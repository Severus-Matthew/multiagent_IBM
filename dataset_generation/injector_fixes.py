from __future__ import annotations

"""Make AIOpsLab's injectors act on the service the scenario actually requested.

Several upstream injectors are written around one application's literals and
therefore mutate nothing when the generator points them at any other service:

* ``inject_wrong_bin_usage`` only rewrites a container whose command already
  contains ``profile``, which is true of exactly one deployment.
* ``inject_container_kill`` receives a container name from the problem class,
  which pins it to one application's container even though the generator varies
  the service.
* ``inject_revoke_auth`` and ``inject_storage_user_unregistered`` hard-filter to
  the only two MongoDB deployments that were started with ``--auth``; the rest
  have no user catalog, so the call is a silent no-op.

A capture produced by a no-op injection is a healthy system carrying a fault
label. It cannot be diagnosed, it cannot be reproduced by any Twin, and training
on it teaches a policy to invent root causes. These patches close that gap.

The replacement mutations are computed by ``fault_mutation_discovery``, the same
module the live Twin uses. Sharing one implementation is deliberate: it makes a
regenerated incident reproducible by the Twin by construction, rather than by two
implementations happening to agree.

Apply with :func:`apply_injector_fixes` before the generator constructs problems.
"""

import json
import subprocess
import time
from typing import Any, Callable

from digital_twin_runtime.fault_mutation_discovery import (
    MutationDiscoveryError,
    config_file_overlay_objects,
    discover_binary_swap,
    discover_config_corruption,
    discover_mongo_access,
    list_mongo_users,
    live_namespace_bundle,
    primary_container,
    read_container_json_file,
    require_auth_catalog,
    resolve_application_source_root,
    select_revocable_user,
    workload_object,
)

# Records what each patched injection actually mutated, so a regeneration run can
# prove the fault was applied instead of assuming it.
INJECTION_JOURNAL: list[dict[str, Any]] = []


def _kubectl(args: list[str], payload: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        input=json.dumps(payload) if payload is not None else None,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _record(mechanism: str, service: str, applied: bool, **details: Any) -> None:
    INJECTION_JOURNAL.append({
        "mechanism": mechanism, "service": service, "applied": bool(applied), **details,
    })


def _wait_for(predicate: Callable[[], bool], timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


def _chaos_manifested(namespace: str, kind: str, name: str) -> bool:
    def observed() -> bool:
        proc = _kubectl(["get", kind, name, "-n", namespace, "-o", "json"])
        if proc.returncode != 0:
            return False
        obj = json.loads(proc.stdout or "{}")
        status = obj.get("status", {}) or {}
        conditions = status.get("conditions", []) or []
        injected = any(
            str(row.get("type") or "").lower() in {"allinjected", "selected"}
            and str(row.get("status") or "").lower() == "true"
            for row in conditions if isinstance(row, dict)
        )
        records = status.get("experiment", {}).get("containerRecords", []) or []
        return injected or bool(records)
    return _wait_for(observed)


def _deployment(namespace: str, service: str) -> dict[str, Any]:
    proc = _kubectl(["get", "deployment", service, "-n", namespace, "-o", "json"])
    if proc.returncode != 0:
        raise MutationDiscoveryError(
            "target_deployment_not_found", service=service, namespace=namespace
        )
    return json.loads(proc.stdout or "{}")


def _apply(obj: dict[str, Any], operation: str) -> None:
    proc = _kubectl(["apply", "-f", "-"], obj)
    if proc.returncode != 0:
        raise RuntimeError(f"{operation} failed: {proc.stderr.strip()}")


def _strip_runtime_fields(obj: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(obj))
    meta = clean.get("metadata", {}) or {}
    for key in ("resourceVersion", "uid", "creationTimestamp", "generation",
                "managedFields", "selfLink"):
        meta.pop(key, None)
    clean.pop("status", None)
    return clean


# --------------------------------------------------------------------------
# wrong binary
# --------------------------------------------------------------------------


_ORIGINAL_COMMANDS: dict[tuple[str, str], list[str] | None] = {}
_ORIGINAL_DEPLOYMENTS: dict[tuple[str, str, str], dict[str, Any]] = {}
_ORIGINAL_SERVICES: dict[tuple[str, str], dict[str, Any]] = {}


def _patched_wrong_bin_usage(self: Any, microservices: list[str]) -> None:
    namespace = self.namespace
    bundle = live_namespace_bundle(namespace)
    for service in microservices:
        try:
            swap = discover_binary_swap(bundle, service)
            live = _strip_runtime_fields(_deployment(namespace, service))
            rows = live["spec"]["template"]["spec"].get("containers", []) or []
            target = next(
                (r for r in rows if str(r.get("name") or "") == swap.container_name),
                rows[0] if rows else None,
            )
            if target is None:
                raise MutationDiscoveryError("no_container_to_mutate", service=service)
            _ORIGINAL_COMMANDS[(namespace, service)] = list(target.get("command") or []) or None
            target["command"] = list(swap.faulted_command)
            _apply(live, f"inject wrong binary into {service}")
            manifested = _wait_for(
                lambda: any(
                    list(row.get("command") or []) == list(swap.faulted_command)
                    for row in (_deployment(namespace, service).get("spec", {})
                                .get("template", {}).get("spec", {})
                                .get("containers", []) or [])
                )
            )
            _record("wrong_binary", service, True,
                    manifested=manifested, command=swap.faulted_command,
                    strategy=swap.strategy)
        except (MutationDiscoveryError, RuntimeError) as exc:
            _record("wrong_binary", service, False, error=str(exc))
            raise


def _patched_recover_wrong_bin_usage(self: Any, microservices: list[str]) -> None:
    namespace = self.namespace
    for service in microservices:
        live = _strip_runtime_fields(_deployment(namespace, service))
        rows = live["spec"]["template"]["spec"].get("containers", []) or []
        original = _ORIGINAL_COMMANDS.get((namespace, service))
        for row in rows:
            if original:
                row["command"] = list(original)
            else:
                row.pop("command", None)
        _apply(live, f"recover wrong binary on {service}")


def _service(namespace: str, service: str) -> dict[str, Any]:
    proc = _kubectl(["get", "service", service, "-n", namespace, "-o", "json"])
    if proc.returncode != 0:
        raise MutationDiscoveryError("target_service_not_found", service=service)
    return json.loads(proc.stdout or "{}")


def _patched_misconfig_k8s(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        live = _strip_runtime_fields(_service(self.namespace, service))
        _ORIGINAL_SERVICES[(self.namespace, service)] = live
        ports = (live.get("spec", {}) or {}).get("ports", []) or []
        if not ports:
            raise MutationDiscoveryError("service_has_no_ports", service=service)
        original = ports[0].get("targetPort", ports[0].get("port"))
        ports[0]["targetPort"] = 65534
        _apply(live, f"misconfigure target port on {service}")
        manifested = _wait_for(
            lambda: ((_service(self.namespace, service).get("spec", {}) or {})
                     .get("ports", [{}])[0].get("targetPort") == 65534)
        )
        _record("target_port_misconfig", service, True, manifested=manifested,
                original_target_port=original, faulted_target_port=65534)


def _patched_recover_misconfig_k8s(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        original = _ORIGINAL_SERVICES.get((self.namespace, service))
        if original:
            _apply(original, f"restore target port on {service}")


def _patched_scale_zero(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        original = _strip_runtime_fields(_deployment(self.namespace, service))
        _ORIGINAL_DEPLOYMENTS[(self.namespace, service, "scale")] = original
        proc = _kubectl(["scale", f"deployment/{service}", "-n", self.namespace, "--replicas=0"])
        if proc.returncode != 0:
            raise RuntimeError(f"scale failed: {proc.stderr.strip()}")
        manifested = _wait_for(
            lambda: int((_deployment(self.namespace, service).get("status", {}) or {})
                        .get("replicas", 0) or 0) == 0
        )
        _record("scale_replicas_zero", service, True, manifested=manifested,
                original_replicas=(original.get("spec", {}) or {}).get("replicas", 1))


def _patched_recover_scale_zero(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        original = _ORIGINAL_DEPLOYMENTS.get((self.namespace, service, "scale"))
        replicas = int(((original or {}).get("spec", {}) or {}).get("replicas", 1) or 0)
        proc = _kubectl(["scale", f"deployment/{service}", "-n", self.namespace,
                         f"--replicas={replicas}"])
        if proc.returncode != 0:
            raise RuntimeError(f"scale recovery failed: {proc.stderr.strip()}")


def _patched_assign_missing_node(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        original = _strip_runtime_fields(_deployment(self.namespace, service))
        _ORIGINAL_DEPLOYMENTS[(self.namespace, service, "node")] = original
        faulted = json.loads(json.dumps(original))
        faulted["spec"]["template"]["spec"]["nodeSelector"] = {
            "kubernetes.io/hostname": "aiops-non-existent.invalid"
        }
        _apply(faulted, f"assign {service} to nonexistent node")
        manifested = _wait_for(
            lambda: (
                _deployment(self.namespace, service).get("spec", {})
                .get("template", {}).get("spec", {}).get("nodeSelector", {})
                .get("kubernetes.io/hostname") == "aiops-non-existent.invalid"
            )
        )
        _record("assign_to_non_existent_node", service, True, manifested=manifested,
                node="aiops-non-existent.invalid")


def _patched_recover_assign_missing_node(self: Any, microservices: list[str]) -> None:
    for service in microservices:
        original = _ORIGINAL_DEPLOYMENTS.get((self.namespace, service, "node"))
        if original:
            _apply(original, f"restore scheduling for {service}")


# --------------------------------------------------------------------------
# container kill
# --------------------------------------------------------------------------


def _patched_container_kill(self: Any, microservice: str, containers: Any) -> None:
    """Resolve the container from the requested workload, not from a fixed literal."""
    namespace = self.namespace
    requested = list(containers) if isinstance(containers, list) else [containers]
    resolved = requested
    try:
        bundle = live_namespace_bundle(namespace)
        workload = workload_object(bundle, microservice)
        available = {
            str(row.get("name"))
            for row in (
                workload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
                or []
            )
        }
        if not available & set(requested):
            resolved = [str(primary_container(workload, microservice).get("name"))]
    except MutationDiscoveryError:
        resolved = requested

    experiment = {
        "apiVersion": "chaos-mesh.org/v1alpha1",
        "kind": "PodChaos",
        "metadata": {"name": "container-kill", "namespace": namespace},
        "spec": {
            "action": "container-kill",
            "mode": "all",
            "duration": "200s",
            "selector": {"labelSelectors": {"io.kompose.service": microservice}},
            "containerNames": resolved,
        },
    }
    self.create_chaos_experiment(experiment, "container-kill")
    manifested = _chaos_manifested(namespace, "podchaos", "container-kill")
    _record("container_kill", microservice, True, manifested=manifested,
            requested_containers=requested, resolved_containers=resolved)


def _inject_chaos(
    self: Any, mechanism: str, microservices: list[str], experiment: dict[str, Any],
    file_name: str, resource: str,
) -> None:
    self.create_chaos_experiment(experiment, file_name)
    manifested = _chaos_manifested(
        self.namespace, resource, str(experiment["metadata"]["name"])
    )
    for service in microservices:
        _record(mechanism, service, True, manifested=manifested,
                chaos_kind=experiment["kind"], chaos_name=experiment["metadata"]["name"])


def _patched_pod_failure(self: Any, microservices: list[str], duration: str = "200s") -> None:
    experiment = {
        "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos",
        "metadata": {"name": "pod-failure-experiment", "namespace": self.namespace},
        "spec": {"action": "pod-failure", "mode": "one", "duration": duration,
                 "selector": {"labelSelectors": {"io.kompose.service": ", ".join(microservices)}}},
    }
    _inject_chaos(self, "pod_failure", microservices, experiment, "pod-failure", "podchaos")


def _patched_pod_kill(self: Any, microservices: list[str], duration: str = "200s") -> None:
    experiment = {
        "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos",
        "metadata": {"name": "pod-kill", "namespace": self.namespace},
        "spec": {"action": "pod-failure", "mode": "one", "duration": duration,
                 "selector": {"labelSelectors": {"io.kompose.service": ", ".join(microservices)}}},
    }
    _inject_chaos(self, "pod_kill", microservices, experiment, "pod-kill", "podchaos")


def _patched_network_loss(self: Any, microservices: list[str], duration: str = "200s") -> None:
    experiment = {
        "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
        "metadata": {"name": "loss", "namespace": self.namespace},
        "spec": {"action": "loss", "mode": "one", "duration": duration,
                 "selector": {"namespaces": [self.namespace],
                              "labelSelectors": {"io.kompose.service": ", ".join(microservices)}},
                 "loss": {"loss": "99", "correlation": "100"}},
    }
    _inject_chaos(self, "network_loss", microservices, experiment, "network-loss", "networkchaos")


def _patched_network_delay(
    self: Any, microservices: list[str], duration: str = "200s",
    latency: str = "10s", jitter: str = "0ms",
) -> None:
    experiment = {
        "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
        "metadata": {"name": "delay", "namespace": self.namespace},
        "spec": {"action": "delay", "mode": "one", "duration": duration,
                 "selector": {"namespaces": [self.namespace],
                              "labelSelectors": {"io.kompose.service": ", ".join(microservices)}},
                 "delay": {"latency": latency, "correlation": "100", "jitter": jitter}},
    }
    _inject_chaos(self, "network_delay", microservices, experiment, "network-delay", "networkchaos")


# --------------------------------------------------------------------------
# MongoDB authorization
# --------------------------------------------------------------------------


def _ensure_authorization(namespace: str, service: str) -> bool:
    """Turn on authorization when the target datastore has none.

    The captured corpus targets several MongoDB deployments that were never
    started with ``--auth``. Upstream skipped those silently. Enabling
    authorization first gives the revoke and drop-user mechanisms a real catalog
    to act on, and mirrors what the Twin's adapter does.
    """
    live = _strip_runtime_fields(_deployment(namespace, service))
    rows = live["spec"]["template"]["spec"].get("containers", []) or []
    if not rows:
        raise MutationDiscoveryError("no_container_to_mutate", service=service)
    args = [str(x) for x in (rows[0].get("args") or [])]
    if "--auth" in args:
        return False
    # Snapshot before mutating so recovery can restore the exact prior template
    # even when the upstream recover method knows nothing about this service.
    _ORIGINAL_DEPLOYMENTS.setdefault((namespace, service, "auth"), json.loads(json.dumps(live)))
    rows[0]["args"] = [*args, "--auth"]
    _apply(live, f"enable MongoDB authorization on {service}")
    _kubectl(["rollout", "status", f"deployment/{service}", "-n", namespace, "--timeout=180s"])
    return True


def _service_pods(namespace: str, service: str) -> list[dict[str, Any]]:
    """Pods that belong to ``service``, resolved from the workload's own selector.

    Charts label pods differently (``io.kompose.service``, ``service``, ``app``),
    so the Deployment's ``matchLabels`` is the only reliable identity.
    """
    try:
        selector = ((_deployment(namespace, service).get("spec", {}) or {})
                    .get("selector", {}) or {}).get("matchLabels", {}) or {}
    except MutationDiscoveryError:
        selector = {}
    if selector:
        label = ",".join(f"{k}={v}" for k, v in sorted(selector.items()))
        proc = _kubectl(["get", "pods", "-n", namespace, "-l", label, "-o", "json"])
        if proc.returncode == 0:
            return list(json.loads(proc.stdout or "{}").get("items", []) or [])
    proc = _kubectl(["get", "pods", "-n", namespace, "-o", "json"])
    if proc.returncode != 0:
        return []
    return [
        pod for pod in (json.loads(proc.stdout or "{}").get("items", []) or [])
        if str((pod.get("metadata", {}) or {}).get("name") or "").startswith(service + "-")
    ]


def _mongo_pod(namespace: str, service: str) -> str:
    names = [
        str((pod.get("metadata", {}) or {}).get("name") or "")
        for pod in _service_pods(namespace, service)
        if str((pod.get("status", {}) or {}).get("phase") or "") == "Running"
        and not (pod.get("metadata", {}) or {}).get("deletionTimestamp")
    ]
    if not names:
        raise MutationDiscoveryError("no_running_pod_for_service", service=service)
    return names[0]


def _mongo_role_mutation(self: Any, microservices: list[str], *, drop_user: bool) -> None:
    namespace = self.namespace
    mechanism = "mongodb_user_unregistered" if drop_user else "mongodb_auth_revoked"
    for service in microservices:
        try:
            _ensure_authorization(namespace, service)
            pod = _mongo_pod(namespace, service)
            bundle = live_namespace_bundle(namespace)
            access = discover_mongo_access(bundle, service, namespace, pod)
            require_auth_catalog(access)
            users = list_mongo_users(access, namespace, pod)
            user = select_revocable_user(access, users)
            admin = "db.getSiblingDB('admin')"
            quoted = json.dumps(user.username)
            script = (
                f"{admin}.dropUser({quoted});" if drop_user
                else f"{admin}.revokeRolesFromUser({quoted}, {json.dumps(user.roles)});"
            )
            proc = _kubectl(["exec", "-n", namespace, pod, "--", *access.shell_command(script)])
            if proc.returncode != 0:
                raise RuntimeError(f"mongo mutation failed: {proc.stderr.strip()}")
            after = list_mongo_users(access, namespace, pod)
            current = next((row for row in after if row.username == user.username), None)
            manifested = current is None if drop_user else bool(current and not current.roles)
            _record(mechanism, service, True, manifested=manifested,
                    user=user.username, roles=user.roles)
        except (MutationDiscoveryError, RuntimeError) as exc:
            _record(mechanism, service, False, error=str(exc))
            raise


def _patched_revoke_auth(self: Any, microservices: list[str]) -> None:
    _mongo_role_mutation(self, microservices, drop_user=False)


def _patched_storage_user_unregistered(self: Any, microservices: list[str]) -> None:
    _mongo_role_mutation(self, microservices, drop_user=True)


def _patched_auth_miss_mongodb(self: Any, microservices: list[str]) -> None:
    """Enforce authorization on the requested datastore.

    Upstream always flips TLS on one specific MongoDB through Helm regardless of
    the requested target. Enforcing authorization on the service the scenario
    names produces the same client-visible failure and actually depends on the
    target.
    """
    for service in microservices:
        try:
            changed = _ensure_authorization(self.namespace, service)
            if not changed:
                raise MutationDiscoveryError(
                    "target_mongodb_already_enforces_authorization", service=service
                )
            pod = _mongo_pod(self.namespace, service)
            access = discover_mongo_access(
                live_namespace_bundle(self.namespace), service, self.namespace, pod
            )
            _record("mongodb_auth_missing", service, True,
                    manifested=bool(access.auth_enabled),
                    strategy="enforce_authorization_on_anonymous_datastore")
        except (MutationDiscoveryError, RuntimeError) as exc:
            _record("mongodb_auth_missing", service, False, error=str(exc))
            raise


# --------------------------------------------------------------------------
# application configuration misconfiguration
# --------------------------------------------------------------------------


def _application_source_roots() -> list[Any]:
    from aiopslab.paths import TARGET_MICROSERVICES

    root = TARGET_MICROSERVICES
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _running_pod(namespace: str, service: str) -> str:
    return _mongo_pod(namespace, service)


def _pod_carries_overlay(namespace: str, service: str, configmap: str, container: str,
                         path: str, key: str, expected: Any) -> bool:
    for pod in _service_pods(namespace, service):
        if str((pod.get("status", {}) or {}).get("phase") or "") != "Running":
            continue
        volumes = ((pod.get("spec", {}) or {}).get("volumes") or [])
        if not any((v.get("configMap") or {}).get("name") == configmap for v in volumes):
            continue
        name = str((pod.get("metadata", {}) or {}).get("name") or "")
        try:
            data = read_container_json_file(namespace, name, container, path)
        except MutationDiscoveryError:
            return False
        return data.get(key) == expected
    return False


def _patched_misconfig_app(self: Any, microservices: list[str]) -> None:
    """Corrupt a real configuration endpoint of the requested service.

    Upstream swaps in one hard-coded image tag whose baked-in ``config.json``
    points the geo service at a wrong MongoDB port; on any other service that
    same image is a functional no-op. Here the corruption is discovered from the
    application's own configuration surface (ConfigMap, environment, arguments,
    or the file the Dockerfile bakes into the image) for the service that was
    actually requested, using the same discovery the Twin adapter runs.
    """
    namespace = self.namespace
    bundle = live_namespace_bundle(namespace)
    for service in microservices:
        try:
            root = resolve_application_source_root(bundle, _application_source_roots())
            corruption = discover_config_corruption(bundle, service, application_source_root=root)
            live = _strip_runtime_fields(_deployment(namespace, service))
            _ORIGINAL_DEPLOYMENTS[(namespace, service, "config")] = live
            if corruption.target_kind != "ConfigFile":
                raise MutationDiscoveryError(
                    "generator_config_fault_supports_image_config_files_only",
                    service=service, discovered=corruption.target_kind,
                )
            pod = _running_pod(namespace, service)
            live_config = read_container_json_file(
                namespace, pod, corruption.container_name, corruption.target_name
            )
            configmap, faulted = config_file_overlay_objects(live, corruption, namespace, live_config)
            _apply(configmap, f"create configuration overlay for {service}")
            _apply(faulted, f"inject configuration corruption into {service}")
            cm_name = str(configmap["metadata"]["name"])
            manifested = _wait_for(lambda: _pod_carries_overlay(
                namespace, service, cm_name, corruption.container_name,
                corruption.target_name, corruption.key, corruption.faulted_value,
            ), timeout=180.0)
            _record("application_config_misconfig", service, True, manifested=manifested,
                    target=corruption.target_name, key=corruption.key,
                    faulted_value=corruption.faulted_value, strategy=corruption.strategy,
                    configmap=cm_name)
        except (MutationDiscoveryError, RuntimeError) as exc:
            _record("application_config_misconfig", service, False, error=str(exc))
            raise


def _patched_recover_misconfig_app(self: Any, microservices: list[str]) -> None:
    namespace = self.namespace
    for service in microservices:
        original = _ORIGINAL_DEPLOYMENTS.get((namespace, service, "config"))
        if original:
            _apply(original, f"restore configuration for {service}")
        from digital_twin_runtime.fault_mutation_discovery import overlay_configmap_name
        _kubectl(["delete", "configmap", overlay_configmap_name(service), "-n", namespace,
                  "--ignore-not-found=true"])


def _restore_datastore_or_upstream(self: Any, microservices: list[str], upstream: Callable[..., Any]) -> None:
    """Undo a MongoDB mutation on the requested services.

    When the fix enabled authorization on a datastore that ran without it, the
    faithful recovery is to restore that exact prior Deployment: with
    authorization off again, a revoked role or dropped user no longer affects any
    client. Datastores that already enforced authorization keep upstream's own
    recovery scripts.
    """
    namespace = self.namespace
    handled: list[str] = []
    for service in microservices:
        original = _ORIGINAL_DEPLOYMENTS.pop((namespace, service, "auth"), None)
        if original is None:
            continue
        _apply(original, f"restore datastore deployment for {service}")
        _kubectl(["rollout", "status", f"deployment/{service}", "-n", namespace, "--timeout=180s"])
        handled.append(service)
    remaining = [service for service in microservices if service not in handled]
    if remaining:
        upstream(self, remaining)


def _make_recover(upstream: Callable[..., Any]) -> Callable[..., Any]:
    def _recover(self: Any, microservices: list[str]) -> None:
        _restore_datastore_or_upstream(self, microservices, upstream)
    return _recover


PATCHES: dict[str, dict[str, Callable[..., Any]]] = {
    "VirtualizationFaultInjector": {
        "inject_misconfig_k8s": _patched_misconfig_k8s,
        "recover_misconfig_k8s": _patched_recover_misconfig_k8s,
        "inject_scale_pods_to_zero": _patched_scale_zero,
        "recover_scale_pods_to_zero": _patched_recover_scale_zero,
        "inject_assign_to_non_existent_node": _patched_assign_missing_node,
        "recover_assign_to_non_existent_node": _patched_recover_assign_missing_node,
        "inject_wrong_bin_usage": _patched_wrong_bin_usage,
        "recover_wrong_bin_usage": _patched_recover_wrong_bin_usage,
    },
    "SymptomFaultInjector": {
        "inject_container_kill": _patched_container_kill,
        "inject_pod_failure": _patched_pod_failure,
        "inject_pod_kill": _patched_pod_kill,
        "inject_network_loss": _patched_network_loss,
        "inject_network_delay": _patched_network_delay,
    },
    "ApplicationFaultInjector": {
        "inject_revoke_auth": _patched_revoke_auth,
        "inject_storage_user_unregistered": _patched_storage_user_unregistered,
        "inject_auth_miss_mongodb": _patched_auth_miss_mongodb,
        "inject_misconfig_app": _patched_misconfig_app,
        "recover_misconfig_app": _patched_recover_misconfig_app,
    },
}

# Recovery wrappers need the upstream implementation, which is only known once
# the class is imported; they are installed by apply_injector_fixes.
_RECOVER_WRAPS = {
    "ApplicationFaultInjector": ("recover_auth_miss_mongodb", "recover_revoke_auth",
                                 "recover_storage_user_unregistered"),
}


def apply_injector_fixes() -> dict[str, list[str]]:
    """Install the target-honouring injector implementations. Idempotent."""
    from aiopslab.generators.fault.inject_app import ApplicationFaultInjector
    from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
    from aiopslab.generators.fault.inject_virtual import VirtualizationFaultInjector

    classes = {
        "ApplicationFaultInjector": ApplicationFaultInjector,
        "SymptomFaultInjector": SymptomFaultInjector,
        "VirtualizationFaultInjector": VirtualizationFaultInjector,
    }
    applied: dict[str, list[str]] = {}
    for class_name, methods in PATCHES.items():
        cls = classes[class_name]
        for method_name, replacement in methods.items():
            if not hasattr(cls, method_name):
                raise AttributeError(
                    f"{class_name}.{method_name} is absent; upstream AIOpsLab changed "
                    "and the injector fix must be revalidated before regenerating data"
                )
            setattr(cls, method_name, replacement)
            applied.setdefault(class_name, []).append(method_name)
    for class_name, method_names in _RECOVER_WRAPS.items():
        cls = classes[class_name]
        for method_name in method_names:
            upstream = getattr(cls, method_name, None)
            if upstream is None:
                raise AttributeError(f"{class_name}.{method_name} is absent; revalidate injector fixes")
            if getattr(upstream, "_wraps_upstream_recover", False):
                continue
            wrapped = _make_recover(upstream)
            wrapped._wraps_upstream_recover = True  # type: ignore[attr-defined]
            setattr(cls, method_name, wrapped)
            applied.setdefault(class_name, []).append(method_name)
    return applied
