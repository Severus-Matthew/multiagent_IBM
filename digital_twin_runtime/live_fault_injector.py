from __future__ import annotations

import json
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from training_pipeline.schemas import FaultLabel, normalize_fault_mechanism

from .sparse_live_session import SparseLiveTwinSession
from .live_capabilities import live_injector_implemented
from .fault_mutation_discovery import (
    MutationDiscoveryError,
    config_file_overlay_objects,
    discover_binary_swap,
    discover_config_corruption,
    read_container_json_file,
    discover_mongo_access,
    list_mongo_users,
    require_auth_catalog,
    select_revocable_user,
)

POD_CHAOS_ACTIONS = {
    "container_kill": "container-kill",
    # Chaos Mesh has no separate container-stop action; container-kill is the
    # closest executable counterfactual for the provider's stop mechanism.
    "container_stop": "container-kill",
    "pod_failure": "pod-failure",
    "pod_kill": "pod-kill",
}


def _run(args: list[str], payload: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str], operation: str) -> str:
    if proc.returncode != 0:
        raise RuntimeError(f"{operation} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _json(args: list[str]) -> dict[str, Any]:
    return json.loads(_ok(_run([*args, "-o", "json"]), "kubectl read") or "{}")


def _deployment_pod(session: SparseLiveTwinSession, deployment: dict[str, Any]) -> str:
    labels = ((deployment.get("spec", {}) or {}).get("selector", {}) or {}).get("matchLabels", {}) or {}
    if not labels:
        raise ValueError("selected Deployment has no pod selector")
    selector = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    items = _json(["get", "pods", "-n", session.namespace, "-l", selector]).get("items", []) or []
    running = [row for row in items if (row.get("status", {}) or {}).get("phase") == "Running"]
    if not running:
        raise RuntimeError("selected Deployment has no running pod")
    return str((running[0].get("metadata", {}) or {}).get("name") or "")


def _pod_exec(namespace: str, pod: str, command: list[str], operation: str) -> str:
    return _ok(_run(["exec", "-n", namespace, pod, "--", *command]), operation)


@dataclass
class FaultManifestation:
    service: str
    mechanism: str
    manifested: bool
    elapsed_seconds: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveFaultHandle:
    def __init__(
        self,
        session: SparseLiveTwinSession,
        fault: FaultLabel,
        original_deployment: dict[str, Any],
        restore_mode: str = "replace_deployment",
        original_service: dict[str, Any] | None = None,
        original_configmap: dict[str, Any] | None = None,
        injected_details: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.fault = fault
        self.original_deployment = original_deployment
        self.restore_mode = restore_mode
        self.original_service = original_service
        self.original_configmap = original_configmap
        self.injected_details = dict(injected_details or {})
        self.restored = False

    def _target_pods(self) -> list[dict[str, Any]]:
        labels = (
            (self.original_deployment.get("spec", {}) or {}).get("selector", {}) or {}
        ).get("matchLabels", {}) or {}
        if not labels:
            return []
        selector = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
        return _json([
            "get", "pods", "-n", self.session.namespace, "-l", selector,
        ]).get("items", []) or []

    def _workload_start_failure(self) -> tuple[bool, list[dict[str, Any]]]:
        """Report whether the target's containers are failing to run."""
        failing = False
        rows: list[dict[str, Any]] = []
        for pod in self._target_pods():
            status = pod.get("status", {}) or {}
            row: dict[str, Any] = {
                "pod": str((pod.get("metadata", {}) or {}).get("name") or ""),
                "phase": str(status.get("phase") or ""),
                "restarts": 0,
                "waiting_reasons": [],
                "terminated_reasons": [],
                "ready": True,
            }
            for container in status.get("containerStatuses", []) or []:
                row["restarts"] = max(int(row["restarts"]), int(container.get("restartCount") or 0))
                if not container.get("ready", False):
                    row["ready"] = False
                state = container.get("state", {}) or {}
                waiting = str((state.get("waiting") or {}).get("reason") or "")
                terminated = str((state.get("terminated") or {}).get("reason") or "")
                if waiting:
                    row["waiting_reasons"].append(waiting)
                if terminated:
                    row["terminated_reasons"].append(terminated)
            if (
                int(row["restarts"]) > 0
                or not row["ready"]
                or row["waiting_reasons"]
                or row["terminated_reasons"]
                or row["phase"] in {"Failed", "Pending"}
            ):
                failing = True
            rows.append(row)
        return failing, rows

    def _configuration_mutation_present(self) -> tuple[bool, dict[str, Any]]:
        """Confirm the corrupted configuration value is live in the cluster."""
        kind = str(self.injected_details.get("target_kind") or "")
        name = str(self.injected_details.get("target_name") or "")
        key = str(self.injected_details.get("key") or "")
        expected = self.injected_details.get("faulted_value")

        if kind == "ConfigMap":
            data = _json([
                "get", "configmap", name, "-n", self.session.namespace,
            ]).get("data", {}) or {}
            return data.get(key) == expected, {"configmap": name, "key": key,
                                               "observed": data.get(key)}

        deployment = _json([
            "get", "deployment", self.fault.service, "-n", self.session.namespace,
        ])
        rows = (
            deployment.get("spec", {}).get("template", {}).get("spec", {})
            .get("containers", []) or []
        )
        if kind == "Env":
            for row in rows:
                for entry in row.get("env", []) or []:
                    if str((entry or {}).get("name") or "") == key:
                        return entry.get("value") == expected, {
                            "container": row.get("name"), "env": key,
                            "observed": entry.get("value"),
                        }
            return False, {"env": key, "observed": None}
        if kind == "Args":
            try:
                index = int(key)
            except ValueError:
                return False, {"args_index": key, "observed": None}
            for row in rows:
                args = list(row.get("args") or [])
                if 0 <= index < len(args):
                    return args[index] == expected, {
                        "container": row.get("name"), "args_index": index,
                        "observed": args[index],
                    }
            return False, {"args_index": index, "observed": None}
        if kind == "ConfigFile":
            # The overlay only manifests once a pod from the mutated template is
            # running and serving the corrupted file; the previous pod keeps the
            # original image file until it is replaced.
            container = str(self.injected_details.get("container") or "")
            for pod in self._target_pods():
                status = pod.get("status", {}) or {}
                if str(status.get("phase") or "") != "Running":
                    continue
                spec = pod.get("spec", {}) or {}
                if not any(
                    (v.get("configMap") or {}).get("name") == self.injected_details.get("configmap")
                    for v in (spec.get("volumes") or [])
                ):
                    continue
                pod_name = str((pod.get("metadata", {}) or {}).get("name") or "")
                try:
                    data = read_container_json_file(self.session.namespace, pod_name, container, name)
                except MutationDiscoveryError as exc:
                    return False, {"pod": pod_name, "file": name, "error": exc.reason}
                return data.get(key) == expected, {
                    "pod": pod_name, "file": name, "key": key, "observed": data.get(key),
                }
            return False, {"file": name, "observed": None, "reason": "no_running_pod_from_mutated_template"}
        return False, {"unsupported_target_kind": kind}

    def wait_for_manifestation(
        self, *, timeout_seconds: float = 60.0, poll_seconds: float = 0.5
    ) -> FaultManifestation:
        started = time.monotonic()
        evidence: list[dict[str, Any]] = []
        while time.monotonic() - started < timeout_seconds:
            mechanism = normalize_fault_mechanism(self.fault.fault_mechanism)
            if mechanism in {"container_kill", "pod_failure", "pod_kill", "container_stop",
                             "network_delay", "network_loss"}:
                kind = str(self.injected_details.get("kind") or "")
                name = str(self.injected_details.get("name") or "")
                proc = _run(["get", kind, name, "-n", self.session.namespace, "-o", "json"])
                if proc.returncode != 0:
                    evidence = [{
                        "kind": kind, "name": name, "resource_exists": False,
                        "kubectl_stderr": proc.stderr.strip()[-1000:],
                    }]
                    time.sleep(max(0.1, poll_seconds))
                    continue
                try:
                    obj = json.loads(proc.stdout or "{}")
                except json.JSONDecodeError:
                    obj = {}
                status = obj.get("status", {}) or {}
                conditions = {
                    str(row.get("type")): str(row.get("status"))
                    for row in (status.get("conditions", []) or [])
                    if isinstance(row, dict)
                }
                records = (status.get("experiment", {}) or {}).get("containerRecords", []) or []
                injected_records = [
                    row for row in records
                    if str((row or {}).get("phase") or "").lower() == "injected"
                ]
                evidence = [{
                    "kind": kind, "name": name, "resource_exists": True,
                    "conditions": conditions,
                    "selected_records": len(records),
                    "injected_records": len(injected_records),
                }]
                # Chaos Mesh reports admission and application separately. Treating
                # mere admission as manifestation lets a chaos object that selected
                # nothing count as a reproduced fault, so require the controller's
                # own AllInjected condition and at least one injected target.
                if conditions.get("AllInjected") == "True" and injected_records:
                    return FaultManifestation(
                        service=self.fault.service, mechanism=mechanism, manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), evidence=evidence,
                        condition=(
                            "Chaos Mesh reported AllInjected=True with "
                            f"{len(injected_records)} injected container record(s)"
                        ),
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "target_port_misconfig":
                service = _json([
                    "get", "service", self.fault.service,
                    "-n", self.session.namespace,
                ])
                ports = (service.get("spec", {}) or {}).get("ports", []) or []
                target_ports = [row.get("targetPort") for row in ports]
                endpoint = _json([
                    "get", "endpoints", self.fault.service,
                    "-n", self.session.namespace,
                ])
                address_count = sum(
                    len((subset or {}).get("addresses", []) or [])
                    for subset in endpoint.get("subsets", []) or []
                )
                evidence = [{
                    "service": self.fault.service,
                    "target_ports": target_ports,
                    "ready_endpoint_addresses": address_count,
                }]
                expected = self.injected_details.get("target_port")
                if expected in target_ports and address_count > 0:
                    return FaultManifestation(
                        service=self.fault.service,
                        mechanism=self.fault.fault_mechanism,
                        manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        evidence=evidence,
                        condition=(
                            f"Service targetPort={expected} while endpoint addresses remain present"
                        ),
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "scale_replicas_zero":
                deployment = _json([
                    "get", "deployment", self.fault.service,
                    "-n", self.session.namespace,
                ])
                desired = int((deployment.get("spec", {}) or {}).get("replicas", -1) or 0)
                ready = int((deployment.get("status", {}) or {}).get("readyReplicas", 0) or 0)
                endpoint = _json([
                    "get", "endpoints", self.fault.service,
                    "-n", self.session.namespace,
                ])
                address_count = sum(
                    len((subset or {}).get("addresses", []) or [])
                    for subset in endpoint.get("subsets", []) or []
                )
                evidence = [{
                    "deployment": self.fault.service,
                    "desired_replicas": desired,
                    "ready_replicas": ready,
                    "ready_endpoint_addresses": address_count,
                }]
                expected = int(self.injected_details.get("replicas", 0))
                manifested = (
                    desired == expected
                    and ((expected == 0 and ready == 0 and address_count == 0)
                         or (expected > 0 and ready == expected and address_count > 0))
                )
                if manifested:
                    return FaultManifestation(
                        service=self.fault.service,
                        mechanism=self.fault.fault_mechanism,
                        manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        evidence=evidence,
                        condition=f"Deployment and endpoints reached requested replica variant {expected}",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "wrong_binary":
                deployment = _json([
                    "get", "deployment", self.fault.service,
                    "-n", self.session.namespace,
                ])
                spec_containers = (
                    deployment.get("spec", {}).get("template", {}).get("spec", {})
                    .get("containers", []) or []
                )
                expected_command = list(self.injected_details.get("command") or [])
                applied = any(
                    list(row.get("command") or []) == expected_command
                    for row in spec_containers
                )
                broken, pod_evidence = self._workload_start_failure()
                evidence = [{
                    "deployment": self.fault.service,
                    "expected_command": expected_command,
                    "commands": [row.get("command") for row in spec_containers],
                    "mutation_applied": applied,
                    "pods": pod_evidence,
                }]
                # A wrong entrypoint that the container happily runs is not the
                # fault we claim to have injected, so require the process to fail.
                if applied and broken:
                    return FaultManifestation(
                        service=self.fault.service, mechanism=mechanism, manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), evidence=evidence,
                        condition="Container entrypoint replaced and the container fails to run",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "application_config_misconfig":
                applied, config_evidence = self._configuration_mutation_present()
                broken, pod_evidence = self._workload_start_failure()
                evidence = [{
                    "target": self.injected_details.get("target"),
                    "strategy": self.injected_details.get("strategy"),
                    "mutation_applied": applied,
                    "observed": config_evidence,
                    "pods": pod_evidence,
                }]
                # A configuration fault often leaves the process running and only
                # fails at request time, so the live object carrying the corrupted
                # value is the manifestation condition. Pod state is recorded as
                # supporting evidence rather than required.
                if applied:
                    return FaultManifestation(
                        service=self.fault.service, mechanism=mechanism, manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), evidence=evidence,
                        condition=(
                            "Live configuration object carries the corrupted endpoint"
                            + (" and the workload is degraded" if broken else "")
                        ),
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "mongodb_auth_missing":
                deployment = _json([
                    "get", "deployment", self.fault.service,
                    "-n", self.session.namespace,
                ])
                spec_containers = (
                    deployment.get("spec", {}).get("template", {}).get("spec", {})
                    .get("containers", []) or []
                )
                enforced = any(
                    "--auth" in [str(x) for x in (row.get("args") or [])]
                    for row in spec_containers
                )
                pods = self._target_pods()
                serving = [
                    pod for pod in pods
                    if str((pod.get("status", {}) or {}).get("phase") or "") == "Running"
                ]
                evidence = [{
                    "deployment": self.fault.service,
                    "authorization_enforced": enforced,
                    "running_pods": len(serving),
                    "observed_generation": (deployment.get("status", {}) or {}).get("observedGeneration"),
                    "generation": (deployment.get("metadata", {}) or {}).get("generation"),
                }]
                # The datastore must be up and demanding credentials; a MongoDB that
                # failed to restart would be an availability fault, not an
                # authentication fault.
                if enforced and serving:
                    return FaultManifestation(
                        service=self.fault.service, mechanism=mechanism, manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), evidence=evidence,
                        condition="Datastore enforces authorization while application clients connect anonymously",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism in {"mongodb_auth_revoked", "mongodb_user_unregistered"}:
                pod = _deployment_pod(self.session, self.original_deployment)
                output = _pod_exec(
                    self.session.namespace, pod,
                    list(self.injected_details["manifestation_probe"]),
                    "probe MongoDB authentication mutation",
                )
                marker = str(self.injected_details["manifestation_marker"])
                evidence = [{"pod": pod, "probe_output": output[-2000:], "expected_marker": marker}]
                if marker in output:
                    return FaultManifestation(
                        service=self.fault.service, mechanism=mechanism, manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), evidence=evidence,
                        condition="MongoDB user/role catalog reflects the injected authentication fault",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            pods = _json([
                "get", "pods", "-n", self.session.namespace,
                "-l", f"service={self.fault.service}",
            ]).get("items", []) or []
            evidence = []
            for pod in pods:
                row = {
                    "pod": str((pod.get("metadata", {}) or {}).get("name") or ""),
                    "phase": str((pod.get("status", {}) or {}).get("phase") or ""),
                    "conditions": [],
                }
                for condition in (pod.get("status", {}) or {}).get("conditions", []) or []:
                    item = {
                        "type": condition.get("type"),
                        "status": condition.get("status"),
                        "reason": condition.get("reason"),
                        "message": condition.get("message"),
                    }
                    row["conditions"].append(item)
                    if (
                        item["type"] == "PodScheduled"
                        and str(item["status"]).lower() == "false"
                        and item["reason"] == "Unschedulable"
                    ):
                        endpoint = _json([
                            "get", "endpoints", self.fault.service,
                            "-n", self.session.namespace,
                        ])
                        address_count = sum(
                            len((subset or {}).get("addresses", []) or [])
                            for subset in endpoint.get("subsets", []) or []
                        )
                        row["ready_endpoint_addresses"] = address_count
                        if address_count != 0:
                            continue
                        evidence.append(row)
                        return FaultManifestation(
                            service=self.fault.service,
                            mechanism=self.fault.fault_mechanism,
                            manifested=True,
                            elapsed_seconds=round(time.monotonic() - started, 3),
                            evidence=evidence,
                            condition="PodScheduled=False/Unschedulable and zero ready endpoints",
                        )
                evidence.append(row)
            time.sleep(max(0.1, poll_seconds))
        return FaultManifestation(
            service=self.fault.service,
            mechanism=self.fault.fault_mechanism,
            manifested=False,
            elapsed_seconds=round(time.monotonic() - started, 3),
            evidence=evidence,
            condition="PodScheduled=False/Unschedulable and zero ready endpoints",
        )

    def restore(self) -> None:
        if self.restore_mode == "delete_custom_resource":
            kind = str(self.injected_details["kind"])
            name = str(self.injected_details["name"])
            proc = _run(["delete", kind, name, "-n", self.session.namespace,
                         "--ignore-not-found=true", "--wait=true", "--timeout=60s"])
            _ok(proc, "remove Chaos Mesh fault")
            self.restored = True
            return
        if self.restore_mode == "apply_service":
            if self.original_service is None:
                raise RuntimeError("original Service snapshot is unavailable")
            _ok(_run(["apply", "-f", "-"], self.original_service), "restore original Service targetPort")
            self.restored = True
            return
        if self.restore_mode == "apply_configmap_and_restart":
            if self.original_configmap is None:
                raise RuntimeError("original ConfigMap snapshot is unavailable")
            _ok(_run(["apply", "-f", "-"], self.original_configmap), "restore MongoDB ConfigMap")
            _ok(_run([
                "rollout", "restart", f"deployment/{self.fault.service}",
                "-n", self.session.namespace,
            ]), "restart MongoDB after ConfigMap restoration")
            self.restored = True
            return
        if self.restore_mode == "apply_deployment":
            _ok(_run(["apply", "-f", "-"], self.original_deployment), "restore original Deployment")
            self.restored = True
            return
        if self.restore_mode == "apply_deployment_and_delete_configmap":
            _ok(_run(["apply", "-f", "-"], self.original_deployment), "restore original Deployment")
            _ok(_run([
                "delete", "configmap", str(self.injected_details["configmap"]),
                "-n", self.session.namespace, "--ignore-not-found=true",
            ]), "remove configuration overlay")
            self.restored = True
            return
        if self.restore_mode == "mongodb_exec":
            pod = _deployment_pod(self.session, self.original_deployment)
            _pod_exec(
                self.session.namespace, pod, list(self.injected_details["restore_command"]),
                "restore MongoDB authentication state",
            )
            self.restored = True
            return
        if self.restore_mode == "scale_replicas":
            replicas = int((self.original_deployment.get("spec", {}) or {}).get("replicas", 1) or 1)
            _ok(_run([
                "scale", f"deployment/{self.fault.service}",
                "-n", self.session.namespace, f"--replicas={replicas}",
            ]), "restore original Deployment replica count")
            self.restored = True
            return
        _ok(_run([
            "delete", "deployment", self.fault.service,
            "-n", self.session.namespace, "--wait=true", "--timeout=60s",
        ]), "remove faulted scheduling Deployment")
        _ok(_run(["apply", "-f", "-"], self.original_deployment), "restore scheduling Deployment")
        self.restored = True


def inject_predicted_fault(
    session: SparseLiveTwinSession,
    fault: FaultLabel,
    *,
    application_source_root: str | None = None,
) -> LiveFaultHandle:
    """Inject exactly the agent-predicted mechanism; never infer from oracle state.

    ``application_source_root`` is the fault-independent application source tree
    already used for static topology; it lets configuration faults reach
    configuration files baked into the image when no ConfigMap/env/argument
    endpoint exists.
    """
    if not session.applied:
        raise RuntimeError("Twin baseline manifests are not applied")
    if not fault.is_injectible():
        raise ValueError("RCA fault is generic, unknown, or mechanism/type-inconsistent")
    if not live_injector_implemented(fault, session.bundle):
        raise NotImplementedError(
            "predicted mechanism is unsupported or required selected objects are absent"
        )
    mechanism = normalize_fault_mechanism(fault.fault_mechanism)

    selected_deployments = {
        row["name"] for row in session.bundle.object_refs
        if row.get("kind") == "Deployment"
    }
    selected_services = {
        row["name"] for row in session.bundle.object_refs
        if row.get("kind") == "Service"
    }
    if fault.service not in selected_deployments:
        raise ValueError("predicted service is not a selected Twin Deployment")
    original = next(
        deepcopy(obj) for obj in session.bundle.objects
        if obj.get("kind") == "Deployment"
        and (obj.get("metadata", {}) or {}).get("name") == fault.service
    )
    if mechanism in {"container_kill", "pod_failure", "pod_kill", "container_stop",
                     "network_delay", "network_loss"}:
        selector = deepcopy((original.get("spec", {}) or {}).get("selector", {}) or {})
        match_labels = selector.get("matchLabels", {}) or {}
        if not match_labels:
            raise ValueError("selected Deployment has no matchLabels for Chaos Mesh targeting")
        suffix = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in fault.service.lower())[:35]
        name = f"twin-{mechanism.replace('_', '-')}-{suffix}"[:63].rstrip("-")
        if mechanism in {"container_kill", "pod_failure", "pod_kill", "container_stop"}:
            containers = (original.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []) or [])
            if not containers:
                raise ValueError("selected Deployment has no container to kill")
            kind = "podchaos"
            chaos_action = POD_CHAOS_ACTIONS[mechanism]
            experiment = {
                "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos",
                "metadata": {
                    "name": name,
                    "namespace": session.namespace,
                    "labels": {
                        "aiopslab.ibm/managed-by": "sparse-live-twin",
                        "aiopslab.ibm/predicted-fault": "true",
                    },
                },
                "spec": {
                    "action": chaos_action,
                    "mode": "all",
                    "selector": {"namespaces": [session.namespace], "labelSelectors": match_labels},
                },
            }
            if mechanism in {"container_kill", "container_stop"}:
                # Upstream pins the container name to one application's literal, so
                # it kills nothing whenever the target differs. Resolve the name
                # from the selected workload instead.
                target_container = next(
                    (
                        row for row in containers
                        if str(row.get("name") or "").strip().lower().endswith(
                            fault.service.strip().lower()
                        )
                    ),
                    containers[0],
                )
                experiment["spec"]["containerNames"] = [str(target_container["name"])]
        else:
            kind = "networkchaos"
            experiment = {
                "apiVersion": "chaos-mesh.org/v1alpha1", "kind": "NetworkChaos",
                "metadata": {
                    "name": name,
                    "namespace": session.namespace,
                    "labels": {
                        "aiopslab.ibm/managed-by": "sparse-live-twin",
                        "aiopslab.ibm/predicted-fault": "true",
                    },
                },
                "spec": {
                    "action": "delay" if mechanism == "network_delay" else "loss",
                    "mode": "all",
                    "selector": {"namespaces": [session.namespace], "labelSelectors": match_labels},
                },
            }
            if mechanism == "network_delay":
                latency = {"delay_100ms": "100ms", "delay_300ms": "300ms",
                           "delay_1000ms": "1000ms"}.get(fault.variant_name, "1000ms")
                experiment["spec"]["delay"] = {"latency": latency, "correlation": "100", "jitter": "0ms"}
            else:
                loss = {"loss_5pct": "5", "loss_20pct": "20", "loss_50pct": "50"}.get(
                    fault.variant_name, "99"
                )
                experiment["spec"]["loss"] = {"loss": loss, "correlation": "100"}
        _ok(_run(["apply", "-f", "-"], experiment), f"inject predicted {mechanism}")
        return LiveFaultHandle(
            session, fault, original, restore_mode="delete_custom_resource",
            injected_details={"kind": kind, "name": name},
        )
    if mechanism == "target_port_misconfig":
        if fault.service not in selected_services:
            raise ValueError("predicted service is not a selected Twin Service")
        original_service = next(
            deepcopy(obj) for obj in session.bundle.objects
            if obj.get("kind") == "Service"
            and (obj.get("metadata", {}) or {}).get("name") == fault.service
        )
        faulted_service = deepcopy(original_service)
        ports = faulted_service["spec"].get("ports", []) or []
        if not ports:
            raise ValueError("selected Service has no ports to corrupt")
        used = {p.get("port") for p in ports} | {p.get("targetPort") for p in ports}
        invalid_port = next(port for port in range(65535, 65000, -1) if port not in used)
        ports[0]["targetPort"] = invalid_port
        _ok(_run(["apply", "-f", "-"], faulted_service), "inject predicted targetPort misconfiguration")
        return LiveFaultHandle(
            session, fault, original, restore_mode="apply_service",
            original_service=original_service,
            injected_details={"target_port": invalid_port},
        )
    if mechanism == "scale_replicas_zero":
        replicas = {"scale_2": 2, "scale_3": 3}.get(str(fault.variant_name), 0)
        _ok(_run([
            "scale", f"deployment/{fault.service}",
            "-n", session.namespace, f"--replicas={replicas}",
        ]), "inject predicted replica-count variant")
        return LiveFaultHandle(
            session, fault, original, restore_mode="scale_replicas",
            injected_details={"replicas": replicas},
        )
    if mechanism == "mongodb_auth_missing":
        # The fault is "the application can no longer authenticate to its
        # datastore". Upstream produces that by switching one specific MongoDB to
        # requireTLS through Helm, which is why it is a no-op for every other
        # target. Enforcing authorization on a server whose clients currently
        # connect anonymously produces the same client-visible failure and is
        # derived entirely from the target's own manifest.
        faulted = deepcopy(original)
        rows = faulted["spec"]["template"]["spec"].get("containers", []) or []
        if not rows:
            raise ValueError("selected Deployment has no container to mutate")
        target = rows[0]
        args = [str(x) for x in (target.get("args") or [])]
        if "--auth" in args:
            raise NotImplementedError(
                "target MongoDB already enforces authorization; "
                "an authentication-missing counterfactual is not available"
            )
        target["args"] = [*args, "--auth"]
        _ok(_run(["apply", "-f", "-"], faulted), "enforce MongoDB authorization")
        return LiveFaultHandle(
            session, fault, original, restore_mode="apply_deployment",
            injected_details={
                "args": target["args"],
                "strategy": "enforce_authorization_on_anonymous_datastore",
                "container": str(target.get("name") or ""),
            },
        )
    if mechanism in {"mongodb_auth_revoked", "mongodb_user_unregistered"}:
        pod = _deployment_pod(session, original)
        access = discover_mongo_access(session.bundle, fault.service, session.namespace, pod)
        require_auth_catalog(access)
        users = list_mongo_users(access, session.namespace, pod)
        user = select_revocable_user(access, users)
        quoted = json.dumps(user.username)
        admin = "db.getSiblingDB('admin')"
        if mechanism == "mongodb_auth_revoked":
            roles = json.dumps(user.roles)
            mutation = f"{admin}.revokeRolesFromUser({quoted}, {roles});"
            probe = (
                f"var u={admin}.getUser({quoted});"
                "print('AIOPS_REVOKED=' + (u != null && (u.roles || []).length === 0));"
            )
            restore = f"{admin}.grantRolesToUser({quoted}, {roles});"
            marker = "AIOPS_REVOKED=true"
        else:
            roles = json.dumps(user.roles)
            mutation = f"{admin}.dropUser({quoted});"
            probe = f"print('AIOPS_USER_MISSING=' + ({admin}.getUser({quoted}) == null));"
            # The password is not recoverable from the catalog, so restoration
            # recreates the account with a deterministic Twin-local credential and
            # the exact role set that was captured before the drop.
            restore = (
                f"{admin}.createUser({{user: {quoted}, "
                f"pwd: {json.dumps('aiops-twin-restored')}, roles: {roles}}});"
            )
            marker = "AIOPS_USER_MISSING=true"
        _pod_exec(
            session.namespace, pod, access.shell_command(mutation),
            f"inject predicted {mechanism}",
        )
        return LiveFaultHandle(
            session, fault, original, restore_mode="mongodb_exec",
            injected_details={
                "manifestation_probe": access.shell_command(probe),
                "manifestation_marker": marker,
                "restore_command": access.shell_command(restore),
                "targeted_user": user.username,
                "targeted_roles": user.roles,
                "shell": access.shell,
                "provenance": access.provenance,
            },
        )
    if mechanism == "wrong_binary":
        swap = discover_binary_swap(session.bundle, fault.service)
        faulted = deepcopy(original)
        rows = faulted["spec"]["template"]["spec"].get("containers", []) or []
        if not rows:
            raise ValueError("selected Deployment has no container to mutate")
        target = next(
            (row for row in rows if str(row.get("name") or "") == swap.container_name),
            rows[0],
        )
        target["command"] = list(swap.faulted_command)
        _ok(_run(["apply", "-f", "-"], faulted), f"inject predicted {mechanism}")
        return LiveFaultHandle(
            session, fault, original, restore_mode="apply_deployment",
            injected_details={
                "command": list(swap.faulted_command),
                "original_command": swap.original_command,
                "container": swap.container_name,
                "strategy": swap.strategy,
                "provenance": swap.provenance,
            },
        )
    if mechanism == "application_config_misconfig":
        corruption = discover_config_corruption(
            session.bundle, fault.service, application_source_root=application_source_root
        )
        details = {
            "target_kind": corruption.target_kind,
            "target_name": corruption.target_name,
            "key": corruption.key,
            "faulted_value": corruption.faulted_value,
            "original_value": corruption.original_value,
            "strategy": corruption.strategy,
            "target": f"{corruption.target_kind}/{corruption.target_name}",
            "provenance": corruption.provenance,
        }
        if corruption.target_kind == "ConfigMap":
            source = next(
                obj for obj in session.bundle.objects
                if obj.get("kind") == "ConfigMap"
                and (obj.get("metadata", {}) or {}).get("name") == corruption.target_name
            )
            original_configmap = deepcopy(source)
            faulted_configmap = deepcopy(source)
            faulted_configmap.setdefault("data", {})[corruption.key] = corruption.faulted_value
            _ok(_run(["apply", "-f", "-"], faulted_configmap),
                "inject predicted configuration corruption")
            _ok(_run([
                "rollout", "restart", f"deployment/{fault.service}", "-n", session.namespace,
            ]), "restart workload onto corrupted configuration")
            return LiveFaultHandle(
                session, fault, original, restore_mode="apply_configmap_and_restart",
                original_configmap=original_configmap, injected_details=details,
            )
        if corruption.target_kind == "ConfigFile":
            # Verify the running image really carries the discovered file and value
            # before overlaying it; a disagreement between source tree and image
            # fails closed rather than injecting an unverified mutation.
            pod = _deployment_pod(session, original)
            live_config = read_container_json_file(
                session.namespace, pod, corruption.container_name, corruption.target_name
            )
            configmap, faulted = config_file_overlay_objects(
                original, corruption, session.namespace, live_config
            )
            _ok(_run(["apply", "-f", "-"], configmap), "create corrupted configuration overlay")
            _ok(_run(["apply", "-f", "-"], faulted), "inject predicted configuration corruption")
            details.update({
                "configmap": str(configmap["metadata"]["name"]),
                "container": corruption.container_name,
                "verified_live_file_pod": pod,
            })
            return LiveFaultHandle(
                session, fault, original, restore_mode="apply_deployment_and_delete_configmap",
                injected_details=details,
            )
        faulted = deepcopy(original)
        rows = faulted["spec"]["template"]["spec"].get("containers", []) or []
        target = next(
            (row for row in rows if str(row.get("name") or "") == corruption.container_name),
            rows[0] if rows else None,
        )
        if target is None:
            raise ValueError("selected Deployment has no container to mutate")
        if corruption.target_kind == "Env":
            for entry in target.setdefault("env", []):
                if str((entry or {}).get("name") or "") == corruption.key:
                    entry["value"] = corruption.faulted_value
                    break
        else:
            target.setdefault("args", [])[int(corruption.key)] = corruption.faulted_value
        _ok(_run(["apply", "-f", "-"], faulted), "inject predicted configuration corruption")
        return LiveFaultHandle(
            session, fault, original, restore_mode="apply_deployment",
            injected_details=details,
        )
    if mechanism != "assign_to_non_existent_node":
        raise NotImplementedError(f"live injector is not implemented for {mechanism!r}")
    faulted = deepcopy(original)
    faulted["spec"]["template"]["spec"]["nodeSelector"] = {
        "kubernetes.io/hostname": "aiops-twin-non-existent-node"
    }
    # Match AIOpsLab's actual mechanism: delete/reapply instead of a rolling
    # patch. This ensures the old healthy ReplicaSet cannot continue serving.
    _ok(_run([
        "delete", "deployment", fault.service,
        "-n", session.namespace, "--wait=true", "--timeout=60s",
    ]), "remove clean Deployment before predicted scheduling fault")
    _ok(_run(["apply", "-f", "-"], faulted), "inject predicted scheduling fault")
    return LiveFaultHandle(session, fault, original)
