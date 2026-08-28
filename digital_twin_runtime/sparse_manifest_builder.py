from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class SparseManifestPlan:
    source_namespace: str
    target_namespace: str
    selected_services: list[str]
    controllers: list[dict[str, str]] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    configmaps: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    serviceaccounts: list[str] = field(default_factory=list)
    persistentvolumeclaims: list[str] = field(default_factory=list)
    missing_controllers: list[str] = field(default_factory=list)
    missing_services: list[str] = field(default_factory=list)
    missing_configmaps: list[str] = field(default_factory=list)
    missing_secrets: list[str] = field(default_factory=list)
    missing_serviceaccounts: list[str] = field(default_factory=list)
    missing_persistentvolumeclaims: list[str] = field(default_factory=list)
    service_port_sanitization: list[str] = field(default_factory=list)
    resource_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kubectl_json(*args: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"kubectl {' '.join(args)} failed")
    return json.loads(proc.stdout or "{}")


def _items(kind: str, namespace: str) -> list[dict[str, Any]]:
    obj = _kubectl_json("get", kind, "-n", namespace)
    return list(obj.get("items", []) or [])


def _name(obj: dict[str, Any]) -> str:
    return str((obj.get("metadata", {}) or {}).get("name") or "")


def _labels(obj: dict[str, Any]) -> dict[str, str]:
    raw = (obj.get("metadata", {}) or {}).get("labels", {}) or {}
    return {str(k): str(v) for k, v in raw.items()}


def _matches_service(obj: dict[str, Any], service: str) -> bool:
    if _name(obj) == service:
        return True
    labels = _labels(obj)
    candidate_values = {
        labels.get("service"),
        labels.get("app"),
        labels.get("app.kubernetes.io/name"),
        labels.get("k8s-app"),
    }
    return service in {x for x in candidate_values if x}


def _matching(items: Iterable[dict[str, Any]], service: str) -> list[dict[str, Any]]:
    return [obj for obj in items if _matches_service(obj, service)]


def _pod_spec(controller: dict[str, Any]) -> dict[str, Any]:
    kind = str(controller.get("kind") or "")
    if kind in {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"}:
        return (((controller.get("spec", {}) or {}).get("template", {}) or {}).get("spec", {}) or {})
    return {}


def _container_specs(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return list(pod_spec.get("initContainers", []) or []) + list(pod_spec.get("containers", []) or [])


def _pod_dependencies(pod_spec: dict[str, Any]) -> dict[str, set[str]]:
    configmaps: set[str] = set()
    secrets: set[str] = set()
    pvcs: set[str] = set()
    serviceaccounts: set[str] = set()

    sa = str(pod_spec.get("serviceAccountName") or "default")
    if sa and sa != "default":
        serviceaccounts.add(sa)

    for item in pod_spec.get("imagePullSecrets", []) or []:
        if isinstance(item, dict) and item.get("name"):
            secrets.add(str(item["name"]))

    for vol in pod_spec.get("volumes", []) or []:
        if not isinstance(vol, dict):
            continue
        cm = vol.get("configMap") or {}
        sec = vol.get("secret") or {}
        pvc = vol.get("persistentVolumeClaim") or {}
        if cm.get("name"):
            configmaps.add(str(cm["name"]))
        if sec.get("secretName"):
            secrets.add(str(sec["secretName"]))
        if pvc.get("claimName"):
            pvcs.add(str(pvc["claimName"]))
        projected = vol.get("projected") or {}
        for source in projected.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            if (source.get("configMap") or {}).get("name"):
                configmaps.add(str(source["configMap"]["name"]))
            if (source.get("secret") or {}).get("name"):
                secrets.add(str(source["secret"]["name"]))

    for container in _container_specs(pod_spec):
        for env_from in container.get("envFrom", []) or []:
            cm = env_from.get("configMapRef") or {}
            sec = env_from.get("secretRef") or {}
            if cm.get("name"):
                configmaps.add(str(cm["name"]))
            if sec.get("name"):
                secrets.add(str(sec["name"]))
        for env in container.get("env", []) or []:
            value_from = env.get("valueFrom") or {}
            cm = value_from.get("configMapKeyRef") or {}
            sec = value_from.get("secretKeyRef") or {}
            if cm.get("name"):
                configmaps.add(str(cm["name"]))
            if sec.get("name"):
                secrets.add(str(sec["name"]))

    return {
        "configmaps": configmaps,
        "secrets": secrets,
        "persistentvolumeclaims": pvcs,
        "serviceaccounts": serviceaccounts,
    }


def _service_needs_port_sanitization(service: dict[str, Any]) -> bool:
    spec = service.get("spec", {}) or {}
    if str(spec.get("type") or "ClusterIP") in {"NodePort", "LoadBalancer"}:
        return True
    for port in spec.get("ports", []) or []:
        if isinstance(port, dict) and port.get("nodePort") is not None:
            return True
    return False


def _resource_names(items: Iterable[dict[str, Any]]) -> set[str]:
    return {_name(x) for x in items if _name(x)}


def build_sparse_manifest_plan(
    *,
    source_namespace: str,
    target_namespace: str,
    selected_services: Iterable[str],
) -> SparseManifestPlan:
    """Build a read-only clone plan from the currently deployed application.

    The function performs only ``kubectl get`` operations. It never creates,
    patches, deletes, or applies resources. The resulting plan is intended to be
    the gate before constructing a dedicated sparse live-Twin namespace.
    """
    selected = sorted({str(s) for s in selected_services if s})
    if not selected:
        raise ValueError("selected_services cannot be empty")
    if not source_namespace:
        raise ValueError("source_namespace cannot be empty")
    if not target_namespace:
        raise ValueError("target_namespace cannot be empty")
    if source_namespace == target_namespace:
        raise ValueError("target_namespace must differ from source_namespace")

    deployments = _items("deployments", source_namespace)
    statefulsets = _items("statefulsets", source_namespace)
    services = _items("services", source_namespace)
    configmaps = _items("configmaps", source_namespace)
    secrets = _items("secrets", source_namespace)
    serviceaccounts = _items("serviceaccounts", source_namespace)
    pvcs = _items("persistentvolumeclaims", source_namespace)

    available_configmaps = _resource_names(configmaps)
    available_secrets = _resource_names(secrets)
    available_serviceaccounts = _resource_names(serviceaccounts)
    available_pvcs = _resource_names(pvcs)

    chosen_controllers: list[dict[str, str]] = []
    chosen_services: set[str] = set()
    required_configmaps: set[str] = set()
    required_secrets: set[str] = set()
    required_serviceaccounts: set[str] = set()
    required_pvcs: set[str] = set()
    missing_controllers: list[str] = []
    missing_services: list[str] = []
    sanitize_ports: set[str] = set()

    controller_pool: list[dict[str, Any]] = []
    for obj in deployments:
        obj = dict(obj)
        obj["kind"] = "Deployment"
        controller_pool.append(obj)
    for obj in statefulsets:
        obj = dict(obj)
        obj["kind"] = "StatefulSet"
        controller_pool.append(obj)

    for service_name in selected:
        controllers = _matching(controller_pool, service_name)
        if not controllers:
            missing_controllers.append(service_name)
        else:
            # Multiple matching controllers are retained deliberately; the later
            # clone stage will reproduce all controller resources associated with
            # the selected logical service.
            for controller in controllers:
                chosen_controllers.append({
                    "kind": str(controller.get("kind") or ""),
                    "name": _name(controller),
                    "logical_service": service_name,
                })
                deps = _pod_dependencies(_pod_spec(controller))
                required_configmaps |= deps["configmaps"]
                required_secrets |= deps["secrets"]
                required_serviceaccounts |= deps["serviceaccounts"]
                required_pvcs |= deps["persistentvolumeclaims"]

        service_objs = _matching(services, service_name)
        if not service_objs:
            missing_services.append(service_name)
        for svc in service_objs:
            chosen_services.add(_name(svc))
            if _service_needs_port_sanitization(svc):
                sanitize_ports.add(_name(svc))

    # kube-root-ca.crt is injected automatically in namespaces and should not be
    # cloned as application configuration.
    required_configmaps.discard("kube-root-ca.crt")

    missing_configmaps = sorted(required_configmaps - available_configmaps)
    missing_secrets = sorted(required_secrets - available_secrets)
    missing_serviceaccounts = sorted(required_serviceaccounts - available_serviceaccounts)
    missing_pvcs = sorted(required_pvcs - available_pvcs)

    blockers = (
        len(missing_controllers)
        + len(missing_services)
        + len(missing_configmaps)
        + len(missing_secrets)
        + len(missing_serviceaccounts)
        + len(missing_pvcs)
    )

    plan = SparseManifestPlan(
        source_namespace=source_namespace,
        target_namespace=target_namespace,
        selected_services=selected,
        controllers=sorted(chosen_controllers, key=lambda x: (x["logical_service"], x["kind"], x["name"])),
        services=sorted(chosen_services),
        configmaps=sorted(required_configmaps),
        secrets=sorted(required_secrets),
        serviceaccounts=sorted(required_serviceaccounts),
        persistentvolumeclaims=sorted(required_pvcs),
        missing_controllers=sorted(missing_controllers),
        missing_services=sorted(missing_services),
        missing_configmaps=missing_configmaps,
        missing_secrets=missing_secrets,
        missing_serviceaccounts=missing_serviceaccounts,
        missing_persistentvolumeclaims=missing_pvcs,
        service_port_sanitization=sorted(sanitize_ports),
        resource_summary={
            "selected_logical_services": len(selected),
            "controller_resources": len(chosen_controllers),
            "service_resources": len(chosen_services),
            "configmap_dependencies": len(required_configmaps),
            "secret_dependencies": len(required_secrets),
            "serviceaccount_dependencies": len(required_serviceaccounts),
            "persistentvolumeclaim_dependencies": len(required_pvcs),
            "blocking_missing_resources": blockers,
            "read_only": True,
            "cluster_mutated": False,
        },
    )
    return plan
