from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable


@dataclass
class SparseManifestPlan:
    source_namespace: str
    selected_services: list[str]
    controllers: list[dict[str, Any]] = field(default_factory=list)
    service_objects: list[dict[str, Any]] = field(default_factory=list)
    configmaps: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    persistent_volume_claims: list[str] = field(default_factory=list)
    service_accounts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    missing_selected_controllers: list[str] = field(default_factory=list)
    missing_required_refs: list[dict[str, str]] = field(default_factory=list)
    source_readiness: dict[str, Any] = field(default_factory=dict)
    resource_counts: dict[str, int] = field(default_factory=dict)
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kubectl_json(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kubectl {' '.join(args)} failed: {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout or "{}")


def _items(kind: str, namespace: str) -> list[dict[str, Any]]:
    obj = _kubectl_json(["get", kind, "-n", namespace])
    return list(obj.get("items", []) or [])


def _name(obj: dict[str, Any]) -> str:
    return str((obj.get("metadata", {}) or {}).get("name") or "")


def _labels(obj: dict[str, Any]) -> dict[str, str]:
    raw = (obj.get("metadata", {}) or {}).get("labels", {}) or {}
    return {str(k): str(v) for k, v in raw.items()}


def _controller_pod_labels(obj: dict[str, Any]) -> dict[str, str]:
    raw = (
        obj.get("spec", {})
        .get("template", {})
        .get("metadata", {})
        .get("labels", {})
        or {}
    )
    return {str(k): str(v) for k, v in raw.items()}


def _matches_service_identity(obj: dict[str, Any], service: str) -> bool:
    if _name(obj) == service:
        return True
    labels = _labels(obj)
    pod_labels = _controller_pod_labels(obj)
    candidates = {
        labels.get("service"), labels.get("app"),
        pod_labels.get("service"), pod_labels.get("app"),
    }
    return service in candidates


def _service_selects_controller(svc: dict[str, Any], ctrl: dict[str, Any]) -> bool:
    selector = (svc.get("spec", {}) or {}).get("selector", {}) or {}
    if not selector:
        return False
    pod_labels = _controller_pod_labels(ctrl)
    return all(str(pod_labels.get(str(k))) == str(v) for k, v in selector.items())


def _controller_spec(ctrl: dict[str, Any]) -> dict[str, Any]:
    return (
        ctrl.get("spec", {})
        .get("template", {})
        .get("spec", {})
        or {}
    )


def _collect_refs(ctrl: dict[str, Any]) -> dict[str, set[str]]:
    pod = _controller_spec(ctrl)
    refs: dict[str, set[str]] = {
        "configmaps": set(),
        "secrets": set(),
        "persistent_volume_claims": set(),
        "service_accounts": set(),
        "images": set(),
    }

    service_account = str(pod.get("serviceAccountName") or "default")
    if service_account:
        refs["service_accounts"].add(service_account)

    for image_pull_secret in pod.get("imagePullSecrets", []) or []:
        if isinstance(image_pull_secret, dict) and image_pull_secret.get("name"):
            refs["secrets"].add(str(image_pull_secret["name"]))

    for volume in pod.get("volumes", []) or []:
        if not isinstance(volume, dict):
            continue
        cm = volume.get("configMap") or {}
        sec = volume.get("secret") or {}
        pvc = volume.get("persistentVolumeClaim") or {}
        if cm.get("name"):
            refs["configmaps"].add(str(cm["name"]))
        if sec.get("secretName"):
            refs["secrets"].add(str(sec["secretName"]))
        if pvc.get("claimName"):
            refs["persistent_volume_claims"].add(str(pvc["claimName"]))

    containers = list(pod.get("initContainers", []) or []) + list(pod.get("containers", []) or [])
    for container in containers:
        if not isinstance(container, dict):
            continue
        if container.get("image"):
            refs["images"].add(str(container["image"]))
        for env_from in container.get("envFrom", []) or []:
            if not isinstance(env_from, dict):
                continue
            cm = env_from.get("configMapRef") or {}
            sec = env_from.get("secretRef") or {}
            if cm.get("name"):
                refs["configmaps"].add(str(cm["name"]))
            if sec.get("name"):
                refs["secrets"].add(str(sec["name"]))
        for env in container.get("env", []) or []:
            if not isinstance(env, dict):
                continue
            value_from = env.get("valueFrom") or {}
            cm = value_from.get("configMapKeyRef") or {}
            sec = value_from.get("secretKeyRef") or {}
            if cm.get("name"):
                refs["configmaps"].add(str(cm["name"]))
            if sec.get("name"):
                refs["secrets"].add(str(sec["name"]))

    return refs


def _controller_readiness(ctrl: dict[str, Any], kind: str) -> dict[str, Any]:
    spec = ctrl.get("spec", {}) or {}
    status = ctrl.get("status", {}) or {}
    desired = int(spec.get("replicas", 1) or 0)
    ready = int(status.get("readyReplicas", 0) or 0)
    available = int(status.get("availableReplicas", ready) or 0)
    return {
        "kind": kind,
        "name": _name(ctrl),
        "desired": desired,
        "ready": ready,
        "available": available,
        "healthy": ready == desired and available == desired,
    }


def discover_sparse_manifest_plan(
    source_namespace: str,
    selected_services: Iterable[str],
) -> SparseManifestPlan:
    """Read-only discovery of Kubernetes resources needed by a sparse Twin.

    The function never creates, patches, deletes, or applies cluster resources.
    It only inspects source manifests and resolves direct Kubernetes references
    (ConfigMaps, Secrets, PVCs, ServiceAccounts, images).  Application dependency
    selection is intentionally handled upstream by the fault-conditioned Twin
    planner; this module must not widen service scope on its own.
    """
    selected = sorted({str(x) for x in selected_services if str(x).strip()})

    deployments = _items("deployments", source_namespace)
    statefulsets = _items("statefulsets", source_namespace)
    services = _items("services", source_namespace)
    configmaps = {_name(x): x for x in _items("configmaps", source_namespace)}
    secrets = {_name(x): x for x in _items("secrets", source_namespace)}
    pvcs = {_name(x): x for x in _items("persistentvolumeclaims", source_namespace)}
    serviceaccounts = {_name(x): x for x in _items("serviceaccounts", source_namespace)}

    selected_controllers: list[tuple[str, dict[str, Any], str]] = []
    missing_controllers: list[str] = []

    for service in selected:
        matches: list[tuple[str, dict[str, Any]]] = []
        for obj in deployments:
            if _matches_service_identity(obj, service):
                matches.append(("Deployment", obj))
        for obj in statefulsets:
            if _matches_service_identity(obj, service):
                matches.append(("StatefulSet", obj))

        # Prefer exact-name matches and fail closed on ambiguous fallback matches.
        exact = [(kind, obj) for kind, obj in matches if _name(obj) == service]
        chosen = exact if exact else matches
        if len(chosen) != 1:
            missing_controllers.append(service)
            continue
        kind, obj = chosen[0]
        selected_controllers.append((kind, obj, service))

    selected_service_objects: list[dict[str, Any]] = []
    seen_service_names: set[str] = set()
    for kind, ctrl, logical_service in selected_controllers:
        candidates = [svc for svc in services if _name(svc) == logical_service]
        if not candidates:
            candidates = [svc for svc in services if _service_selects_controller(svc, ctrl)]
        for svc in candidates:
            name = _name(svc)
            if name and name not in seen_service_names:
                selected_service_objects.append(svc)
                seen_service_names.add(name)

    refs = {
        "configmaps": set(),
        "secrets": set(),
        "persistent_volume_claims": set(),
        "service_accounts": set(),
        "images": set(),
    }
    readiness = []
    controller_rows = []
    for kind, ctrl, logical_service in selected_controllers:
        local = _collect_refs(ctrl)
        for key in refs:
            refs[key].update(local[key])
        readiness.append(_controller_readiness(ctrl, kind))
        controller_rows.append({
            "logical_service": logical_service,
            "kind": kind,
            "name": _name(ctrl),
            "pod_labels": _controller_pod_labels(ctrl),
        })

    missing_refs: list[dict[str, str]] = []
    available_by_kind = {
        "ConfigMap": configmaps,
        "Secret": secrets,
        "PersistentVolumeClaim": pvcs,
        "ServiceAccount": serviceaccounts,
    }
    refs_by_kind = {
        "ConfigMap": refs["configmaps"],
        "Secret": refs["secrets"],
        "PersistentVolumeClaim": refs["persistent_volume_claims"],
        "ServiceAccount": refs["service_accounts"],
    }
    for kind, names in refs_by_kind.items():
        available = available_by_kind[kind]
        for name in sorted(names):
            if name not in available:
                missing_refs.append({"kind": kind, "name": name})

    resource_counts = {
        "selected_logical_services": len(selected),
        "controllers": len(controller_rows),
        "service_objects": len(selected_service_objects),
        "configmaps": len(refs["configmaps"]),
        "secrets": len(refs["secrets"]),
        "persistent_volume_claims": len(refs["persistent_volume_claims"]),
        "service_accounts": len(refs["service_accounts"]),
        "images": len(refs["images"]),
    }

    return SparseManifestPlan(
        source_namespace=source_namespace,
        selected_services=selected,
        controllers=controller_rows,
        service_objects=[
            {
                "name": _name(svc),
                "type": str((svc.get("spec", {}) or {}).get("type") or "ClusterIP"),
                "selector": (svc.get("spec", {}) or {}).get("selector", {}) or {},
                "ports": (svc.get("spec", {}) or {}).get("ports", []) or [],
            }
            for svc in selected_service_objects
        ],
        configmaps=sorted(refs["configmaps"]),
        secrets=sorted(refs["secrets"]),
        persistent_volume_claims=sorted(refs["persistent_volume_claims"]),
        service_accounts=sorted(refs["service_accounts"]),
        images=sorted(refs["images"]),
        missing_selected_controllers=sorted(missing_controllers),
        missing_required_refs=missing_refs,
        source_readiness={
            "controllers": readiness,
            "all_selected_source_controllers_healthy": bool(readiness) and all(x["healthy"] for x in readiness),
            "note": "source readiness is diagnostic only; sparse Twin manifests are cloned into a separate namespace",
        },
        resource_counts=resource_counts,
        read_only=True,
    )
