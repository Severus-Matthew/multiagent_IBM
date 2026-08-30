from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
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


@dataclass
class SparseManifestBundle:
    """Sanitized, namespace-local objects ready for a later apply stage.

    Rendering is read-only.  In particular, this object does not imply that the
    manifests were applied or that a namespace was created.
    """

    source_namespace: str
    target_namespace: str
    objects: list[dict[str, Any]] = field(default_factory=list)
    object_refs: list[dict[str, str]] = field(default_factory=list)
    rejected_refs: list[dict[str, str]] = field(default_factory=list)
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


def _object(kind: str, name: str, namespace: str) -> dict[str, Any]:
    return _kubectl_json(["get", kind, name, "-n", namespace])


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


_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_SERVER_METADATA = {
    "annotations",
    "creationTimestamp",
    "deletionGracePeriodSeconds",
    "deletionTimestamp",
    "finalizers",
    "generateName",
    "generation",
    "managedFields",
    "ownerReferences",
    "resourceVersion",
    "selfLink",
    "uid",
}


def _validate_namespace(namespace: str) -> str:
    value = str(namespace or "").strip()
    if len(value) > 63 or not _DNS_LABEL.fullmatch(value):
        raise ValueError(f"invalid target namespace: {namespace!r}")
    return value


def _sanitize_metadata(obj: dict[str, Any], target_namespace: str) -> None:
    metadata = obj.setdefault("metadata", {})
    for key in _SERVER_METADATA:
        metadata.pop(key, None)
    metadata["namespace"] = target_namespace
    labels = metadata.setdefault("labels", {})
    labels["aiopslab.ibm/twin"] = "sparse-live"
    labels["aiopslab.ibm/managed-by"] = "sparse-live-twin"


def _sanitize_for_clone(
    obj: dict[str, Any], target_namespace: str, source_namespace: str | None = None
) -> dict[str, Any]:
    clone = deepcopy(obj)
    clone.pop("status", None)
    _sanitize_metadata(clone, target_namespace)

    kind = str(clone.get("kind") or "")
    spec = clone.get("spec", {}) or {}
    if kind == "Service":
        # These fields are allocated by the destination cluster.  Retaining
        # them can collide with the source Service or create a non-portable
        # manifest.  External/LB identity is deliberately not cloned.
        for key in (
            "clusterIP", "clusterIPs", "healthCheckNodePort", "ipFamilies",
            "ipFamilyPolicy", "loadBalancerClass", "loadBalancerIP",
        ):
            spec.pop(key, None)
        spec["type"] = "ClusterIP"
        for port in spec.get("ports", []) or []:
            if isinstance(port, dict):
                port.pop("nodePort", None)
    elif kind == "ServiceAccount":
        # Bound token secrets are namespace/runtime state, not portable config.
        clone.pop("secrets", None)
        spec.pop("secrets", None)
        spec.pop("imagePullSecrets", None)
    elif kind in {"Deployment", "StatefulSet"}:
        spec.pop("progressDeadlineSeconds", None)
        if kind == "Deployment":
            spec.pop("revisionHistoryLimit", None)
        # A sparse Twin starts with one replica per selected logical service;
        # scale faults are applied later to this clean baseline.
        spec["replicas"] = 1
        template_meta = (
            spec.setdefault("template", {}).setdefault("metadata", {})
        )
        for key in _SERVER_METADATA:
            template_meta.pop(key, None)
    elif kind == "ConfigMap":
        clone.pop("immutable", None)
    elif kind == "Secret":
        # Service-account tokens are runtime credentials and must never be
        # copied across namespaces.
        if str(clone.get("type") or "") == "kubernetes.io/service-account-token":
            raise ValueError(
                f"refusing to clone service-account token Secret {_name(clone)!r}"
            )
    clone["spec"] = spec if "spec" in clone else clone.get("spec")
    if clone.get("spec") is None:
        clone.pop("spec", None)
    if source_namespace:
        source_suffix = f".{source_namespace}.svc.cluster.local"
        target_suffix = f".{target_namespace}.svc.cluster.local"

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(source_suffix, target_suffix)
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            return value

        clone = rewrite(clone)
    return clone


def render_sparse_manifest_bundle(
    plan: SparseManifestPlan,
    target_namespace: str,
) -> SparseManifestBundle:
    """Read source objects and render portable copies without mutating K8s.

    PVC cloning is intentionally unsupported until a storage policy is chosen;
    silently pointing a Twin at source storage would violate isolation.
    Missing or ambiguous discovery results also fail closed.
    """
    target = _validate_namespace(target_namespace)
    if target == plan.source_namespace:
        raise ValueError("target namespace must differ from source namespace")
    if plan.missing_selected_controllers or plan.missing_required_refs:
        raise ValueError("cannot render an incomplete sparse manifest plan")
    if plan.persistent_volume_claims:
        raise ValueError("PVC-backed sparse Twins require an explicit storage clone policy")

    refs: list[tuple[str, str]] = []
    for row in plan.controllers:
        refs.append((str(row["kind"]), str(row["name"])))
    refs.extend(("Service", str(row["name"])) for row in plan.service_objects)
    refs.extend(("ConfigMap", name) for name in plan.configmaps)
    refs.extend(("Secret", name) for name in plan.secrets)
    # The destination namespace creates its own default ServiceAccount.
    refs.extend(
        ("ServiceAccount", name)
        for name in plan.service_accounts
        if name != "default"
    )

    objects: list[dict[str, Any]] = []
    object_refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, name in refs:
        key = (kind, name)
        if key in seen:
            continue
        seen.add(key)
        raw = _object(kind.lower(), name, plan.source_namespace)
        clone = _sanitize_for_clone(raw, target, plan.source_namespace)
        objects.append(clone)
        object_refs.append({"kind": kind, "name": name})

    order = {
        "ServiceAccount": 0,
        "Secret": 1,
        "ConfigMap": 2,
        "Service": 3,
        "Deployment": 4,
        "StatefulSet": 4,
    }
    paired = sorted(
        zip(objects, object_refs),
        key=lambda pair: (order.get(pair[1]["kind"], 99), pair[1]["kind"], pair[1]["name"]),
    )
    return SparseManifestBundle(
        source_namespace=plan.source_namespace,
        target_namespace=target,
        objects=[pair[0] for pair in paired],
        object_refs=[pair[1] for pair in paired],
        read_only=True,
    )
