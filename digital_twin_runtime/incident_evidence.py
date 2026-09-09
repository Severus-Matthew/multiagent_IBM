"""Observable comparisons with a frozen healthy reference; never oracle labels."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def reference_state_from_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    system: dict[str, Any] = {}
    for obj in objects:
        kind, name = obj.get("kind"), obj.get("metadata", {}).get("name")
        spec = obj.get("spec", {}) or {}
        if not name:
            continue
        if kind in {"Deployment", "StatefulSet"}:
            pod_spec = spec.get("template", {}).get("spec", {})
            system.setdefault(name, {})["deployment"] = {
                "replicas_desired": spec.get("replicas", 1),
                "node_selector": pod_spec.get("nodeSelector", {}),
                "containers": [{k: c[k] for k in ("name", "image", "command", "args", "ports") if k in c}
                               for c in pod_spec.get("containers", [])],
            }
        elif kind == "Service":
            system.setdefault(name, {})["service"] = {
                "selector": spec.get("selector", {}),
                "ports": [{"port": p.get("port"), "target_port": p.get("targetPort", p.get("port")),
                           "protocol": p.get("protocol", "TCP")} for p in spec.get("ports", [])],
            }
    return {"system": system}


_CONTROLLER_KINDS = {"Deployment", "StatefulSet"}


def _require_complete_object(obj: Any, kind: str, name: str) -> dict[str, Any]:
    metadata = (obj.get("metadata") if isinstance(obj, dict) else None) or {}
    if (not isinstance(obj, dict) or str(obj.get("kind") or "") != kind
            or str(metadata.get("name") or "") != name or not isinstance(obj.get("spec"), dict)):
        raise ValueError(f"resolved {kind}/{name} is not a complete Kubernetes object")
    return obj


def resolve_reference_objects(plan: Any, namespace: str, fetch: Any) -> list[dict[str, Any]]:
    """Resolve a manifest plan's controller/Service summaries to real objects.

    ``discover_sparse_manifest_plan`` returns summaries (``kind``, ``name``,
    ``logical_service``, ``pod_labels``; Service rows carry ``name``,
    ``selector``, ``ports``), not Kubernetes objects. The healthy reference
    state and the environment fingerprint must be built from actual specs, so
    every reference is fetched with an explicit kind and validated.
    """
    objects: list[dict[str, Any]] = []
    for row in getattr(plan, "controllers", []) or []:
        kind, name = str(row.get("kind") or ""), str(row.get("name") or "")
        if kind not in _CONTROLLER_KINDS or not name:
            raise ValueError(f"controller reference lacks a supported kind/name: {row!r}")
        objects.append(_require_complete_object(fetch(kind, name, namespace), kind, name))
    for row in getattr(plan, "service_objects", []) or []:
        name = str(row.get("name") or "")
        if not name:
            raise ValueError(f"Service reference lacks a name: {row!r}")
        objects.append(_require_complete_object(fetch("Service", name, namespace), "Service", name))
    return objects


def with_reference_deviations(state: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(state)
    deviations: dict[str, Any] = {}
    for service, item in (state.get("system") or {}).items():
        expected = (reference.get("system") or {}).get(service, {})
        if not isinstance(item, dict):
            continue
        got = (item.get("deployment") or {}).get("replicas_desired")
        desired = (expected.get("deployment") or {}).get("replicas_desired")
        reasons = []
        if got is not None and desired is not None and got != desired:
            reasons.append({"field": "replicas_desired", "observed": got, "reference": desired})
        actual_service = item.get("service") or item.get("services") or {}
        expected_service = expected.get("service") or {}
        if actual_service and expected_service:
            def ports(x):
                return sorted((str(p.get("port")), str(p.get("target_port", p.get("targetPort", p.get("port")))),
                               str(p.get("protocol", "TCP"))) for p in x.get("ports", []))
            for field in ("selector", "ports"):
                observed = ports(actual_service) if field == "ports" else actual_service.get(field)
                target = ports(expected_service) if field == "ports" else expected_service.get(field)
                if observed is not None and observed != target:
                    reasons.append({"field": "service." + field, "observed": observed, "reference": target})
        if reasons:
            deviations[service] = reasons
    out["observed_deviations"] = deviations
    return out
