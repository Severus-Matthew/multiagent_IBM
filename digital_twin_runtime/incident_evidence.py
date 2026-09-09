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
