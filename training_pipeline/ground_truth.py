from __future__ import annotations

from typing import Any
from .schemas import FaultLabel, normalize_fault_type


def labels_from_fault_context(fault_context: dict[str, Any]) -> list[FaultLabel]:
    """Extract oracle labels from full state only. Never call from agents."""
    labels: list[FaultLabel] = []
    instances = fault_context.get("fault_instances") or []
    if not instances and fault_context.get("faulty_service"):
        instances = [{
            "faulty_service": fault_context.get("faulty_service"),
            "fault_family": fault_context.get("fault_family"),
            "variant_name": fault_context.get("variant_name", "default"),
            "variant_params": fault_context.get("variant_params", {}),
        }]
    for inst in instances:
        svc = inst.get("faulty_service") or inst.get("service")
        if not svc:
            continue
        family = inst.get("fault_family") or fault_context.get("fault_family") or ""
        labels.append(FaultLabel(
            service=str(svc),
            fault_type=normalize_fault_type(family),
            fault_family=str(family),
            variant_name=str(inst.get("variant_name", "default")),
            metadata={
                "variant_params": inst.get("variant_params", {}),
                "task": inst.get("task") or fault_context.get("task"),
                "app": inst.get("app") or fault_context.get("app"),
            },
        ))
    return labels


def labels_from_full_state(full_state: dict[str, Any]) -> list[FaultLabel]:
    return labels_from_fault_context(full_state.get("fault_context", {}) or {})


def ground_truth_summary(full_state: dict[str, Any]) -> dict[str, Any]:
    labels = labels_from_full_state(full_state)
    return {
        "scenario_id": full_state.get("scenario_id"),
        "num_faults": len(labels),
        "is_multifault": len(labels) > 1 or bool((full_state.get("fault_context", {}) or {}).get("is_multifault")),
        "labels": [x.to_dict() for x in labels],
    }
