from __future__ import annotations

"""Audited live-Twin mechanism/workload capability registry."""

from typing import Any

from training_pipeline.schemas import FaultLabel, normalize_fault_mechanism


# A pair is admitted only after injection, root-reaching workload, telemetry,
# remediation, recovery, and cleanup have all passed live Kubernetes audits.
AUDITED_LIVE_FAULT_WORKLOADS: dict[tuple[str, str], dict[str, Any]] = {
    ("user-timeline-service", "assign_to_non_existent_node"): {
        "application": "social-network",
        "workload": "read-user-timeline",
        "remediation": "remove_node_selector",
        "audit": "PASS_SPARSE_LIVE_JOINT_ROLLOUT",
        "supported_variants": ["default"],
    },
    ("user-timeline-service", "scale_replicas_zero"): {
        "application": "social-network",
        "workload": "read-user-timeline",
        "remediation": "scale_to_one_replica",
        "audit": "PASS_LIVE_ACTION_REMEDIATION",
        "supported_variants": ["default", "scale_0"],
    },
    ("user-timeline-service", "target_port_misconfig"): {
        "application": "social-network",
        "workload": "read-user-timeline",
        "remediation": "restore_service_target_port_9090",
        "audit": "PASS_LIVE_ACTION_REMEDIATION",
        "supported_variants": ["default"],
    },
}

IMPLEMENTED_LIVE_INJECTORS = {
    ("user-timeline-service", "assign_to_non_existent_node"): {"default"},
    ("user-timeline-service", "scale_replicas_zero"): {"default", "scale_0"},
    ("user-timeline-service", "target_port_misconfig"): {"default"},
}


def live_capability(fault: FaultLabel) -> dict[str, Any] | None:
    key = (str(fault.service), normalize_fault_mechanism(fault.fault_mechanism))
    row = AUDITED_LIVE_FAULT_WORKLOADS.get(key)
    if not row or str(fault.variant_name or "default") not in row["supported_variants"]:
        return None
    return dict(row)


def live_injector_implemented(fault: FaultLabel) -> bool:
    key = (str(fault.service), normalize_fault_mechanism(fault.fault_mechanism))
    variants = IMPLEMENTED_LIVE_INJECTORS.get(key, set())
    return str(fault.variant_name or "default") in variants


def audit_live_training_records(records: list[Any]) -> dict[str, Any]:
    """Check evaluator labels only; these details never enter policy inputs."""
    from training_pipeline.ground_truth import labels_from_full_state

    supported: list[str] = []
    unsupported: list[dict[str, Any]] = []
    for record in records:
        labels = labels_from_full_state(record.full_state)
        missing = [
            label.to_dict() for label in labels
            if live_capability(label) is None
        ]
        if labels and not missing:
            supported.append(str(record.scenario_id))
        else:
            unsupported.append({
                "scenario_id": str(record.scenario_id),
                "reason": "missing_or_unaudited_fault_workload_pair",
                "unsupported_labels": missing,
            })
    return {
        "num_records": len(records),
        "num_supported": len(supported),
        "num_unsupported": len(unsupported),
        "supported_scenario_ids": supported,
        "unsupported": unsupported,
        "all_supported": bool(records) and not unsupported,
        "registry": [
            {"service": service, "mechanism": mechanism, **metadata}
            for (service, mechanism), metadata in sorted(AUDITED_LIVE_FAULT_WORKLOADS.items())
        ],
    }
