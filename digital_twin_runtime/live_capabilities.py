from __future__ import annotations

"""Capability-driven admission for live-Twin fault replay.

Capabilities are keyed by mechanism and required Kubernetes object shape, never
by scenario id or service name. Runtime admission inspects the sparse bundle.
"""

from dataclasses import asdict, dataclass
from typing import Any

from training_pipeline.schemas import FaultLabel, normalize_fault_mechanism


@dataclass(frozen=True)
class LiveMechanismCapability:
    mechanism: str
    required_kinds: tuple[str, ...]
    supported_variants: tuple[str, ...]
    remediation: str
    audit: str
    live_reward_eligible: bool = True
    # Distinguishes "a generic adapter exists and fails closed when the target
    # cannot support the mechanism" from "the adapter has been exercised against a
    # live cluster". Admission to live reward requires the former; the latter is
    # tracked so results are never reported as live-verified before they are.
    live_audit_status: str = "verified_live"
    # Evidence channels in which this mechanism leaves a signature that the Twin
    # comparison can score. "structural" means Deployment/endpoint state (replica
    # counts, readiness, scheduling), which every capture records; "traces" and
    # "logs" are request-level channels a historical capture may have missed.
    # Dataset admission uses this to decide whether an incident whose trace
    # export failed still carries comparable evidence for its labeled mechanism.
    evidence_channels: tuple[str, ...] = ("traces", "logs")


STRUCTURAL_EVIDENCE = ("structural", "traces", "logs")
REQUEST_LEVEL_EVIDENCE = ("traces", "logs")


LIVE_MECHANISM_CAPABILITIES: dict[str, LiveMechanismCapability] = {
    # Every adapter below resolves its target from the rendered Twin bundle. The
    # previously disabled entries were disabled because the adapters mirrored
    # AIOpsLab's hard-coded literals and therefore mutated nothing unless the
    # target happened to be the one service those literals named. They are now
    # discovery-driven (see fault_mutation_discovery.py) and refuse the injection
    # when the target genuinely cannot support it.
    "mongodb_auth_missing": LiveMechanismCapability(
        "mongodb_auth_missing", ("Deployment",), ("default",),
        "restore_deployment_snapshot",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "mongodb_auth_revoked": LiveMechanismCapability(
        "mongodb_auth_revoked", ("Deployment",), ("default",),
        "restore_mongodb_role",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "mongodb_user_unregistered": LiveMechanismCapability(
        "mongodb_user_unregistered", ("Deployment",), ("default",),
        "restore_mongodb_user",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "application_config_misconfig": LiveMechanismCapability(
        "application_config_misconfig", ("Deployment",), ("default",),
        "restore_configmap_or_deployment_snapshot",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03_IMAGE_CONFIG_FILE_OVERLAY",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "wrong_binary": LiveMechanismCapability(
        "wrong_binary", ("Deployment",), ("default",),
        "restore_deployment_snapshot",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "assign_to_non_existent_node": LiveMechanismCapability(
        "assign_to_non_existent_node", ("Deployment",), ("default",),
        "restore_deployment_snapshot", "PASS_GENERIC_MANIFEST_DERIVED_SCHEDULING_ADAPTER",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "scale_replicas_zero": LiveMechanismCapability(
        "scale_replicas_zero", ("Deployment",),
        ("default", "scale_0", "scale_2", "scale_3"),
        "restore_original_replica_count", "PASS_GENERIC_MANIFEST_DERIVED_SCALE_ADAPTER",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "target_port_misconfig": LiveMechanismCapability(
        "target_port_misconfig", ("Deployment", "Service"), ("default",),
        "restore_service_snapshot", "PASS_GENERIC_MANIFEST_DERIVED_SERVICE_PORT_ADAPTER",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "container_kill": LiveMechanismCapability(
        "container_kill", ("Deployment",), ("default",),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "pod_failure": LiveMechanismCapability(
        "pod_failure", ("Deployment",), ("default",),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "pod_kill": LiveMechanismCapability(
        "pod_kill", ("Deployment",), ("default",),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "container_stop": LiveMechanismCapability(
        "container_stop", ("Deployment",), ("default",),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03_SYNTHETIC_CORPUS_DERIVED_TARGETS",
        evidence_channels=STRUCTURAL_EVIDENCE,
    ),
    "network_delay": LiveMechanismCapability(
        "network_delay", ("Deployment",),
        ("default", "delay_100ms", "delay_300ms", "delay_1000ms"),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
    "network_loss": LiveMechanismCapability(
        "network_loss", ("Deployment",),
        ("default", "loss_5pct", "loss_20pct", "loss_50pct"),
        "delete_chaos_experiment",
        "VERIFIED_2_OF_2_FULL_LIFECYCLE_2026_09_03",
        evidence_channels=REQUEST_LEVEL_EVIDENCE,
    ),
}

# Historical evidence only; never admitted by the current assessor.
# Current production qualification is a content-bound matched-control manifest.
LIVE_REWARD_CALIBRATION: dict[str, dict[str, Any]] = {
    "assign_to_non_existent_node": {
        "threshold": 0.4702,
        "evidence": "live-threshold-controls-matched-v3",
        "status": "calibrated_matched_positive_negative_controls",
    },
    "scale_replicas_zero": {
        "threshold": 0.4702,
        "evidence": "live-threshold-controls-matched-v3",
        "status": "calibrated_matched_positive_negative_controls",
    },
}


def assess_live_reward_calibration(labels: list[FaultLabel], *, calibration_path: str | None = None,
                                   application_state: dict[str, Any] | None = None) -> dict[str, Any]:
    from .reward_calibration import assess_calibration
    return assess_calibration(labels, calibration_path, application_state)


def _object_refs(bundle: Any | None) -> set[tuple[str, str]]:
    if bundle is None:
        return set()
    return {
        (str(row.get("kind") or ""), str(row.get("name") or ""))
        for row in getattr(bundle, "object_refs", []) if isinstance(row, dict)
    }


def assess_live_capability(
    fault: FaultLabel,
    bundle: Any | None = None,
    *,
    require_verified_live: bool = True,
) -> dict[str, Any]:
    mechanism = normalize_fault_mechanism(fault.fault_mechanism)
    capability = LIVE_MECHANISM_CAPABILITIES.get(mechanism)
    if capability is None:
        return {"supported": False, "reason": "mechanism_adapter_not_implemented", "mechanism": mechanism}
    if not capability.live_reward_eligible:
        return {
            "supported": False, "reason": "adapter_failed_live_reward_audit",
            "mechanism": mechanism, "audit": capability.audit,
        }
    if require_verified_live and capability.live_audit_status != "verified_live":
        return {
            "supported": False,
            "reason": "adapter_not_verified_live",
            "mechanism": mechanism,
            "audit": capability.audit,
            "live_audit_status": capability.live_audit_status,
        }
    variant = str(fault.variant_name or "default")
    if variant not in capability.supported_variants:
        return {
            "supported": False, "reason": "variant_not_supported", "mechanism": mechanism,
            "variant": variant, "supported_variants": list(capability.supported_variants),
        }
    if not fault.is_injectible():
        return {"supported": False, "reason": "fault_type_mechanism_mismatch", "mechanism": mechanism}
    if bundle is not None:
        refs = _object_refs(bundle)
        missing = [kind for kind in capability.required_kinds if (kind, fault.service) not in refs]
        if missing:
            return {
                "supported": False, "reason": "required_selected_objects_missing",
                "mechanism": mechanism, "service": fault.service, "missing_kinds": missing,
            }
    return {"supported": True, **asdict(capability), "service": fault.service, "variant": variant}


def live_capability(fault: FaultLabel, bundle: Any | None = None) -> dict[str, Any] | None:
    assessment = assess_live_capability(fault, bundle)
    return assessment if assessment["supported"] else None


def live_injector_implemented(fault: FaultLabel, bundle: Any | None = None) -> bool:
    return bool(assess_live_capability(
        fault, bundle, require_verified_live=False
    )["supported"])


def audit_live_training_records(
    records: list[Any],
    *,
    admit_weak_evidence: bool = False,
    require_reward_calibration: bool = True,
    calibration_path: str | None = None,
) -> dict[str, Any]:
    """Dataset-level adapter audit; object/workload checks happen at runtime."""
    from training_pipeline.ground_truth import labels_from_full_state
    from training_pipeline.live_dataset_admission import assess_record_for_live_reward

    supported: list[str] = []
    unsupported: list[dict[str, Any]] = []
    for record in records:
        labels = labels_from_full_state(record.full_state)
        record_admission = assess_record_for_live_reward(record, admit_weak_evidence=admit_weak_evidence)
        reward_calibration = assess_live_reward_calibration(labels, calibration_path=calibration_path,
                                                     application_state=record.compressed_state)
        missing = [
            {**label.to_dict(), "capability": assess_live_capability(label)}
            for label in labels if not assess_live_capability(label)["supported"]
        ]
        calibration_ok = bool(
            reward_calibration["eligible"] or not require_reward_calibration
        )
        if labels and not missing and record_admission["eligible"] and calibration_ok:
            supported.append(str(record.scenario_id))
        else:
            unsupported.append({
                "scenario_id": str(record.scenario_id),
                "reason": "live_reward_admission_failed",
                "unsupported_labels": missing,
                "record_admission": record_admission,
                "reward_calibration": reward_calibration,
            })
    return {
        "num_records": len(records), "num_supported": len(supported),
        "num_unsupported": len(unsupported), "supported_scenario_ids": supported,
        "unsupported": unsupported, "all_supported": bool(records) and not unsupported,
        "registry_scope": "mechanism_and_required_object_shape_not_service_or_scenario",
        "registry": [asdict(row) for row in LIVE_MECHANISM_CAPABILITIES.values()],
        "require_reward_calibration": bool(require_reward_calibration),
        "reward_calibration_manifest": calibration_path,
    }
