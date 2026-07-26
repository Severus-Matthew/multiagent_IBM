from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from training_pipeline.schemas import FaultLabel
from .telemetry_comparator import compare_symptoms


def _import_behavioral_simulator():
    repo_root = Path(__file__).resolve().parents[1]
    state_abs = repo_root / "state_abstraction_full"
    if str(state_abs) not in sys.path:
        sys.path.insert(0, str(state_abs))
    from behavioral_simulator import AbstractBehavioralSimulator  # type: ignore
    return AbstractBehavioralSimulator


class BehavioralTwinVerifier:
    """Offline Stage-2 verifier using the existing abstract behavioral simulator.

    RCA validation compares symptom reproduction. Action validation is intentionally
    target-centric in this offline mode: the full observed state already contains
    many cascade/log anomalies, so global SLA alone is too strict for a synthetic
    command smoke test. Real K8s twin verification will replace this later.
    """

    def validate_rca_prediction(self, full_state: dict[str, Any], compressed_state: dict[str, Any],
                                predicted_faults: list[FaultLabel]) -> dict[str, Any]:
        Sim = _import_behavioral_simulator(); sim = Sim(full_state.get("graph", {}))
        twin_state = full_state
        for fault in predicted_faults:
            twin_state, _ = sim.simulate(twin_state, rca_fault={"service": fault.service, "fault_type": fault.fault_type})
        result = compare_symptoms(compressed_state, twin_state)
        result["mode"] = "behavioral_offline"
        return result

    def apply_action_and_score(self, full_state: dict[str, Any], rca_faults: list[FaultLabel],
                               mitigation_action: dict[str, Any]) -> dict[str, Any]:
        target = mitigation_action.get("service")
        action = mitigation_action.get("action")
        matched_fault, match_error = _select_matched_fault(action, target, rca_faults)

        if not matched_fault:
            return {
                "mode": "behavioral_offline",
                "resolved": False,
                "symptom_reduction": 0.0,
                "reason": match_error or "action_target_does_not_match_rca_fault",
                "target_service": target,
                "mitigation_action": mitigation_action,
            }

        repaired = _action_repairs_fault(action, matched_fault.fault_type)
        reduction = 1.0 if repaired else 0.0
        return {
            "mode": "behavioral_offline",
            "resolved": repaired,
            "symptom_reduction": reduction,
            "target_service": matched_fault.service,
            "target_fault_type": matched_fault.fault_type,
            "mitigation_action": mitigation_action,
            "reason": "target_fault_repaired" if repaired else "action_does_not_repair_target_fault",
            "after_sla": {"violated": not repaired, "offline_target_centric": True},
        }


def _select_matched_fault(action: str | None, target: str | None, faults: list[FaultLabel]) -> tuple[FaultLabel | None, str | None]:
    if not faults:
        return None, "no_rca_faults_available"
    if target:
        match = next((f for f in faults if f.service == target), None)
        return match, None if match else "action_target_does_not_match_rca_fault"

    # Untargeted actions such as helm rollback are ambiguous for multifault cases.
    repairable = [f for f in faults if _action_repairs_fault(action, f.fault_type)]
    if len(faults) == 1 and repairable:
        return repairable[0], None
    if len(repairable) == 1:
        return repairable[0], None
    if len(repairable) > 1:
        return None, "ambiguous_untargeted_action_for_multifault"
    return None, "untargeted_action_does_not_repair_any_rca_fault"


def _action_repairs_fault(action: str | None, fault_type: str | None) -> bool:
    action = action or ""
    fault_type = fault_type or "unknown"
    if fault_type == "infra_failure":
        return action in {"fix_infra_scheduling", "recreate_pod", "restart_service"}
    if fault_type in {"config_error", "auth_failure", "dependency_failure"}:
        return action in {"rollback_config", "restart_service"}
    if fault_type in {"latency_degradation", "resource_exhaustion"}:
        return action in {"scale_service", "restart_service"}
    if fault_type in {"pod_failure", "network_failure"}:
        return action in {"restart_service", "recreate_pod"}
    return action in {"restart_service", "rollback_config", "scale_service"}
