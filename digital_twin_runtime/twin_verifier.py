from __future__ import annotations

from typing import Any

from training_pipeline.schemas import FaultLabel

from .sla_verifier import signature_summary, sla_verdict_from_signature, symptom_reduction
from .telemetry_comparator import graph_neighborhood, score_prediction_reproduction, symptom_signature


class BehavioralTwinVerifier:
    """Offline Stage-2 verifier.

    RCA validation uses a prediction-sensitive behavioral proxy over the redacted
    compressed state. It scores whether the predicted RCA service/fault type is
    supported by observed degraded services, log-error services, trace-edge
    anomalies, metric anomalies, and graph-neighborhood coverage.

    Action validation simulates the predicted remediation on the same observable
    symptom abstraction and computes before/after SLA-style violation counts. The
    live K8s twin should replace this simulator later, but the interface already
    separates command safety, twin outcome, and SLA restoration.
    """

    def validate_rca_prediction(
        self,
        full_state: dict[str, Any],
        compressed_state: dict[str, Any],
        predicted_faults: list[FaultLabel],
    ) -> dict[str, Any]:
        # Use only agent-visible evidence plus the predicted RCA labels for the
        # reproduction score. The previous version simulated on top of the
        # already-faulted full state, which made overlap almost constant.
        result = score_prediction_reproduction(
            compressed_state,
            [f.to_dict() for f in predicted_faults],
        )
        result["mode"] = "behavioral_offline_proxy"
        result["uses_full_state_for_rca_score"] = False
        result["uses_oracle_labels"] = False
        result["predicted_fault_injection_checked"] = True
        return result

    def apply_action_and_score(
        self,
        full_state: dict[str, Any],
        rca_faults: list[FaultLabel],
        mitigation_action: dict[str, Any],
        compressed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a remediation action using behavioral twin + SLA-style verifier.

        Args:
            full_state: kept for API compatibility and future live-twin hooks.
            rca_faults: final RCA faults that passed the RCA gate.
            mitigation_action: normalized command action from command_normalizer.
            compressed_state: redacted symptom abstraction to simulate before/after.
        """
        observable_state = compressed_state or full_state
        target = mitigation_action.get("service")
        action = mitigation_action.get("action")
        matched_fault, match_error = _select_matched_fault(action, target, rca_faults)
        before_signature = symptom_signature(observable_state)
        before_sla = sla_verdict_from_signature(before_signature)

        if not matched_fault:
            return {
                "mode": "behavioral_offline",
                "resolved": False,
                "twin_resolved": False,
                "sla_restored": False,
                "target_sla_restored": False,
                "symptom_reduction": 0.0,
                "global_symptom_reduction": 0.0,
                "target_symptom_reduction": 0.0,
                "reason": match_error or "action_target_does_not_match_rca_fault",
                "target_service": target,
                "mitigation_action": mitigation_action,
                "before_sla": before_sla,
                "after_sla": before_sla,
                "before_signature_summary": signature_summary(before_signature),
                "after_signature_summary": signature_summary(before_signature),
            }

        repaired = _action_repairs_fault(action, matched_fault.fault_type)
        after_signature = _simulate_after_signature(
            observable_state,
            before_signature,
            matched_fault=matched_fault,
            mitigation_action=mitigation_action,
            repaired=repaired,
        )
        after_sla = sla_verdict_from_signature(after_signature)
        global_reduction = symptom_reduction(before_sla, after_sla)
        target_before = _target_sla(before_signature, matched_fault.service)
        target_after = _target_sla(after_signature, matched_fault.service)
        target_reduction = symptom_reduction(target_before, target_after)
        target_sla_restored = bool(target_after.get("sla_restored"))
        sla_restored = bool(after_sla.get("sla_restored"))

        # In offline mode, allow target-centric success if the repaired root-cause
        # service no longer violates its local SLA, even when unrelated cascade
        # artifacts remain in the compressed abstraction.
        twin_resolved = bool(repaired and target_sla_restored)
        resolved = bool(twin_resolved and (sla_restored or target_reduction >= 0.95))

        return {
            "mode": "behavioral_offline",
            "resolved": resolved,
            "twin_resolved": twin_resolved,
            "sla_restored": sla_restored,
            "target_sla_restored": target_sla_restored,
            "symptom_reduction": global_reduction,
            "global_symptom_reduction": global_reduction,
            "target_symptom_reduction": target_reduction,
            "target_service": matched_fault.service,
            "target_fault_type": matched_fault.fault_type,
            "mitigation_action": mitigation_action,
            "action_repairs_fault_type": repaired,
            "reason": _action_reason(repaired, resolved, target_sla_restored, sla_restored),
            "before_sla": before_sla,
            "after_sla": after_sla,
            "target_before_sla": target_before,
            "target_after_sla": target_after,
            "before_signature_summary": signature_summary(before_signature),
            "after_signature_summary": signature_summary(after_signature),
            "after_sla_object": {"violated": not sla_restored, "offline_behavioral": True},
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


def _simulate_after_signature(
    state: dict[str, Any],
    before: dict[str, Any],
    matched_fault: FaultLabel,
    mitigation_action: dict[str, Any],
    repaired: bool,
) -> dict[str, Any]:
    after = {k: list(v) if isinstance(v, list) else v for k, v in before.items()}
    if not repaired:
        return after

    target = matched_fault.service
    radius = 1
    if mitigation_action.get("action") in {"scale_service", "rollback_config", "fix_infra_scheduling"}:
        radius = 0
    affected_scope = set(graph_neighborhood(state, target, radius=radius)) | {target}

    for field in [
        "degraded_services",
        "top_error_services",
        "trace_sources",
        "trace_targets",
        "metric_anomaly_services",
        "affected_services",
    ]:
        values = after.get(field, []) or []
        if isinstance(values, list):
            after[field] = [v for v in values if str(v) not in affected_scope]

    failed_edges = after.get("failed_edges", []) or []
    if isinstance(failed_edges, list):
        cleaned = []
        for edge in failed_edges:
            e = str(edge)
            touched = False
            for svc in affected_scope:
                if e == svc or e.startswith(svc + "->") or e.endswith("->" + svc) or ("->" + svc + "->") in e:
                    touched = True
                    break
            if not touched:
                cleaned.append(edge)
        after["failed_edges"] = cleaned

    return after


def _target_sla(signature: dict[str, Any], service: str) -> dict[str, Any]:
    target_sig: dict[str, Any] = {}
    for field in ["degraded_services", "top_error_services", "trace_sources", "trace_targets", "metric_anomaly_services", "affected_services"]:
        values = signature.get(field, []) or []
        target_sig[field] = [v for v in values if str(v) == service] if isinstance(values, list) else []
    failed_edges = signature.get("failed_edges", []) or []
    if isinstance(failed_edges, list):
        target_sig["failed_edges"] = [
            e for e in failed_edges
            if str(e).startswith(service + "->") or str(e).endswith("->" + service)
        ]
    else:
        target_sig["failed_edges"] = []
    return sla_verdict_from_signature(target_sig)


def _action_reason(repaired: bool, resolved: bool, target_sla_restored: bool, sla_restored: bool) -> str:
    if not repaired:
        return "action_does_not_repair_target_fault_type"
    if resolved:
        return "target_fault_repaired_and_sla_improved"
    if target_sla_restored and not sla_restored:
        return "target_fault_repaired_but_global_cascade_remains"
    return "action_valid_but_sla_not_restored"
