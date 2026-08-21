from __future__ import annotations

from typing import Any

from training_pipeline.ground_truth import labels_from_full_state
from training_pipeline.schemas import FaultLabel, normalize_fault_type

from .counterfactual_replay import score_counterfactual_reproduction
from .sla_verifier import signature_summary, sla_verdict_from_signature, symptom_reduction
from .telemetry_comparator import graph_neighborhood, symptom_signature


class BehavioralTwinVerifier:
    """Offline Stage-2 verifier.

    RCA validation uses an independent counterfactual replay score:
      prediction -> mechanism-specific synthetic symptom footprint -> comparison
      against the observed redacted incident state.

    The hidden injection manifest is logged only as an evaluator-side audit label.
    It never changes the reproduction score. Action validation still uses an
    offline symptom simulator. A live K8s twin can replace both replay functions
    behind the same interface later.
    """

    def validate_rca_prediction(
        self,
        full_state: dict[str, Any],
        compressed_state: dict[str, Any],
        predicted_faults: list[FaultLabel],
    ) -> dict[str, Any]:
        result = score_counterfactual_reproduction(
            compressed_state,
            [f.to_dict() for f in predicted_faults],
        )
        result = dict(result)
        result["offline_injection_anchor_audit"] = _hidden_injection_anchor(full_state, predicted_faults)
        result["uses_full_state_for_rca_score"] = False
        result["uses_oracle_labels"] = False
        result["uses_hidden_injection_manifest_for_score"] = False
        result["counterfactual_prediction_replayed"] = True
        result["predicted_fault_injection_checked"] = False
        return result

    def apply_action_and_score(
        self,
        full_state: dict[str, Any],
        rca_faults: list[FaultLabel],
        mitigation_action: dict[str, Any],
        compressed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a remediation action using offline twin + SLA-style verifier."""
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


def _hidden_injection_anchor(full_state: dict[str, Any], predicted_faults: list[FaultLabel]) -> dict[str, Any]:
    """Evaluator-only audit of prediction vs hidden injected fault manifest."""
    hidden = labels_from_full_state(full_state or {})
    if not hidden:
        return {"available": False, "reason": "no_hidden_injection_manifest"}
    if not predicted_faults:
        return {
            "available": True,
            "all_predicted_faults_match_hidden_injection": False,
            "same_service_wrong_type": False,
            "reason": "empty_prediction",
            "hidden_fault_count": len(hidden),
        }

    exact = 0
    same_service_wrong_type = 0
    unmatched = 0
    per_fault = []
    for pred in predicted_faults:
        pred_type = normalize_fault_type(pred.fault_type or pred.fault_family)
        same_service = [gt for gt in hidden if _same_service(pred.service, gt.service)]
        exact_match = next(
            (
                gt for gt in same_service
                if normalize_fault_type(gt.fault_type or gt.fault_family) == pred_type
            ),
            None,
        )
        if exact_match:
            exact += 1
            status = "exact_hidden_injection_match"
        elif same_service:
            same_service_wrong_type += 1
            status = "same_service_wrong_type"
        else:
            unmatched += 1
            status = "no_hidden_injection_match"
        per_fault.append({
            "service": pred.service,
            "fault_type": pred_type,
            "status": status,
        })

    all_match = exact == len(predicted_faults)
    return {
        "available": True,
        "audit_only": True,
        "used_for_reproduction_score": False,
        "all_predicted_faults_match_hidden_injection": all_match,
        "same_service_wrong_type": bool(same_service_wrong_type),
        "exact_predicted_fault_matches": exact,
        "same_service_wrong_type_count": same_service_wrong_type,
        "unmatched_prediction_count": unmatched,
        "hidden_fault_count": len(hidden),
        "predicted_fault_count": len(predicted_faults),
        "per_fault": per_fault,
    }


def _service_aliases(service: str | None) -> set[str]:
    s = str(service or "").strip()
    if not s:
        return set()
    low = s.lower()
    out = {s, low}
    if low.startswith("hotel-reserv-"):
        out.add(low[len("hotel-reserv-"):])
    if low.endswith("-mongo"):
        base = low[:-len("-mongo")].split("-")[-1]
        out.update({base, "mongodb-" + base})
    if low.startswith("mongodb-"):
        base = low.replace("mongodb-", "")
        out.update({base, "hotel-reserv-" + base + "-mongo"})
    return {x for x in out if x}


def _same_service(a: str | None, b: str | None) -> bool:
    return bool(_service_aliases(a) & _service_aliases(b))


def _edge_touches_service(edge: Any, service: str) -> bool:
    text = str(edge or "")
    if "->" not in text:
        return _same_service(text, service)
    parts = [p.strip() for p in text.split("->") if p.strip()]
    return any(_same_service(part, service) for part in parts)


def _select_matched_fault(action: str | None, target: str | None, faults: list[FaultLabel]) -> tuple[FaultLabel | None, str | None]:
    if not faults:
        return None, "no_rca_faults_available"
    if target:
        match = next((f for f in faults if _same_service(f.service, str(target))), None)
        return match, None if match else "action_target_does_not_match_rca_fault"

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
    fault_type = normalize_fault_type(fault_type or "unknown")
    if fault_type == "infra_failure":
        return action in {"fix_infra_scheduling", "recreate_pod", "restart_service"}
    if fault_type in {"config_error", "auth_failure", "dependency_failure"}:
        return action in {"rollback_config", "restart_service"}
    if fault_type in {"latency_degradation", "resource_exhaustion"}:
        return action in {"scale_service", "restart_service"}
    if fault_type == "network_failure":
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
            after[field] = [
                value for value in values
                if not any(_same_service(str(value), str(scope_service)) for scope_service in affected_scope)
            ]

    failed_edges = after.get("failed_edges", []) or []
    if isinstance(failed_edges, list):
        after["failed_edges"] = [
            edge for edge in failed_edges
            if not any(_edge_touches_service(edge, str(scope_service)) for scope_service in affected_scope)
        ]

    return after


def _target_sla(signature: dict[str, Any], service: str) -> dict[str, Any]:
    target_sig: dict[str, Any] = {}
    for field in [
        "degraded_services",
        "top_error_services",
        "trace_sources",
        "trace_targets",
        "metric_anomaly_services",
        "affected_services",
    ]:
        values = signature.get(field, []) or []
        target_sig[field] = [
            value for value in values if _same_service(str(value), service)
        ] if isinstance(values, list) else []
    failed_edges = signature.get("failed_edges", []) or []
    target_sig["failed_edges"] = [
        edge for edge in failed_edges if _edge_touches_service(edge, service)
    ] if isinstance(failed_edges, list) else []
    return sla_verdict_from_signature(target_sig)


def _action_reason(repaired: bool, resolved: bool, target_sla_restored: bool, sla_restored: bool) -> str:
    if not repaired:
        return "action_does_not_repair_target_fault_type"
    if resolved:
        return "target_fault_repaired_and_sla_improved"
    if target_sla_restored and not sla_restored:
        return "target_fault_repaired_but_global_cascade_remains"
    return "action_valid_but_sla_not_restored"
