from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from training_pipeline.schemas import FaultLabel
from .twin_spec_builder import build_oracle_twin_spec, build_predicted_twin_spec
from .twin_verifier import BehavioralTwinVerifier


@dataclass
class TwinPreflightResult:
    """Preflight diagnostics for the RCA-verification twin path.

    In behavioral mode this does not create a live Kubernetes namespace. It checks
    that the state has enough observable structure for the counterfactual replay
    twin and that the verifier can produce a prediction-sensitive RCA validation
    object without using oracle labels for the score.
    """

    ok: bool
    mode: str
    scenario_id: str
    namespace: str | None
    state_hash: str
    num_services: int
    num_graph_edges: int
    predicted_spec: dict[str, Any]
    oracle_spec_summary: dict[str, Any] | None
    verifier_probe: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight_behavioral_twin(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    include_oracle_summary: bool = True,
) -> dict[str, Any]:
    """Run an offline twin preflight before RCA/action rollouts.

    The agent never sees this object. It is intended for run logs, W&B artifacts,
    and guardrails before large experiments.
    """
    warnings: list[str] = []
    errors: list[str] = []

    scenario_id = str(compressed_state.get("scenario_id") or full_state.get("scenario_id") or "unknown")
    namespace = compressed_state.get("namespace") or (full_state.get("fault_context", {}) or {}).get("target_namespace")
    services = list(compressed_state.get("services", []) or [])
    graph_edges = list(((compressed_state.get("graph", {}) or {}).get("edges", []) or []))

    if not services:
        errors.append("compressed_state_has_no_services")
    if not graph_edges:
        warnings.append("compressed_state_has_no_graph_edges")

    # Use an empty prediction to verify spec construction without injecting any
    # hidden label into the agent-facing predicted-twin path.
    predicted_spec = build_predicted_twin_spec(compressed_state, [])

    oracle_summary = None
    if include_oracle_summary:
        oracle_labels = _labels_from_full_state_safe(full_state)
        oracle_spec = build_oracle_twin_spec(full_state, oracle_labels)
        oracle_summary = {
            "mode": oracle_spec.mode,
            "num_services_to_keep": len(oracle_spec.services_to_keep),
            "num_services_to_prune": len(oracle_spec.services_to_prune),
            "num_target_faults": len(oracle_spec.target_faults),
            "note": "offline_evaluator_only_do_not_show_to_agent",
        }

    verifier_probe: dict[str, Any]
    try:
        verifier = BehavioralTwinVerifier()
        probe_fault = _probe_fault_from_redacted_state(compressed_state)
        verifier_probe = verifier.validate_rca_prediction(full_state, compressed_state, [probe_fault])
        if "reproduction_score" not in verifier_probe:
            errors.append("verifier_probe_missing_reproduction_score")
        if verifier_probe.get("uses_oracle_labels"):
            errors.append("behavioral_verifier_reports_oracle_label_use")
        if verifier_probe.get("uses_full_state_for_rca_score"):
            errors.append("behavioral_verifier_reports_full_state_rca_scoring")
    except Exception as exc:
        verifier_probe = {"error": repr(exc)}
        errors.append("behavioral_verifier_probe_failed")

    result = TwinPreflightResult(
        ok=not errors,
        mode="counterfactual_offline_twin",
        scenario_id=scenario_id,
        namespace=str(namespace) if namespace else None,
        state_hash=_stable_hash(compressed_state),
        num_services=len(services),
        num_graph_edges=len(graph_edges),
        predicted_spec={
            "mode": predicted_spec.mode,
            "num_services_to_keep": len(predicted_spec.services_to_keep),
            "num_services_to_prune": len(predicted_spec.services_to_prune),
            "num_target_faults": len(predicted_spec.target_faults),
        },
        oracle_spec_summary=oracle_summary,
        verifier_probe=verifier_probe,
        warnings=warnings,
        errors=errors,
    )
    return result.to_dict()


def require_twin_preflight_ok(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        raise RuntimeError("twin preflight failed: " + json.dumps(result.get("errors", []), sort_keys=True))


def rca_twin_gate(
    twin_result: dict[str, Any] | None,
    min_reproduction_score: float = 0.0,
    rca_success: bool | None = None,
) -> dict[str, Any]:
    """Turn a per-attempt RCA twin result into an explicit gate object.

    A prediction can be verified by either the current offline counterfactual
    replay path or a future live reinjection path. The gate requires:
      1. RCA success when `rca_success` is explicitly provided;
      2. a prediction-derived signature and an observed/evidence signature;
      3. reproduction score >= `min_reproduction_score`;
      4. no oracle/full-state information used to compute that score.

    `predicted_fault_injection_checked` is reserved for true live reinjection.
    Offline counterfactual replay is exposed separately as
    `counterfactual_replay_checked` so the two modes cannot be confused.
    """
    if not twin_result:
        return {
            "rca_twin_verified": False,
            "reason": "missing_twin_result",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
            "same_error_pattern_score": 0.0,
            "predicted_fault_injection_checked": False,
            "counterfactual_replay_checked": False,
            "same_error_pattern_verified": False,
            "rca_success_required": rca_success is not None,
            "rca_success": bool(rca_success) if rca_success is not None else None,
        }

    score = float(twin_result.get("reproduction_score", 0.0) or 0.0)
    mode = twin_result.get("mode", "unknown")
    uses_oracle = bool(twin_result.get("uses_oracle_labels", False) or twin_result.get("uses_oracle_labels_for_score", False))
    uses_full_state = bool(twin_result.get("uses_full_state_for_rca_score", False) or twin_result.get("uses_full_state_for_score", False))

    predicted_signature = twin_result.get("predicted_signature")
    observed_signature = twin_result.get("observed_signature") or twin_result.get("evidence_signature")
    replay_checked = bool(predicted_signature) and bool(observed_signature)
    live_injection_checked = bool(twin_result.get("predicted_fault_injection_checked", False))
    same_error_pattern_verified = replay_checked and score >= float(min_reproduction_score)
    rca_ok = True if rca_success is None else bool(rca_success)
    ok = rca_ok and same_error_pattern_verified and not uses_oracle and not uses_full_state

    if not rca_ok:
        reason = "rca_attempt_not_successful"
    elif uses_oracle:
        reason = "twin_score_used_oracle_labels"
    elif uses_full_state:
        reason = "twin_score_used_full_state_rca_scoring"
    elif not replay_checked:
        reason = "missing_predicted_or_observed_signature"
    elif not same_error_pattern_verified:
        reason = "score_below_threshold"
    else:
        reason = "rca_success_and_counterfactual_pattern_reproduced"

    return {
        "rca_twin_verified": ok,
        "reason": reason,
        "mode": mode,
        "min_reproduction_score": float(min_reproduction_score),
        "reproduction_score": round(score, 6),
        "same_error_pattern_score": round(score, 6),
        "predicted_fault_injection_checked": live_injection_checked,
        "counterfactual_replay_checked": replay_checked,
        "same_error_pattern_verified": same_error_pattern_verified,
        "rca_success_required": rca_success is not None,
        "rca_success": bool(rca_success) if rca_success is not None else None,
        "uses_oracle_labels": uses_oracle,
        "uses_full_state_for_rca_score": uses_full_state,
        "predicted_signature": predicted_signature,
        "observed_signature_summary": _signature_summary(observed_signature or {}),
        # Backward-compatible key used by some older logs/readers.
        "evidence_signature_summary": _signature_summary(observed_signature or {}),
        "per_fault": twin_result.get("per_fault", []),
    }


def _signature_summary(sig: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sig, dict):
        return {}
    keys = [
        "degraded_services",
        "failed_edges",
        "top_error_services",
        "trace_sources",
        "trace_targets",
        "metric_anomaly_services",
        "affected_services",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = sig.get(key, []) or []
        out[f"num_{key}"] = len(vals) if isinstance(vals, list) else 0
        if isinstance(vals, list):
            out[f"sample_{key}"] = vals[:10]
    return out


def _probe_fault_from_redacted_state(compressed_state: dict[str, Any]) -> FaultLabel:
    services = compressed_state.get("services", []) or []
    service = "unknown"
    system = compressed_state.get("system", {}) or {}
    for svc, info in system.items():
        health = info.get("health", {}) if isinstance(info, dict) else {}
        if health.get("infra_issue_flag") or float(health.get("pods_unready", 0) or 0) > 0:
            service = str(svc)
            break
    if service == "unknown" and services:
        service = str(services[0])
    return FaultLabel(service=service, fault_type="unknown")


def _labels_from_full_state_safe(full_state: dict[str, Any]) -> list[FaultLabel]:
    # Local import avoids a dependency cycle. This is evaluator-only preflight
    # metadata and must never be placed in policy prompts.
    from training_pipeline.ground_truth import labels_from_full_state

    return labels_from_full_state(full_state)


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
