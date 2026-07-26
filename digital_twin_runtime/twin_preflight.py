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
    that the state has enough observable structure for the behavioral twin and
    that the verifier can produce a prediction-sensitive RCA validation object.
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
        mode="behavioral_offline_proxy",
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
) -> dict[str, Any]:
    """Turn a per-attempt RCA twin result into an explicit gate object."""
    if not twin_result:
        return {
            "rca_twin_verified": False,
            "reason": "missing_twin_result",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
        }
    score = float(twin_result.get("reproduction_score", 0.0) or 0.0)
    mode = twin_result.get("mode", "unknown")
    ok = score >= float(min_reproduction_score)
    return {
        "rca_twin_verified": ok,
        "reason": "score_above_threshold" if ok else "score_below_threshold",
        "mode": mode,
        "min_reproduction_score": float(min_reproduction_score),
        "reproduction_score": round(score, 6),
        "uses_oracle_labels": bool(twin_result.get("uses_oracle_labels", False)),
        "uses_full_state_for_rca_score": bool(twin_result.get("uses_full_state_for_rca_score", False)),
    }


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
