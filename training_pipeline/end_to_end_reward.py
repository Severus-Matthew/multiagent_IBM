from __future__ import annotations

import math
from typing import Any


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _bounded_local_reward(value: Any, scale: float) -> float:
    """Map an unbounded/debug local reward to approximately [-1, 1]."""
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return math.tanh(x / max(scale, 1e-6))


def _final_rca_attempt(rca_result: dict[str, Any]) -> dict[str, Any]:
    attempts = rca_result.get("attempts", []) or []
    return attempts[-1] if attempts else {}


def _final_action_attempt(action_result: dict[str, Any]) -> dict[str, Any]:
    attempts = action_result.get("attempts", []) or []
    return attempts[-1] if attempts else {}


def end_to_end_reward(
    rca_result: dict[str, Any],
    action_result: dict[str, Any],
    *,
    reward_mode: str = "offline_diagnostic_joint_v1",
) -> dict[str, Any]:
    """Score one complete RCA -> twin -> action -> recovery trajectory.

    The scalar return is shared by every trainable policy decision in the same
    trajectory. Local RCA/action rewards remain available only as shaping and
    diagnostics. Final success is recovery-centric: a trajectory succeeds when a
    safe action resolves the verifier environment and restores the target/global
    SLA. It does not require the private RCA exact-label flag.

    `offline_diagnostic_joint_v1` is for plumbing/smoke tests only because the
    current action verifier is still an offline behavioral simulator. A live-twin
    implementation can use the same contract without changing the joint trainer.
    """
    rca_attempt = _final_rca_attempt(rca_result)
    action_attempt = _final_action_attempt(action_result)

    rca_components = rca_attempt.get("reward_components", {}) or {}
    action_components = action_attempt.get("reward_components", {}) or {}
    verifier = action_attempt.get("verifier_result", {}) or {}
    gate = action_result.get("public_rca_twin_gate") or action_result.get("rca_twin_gate") or {}

    rca_local_raw = float(rca_attempt.get("reward", 0.0) or 0.0)
    action_local_raw = float(action_attempt.get("reward", 0.0) or 0.0)
    rca_local = _bounded_local_reward(rca_local_raw, 4.0)
    action_local = _bounded_local_reward(action_local_raw, 6.0)

    twin_score = _clamp01(gate.get("reproduction_score", rca_components.get("twin_reproduction_score", 0.0)))
    safe = bool(action_components.get("safe", False))
    target_reduction = _clamp01(verifier.get("target_symptom_reduction", action_components.get("target_symptom_reduction", 0.0)))
    global_reduction = _clamp01(verifier.get("global_symptom_reduction", action_components.get("global_symptom_reduction", 0.0)))
    target_sla_restored = bool(verifier.get("target_sla_restored", action_components.get("target_sla_restored", False)))
    sla_restored = bool(verifier.get("sla_restored", action_components.get("sla_restored", False)))
    resolved = bool(verifier.get("resolved", action_components.get("resolved", False)))
    skipped_action = bool(action_result.get("skipped_action", False))
    has_rca_prediction = bool(str(rca_result.get("final_prediction") or "").strip())

    unsafe_penalty = 0.0 if safe or skipped_action else 1.50
    skipped_penalty = 0.75 if skipped_action else 0.0
    empty_rca_penalty = 0.75 if not has_rca_prediction else 0.0

    # Recovery dominates the return. RCA/twin and local action terms provide
    # denser credit before the live environment is consistently solved.
    reward = (
        0.30 * rca_local
        + 0.75 * twin_score
        + 0.30 * action_local
        + 0.75 * target_reduction
        + 0.75 * global_reduction
        + (1.00 if target_sla_restored else 0.0)
        + (1.50 if sla_restored else 0.0)
        + (1.25 if resolved else 0.0)
        - unsafe_penalty
        - skipped_penalty
        - empty_rca_penalty
    )

    success = bool(safe and resolved and (target_sla_restored or sla_restored))
    return {
        "reward": round(float(reward), 6),
        "success": success,
        "reward_mode": reward_mode,
        "components": {
            "rca_local_reward_raw": round(rca_local_raw, 6),
            "rca_local_reward_bounded": round(rca_local, 6),
            "private_rca_exact_set_match": bool(rca_components.get("exact_set_match", False)),
            "counterfactual_twin_reproduction_score": round(twin_score, 6),
            "action_local_reward_raw": round(action_local_raw, 6),
            "action_local_reward_bounded": round(action_local, 6),
            "safe": safe,
            "target_symptom_reduction": round(target_reduction, 6),
            "global_symptom_reduction": round(global_reduction, 6),
            "target_sla_restored": target_sla_restored,
            "sla_restored": sla_restored,
            "resolved": resolved,
            "skipped_action": skipped_action,
            "has_rca_prediction": has_rca_prediction,
            "unsafe_penalty": unsafe_penalty,
            "skipped_penalty": skipped_penalty,
            "empty_rca_penalty": empty_rca_penalty,
        },
        "note": (
            "Private RCA exact-match is logged only as evaluator diagnostics and is never exposed to downstream agents. "
            "Final trajectory success is defined by safe recovery/SLA restoration."
        ),
    }
