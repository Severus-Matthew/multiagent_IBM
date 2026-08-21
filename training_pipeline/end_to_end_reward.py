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
    """Map an unbounded local reward to approximately [-1, 1]."""
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
    reward_mode: str = "factorized_joint_pipeline_v1",
    rca_downstream_credit_weight: float = 0.15,
    action_system_credit_weight: float = 0.25,
) -> dict[str, Any]:
    """Score one complete RCA -> twin -> Action -> recovery trajectory.

    The pipeline executes jointly, but credit is factorized by trainable policy:

      * RCA policy return is dominated by RCA-local correctness/evidence quality
        and counterfactual-twin reproduction, with only a small downstream recovery
        term.
      * Action policy return is dominated by action-local/safety/recovery quality,
        with a moderate system-level recovery term.
      * A separate system reward is logged for end-to-end evaluation and model
        selection; it is not blindly reused as the optimizer return for both agents.

    This avoids the high-variance credit assignment of a single terminal reward
    while preserving downstream coupling between the two agents.
    """
    rca_attempt = _final_rca_attempt(rca_result)
    action_attempt = _final_action_attempt(action_result)

    rca_components = rca_attempt.get("reward_components", {}) or {}
    action_components = action_attempt.get("reward_components", {}) or {}
    verifier = action_attempt.get("verifier_result", {}) or {}
    gate = action_result.get("public_rca_twin_gate") or action_result.get("rca_twin_gate") or {}

    rca_local_raw = float(rca_attempt.get("reward", 0.0) or 0.0)
    action_local_raw = float(action_attempt.get("reward", 0.0) or 0.0)
    rca_local_bounded = _bounded_local_reward(rca_local_raw, 4.0)
    action_local_bounded = _bounded_local_reward(action_local_raw, 6.0)

    pair_score = _clamp01(rca_components.get("pair_score", 0.0))
    exact_set_match = bool(rca_components.get("exact_set_match", False))
    twin_score = _clamp01(gate.get("reproduction_score", rca_components.get("twin_reproduction_score", 0.0)))

    safe = bool(action_components.get("safe", False))
    action_repairs = bool(action_components.get("action_repairs_fault_type", False))
    target_reduction = _clamp01(
        verifier.get("target_symptom_reduction", action_components.get("target_symptom_reduction", 0.0))
    )
    global_reduction = _clamp01(
        verifier.get("global_symptom_reduction", action_components.get("global_symptom_reduction", 0.0))
    )
    target_sla_restored = bool(
        verifier.get("target_sla_restored", action_components.get("target_sla_restored", False))
    )
    sla_restored = bool(verifier.get("sla_restored", action_components.get("sla_restored", False)))
    resolved = bool(verifier.get("resolved", action_components.get("resolved", False)))
    skipped_action = bool(action_result.get("skipped_action", False))
    has_rca_prediction = bool(str(rca_result.get("final_prediction") or "").strip())

    # Observable end-to-end recovery quality in [0, 1]. This is shared context,
    # not the sole training return.
    system_quality = (
        0.10 * float(safe)
        + 0.20 * target_reduction
        + 0.20 * global_reduction
        + 0.15 * float(target_sla_restored)
        + 0.15 * float(sla_restored)
        + 0.20 * float(resolved)
    )
    system_quality = _clamp01(system_quality)

    # Private/evaluator-only RCA intrinsic score. Exact-label information is used
    # only for reward computation and never passed into RCA history or Action input.
    rca_intrinsic = (
        0.40 * pair_score
        + 0.20 * float(exact_set_match)
        + 0.40 * twin_score
    )
    rca_intrinsic = _clamp01(rca_intrinsic)

    action_intrinsic = (
        0.10 * float(safe)
        + 0.15 * float(action_repairs)
        + 0.25 * target_reduction
        + 0.15 * global_reduction
        + 0.15 * float(target_sla_restored)
        + 0.10 * float(sla_restored)
        + 0.10 * float(resolved)
    )
    action_intrinsic = _clamp01(action_intrinsic)

    rca_downstream_weight = _clamp01(rca_downstream_credit_weight)
    action_system_weight = _clamp01(action_system_credit_weight)

    rca_local_mix = 0.70 * rca_intrinsic + 0.30 * rca_local_bounded
    action_local_mix = 0.70 * action_intrinsic + 0.30 * action_local_bounded

    rca_policy_return = (
        (1.0 - rca_downstream_weight) * rca_local_mix
        + rca_downstream_weight * system_quality
    )
    action_policy_return = (
        (1.0 - action_system_weight) * action_local_mix
        + action_system_weight * system_quality
    )

    # System reward is retained for end-to-end evaluation. Recovery dominates and
    # unsafe/skipped/empty trajectories are explicitly penalized.
    unsafe_penalty = 0.0 if safe or skipped_action else 1.50
    skipped_penalty = 0.75 if skipped_action else 0.0
    empty_rca_penalty = 0.75 if not has_rca_prediction else 0.0
    system_reward = (
        4.0 * system_quality
        + 0.50 * twin_score
        + 0.25 * rca_intrinsic
        - unsafe_penalty
        - skipped_penalty
        - empty_rca_penalty
    )

    success = bool(safe and resolved and (target_sla_restored or sla_restored))
    return {
        # `reward` is kept as a compatibility alias for system-level evaluation.
        "reward": round(float(system_reward), 6),
        "system_reward": round(float(system_reward), 6),
        "system_quality": round(float(system_quality), 6),
        "rca_policy_return": round(float(rca_policy_return), 6),
        "action_policy_return": round(float(action_policy_return), 6),
        "success": success,
        "reward_mode": reward_mode,
        "credit_assignment_mode": "joint_rollout_factorized_policy_returns_v1",
        "components": {
            "rca_local_reward_raw": round(rca_local_raw, 6),
            "rca_local_reward_bounded": round(rca_local_bounded, 6),
            "rca_intrinsic": round(rca_intrinsic, 6),
            "pair_score": round(pair_score, 6),
            "private_rca_exact_set_match": exact_set_match,
            "counterfactual_twin_reproduction_score": round(twin_score, 6),
            "rca_downstream_credit_weight": round(rca_downstream_weight, 6),
            "action_local_reward_raw": round(action_local_raw, 6),
            "action_local_reward_bounded": round(action_local_bounded, 6),
            "action_intrinsic": round(action_intrinsic, 6),
            "action_system_credit_weight": round(action_system_weight, 6),
            "safe": safe,
            "action_repairs_fault_type": action_repairs,
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
            "The incident executes end-to-end, but RCA and Action receive separate returns. "
            "Private RCA exact-match is evaluator-only and never exposed to downstream agents."
        ),
    }
