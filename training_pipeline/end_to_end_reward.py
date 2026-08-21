from __future__ import annotations

from typing import Any


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _clamp_signed(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(-1.0, min(1.0, x))


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
    reward_mode: str = "factorized_joint_pipeline_v2_no_double_count",
    rca_downstream_credit_weight: float = 0.15,
    action_system_credit_weight: float = 0.25,
) -> dict[str, Any]:
    """Score one complete RCA -> twin -> Action -> recovery trajectory.

    Execution is joint, but RCA and Action receive separate returns. Raw local
    scalar rewards are diagnostics only; they are not reused here, which avoids
    double-counting the same RCA/twin/recovery components.
    """
    rca_attempt = _final_rca_attempt(rca_result)
    action_attempt = _final_action_attempt(action_result)

    rca_components = rca_attempt.get("reward_components", {}) or {}
    action_components = action_attempt.get("reward_components", {}) or {}
    verifier = action_attempt.get("verifier_result", {}) or {}
    gate = action_result.get("public_rca_twin_gate") or action_result.get("rca_twin_gate") or {}

    # RCA intrinsic signal. Positive weights sum to one.
    pair_score = _clamp01(rca_components.get("pair_score", 0.0))
    exact_set_match = bool(rca_components.get("exact_set_match", False))
    twin_score = _clamp01(
        gate.get("reproduction_score", rca_components.get("twin_reproduction_score", 0.0))
    )
    rca_intrinsic = _clamp01(
        0.40 * pair_score
        + 0.20 * float(exact_set_match)
        + 0.40 * twin_score
    )

    invalid_format = bool(rca_components.get("invalid_format", False))
    count_mismatch = max(0.0, float(rca_components.get("count_mismatch", 0.0) or 0.0))
    num_gt = max(1.0, float(rca_components.get("num_gt", 1.0) or 1.0))
    count_mismatch_rate = _clamp01(count_mismatch / num_gt)
    repeated_wrong_guess = bool(rca_components.get("repeated_wrong_guess", False))
    rca_iteration = max(0.0, float(rca_components.get("iteration_index", 0.0) or 0.0))
    rca_instruction_tokens = max(0.0, float(rca_components.get("instruction_tokens", 0.0) or 0.0))

    rca_penalty = (
        0.20 * float(invalid_format)
        + 0.10 * count_mismatch_rate
        + 0.05 * float(repeated_wrong_guess)
        + 0.03 * _clamp01(rca_iteration / 4.0)
        + 0.02 * _clamp01(max(0.0, rca_instruction_tokens - 120.0) / 240.0)
    )
    rca_local_score = _clamp_signed(rca_intrinsic - rca_penalty)

    # Action/recovery observables.
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
    has_verify = bool(action_components.get("has_verification_command", False))
    has_mutation = bool(action_components.get("has_mutating_command", False))
    num_commands = max(0.0, float(action_components.get("num_commands", 0.0) or 0.0))
    action_iteration = max(0.0, float(action_components.get("iteration_index", 0.0) or 0.0))

    # Recovery quality contains no free reward for merely being safe. Safety and
    # an actual mutation are gates. Thus a safe no-op cannot receive positive
    # system credit simply because it avoided damage.
    recovery_quality = _clamp01(
        0.25 * target_reduction
        + 0.25 * global_reduction
        + 0.15 * float(target_sla_restored)
        + 0.15 * float(sla_restored)
        + 0.20 * float(resolved)
    )
    system_quality = recovery_quality if (safe and has_mutation) else 0.0

    # Action intrinsic quality keeps safety and repair compatibility as dense
    # shaping. Positive weights sum to one before penalties.
    action_intrinsic = _clamp01(
        0.10 * float(safe)
        + 0.15 * float(action_repairs)
        + 0.25 * target_reduction
        + 0.15 * global_reduction
        + 0.15 * float(target_sla_restored)
        + 0.10 * float(sla_restored)
        + 0.10 * float(resolved)
    )

    action_penalty = (
        0.35 * float(not safe)
        + 0.05 * float(not has_verify)
        + 0.08 * float(not has_mutation)
        + 0.02 * _clamp01(max(0.0, num_commands - 3.0) / 10.0)
        + 0.03 * _clamp01(action_iteration / 4.0)
    )
    action_local_score = _clamp_signed(action_intrinsic - action_penalty)

    # Factorized policy returns: bounded convex mixtures with explicit cross-stage
    # coupling. These are the values normalized into policy advantages.
    rca_downstream_weight = _clamp01(rca_downstream_credit_weight)
    action_system_weight = _clamp01(action_system_credit_weight)

    rca_policy_return = (
        (1.0 - rca_downstream_weight) * rca_local_score
        + rca_downstream_weight * system_quality
    )
    action_policy_return = (
        (1.0 - action_system_weight) * action_local_score
        + action_system_weight * system_quality
    )

    # System reward is diagnostic/model-selection only and does not contain the
    # private RCA exact-match signal.
    unsafe_system_penalty = 0.50 if (not safe and not skipped_action) else 0.0
    skipped_system_penalty = 0.25 if skipped_action else 0.0
    empty_rca_system_penalty = 0.25 if not has_rca_prediction else 0.0
    system_reward = (
        system_quality
        - unsafe_system_penalty
        - skipped_system_penalty
        - empty_rca_system_penalty
    )

    success = bool(
        safe
        and has_mutation
        and resolved
        and (target_sla_restored or sla_restored)
    )

    rca_local_raw = float(rca_attempt.get("reward", 0.0) or 0.0)
    action_local_raw = float(action_attempt.get("reward", 0.0) or 0.0)

    return {
        "reward": round(float(system_reward), 6),
        "system_reward": round(float(system_reward), 6),
        "system_quality": round(float(system_quality), 6),
        "rca_policy_return": round(float(rca_policy_return), 6),
        "action_policy_return": round(float(action_policy_return), 6),
        "success": success,
        "reward_mode": reward_mode,
        "credit_assignment_mode": "joint_rollout_factorized_policy_returns_v2",
        "components": {
            "rca_local_reward_raw_diagnostic_only": round(rca_local_raw, 6),
            "rca_intrinsic": round(rca_intrinsic, 6),
            "rca_penalty": round(rca_penalty, 6),
            "rca_local_score": round(rca_local_score, 6),
            "pair_score": round(pair_score, 6),
            "private_rca_exact_set_match": exact_set_match,
            "counterfactual_twin_reproduction_score": round(twin_score, 6),
            "count_mismatch_rate": round(count_mismatch_rate, 6),
            "rca_downstream_credit_weight": round(rca_downstream_weight, 6),
            "action_local_reward_raw_diagnostic_only": round(action_local_raw, 6),
            "action_intrinsic": round(action_intrinsic, 6),
            "action_penalty": round(action_penalty, 6),
            "action_local_score": round(action_local_score, 6),
            "action_system_credit_weight": round(action_system_weight, 6),
            "safe": safe,
            "action_repairs_fault_type": action_repairs,
            "target_symptom_reduction": round(target_reduction, 6),
            "global_symptom_reduction": round(global_reduction, 6),
            "target_sla_restored": target_sla_restored,
            "sla_restored": sla_restored,
            "resolved": resolved,
            "recovery_quality": round(recovery_quality, 6),
            "system_quality_requires_safe_mutation": True,
            "skipped_action": skipped_action,
            "has_rca_prediction": has_rca_prediction,
            "has_verification_command": has_verify,
            "has_mutating_command": has_mutation,
            "unsafe_system_penalty": unsafe_system_penalty,
            "skipped_system_penalty": skipped_system_penalty,
            "empty_rca_system_penalty": empty_rca_system_penalty,
        },
        "note": (
            "Joint execution with separate, non-duplicated RCA/Action returns. "
            "System credit requires observable recovery from a safe mutating action."
        ),
    }
