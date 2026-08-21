from __future__ import annotations

import json
import math
from typing import Any

from .end_to_end_reward import end_to_end_reward
from .grpo_math import (
    clipped_grpo_surrogate,
    group_relative_advantages,
    schulman_reverse_kl_estimate,
)
from .rca_reward import optimal_match
from .schemas import FaultLabel


def _assert_close(a: float, b: float, tol: float = 1e-8, message: str = "") -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(message or f"expected {a} ~= {b} within {tol}")


def _rca_result(
    *,
    pair: float,
    exact: bool,
    twin: float,
    raw_reward: float = 0.0,
    invalid: bool = False,
    count_mismatch: int = 0,
    num_gt: int = 1,
    repeated: bool = False,
    iteration: int = 0,
) -> dict[str, Any]:
    return {
        "final_prediction": "svc::infra_failure",
        "attempts": [{
            "reward": raw_reward,
            "reward_components": {
                "pair_score": pair,
                "exact_set_match": exact,
                "twin_reproduction_score": twin,
                "invalid_format": invalid,
                "count_mismatch": count_mismatch,
                "num_gt": num_gt,
                "repeated_wrong_guess": repeated,
                "iteration_index": iteration,
                "instruction_tokens": 80,
            },
        }],
    }


def _action_result(
    *,
    safe: bool,
    repairs: bool,
    target_reduction: float,
    global_reduction: float,
    target_sla: bool,
    sla: bool,
    resolved: bool,
    twin_score: float = 0.0,
    raw_reward: float = 0.0,
    has_verify: bool = True,
    has_mutation: bool = True,
    num_commands: int = 2,
    iteration: int = 0,
) -> dict[str, Any]:
    return {
        "skipped_action": False,
        "public_rca_twin_gate": {"reproduction_score": twin_score},
        "attempts": [{
            "reward": raw_reward,
            "reward_components": {
                "safe": safe,
                "action_repairs_fault_type": repairs,
                "target_symptom_reduction": target_reduction,
                "global_symptom_reduction": global_reduction,
                "target_sla_restored": target_sla,
                "sla_restored": sla,
                "resolved": resolved,
                "has_verification_command": has_verify,
                "has_mutating_command": has_mutation,
                "num_commands": num_commands,
                "iteration_index": iteration,
            },
            "verifier_result": {
                "target_symptom_reduction": target_reduction,
                "global_symptom_reduction": global_reduction,
                "target_sla_restored": target_sla,
                "sla_restored": sla,
                "resolved": resolved,
            },
        }],
    }


def run_audit() -> dict[str, Any]:
    checks: list[str] = []

    # 1) Standard group-relative normalization invariants. Current common TRL
    # implementations use torch.std with correction=1 plus ~1e-4 in denominator.
    vals = [1.0, 2.0, 4.0, 8.0]
    g = group_relative_advantages(vals)
    assert g.std_correction == 1
    _assert_close(g.normalization_epsilon, 1e-4, 1e-15)
    _assert_close(sum(g.advantages) / len(g.advantages), 0.0, 1e-12)

    mu = sum(vals) / len(vals)
    expected_std = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
    _assert_close(g.std, expected_std, 1e-12)
    for value, advantage in zip(vals, g.advantages):
        _assert_close(advantage, (value - mu) / (expected_std + 1e-4), 1e-12)
    checks.append("group_advantage_matches_sample_std_plus_epsilon")

    # With epsilon disabled, positive affine transforms preserve standardized
    # advantages exactly. This isolates the normalization math from the deliberate
    # numerical epsilon used by the production path.
    g_noeps = group_relative_advantages(vals, normalization_epsilon=0.0)
    g2_noeps = group_relative_advantages([7.0 * x + 13.0 for x in vals], normalization_epsilon=0.0)
    for a, b in zip(g_noeps.advantages, g2_noeps.advantages):
        _assert_close(a, b, 1e-10)
    checks.append("group_advantage_affine_invariance_without_numerical_epsilon")

    # Constant groups carry no relative signal.
    gz = group_relative_advantages([3.0, 3.0, 3.0, 3.0])
    assert gz.zero_variance
    assert all(a == 0.0 for a in gz.advantages)
    checks.append("zero_variance_group_yields_zero_advantage")

    # Singleton groups are not valid relative-comparison groups and also carry no
    # relative signal.
    gs = group_relative_advantages([3.0])
    assert gs.zero_variance and gs.advantages == (0.0,)
    checks.append("singleton_group_yields_zero_advantage")

    # 2) PPO/GRPO clip sign behavior.
    pos = clipped_grpo_surrogate(math.log(1.5), 0.0, 1.0, epsilon_low=0.2, epsilon_high=0.2)
    _assert_close(float(pos["clipped_ratio"]), 1.2, 1e-10)
    _assert_close(float(pos["loss"]), -1.2, 1e-10)

    neg = clipped_grpo_surrogate(math.log(0.5), 0.0, -1.0, epsilon_low=0.2, epsilon_high=0.2)
    _assert_close(float(neg["clipped_ratio"]), 0.8, 1e-10)
    _assert_close(float(neg["loss"]), 0.8, 1e-10)
    checks.append("clipped_surrogate_positive_and_negative_advantage_signs")

    # 3) Sampled KL estimator must be non-negative and zero at equal policies.
    _assert_close(schulman_reverse_kl_estimate(-2.0, -2.0), 0.0, 1e-12)
    assert schulman_reverse_kl_estimate(-1.0, -2.0) >= 0.0
    assert schulman_reverse_kl_estimate(-2.0, -1.0) >= 0.0
    checks.append("schulman_kl_estimator_nonnegative")

    good_rca = _rca_result(pair=1.0, exact=True, twin=0.9, raw_reward=100.0)
    bad_rca = _rca_result(pair=0.1, exact=False, twin=0.1, raw_reward=-100.0)
    bad_action = _action_result(
        safe=True, repairs=False, target_reduction=0.0, global_reduction=0.0,
        target_sla=False, sla=False, resolved=False, twin_score=0.9, raw_reward=-100.0,
    )
    good_action = _action_result(
        safe=True, repairs=True, target_reduction=1.0, global_reduction=1.0,
        target_sla=True, sla=True, resolved=True, twin_score=0.1, raw_reward=100.0,
    )

    a = end_to_end_reward(good_rca, bad_action)
    b = end_to_end_reward(bad_rca, good_action)
    assert a["rca_policy_return"] > a["action_policy_return"]
    assert b["action_policy_return"] > b["rca_policy_return"]
    checks.append("factorized_credit_separates_rca_and_action_quality")

    # Raw local scalar rewards are diagnostic only. Altering them with all causal
    # reward components held fixed must not change policy returns.
    good_rca_low_raw = _rca_result(pair=1.0, exact=True, twin=0.9, raw_reward=-999.0)
    bad_action_high_raw = _action_result(
        safe=True, repairs=False, target_reduction=0.0, global_reduction=0.0,
        target_sla=False, sla=False, resolved=False, twin_score=0.9, raw_reward=999.0,
    )
    a2 = end_to_end_reward(good_rca_low_raw, bad_action_high_raw)
    _assert_close(a["rca_policy_return"], a2["rca_policy_return"], 1e-12)
    _assert_close(a["action_policy_return"], a2["action_policy_return"], 1e-12)
    checks.append("no_duplicate_raw_local_reward_shaping")

    # Better RCA evidence must monotonically improve RCA return when downstream
    # recovery is held fixed.
    rca_low = end_to_end_reward(_rca_result(pair=0.2, exact=False, twin=0.2), bad_action)
    rca_high = end_to_end_reward(_rca_result(pair=0.8, exact=False, twin=0.8), bad_action)
    assert rca_high["rca_policy_return"] > rca_low["rca_policy_return"]
    checks.append("rca_return_monotone_in_match_and_twin_quality")

    # Better recovery must improve Action return and system quality.
    action_low = end_to_end_reward(good_rca, bad_action)
    action_high = end_to_end_reward(good_rca, good_action)
    assert action_high["action_policy_return"] > action_low["action_policy_return"]
    assert action_high["system_quality"] > action_low["system_quality"]
    checks.append("action_and_system_returns_monotone_in_recovery")

    # System reward must not depend on private RCA exact correctness when the
    # observable recovery trajectory is unchanged.
    same_action_1 = end_to_end_reward(good_rca, good_action)
    same_action_2 = end_to_end_reward(bad_rca, good_action)
    _assert_close(same_action_1["system_reward"], same_action_2["system_reward"], 1e-12)
    checks.append("system_reward_independent_of_private_rca_exact_match")

    # 4) Multifault matching must be order-invariant and exact for reversed lists.
    full_state = {
        "graph": {
            "edges": [
                {"src": "svc-a", "dst": "svc-b"},
                ["svc-b", "svc-c"],
            ]
        }
    }
    gt = [
        FaultLabel(service="svc-a", fault_type="infra_failure"),
        FaultLabel(service="svc-b", fault_type="auth_failure"),
    ]
    pred = [
        FaultLabel(service="svc-b", fault_type="auth_failure"),
        FaultLabel(service="svc-a", fault_type="infra_failure"),
    ]
    _, score1 = optimal_match(gt, pred, full_state)
    _, score2 = optimal_match(list(reversed(gt)), pred, full_state)
    _assert_close(score1, 1.0, 1e-12)
    _assert_close(score2, 1.0, 1e-12)
    checks.append("multifault_matching_exact_and_order_invariant")

    return {
        "ok": True,
        "num_checks": len(checks),
        "checks": checks,
        "reward_mode": a["reward_mode"],
        "credit_assignment_mode": a["credit_assignment_mode"],
        "grpo_std_correction": g.std_correction,
        "grpo_normalization_epsilon": g.normalization_epsilon,
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
