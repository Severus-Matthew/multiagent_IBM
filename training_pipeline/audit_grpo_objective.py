from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .end_to_end_reward import end_to_end_reward
from .grpo_dataset import load_grpo_dataset
from .grpo_math import (
    clipped_grpo_surrogate,
    group_relative_advantages,
    schulman_reverse_kl_estimate,
)


def _assert_close(actual: float, expected: float, tol: float = 1e-8, message: str = "") -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(message or f"expected {expected}, got {actual}")


def _rca_result(pair: float, exact: bool, twin: float, *, invalid: bool = False) -> dict[str, Any]:
    return {
        "final_prediction": "svc::infra_failure" if not invalid else "",
        "attempts": [{
            "reward": 999.0,  # deliberately absurd: factorized return must not reuse it
            "reward_components": {
                "pair_score": pair,
                "exact_set_match": exact,
                "twin_reproduction_score": twin,
                "invalid_format": invalid,
                "count_mismatch": 0,
                "num_gt": 1,
                "repeated_wrong_guess": False,
                "iteration_index": 0,
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
    mutation: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    return {
        "skipped_action": False,
        "public_rca_twin_gate": {"reproduction_score": 0.5},
        "attempts": [{
            "reward": -999.0,  # deliberately absurd: must remain diagnostic only
            "reward_components": {
                "safe": safe,
                "action_repairs_fault_type": repairs,
                "target_symptom_reduction": target_reduction,
                "global_symptom_reduction": global_reduction,
                "target_sla_restored": target_sla,
                "sla_restored": sla,
                "resolved": resolved,
                "has_verification_command": verify,
                "has_mutating_command": mutation,
                "num_commands": 2 if mutation else 1,
                "iteration_index": 0,
                "instruction_tokens": 80,
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


def audit_advantage_math() -> dict[str, Any]:
    result = group_relative_advantages([1.0, 2.0, 3.0, 4.0])
    _assert_close(result.mean, 2.5)
    expected_std = math.sqrt(5.0 / 3.0)
    _assert_close(result.std, expected_std)
    _assert_close(sum(result.advantages), 0.0, tol=1e-10)
    if result.std_correction != 1:
        raise AssertionError("GRPO advantage normalization must use explicit sample std correction=1")
    if result.normalization_epsilon != 1e-4:
        raise AssertionError("unexpected GRPO normalization epsilon")

    zero = group_relative_advantages([5.0, 5.0, 5.0, 5.0])
    if not zero.zero_variance or any(x != 0.0 for x in zero.advantages):
        raise AssertionError("zero-variance group must produce exactly zero advantages")

    shifted = group_relative_advantages([11.0, 12.0, 13.0, 14.0])
    for a, b in zip(result.advantages, shifted.advantages):
        _assert_close(a, b, tol=1e-12, message="advantage must be invariant to reward translation")

    return {
        "known_group_mean": result.mean,
        "known_group_sample_std": result.std,
        "known_group_advantages": list(result.advantages),
        "zero_variance_advantages": list(zero.advantages),
        "translation_invariance": True,
    }


def audit_clipped_surrogate() -> dict[str, Any]:
    # Positive A, ratio above upper clip -> clipped objective.
    pos = clipped_grpo_surrogate(math.log(1.5), 0.0, 1.0, epsilon_low=0.2, epsilon_high=0.2)
    _assert_close(float(pos["objective"]), 1.2)
    if not pos["was_clipped"]:
        raise AssertionError("positive-advantage high ratio should be clipped")

    # Negative A, ratio below lower clip -> clipped objective is more negative.
    neg = clipped_grpo_surrogate(math.log(0.5), 0.0, -1.0, epsilon_low=0.2, epsilon_high=0.2)
    _assert_close(float(neg["objective"]), -0.8)
    if not neg["was_clipped"]:
        raise AssertionError("negative-advantage low ratio should be clipped")

    same = clipped_grpo_surrogate(0.0, 0.0, 0.37)
    _assert_close(float(same["objective"]), 0.37)

    return {"positive_case": pos, "negative_case": neg, "ratio_one_case": same}


def audit_kl_math() -> dict[str, Any]:
    zero = schulman_reverse_kl_estimate(-2.0, -2.0)
    _assert_close(zero, 0.0)
    a = schulman_reverse_kl_estimate(-1.0, -2.0)
    b = schulman_reverse_kl_estimate(-2.0, -1.0)
    if a < 0.0 or b < 0.0:
        raise AssertionError("sampled reverse-KL estimator must be non-negative")
    return {"equal_logprob_kl": zero, "case_a": a, "case_b": b}


def audit_factorized_reward() -> dict[str, Any]:
    bad_action = _action_result(
        safe=True, repairs=False, target_reduction=0.0, global_reduction=0.0,
        target_sla=False, sla=False, resolved=False,
    )
    good_action = _action_result(
        safe=True, repairs=True, target_reduction=1.0, global_reduction=1.0,
        target_sla=True, sla=True, resolved=True,
    )

    good_rca = _rca_result(1.0, True, 1.0)
    bad_rca = _rca_result(0.0, False, 0.0)

    good_r_bad_a = end_to_end_reward(good_rca, bad_action)
    bad_r_good_a = end_to_end_reward(bad_rca, good_action)
    good_both = end_to_end_reward(good_rca, good_action)
    bad_both = end_to_end_reward(bad_rca, bad_action)

    if not (good_r_bad_a["rca_policy_return"] > good_r_bad_a["action_policy_return"]):
        raise AssertionError("good RCA / bad Action must favor RCA credit")
    if not (bad_r_good_a["action_policy_return"] > bad_r_good_a["rca_policy_return"]):
        raise AssertionError("bad RCA / good Action must favor Action credit")
    if not (good_both["rca_policy_return"] > bad_both["rca_policy_return"]):
        raise AssertionError("RCA return must increase with RCA correctness/twin support")
    if not (good_both["action_policy_return"] > bad_both["action_policy_return"]):
        raise AssertionError("Action return must increase with recovery quality")

    # System reward must not depend on private RCA exact/pair correctness when the
    # observable action/recovery outcome and twin gate are held fixed.
    sys_good_rca = end_to_end_reward(good_rca, good_action)["system_reward"]
    sys_bad_rca = end_to_end_reward(bad_rca, good_action)["system_reward"]
    _assert_close(float(sys_good_rca), float(sys_bad_rca), tol=1e-12,
                  message="system reward leaked private RCA correctness")

    # Raw local scalar rewards are intentionally extreme in fixtures; factorized
    # returns must remain bounded and therefore cannot be reusing those raw values.
    for name, obj in {
        "good_r_bad_a": good_r_bad_a,
        "bad_r_good_a": bad_r_good_a,
        "good_both": good_both,
        "bad_both": bad_both,
    }.items():
        if not (-1.0 <= float(obj["rca_policy_return"]) <= 1.0):
            raise AssertionError(f"{name}: RCA policy return out of bounds")
        if not (-1.0 <= float(obj["action_policy_return"]) <= 1.0):
            raise AssertionError(f"{name}: Action policy return out of bounds")

    no_op = _action_result(
        safe=True, repairs=False, target_reduction=0.0, global_reduction=0.0,
        target_sla=False, sla=False, resolved=False, mutation=False, verify=True,
    )
    no_op_reward = end_to_end_reward(good_rca, no_op)
    if float(no_op_reward["system_quality"]) != 0.0:
        raise AssertionError("safe no-op must not receive positive system quality")
    if no_op_reward["success"]:
        raise AssertionError("safe no-op must not be an end-to-end success")

    return {
        "good_rca_bad_action": good_r_bad_a,
        "bad_rca_good_action": bad_r_good_a,
        "good_both": good_both,
        "bad_both": bad_both,
        "safe_noop": no_op_reward,
        "system_reward_private_rca_independence": True,
        "raw_local_reward_not_reused": True,
    }


def audit_buffer(path: str | Path) -> dict[str, Any]:
    rows = load_grpo_dataset(path, require_policy_credit=True, require_old_logprobs=False)
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group[str(row["optimizer_group_id"])][str(row["trajectory_id"])].append(row)

    for gid, trajectories in by_group.items():
        if len(trajectories) < 2:
            raise AssertionError(f"{gid}: fewer than two trajectories")
        for tid, trows in trajectories.items():
            weights = [float(r["optimizer_sample_weight"]) for r in trows]
            _assert_close(sum(weights), 1.0, tol=1e-10,
                          message=f"{gid}/{tid}: decision weights must sum to 1")

    return {
        "path": str(Path(path).expanduser()),
        "num_rows": len(rows),
        "num_optimizer_groups": len(by_group),
        "equal_trajectory_weighting": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit factorized trajectory-GRPO math and optional rollout buffers.")
    ap.add_argument("--rca_buffer", default=None)
    ap.add_argument("--action_buffer", default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {
        "advantage_math": audit_advantage_math(),
        "clipped_surrogate_math": audit_clipped_surrogate(),
        "kl_math": audit_kl_math(),
        "factorized_reward": audit_factorized_reward(),
    }
    if args.rca_buffer:
        report["rca_buffer"] = audit_buffer(args.rca_buffer)
    if args.action_buffer:
        report["action_buffer"] = audit_buffer(args.action_buffer)

    report["status"] = "PASS"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
