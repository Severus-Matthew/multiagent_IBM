from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from .grpo_math import group_relative_advantages

REQUIRED_FIELDS = [
    "stage", "scenario_id", "group_id", "sample_id", "policy_role",
    "policy_prompt", "completion", "reward", "advantage",
]

FACTORIZED_REQUIRED_FIELDS = [
    "policy_reward",
    "policy_advantage",
    "optimizer_group_id",
    "trajectory_group_id",
    "trajectory_id",
]


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(Path(path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _finite_float_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_finite_number(x) for x in value)


def _valid_token_id_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for x in value:
        if isinstance(x, bool):
            return False
        try:
            i = int(x)
        except Exception:
            return False
        if i < 0 or float(x) != float(i):
            return False
    return True


def validate_grpo_sample(
    row: dict[str, Any],
    *,
    require_policy_credit: bool = False,
    require_old_logprobs: bool = False,
) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"missing:{field}")

    if not str(row.get("policy_prompt", "")).strip():
        errors.append("empty:policy_prompt")
    if not str(row.get("completion", "")).strip():
        errors.append("empty:completion")

    if require_policy_credit:
        for field in FACTORIZED_REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"missing:{field}")
        if not _is_finite_number(row.get("policy_reward")):
            errors.append("invalid:policy_reward")
        if not _is_finite_number(row.get("policy_advantage")):
            errors.append("invalid:policy_advantage")

        metadata = row.get("metadata", {}) or {}
        stage = str(row.get("stage") or "")
        expected_role = {"rca": "rca_policy", "action": "action_policy"}.get(stage)
        expected_adapter = {"rca": "lora_rca", "action": "lora_action"}.get(stage)
        if not metadata.get("optimizer_role"):
            errors.append("missing:metadata.optimizer_role")
        elif expected_role and metadata.get("optimizer_role") != expected_role:
            errors.append("invalid:metadata.optimizer_role")
        if metadata.get("optimizer_advantage_field") != "policy_advantage":
            errors.append("invalid:metadata.optimizer_advantage_field")
        if row.get("adapter_id") is not None and expected_adapter and row.get("adapter_id") != expected_adapter:
            errors.append("invalid:adapter_id")

    if require_old_logprobs:
        old_logprobs = row.get("old_logprobs")
        token_ids = row.get("completion_token_ids")
        if not _finite_float_list(old_logprobs):
            errors.append("invalid:old_logprobs")
        if not _valid_token_id_list(token_ids):
            errors.append("invalid:completion_token_ids")
        if isinstance(old_logprobs, list) and isinstance(token_ids, list) and len(old_logprobs) != len(token_ids):
            errors.append("mismatch:old_logprobs_vs_completion_token_ids")

        old_sum = row.get("old_logprob_sum")
        if not _is_finite_number(old_sum):
            errors.append("invalid:old_logprob_sum")
        elif _finite_float_list(old_logprobs):
            expected = float(sum(float(x) for x in old_logprobs))
            if abs(float(old_sum) - expected) > 1e-5 * max(1.0, abs(expected)):
                errors.append("mismatch:old_logprob_sum")

        ref_logprobs = row.get("ref_logprobs")
        if ref_logprobs is not None:
            if not _finite_float_list(ref_logprobs):
                errors.append("invalid:ref_logprobs")
            elif isinstance(token_ids, list) and len(ref_logprobs) != len(token_ids):
                errors.append("mismatch:ref_logprobs_vs_completion_token_ids")

    return errors


def _validate_factorized_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute expected trajectory-level advantages and compare exactly.

    Individual decision rows from the same trajectory intentionally share one
    policy reward/advantage. The group baseline must therefore be computed over
    unique trajectories, never over duplicated decision rows.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        optimizer_group_id = str(row.get("optimizer_group_id") or "")
        trajectory_id = str(row.get("trajectory_id") or "")
        grouped.setdefault(optimizer_group_id, {}).setdefault(trajectory_id, []).append(row)

    errors: list[dict[str, Any]] = []
    for optimizer_group_id, trajectories in grouped.items():
        if not optimizer_group_id:
            errors.append({"optimizer_group_id": optimizer_group_id, "error": "empty_optimizer_group_id"})
            continue
        if len(trajectories) < 2:
            errors.append({
                "optimizer_group_id": optimizer_group_id,
                "error": "optimizer_group_has_fewer_than_two_trajectories",
                "num_trajectories": len(trajectories),
            })
            continue

        ordered_ids = sorted(trajectories)
        rewards: list[float] = []
        stored_advantages: dict[str, float] = {}
        consistent = True

        for trajectory_id in ordered_ids:
            trows = trajectories[trajectory_id]
            t_rewards = {round(float(r.get("policy_reward", 0.0)), 12) for r in trows}
            t_advantages = {round(float(r.get("policy_advantage", 0.0)), 12) for r in trows}
            if len(t_rewards) != 1:
                errors.append({
                    "optimizer_group_id": optimizer_group_id,
                    "trajectory_id": trajectory_id,
                    "error": "inconsistent_policy_reward_within_trajectory",
                })
                consistent = False
                continue
            if len(t_advantages) != 1:
                errors.append({
                    "optimizer_group_id": optimizer_group_id,
                    "trajectory_id": trajectory_id,
                    "error": "inconsistent_policy_advantage_within_trajectory",
                })
                consistent = False
                continue
            rewards.append(next(iter(t_rewards)))
            stored_advantages[trajectory_id] = next(iter(t_advantages))

        if not consistent or len(rewards) != len(ordered_ids):
            continue

        expected = group_relative_advantages(rewards, scale_by_std=True)
        for trajectory_id, expected_adv in zip(ordered_ids, expected.advantages):
            actual = stored_advantages[trajectory_id]
            if abs(actual - expected_adv) > 2e-5:
                errors.append({
                    "optimizer_group_id": optimizer_group_id,
                    "trajectory_id": trajectory_id,
                    "error": "policy_advantage_math_mismatch",
                    "stored": actual,
                    "expected": expected_adv,
                })

    return errors


def load_grpo_dataset(
    path: str | Path,
    require_old_logprobs: bool = False,
    require_policy_credit: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for i, row in enumerate(iter_jsonl(path), start=1):
        errors = validate_grpo_sample(
            row,
            require_policy_credit=require_policy_credit,
            require_old_logprobs=require_old_logprobs,
        )
        if errors:
            bad.append({"line": i, "sample_id": row.get("sample_id"), "errors": errors})
        else:
            rows.append(row)

    if require_policy_credit and not bad:
        group_errors = _validate_factorized_groups(rows)
        if group_errors:
            bad.extend({"group_error": x} for x in group_errors[:50])

    if bad:
        raise ValueError(json.dumps({"invalid_samples": bad[:50], "num_invalid": len(bad)}, indent=2))
    return rows


def _unique_trajectory_policy_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    vals: dict[tuple[str, str], float] = {}
    for row in rows:
        gid = str(row.get("optimizer_group_id") or "")
        tid = str(row.get("trajectory_id") or "")
        if gid and tid and row.get(field) is not None:
            vals[(gid, tid)] = float(row[field])
    return list(vals.values())


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(x.get("reward", 0.0) or 0.0) for x in rows]
    advantages = [float(x.get("advantage", 0.0) or 0.0) for x in rows if x.get("advantage") is not None]
    policy_rewards = _unique_trajectory_policy_values(rows, "policy_reward")
    policy_advantages = _unique_trajectory_policy_values(rows, "policy_advantage")
    groups = {x.get("group_id") for x in rows}
    optimizer_groups = {x.get("optimizer_group_id") for x in rows if x.get("optimizer_group_id")}
    scenarios = {x.get("scenario_id") for x in rows}
    adapters = {x.get("adapter_id") for x in rows if x.get("adapter_id")}
    return {
        "num_samples": len(rows),
        "num_groups": len(groups),
        "num_optimizer_groups": len(optimizer_groups),
        "num_unique_trajectory_policy_values": len(policy_rewards),
        "num_scenarios": len(scenarios),
        "reward_mean_row_level": round(mean(rewards), 6) if rewards else None,
        "reward_std_row_level": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "advantage_mean_row_level": round(mean(advantages), 6) if advantages else None,
        "advantage_std_row_level": round(pstdev(advantages), 6) if len(advantages) > 1 else 0.0,
        "policy_reward_mean_trajectory_level": round(mean(policy_rewards), 6) if policy_rewards else None,
        "policy_reward_std_trajectory_level": round(pstdev(policy_rewards), 6) if len(policy_rewards) > 1 else 0.0,
        "policy_advantage_mean_trajectory_level": round(mean(policy_advantages), 6) if policy_advantages else None,
        "policy_advantage_std_trajectory_level": round(pstdev(policy_advantages), 6) if len(policy_advantages) > 1 else 0.0,
        "missing_old_logprob_sum": sum(1 for x in rows if x.get("old_logprob_sum") is None),
        "missing_old_logprobs": sum(1 for x in rows if not x.get("old_logprobs")),
        "missing_completion_token_ids": sum(1 for x in rows if not x.get("completion_token_ids")),
        "success_samples": sum(1 for x in rows if x.get("success")),
        "adapter_ids": sorted(str(x) for x in adapters),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate and summarize GRPO sample jsonl.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--require_old_logprobs", action="store_true")
    ap.add_argument(
        "--require_policy_credit",
        action="store_true",
        help="Require and mathematically verify factorized trajectory-level policy credit.",
    )
    args = ap.parse_args()
    rows = load_grpo_dataset(
        args.input,
        require_old_logprobs=args.require_old_logprobs,
        require_policy_credit=args.require_policy_credit,
    )
    print(json.dumps(summarize_dataset(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
