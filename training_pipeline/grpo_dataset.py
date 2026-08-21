from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

REQUIRED_FIELDS = [
    "stage", "scenario_id", "group_id", "sample_id", "policy_role",
    "policy_prompt", "completion", "reward", "advantage",
]

FACTORized_REQUIRED_FIELDS = [
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


def validate_grpo_sample(row: dict[str, Any], require_policy_credit: bool = False) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"missing:{field}")
    if require_policy_credit:
        for field in FACTORized_REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"missing:{field}")
        metadata = row.get("metadata", {}) or {}
        if not metadata.get("optimizer_role"):
            errors.append("missing:metadata.optimizer_role")
        if metadata.get("optimizer_advantage_field") != "policy_advantage":
            errors.append("invalid:metadata.optimizer_advantage_field")
    if not str(row.get("policy_prompt", "")).strip():
        errors.append("empty:policy_prompt")
    if not str(row.get("completion", "")).strip():
        errors.append("empty:completion")
    if row.get("old_logprob_sum") is None:
        # This is acceptable for debug rollouts. Real GRPO training must fill it.
        pass
    return errors


def load_grpo_dataset(
    path: str | Path,
    require_old_logprobs: bool = False,
    require_policy_credit: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    bad = []
    for i, row in enumerate(iter_jsonl(path), start=1):
        errors = validate_grpo_sample(row, require_policy_credit=require_policy_credit)
        if require_old_logprobs and row.get("old_logprob_sum") is None:
            errors.append("missing:old_logprob_sum")
        if errors:
            bad.append({"line": i, "sample_id": row.get("sample_id"), "errors": errors})
        else:
            rows.append(row)
    if bad:
        raise ValueError(json.dumps({"invalid_samples": bad[:20], "num_invalid": len(bad)}, indent=2))
    return rows


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(x.get("reward", 0.0) or 0.0) for x in rows]
    advantages = [float(x.get("advantage", 0.0) or 0.0) for x in rows if x.get("advantage") is not None]
    policy_rewards = [float(x.get("policy_reward", 0.0) or 0.0) for x in rows if x.get("policy_reward") is not None]
    policy_advantages = [float(x.get("policy_advantage", 0.0) or 0.0) for x in rows if x.get("policy_advantage") is not None]
    groups = {x.get("group_id") for x in rows}
    optimizer_groups = {x.get("optimizer_group_id") for x in rows if x.get("optimizer_group_id")}
    scenarios = {x.get("scenario_id") for x in rows}
    adapters = {x.get("adapter_id") for x in rows if x.get("adapter_id")}
    return {
        "num_samples": len(rows),
        "num_groups": len(groups),
        "num_optimizer_groups": len(optimizer_groups),
        "num_scenarios": len(scenarios),
        "reward_mean": round(mean(rewards), 6) if rewards else None,
        "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "advantage_mean": round(mean(advantages), 6) if advantages else None,
        "advantage_std": round(pstdev(advantages), 6) if len(advantages) > 1 else 0.0,
        "policy_reward_mean": round(mean(policy_rewards), 6) if policy_rewards else None,
        "policy_reward_std": round(pstdev(policy_rewards), 6) if len(policy_rewards) > 1 else 0.0,
        "policy_advantage_mean": round(mean(policy_advantages), 6) if policy_advantages else None,
        "policy_advantage_std": round(pstdev(policy_advantages), 6) if len(policy_advantages) > 1 else 0.0,
        "missing_old_logprob_sum": sum(1 for x in rows if x.get("old_logprob_sum") is None),
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
        help="Require factorized end-to-end policy_reward/policy_advantage fields.",
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
