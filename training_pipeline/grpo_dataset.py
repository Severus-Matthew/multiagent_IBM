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


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(Path(path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_grpo_sample(row: dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"missing:{field}")
    if not str(row.get("policy_prompt", "")).strip():
        errors.append("empty:policy_prompt")
    if not str(row.get("completion", "")).strip():
        errors.append("empty:completion")
    if row.get("old_logprob_sum") is None:
        # This is acceptable for debug rollouts. Real GRPO training must fill it.
        pass
    return errors


def load_grpo_dataset(path: str | Path, require_old_logprobs: bool = False) -> list[dict[str, Any]]:
    rows = []
    bad = []
    for i, row in enumerate(iter_jsonl(path), start=1):
        errors = validate_grpo_sample(row)
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
    groups = {x.get("group_id") for x in rows}
    scenarios = {x.get("scenario_id") for x in rows}
    return {
        "num_samples": len(rows),
        "num_groups": len(groups),
        "num_scenarios": len(scenarios),
        "reward_mean": round(mean(rewards), 6) if rewards else None,
        "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "advantage_mean": round(mean(advantages), 6) if advantages else None,
        "advantage_std": round(pstdev(advantages), 6) if len(advantages) > 1 else 0.0,
        "missing_old_logprob_sum": sum(1 for x in rows if x.get("old_logprob_sum") is None),
        "success_samples": sum(1 for x in rows if x.get("success")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate and summarize GRPO sample jsonl.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--require_old_logprobs", action="store_true")
    args = ap.parse_args()
    rows = load_grpo_dataset(args.input, require_old_logprobs=args.require_old_logprobs)
    print(json.dumps(summarize_dataset(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
