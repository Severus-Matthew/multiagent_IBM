from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .grpo_math import group_relative_advantages


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(Path(path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def compute_legacy_group_advantages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy single-stage group normalization over individual sample rewards."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id", ""))].append(row)

    out: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        result = group_relative_advantages(
            [float(x.get("reward", 0.0) or 0.0) for x in group],
            scale_by_std=True,
        )
        for row, advantage in zip(group, result.advantages):
            updated = dict(row)
            updated["group_reward_mean"] = round(result.mean, 6)
            updated["group_reward_std"] = round(result.std, 6)
            updated["advantage"] = round(float(advantage), 6)
            out.append(updated)
    return out


def compute_factorized_policy_advantages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute policy advantages correctly for factorized joint buffers.

    A joint trajectory can contain multiple RCA or Action decision rows. Those
    rows all belong to one sampled trajectory and share one policy return. The
    GRPO baseline must therefore be computed over *unique trajectories* in the
    optimizer group, not over decision rows. Row-level normalization would bias
    the baseline toward trajectories with more retry decisions.
    """
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    original_order: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))

    for row in rows:
        gid = str(row.get("optimizer_group_id") or "")
        tid = str(row.get("trajectory_id") or "")
        if not gid or not tid:
            raise ValueError("factorized rows require optimizer_group_id and trajectory_id")
        groups[gid][tid].append(row)

    trajectory_credit: dict[tuple[str, str], tuple[float, float, float]] = {}
    for gid, trajectories in groups.items():
        tids = sorted(trajectories)
        rewards: list[float] = []
        for tid in tids:
            vals = {round(float(r.get("policy_reward", 0.0)), 12) for r in trajectories[tid]}
            if len(vals) != 1:
                raise ValueError(f"inconsistent policy_reward within trajectory {tid}: {sorted(vals)}")
            rewards.append(next(iter(vals)))

        result = group_relative_advantages(rewards, scale_by_std=True)
        for tid, reward, advantage in zip(tids, rewards, result.advantages):
            trajectory_credit[(gid, tid)] = (reward, float(advantage), result.std)

    out: list[dict[str, Any]] = []
    for _, row in original_order:
        gid = str(row.get("optimizer_group_id") or "")
        tid = str(row.get("trajectory_id") or "")
        reward, advantage, std = trajectory_credit[(gid, tid)]
        updated = dict(row)
        updated["policy_reward"] = round(float(reward), 6)
        updated["policy_advantage"] = round(float(advantage), 6)
        metadata = dict(updated.get("metadata", {}) or {})
        metadata["policy_advantage_recomputed_from_unique_trajectories"] = True
        metadata["optimizer_group_return_std"] = round(float(std), 6)
        updated["metadata"] = metadata
        out.append(updated)
    return out


def compute_group_advantages(rows: list[dict[str, Any]], mode: str = "auto") -> list[dict[str, Any]]:
    if mode not in {"auto", "legacy", "factorized"}:
        raise ValueError("mode must be auto, legacy, or factorized")
    detected_factorized = any(
        "policy_reward" in row or "optimizer_group_id" in row or "trajectory_id" in row
        for row in rows
    )
    if mode == "factorized" or (mode == "auto" and detected_factorized):
        return compute_factorized_policy_advantages(rows)
    return compute_legacy_group_advantages(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(x.get("reward", 0.0) or 0.0) for x in rows]
    advantages = [float(x.get("advantage", 0.0) or 0.0) for x in rows if x.get("advantage") is not None]
    policy_rewards = [float(x.get("policy_reward")) for x in rows if x.get("policy_reward") is not None]
    policy_advantages = [float(x.get("policy_advantage")) for x in rows if x.get("policy_advantage") is not None]
    groups = {x.get("group_id") for x in rows}
    optimizer_groups = {x.get("optimizer_group_id") for x in rows if x.get("optimizer_group_id")}
    return {
        "num_samples": len(rows),
        "num_groups": len(groups),
        "num_optimizer_groups": len(optimizer_groups),
        "reward_mean_row_level": round(mean(rewards), 6) if rewards else None,
        "reward_std_row_level": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "advantage_mean_row_level": round(mean(advantages), 6) if advantages else None,
        "policy_reward_mean_row_level": round(mean(policy_rewards), 6) if policy_rewards else None,
        "policy_advantage_mean_row_level": round(mean(policy_advantages), 6) if policy_advantages else None,
        "success_samples": sum(1 for x in rows if x.get("success")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute or recompute GRPO advantages without corrupting factorized trajectory credit.")
    ap.add_argument("--input", required=True, help="Input GRPO sample jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl with advantage fields")
    ap.add_argument("--summary", default=None, help="Optional summary json path")
    ap.add_argument("--mode", choices=["auto", "legacy", "factorized"], default="auto")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    out = compute_group_advantages(rows, mode=args.mode)
    write_jsonl(args.output, out)
    summary = summarize(out)
    summary["mode"] = args.mode
    if args.summary:
        Path(args.summary).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
