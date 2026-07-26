from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


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


def compute_group_advantages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id", ""))].append(row)

    out = []
    for group_id, group in groups.items():
        rewards = [float(x.get("reward", 0.0) or 0.0) for x in group]
        mu = mean(rewards) if rewards else 0.0
        sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
        denom = sigma if sigma > 1e-8 else 1.0
        for row in group:
            row = dict(row)
            row["group_reward_mean"] = round(mu, 6)
            row["group_reward_std"] = round(sigma, 6)
            row["advantage"] = round((float(row.get("reward", 0.0) or 0.0) - mu) / denom, 6)
            out.append(row)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(x.get("reward", 0.0) or 0.0) for x in rows]
    advantages = [float(x.get("advantage", 0.0) or 0.0) for x in rows if x.get("advantage") is not None]
    groups = {x.get("group_id") for x in rows}
    return {
        "num_samples": len(rows),
        "num_groups": len(groups),
        "reward_mean": round(mean(rewards), 6) if rewards else None,
        "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "advantage_mean": round(mean(advantages), 6) if advantages else None,
        "advantage_std": round(pstdev(advantages), 6) if len(advantages) > 1 else 0.0,
        "success_samples": sum(1 for x in rows if x.get("success")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute or recompute group-normalized GRPO advantages.")
    ap.add_argument("--input", required=True, help="Input grpo_samples.jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl with advantage fields")
    ap.add_argument("--summary", default=None, help="Optional summary json path")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    out = compute_group_advantages(rows)
    write_jsonl(args.output, out)
    summary = summarize(out)
    if args.summary:
        Path(args.summary).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
