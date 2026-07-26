from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .wandb_logger import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare RCA prompt policies on the same scenario split.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--policies", default="heuristic,operator,qwen_stub",
                    help="Comma-separated policies: heuristic,operator,qwen_stub")
    ap.add_argument("--use_behavioral_twin", action="store_true")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_iterations", type=int, default=5)
    ap.add_argument("--selection_strategy", choices=["best", "sample0"], default="best")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rca_solver", choices=["heuristic"], default="heuristic")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--wandb_entity", default=DEFAULT_WANDB_ENTITY)
    ap.add_argument("--wandb_tags", default="comparison,rca")
    args = ap.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    rows: dict[str, Any] = {}
    for policy in policies:
        out_dir = output_root / policy
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[COMPARE] Running policy={policy} -> {out_dir}")

        train_cmd = [
            sys.executable, "-m", "training_pipeline.train_rca_grpo",
            "--processed_states", args.processed_states,
            "--scenario_ids", args.scenario_ids,
            "--output_dir", str(out_dir),
            "--instruction_policy", policy,
            "--rca_solver", args.rca_solver,
            "--group_size", str(args.group_size),
            "--max_iterations", str(args.max_iterations),
            "--selection_strategy", args.selection_strategy,
        ]
        if args.limit is not None:
            train_cmd += ["--limit", str(args.limit)]
        if args.use_behavioral_twin:
            train_cmd.append("--use_behavioral_twin")
        if args.wandb:
            train_cmd += [
                "--wandb",
                "--wandb_project", args.wandb_project,
                "--wandb_entity", args.wandb_entity,
                "--wandb_run_name", f"rca-{policy}-{output_root.name}",
                "--wandb_tags", args.wandb_tags + f",{policy}",
            ]

        subprocess.run(train_cmd, check=True)

        audit_cmd = [
            sys.executable, "-m", "training_pipeline.rca_reward_audit",
            "--rollout_dir", str(out_dir),
            "--processed_states", args.processed_states,
            "--output", str(out_dir / "reward_audit.json"),
        ]
        subprocess.run(audit_cmd, check=True)

        rows[policy] = _compact_metrics(out_dir)

    summary = {
        "processed_states": args.processed_states,
        "scenario_ids": args.scenario_ids,
        "output_root": str(output_root),
        "policies": policies,
        "rca_solver": args.rca_solver,
        "group_size": args.group_size,
        "max_iterations": args.max_iterations,
        "selection_strategy": args.selection_strategy,
        "use_behavioral_twin": args.use_behavioral_twin,
        "results": rows,
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (output_root / "comparison_table.md").write_text(_markdown_table(rows), encoding="utf-8")
    print("\n[COMPARE] Summary")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"[COMPARE] wrote {output_root / 'comparison_summary.json'}")
    print(f"[COMPARE] wrote {output_root / 'comparison_table.md'}")


def _compact_metrics(out_dir: Path) -> dict[str, Any]:
    summary = _read_json(out_dir / "summary.json")
    audit = _read_json(out_dir / "reward_audit.json")
    reward = ((audit.get("samples", {}) or {}).get("reward", {}) or {})
    pair = ((audit.get("reward_components", {}) or {}).get("pair_score", {}) or {})
    twin = ((audit.get("reward_components", {}) or {}).get("twin_reproduction_score", {}) or {})
    groups = audit.get("groups", {}) or {}
    return {
        "total": summary.get("total"),
        "passed": summary.get("passed"),
        "failed": summary.get("failed"),
        "success_rate": summary.get("success_rate"),
        "grpo_samples": summary.get("grpo_samples"),
        "reward_mean": reward.get("mean"),
        "reward_std": reward.get("std"),
        "pair_score_mean": pair.get("mean"),
        "pair_score_std": pair.get("std"),
        "twin_score_mean": twin.get("mean"),
        "twin_score_std": twin.get("std"),
        "count_mismatch_rate": (audit.get("reward_components", {}) or {}).get("count_mismatch_rate"),
        "invalid_format_rate": (audit.get("reward_components", {}) or {}).get("invalid_format_rate"),
        "groups_with_nonzero_reward_std": groups.get("groups_with_nonzero_reward_std"),
        "group_reward_std_mean": ((groups.get("group_reward_std", {}) or {}).get("mean")),
        "warnings": audit.get("warnings", []),
        "output_dir": str(out_dir),
    }


def _markdown_table(rows: dict[str, Any]) -> str:
    headers = [
        "policy", "passed", "total", "success_rate", "reward_mean", "pair_mean",
        "twin_mean", "twin_std", "count_mismatch", "nonzero_groups",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for policy, r in rows.items():
        vals = [
            policy,
            r.get("passed"),
            r.get("total"),
            _fmt(r.get("success_rate")),
            _fmt(r.get("reward_mean")),
            _fmt(r.get("pair_score_mean")),
            _fmt(r.get("twin_score_mean")),
            _fmt(r.get("twin_score_std")),
            _fmt(r.get("count_mismatch_rate")),
            r.get("groups_with_nonzero_reward_std"),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


if __name__ == "__main__":
    main()
