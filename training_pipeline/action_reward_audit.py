from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    from .rca_reward_audit import _family_from_scenario_id, _metadata_from_processed_states, _rate, _read_jsonl, _stats, _task_from_scenario_id, _write_json
except Exception:  # pragma: no cover
    _family_from_scenario_id = None
    _metadata_from_processed_states = None
    _rate = None
    _read_jsonl = None
    _stats = None
    _task_from_scenario_id = None
    _write_json = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _fallback_rate(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def _fallback_stats(xs: Iterable[Any]) -> dict[str, Any]:
    vals = [_safe_float(x) for x in xs]
    if not vals:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": round(mean(vals), 6),
        "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def _rate2(num: int | float, den: int | float) -> float:
    return _rate(num, den) if _rate else _fallback_rate(num, den)


def _stats2(xs: Iterable[Any]) -> dict[str, Any]:
    return _stats(xs) if _stats else _fallback_stats(xs)


def _read_jsonl2(path: str | Path | None) -> list[dict[str, Any]]:
    if _read_jsonl:
        return _read_jsonl(path)
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        return []
    rows = []
    with p.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json2(obj: dict[str, Any], path: str | Path | None) -> None:
    if _write_json:
        _write_json(obj, path)
        return
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _family(sid: str) -> str:
    return _family_from_scenario_id(sid) if _family_from_scenario_id else str(sid or "unknown").split("-", 1)[0]


def _task(sid: str) -> str:
    return _task_from_scenario_id(sid) if _task_from_scenario_id else "unknown"


def _metadata(processed_states: str | Path | None) -> dict[str, dict[str, Any]]:
    return _metadata_from_processed_states(processed_states) if _metadata_from_processed_states else {}


def _components(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("reward_components", {}) or {}


def _group_stats(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[str(s.get("group_id") or "missing_group")].append(s)

    group_reward_stds: list[float] = []
    group_adv_stds: list[float] = []
    nonzero_reward_std = nonzero_adv_std = 0
    successful_groups = selected_groups = 0
    groups_with_multiple_action_families = 0

    for rows in groups.values():
        rewards = [_safe_float(r.get("reward")) for r in rows]
        advs = [_safe_float(r.get("advantage")) for r in rows if r.get("advantage") is not None]
        r_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        a_std = pstdev(advs) if len(advs) > 1 else 0.0
        group_reward_stds.append(r_std)
        group_adv_stds.append(a_std)
        nonzero_reward_std += int(r_std > 1e-8)
        nonzero_adv_std += int(a_std > 1e-8)
        successful_groups += int(any(r.get("success") for r in rows))
        selected_groups += int(any((r.get("metadata") or {}).get("selected_for_episode_history") for r in rows))
        families = {str((r.get("metadata") or {}).get("action_family") or "unknown") for r in rows}
        groups_with_multiple_action_families += int(len(families) > 1)

    return {
        "num_groups": len(groups),
        "groups_with_success_sample": successful_groups,
        "groups_with_selected_sample": selected_groups,
        "groups_with_multiple_action_families": groups_with_multiple_action_families,
        "groups_with_zero_reward_std": len(groups) - nonzero_reward_std,
        "groups_with_nonzero_reward_std": nonzero_reward_std,
        "groups_with_zero_advantage_std": len(groups) - nonzero_adv_std,
        "groups_with_nonzero_advantage_std": nonzero_adv_std,
        "group_reward_std": _stats2(group_reward_stds),
        "group_advantage_std": _stats2(group_adv_stds),
    }, groups


def _episode_terminal_failure(ep: dict[str, Any]) -> bool:
    term = ep.get("terminal") or {}
    if isinstance(term, dict) and (term.get("components", {}) or {}).get("terminal_failure"):
        return True
    return bool(term and not ep.get("success"))


def _action_family_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        buckets[str((s.get("metadata") or {}).get("action_family") or "unknown")].append(s)

    out: dict[str, Any] = {}
    for name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        comps = [_components(r) for r in rows]
        out[name] = {
            "samples": len(rows),
            "success_samples": sum(1 for r in rows if r.get("success")),
            "success_sample_rate": _rate2(sum(1 for r in rows if r.get("success")), len(rows)),
            "reward": _stats2([r.get("reward") for r in rows]),
            "selected_count": sum(1 for r in rows if (r.get("metadata") or {}).get("selected_for_episode_history")),
            "safe_rate": _rate2(sum(1 for c in comps if c.get("safe")), len(comps)),
            "resolved_rate": _rate2(sum(1 for c in comps if c.get("resolved")), len(comps)),
            "twin_resolved_rate": _rate2(sum(1 for c in comps if c.get("twin_resolved")), len(comps)),
            "target_sla_restored_rate": _rate2(sum(1 for c in comps if c.get("target_sla_restored")), len(comps)),
            "sla_restored_rate": _rate2(sum(1 for c in comps if c.get("sla_restored")), len(comps)),
            "action_repairs_fault_type_rate": _rate2(sum(1 for c in comps if c.get("action_repairs_fault_type")), len(comps)),
            "global_symptom_reduction": _stats2([c.get("global_symptom_reduction", c.get("symptom_reduction", 0.0)) for c in comps]),
            "target_symptom_reduction": _stats2([c.get("target_symptom_reduction", 0.0) for c in comps]),
        }
    return out


def _bucket_summary(items: list[dict[str, Any]], meta: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        sid = str(row.get("scenario_id") or "")
        default = _family(sid) if key == "family" else _task(sid)
        value = str((meta.get(sid, {}) or {}).get(key) or default or "unknown")
        buckets[value].append(row)
    out: dict[str, Any] = {}
    for name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        comps = [_components(r) for r in rows]
        out[name] = {
            "samples": len(rows),
            "success_samples": sum(1 for r in rows if r.get("success")),
            "success_sample_rate": _rate2(sum(1 for r in rows if r.get("success")), len(rows)),
            "reward": _stats2([r.get("reward") for r in rows]),
            "target_sla_restored_rate": _rate2(sum(1 for c in comps if c.get("target_sla_restored")), len(comps)),
            "sla_restored_rate": _rate2(sum(1 for c in comps if c.get("sla_restored")), len(comps)),
            "twin_resolved_rate": _rate2(sum(1 for c in comps if c.get("twin_resolved")), len(comps)),
            "target_symptom_reduction_mean": round(mean([_safe_float(c.get("target_symptom_reduction")) for c in comps]), 6) if comps else 0.0,
        }
    return out


def _episode_bucket_summary(episodes: list[dict[str, Any]], meta: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        sid = str(ep.get("scenario_id") or "")
        default = _family(sid) if key == "family" else _task(sid)
        value = str((meta.get(sid, {}) or {}).get(key) or default or "unknown")
        buckets[value].append(ep)
    out: dict[str, Any] = {}
    for name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        passed = sum(1 for r in rows if r.get("success"))
        skipped = sum(1 for r in rows if r.get("skipped_action"))
        out[name] = {
            "episodes": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "success_rate": _rate2(passed, len(rows)),
            "skipped_action_gate": skipped,
            "skipped_action_gate_rate": _rate2(skipped, len(rows)),
            "terminal_failures": sum(1 for r in rows if _episode_terminal_failure(r)),
            "avg_attempts": round(mean([len(r.get("attempts") or []) for r in rows]), 6) if rows else 0.0,
        }
    return out


def audit_action_rollout(
    rollout_dir: str | Path | None = None,
    grpo_samples_path: str | Path | None = None,
    rollouts_path: str | Path | None = None,
    processed_states: str | Path | None = None,
) -> dict[str, Any]:
    if rollout_dir:
        rd = Path(rollout_dir).expanduser()
        grpo_samples_path = grpo_samples_path or rd / "grpo_samples.jsonl"
        rollouts_path = rollouts_path or rd / "rollouts.jsonl"

    samples_path = Path(grpo_samples_path).expanduser() if grpo_samples_path else None
    episodes_path = Path(rollouts_path).expanduser() if rollouts_path else None
    samples = [s for s in _read_jsonl2(samples_path) if s.get("stage") == "action"]
    episodes = [e for e in _read_jsonl2(episodes_path) if e.get("stage") == "action"]
    meta = _metadata(processed_states)

    comps = [_components(s) for s in samples]
    rewards = [_safe_float(s.get("reward")) for s in samples]
    advantages = [_safe_float(s.get("advantage")) for s in samples if s.get("advantage") is not None]
    group_summary, _ = _group_stats(samples)

    sample_success = sum(1 for s in samples if s.get("success"))
    episode_success = sum(1 for e in episodes if e.get("success"))
    skipped_gate = sum(1 for e in episodes if e.get("skipped_action"))

    summary = {
        "inputs": {
            "rollout_dir": str(Path(rollout_dir).expanduser()) if rollout_dir else None,
            "grpo_samples_jsonl": str(samples_path) if samples_path else None,
            "grpo_samples_jsonl_exists": bool(samples_path and samples_path.exists()),
            "rollouts_jsonl": str(episodes_path) if episodes_path else None,
            "rollouts_jsonl_exists": bool(episodes_path and episodes_path.exists()),
            "processed_states": str(Path(processed_states).expanduser()) if processed_states else None,
        },
        "episodes": {
            "num_episodes": len(episodes),
            "passed": episode_success,
            "failed": len(episodes) - episode_success,
            "success_rate": _rate2(episode_success, len(episodes)),
            "skipped_action_gate": skipped_gate,
            "skipped_action_gate_rate": _rate2(skipped_gate, len(episodes)),
            "avg_attempts": round(mean([len(e.get("attempts") or []) for e in episodes]), 6) if episodes else 0.0,
            "terminal_failures": sum(1 for e in episodes if _episode_terminal_failure(e)),
        },
        "samples": {
            "num_samples": len(samples),
            "success_samples": sample_success,
            "success_sample_rate": _rate2(sample_success, len(samples)),
            "missing_old_logprob_sum": sum(1 for s in samples if s.get("old_logprob_sum") is None),
            "reward": _stats2(rewards),
            "advantage": _stats2(advantages),
        },
        "groups": group_summary,
        "reward_components": {
            "safe_rate": _rate2(sum(1 for c in comps if c.get("safe")), len(comps)),
            "resolved_rate": _rate2(sum(1 for c in comps if c.get("resolved")), len(comps)),
            "twin_resolved_rate": _rate2(sum(1 for c in comps if c.get("twin_resolved")), len(comps)),
            "target_sla_restored_rate": _rate2(sum(1 for c in comps if c.get("target_sla_restored")), len(comps)),
            "sla_restored_rate": _rate2(sum(1 for c in comps if c.get("sla_restored")), len(comps)),
            "action_repairs_fault_type_rate": _rate2(sum(1 for c in comps if c.get("action_repairs_fault_type")), len(comps)),
            "has_verification_command_rate": _rate2(sum(1 for c in comps if c.get("has_verification_command")), len(comps)),
            "has_mutating_command_rate": _rate2(sum(1 for c in comps if c.get("has_mutating_command")), len(comps)),
            "global_symptom_reduction": _stats2([c.get("global_symptom_reduction", c.get("symptom_reduction", 0.0)) for c in comps]),
            "target_symptom_reduction": _stats2([c.get("target_symptom_reduction", 0.0) for c in comps]),
            "avg_instruction_tokens": round(mean([_safe_float(c.get("instruction_tokens")) for c in comps]), 6) if comps else 0.0,
        },
        "by_action_family": _action_family_summary(samples),
        "by_family_samples": _bucket_summary(samples, meta, "family"),
        "by_family_episodes": _episode_bucket_summary(episodes, meta, "family"),
        "by_task_samples": _bucket_summary(samples, meta, "task"),
        "by_task_episodes": _episode_bucket_summary(episodes, meta, "task"),
        "warnings": [],
    }

    if samples_path and not samples_path.exists():
        summary["warnings"].append(f"grpo_samples_jsonl not found: {samples_path}")
    if episodes_path and not episodes_path.exists():
        summary["warnings"].append(f"rollouts_jsonl not found: {episodes_path}")
    if not samples:
        summary["warnings"].append("No action GRPO samples found; run train_action_grpo with --group_size > 1 before auditing.")
    if not episodes:
        summary["warnings"].append("No action episode rollouts found; run train_action_grpo before auditing.")
    if samples and group_summary["groups_with_nonzero_reward_std"] == 0:
        summary["warnings"].append("All action GRPO groups have zero reward variance; no group-relative action learning signal yet.")
    if samples and summary["samples"]["missing_old_logprob_sum"] == len(samples):
        summary["warnings"].append("All action samples are missing old_logprob_sum; expected for structured/fixed debug policies, not for real GRPO model updates.")
    if samples and summary["reward_components"]["safe_rate"] < 1.0:
        summary["warnings"].append("Some action samples produced unsafe commands; check command policy and safety filters.")
    if samples and summary["reward_components"]["target_sla_restored_rate"] == 0.0:
        summary["warnings"].append("No action sample restored target SLA; action family mapping or verifier may be too strict.")
    if episodes and skipped_gate == len(episodes):
        summary["warnings"].append("All action episodes were skipped by RCA gate; improve RCA before action training can collect samples.")
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    print(json.dumps({
        "inputs": summary["inputs"],
        "episodes": summary["episodes"],
        "samples": summary["samples"],
        "groups": summary["groups"],
        "reward_components": summary["reward_components"],
        "warnings": summary["warnings"],
    }, indent=2, sort_keys=True))

    print("\nAction-family sample buckets:")
    for name, row in list(summary["by_action_family"].items())[:20]:
        print(
            f"  {name:24s} samples={int(row.get('samples', 0)):5d} "
            f"succ={_safe_float(row.get('success_sample_rate')):.3f} "
            f"reward={_safe_float((row.get('reward') or {}).get('mean')):.3f} "
            f"target_sla={_safe_float(row.get('target_sla_restored_rate')):.3f} "
            f"global_sla={_safe_float(row.get('sla_restored_rate')):.3f} "
            f"selected={int(row.get('selected_count', 0))}"
        )

    print("\nEpisode pass/skipped rate by family:")
    for name, row in list(summary["by_family_episodes"].items())[:20]:
        print(
            f"  {name:55s} episodes={int(row.get('episodes', 0)):4d} "
            f"pass={_safe_float(row.get('success_rate')):.3f} "
            f"skip={_safe_float(row.get('skipped_action_gate_rate')):.3f} "
            f"avg_attempts={_safe_float(row.get('avg_attempts')):.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit action reward and GRPO rollout quality.")
    ap.add_argument("--rollout_dir", default=None, help="Directory containing action rollouts.jsonl and grpo_samples.jsonl.")
    ap.add_argument("--grpo_samples", default=None, help="Optional explicit grpo_samples.jsonl path.")
    ap.add_argument("--rollouts", default=None, help="Optional explicit rollouts.jsonl path.")
    ap.add_argument("--processed_states", default=None, help="Optional processed_states directory for exact family/task metadata.")
    ap.add_argument("--output", default=None, help="Optional path to write full audit JSON.")
    ap.add_argument("--quiet", action="store_true", help="Only write --output, do not print human summary.")
    args = ap.parse_args()

    summary = audit_action_rollout(
        rollout_dir=args.rollout_dir,
        grpo_samples_path=args.grpo_samples,
        rollouts_path=args.rollouts,
        processed_states=args.processed_states,
    )
    _write_json2(summary, args.output)
    if not args.quiet:
        _print_human(summary)


if __name__ == "__main__":
    main()
