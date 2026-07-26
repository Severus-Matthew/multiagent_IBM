from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    from .data_loader import iter_scenarios
except Exception:  # pragma: no cover - keeps script usable if imported outside package context
    iter_scenarios = None


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"failed to parse {p}:{line_no}: {e}") from e
    return rows


def _write_json(obj: dict[str, Any], path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _rate(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def _stats(xs: Iterable[Any]) -> dict[str, Any]:
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


def _family_from_scenario_id(scenario_id: str) -> str:
    sid = str(scenario_id or "")
    if sid.startswith("gen_multifault__"):
        return "multifault"
    if sid.startswith("gen_"):
        sid = sid[len("gen_"):]
    if "-" in sid:
        return sid.split("-", 1)[0]
    return sid or "unknown"


def _task_from_scenario_id(scenario_id: str) -> str:
    sid = str(scenario_id or "")
    for task in ("detection", "localization", "analysis", "mitigation"):
        if f"-{task}-" in sid or sid.endswith(f"__{task}") or sid.endswith(f"-{task}"):
            return task
    if sid.startswith("gen_multifault__"):
        tail = sid.rsplit("__", 1)[-1]
        if tail in {"detection", "localization", "analysis", "mitigation"}:
            return tail
    return "unknown"


def _metadata_from_processed_states(processed_states: str | Path | None) -> dict[str, dict[str, Any]]:
    if not processed_states or iter_scenarios is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rec in iter_scenarios(processed_states, require_safe_redaction=False):
        fc = rec.full_state.get("fault_context", {}) or {}
        out[rec.scenario_id] = {
            "family": fc.get("fault_family") or _family_from_scenario_id(rec.scenario_id),
            "task": fc.get("task") or _task_from_scenario_id(rec.scenario_id),
            "is_multifault": bool(fc.get("is_multifault")) or (fc.get("fault_family") == "multifault"),
            "num_oracle_faults": len(fc.get("fault_instances") or []) or (1 if fc.get("faulty_service") else 0),
        }
    return out


def _sample_components(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("reward_components", {}) or {}


def _episode_terminal_failure(ep: dict[str, Any]) -> bool:
    term = ep.get("terminal") or {}
    if isinstance(term, dict) and (term.get("components", {}) or {}).get("terminal_failure"):
        return True
    return bool(term and not ep.get("success"))


def _group_stats(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[str(s.get("group_id"))].append(s)

    group_reward_stds: list[float] = []
    group_adv_stds: list[float] = []
    zero_reward_std = nonzero_reward_std = 0
    zero_adv_std = nonzero_adv_std = 0
    single_sample_groups = 0
    successful_groups = 0
    terminal_groups = 0

    for rows in groups.values():
        if len(rows) <= 1:
            single_sample_groups += 1
        rewards = [_safe_float(r.get("reward")) for r in rows]
        advs = [_safe_float(r.get("advantage")) for r in rows if r.get("advantage") is not None]
        r_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        a_std = pstdev(advs) if len(advs) > 1 else 0.0
        group_reward_stds.append(r_std)
        group_adv_stds.append(a_std)
        if r_std > 1e-8:
            nonzero_reward_std += 1
        else:
            zero_reward_std += 1
        if a_std > 1e-8:
            nonzero_adv_std += 1
        else:
            zero_adv_std += 1
        if any(r.get("success") for r in rows):
            successful_groups += 1
        if any(r.get("terminal") for r in rows):
            terminal_groups += 1

    return {
        "num_groups": len(groups),
        "single_sample_groups": single_sample_groups,
        "groups_with_success_sample": successful_groups,
        "groups_with_terminal_sample": terminal_groups,
        "groups_with_zero_reward_std": zero_reward_std,
        "groups_with_nonzero_reward_std": nonzero_reward_std,
        "groups_with_zero_advantage_std": zero_adv_std,
        "groups_with_nonzero_advantage_std": nonzero_adv_std,
        "group_reward_std": _stats(group_reward_stds),
        "group_advantage_std": _stats(group_adv_stds),
    }, groups


def _bucket_summary(items: list[dict[str, Any]], meta: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        sid = str(row.get("scenario_id") or "")
        default = _family_from_scenario_id(sid) if key == "family" else _task_from_scenario_id(sid)
        value = str((meta.get(sid, {}) or {}).get(key) or default or "unknown")
        buckets[value].append(row)

    out: dict[str, Any] = {}
    for name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rewards = [_safe_float(r.get("reward")) for r in rows]
        comps = [_sample_components(r) for r in rows]
        out[name] = {
            "samples": len(rows),
            "success_samples": sum(1 for r in rows if r.get("success")),
            "success_sample_rate": _rate(sum(1 for r in rows if r.get("success")), len(rows)),
            "terminal_samples": sum(1 for r in rows if r.get("terminal")),
            "reward": _stats(rewards),
            "pair_score_mean": round(mean([_safe_float(c.get("pair_score")) for c in comps]), 6) if comps else 0.0,
            "twin_reproduction_score_mean": round(mean([_safe_float(c.get("twin_reproduction_score")) for c in comps]), 6) if comps else 0.0,
            "invalid_format_rate": _rate(sum(1 for c in comps if c.get("invalid_format")), len(comps)),
            "count_mismatch_rate": _rate(sum(1 for c in comps if _safe_float(c.get("count_mismatch")) > 0), len(comps)),
            "repeated_guess_rate": _rate(sum(1 for c in comps if c.get("repeated_wrong_guess")), len(comps)),
        }
    return out


def _episode_bucket_summary(episodes: list[dict[str, Any]], meta: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        sid = str(ep.get("scenario_id") or "")
        default = _family_from_scenario_id(sid) if key == "family" else _task_from_scenario_id(sid)
        value = str((meta.get(sid, {}) or {}).get(key) or default or "unknown")
        buckets[value].append(ep)
    out: dict[str, Any] = {}
    for name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        passed = sum(1 for r in rows if r.get("success"))
        terminal_failures = sum(1 for r in rows if _episode_terminal_failure(r))
        out[name] = {
            "episodes": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "success_rate": _rate(passed, len(rows)),
            "terminal_failures": terminal_failures,
            "terminal_failure_rate": _rate(terminal_failures, len(rows)),
            "avg_attempts": round(mean([len(r.get("attempts") or []) for r in rows]), 6) if rows else 0.0,
        }
    return out


def audit_rollout(rollout_dir: str | Path | None = None,
                  grpo_samples_path: str | Path | None = None,
                  rollouts_path: str | Path | None = None,
                  processed_states: str | Path | None = None) -> dict[str, Any]:
    if rollout_dir:
        rd = Path(rollout_dir).expanduser()
        grpo_samples_path = grpo_samples_path or rd / "grpo_samples.jsonl"
        rollouts_path = rollouts_path or rd / "rollouts.jsonl"

    samples_path = Path(grpo_samples_path).expanduser() if grpo_samples_path else None
    episodes_path = Path(rollouts_path).expanduser() if rollouts_path else None
    samples = _read_jsonl(samples_path)
    episodes = _read_jsonl(episodes_path)
    meta = _metadata_from_processed_states(processed_states)

    comps = [_sample_components(s) for s in samples]
    rewards = [_safe_float(s.get("reward")) for s in samples]
    advantages = [_safe_float(s.get("advantage")) for s in samples if s.get("advantage") is not None]
    group_summary, _ = _group_stats(samples)

    scenario_ids = {str(s.get("scenario_id")) for s in samples if s.get("scenario_id")}
    episode_ids = {str(e.get("scenario_id")) for e in episodes if e.get("scenario_id")}

    sample_success = sum(1 for s in samples if s.get("success"))
    episode_success = sum(1 for e in episodes if e.get("success"))
    terminal_episodes = sum(1 for e in episodes if _episode_terminal_failure(e))

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
            "num_episode_scenarios": len(episode_ids),
            "passed": episode_success,
            "failed": len(episodes) - episode_success,
            "success_rate": _rate(episode_success, len(episodes)),
            "terminal_failures": terminal_episodes,
            "terminal_failure_rate": _rate(terminal_episodes, len(episodes)),
            "avg_attempts": round(mean([len(e.get("attempts") or []) for e in episodes]), 6) if episodes else 0.0,
        },
        "samples": {
            "num_samples": len(samples),
            "num_sample_scenarios": len(scenario_ids),
            "success_samples": sample_success,
            "success_sample_rate": _rate(sample_success, len(samples)),
            "terminal_samples": sum(1 for s in samples if s.get("terminal")),
            "missing_old_logprob_sum": sum(1 for s in samples if s.get("old_logprob_sum") is None),
            "reward": _stats(rewards),
            "advantage": _stats(advantages),
        },
        "groups": group_summary,
        "reward_components": {
            "pair_score": _stats([c.get("pair_score") for c in comps]),
            "twin_reproduction_score": _stats([c.get("twin_reproduction_score") for c in comps]),
            "count_mismatch_rate": _rate(sum(1 for c in comps if _safe_float(c.get("count_mismatch")) > 0), len(comps)),
            "invalid_format_rate": _rate(sum(1 for c in comps if c.get("invalid_format")), len(comps)),
            "repeated_guess_rate": _rate(sum(1 for c in comps if c.get("repeated_wrong_guess")), len(comps)),
            "exact_set_match_rate": _rate(sum(1 for c in comps if c.get("exact_set_match")), len(comps)),
            "soft_success_rate": _rate(sum(1 for c in comps if c.get("soft_success")), len(comps)),
            "avg_instruction_tokens": round(mean([_safe_float(c.get("instruction_tokens")) for c in comps]), 6) if comps else 0.0,
        },
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
        summary["warnings"].append("No GRPO samples found; run train_rca_grpo before auditing rewards.")
    if not episodes:
        summary["warnings"].append("No episode rollouts found; run train_rca_grpo before auditing episodes.")
    if samples and group_summary["groups_with_nonzero_reward_std"] == 0:
        summary["warnings"].append(
            "All GRPO groups have zero within-group reward variance; there is no group-relative learning signal yet."
        )
    if samples and summary["samples"]["missing_old_logprob_sum"] == len(samples):
        summary["warnings"].append(
            "All samples are missing old_logprob_sum; this is expected for heuristic/debug policies but not for real GRPO training."
        )
    if summary["reward_components"]["invalid_format_rate"] > 0.05:
        summary["warnings"].append("High invalid-format rate; solver output schema may need stronger prompting or parsing.")
    if summary["reward_components"]["count_mismatch_rate"] > 0.25:
        summary["warnings"].append("High count-mismatch rate; multifault/root-cause-count handling may be weak.")
    if samples and summary["reward_components"]["twin_reproduction_score"]["std"] == 0.0:
        summary["warnings"].append("Twin reproduction score has zero variance; twin reward may be too coarse for learning signal.")
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

    print("\nTop sample buckets by family:")
    for name, row in list(summary["by_family_samples"].items())[:20]:
        print(
            f"  {name:55s} samples={int(row.get('samples', 0)):5d} "
            f"succ={_safe_float(row.get('success_sample_rate')):.3f} "
            f"reward_mean={_safe_float((row.get('reward') or {}).get('mean')):.3f} "
            f"pair={_safe_float(row.get('pair_score_mean')):.3f} "
            f"twin={_safe_float(row.get('twin_reproduction_score_mean')):.3f}"
        )

    print("\nEpisode pass rate by family:")
    for name, row in list(summary["by_family_episodes"].items())[:20]:
        print(
            f"  {name:55s} episodes={int(row.get('episodes', 0)):4d} "
            f"pass={_safe_float(row.get('success_rate')):.3f} "
            f"terminal={_safe_float(row.get('terminal_failure_rate')):.3f} "
            f"avg_attempts={_safe_float(row.get('avg_attempts')):.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit RCA reward and GRPO rollout quality.")
    ap.add_argument("--rollout_dir", default=None, help="Directory containing rollouts.jsonl and grpo_samples.jsonl.")
    ap.add_argument("--grpo_samples", default=None, help="Optional explicit grpo_samples.jsonl path.")
    ap.add_argument("--rollouts", default=None, help="Optional explicit rollouts.jsonl path.")
    ap.add_argument("--processed_states", default=None, help="Optional processed_states directory for exact family/task metadata.")
    ap.add_argument("--output", default=None, help="Optional path to write full audit JSON.")
    ap.add_argument("--quiet", action="store_true", help="Only write --output, do not print human summary.")
    args = ap.parse_args()

    summary = audit_rollout(
        rollout_dir=args.rollout_dir,
        grpo_samples_path=args.grpo_samples,
        rollouts_path=args.rollouts,
        processed_states=args.processed_states,
    )
    _write_json(summary, args.output)
    if not args.quiet:
        _print_human(summary)


if __name__ == "__main__":
    main()
