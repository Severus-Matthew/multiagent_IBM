from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_preflight import preflight_behavioral_twin, require_twin_preflight_ok
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .action_loop import run_action_prompt_optimizer_loop
from .action_prompt_policy import CANONICAL_ACTION_STRATEGIES, StructuredActionPromptPolicy
from .data_loader import iter_scenarios
from .fixed_action_agent import FixedActionAgent
from .ground_truth import labels_from_full_state
from .rollout_logger import RolloutLogger
from .schemas import FaultLabel, parse_fault_lines
from .split_utils import read_scenario_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 action-loop rollout generation")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Maximum selected labeled scenarios to run after filtering.")
    ap.add_argument("--scenario_ids", default=None, help="Optional file with one allowed scenario_id per line.")
    ap.add_argument("--rca_rollout_dir", default=None,
                    help="Optional RCA rollout directory. When set, action uses final RCA predictions from rollouts.jsonl instead of oracle labels.")
    ap.add_argument("--twin_mode", choices=["behavioral"], default="behavioral")
    ap.add_argument("--twin_preflight", action="store_true", help="Run/log behavioral twin preflight before each action episode.")
    ap.add_argument("--abort_on_twin_preflight_failure", action="store_true")
    ap.add_argument("--require_rca_twin_verification", action="store_true",
                    help="Skip action unless the upstream RCA result passes the RCA twin gate.")
    ap.add_argument("--skip_action_if_rca_unverified", action="store_true", default=True,
                    help="When RCA twin verification is required, do not generate commands if RCA is unverified.")
    ap.add_argument("--allow_action_if_rca_unverified", action="store_true",
                    help="Override the default skip behavior and still run action even if RCA is unverified.")
    ap.add_argument("--min_twin_reproduction_score", type=float, default=0.0)
    ap.add_argument("--max_iterations", type=int, default=5)

    # Active action policy/agent controls.
    ap.add_argument("--action_prompt_policy", choices=["structured"], default="structured",
                    help="Prompt optimizer used before the fixed ActionAgent. Current active option: structured.")
    ap.add_argument("--action_strategy", choices=CANONICAL_ACTION_STRATEGIES, default="auto",
                    help="Structured action strategy family.")
    ap.add_argument("--action_agent", choices=["fixed", "llm"], default="fixed",
                    help="fixed is deterministic/safe for offline rollouts; llm uses agents.action_agent.ActionAgent.")
    ap.add_argument("--llm_provider", choices=["claude", "openai"], default="claude")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--max_commands", type=int, default=15)
    args = ap.parse_args()

    if args.allow_action_if_rca_unverified:
        args.skip_action_if_rca_unverified = False

    allowed_ids = read_scenario_ids(args.scenario_ids)
    rca_rows = _load_rca_rollouts(args.rca_rollout_dir) if args.rca_rollout_dir else {}
    logger = RolloutLogger(args.output_dir)
    twin = BehavioralTwinVerifier()
    preflight_path = Path(args.output_dir).expanduser() / "twin_preflight.jsonl"
    prompt_policy = _build_action_prompt_policy(args)
    action_agent = _build_action_agent(args)

    total = passed = skipped_unlabeled = skipped_filter = skipped_missing_rca = skipped_action_gate = 0
    twin_preflight_count = twin_preflight_failed = 0
    action_attempt_count = sla_restored_count = target_sla_restored_count = 0
    uses_oracle_rca = not bool(args.rca_rollout_dir)

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            skipped_filter += 1
            continue
        gt = labels_from_full_state(rec.full_state)
        if not gt:
            skipped_unlabeled += 1
            continue
        if args.limit is not None and total >= args.limit:
            break

        if args.twin_preflight or args.require_rca_twin_verification:
            preflight = preflight_behavioral_twin(rec.full_state, rec.compressed_state)
            twin_preflight_count += 1
            twin_preflight_failed += int(not preflight.get("ok", False))
            _append_jsonl(preflight_path, {"stage": "action_twin_preflight", **preflight})
            if args.abort_on_twin_preflight_failure:
                require_twin_preflight_ok(preflight)

        if args.rca_rollout_dir:
            upstream = rca_rows.get(rec.scenario_id)
            if not upstream:
                skipped_missing_rca += 1
                continue
            rca_result, rca_faults, rca_gate = _rca_context_from_rollout(upstream)
        else:
            rca_faults = gt
            rca_result = _rca_result_from_faults(gt, source="oracle_debug")
            rca_gate = None

        total += 1
        result = run_action_prompt_optimizer_loop(
            rec.full_state,
            rec.compressed_state,
            rca_result,
            rca_faults,
            prompt_policy,
            action_agent,
            twin,
            max_iterations=args.max_iterations,
            require_rca_twin_verification=args.require_rca_twin_verification,
            skip_action_if_rca_unverified=args.skip_action_if_rca_unverified,
            min_twin_reproduction_score=args.min_twin_reproduction_score,
            rca_twin_gate=rca_gate,
        )
        result["action_policy_metadata"] = {
            "action_prompt_policy": args.action_prompt_policy,
            "action_strategy": args.action_strategy,
            "action_agent": args.action_agent,
            "llm_provider": args.llm_provider if args.action_agent == "llm" else None,
            "llm_model": args.llm_model if args.action_agent == "llm" else None,
            "max_commands": args.max_commands,
        }
        skipped_action_gate += int(bool(result.get("skipped_action")))
        attempts = result.get("attempts", []) or []
        action_attempt_count += len(attempts)
        if attempts:
            final_verifier = attempts[-1].get("verifier_result", {}) or {}
            sla_restored_count += int(bool(final_verifier.get("sla_restored")))
            target_sla_restored_count += int(bool(final_verifier.get("target_sla_restored")))
        passed += int(result["success"])
        logger.log({"stage": "action", **result})
        print(
            f"[ACTION] {total} {rec.scenario_id} "
            f"success={result['success']} skipped_action={bool(result.get('skipped_action'))} attempts={len(attempts)}"
        )

    summary = {
        "stage": "action",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / max(total, 1),
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_filter": skipped_filter,
        "skipped_missing_rca": skipped_missing_rca,
        "skipped_action_gate": skipped_action_gate,
        "action_attempt_count": action_attempt_count,
        "sla_restored_count": sla_restored_count,
        "target_sla_restored_count": target_sla_restored_count,
        "scenario_ids_file": args.scenario_ids,
        "rca_rollout_dir": args.rca_rollout_dir,
        "uses_oracle_rca": uses_oracle_rca,
        "twin_mode": args.twin_mode,
        "twin_preflight": args.twin_preflight,
        "twin_preflight_jsonl": str(preflight_path) if (args.twin_preflight or args.require_rca_twin_verification) else None,
        "twin_preflight_count": twin_preflight_count,
        "twin_preflight_failed": twin_preflight_failed,
        "require_rca_twin_verification": args.require_rca_twin_verification,
        "skip_action_if_rca_unverified": args.skip_action_if_rca_unverified,
        "min_twin_reproduction_score": args.min_twin_reproduction_score,
        "max_iterations": args.max_iterations,
        "action_prompt_policy": args.action_prompt_policy,
        "action_strategy": args.action_strategy,
        "action_agent": args.action_agent,
        "max_commands": args.max_commands,
        "uses_real_llm_action_agent": args.action_agent == "llm",
        "uses_real_training_update": False,
    }
    logger.write_summary(summary)
    print(json.dumps(summary, indent=2))


def _build_action_prompt_policy(args):
    if args.action_prompt_policy == "structured":
        return StructuredActionPromptPolicy(strategy=args.action_strategy)
    raise ValueError(f"unknown action prompt policy {args.action_prompt_policy!r}")


def _build_action_agent(args):
    if args.action_agent == "fixed":
        return FixedActionAgent(max_commands=args.max_commands)
    if args.action_agent == "llm":
        # Optional live LLM path. Import lazily so offline smoke tests do not need
        # API packages or API keys.
        from agents.action_agent import ActionAgent
        from agents.llm_client import LLMClient

        return ActionAgent(client=LLMClient(provider=args.llm_provider, model=args.llm_model), max_commands=args.max_commands)
    raise ValueError(f"unknown action agent {args.action_agent!r}")


def _load_rca_rollouts(rca_rollout_dir: str | None) -> dict[str, dict[str, Any]]:
    if not rca_rollout_dir:
        return {}
    path = Path(rca_rollout_dir).expanduser() / "rollouts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"RCA rollouts.jsonl not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != "rca":
                continue
            sid = str(row.get("scenario_id") or "")
            if sid:
                rows[sid] = row
    return rows


def _rca_context_from_rollout(row: dict[str, Any]) -> tuple[dict[str, Any], list[FaultLabel], dict[str, Any] | None]:
    faults = parse_fault_lines(row.get("final_prediction", ""))
    if not faults:
        attempts = row.get("attempts", []) or []
        if attempts:
            faults = parse_fault_lines(attempts[-1].get("prediction_text", ""))
    result = _rca_result_from_faults(faults, source="upstream_rca_rollout")
    result.update({
        "upstream_success": bool(row.get("success")),
        "upstream_success_before_twin_gate": bool(row.get("success_before_twin_gate", False)),
        "final_prediction": row.get("final_prediction", ""),
        "rca_twin_gate": row.get("rca_twin_gate"),
    })
    return result, faults, row.get("rca_twin_gate")


def _rca_result_from_faults(faults: list[FaultLabel], source: str) -> dict[str, Any]:
    first = faults[0] if faults else FaultLabel(service="unknown", fault_type="unknown")
    return {
        "root_cause_service": first.service,
        "fault_type": first.fault_type,
        "affected_services": [],
        "num_root_causes": len(faults),
        "root_causes": [f.to_dict() for f in faults],
        "source": source,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
