from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .prompt_policy_factory import (
    build_rca_instruction_policy,
    build_rca_solver,
    default_policy_model_name,
    policy_metadata,
)
from .rca_loop import run_rca_grpo_episode
from .rollout_logger import RolloutLogger
from .schemas import parse_fault_lines
from .split_utils import read_scenario_ids
from .wandb_logger import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, WandbRunLogger, parse_tags
from digital_twin_runtime.twin_preflight import (
    preflight_behavioral_twin,
    rca_twin_gate,
    require_twin_preflight_ok,
)
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 RCA GRPO-ready rollout generation")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Maximum selected scenarios to run after filtering.")
    ap.add_argument("--scenario_ids", default=None, help="Optional file with one allowed scenario_id per line.")
    ap.add_argument("--include_unlabeled", action="store_true", help="Include scenarios without oracle labels. Not recommended for reward training.")
    ap.add_argument("--agent_input_mode", choices=["legacy", "training_safe"], default="legacy",
                    help="training_safe removes oracle/candidate-root-cause menus before the policy/solver sees the state.")
    ap.add_argument("--allow_candidate_sweep_audit", action="store_true",
                    help="Required when --llm_model candidate_sweep is used. Candidate sweep is audit-only, not training.")
    ap.add_argument("--use_behavioral_twin", action="store_true")
    ap.add_argument("--twin_mode", choices=["none", "behavioral"], default=None,
                    help="Twin verifier mode. Defaults to behavioral when --use_behavioral_twin is set, otherwise none. Live mode will be added after the K8s twin builder is wired.")
    ap.add_argument("--twin_preflight", action="store_true",
                    help="Run and log a twin preflight before each selected RCA episode.")
    ap.add_argument("--abort_on_twin_preflight_failure", action="store_true",
                    help="Stop the rollout immediately if twin preflight fails.")
    ap.add_argument("--require_rca_twin_verification", action="store_true",
                    help="Compute the RCA twin gate for the final attempt. Blocking behavior is controlled by --rca_twin_gate_mode.")
    ap.add_argument("--rca_twin_gate_mode", choices=["diagnostic", "strict"], default="diagnostic",
                    help="diagnostic logs twin reproduction without changing RCA success; strict blocks label successes that fail the twin gate.")
    ap.add_argument("--min_twin_reproduction_score", type=float, default=0.0,
                    help="Minimum RCA twin reproduction score for the gate.")
    ap.add_argument("--max_iterations", type=int, default=5)
    ap.add_argument("--group_size", type=int, default=4, help="Number of instruction candidates per state/history group.")
    ap.add_argument("--selection_strategy", choices=["best", "sample0"], default="best",
                    help="Use best for verifier-guided data generation; use sample0 for stricter on-policy debugging/training smoke tests.")

    ap.add_argument("--instruction_policy", choices=["heuristic", "operator", "qwen_stub"], default="heuristic",
                    help="Control policy family that emits instructions for the fixed RCA solver.")
    ap.add_argument("--operator_profile", choices=["auto", "system_first", "trace_first", "log_first", "multifault_first"], default="auto",
                    help="Structured prompt-operator profile used when --instruction_policy operator.")
    ap.add_argument("--operator_max_focus_services", type=int, default=6)

    ap.add_argument("--qwen_model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--qwen_adapter_path", default=None)
    ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
    ap.add_argument("--qwen_temperature", type=float, default=0.7)
    ap.add_argument("--qwen_top_p", type=float, default=0.9)

    ap.add_argument("--rca_solver", choices=["heuristic", "llm", "safe_llm"], default="heuristic",
                    help="Fixed RCA solver consumed by the prompt policy. safe_llm hides candidate/root-cause menus from the model.")
    ap.add_argument("--llm_provider", choices=["openai", "claude"], default="openai",
                    help="Provider for --rca_solver llm/safe_llm. API key is read by agents.llm_client from environment variables.")
    ap.add_argument("--llm_model", default=None,
                    help="Provider model name, or candidate_sweep for audit-only reachability mode.")
    ap.add_argument("--llm_max_tokens", type=int, default=300)
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_state_char_budget", type=int, default=24000)
    ap.add_argument("--llm_max_root_causes", type=int, default=1,
                    help="Maximum root-cause lines returned by the LLM RCA postprocessor. Use 1 for single-fault smoke; use 2+ for multifault experiments.")
    ap.add_argument("--llm_cache_path", default=None,
                    help="Optional JSONL cache for LLM RCA calls, useful for repeated smoke tests.")

    ap.add_argument("--policy_model_name", default=None)
    ap.add_argument("--policy_version", default="v0")

    ap.add_argument("--wandb", action="store_true", help="Enable W&B scalar logging and artifact upload.")
    ap.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--wandb_entity", default=DEFAULT_WANDB_ENTITY)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_tags", default="")
    args = ap.parse_args()

    twin_mode = args.twin_mode or ("behavioral" if args.use_behavioral_twin else "none")
    if args.require_rca_twin_verification and twin_mode == "none":
        raise ValueError("--require_rca_twin_verification requires --use_behavioral_twin or --twin_mode behavioral")
    if twin_mode == "behavioral":
        args.use_behavioral_twin = True

    is_candidate_sweep = args.rca_solver == "llm" and str(args.llm_model or "").lower() in {"candidate_sweep", "candidate-sweep", "sweep"}
    if is_candidate_sweep and not args.allow_candidate_sweep_audit:
        raise ValueError("candidate_sweep is audit-only. Re-run with --allow_candidate_sweep_audit, and do not report it as training/final inference.")
    if is_candidate_sweep and args.agent_input_mode == "training_safe":
        raise ValueError("candidate_sweep cannot be combined with --agent_input_mode training_safe; it intentionally uses a candidate menu.")
    if args.rca_solver == "safe_llm" and args.agent_input_mode != "training_safe":
        args.agent_input_mode = "training_safe"

    allowed_ids = read_scenario_ids(args.scenario_ids)
    logger = RolloutLogger(args.output_dir)
    preflight_path = Path(args.output_dir).expanduser() / "twin_preflight.jsonl"
    agent_safety_path = Path(args.output_dir).expanduser() / "agent_input_safety.jsonl"
    if args.rca_solver in {"llm", "safe_llm"} and not args.llm_cache_path and not is_candidate_sweep:
        args.llm_cache_path = str(Path(args.output_dir).expanduser() / "llm_rca_cache.jsonl")
    policy = build_rca_instruction_policy(args)
    solver = build_rca_solver(args)
    twin = BehavioralTwinVerifier() if twin_mode == "behavioral" else None
    total = passed = skipped_unlabeled = skipped_filter = sample_count = 0
    twin_preflight_count = twin_preflight_failed = twin_verified_successes = twin_blocked_successes = 0
    twin_diagnostic_label_success_but_twin_fail = 0
    same_error_pattern_verified_count = 0
    agent_input_safety_failed = 0

    policy_model_name = args.policy_model_name or default_policy_model_name(args)
    run_config = {
        "stage": "rca",
        "processed_states": args.processed_states,
        "output_dir": args.output_dir,
        "scenario_ids_file": args.scenario_ids,
        "include_unlabeled": args.include_unlabeled,
        "agent_input_mode": args.agent_input_mode,
        "allow_candidate_sweep_audit": args.allow_candidate_sweep_audit,
        "candidate_sweep_audit_only": is_candidate_sweep,
        "use_behavioral_twin": args.use_behavioral_twin,
        "twin_mode": twin_mode,
        "twin_preflight": args.twin_preflight,
        "abort_on_twin_preflight_failure": args.abort_on_twin_preflight_failure,
        "require_rca_twin_verification": args.require_rca_twin_verification,
        "rca_twin_gate_mode": args.rca_twin_gate_mode,
        "min_twin_reproduction_score": args.min_twin_reproduction_score,
        "max_iterations": args.max_iterations,
        "group_size": args.group_size,
        "selection_strategy": args.selection_strategy,
        "policy_model_name": policy_model_name,
        "policy_version": args.policy_version,
        **policy_metadata(args),
    }
    wandb_logger = WandbRunLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        config=run_config,
        tags=parse_tags(args.wandb_tags),
    )
    wandb_logger.start()

    try:
        for rec in iter_scenarios(args.processed_states):
            if allowed_ids is not None and rec.scenario_id not in allowed_ids:
                skipped_filter += 1
                continue
            if not args.include_unlabeled and not labels_from_full_state(rec.full_state):
                skipped_unlabeled += 1
                continue
            if args.limit is not None and total >= args.limit:
                break

            total += 1
            twin_preflight = None
            if args.twin_preflight or args.require_rca_twin_verification:
                if twin_mode != "behavioral":
                    raise ValueError("twin preflight currently supports --twin_mode behavioral only")
                twin_preflight = preflight_behavioral_twin(rec.full_state, rec.compressed_state)
                twin_preflight_count += 1
                twin_preflight_failed += int(not twin_preflight.get("ok", False))
                _append_jsonl(preflight_path, {"stage": "twin_preflight", **twin_preflight})
                if args.abort_on_twin_preflight_failure:
                    require_twin_preflight_ok(twin_preflight)

            if args.agent_input_mode == "training_safe":
                agent_state = sanitize_agent_state(rec.compressed_state, mode="training_safe")
                agent_safety = agent_input_safety_report(agent_state)
                agent_input_safety_failed += int(not agent_safety.get("safe_for_training_agent", False))
                _append_jsonl(agent_safety_path, {"scenario_id": rec.scenario_id, **agent_safety})
            else:
                agent_state = rec.compressed_state
                agent_safety = {"safe_for_training_agent": None, "sanitizer_version": "legacy_unsanitized"}

            result = run_rca_grpo_episode(
                rec.full_state,
                rec.compressed_state,
                policy,
                solver,
                twin_validator=twin,
                max_iterations=args.max_iterations,
                group_size=args.group_size,
                selection_strategy=args.selection_strategy,
                policy_model_name=policy_model_name,
                policy_version=args.policy_version,
                agent_state=agent_state,
                agent_input_mode=args.agent_input_mode,
                agent_input_safety=agent_safety,
            )
            if twin_preflight is not None:
                result["twin_preflight"] = twin_preflight

            if args.require_rca_twin_verification:
                gate = _final_rca_twin_gate(rec.full_state, rec.compressed_state, twin, result, args.min_twin_reproduction_score)
                result["rca_twin_gate"] = gate
                result["rca_twin_gate_mode"] = args.rca_twin_gate_mode
                same_error_pattern_verified_count += int(bool(gate.get("same_error_pattern_verified")))
                if result.get("success") and gate.get("rca_twin_verified"):
                    twin_verified_successes += 1
                elif result.get("success") and not gate.get("rca_twin_verified"):
                    if args.rca_twin_gate_mode == "strict":
                        result["success_before_twin_gate"] = True
                        result["success"] = False
                        twin_blocked_successes += 1
                    else:
                        result["twin_gate_diagnostic_only"] = True
                        twin_diagnostic_label_success_but_twin_fail += 1

            samples = result.pop("grpo_samples", [])
            for sample in samples:
                logger.log_grpo_sample(sample)
            sample_count += len(samples)
            passed += int(result["success"])
            logger.log({"stage": "rca", **result})
            wandb_logger.log_episode(total, result, samples, passed_so_far=passed, total_so_far=total)
            print(f"[RCA] {total} {rec.scenario_id} success={result['success']} samples={len(samples)}")

        summary = {
            "stage": "rca",
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / max(total, 1),
            "skipped_unlabeled": skipped_unlabeled,
            "skipped_filter": skipped_filter,
            "scenario_ids_file": args.scenario_ids,
            "group_size": args.group_size,
            "max_iterations": args.max_iterations,
            "selection_strategy": args.selection_strategy,
            "grpo_samples": sample_count,
            "rollouts_jsonl": str(logger.jsonl_path),
            "grpo_samples_jsonl": str(logger.grpo_jsonl_path),
            "agent_input_mode": args.agent_input_mode,
            "agent_input_safety_jsonl": str(agent_safety_path) if args.agent_input_mode == "training_safe" else None,
            "agent_input_safety_failed": agent_input_safety_failed,
            "candidate_sweep_audit_only": is_candidate_sweep,
            "twin_preflight_jsonl": str(preflight_path) if (args.twin_preflight or args.require_rca_twin_verification) else None,
            "twin_mode": twin_mode,
            "twin_preflight_count": twin_preflight_count,
            "twin_preflight_failed": twin_preflight_failed,
            "require_rca_twin_verification": args.require_rca_twin_verification,
            "rca_twin_gate_mode": args.rca_twin_gate_mode,
            "min_twin_reproduction_score": args.min_twin_reproduction_score,
            "same_error_pattern_verified_count": same_error_pattern_verified_count,
            "twin_verified_successes": twin_verified_successes,
            "twin_blocked_successes": twin_blocked_successes,
            "twin_diagnostic_label_success_but_twin_fail": twin_diagnostic_label_success_but_twin_fail,
            "policy_model_name": policy_model_name,
            "policy_version": args.policy_version,
            "instruction_policy": args.instruction_policy,
            "rca_solver": args.rca_solver,
            "policy_metadata": policy_metadata(args),
            "uses_real_llm": args.rca_solver in {"llm", "safe_llm"} and not is_candidate_sweep,
            "uses_real_training_update": False,
            "wandb_enabled": bool(args.wandb),
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        }
        logger.write_summary(summary)
        wandb_logger.log_summary(summary, args.output_dir)
        print(json.dumps(summary, indent=2))
    finally:
        wandb_logger.finish()


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _final_rca_twin_gate(full_state: dict, compressed_state: dict, twin, result: dict, min_score: float) -> dict:
    attempts = result.get("attempts", []) or []
    if not attempts:
        return rca_twin_gate(None, min_score, rca_success=False)
    final = attempts[-1]
    rca_success = bool(final.get("success"))
    faults = parse_fault_lines(final.get("prediction_text", ""))
    twin_result = None
    if twin is not None and faults:
        twin_result = twin.validate_rca_prediction(full_state, compressed_state, faults)
    gate = rca_twin_gate(twin_result, min_score, rca_success=rca_success)
    gate["final_attempt_success"] = rca_success
    gate["final_prediction"] = final.get("prediction_text", "")
    return gate


if __name__ == "__main__":
    main()
