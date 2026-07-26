from __future__ import annotations

import argparse
import json
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver, run_rca_grpo_episode
from .rollout_logger import RolloutLogger
from .split_utils import read_scenario_ids
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 RCA GRPO-ready rollout generation")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Maximum selected scenarios to run after filtering.")
    ap.add_argument("--scenario_ids", default=None, help="Optional file with one allowed scenario_id per line.")
    ap.add_argument("--include_unlabeled", action="store_true", help="Include scenarios without oracle labels. Not recommended for reward training.")
    ap.add_argument("--use_behavioral_twin", action="store_true")
    ap.add_argument("--max_iterations", type=int, default=5)
    ap.add_argument("--group_size", type=int, default=4, help="Number of instruction candidates per state/history group.")
    ap.add_argument("--selection_strategy", choices=["best", "sample0"], default="best",
                    help="Which candidate advances episode history. Use sample0 for stricter on-policy debugging; best for offline verifier-guided data generation.")
    ap.add_argument("--policy_model_name", default="debug-heuristic-policy")
    ap.add_argument("--policy_version", default="v0")
    args = ap.parse_args()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    logger = RolloutLogger(args.output_dir)
    policy = HeuristicRCAInstructionPolicy()
    solver = HeuristicRCASolver()
    twin = BehavioralTwinVerifier() if args.use_behavioral_twin else None
    total = passed = skipped_unlabeled = skipped_filter = sample_count = 0

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
        result = run_rca_grpo_episode(
            rec.full_state,
            rec.compressed_state,
            policy,
            solver,
            twin_validator=twin,
            max_iterations=args.max_iterations,
            group_size=args.group_size,
            selection_strategy=args.selection_strategy,
            policy_model_name=args.policy_model_name,
            policy_version=args.policy_version,
        )
        samples = result.pop("grpo_samples", [])
        for sample in samples:
            logger.log_grpo_sample(sample)
        sample_count += len(samples)
        passed += int(result["success"])
        logger.log({"stage": "rca", **result})
        print(f"[RCA] {total} {rec.scenario_id} success={result['success']} samples={len(samples)}")

    summary = {
        "stage": "rca",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_filter": skipped_filter,
        "scenario_ids_file": args.scenario_ids,
        "group_size": args.group_size,
        "max_iterations": args.max_iterations,
        "selection_strategy": args.selection_strategy,
        "grpo_samples": sample_count,
        "rollouts_jsonl": str(logger.jsonl_path),
        "grpo_samples_jsonl": str(logger.grpo_jsonl_path),
        "policy_model_name": args.policy_model_name,
        "policy_version": args.policy_version,
        "uses_real_llm": False,
        "uses_real_training_update": False,
    }
    logger.write_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
