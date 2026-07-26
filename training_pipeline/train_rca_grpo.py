from __future__ import annotations

import argparse, json
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver, run_rca_self_prompting_loop
from .rollout_logger import RolloutLogger
from .split_utils import read_scenario_ids
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 RCA rollout generation / smoke-test")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Maximum selected scenarios to run after filtering.")
    ap.add_argument("--scenario_ids", default=None, help="Optional file with one allowed scenario_id per line.")
    ap.add_argument("--include_unlabeled", action="store_true", help="Include scenarios without oracle labels. Not recommended for reward training.")
    ap.add_argument("--use_behavioral_twin", action="store_true")
    args = ap.parse_args()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    logger = RolloutLogger(args.output_dir)
    policy = HeuristicRCAInstructionPolicy(); solver = HeuristicRCASolver()
    twin = BehavioralTwinVerifier() if args.use_behavioral_twin else None
    total = passed = skipped_unlabeled = skipped_filter = 0

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
        result = run_rca_self_prompting_loop(rec.full_state, rec.compressed_state, policy, solver, twin_validator=twin)
        passed += int(result["success"])
        logger.log({"stage": "rca", **result})
        print(f"[RCA] {total} {rec.scenario_id} success={result['success']}")

    summary = {
        "stage": "rca",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_filter": skipped_filter,
        "scenario_ids_file": args.scenario_ids,
    }
    logger.write_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
