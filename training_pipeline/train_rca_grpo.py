from __future__ import annotations

import argparse, json
from .data_loader import iter_scenarios
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver, run_rca_self_prompting_loop
from .rollout_logger import RolloutLogger
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 RCA rollout generation / smoke-test")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use_behavioral_twin", action="store_true")
    args = ap.parse_args()
    logger = RolloutLogger(args.output_dir)
    policy = HeuristicRCAInstructionPolicy(); solver = HeuristicRCASolver()
    twin = BehavioralTwinVerifier() if args.use_behavioral_twin else None
    total = passed = 0
    for rec in iter_scenarios(args.processed_states, limit=args.limit):
        total += 1
        result = run_rca_self_prompting_loop(rec.full_state, rec.compressed_state, policy, solver, twin_validator=twin)
        passed += int(result["success"])
        logger.log({"stage": "rca", **result})
        print(f"[RCA] {total} {rec.scenario_id} success={result['success']}")
    summary = {"stage": "rca", "total": total, "passed": passed, "failed": total - passed}
    logger.write_summary(summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
