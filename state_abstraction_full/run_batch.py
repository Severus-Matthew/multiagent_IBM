"""
Batch runner: processes every completed scenario from gen_and_telmetry.py output
through the full state-abstraction pipeline.

Usage:
    python run_batch.py \
        --telemetry_dir /path/to/dynamic_generated_scenario_results_all_new/telemetry_outputs \
        [--output_base /path/to/processed_states]   # default: <telemetry_dir>/../processed_states
        [--workers 4]
        [--skip_simulator]
        [--skip_compress]
        [--limit N]            # process at most N scenarios (for testing)
        [--resume]             # skip scenarios whose processed_state/state_summary.json already exists
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from build_state import build_and_write
from build_simulation_specs import build_specs
from run_rca_simulation_check import run_check
from sla import evaluate_states_file
from compress import compress_state
from utils import write_json, read_json, ensure_dir


def _process_one(args):
    """Worker function — runs the full pipeline for a single scenario dir."""
    run_dir, output_dir, skip_simulator, skip_compress = args
    run_dir   = Path(run_dir)
    output_dir = Path(output_dir)

    result = {
        "scenario_id": run_dir.name,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "ok": False,
        "error": None,
    }

    try:
        ensure_dir(output_dir)

        # 1. Build state
        state = build_and_write(run_dir, output_dir)

        # 2. SLA
        evaluate_states_file(output_dir / "states.jsonl", output_dir / "sla_results.json")

        # 3. Simulator specs + optional simulation
        build_specs(output_dir)

        if not skip_simulator:
            rca       = state.get("rca_features", {})
            hyp       = rca.get("hypothesis", {})
            fault_ctx = state.get("fault_context", {})
            service    = hyp.get("service")    or fault_ctx.get("faulty_service") or "unknown"
            fault_type = hyp.get("fault_type") or "dependency_failure"
            run_check(output_dir, service=service, fault_type=fault_type, action="rollback_config")

        # 4. Compress
        if not skip_compress:
            compressed = compress_state(state)
            write_json(compressed, output_dir / "state_abstraction_compressed.json")

        result["ok"] = True
        result["most_suspicious_service"] = (
            state.get("rca_features", {}).get("most_suspicious_service")
        )
        result["sla_violated"] = state.get("sla", {}).get("violated", False)

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry_dir", required=True,
                    help="Path to telemetry_outputs/ from gen_and_telmetry.py")
    ap.add_argument("--output_base", default=None,
                    help="Root for processed state dirs. Default: <telemetry_dir>/../processed_states")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (set >1 only if cluster can handle concurrent kubectl)")
    ap.add_argument("--skip_simulator", action="store_true")
    ap.add_argument("--skip_compress",  action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max scenarios to process (useful for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip scenarios that already have state_summary.json")
    args = ap.parse_args()

    telemetry_dir = Path(args.telemetry_dir)
    output_base   = Path(args.output_base) if args.output_base else telemetry_dir.parent / "processed_states"
    ensure_dir(output_base)

    # Discover scenario dirs — only those with a DONE.json (completed scenarios)
    scenario_dirs = sorted(
        d for d in telemetry_dir.iterdir()
        if d.is_dir() and (d / "DONE.json").exists()
    )

    if args.resume:
        before = len(scenario_dirs)
        scenario_dirs = [
            d for d in scenario_dirs
            if not (output_base / d.name / "state_summary.json").exists()
        ]
        print(f"[RESUME] Skipped {before - len(scenario_dirs)} already-processed scenarios.")

    if args.limit:
        scenario_dirs = scenario_dirs[:args.limit]

    total = len(scenario_dirs)
    print(f"[BATCH] Processing {total} scenarios with {args.workers} worker(s).")
    print(f"        Telemetry dir : {telemetry_dir}")
    print(f"        Output base   : {output_base}")

    tasks = [
        (str(d), str(output_base / d.name), args.skip_simulator, args.skip_compress)
        for d in scenario_dirs
    ]

    passed = 0
    failed = 0
    batch_log = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                _log_result(res, i, total)
                batch_log.append(res)
                if res["ok"]:
                    passed += 1
                else:
                    failed += 1
    else:
        for i, task in enumerate(tasks, 1):
            res = _process_one(task)
            _log_result(res, i, total)
            batch_log.append(res)
            if res["ok"]:
                passed += 1
            else:
                failed += 1

    # Write batch summary
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": batch_log,
    }
    log_path = output_base / "batch_run_log.json"
    write_json(summary, log_path)

    print(f"\n[DONE] {passed}/{total} passed, {failed} failed.")
    print(f"       Log: {log_path}")


def _log_result(res, i, total):
    status = "PASS" if res["ok"] else "FAIL"
    sid    = res["scenario_id"]
    extra  = f"  rca={res.get('most_suspicious_service')}  sla_violated={res.get('sla_violated')}" if res["ok"] else f"  error={res.get('error', '')[:80]}"
    print(f"[{i:4d}/{total}] [{status}] {sid}{extra}")


if __name__ == "__main__":
    main()
