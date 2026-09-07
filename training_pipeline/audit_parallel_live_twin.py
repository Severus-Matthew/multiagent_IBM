from __future__ import annotations

"""Two-scenario concurrent live-Twin infrastructure smoke (not a training rollout)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--source_namespace", default="test-social-network")
    ap.add_argument("--application_source_root", default="AIOpsLab/aiopslab-applications/socialNetwork")
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--baseline_timeout_seconds", type=float, default=300.0)
    args = ap.parse_args()
    wanted = [x.strip() for x in Path(args.scenario_ids).read_text().splitlines()
              if x.strip() and not x.startswith("#")][:2]
    records = [r for r in iter_scenarios(args.processed_states) if r.scenario_id in wanted]
    records.sort(key=lambda r: wanted.index(r.scenario_id))
    if len(records) != 2:
        raise RuntimeError(f"exactly two matching scenarios required; found {len(records)}")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(index: int, rec: object) -> dict:
        verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
            source_namespace=args.source_namespace,
            application_source_root=str(Path(args.application_source_root).resolve()),
            state_abstraction_root=str(Path(args.state_abstraction_root).resolve()),
            artifact_root=str(out / "twin_artifacts" / f"worker-{index}"),
            baseline_timeout_seconds=args.baseline_timeout_seconds,
            require_reward_calibration=False,
        ))
        verifier.begin_trajectory(f"parallel-smoke:{rec.scenario_id}")
        try:
            result = verifier.validate_rca_prediction(
                rec.full_state, rec.compressed_state, labels_from_full_state(rec.full_state))
            return {"worker": index, "scenario_id": rec.scenario_id, "result": result}
        finally:
            verifier.end_trajectory()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda pair: run(*pair), enumerate(records)))
    summary = {
        "status": "PASS" if all(r["result"].get("predicted_fault_injection_checked") for r in rows) else "FAIL",
        "mode": "parallel_live_twin_oracle_infrastructure_smoke_not_training",
        "parallel_workers": 2, "results": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
