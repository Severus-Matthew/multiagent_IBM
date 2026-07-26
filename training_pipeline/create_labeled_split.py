from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .split_utils import write_scenario_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Create labeled/unlabeled scenario split files from processed_states.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    labeled: list[str] = []
    unlabeled: list[str] = []
    missing_examples: list[dict] = []

    for rec in iter_scenarios(args.processed_states):
        labels = labels_from_full_state(rec.full_state)
        if labels:
            labeled.append(rec.scenario_id)
        else:
            unlabeled.append(rec.scenario_id)
            if len(missing_examples) < 20:
                fc = rec.full_state.get("fault_context", {}) or {}
                missing_examples.append({
                    "scenario_id": rec.scenario_id,
                    "fault_family": fc.get("fault_family"),
                    "task": fc.get("task"),
                    "faulty_service": fc.get("faulty_service"),
                    "expected_faulty_services": fc.get("expected_faulty_services"),
                })

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_scenario_ids(out / f"labeled_{len(labeled)}.txt", labeled)
    write_scenario_ids(out / f"unlabeled_{len(unlabeled)}.txt", unlabeled)

    summary = {
        "processed_states": str(Path(args.processed_states).expanduser()),
        "total": len(labeled) + len(unlabeled),
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "labeled_file": str(out / f"labeled_{len(labeled)}.txt"),
        "unlabeled_file": str(out / f"unlabeled_{len(unlabeled)}.txt"),
        "missing_examples": missing_examples,
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
