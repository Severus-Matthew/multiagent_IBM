from __future__ import annotations

import argparse
import json

from digital_twin_runtime.live_capabilities import audit_live_training_records

from .data_loader import iter_scenarios
from .split_utils import read_scenario_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed live GRPO dataset preflight")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", required=True)
    args = ap.parse_args()
    selected = read_scenario_ids(args.scenario_ids) or set()
    records = [
        row for row in iter_scenarios(args.processed_states, allowed_ids=selected)
    ]
    found = {row.scenario_id for row in records}
    missing = sorted(selected - found)
    report = audit_live_training_records(records)
    report["missing_scenario_ids"] = missing
    report["all_supported"] = bool(records) and not missing and report["all_supported"]
    report["status"] = "PASS_LIVE_TRAINING_DATASET" if report["all_supported"] else "FAIL_LIVE_TRAINING_DATASET"
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_supported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
