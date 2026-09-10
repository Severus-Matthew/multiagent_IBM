from __future__ import annotations

"""Dataset-level live-reward admission audit.

For every record in the train and test id lists this reports, without touching
Kubernetes:

* source faithfulness (did the injector mutate the labeled service),
* comparable symptom evidence and its strength,
* whether every labeled mechanism has a live-verified Twin adapter,

and writes id lists that partition the split into records usable now, records
usable only with weak evidence, records usable after label correction, and
records that must be re-recorded because their capture shows a healthy system.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.live_capabilities import assess_live_capability
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .label_corrections import load_manifest
from .live_dataset_admission import assess_record_for_live_reward
from .split_utils import read_scenario_ids, write_scenario_ids


def _row(record: Any, admit_weak_evidence: bool) -> dict[str, Any]:
    admission = assess_record_for_live_reward(record, admit_weak_evidence=admit_weak_evidence)
    capabilities = [assess_live_capability(x) for x in labels_from_full_state(record.full_state)]
    verified = bool(capabilities) and all(x.get("supported") for x in capabilities)
    return {**admission, "all_adapters_verified_live": verified, "capabilities": capabilities}


def _audit(
    name: str, path: str, processed_states: str, *,
    admit_weak_evidence: bool, corrections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    requested = read_scenario_ids(path)
    records = list(iter_scenarios(processed_states, allowed_ids=requested,
                                  label_corrections=corrections))
    rows = []
    counts: Counter[str] = Counter()
    buckets: dict[str, list[str]] = {
        "eligible": [], "fully_live_reward_admissible": [], "weak_evidence_only": [],
        "needs_regeneration": [], "adapter_pending": [], "label_corrected": [],
    }
    reason_counts: Counter[str] = Counter()
    for record in records:
        row = _row(record, admit_weak_evidence)
        rows.append(row)
        sid = record.scenario_id
        reasons = set(row["reasons"])
        strength = row["comparable_evidence"]["strength"]
        counts["source_faithful"] += int(row["source_injection"]["valid"])
        counts["collected_trace_edges"] += int(row["has_collected_trace_edges"])
        counts[f"evidence_{strength}"] += 1
        counts["source_faithful_and_comparable"] += int(row["eligible"])
        counts["verified_live_adapter"] += int(row["all_adapters_verified_live"])
        counts["fully_live_reward_admissible"] += int(row["eligible"] and row["all_adapters_verified_live"])
        if row["label_correction"]:
            counts["label_corrected"] += 1
            buckets["label_corrected"].append(sid)
        for reason in row["reasons"]:
            reason_counts[reason] += 1
        if row["eligible"]:
            buckets["eligible"].append(sid)
        if row["eligible"] and row["all_adapters_verified_live"]:
            buckets["fully_live_reward_admissible"].append(sid)
        elif row["eligible"]:
            buckets["adapter_pending"].append(sid)
        # A capture is re-recorded when its telemetry cannot support live reward
        # under any admission policy: the injector did not touch the labeled
        # service, or no scored channel holds evidence at the target.
        if (
            "source_injection_not_faithful_to_label" in reasons
            or "historical_capture_has_no_comparable_symptom_evidence" in reasons
        ):
            buckets["needs_regeneration"].append(sid)
        elif "historical_capture_has_only_weak_symptom_evidence" in reasons:
            buckets["weak_evidence_only"].append(sid)
    return {
        "name": name,
        "path": str(Path(path).resolve()),
        "requested_ids": len(requested),
        "unique_requested_ids": len(set(requested)),
        "loadable_records": len(records),
        "missing_ids": sorted(set(requested) - {r.scenario_id for r in records}),
        "counts": dict(counts),
        "rejection_reasons": dict(reason_counts),
        "buckets": {key: sorted(value) for key, value in buckets.items()},
        "source_faithful_and_traced_ids": sorted(buckets["eligible"]),
        "fully_live_reward_admissible_ids": sorted(buckets["fully_live_reward_admissible"]),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--selection_dir")
    ap.add_argument("--admit_weak_evidence", action="store_true",
                    help="admit captures whose only target evidence is service-health flags or log error counts")
    ap.add_argument("--label_corrections",
                    help="manifest from training_pipeline.label_corrections; applied to private labels only")
    args = ap.parse_args()
    corrections = load_manifest(args.label_corrections)
    kwargs = {"admit_weak_evidence": args.admit_weak_evidence, "corrections": corrections}
    train = _audit("train", args.train, args.processed_states, **kwargs)
    test = _audit("test", args.test, args.processed_states, **kwargs)
    train_ids, test_ids = read_scenario_ids(args.train), read_scenario_ids(args.test)
    contract = "source-faithful capture + comparable symptom evidence + verified-live mechanism adapter"
    report = {
        "contract": contract,
        "evidence_policy": (
            "weak evidence admitted" if args.admit_weak_evidence
            else "structural or trace evidence at the labeled target required"
        ),
        "label_corrections": str(Path(args.label_corrections).resolve()) if args.label_corrections else None,
        "train": train,
        "test": test,
        "train_test_overlap": sorted(train_ids & test_ids),
        "split_valid": not (train_ids & test_ids) and not train["missing_ids"] and not test["missing_ids"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.selection_dir:
        selection = Path(args.selection_dir)
        selection.mkdir(parents=True, exist_ok=True)
        for split in (train, test):
            for bucket, ids in split["buckets"].items():
                write_scenario_ids(selection / f"{split['name']}_{bucket}.txt", ids)
        # Backwards-compatible names used by earlier curricula.
        write_scenario_ids(selection / "train_source_faithful_traced.txt", train["source_faithful_and_traced_ids"])
        write_scenario_ids(selection / "test_source_faithful_traced.txt", test["source_faithful_and_traced_ids"])
        write_scenario_ids(selection / "train_verified_live_reward.txt", train["fully_live_reward_admissible_ids"])
        write_scenario_ids(selection / "test_verified_live_reward.txt", test["fully_live_reward_admissible_ids"])
    compact = {
        key: {
            "requested": value["requested_ids"],
            **value["counts"],
            "rejection_reasons": value["rejection_reasons"],
            "needs_regeneration": len(value["buckets"]["needs_regeneration"]),
            "weak_evidence_only": len(value["buckets"]["weak_evidence_only"]),
            "adapter_pending": len(value["buckets"]["adapter_pending"]),
        }
        for key, value in (("train", train), ("test", test))
    }
    compact["split_valid"] = report["split_valid"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
