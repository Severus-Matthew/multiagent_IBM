from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from typing import Any

from digital_twin_runtime.live_capabilities import LIVE_MECHANISM_CAPABILITIES

from .audit_live_twin_controls import _run_control
from .data_loader import iter_scenarios
from .label_corrections import load_manifest
from .ground_truth import labels_from_full_state
from .live_dataset_admission import assess_record_for_live_reward
from .schemas import FaultLabel
from .split_utils import read_scenario_ids


def _pending() -> list[str]:
    return sorted(
        name for name, capability in LIVE_MECHANISM_CAPABILITIES.items()
        if capability.live_audit_status == "pending_live_audit"
    )


def _select_cases(
    records: list[Any], per_mechanism: int, seed: int, mechanisms: list[str]
) -> list[tuple[Any, FaultLabel]]:
    buckets: dict[str, list[tuple[Any, FaultLabel]]] = defaultdict(list)
    fallback: list[tuple[Any, FaultLabel]] = []
    for record in records:
        labels = labels_from_full_state(record.full_state)
        if labels:
            fallback.append((record, labels[0]))
        for label in labels:
            if label.fault_mechanism in _pending():
                buckets[label.fault_mechanism].append((record, label))
    rng = random.Random(seed)
    selected: list[tuple[Any, FaultLabel]] = []
    for mechanism in mechanisms:
        candidates = buckets.get(mechanism, [])
        single = [pair for pair in candidates if len(labels_from_full_state(pair[0].full_state)) == 1]
        multi = [pair for pair in candidates if len(labels_from_full_state(pair[0].full_state)) != 1]
        rng.shuffle(single); rng.shuffle(multi)
        candidates = single + multi
        if mechanism == "container_stop" and not candidates:
            # No current corpus label uses this adapter. Exercise it against
            # corpus-derived services; this is adapter evidence, not corpus
            # reproduction evidence and is reported as synthetic below.
            candidates = [
                (record, FaultLabel(
                    service=base.service,
                    fault_type="infra_failure",
                    fault_mechanism="container_stop",
                    variant_name="default",
                ))
                for record, base in fallback
            ]
            rng.shuffle(candidates)
        selected.extend(candidates[:per_mechanism])
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--per_mechanism", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument(
        "--mechanisms", nargs="*", choices=_pending(), default=None,
        help="Optional pending-mechanism subset for focused repair validation.",
    )
    ap.add_argument("--application_source_root", default="AIOpsLab/aiopslab-applications/socialNetwork")
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--baseline_timeout_seconds", type=float, default=300.0)
    ap.add_argument("--label_corrections", default=None,
                    help="label-correction manifest applied to private labels only")
    args = ap.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    allowed = read_scenario_ids(args.scenario_ids) or set()
    records = [
        record for record in iter_scenarios(args.processed_states, allowed_ids=allowed,
                                            label_corrections=load_manifest(args.label_corrections))
        if assess_record_for_live_reward(record)["eligible"]
    ]
    mechanisms = args.mechanisms or _pending()
    cases = _select_cases(records, args.per_mechanism, args.seed, mechanisms)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_control, record, label, "adapter_lifecycle", args): (record, label)
            for record, label in cases
        }
        for future in as_completed(futures):
            record, label = futures[future]
            row = future.result()
            row["mechanism"] = label.fault_mechanism
            row["synthetic_adapter_probe"] = not any(
                actual.fault_mechanism == label.fault_mechanism
                for actual in labels_from_full_state(record.full_state)
            )
            rows.append(row)
            print(json.dumps({
                "mechanism": label.fault_mechanism,
                "scenario_id": record.scenario_id,
                "passed": row["lifecycle_passed"],
                "error": row["error"],
            }), flush=True)
    by_mechanism: dict[str, dict[str, Any]] = {}
    for mechanism in mechanisms:
        tested = [row for row in rows if row["mechanism"] == mechanism]
        by_mechanism[mechanism] = {
            "tested": len(tested),
            "passed": sum(row["lifecycle_passed"] for row in tested),
            "all_lifecycle_passed": (
                len(tested) == args.per_mechanism
                and all(row["lifecycle_passed"] for row in tested)
            ),
            "promotion_eligible": (
                len(tested) == args.per_mechanism
                and all(row["lifecycle_passed"] for row in tested)
            ),
        }
    payload = {
        "policy": "two independent full-lifecycle probes per pending mechanism",
        "per_mechanism_required": args.per_mechanism,
        "rows": rows,
        "mechanisms": by_mechanism,
        "all_passed": bool(by_mechanism) and all(
            result["promotion_eligible"] for result in by_mechanism.values()
        ),
    }
    path = Path(args.output_dir) / "pending_adapter_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"artifact": str(path), **by_mechanism}, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
