from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .data_loader import iter_scenarios
from .ground_truth import ground_truth_summary, labels_from_full_state
from .schemas import FaultLabel, normalize_fault_type
from .split_utils import read_scenario_ids


FAULT_TYPES = [
    "infra_failure",
    "auth_failure",
    "dependency_failure",
    "resource_exhaustion",
    "latency_degradation",
    "network_failure",
    "config_error",
    "unknown",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit which same-service wrong RCA mechanisms the counterfactual twin over-accepts."
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "type_confusion_rows.jsonl"
    summary_path = out_dir / "summary.json"

    allowed_ids = read_scenario_ids(args.scenario_ids)
    twin = BehavioralTwinVerifier()

    total_labels = 0
    total_wrong_type_trials = 0
    skipped_true_alternate_type_trials = 0
    wrong_type_fp = 0
    oracle_pass = 0
    oracle_scores: list[float] = []
    wrong_type_scores: list[float] = []
    confusion = Counter()
    by_gt_type: dict[str, Counter] = defaultdict(Counter)
    by_wrong_type: Counter = Counter()
    service_rows = Counter()

    with rows_path.open("w", encoding="utf-8") as f:
        scenario_count = 0
        for rec in iter_scenarios(args.processed_states):
            if allowed_ids is not None and rec.scenario_id not in allowed_ids:
                continue
            gt = labels_from_full_state(rec.full_state)
            if not gt:
                continue
            if args.limit is not None and scenario_count >= args.limit:
                break
            scenario_count += 1

            true_types_by_service: dict[str, set[str]] = defaultdict(set)
            for gt_label in gt:
                true_types_by_service[str(gt_label.service)].add(normalize_fault_type(gt_label.fault_type))

            for label in gt:
                total_labels += 1
                gt_type = normalize_fault_type(label.fault_type)
                oracle_result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, [label])
                oracle_score = _score(oracle_result)
                oracle_scores.append(oracle_score)
                oracle_ok = oracle_score >= args.threshold
                oracle_pass += int(oracle_ok)

                row = {
                    "scenario_id": rec.scenario_id,
                    "ground_truth_summary": ground_truth_summary(rec.full_state),
                    "service": label.service,
                    "gt_fault_type": gt_type,
                    "oracle_score": oracle_score,
                    "oracle_pass": oracle_ok,
                    "wrong_type_trials": [],
                }

                true_types_here = true_types_by_service.get(str(label.service), set())
                for wrong_type in FAULT_TYPES:
                    if wrong_type == gt_type:
                        continue
                    # In a multifault incident the same service may genuinely have
                    # multiple injected mechanisms. Such a type is not a negative
                    # control and must not be counted as a false positive.
                    if wrong_type in true_types_here:
                        skipped_true_alternate_type_trials += 1
                        row["wrong_type_trials"].append({
                            "wrong_fault_type": wrong_type,
                            "skipped": True,
                            "skip_reason": "fault_type_is_also_ground_truth_for_same_service",
                        })
                        continue

                    total_wrong_type_trials += 1
                    pred = [FaultLabel(service=label.service, fault_type=wrong_type)]
                    result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, pred)
                    score = _score(result)
                    ok = score >= args.threshold
                    wrong_type_scores.append(score)
                    wrong_type_fp += int(ok)
                    if ok:
                        confusion[(gt_type, wrong_type)] += 1
                        by_gt_type[gt_type][wrong_type] += 1
                        by_wrong_type[wrong_type] += 1
                        service_rows[str(label.service)] += 1
                    row["wrong_type_trials"].append({
                        "wrong_fault_type": wrong_type,
                        "score": score,
                        "false_positive": ok,
                        "result_mode": (result or {}).get("mode"),
                        "result_components": _compact_components(result),
                    })
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    summary = {
        "mode": "counterfactual_offline_twin_type_confusion_audit",
        "threshold": args.threshold,
        "total_gt_labels": total_labels,
        "total_wrong_type_trials": total_wrong_type_trials,
        "skipped_true_alternate_type_trials": skipped_true_alternate_type_trials,
        "oracle_pass": oracle_pass,
        "oracle_pass_rate_per_label": oracle_pass / max(total_labels, 1),
        "wrong_type_false_positive": wrong_type_fp,
        "wrong_type_false_positive_rate_per_trial": wrong_type_fp / max(total_wrong_type_trials, 1),
        "oracle_score_summary": _score_summary(oracle_scores),
        "wrong_type_score_summary": _score_summary(wrong_type_scores),
        "top_confusions": [
            {"gt_fault_type": k[0], "wrong_fault_type": k[1], "count": v}
            for k, v in confusion.most_common(30)
        ],
        "by_gt_type": {
            gt_type: {wrong: count for wrong, count in counter.most_common()}
            for gt_type, counter in sorted(by_gt_type.items())
        },
        "by_wrong_type": {wrong: count for wrong, count in by_wrong_type.most_common()},
        "top_services_with_wrong_type_fp": [
            {"service": svc, "false_positive_count": count}
            for svc, count in service_rows.most_common(30)
        ],
        "rows_jsonl": str(rows_path),
        "interpretation": (
            "Negative controls exclude any fault type that is also a true injected mechanism on the same service. "
            "The counterfactual twin score itself does not use those hidden labels; they are used here only to construct a valid evaluation set."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _score(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    try:
        return float(result.get("reproduction_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _score_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    xs = sorted(values)
    n = len(xs)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return xs[idx]

    return {
        "min": round(xs[0], 4),
        "p25": round(q(0.25), 4),
        "median": round(q(0.50), 4),
        "p75": round(q(0.75), 4),
        "max": round(xs[-1], 4),
        "mean": round(sum(xs) / n, 4),
    }


def _compact_components(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    out = {}
    for key in (
        "counterfactual_overlap_score",
        "mechanism_channel_presence_score",
        "predicted_service_support",
        "channel_scores",
        "channel_presence",
        "uses_oracle_labels_for_score",
        "uses_full_state_for_score",
    ):
        if key in result:
            out[key] = result[key]
    return out


if __name__ == "__main__":
    main()
