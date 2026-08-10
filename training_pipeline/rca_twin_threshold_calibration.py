from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
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
    ap = argparse.ArgumentParser(description="Calibrate behavioral RCA twin threshold using oracle labels and negative controls.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "twin_threshold_scores.jsonl"
    table_path = out_dir / "threshold_calibration.json"
    summary_path = out_dir / "summary.json"

    allowed_ids = read_scenario_ids(args.scenario_ids)
    twin = BehavioralTwinVerifier()
    rows: list[dict[str, Any]] = []

    with rows_path.open("w", encoding="utf-8") as f:
        total = 0
        for rec in iter_scenarios(args.processed_states):
            if allowed_ids is not None and rec.scenario_id not in allowed_ids:
                continue
            gt = labels_from_full_state(rec.full_state)
            if not gt:
                continue
            if args.limit is not None and total >= args.limit:
                break
            total += 1

            wrong_service = _wrong_service_control(rec.compressed_state, gt)
            wrong_type = _wrong_type_control(gt)
            row = {
                "scenario_id": rec.scenario_id,
                "oracle_score": _score(twin.validate_rca_prediction(rec.full_state, rec.compressed_state, gt)),
                "wrong_service_score": _score(twin.validate_rca_prediction(rec.full_state, rec.compressed_state, wrong_service)),
                "wrong_type_score": _score(twin.validate_rca_prediction(rec.full_state, rec.compressed_state, wrong_type)),
                "num_faults": len(gt),
            }
            rows.append(row)
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    thresholds = [round(x / 100.0, 2) for x in range(0, 81, 5)]
    table = []
    for th in thresholds:
        oracle_pass = sum(1 for r in rows if r["oracle_score"] >= th)
        wrong_service_pass = sum(1 for r in rows if r["wrong_service_score"] >= th)
        wrong_type_pass = sum(1 for r in rows if r["wrong_type_score"] >= th)
        total = max(1, len(rows))
        table.append({
            "threshold": th,
            "oracle_pass": oracle_pass,
            "oracle_pass_rate": oracle_pass / total,
            "wrong_service_false_positive": wrong_service_pass,
            "wrong_service_false_positive_rate": wrong_service_pass / total,
            "wrong_type_false_positive": wrong_type_pass,
            "wrong_type_false_positive_rate": wrong_type_pass / total,
            "separation_margin": (oracle_pass / total) - max(wrong_service_pass / total, wrong_type_pass / total),
            "usable_for_strict_reward": (oracle_pass / total) >= 0.85 and max(wrong_service_pass / total, wrong_type_pass / total) <= 0.10,
        })

    usable = [r for r in table if r["usable_for_strict_reward"]]
    recommendation = usable[0]["threshold"] if usable else None
    summary = {
        "mode": "behavioral_offline_proxy_threshold_calibration",
        "total": len(rows),
        "rows_jsonl": str(rows_path),
        "threshold_calibration_json": str(table_path),
        "recommended_threshold": recommendation,
        "has_usable_strict_threshold": recommendation is not None,
        "interpretation": (
            "Use behavioral twin as strict RCA reward only if a threshold has high oracle pass rate and low negative-control pass rate. "
            "Otherwise keep twin score diagnostic and use label match for supervised/RL reward."
        ),
        "best_by_margin": max(table, key=lambda x: x["separation_margin"]) if table else None,
        "thresholds": table,
    }
    table_path.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _score(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    try:
        return float(result.get("reproduction_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _wrong_type_control(gt: list[FaultLabel]) -> list[FaultLabel]:
    out: list[FaultLabel] = []
    for label in gt:
        current = normalize_fault_type(label.fault_type)
        wrong = next((ft for ft in FAULT_TYPES if ft != current), "unknown")
        out.append(FaultLabel(service=label.service, fault_type=wrong))
    return out


def _wrong_service_control(state: dict[str, Any], gt: list[FaultLabel]) -> list[FaultLabel]:
    services = _observable_services(state)
    gt_services = {str(x.service) for x in gt}
    fallback = next((s for s in services if s not in gt_services), None) or "unknown"
    return [FaultLabel(service=fallback, fault_type=normalize_fault_type(label.fault_type)) for label in gt]


def _observable_services(obj: Any) -> list[str]:
    services: set[str] = set()
    if isinstance(obj, dict):
        for key in ("system", "service_health", "metrics"):
            val = obj.get(key)
            if isinstance(val, dict):
                services.update(str(k) for k in val.keys())
        llm_view = obj.get("llm_view", {}) if isinstance(obj.get("llm_view", {}), dict) else {}
        for row in llm_view.get("top_log_error_services", []) or []:
            if isinstance(row, dict) and row.get("service"):
                services.add(str(row["service"]))
        traces = obj.get("traces", {}) if isinstance(obj.get("traces", {}), dict) else {}
        per_edge = traces.get("per_edge", {}) if isinstance(traces.get("per_edge", {}), dict) else {}
        for edge, feats in per_edge.items():
            if isinstance(feats, dict):
                for field in ("source", "target"):
                    if feats.get(field):
                        services.add(str(feats[field]))
            if "->" in str(edge):
                a, b = str(edge).split("->", 1)
                services.add(a)
                services.add(b)
    return sorted(s for s in services if s and s != "unknown")


if __name__ == "__main__":
    main()
