from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .data_loader import iter_scenarios
from .ground_truth import ground_truth_summary, labels_from_full_state
from .rca_reward import exact_set_match
from .schemas import FaultLabel, parse_fault_lines, normalize_fault_type
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
    ap = argparse.ArgumentParser(description="Audit whether RCA labels and predictions reproduce observed symptoms in the behavioral twin proxy.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--rollouts_jsonl", default=None, help="Optional RCA rollout file whose final predictions should be audited.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "twin_reproduction_audit.jsonl"
    summary_path = out_dir / "summary.json"
    failures_path = out_dir / "failed_oracle_reproduction_cases.json"

    allowed_ids = read_scenario_ids(args.scenario_ids)
    rollout_predictions = _load_rollout_predictions(args.rollouts_jsonl)
    twin = BehavioralTwinVerifier()

    rows: list[dict[str, Any]] = []
    total = 0
    oracle_pass = 0
    wrong_service_fp = 0
    wrong_type_fp = 0
    rollout_available = 0
    rollout_label_match = 0
    rollout_twin_pass = 0
    rollout_both_pass = 0
    rollout_label_pass_twin_fail = 0
    rollout_label_fail_twin_pass = 0

    with rows_path.open("w", encoding="utf-8") as f:
        for rec in iter_scenarios(args.processed_states):
            if allowed_ids is not None and rec.scenario_id not in allowed_ids:
                continue
            gt = labels_from_full_state(rec.full_state)
            if not gt:
                continue
            if args.limit is not None and total >= args.limit:
                break
            total += 1

            oracle_result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, gt)
            oracle_score = _score(oracle_result)
            oracle_ok = oracle_score >= args.threshold
            oracle_pass += int(oracle_ok)

            wrong_service = _wrong_service_control(rec.compressed_state, gt)
            wrong_service_result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, wrong_service) if wrong_service else None
            wrong_service_score = _score(wrong_service_result)
            wrong_service_ok = wrong_service_score >= args.threshold
            wrong_service_fp += int(wrong_service_ok)

            wrong_type = _wrong_type_control(gt)
            wrong_type_result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, wrong_type) if wrong_type else None
            wrong_type_score = _score(wrong_type_result)
            wrong_type_ok = wrong_type_score >= args.threshold
            wrong_type_fp += int(wrong_type_ok)

            pred_text = rollout_predictions.get(rec.scenario_id)
            pred_labels = parse_fault_lines(pred_text or "")
            pred_result = twin.validate_rca_prediction(rec.full_state, rec.compressed_state, pred_labels) if pred_labels else None
            pred_score = _score(pred_result)
            pred_twin_ok = pred_score >= args.threshold
            pred_label_ok = exact_set_match(gt, pred_labels) if pred_labels else False
            if pred_text is not None:
                rollout_available += 1
                rollout_label_match += int(pred_label_ok)
                rollout_twin_pass += int(pred_twin_ok)
                rollout_both_pass += int(pred_label_ok and pred_twin_ok)
                rollout_label_pass_twin_fail += int(pred_label_ok and not pred_twin_ok)
                rollout_label_fail_twin_pass += int((not pred_label_ok) and pred_twin_ok)

            row = {
                "scenario_id": rec.scenario_id,
                "threshold": args.threshold,
                "ground_truth_summary": ground_truth_summary(rec.full_state),
                "oracle": {
                    "score": oracle_score,
                    "pass": oracle_ok,
                    "result": _compact_twin_result(oracle_result),
                },
                "negative_controls": {
                    "wrong_service": {
                        "prediction": _labels_to_text(wrong_service),
                        "score": wrong_service_score,
                        "false_positive": wrong_service_ok,
                        "result": _compact_twin_result(wrong_service_result),
                    },
                    "wrong_type": {
                        "prediction": _labels_to_text(wrong_type),
                        "score": wrong_type_score,
                        "false_positive": wrong_type_ok,
                        "result": _compact_twin_result(wrong_type_result),
                    },
                },
                "rollout_prediction": {
                    "available": pred_text is not None,
                    "prediction_text": pred_text,
                    "label_match": pred_label_ok,
                    "twin_score": pred_score,
                    "twin_pass": pred_twin_ok,
                    "fully_verified_correct": bool(pred_label_ok and pred_twin_ok),
                    "label_pass_twin_fail": bool(pred_label_ok and not pred_twin_ok),
                    "label_fail_twin_pass": bool((not pred_label_ok) and pred_twin_ok),
                    "result": _compact_twin_result(pred_result),
                },
                "audit_note": "Behavioral proxy only. A future live K8s twin audit should replace or confirm these scores.",
            }
            rows.append(row)
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    summary = {
        "total": total,
        "threshold": args.threshold,
        "oracle_reproduction_pass": oracle_pass,
        "oracle_reproduction_pass_rate": oracle_pass / max(total, 1),
        "wrong_service_false_positive": wrong_service_fp,
        "wrong_service_false_positive_rate": wrong_service_fp / max(total, 1),
        "wrong_type_false_positive": wrong_type_fp,
        "wrong_type_false_positive_rate": wrong_type_fp / max(total, 1),
        "rollout_predictions_available": rollout_available,
        "rollout_label_match": rollout_label_match,
        "rollout_label_match_rate": rollout_label_match / max(rollout_available, 1),
        "rollout_twin_pass": rollout_twin_pass,
        "rollout_twin_pass_rate": rollout_twin_pass / max(rollout_available, 1),
        "rollout_fully_verified_correct": rollout_both_pass,
        "rollout_fully_verified_correct_rate": rollout_both_pass / max(rollout_available, 1),
        "rollout_label_pass_twin_fail": rollout_label_pass_twin_fail,
        "rollout_label_fail_twin_pass": rollout_label_fail_twin_pass,
        "rows_jsonl": str(rows_path),
        "failed_oracle_reproduction_cases_json": str(failures_path),
        "mode": "behavioral_offline_proxy_audit",
        "strict_interpretation": "Count final RCA as fully verified only when label_match and twin_pass are both true. If oracle_reproduction_pass is low, audit the twin/state abstraction before training with twin reward.",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    failed = [r for r in rows if not r["oracle"]["pass"]]
    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, sort_keys=True, default=str)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_rollout_predictions(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            row = json.loads(raw)
            sid = row.get("scenario_id")
            pred = row.get("final_prediction")
            if sid and isinstance(pred, str):
                out[str(sid)] = pred
    return out


def _score(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    try:
        return float(result.get("reproduction_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _compact_twin_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        "mode": result.get("mode"),
        "reproduction_score": result.get("reproduction_score"),
        "same_error_pattern_score": result.get("same_error_pattern_score"),
        "per_fault": result.get("per_fault"),
        "predicted_signature": result.get("predicted_signature"),
        "evidence_signature_summary": result.get("evidence_signature_summary"),
        "uses_oracle_labels": result.get("uses_oracle_labels"),
        "uses_full_state_for_rca_score": result.get("uses_full_state_for_rca_score"),
    }


def _labels_to_text(labels: list[FaultLabel] | None) -> str | None:
    if labels is None:
        return None
    return "\n".join(f"{x.service}::{normalize_fault_type(x.fault_type)}" for x in labels)


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
    fallback = next((s for s in services if s not in gt_services), None)
    if fallback is None:
        fallback = "unknown"
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
                services.add(a); services.add(b)
    return sorted(s for s in services if s and s != "unknown")


if __name__ == "__main__":
    main()
