from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from typing import Any

from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .live_dataset_admission import assess_record_for_live_reward
from .schemas import FaultLabel
from .split_utils import read_scenario_ids


_COMPATIBLE_WRONG_MECHANISM = {
    "assign_to_non_existent_node": "scale_replicas_zero",
    "scale_replicas_zero": "assign_to_non_existent_node",
}


def _verifier(args: argparse.Namespace, case_id: str) -> SparseLiveTwinVerifier:
    return SparseLiveTwinVerifier(SparseLiveVerifierConfig(
        source_namespace="unused-dynamic-profile",
        application_source_root=args.application_source_root,
        state_abstraction_root=args.state_abstraction_root,
        baseline_timeout_seconds=args.baseline_timeout_seconds,
        reproduction_threshold=0.0,
        require_reward_calibration=False,
        artifact_root=str(Path(args.output_dir) / "telemetry"),
    ))


def _wrong_service(record: Any, label: FaultLabel, args: argparse.Namespace) -> str:
    verifier = _verifier(args, "planning-only")
    profile = verifier._profile(record.compressed_state)
    state = verifier._planner_state(record.compressed_state, profile)
    candidates = sorted(set(record.compressed_state.get("services", []) or []) - {label.service})
    for service in candidates:
        candidate = FaultLabel(
            service=service, fault_type=label.fault_type,
            fault_mechanism=label.fault_mechanism, variant_name=label.variant_name,
        )
        spec = build_sparse_live_twin_spec(state, [candidate])
        if spec.services_to_keep and not spec.resource_summary.get("invalid_topology"):
            return service
    raise RuntimeError(f"no reachable wrong-service control for {record.scenario_id}")


def _run_control(
    record: Any, fault: FaultLabel, control: str, args: argparse.Namespace
) -> dict[str, Any]:
    case_id = f"{record.scenario_id}--{control}"
    verifier = _verifier(args, case_id)
    verifier.begin_trajectory(case_id)
    result: dict[str, Any] = {}
    recovery = None
    recovery_workload = None
    error = None
    try:
        result = verifier.validate_rca_prediction(
            record.full_state, record.compressed_state, [fault]
        )
        if not result.get("predicted_fault_injection_checked"):
            raise RuntimeError(str(result.get("reason") or "live control injection failed"))
        for handle in reversed(verifier.handles):
            handle.restore()
        assert verifier.session is not None
        recovery = verifier.session.wait_for_clean_baseline(
            timeout_seconds=args.baseline_timeout_seconds
        )
        if not recovery.ready:
            raise RuntimeError("control did not recover")
        recovery_workload = verifier._run_workload(fault.service)
        if (
            not recovery_workload.completed or recovery_workload.failed
            or recovery_workload.application_failures > 0
            or (recovery_workload.required_ready_endpoints or 0) <= 0
        ):
            raise RuntimeError("recovery workload did not restore selected path")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        verifier.end_trajectory()
    lifecycle = bool(
        not error
        and (result.get("baseline") or {}).get("ready")
        and result.get("predicted_fault_injection_checked")
        and recovery and recovery.ready
        and recovery_workload and recovery_workload.completed
    )
    return {
        "case_id": case_id,
        "scenario_id": record.scenario_id,
        "control": control,
        "fault": fault.to_dict(),
        "score": float(result.get("reproduction_score", 0.0) or 0.0),
        "lifecycle_passed": lifecycle,
        "error": error,
        "result": result,
        "recovery": recovery.to_dict() if recovery else None,
        "recovery_workload": recovery_workload.to_dict() if recovery_workload else None,
    }


def _run_triplet(
    record: Any, args: argparse.Namespace, matched_wrong_services: dict[str, list[str]]
) -> list[dict[str, Any]]:
    positive = labels_from_full_state(record.full_state)[0]
    matched = [
        service for service in matched_wrong_services.get(positive.fault_mechanism, [])
        if service != positive.service
    ]
    wrong_service = FaultLabel(
        service=(matched[0] if matched else _wrong_service(record, positive, args)),
        fault_type=positive.fault_type,
        fault_mechanism=positive.fault_mechanism,
        variant_name=positive.variant_name,
    )
    wrong_mechanism_name = _COMPATIBLE_WRONG_MECHANISM[positive.fault_mechanism]
    wrong_mechanism = FaultLabel(
        service=positive.service,
        fault_type="infra_failure",
        fault_mechanism=wrong_mechanism_name,
        variant_name="default",
    )
    return [
        _run_control(record, positive, "positive", args),
        _run_control(record, wrong_service, "wrong_service", args),
        _run_control(record, wrong_mechanism, "wrong_mechanism", args),
    ]


def _calibrate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["lifecycle_passed"]]
    positives = [r["score"] for r in usable if r["control"] == "positive"]
    negatives = [r["score"] for r in usable if r["control"] != "positive"]
    thresholds = sorted({0.0, 1.0, *positives, *negatives})
    table = []
    for threshold in thresholds:
        tpr = sum(x >= threshold for x in positives) / max(1, len(positives))
        fpr = sum(x >= threshold for x in negatives) / max(1, len(negatives))
        table.append({
            "threshold": threshold, "positive_pass_rate": tpr,
            "negative_false_positive_rate": fpr, "youden_j": tpr - fpr,
        })
    strict = [x for x in table if x["positive_pass_rate"] >= 0.8 and x["negative_false_positive_rate"] <= 0.1]
    recommended = max(strict, key=lambda x: (x["youden_j"], x["threshold"]), default=None)
    if positives and negatives and min(positives) > max(negatives):
        # Choose the midpoint of the observed margin rather than putting a
        # boundary exactly on the weakest positive observation.
        margin_threshold = (min(positives) + max(negatives)) / 2.0
        recommended = {
            "threshold": round(margin_threshold, 4),
            "positive_pass_rate": 1.0,
            "negative_false_positive_rate": 0.0,
            "youden_j": 1.0,
            "selection": "midpoint_between_worst_positive_and_worst_negative",
            "observed_margin": round(min(positives) - max(negatives), 4),
        }
    return {
        "positive_scores": positives,
        "negative_scores": negatives,
        "all_controls_completed": len(usable) == len(rows),
        "recommended": recommended,
        "has_strict_separation": recommended is not None,
        "threshold_table": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--application_source_root", default="AIOpsLab/aiopslab-applications/socialNetwork")
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--baseline_timeout_seconds", type=float, default=240.0)
    args = ap.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    allowed = read_scenario_ids(args.scenario_ids) or set()
    candidates = []
    for record in iter_scenarios(args.processed_states, allowed_ids=allowed):
        labels = labels_from_full_state(record.full_state)
        if (
            len(labels) == 1
            and labels[0].fault_mechanism in _COMPATIBLE_WRONG_MECHANISM
            and assess_record_for_live_reward(record)["eligible"]
        ):
            candidates.append(record)
    random.Random(args.seed).shuffle(candidates)
    # Balance the two mechanism directions when possible.
    selected = []
    for mechanism in sorted(_COMPATIBLE_WRONG_MECHANISM):
        selected.extend([r for r in candidates if labels_from_full_state(r.full_state)[0].fault_mechanism == mechanism][: max(1, args.count // 2)])
    selected = selected[:args.count]
    matched_wrong_services: dict[str, list[str]] = {}
    for record in selected:
        label = labels_from_full_state(record.full_state)[0]
        matched_wrong_services.setdefault(label.fault_mechanism, []).append(label.service)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_triplet, record, args, matched_wrong_services): record
            for record in selected
        }
        for future in as_completed(futures):
            triplet = future.result(); rows.extend(triplet)
            print(json.dumps({"scenario_id": triplet[0]["scenario_id"], "scores": {r["control"]: r["score"] for r in triplet}, "lifecycles": {r["control"]: r["lifecycle_passed"] for r in triplet}}), flush=True)
    calibration = _calibrate(rows)
    payload = {"selected_scenarios": [r.scenario_id for r in selected], "rows": rows, "calibration": calibration}
    (out / "live_threshold_controls.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(calibration, indent=2, sort_keys=True))
    if not calibration["all_controls_completed"] or not calibration["has_strict_separation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
