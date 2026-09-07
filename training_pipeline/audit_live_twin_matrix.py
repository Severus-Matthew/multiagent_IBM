from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import subprocess
import threading
from typing import Any

from digital_twin_runtime.live_capabilities import assess_live_capability
from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig

from .data_loader import iter_scenarios
from .label_corrections import load_manifest
from .ground_truth import labels_from_full_state
from .live_dataset_admission import assess_record_for_live_reward
from .split_utils import read_scenario_ids


def _application(record: Any) -> str:
    context = record.full_state.get("fault_context", {}) or {}
    return str(context.get("app") or context.get("target_namespace") or "unclassified")


def _eligible(record: Any, *, include_pending_adapters: bool = False) -> bool:
    labels = labels_from_full_state(record.full_state)
    return (
        bool(labels)
        and assess_record_for_live_reward(record)["eligible"]
        and all(assess_live_capability(
            label, require_verified_live=not include_pending_adapters
        )["supported"] for label in labels)
    )


def _select(records: list[Any], count: int, seed: int) -> list[Any]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        labels = labels_from_full_state(record.full_state)
        mechanisms = "+".join(sorted({label.fault_mechanism for label in labels}))
        buckets[f"{_application(record)}:{'multi' if len(labels)>1 else 'single'}:{mechanisms}"].append(record)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[Any] = []
    keys = sorted(buckets)
    while keys and len(selected) < count:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def _namespace_exists(namespace: str | None) -> bool:
    if not namespace:
        return False
    return subprocess.run(
        ["kubectl", "get", "namespace", namespace, "-o", "name"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _run_case(record: Any, args: argparse.Namespace) -> dict[str, Any]:
    labels = labels_from_full_state(record.full_state)
    verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
        source_namespace="unused-dynamic-profile",
        application_source_root=args.application_source_root,
        state_abstraction_root=args.state_abstraction_root,
        baseline_timeout_seconds=args.baseline_timeout_seconds,
        reproduction_threshold=args.reproduction_threshold,
        require_reward_calibration=False,
        artifact_root=str(Path(args.artifact_root).resolve() / record.scenario_id),
    ))
    namespace = None
    result: dict[str, Any] = {}
    recovery = None
    recovery_workload = None
    cleanup = False
    error = None
    verifier.begin_trajectory(record.scenario_id)
    try:
        result = verifier.validate_rca_prediction(record.full_state, record.compressed_state, labels)
        namespace = verifier.action_namespace()
        if not result.get("predicted_fault_injection_checked"):
            raise RuntimeError(str(result.get("reason") or "RCA live injection/workload failed"))
        for handle in reversed(verifier.handles):
            handle.restore()
        assert verifier.session is not None
        recovery = verifier.session.wait_for_clean_baseline(
            timeout_seconds=args.baseline_timeout_seconds
        )
        if not recovery.ready:
            raise RuntimeError("restored Twin did not return to clean baseline")
        recovery_workload = verifier._run_workload(labels[0].service)
        if not recovery_workload.completed or recovery_workload.failed:
            raise RuntimeError("recovery workload did not complete")
        if float(result.get("reproduction_score", 0.0) or 0.0) < args.reproduction_threshold:
            error = "ReproductionThresholdNotMet: live controls passed but score is below calibrated gate"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        verifier.end_trajectory()
        cleanup = not _namespace_exists(namespace)
    gates = {
        "baseline": bool((result.get("baseline") or {}).get("ready")),
        "manifestation": bool(result.get("manifestations")) and all(
            row.get("manifested") for row in result.get("manifestations", [])
        ),
        "workload": bool(
            ((result.get("workload") or {}).get("completed")
             or (result.get("workload") or {}).get("failed"))
            and (
                str((result.get("workload") or {}).get("output") or "").strip()
                or (result.get("workload") or {}).get("execution_started")
            )
        ),
        "telemetry_comparison": "reproduction_score" in result,
        "reproduction_threshold": float(result.get("reproduction_score", 0.0)) >= args.reproduction_threshold,
        "recovery": bool(recovery and recovery.ready),
        "recovery_workload": bool(recovery_workload and recovery_workload.completed and not recovery_workload.failed),
        "cleanup": cleanup,
    }
    return {
        "scenario_id": record.scenario_id, "application": _application(record),
        "faults": [label.to_dict() for label in labels], "gates": gates,
        "passed": all(gates.values()) and not error, "error": error,
        "reproduction_score": result.get("reproduction_score"),
        "services_selected": result.get("services_selected"),
        "service_reduction_percent": result.get("service_reduction_percent"),
        "telemetry_artifact_path": result.get("telemetry_artifact_path"),
        "recovery_workload": recovery_workload.to_dict() if recovery_workload else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_states", required=True)
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--application_source_root", default="AIOpsLab/aiopslab-applications/socialNetwork")
    parser.add_argument("--state_abstraction_root", default="state_abstraction_full")
    parser.add_argument("--baseline_timeout_seconds", type=float, default=240.0)
    parser.add_argument("--reproduction_threshold", type=float, default=0.4702)
    parser.add_argument("--label_corrections", default=None,
                        help="label-correction manifest applied to private labels only")
    parser.add_argument(
        "--include_pending_adapters", action="store_true",
        help="Exercise implemented pending adapters; never implies live-reward admission or promotion.",
    )
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    args.artifact_root = str(out / "telemetry")
    allowed = read_scenario_ids(args.train_ids) or set()
    records = [
        r for r in iter_scenarios(args.processed_states, allowed_ids=allowed,
                                  label_corrections=load_manifest(args.label_corrections))
        if _eligible(r, include_pending_adapters=args.include_pending_adapters)
    ]
    selected = _select(records, args.count, args.seed)
    (out / "selected_ids.txt").write_text("\n".join(r.scenario_id for r in selected) + "\n")
    results_path = out / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line); completed[row["scenario_id"]] = row
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_case, r, args): r for r in selected if r.scenario_id not in completed}
        for future in as_completed(futures):
            row = future.result(); completed[row["scenario_id"]] = row
            with lock, results_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"scenario_id": row["scenario_id"], "passed": row["passed"], "error": row["error"]}))
    rows = [completed[r.scenario_id] for r in selected if r.scenario_id in completed]
    mechanism = Counter()
    for row in rows:
        for fault in row["faults"]:
            mechanism[f"{fault['fault_mechanism']}:{'pass' if row['passed'] else 'fail'}"] += 1
    summary = {
        "selected": len(selected), "completed": len(rows),
        "passed": sum(bool(r["passed"]) for r in rows),
        "failed": sum(not r["passed"] for r in rows),
        "all_passed": len(rows) == len(selected) and all(r["passed"] for r in rows),
        "mechanism_results": dict(sorted(mechanism.items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
