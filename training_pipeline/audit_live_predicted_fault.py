from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from digital_twin_runtime.live_fault_injector import inject_predicted_fault
from digital_twin_runtime.sparse_live_manifest import discover_sparse_manifest_plan, render_sparse_manifest_bundle
from digital_twin_runtime.sparse_live_session import SparseLiveTwinSession
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.audit_sparse_live_twin_spec import (
    _augment_with_static_topology, _default_application_source_root, _find_scenario,
)
from training_pipeline.schemas import parse_fault_lines


def _exists(namespace: str) -> bool:
    return subprocess.run(
        ["kubectl", "get", "namespace", namespace, "-o", "name"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit fresh injection of an RCA-predicted fault in a sparse live Twin.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument("--source_namespace", required=True)
    ap.add_argument("--target_namespace", required=True)
    ap.add_argument("--fault", required=True, help="service::fault_type::injectible_mechanism")
    ap.add_argument("--timeout_seconds", type=float, default=180.0)
    args = ap.parse_args()

    faults = parse_fault_lines(args.fault)
    if len(faults) != 1 or not faults[0].is_injectible():
        raise SystemExit("--fault must contain one valid injectible RCA hypothesis")
    if _exists(args.target_namespace):
        raise SystemExit("target namespace already exists")
    rec = _find_scenario(args.processed_states, args.scenario_id)
    app_root = _default_application_source_root(args.processed_states)
    planner_state, _ = _augment_with_static_topology(rec.compressed_state, Path(app_root))
    spec = build_sparse_live_twin_spec(planner_state, faults, downstream_support_hops=2)
    plan = discover_sparse_manifest_plan(args.source_namespace, spec.services_to_keep)
    bundle = render_sparse_manifest_bundle(plan, args.target_namespace)
    session = SparseLiveTwinSession(bundle)
    baseline = manifestation = recovery = None
    error = cleanup_error = None
    try:
        session.create_namespace()
        session.apply_manifests()
        baseline = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
        if not baseline.ready:
            raise RuntimeError("clean sparse Twin baseline did not become ready")
        handle = inject_predicted_fault(session, faults[0])
        try:
            manifestation = handle.wait_for_manifestation(timeout_seconds=60)
        finally:
            handle.restore()
        recovery = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if session.created:
            try:
                session.destroy()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"

    removed = not _exists(args.target_namespace)
    passed = bool(
        baseline and baseline.ready and manifestation and manifestation.manifested
        and recovery and recovery.ready and removed and not error and not cleanup_error
    )
    print(json.dumps({
        "status": "PASS_LIVE_PREDICTED_FAULT" if passed else "FAIL_LIVE_PREDICTED_FAULT",
        "predicted_fault": faults[0].to_dict(),
        "oracle_fault_used": False,
        "scenario_id_used_for_injection": False,
        "baseline": baseline.to_dict() if baseline else None,
        "manifestation": manifestation.to_dict() if manifestation else None,
        "recovery": recovery.to_dict() if recovery else None,
        "namespace_removed": removed,
        "error": error,
        "cleanup_error": cleanup_error,
        "next_gate": "targeted_workload_and_telemetry" if passed else "fix_live_predicted_fault",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
