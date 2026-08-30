from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from digital_twin_runtime.sparse_live_manifest import (
    discover_sparse_manifest_plan,
    render_sparse_manifest_bundle,
)
from digital_twin_runtime.sparse_live_session import SparseLiveTwinSession
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.audit_sparse_live_twin_spec import (
    _augment_with_static_topology,
    _default_application_source_root,
    _find_scenario,
    _parse_fault,
)


def _namespace_exists(name: str) -> bool:
    return subprocess.run(
        ["kubectl", "get", "namespace", name, "-o", "name"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Create, validate, and destroy one clean sparse Twin baseline.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument("--source_namespace", required=True)
    ap.add_argument("--target_namespace", required=True)
    ap.add_argument("--fault", action="append", required=True)
    ap.add_argument("--application_source_root", default=None)
    ap.add_argument("--upstream_hops", type=int, default=2)
    ap.add_argument("--downstream_support_hops", type=int, default=2)
    ap.add_argument("--max_entry_path_hops", type=int, default=8)
    ap.add_argument("--timeout_seconds", type=float, default=180.0)
    args = ap.parse_args()

    if _namespace_exists(args.target_namespace):
        raise SystemExit(f"target namespace already exists: {args.target_namespace}")
    rec = _find_scenario(args.processed_states, args.scenario_id)
    faults = [_parse_fault(x) for x in args.fault]
    app_root = Path(args.application_source_root).resolve() if args.application_source_root else _default_application_source_root(args.processed_states)
    planner_state, _ = _augment_with_static_topology(rec.compressed_state, app_root)
    spec = build_sparse_live_twin_spec(
        planner_state, faults,
        upstream_hops=args.upstream_hops,
        downstream_support_hops=args.downstream_support_hops,
        max_entry_path_hops=args.max_entry_path_hops,
    )
    plan = discover_sparse_manifest_plan(args.source_namespace, spec.services_to_keep)
    bundle = render_sparse_manifest_bundle(plan, args.target_namespace)
    session = SparseLiveTwinSession(bundle)
    started = time.monotonic()
    baseline = None
    error = None
    cleanup_error = None
    try:
        session.create_namespace()
        session.apply_manifests()
        baseline = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if session.created:
            try:
                session.destroy()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"

    namespace_removed = not _namespace_exists(args.target_namespace)
    passed = bool(baseline and baseline.ready and namespace_removed and not error and not cleanup_error)
    print(json.dumps({
        "status": "PASS_SPARSE_TWIN_CLEAN_BASELINE" if passed else "FAIL_SPARSE_TWIN_CLEAN_BASELINE",
        "scenario_id": rec.scenario_id,
        "selected_services": spec.services_to_keep,
        "service_reduction_percent": spec.resource_summary.get("service_reduction_percent"),
        "baseline": baseline.to_dict() if baseline else None,
        "total_session_seconds": round(time.monotonic() - started, 3),
        "namespace_removed": namespace_removed,
        "error": error,
        "cleanup_error": cleanup_error,
        "next_gate": "injectible_rca_mechanism_contract" if passed else "fix_clean_baseline",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
