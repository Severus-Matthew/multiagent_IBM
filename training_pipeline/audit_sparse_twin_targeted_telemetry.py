from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from digital_twin_runtime.live_fault_injector import inject_predicted_fault
from digital_twin_runtime.sparse_live_manifest import discover_sparse_manifest_plan, render_sparse_manifest_bundle
from digital_twin_runtime.sparse_live_session import SparseLiveTwinSession
from digital_twin_runtime.targeted_telemetry import collect_targeted_telemetry
from digital_twin_runtime.targeted_workload import run_targeted_wrk
from digital_twin_runtime.telemetry_comparator import compare_symptoms_scoped
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
    ap = argparse.ArgumentParser(description="Audit targeted sparse-Twin workload, telemetry, abstraction, and scoped comparison.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument("--source_namespace", required=True)
    ap.add_argument("--target_namespace", required=True)
    ap.add_argument("--fault", required=True)
    ap.add_argument("--timeout_seconds", type=float, default=180.0)
    args = ap.parse_args()

    faults = parse_fault_lines(args.fault)
    if len(faults) != 1 or not faults[0].is_injectible():
        raise SystemExit("one injectible predicted fault is required")
    if _exists(args.target_namespace):
        raise SystemExit("target namespace already exists")
    rec = _find_scenario(args.processed_states, args.scenario_id)
    app_root = _default_application_source_root(args.processed_states)
    planner_state, _ = _augment_with_static_topology(rec.compressed_state, Path(app_root))
    spec = build_sparse_live_twin_spec(planner_state, faults, downstream_support_hops=2)
    plan = discover_sparse_manifest_plan(args.source_namespace, spec.services_to_keep)
    bundle = render_sparse_manifest_bundle(plan, args.target_namespace)
    session = SparseLiveTwinSession(bundle)

    baseline = manifestation = recovery = workload = collection = comparison = None
    abstraction = None
    error = cleanup_error = None
    with tempfile.TemporaryDirectory(prefix="aiops-sparse-telemetry-") as temp_dir:
        run_dir = Path(temp_dir) / "opaque-twin-incident"
        output_dir = Path(temp_dir) / "processed"
        try:
            session.create_namespace()
            session.apply_manifests()
            baseline = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
            if not baseline.ready:
                raise RuntimeError("clean baseline did not stabilize")
            handle = inject_predicted_fault(session, faults[0])
            try:
                manifestation = handle.wait_for_manifestation(timeout_seconds=60)
                if not manifestation.manifested:
                    raise RuntimeError("predicted fault did not manifest")
                workload = run_targeted_wrk(
                    session,
                    payload_script=(
                        Path(app_root) / "wrk2" / "scripts" / "social-network"
                        / "read-user-timeline.lua"
                    ),
                    endpoint="http://nginx-thrift:8080/wrk2-api/user-timeline/read",
                    required_service=faults[0].service,
                    duration_seconds=10,
                    timeout_seconds=90,
                )
                collection = collect_targeted_telemetry(
                    session, run_dir, workload=workload
                )
            finally:
                handle.restore()
            recovery = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)

            pipeline = subprocess.run(
                [
                    sys.executable, "run_pipeline.py", "--run_dir", str(run_dir),
                    "--output_dir", str(output_dir), "--skip_simulator",
                ],
                cwd=Path(__file__).resolve().parents[1] / "state_abstraction_full",
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if pipeline.returncode != 0:
                raise RuntimeError(f"state abstraction failed: {pipeline.stderr[-2000:]}")
            twin_state = json.loads((output_dir / "state_abstraction_compressed.json").read_text())
            abstraction = {
                "services": twin_state.get("services", []),
                "metrics_present": bool(twin_state.get("metrics")),
                "logs_present": bool(twin_state.get("logs")),
                "traces_present": bool(twin_state.get("traces")),
                "system_present": bool(twin_state.get("system")),
                "pipeline_stdout_tail": pipeline.stdout[-1000:],
            }
            comparison = compare_symptoms_scoped(
                rec.compressed_state, twin_state, spec.services_to_keep
            )
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
        and workload and workload.completed and (workload.total_requests or 0) > 0
        and (workload.non_success_responses > 0 or workload.application_failures > 0)
        and collection and not collection.errors
        and abstraction and abstraction["system_present"] and abstraction["logs_present"]
        and not any("__" in service for service in abstraction["services"])
        and comparison and recovery and recovery.ready and removed
        and not error and not cleanup_error
    )
    print(json.dumps({
        "status": "PASS_SPARSE_TWIN_TARGETED_TELEMETRY" if passed else "FAIL_SPARSE_TWIN_TARGETED_TELEMETRY",
        "predicted_fault": faults[0].to_dict(),
        "selected_services": spec.services_to_keep,
        "service_reduction_percent": spec.resource_summary.get("service_reduction_percent"),
        "baseline": baseline.to_dict() if baseline else None,
        "manifestation": manifestation.to_dict() if manifestation else None,
        "workload": workload.to_dict() if workload else None,
        "collection": collection.to_dict() if collection else None,
        "abstraction": abstraction,
        "scoped_comparison": comparison,
        "recovery": recovery.to_dict() if recovery else None,
        "namespace_removed": removed,
        "oracle_fault_used": False,
        "error": error,
        "cleanup_error": cleanup_error,
        "next_gate": "live_action_remediation" if passed else "fix_targeted_telemetry",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
