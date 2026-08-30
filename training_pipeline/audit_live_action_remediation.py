from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from digital_twin_runtime.live_action_executor import execute_twin_commands
from digital_twin_runtime.live_fault_injector import inject_predicted_fault
from digital_twin_runtime.sparse_live_manifest import discover_sparse_manifest_plan, render_sparse_manifest_bundle
from digital_twin_runtime.sparse_live_session import SparseLiveTwinSession
from digital_twin_runtime.targeted_telemetry import collect_targeted_telemetry
from digital_twin_runtime.targeted_workload import run_targeted_wrk
from digital_twin_runtime.telemetry_comparator import score_resolution
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.action_reward import action_reward
from training_pipeline.audit_sparse_live_twin_spec import _augment_with_static_topology, _default_application_source_root, _find_scenario
from training_pipeline.command_safety import check_command_safety
from training_pipeline.fixed_action_agent import FixedActionAgent
from training_pipeline.schemas import parse_fault_lines


def _exists(namespace: str) -> bool:
    return subprocess.run(
        ["kubectl", "get", "namespace", namespace, "-o", "name"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _abstract(run_dir: Path, output_dir: Path) -> dict:
    pipeline = subprocess.run(
        [sys.executable, "run_pipeline.py", "--run_dir", str(run_dir),
         "--output_dir", str(output_dir), "--skip_simulator"],
        cwd=Path(__file__).resolve().parents[1] / "state_abstraction_full",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if pipeline.returncode != 0:
        raise RuntimeError(f"state abstraction failed: {pipeline.stderr[-2000:]}")
    return json.loads((output_dir / "state_abstraction_compressed.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a safety-checked Action Agent remediation on the live sparse Twin.")
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
    fault = faults[0]
    rec = _find_scenario(args.processed_states, args.scenario_id)
    app_root = _default_application_source_root(args.processed_states)
    planner_state, _ = _augment_with_static_topology(rec.compressed_state, Path(app_root))
    spec = build_sparse_live_twin_spec(planner_state, faults, downstream_support_hops=2)
    plan = discover_sparse_manifest_plan(args.source_namespace, spec.services_to_keep)
    session = SparseLiveTwinSession(render_sparse_manifest_bundle(plan, args.target_namespace))

    commands: list[str] = []
    execution = before_workload = after_workload = recovery = None
    resolution = reward = None
    error = cleanup_error = None
    with tempfile.TemporaryDirectory(prefix="aiops-live-action-") as temp_dir:
        temp = Path(temp_dir)
        try:
            session.create_namespace()
            session.apply_manifests()
            baseline = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
            if not baseline.ready:
                raise RuntimeError("clean baseline did not stabilize")
            handle = inject_predicted_fault(session, fault)
            manifestation = handle.wait_for_manifestation(timeout_seconds=60)
            if not manifestation.manifested:
                raise RuntimeError("predicted fault did not manifest")
            before_workload = run_targeted_wrk(
                session,
                payload_script=Path(app_root) / "wrk2/scripts/social-network/read-user-timeline.lua",
                endpoint="http://nginx-thrift:8080/wrk2-api/user-timeline/read",
                required_service=fault.service,
            )
            collect_targeted_telemetry(session, temp / "before", workload=before_workload)
            before_state = _abstract(temp / "before", temp / "before-processed")

            action_plan = {
                "contract": "ACTION_PLAN_V1",
                "namespace": session.namespace,
                "plans": [{
                    "service": fault.service,
                    "fault_type": fault.fault_type,
                    "fault_mechanism": fault.fault_mechanism,
                    "action_family": (
                        "scale_service" if fault.fault_mechanism == "scale_replicas_zero"
                        else "repair_service_port" if fault.fault_mechanism == "target_port_misconfig"
                        else "infra_patch_first"
                    ),
                }],
            }
            instruction = "ACTION_PLAN_JSON:\n" + json.dumps(action_plan)
            commands = FixedActionAgent().get_commands(instruction, {"namespace": session.namespace})
            execution = execute_twin_commands(session, commands)
            if not execution.executed:
                raise RuntimeError(f"live Action commands failed: {execution.to_dict()}")
            recovery = session.wait_for_clean_baseline(timeout_seconds=args.timeout_seconds)
            if not recovery.ready:
                raise RuntimeError("Action did not restore stable controller health")
            after_workload = run_targeted_wrk(
                session,
                payload_script=Path(app_root) / "wrk2/scripts/social-network/read-user-timeline.lua",
                endpoint="http://nginx-thrift:8080/wrk2-api/user-timeline/read",
                required_service=fault.service,
            )
            collect_targeted_telemetry(session, temp / "after", workload=after_workload)
            after_state = _abstract(temp / "after", temp / "after-processed")
            resolution = score_resolution(before_state, after_state)
            endpoint_restored = (after_workload.required_ready_endpoints or 0) > 0
            workload_restored = bool(
                after_workload.completed
                and not after_workload.failed
                and (after_workload.total_requests or 0) > 0
                and after_workload.application_failures == 0
            )
            verifier = {
                "resolved": endpoint_restored and workload_restored,
                "twin_resolved": endpoint_restored and workload_restored,
                "sla_restored": bool(resolution.get("resolved", False)),
                "target_sla_restored": endpoint_restored and workload_restored,
                "target_symptom_reduction": 1.0 if endpoint_restored else 0.0,
                "global_symptom_reduction": resolution.get("symptom_reduction", 0.0),
                "action_repairs_fault_type": endpoint_restored,
                "reason": "live_sparse_twin_execution",
            }
            reward = action_reward(commands, check_command_safety(commands), verifier)
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
        execution and execution.safe and execution.executed and recovery and recovery.ready
        and before_workload and (before_workload.completed or before_workload.failed)
        and (before_workload.total_requests or 0) > 0
        and (
            before_workload.application_failures > 0
            or before_workload.non_success_responses > 0
        )
        and after_workload and after_workload.completed and not after_workload.failed
        and (after_workload.total_requests or 0) > 0
        and after_workload.application_failures == 0
        and (after_workload.required_ready_endpoints or 0) > 0
        and reward and reward["success"] and removed and not error and not cleanup_error
    )
    print(json.dumps({
        "status": "PASS_LIVE_ACTION_REMEDIATION" if passed else "FAIL_LIVE_ACTION_REMEDIATION",
        "predicted_fault": fault.to_dict(),
        "commands": commands,
        "execution": execution.to_dict() if execution else None,
        "before_workload": before_workload.to_dict() if before_workload else None,
        "after_workload": after_workload.to_dict() if after_workload else None,
        "recovery": recovery.to_dict() if recovery else None,
        "resolution": resolution,
        "action_reward": reward,
        "namespace_removed": removed,
        "oracle_action_used": False,
        "error": error,
        "cleanup_error": cleanup_error,
        "next_gate": "live_verifier_rollout_integration" if passed else "fix_live_action_remediation",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
