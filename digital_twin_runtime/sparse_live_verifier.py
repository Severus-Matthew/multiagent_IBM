from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_pipeline.schemas import FaultLabel

from .application_topology import discover_application_topology
from .live_action_executor import execute_twin_commands
from .live_fault_injector import LiveFaultHandle, inject_predicted_fault
from .sparse_live_manifest import discover_sparse_manifest_plan, render_sparse_manifest_bundle
from .sparse_live_session import SparseLiveTwinSession
from .targeted_telemetry import collect_targeted_telemetry
from .targeted_workload import WorkloadResult, run_targeted_wrk
from .telemetry_comparator import compare_symptoms_scoped, score_resolution
from .twin_spec_builder import build_sparse_live_twin_spec


@dataclass(frozen=True)
class SparseLiveVerifierConfig:
    source_namespace: str
    application_source_root: str
    state_abstraction_root: str
    baseline_timeout_seconds: float = 180.0
    reproduction_threshold: float = 0.1
    upstream_hops: int = 2
    downstream_support_hops: int = 2
    max_entry_path_hops: int = 8


class SparseLiveTwinVerifier:
    """Stateful live verifier spanning RCA injection and Action execution.

    One instance may be reused sequentially, but each trajectory receives a new
    opaque namespace. A new RCA validation closes any prior active session, so a
    retry cannot leak live state into the next hypothesis.
    """

    is_live = True

    def __init__(self, config: SparseLiveVerifierConfig) -> None:
        self.config = config
        self.session: SparseLiveTwinSession | None = None
        self.handles: list[LiveFaultHandle] = []
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.before_state: dict[str, Any] | None = None
        self.before_workload: WorkloadResult | None = None
        self.predicted_faults: list[FaultLabel] = []
        self.selected_services: list[str] = []
        self.last_rca_result: dict[str, Any] | None = None

    def begin_trajectory(self, trajectory_id: str | None = None) -> None:
        self.end_trajectory()

    def end_trajectory(self) -> None:
        if self.session and self.session.created:
            try:
                self.session.destroy()
            finally:
                self.session = None
        self.handles = []
        self.before_state = None
        self.before_workload = None
        self.predicted_faults = []
        self.selected_services = []
        self.last_rca_result = None
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def action_namespace(self) -> str | None:
        return self.session.namespace if self.session and self.session.created else None

    def current_rca_gate(self, faults: list[FaultLabel]) -> dict[str, Any] | None:
        if not self.last_rca_result:
            return None
        if [x.injection_key() for x in faults] != [x.injection_key() for x in self.predicted_faults]:
            return None
        score = float(self.last_rca_result.get("reproduction_score", 0.0) or 0.0)
        return {
            **self.last_rca_result,
            "rca_twin_verified": bool(
                self.last_rca_result.get("predicted_fault_injection_checked")
                and score >= self.config.reproduction_threshold
            ),
            "min_reproduction_score": self.config.reproduction_threshold,
            "source": "active_sparse_live_twin_session",
        }

    def _planner_state(self, compressed_state: dict[str, Any]) -> dict[str, Any]:
        state = copy.deepcopy(compressed_state)
        services = [str(x) for x in state.get("services", []) or [] if x]
        topology = discover_application_topology(
            self.config.application_source_root, services
        )
        graph = state.setdefault("graph", {})
        raw_edges = list(graph.get("edges", []) or [])
        seen = {
            (str(row.get("src")), str(row.get("dst")))
            for row in raw_edges if isinstance(row, dict)
        }
        for src, dst in topology.edges:
            if (src, dst) not in seen:
                raw_edges.append({"src": src, "dst": dst, "source": topology.source_mode})
        graph["edges"] = raw_edges
        for entry in topology.entrypoints:
            raw_edges.append({"src": "ROOT", "dst": entry, "source": topology.source_mode})
        return state

    def _workload(self, required_service: str) -> tuple[Path, str]:
        root = Path(self.config.application_source_root)
        # First scientifically audited request-path contract. Unsupported roots
        # fail closed until a source-derived workload catalog is added.
        if required_service == "user-timeline-service" and (root / "wrk2/scripts/social-network/read-user-timeline.lua").exists():
            return (
                root / "wrk2/scripts/social-network/read-user-timeline.lua",
                "http://nginx-thrift:8080/wrk2-api/user-timeline/read",
            )
        raise NotImplementedError(
            f"no audited root-reaching sparse workload for {required_service!r}"
        )

    def _abstract(self, run_dir: Path, output_dir: Path) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--run_dir", str(run_dir),
             "--output_dir", str(output_dir), "--skip_simulator"],
            cwd=self.config.state_abstraction_root,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"state abstraction failed: {proc.stderr[-2000:]}")
        return json.loads((output_dir / "state_abstraction_compressed.json").read_text())

    def validate_rca_prediction(
        self,
        full_state: dict[str, Any],
        compressed_state: dict[str, Any],
        predicted_faults: list[FaultLabel],
    ) -> dict[str, Any]:
        del full_state  # Explicit: private evaluator state is not used here.
        self.end_trajectory()
        if not predicted_faults or any(not fault.is_injectible() for fault in predicted_faults):
            return {
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "reason": "prediction_not_injectible",
                "uses_oracle_labels": False,
            }
        try:
            planner_state = self._planner_state(compressed_state)
            spec = build_sparse_live_twin_spec(
                planner_state, predicted_faults,
                upstream_hops=self.config.upstream_hops,
                downstream_support_hops=self.config.downstream_support_hops,
                max_entry_path_hops=self.config.max_entry_path_hops,
            )
            if not spec.services_to_keep or spec.resource_summary.get("invalid_topology"):
                raise RuntimeError("invalid sparse Twin specification")
            plan = discover_sparse_manifest_plan(
                self.config.source_namespace, spec.services_to_keep
            )
            namespace = "aiops-twin-" + uuid.uuid4().hex[:12]
            self.session = SparseLiveTwinSession(
                render_sparse_manifest_bundle(plan, namespace)
            )
            self.temp_dir = tempfile.TemporaryDirectory(prefix="aiops-live-verifier-")
            self.session.create_namespace()
            self.session.apply_manifests()
            baseline = self.session.wait_for_clean_baseline(
                timeout_seconds=self.config.baseline_timeout_seconds
            )
            if not baseline.ready:
                raise RuntimeError("sparse Twin baseline did not stabilize")
            manifestations = []
            for fault in predicted_faults:
                handle = inject_predicted_fault(self.session, fault)
                self.handles.append(handle)
                manifestation = handle.wait_for_manifestation(timeout_seconds=60)
                manifestations.append(manifestation.to_dict())
                if not manifestation.manifested:
                    raise RuntimeError("predicted fault failed to manifest")
            script, endpoint = self._workload(predicted_faults[0].service)
            workload = run_targeted_wrk(
                self.session, payload_script=script, endpoint=endpoint,
                required_service=predicted_faults[0].service,
            )
            root = Path(self.temp_dir.name)
            collect_targeted_telemetry(self.session, root / "before", workload=workload)
            twin_state = self._abstract(root / "before", root / "before-processed")
            comparison = compare_symptoms_scoped(
                compressed_state, twin_state, spec.services_to_keep
            )
            injection_checked = bool(
                all(row.get("manifested") for row in manifestations)
                and (workload.completed or workload.failed)
                and (workload.total_requests or 0) > 0
                and (
                    workload.application_failures > 0
                    or workload.non_success_responses > 0
                )
            )
            score = float(comparison.get("reproduction_score", 0.0) or 0.0)
            result = {
                **comparison,
                "mode": "sparse_live_kubernetes_v1",
                "predicted_fault_injection_checked": injection_checked,
                "counterfactual_prediction_replayed": True,
                "uses_full_state_for_rca_score": False,
                "uses_oracle_labels": False,
                "uses_hidden_injection_manifest_for_score": False,
                "rca_twin_verified": injection_checked and score >= self.config.reproduction_threshold,
                "manifestations": manifestations,
                "workload": workload.to_dict(),
                "twin_namespace_opaque": True,
                "services_selected": len(spec.services_to_keep),
                "service_reduction_percent": spec.resource_summary.get("service_reduction_percent"),
            }
            self.before_state = twin_state
            self.before_workload = workload
            self.predicted_faults = list(predicted_faults)
            self.selected_services = list(spec.services_to_keep)
            self.last_rca_result = result
            return result
        except Exception as exc:
            self.end_trajectory()
            return {
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "uses_oracle_labels": False,
            }

    def apply_commands_and_score(
        self,
        full_state: dict[str, Any],
        rca_faults: list[FaultLabel],
        mitigation_action: dict[str, Any],
        commands: list[str],
        compressed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del full_state, compressed_state
        if not self.session or not self.before_state or not self.before_workload:
            return {"resolved": False, "reason": "no_active_verified_live_twin"}
        if [x.injection_key() for x in rca_faults] != [x.injection_key() for x in self.predicted_faults]:
            return {"resolved": False, "reason": "action_rca_does_not_match_active_twin"}
        execution = execute_twin_commands(self.session, commands)
        if not execution.executed:
            return {
                "resolved": False, "twin_resolved": False,
                "sla_restored": False, "target_sla_restored": False,
                "action_repairs_fault_type": False,
                "reason": "live_command_execution_failed",
                "execution": execution.to_dict(),
            }
        recovery = self.session.wait_for_clean_baseline(
            timeout_seconds=self.config.baseline_timeout_seconds
        )
        script, endpoint = self._workload(rca_faults[0].service)
        after_workload = run_targeted_wrk(
            self.session, payload_script=script, endpoint=endpoint,
            required_service=rca_faults[0].service,
        )
        assert self.temp_dir is not None
        root = Path(self.temp_dir.name)
        collect_targeted_telemetry(self.session, root / "after", workload=after_workload)
        after_state = self._abstract(root / "after", root / "after-processed")
        resolution = score_resolution(self.before_state, after_state)
        target_restored = bool(
            recovery.ready
            and after_workload.completed
            and not after_workload.failed
            and (after_workload.total_requests or 0) > 0
            and (after_workload.required_ready_endpoints or 0) > 0
            and after_workload.application_failures == 0
        )
        global_restored = bool(resolution.get("resolved", False))
        return {
            "mode": "sparse_live_kubernetes_action_v1",
            "resolved": target_restored,
            "twin_resolved": target_restored,
            "sla_restored": global_restored,
            "target_sla_restored": target_restored,
            "symptom_reduction": resolution.get("symptom_reduction", 0.0),
            "global_symptom_reduction": resolution.get("symptom_reduction", 0.0),
            "target_symptom_reduction": 1.0 if target_restored else 0.0,
            "action_repairs_fault_type": target_restored,
            "reason": "live_target_recovered" if target_restored else "live_target_not_recovered",
            "target_service": mitigation_action.get("service"),
            "execution": execution.to_dict(),
            "recovery": recovery.to_dict(),
            "before_workload": self.before_workload.to_dict(),
            "after_workload": after_workload.to_dict(),
            "resolution": resolution,
        }
