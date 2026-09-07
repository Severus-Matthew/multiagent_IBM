from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_pipeline.schemas import FaultLabel

from .application_topology import discover_application_topology
from .live_action_executor import execute_twin_commands
from .live_capabilities import assess_live_reward_calibration
from .live_fault_injector import LiveFaultHandle, inject_predicted_fault
from .sparse_live_manifest import discover_sparse_manifest_plan, render_sparse_manifest_bundle
from .sparse_live_session import SparseLiveTwinSession
from .targeted_telemetry import collect_targeted_telemetry
from .targeted_workload import WorkloadResult, run_targeted_wrk
from .telemetry_comparator import _service_aliases, compare_symptoms_scoped, score_resolution
from .twin_spec_builder import build_sparse_live_twin_spec


class TwinTelemetryIncomplete(RuntimeError):
    """Raised when a Twin abstraction lacks a channel the comparison scores on."""

    def __init__(self, phase: str, channel: str, observed: dict[str, Any]) -> None:
        super().__init__(
            f"{phase} Twin telemetry is missing the {channel} channel; "
            f"observed {observed}"
        )
        self.phase = phase
        self.channel = channel
        self.observed = observed


def _with_live_route(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("reward_route", "live")
    out.setdefault("live_reward_eligible", True)
    return out


@dataclass(frozen=True)
class SparseLiveVerifierConfig:
    source_namespace: str
    application_source_root: str
    state_abstraction_root: str
    baseline_timeout_seconds: float = 180.0
    # Conservative provisional gate. Reported experiments must replace this
    # with a threshold calibrated from live positive/negative controls.
    # Calibrated 2026-09-02 from four lifecycle-complete matched triplets:
    # min positive=.5400, max wrong-service/mechanism=.4004, midpoint=.4702.
    reproduction_threshold: float = 0.4702
    upstream_hops: int = 2
    # One direct hop of runtime dependencies. Larger budgets saturate the
    # dependency graph and deploy essentially the whole application, which
    # destroys the resource reduction the sparse Twin is measured on.
    downstream_support_hops: int = 1
    max_entry_path_hops: int = 8
    artifact_root: str | None = None
    telemetry_settle_seconds: float = 5.0
    require_reward_calibration: bool = True


@dataclass(frozen=True)
class _RuntimeProfile:
    name: str
    source_namespace: str
    source_root: Path
    payload_script: Path | None
    endpoint: str
    frontend_service: str
    frontend_container: str | None
    frontend_port: int
    discovery: dict[str, Any]


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
        self.work_root: Path | None = None
        self.trajectory_id: str | None = None
        self.before_state: dict[str, Any] | None = None
        self.before_workload: WorkloadResult | None = None
        self.predicted_faults: list[FaultLabel] = []
        self.selected_services: list[str] = []
        self.selected_paths: list[list[str]] = []
        self.last_rca_result: dict[str, Any] | None = None
        self.runtime_profile: _RuntimeProfile | None = None

    def begin_trajectory(self, trajectory_id: str | None = None) -> None:
        self.end_trajectory()
        self.trajectory_id = trajectory_id

    def public_agent_state(self, compressed_state: dict[str, Any]) -> dict[str, Any]:
        """Expose the same fault-independent topology prior used by the planner.

        This prevents the policy from seeing four/partial telemetry edges while
        the Twin silently plans with a richer source-derived graph. No label,
        scenario name, predicted fault, or private full state is consulted.
        """
        profile = self._profile(compressed_state)
        return self._planner_state(compressed_state, profile)

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
        self.selected_paths = []
        self.last_rca_result = None
        self.runtime_profile = None
        self.work_root = None
        self.trajectory_id = None
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def action_namespace(self) -> str | None:
        return self.session.namespace if self.session and self.session.created else None

    def _actionable_fault_resources(self) -> list[dict[str, str]]:
        """Public resources created solely from the agent's own prediction."""
        rows: list[dict[str, str]] = []
        for handle in self.handles:
            if handle.restore_mode != "delete_custom_resource":
                continue
            kind = str(handle.injected_details.get("kind") or "").strip().lower()
            name = str(handle.injected_details.get("name") or "").strip()
            if kind and name:
                rows.append({
                    "kind": kind,
                    "name": name,
                    "remediation": "delete_exact_resource",
                })
        return rows

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
                and (
                    self.last_rca_result.get("live_reward_calibrated")
                    or not self.config.require_reward_calibration
                )
                and score >= self.config.reproduction_threshold
            ),
            "min_reproduction_score": self.config.reproduction_threshold,
            "source": "active_sparse_live_twin_session",
        }

    def _profile(self, compressed_state: dict[str, Any]) -> _RuntimeProfile:
        services = {str(x) for x in compressed_state.get("services", []) or [] if x}
        configured_root = Path(self.config.application_source_root).expanduser().resolve()
        proc = subprocess.run(
            ["kubectl", "get", "deployments,statefulsets,services", "-A", "-o", "json"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"runtime profile discovery failed: {proc.stderr.strip()}")
        items = json.loads(proc.stdout or "{}").get("items", []) or []
        by_namespace: dict[str, set[str]] = {}
        service_objects: dict[tuple[str, str], dict[str, Any]] = {}
        for obj in items:
            meta = obj.get("metadata", {}) or {}
            ns, name = str(meta.get("namespace") or ""), str(meta.get("name") or "")
            if not ns or ns.startswith("aiops-twin-") or not name:
                continue
            by_namespace.setdefault(ns, set()).add(name)
            if obj.get("kind") == "Service":
                service_objects[(ns, name)] = obj
        ranked_ns = sorted(
            ((len(services & names), ns) for ns, names in by_namespace.items()),
            key=lambda row: (-row[0], row[1]),
        )
        if not ranked_ns or ranked_ns[0][0] == 0:
            raise NotImplementedError("no source namespace overlaps observable services")
        overlap, source_namespace = ranked_ns[0]

        parent = configured_root.parent
        roots = [configured_root] + ([p for p in parent.iterdir() if p.is_dir()] if parent.exists() else [])
        root_rows: list[tuple[int, int, str, Path, Any]] = []
        for root in {p.resolve() for p in roots}:
            topology = discover_application_topology(root, services)
            covered = {x for edge in topology.edges for x in edge} | set(topology.entrypoints)
            root_rows.append((len(covered & services), topology.source_files_scanned, str(root), root, topology))
        _, _, _, root, topology = sorted(root_rows, key=lambda row: (-row[0], -row[1], row[2]))[0]

        entrypoints = [x for x in topology.entrypoints if (source_namespace, x) in service_objects]
        if not entrypoints:
            # Bundle-derived fallback: choose a Service with a port that is not a
            # known datastore and that has no incoming observable dependency.
            incoming = {dst for _, dst in topology.edges}
            candidates = sorted(
                s for s in services
                if (source_namespace, s) in service_objects and s not in incoming
            )
            entrypoints = candidates
        if not entrypoints:
            raise NotImplementedError("no request entrypoint discovered from source topology or Services")
        frontend = entrypoints[0]
        svc = service_objects[(source_namespace, frontend)]
        ports = (svc.get("spec", {}) or {}).get("ports", []) or []
        if not ports:
            raise NotImplementedError(f"entrypoint Service {frontend} exposes no port")
        port = int(ports[0].get("port") or 0)
        if port <= 0:
            raise NotImplementedError(f"entrypoint Service {frontend} has invalid port")

        scripts = sorted(root.glob("wrk2/**/*.lua"))
        mixed = [p for p in scripts if "mixed" in p.name.lower()]
        payload = (mixed or scripts or [None])[0]
        return _RuntimeProfile(
            name=root.name,
            source_namespace=source_namespace,
            source_root=root,
            payload_script=payload,
            endpoint=f"http://{frontend}:{port}",
            frontend_service=frontend,
            frontend_container=None,
            frontend_port=port,
            discovery={
                "mode": "observable_bundle_and_cluster_discovery_v1",
                "namespace_overlap": overlap,
                "observable_service_count": len(services),
                "source_topology_mode": topology.source_mode,
                "payload": str(payload) if payload else "generated_http_get",
            },
        )

    def _planner_state(self, compressed_state: dict[str, Any], profile: _RuntimeProfile) -> dict[str, Any]:
        state = copy.deepcopy(compressed_state)
        services = [str(x) for x in state.get("services", []) or [] if x]
        topology = discover_application_topology(
            profile.source_root, services
        )
        graph = state.setdefault("graph", {})
        raw_edges = list(graph.get("edges", []) or [])
        seen = {
            (str(row.get("src")), str(row.get("dst")))
            for row in raw_edges if isinstance(row, dict)
        }
        startup = set(topology.startup_edges)
        for src, dst in topology.edges:
            if (src, dst) not in seen:
                raw_edges.append({
                    "src": src, "dst": dst, "source": topology.source_mode,
                    "startup_required": (src, dst) in startup,
                })
            elif (src, dst) in startup:
                for row in raw_edges:
                    if isinstance(row, dict) and (str(row.get("src")), str(row.get("dst"))) == (src, dst):
                        row["startup_required"] = True
        graph["edges"] = raw_edges
        for entry in topology.entrypoints:
            raw_edges.append({"src": "ROOT", "dst": entry, "source": topology.source_mode})
        return state

    def _workload(self, required_service: str) -> tuple[Path, str]:
        if not self.runtime_profile:
            raise NotImplementedError("runtime profile has not been discovered")
        script = self.runtime_profile.payload_script
        # Prefer an existing, purpose-built wrk2 workload script over one
        # auto-generated from raw handler source: a pre-built script (e.g.
        # DeathStarBench's compose-post.lua) constructs realistic field values
        # (a real username/user_id range, JSON-array media_ids, etc.), while
        # the handler-derived fallback below fills every captured field with a
        # meaningless placeholder ("field=1", "field=2", ...) — which silently
        # fails application-side validation on any endpoint that checks field
        # shape/identity (e.g. compose-post's user_id), producing a workload
        # that runs, returns non-2xx on every request, and never actually
        # reaches the target service — yielding zero trace edges and, via
        # _require_observable_channels' fail-closed check, a Twin that can
        # never verify a prediction for that service at all. Confirmed via a
        # direct reproduction for gen_assign_to_non_existent_node_social_net-
        # mitigation-unique-id-service-default: the auto-generated script sent
        # "media_ids=1&media_types=2&post_type=3&text=4&user_id=5&username=6"
        # to /wrk2-api/post/compose, got a login-page redirect on all 117
        # requests, and the clean baseline's trace collection came back with
        # collected_trace_edges: 0, failing every one of that scenario's RCA
        # attempts (across every task-phase variant in the dataset) with
        # "twin_telemetry_incomplete:traces" before a single fault was ever
        # injected.
        candidates = sorted(self.runtime_profile.source_root.glob("wrk2/**/*.lua"))
        path_tokens = {
            token.lower().replace("-service", "").replace("-", "_")
            for path in self.selected_paths for token in path
        }
        target_token = required_service.lower().replace("-service", "").replace("-", "_")
        path_tokens.add(target_token)
        scored: list[tuple[int, str, Path]] = []
        for candidate in candidates:
            try:
                body = candidate.read_text(errors="ignore").lower().replace("-", "_")
            except OSError:
                continue
            score = sum(1 for token in path_tokens if token and token in body)
            scored.append((score, str(candidate), candidate))
        prebuilt_match = (
            sorted(scored, key=lambda row: (-row[0], row[1]))[0][2]
            if scored and max(row[0] for row in scored) > 0 else None
        )
        if prebuilt_match is not None:
            script = prebuilt_match
        else:
            # No purpose-built script references this service/path — fall back
            # to deriving a request from the raw API handler source. This is
            # last-resort and known to misfire on endpoints that validate
            # field values rather than merely field presence (see above).
            token_variants = {
                required_service,
                required_service.replace("-service", ""),
                required_service.replace("-", "_"),
            }
            handlers = sorted(self.runtime_profile.source_root.rglob("lua-scripts/wrk2-api/**/*.lua"))
            matching_handlers: list[Path] = []
            for handler in handlers:
                try:
                    body = handler.read_text(errors="ignore")
                except OSError:
                    continue
                if any(token and token in body for token in token_variants):
                    matching_handlers.append(handler)
            if matching_handlers:
                handler = matching_handlers[0]
                body = handler.read_text(errors="ignore")
                marker = "/lua-scripts/"
                normalized = str(handler).replace("\\", "/")
                route = "/" + normalized.split(marker, 1)[1].removesuffix(".lua")
                form_fields = sorted(set(re.findall(r"\bpost\.([A-Za-z_][A-Za-z0-9_]*)", body)))
                query_fields = sorted(set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", body)))
                if self.work_root is None:
                    raise RuntimeError("work root is unavailable for generated targeted workload")
                script = self.work_root / "generated-selected-path.lua"
                if form_fields:
                    pairs = [f"{name}={index + 1}" for index, name in enumerate(form_fields)]
                    script.write_text(
                        'wrk.method = "POST"\n'
                        f'wrk.path = "{route}"\n'
                        'wrk.headers["Content-Type"] = "application/x-www-form-urlencoded"\n'
                        f'wrk.body = "{"&".join(pairs)}"\n'
                    )
                else:
                    pairs = [f"{name}={index + 1}" for index, name in enumerate(query_fields)]
                    suffix = ("?" + "&".join(pairs)) if pairs else ""
                    script.write_text(f'wrk.method = "GET"\nwrk.path = "{route}{suffix}"\n')
        if script is None:
            if self.work_root is None:
                raise RuntimeError("work root is unavailable for generated workload")
            script = self.work_root / "generated-targeted-get.lua"
            script.write_text('wrk.method = "GET"\nwrk.path = "/"\n')
        if not script.exists():
            raise NotImplementedError("discovered workload script does not exist")
        return script, self.runtime_profile.endpoint

    def _run_workload(self, required_service: str) -> WorkloadResult:
        assert self.session is not None and self.runtime_profile is not None
        script, endpoint = self._workload(required_service)
        return run_targeted_wrk(
            self.session, payload_script=script, endpoint=endpoint,
            required_service=required_service,
            frontend_service=self.runtime_profile.frontend_service,
            frontend_container=self.runtime_profile.frontend_container,
            frontend_port=self.runtime_profile.frontend_port,
        )

    def _run_predicted_root_workloads(
        self, faults: list[FaultLabel]
    ) -> tuple[WorkloadResult, list[WorkloadResult]]:
        """Exercise every predicted root path, not only the first multifault root."""
        services = list(dict.fromkeys(
            str(fault.service) for fault in faults if str(fault.service).strip()
        ))
        if not services:
            raise ValueError("no predicted root services are available for workload execution")
        rows = [self._run_workload(service) for service in services]
        if len(rows) == 1:
            return rows[0], rows

        socket_keys = {key for row in rows for key in row.socket_errors}
        aggregate = WorkloadResult(
            name="twin-wrk2-multiroot",
            endpoint=",".join(dict.fromkeys(row.endpoint for row in rows)),
            completed=all(row.completed for row in rows),
            failed=any(row.failed for row in rows),
            elapsed_seconds=round(sum(row.elapsed_seconds for row in rows), 3),
            requests_per_second=sum(row.requests_per_second or 0.0 for row in rows),
            total_requests=sum(row.total_requests or 0 for row in rows),
            non_success_responses=sum(row.non_success_responses for row in rows),
            application_failures=sum(row.application_failures for row in rows),
            probe_http_status=next(
                (row.probe_http_status for row in rows if row.probe_http_status and row.probe_http_status >= 400),
                rows[-1].probe_http_status,
            ),
            probe_body="\n\n".join(row.probe_body for row in rows if row.probe_body)[:4000],
            required_service=",".join(services),
            required_ready_endpoints=min(
                (row.required_ready_endpoints or 0) for row in rows
            ),
            socket_errors={
                key: sum(row.socket_errors.get(key, 0) for row in rows)
                for key in sorted(socket_keys)
            },
            output="\n\n".join(row.output for row in rows),
            execution_started=all(row.execution_started for row in rows),
            scope_policy="union_of_all_predicted_root_request_paths",
        )
        return aggregate, rows

    @staticmethod
    def _require_predicted_roots_observed(
        state: dict[str, Any], faults: list[FaultLabel], phase: str
    ) -> dict[str, Any]:
        """Require clean/recovered traces to prove every predicted root was exercised."""
        traces = state.get("traces") or {}
        per_edge = traces.get("per_edge", {}) if isinstance(traces, dict) else {}
        endpoints: set[str] = set()
        if isinstance(per_edge, dict):
            for edge_id, features in per_edge.items():
                features = features if isinstance(features, dict) else {}
                src = features.get("source")
                dst = features.get("target")
                if (not src or not dst) and "->" in str(edge_id):
                    src, dst = str(edge_id).split("->", 1)
                if src:
                    endpoints.add(str(src))
                if dst:
                    endpoints.add(str(dst))
        targets = list(dict.fromkeys(str(fault.service) for fault in faults))
        known_services = {
            str(service).strip().lower()
            for service in (state.get("services") or [])
            if str(service).strip()
        }
        observed_normalized = {
            str(service).strip().lower() for service in endpoints
        }

        def observed_target(target: str) -> bool:
            target_normalized = target.strip().lower()
            if target_normalized in observed_normalized:
                return True
            # Alias fallback is needed for provider-specific Mongo names, but an
            # alias that is itself another declared service (e.g. ``rate`` vs
            # ``mongodb-rate``) must not satisfy coverage for the wrong target.
            aliases = {
                alias.strip().lower() for alias in _service_aliases(target)
            }
            aliases -= known_services - {target_normalized}
            return bool(aliases & observed_normalized)

        missing = [
            target for target in targets
            if not observed_target(target)
        ]
        report = {
            "predicted_root_services": targets,
            "trace_endpoint_services": sorted(endpoints),
            "missing_predicted_root_services": missing,
        }
        if missing:
            raise TwinTelemetryIncomplete(phase, "predicted_root_trace_coverage", report)
        return report

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

    @staticmethod
    def _require_observable_channels(
        state: dict[str, Any], phase: str, *, trace_collection_prevalidated: bool = False
    ) -> dict[str, Any]:
        """Reject a Twin abstraction that is missing the channels we score on.

        The abstraction pipeline does not fail when telemetry is absent: a run with
        no trace export silently yields ``traces: {}`` and ``graph.edges: []``.
        Scoring that against the incident makes the reproduction score a function
        of collection races rather than of the counterfactual, and inside a GRPO
        group two trajectories with the same hypothesis then receive different
        advantages. Fail closed and label the failure so it is attributable to
        collection rather than to a wrong prediction.
        """
        traces = state.get("traces") or {}
        system = state.get("system") or {}
        per_edge = traces.get("per_edge", {}) if isinstance(traces, dict) else {}
        trace_summary = traces.get("summary", {}) if isinstance(traces, dict) else {}
        collected_trace_edges = len(per_edge) if isinstance(per_edge, dict) else 0
        collected_trace_edges = max(
            collected_trace_edges,
            int(trace_summary.get("num_edges", 0) or 0) if isinstance(trace_summary, dict) else 0,
        )
        graph_edges = ((state.get("graph") or {}).get("edges") or [])
        channels = {
            "collected_trace_edges": collected_trace_edges,
            "graph_edges_planning_only": len(graph_edges),
            "system_services": len(system),
        }
        if not system:
            raise TwinTelemetryIncomplete(phase, "system", channels)
        if collected_trace_edges <= 0 and not trace_collection_prevalidated:
            raise TwinTelemetryIncomplete(phase, "traces", channels)
        channels["trace_collection_prevalidated"] = trace_collection_prevalidated
        channels["zero_trace_edges_is_observed_result"] = bool(
            trace_collection_prevalidated and collected_trace_edges <= 0
        )
        return channels

    def validate_rca_prediction(
        self,
        full_state: dict[str, Any],
        compressed_state: dict[str, Any],
        predicted_faults: list[FaultLabel],
    ) -> dict[str, Any]:
        del full_state  # Explicit: private evaluator state is not used here.
        trajectory_id = self.trajectory_id
        self.end_trajectory()
        self.trajectory_id = trajectory_id
        if not predicted_faults or any(not fault.is_injectible() for fault in predicted_faults):
            return _with_live_route({
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "reason": "prediction_not_injectible",
                "uses_oracle_labels": False,
            })
        reward_calibration = assess_live_reward_calibration(predicted_faults)
        if (
            reward_calibration["eligible"]
            and abs(
                float(self.config.reproduction_threshold)
                - float(reward_calibration["threshold"])
            ) > 1e-9
        ):
            reward_calibration = {
                **reward_calibration,
                "eligible": False,
                "reason": "configured_reproduction_threshold_not_calibrated",
                "configured_threshold": float(self.config.reproduction_threshold),
            }
        if self.config.require_reward_calibration and not reward_calibration["eligible"]:
            return _with_live_route({
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "reason": str(reward_calibration["reason"]),
                "live_reward_calibrated": False,
                "reward_calibration": reward_calibration,
                "uses_oracle_labels": False,
            })
        try:
            profile = self._profile(compressed_state)
            self.runtime_profile = profile
            planner_state = self._planner_state(compressed_state, profile)
            spec = build_sparse_live_twin_spec(
                planner_state, predicted_faults,
                upstream_hops=self.config.upstream_hops,
                downstream_support_hops=self.config.downstream_support_hops,
                max_entry_path_hops=self.config.max_entry_path_hops,
            )
            self.selected_paths = [list(path) for path in spec.selected_paths]
            if not spec.services_to_keep or spec.resource_summary.get("invalid_topology"):
                raise RuntimeError("invalid sparse Twin specification")
            plan = discover_sparse_manifest_plan(
                profile.source_namespace, spec.services_to_keep
            )
            # Manifest rendering resolves ConfigMaps/Secrets/PVCs for the
            # planner-selected workloads, but must never widen workload scope.
            # Shared service registries often name the full application and are
            # not evidence that every named service is required. Fail closed if
            # this boundary regresses.
            if set(plan.selected_services) != set(spec.services_to_keep):
                raise RuntimeError(
                    "sparse manifest discovery changed the causal service scope: "
                    f"planned={sorted(spec.services_to_keep)} "
                    f"rendered={sorted(plan.selected_services)}"
                )
            namespace = "aiops-twin-" + uuid.uuid4().hex[:12]
            self.session = SparseLiveTwinSession(
                render_sparse_manifest_bundle(plan, namespace, pvc_policy="ephemeral_empty")
            )
            if self.config.artifact_root:
                base = Path(self.config.artifact_root).expanduser().resolve()
                base.mkdir(parents=True, exist_ok=True)
                safe_id = "".join(c if c.isalnum() or c in "-_" else "_"
                                  for c in (self.trajectory_id or "trajectory"))
                self.work_root = base / f"{safe_id}-{uuid.uuid4().hex[:8]}"
                self.work_root.mkdir(parents=True)
            else:
                self.temp_dir = tempfile.TemporaryDirectory(prefix="aiops-live-verifier-")
                self.work_root = Path(self.temp_dir.name)
            self.session.create_namespace()
            self.session.apply_manifests()
            baseline = self.session.wait_for_clean_baseline(
                timeout_seconds=self.config.baseline_timeout_seconds
            )
            if not baseline.ready:
                raise RuntimeError("sparse Twin baseline did not stabilize")
            assert self.work_root is not None
            root = self.work_root
            clean_workload, clean_workloads = self._run_predicted_root_workloads(
                predicted_faults
            )
            if (
                not clean_workload.completed or clean_workload.failed
                or (clean_workload.total_requests or 0) <= 0
                or clean_workload.application_failures > 0
                or (clean_workload.required_ready_endpoints or 0) <= 0
            ):
                raise RuntimeError("sparse Twin clean workload did not exercise the selected path")
            time.sleep(max(0.0, self.config.telemetry_settle_seconds))
            collect_targeted_telemetry(self.session, root / "clean", workload=clean_workload)
            clean_state = self._abstract(root / "clean", root / "clean-processed")
            clean_channels = self._require_observable_channels(clean_state, "clean_baseline")
            clean_target_coverage = self._require_predicted_roots_observed(
                clean_state, predicted_faults, "clean_baseline"
            )
            manifestations = []
            for fault in predicted_faults:
                handle = inject_predicted_fault(
                    self.session, fault,
                    application_source_root=str(profile.source_root),
                )
                self.handles.append(handle)
                manifestation = handle.wait_for_manifestation(timeout_seconds=60)
                manifestations.append(manifestation.to_dict())
                if not manifestation.manifested:
                    raise RuntimeError("predicted fault failed to manifest")
            workload, workloads = self._run_predicted_root_workloads(predicted_faults)
            time.sleep(max(0.0, self.config.telemetry_settle_seconds))
            collect_targeted_telemetry(self.session, root / "before", workload=workload)
            twin_state = self._abstract(root / "before", root / "before-processed")
            twin_channels = self._require_observable_channels(
                twin_state, "post_injection", trace_collection_prevalidated=True
            )
            comparison = compare_symptoms_scoped(
                compressed_state, twin_state, spec.services_to_keep,
                target_services=[fault.service for fault in predicted_faults],
            )
            comparison["twin_observable_channels"] = twin_channels
            injection_checked = bool(
                all(row.get("manifested") for row in manifestations)
                and (workload.completed or workload.failed)
                and (
                    (workload.total_requests or 0) > 0
                    or bool(workload.output.strip())
                    or bool(workload.socket_errors)
                    or workload.execution_started
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
                "live_reward_calibrated": bool(reward_calibration["eligible"]),
                "reward_calibration": reward_calibration,
                "uncalibrated_reward_override_used": bool(
                    not reward_calibration["eligible"]
                    and not self.config.require_reward_calibration
                ),
                "rca_twin_verified": bool(
                    injection_checked
                    and (
                        reward_calibration["eligible"]
                        or not self.config.require_reward_calibration
                    )
                    and score >= self.config.reproduction_threshold
                ),
                "manifestations": manifestations,
                "baseline": baseline.to_dict(),
                "clean_workload": clean_workload.to_dict(),
                "clean_workloads": [row.to_dict() for row in clean_workloads],
                "clean_observable_channels": clean_channels,
                "clean_predicted_root_trace_coverage": clean_target_coverage,
                "workload": workload.to_dict(),
                "workloads": [row.to_dict() for row in workloads],
                "twin_namespace_opaque": True,
                "services_selected": len(spec.services_to_keep),
                "service_reduction_percent": spec.resource_summary.get("service_reduction_percent"),
                "selected_paths": spec.selected_paths,
                "actionable_fault_resources": self._actionable_fault_resources(),
                "runtime_profile_discovery": profile.discovery,
                "telemetry_artifact_path": str(root) if self.config.artifact_root else None,
            }
            result = _with_live_route(result)
            self.before_state = twin_state
            self.before_workload = workload
            self.predicted_faults = list(predicted_faults)
            self.selected_services = list(spec.services_to_keep)
            self.last_rca_result = result
            return result
        except TwinTelemetryIncomplete as exc:
            self.end_trajectory()
            # Distinct from a wrong hypothesis: the Twin ran but the observation
            # channel needed to judge it was not collected.
            return _with_live_route({
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "telemetry_incomplete": True,
                "missing_channel": exc.channel,
                "observed_channels": exc.observed,
                "reason": f"twin_telemetry_incomplete:{exc.channel}",
                "uses_oracle_labels": False,
            })
        except Exception as exc:
            self.end_trajectory()
            return _with_live_route({
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.0,
                "predicted_fault_injection_checked": False,
                "rca_twin_verified": False,
                "telemetry_incomplete": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "uses_oracle_labels": False,
            })

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
            return _with_live_route({"resolved": False, "reason": "no_active_verified_live_twin"})
        if [x.injection_key() for x in rca_faults] != [x.injection_key() for x in self.predicted_faults]:
            return _with_live_route({"resolved": False, "reason": "action_rca_does_not_match_active_twin"})
        execution = execute_twin_commands(
            self.session,
            commands,
            owned_runtime_objects=self._actionable_fault_resources(),
        )
        if not execution.executed:
            return _with_live_route({
                "resolved": False, "twin_resolved": False,
                "sla_restored": False, "target_sla_restored": False,
                "action_repairs_fault_type": False,
                "reason": "live_command_execution_failed",
                "execution": execution.to_dict(),
            })
        recovery = self.session.wait_for_clean_baseline(
            timeout_seconds=self.config.baseline_timeout_seconds
        )
        after_workload, after_workloads = self._run_predicted_root_workloads(
            rca_faults
        )
        time.sleep(max(0.0, self.config.telemetry_settle_seconds))
        assert self.work_root is not None
        root = self.work_root
        collect_targeted_telemetry(self.session, root / "after", workload=after_workload)
        after_state = self._abstract(root / "after", root / "after-processed")
        try:
            after_channels = self._require_observable_channels(after_state, "post_remediation")
            after_target_coverage = self._require_predicted_roots_observed(
                after_state, rca_faults, "post_remediation"
            )
        except TwinTelemetryIncomplete as exc:
            # An uncollected after-state is indistinguishable from a fully healed
            # one by symptom counting, so recovery credit must not be granted.
            return _with_live_route({
                "mode": "sparse_live_kubernetes_action_v1",
                "resolved": False, "twin_resolved": False,
                "sla_restored": False, "target_sla_restored": False,
                "symptom_reduction": 0.0, "global_symptom_reduction": 0.0,
                "target_symptom_reduction": 0.0,
                "action_repairs_fault_type": False,
                "telemetry_incomplete": True,
                "missing_channel": exc.channel,
                "observed_channels": exc.observed,
                "reason": f"twin_telemetry_incomplete:{exc.channel}",
                "execution": execution.to_dict(),
                "recovery": recovery.to_dict(),
            })
        resolution = score_resolution(self.before_state, after_state)
        before_sla = self.before_state.get("sla", {}) or {}
        after_sla = after_state.get("sla", {}) or {}
        before_sla_violated = bool(
            before_sla.get("violated")
            or not (before_sla.get("global_sla", {}) or {}).get("healthy", True)
        )
        after_sla_healthy = bool(
            not after_sla.get("violated")
            and (after_sla.get("global_sla", {}) or {}).get("healthy", False)
        )
        sla_transition_restored = bool(before_sla_violated and after_sla_healthy)
        target_restored = bool(
            recovery.ready
            and after_workload.completed
            and not after_workload.failed
            and (after_workload.total_requests or 0) > 0
            and (after_workload.required_ready_endpoints or 0) > 0
            and after_workload.application_failures == 0
            and resolution.get("resolved", False)
            and sla_transition_restored
        )
        global_restored = bool(
            resolution.get("resolved", False) and sla_transition_restored
        )
        return _with_live_route({
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
            "after_workloads": [row.to_dict() for row in after_workloads],
            "after_predicted_root_trace_coverage": after_target_coverage,
            "resolution": resolution,
            "before_sla": before_sla,
            "after_sla": after_sla,
            "sla_transition_restored": sla_transition_restored,
            "telemetry_incomplete": False,
            "observed_channels": after_channels,
            "telemetry_artifact_path": str(root) if self.config.artifact_root else None,
        })
