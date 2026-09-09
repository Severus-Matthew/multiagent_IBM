from __future__ import annotations

import copy
import json
import hashlib
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
from .targeted_telemetry import (collect_targeted_telemetry, ObservationWindow,
                                 TelemetryCollectionError, MEASUREMENT_CONTRACT,
                                 discover_prometheus_scrape_interval,
                                 require_phase_window_covers_scrapes)
from .incident_evidence import (reference_state_from_objects, resolve_reference_objects,
                                with_reference_deviations)
from .targeted_workload import WorkloadResult, run_targeted_wrk
from .telemetry_comparator import _service_aliases, compare_symptoms_scoped, score_resolution
from .twin_spec_builder import build_incident_twin_spec


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
    # Debug-only floor; production decisions use current matched controls.
    reproduction_threshold: float = 0.0
    upstream_hops: int = 2
    # One direct hop of runtime dependencies. Larger budgets saturate the
    # dependency graph and deploy essentially the whole application, which
    # destroys the resource reduction the sparse Twin is measured on.
    downstream_support_hops: int = 1
    max_entry_path_hops: int = 8
    artifact_root: str | None = None
    telemetry_settle_seconds: float = 5.0
    require_reward_calibration: bool = True
    calibration_path: str | None = None
    workload_rate: int = 10
    workload_duration_seconds: int = 30


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
        self._incident_key: str | None = None
        self._incident_state: dict[str, Any] | None = None
        self._incident_spec = None
        self._incident_profile = None
        self._incident_template = None
        self._incident_reference: dict[str, Any] = {}
        self._clean_capture: dict[str, Any] | None = None
        self._action_attempt_count = 0
        self.environment_sha256: str | None = None
        self.scrape_interval_seconds: float | None = None

    def begin_trajectory(self, trajectory_id: str | None = None) -> None:
        self.end_trajectory()
        self.trajectory_id = trajectory_id

    def public_agent_state(self, compressed_state: dict[str, Any]) -> dict[str, Any]:
        """Expose the same fault-independent topology prior used by the planner.

        This prevents the policy from seeing four/partial telemetry edges while
        the Twin silently plans with a richer source-derived graph. No label,
        scenario name, predicted fault, or private full state is consulted.
        """
        self.prepare_scenario({}, compressed_state)
        return copy.deepcopy(self._incident_state)

    def prepare_scenario(self, full_state: dict[str, Any], compressed_state: dict[str, Any]) -> None:
        del full_state
        from training_pipeline.agent_input_safety import sanitize_agent_state
        state = sanitize_agent_state(compressed_state)
        key = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        if key == self._incident_key:
            return
        profile = self._profile(state)
        planner_state = self._planner_state(state, profile)
        # Phase-bounded Prometheus range functions need at least two scrapes
        # inside every phase window. Fail closed before any Twin is created.
        self.scrape_interval_seconds = discover_prometheus_scrape_interval()
        require_phase_window_covers_scrapes(self.config.workload_duration_seconds, self.scrape_interval_seconds)
        reference_plan = discover_sparse_manifest_plan(profile.source_namespace, state.get("services") or [])
        from .sparse_live_manifest import _object
        # The plan carries controller/Service summaries, not Kubernetes objects.
        # Resolve every reference so the healthy reference state and the
        # environment fingerprint come from the actual specs.
        reference_objects = resolve_reference_objects(reference_plan, profile.source_namespace, _object)
        source_contract = {
            "objects": sorted([
                {"kind": o["kind"], "name": o["metadata"]["name"], "spec": o.get("spec", {})}
                for o in reference_objects],
                key=lambda o: (o["kind"], o["name"])),
            "configmaps": {name: _object("configmap", name, profile.source_namespace).get("data", {})
                           for name in sorted(reference_plan.configmaps)},
            "payloads": {str(p.relative_to(profile.source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(profile.source_root.rglob("*.lua"))},
            "rate": self.config.workload_rate, "duration": self.config.workload_duration_seconds,
            "prometheus_scrape_interval_seconds": self.scrape_interval_seconds,
            "sla_definition": (state.get("sla") or {}).get("definition"),
            "measurement_contract": MEASUREMENT_CONTRACT,
        }
        self.environment_sha256 = hashlib.sha256(json.dumps(source_contract, sort_keys=True).encode()).hexdigest()
        reference = reference_state_from_objects(reference_objects)
        # Reference capacity/routing and deviations are observable evidence, but
        # they are derived from source objects after the incident state was
        # sanitized; apply the same public-input boundary to them.
        planner_state = sanitize_agent_state(with_reference_deviations(planner_state, reference))
        # This is observable reference capacity/routing, not the private fault.
        planner_state["reference_configuration"] = sanitize_agent_state(reference)
        # Inventory names without a controller in the healthy reference (volumes,
        # container names, Chaos objects) cannot be deployed or be "outside" scope.
        deployable = (set(state.get("services") or [])
                      - set(getattr(reference_plan, "missing_selected_controllers", None) or []))
        spec = build_incident_twin_spec(
            planner_state, deployable_services=deployable,
            upstream_hops=self.config.upstream_hops,
            downstream_support_hops=self.config.downstream_support_hops,
            max_entry_path_hops=self.config.max_entry_path_hops,
        )
        if not spec.services_to_keep or spec.resource_summary.get("invalid_topology"):
            raise RuntimeError("invalid incident Twin topology")
        plan = discover_sparse_manifest_plan(profile.source_namespace, spec.services_to_keep)
        if set(plan.selected_services) != set(spec.services_to_keep):
            raise RuntimeError("manifest discovery changed the incident service scope")
        template = render_sparse_manifest_bundle(plan, "aiops-twin-template", pvc_policy="ephemeral_empty")
        self._incident_key, self._incident_state = key, planner_state
        self._incident_reference, self._incident_spec = reference, spec
        self._incident_profile, self._incident_template = profile, template

    def _incident_targets(self) -> list[FaultLabel]:
        """Affected services on the request graph: they select the phase workloads."""
        return [FaultLabel(service=service, fault_type="unknown")
                for service in self._incident_spec.resource_summary["incident_request_path_targets"]]

    def _trace_observable_targets(self) -> list[FaultLabel]:
        """Affected services the incident's own traces observed; only these can be
        required to appear in a clean/recovered Twin trace export."""
        return [FaultLabel(service=service, fault_type="unknown")
                for service in self._incident_spec.resource_summary["incident_trace_observable_targets"]]

    def _capture_phase(self, phase: str, *, require_trace_coverage: bool) -> dict[str, Any]:
        assert self.session is not None and self.work_root is not None
        from .targeted_telemetry import capture_pod_inventory
        inventory = capture_pod_inventory(self.session)
        started = time.time()
        workload, workloads = self._run_predicted_root_workloads(self._incident_targets())
        window = ObservationWindow(started, time.time(), phase)
        time.sleep(max(0.0, self.config.telemetry_settle_seconds))
        root = self.work_root / (phase + "-" + uuid.uuid4().hex[:8])
        collection = collect_targeted_telemetry(self.session, root, window=window, workload=workload,
                                                 initial_pod_inventory=inventory,
                                                 scrape_interval_seconds=self.scrape_interval_seconds)
        contract = [{"service": w.required_service, "rate": w.requested_rate,
                     "duration_seconds": w.requested_duration_seconds, "payload_sha256": w.payload_sha256,
                     "endpoint": w.endpoint.replace(self.session.namespace, "application-namespace")}
                    for w in workloads]
        collection.resources["workload_contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        collection.resources["workload_healthy"] = bool(workload.completed and not workload.failed
            and not workload.application_failures and (workload.total_requests or 0) > 0)
        collection.resources["observed_request_count"] = workload.total_requests
        collection.resources["reference_environment_sha256"] = self.environment_sha256
        collection.resources["effective_requests_per_second"] = (workload.total_requests or 0) / (window.end_unix - window.start_unix)
        (root / "collection_metadata.json").write_text(json.dumps(collection.to_dict(), indent=2))
        state = self._abstract(root, root.with_name(root.name + "-processed"))
        state = with_reference_deviations(state, self._incident_reference)
        state["collection_quality"] = {"contract": collection.collection_mode,
                                       "channels": collection.channels, "errors": collection.errors}
        channels = self._require_observable_channels(
            state, phase, trace_collection_prevalidated=not require_trace_coverage)
        coverage = (self._require_predicted_roots_observed(state, self._trace_observable_targets(), phase)
                    if require_trace_coverage else {})
        return {"state": state, "workload": workload, "workloads": workloads,
                "channels": channels, "coverage": coverage, "collection": collection.to_dict()}

    def prepare_incident_twin(self, compressed_state: dict[str, Any]) -> None:
        """Create the frozen incident-scope live baseline before the RCA policy runs."""
        if compressed_state is not self._incident_state:
            self.prepare_scenario({}, compressed_state)
        if self.session and not self.handles and self._clean_capture:
            return
        trajectory_id = self.trajectory_id
        self.end_trajectory()
        self.trajectory_id = trajectory_id
        self.runtime_profile = self._incident_profile
        self.selected_services = list(self._incident_spec.services_to_keep)
        self.selected_paths = copy.deepcopy(self._incident_spec.selected_paths)
        namespace = "aiops-twin-" + uuid.uuid4().hex[:12]
        bundle = copy.deepcopy(self._incident_template)
        def rebind(value):
            if isinstance(value, str):
                return value.replace("aiops-twin-template", namespace)
            if isinstance(value, list):
                return [rebind(v) for v in value]
            if isinstance(value, dict):
                return {k: rebind(v) for k, v in value.items()}
            return value
        bundle.target_namespace = namespace
        bundle.objects, bundle.object_refs = rebind(bundle.objects), rebind(bundle.object_refs)
        self.session = SparseLiveTwinSession(bundle)
        if self.config.artifact_root:
            base = Path(self.config.artifact_root).expanduser().resolve()
            self.work_root = base / namespace
            self.work_root.mkdir(parents=True, exist_ok=False)
        else:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="aiops-live-verifier-")
            self.work_root = Path(self.temp_dir.name)
        self.session.create_namespace()
        self.session.apply_manifests()
        baseline = self.session.wait_for_clean_baseline(timeout_seconds=self.config.baseline_timeout_seconds)
        if not baseline.ready:
            raise RuntimeError("incident Twin healthy reference did not stabilize")
        clean = self._capture_phase("clean_baseline", require_trace_coverage=True)
        workload = clean["workload"]
        if (not workload.completed or workload.failed or workload.application_failures
                or (workload.total_requests or 0) <= 0 or (workload.required_ready_endpoints or 0) <= 0):
            raise RuntimeError("incident Twin clean workload failed")
        clean["baseline"] = baseline
        self._clean_capture = clean

    def prepare_action_attempt(self, faults: list[FaultLabel]) -> dict[str, Any]:
        """Every candidate repair starts from the same frozen baseline plus faults."""
        count = self._action_attempt_count
        if count:
            result = self.validate_rca_prediction({}, self._incident_state, faults)
        else:
            result = self.current_rca_gate(faults) or {}
        self._action_attempt_count = count + 1
        if not result.get("rca_twin_verified"):
            raise RuntimeError("fresh fault-state qualification failed before action attempt")
        return result

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
        self._clean_capture = None
        self._action_attempt_count = 0
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
        return {
            **self.last_rca_result,
            "rca_twin_verified": bool(self.last_rca_result.get("rca_twin_verified")),
            "min_reproduction_score": self.last_rca_result.get("decision_threshold", self.config.reproduction_threshold),
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
        if self.config.source_namespace in by_namespace and services.issubset(by_namespace[self.config.source_namespace]):
            overlap, source_namespace = len(services), self.config.source_namespace
        else:
            if len(ranked_ns) > 1 and ranked_ns[0][0] == ranked_ns[1][0]:
                raise RuntimeError("ambiguous healthy reference namespace; configure source_namespace")
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
            rate=self.config.workload_rate, duration_seconds=self.config.workload_duration_seconds,
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
        # Several targets commonly resolve to the same request path; run each
        # distinct payload/endpoint once so phase length does not scale with
        # the number of symptomatic services.
        distinct: dict[tuple[str, str], str] = {}
        for service in services:
            script, endpoint = self._workload(service)
            distinct.setdefault((str(script), str(endpoint)), service)
        rows = [self._run_workload(service) for service in distinct.values()]
        if len(rows) == 1:
            return rows[0], rows

        socket_keys = {key for row in rows for key in row.socket_errors}
        aggregate = WorkloadResult(
            name="twin-wrk2-multiroot",
            endpoint=",".join(dict.fromkeys(row.endpoint for row in rows)),
            completed=all(row.completed for row in rows),
            failed=any(row.failed for row in rows),
            elapsed_seconds=round(sum(row.elapsed_seconds for row in rows), 3),
            requests_per_second=sum(row.total_requests or 0 for row in rows) / max(1e-9, sum(row.elapsed_seconds for row in rows)),
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
        definition = (self._incident_state or {}).get("sla", {}).get("definition")
        sla_args = []
        if definition:
            sla_path = run_dir / "sla_definition.json"
            sla_path.write_text(json.dumps(definition))
            sla_args = ["--sla_config", str(sla_path)]
        proc = subprocess.run(
            [sys.executable, "run_pipeline.py", "--run_dir", str(run_dir),
             "--output_dir", str(output_dir), "--skip_simulator", *sla_args],
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
        quality = state.get("collection_quality") or {}
        channels_now = quality.get("channels") or {}
        if (quality.get("contract") != MEASUREMENT_CONTRACT or quality.get("errors")
                or not all(channels_now.get(k, {}).get("query_succeeded")
                           for k in ("system", "traces", "metrics", "logs"))):
            raise TwinTelemetryIncomplete(phase, "current_phase_collection_status", quality)
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
        del full_state
        if not predicted_faults or any(not fault.is_injectible() for fault in predicted_faults):
            return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                                     "reason": "prediction_not_injectible", "uses_oracle_labels": False})
        reward_calibration = assess_live_reward_calibration(
            predicted_faults, calibration_path=self.config.calibration_path,
            application_state=compressed_state)
        if self.config.require_reward_calibration and not reward_calibration["eligible"]:
            return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                                     "reason": reward_calibration["reason"], "live_reward_calibrated": False,
                                     "reward_calibration": reward_calibration, "uses_oracle_labels": False})
        threshold = float(reward_calibration.get("threshold", self.config.reproduction_threshold))
        try:
            # _incident_state is enriched; reuse the frozen plan when resetting an action.
            if compressed_state is not self._incident_state:
                self.prepare_scenario({}, compressed_state)
            if reward_calibration.get("eligible") and reward_calibration.get("environment_sha256") != self.environment_sha256:
                return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                    "reason": "reference_environment_differs_from_matched_controls",
                    "live_reward_calibrated": False, "uses_oracle_labels": False})
            if not set(f.service for f in predicted_faults).issubset(self._incident_spec.services_to_keep):
                return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                                         "reason": "hypothesis_outside_observed_incident_scope",
                                         "live_reward_calibrated": bool(reward_calibration["eligible"]),
                                         "uses_oracle_labels": False})
            if self.handles:
                trajectory_id = self.trajectory_id
                self.end_trajectory()
                self.trajectory_id = trajectory_id
            if not self._clean_capture:
                # Do not recompute the frozen plan from its enriched public projection.
                self.prepare_incident_twin(compressed_state)
            assert self._clean_capture is not None
            clean = self._clean_capture
            manifestations = []
            for fault in predicted_faults:
                handle = inject_predicted_fault(self.session, fault,
                    application_source_root=str(self.runtime_profile.source_root))
                self.handles.append(handle)
                manifestation = handle.wait_for_manifestation(timeout_seconds=60)
                manifestations.append(manifestation.to_dict())
                if not manifestation.manifested:
                    raise RuntimeError("predicted fault failed to manifest")
            if len(self.handles) > 1:
                # A later mutation must not have undone an earlier component.
                manifestations = [h.wait_for_manifestation(timeout_seconds=60).to_dict() for h in self.handles]
                if not all(m["manifested"] for m in manifestations):
                    raise RuntimeError("joint fault components are not simultaneously manifested")
            capture = self._capture_phase("post_injection", require_trace_coverage=False)
            state, workload = capture["state"], capture["workload"]
            scope = self._incident_spec.services_to_keep
            attributable = self._incident_spec.resource_summary["deployable_services"]
            comparison = compare_symptoms_scoped(self._incident_state, state, scope, target_services=scope,
                                                 attributable_services=attributable)
            clean_comparison = compare_symptoms_scoped(self._incident_state, clean["state"], scope, target_services=scope,
                                                       attributable_services=attributable)
            score = float(comparison["reproduction_score"])
            clean_score = float(clean_comparison["reproduction_score"])
            injection_checked = bool(all(m["manifested"] for m in manifestations)
                                     and (workload.completed or workload.failed)
                                     and workload.execution_started)
            evidence_gate = bool(comparison.get("positive_incident_evidence")
                                 and comparison.get("incident_scope_coverage_complete")
                                 and score > clean_score)
            result = _with_live_route({
                **comparison, "mode": "incident_sparse_live_kubernetes_v2",
                "measurement_contract": MEASUREMENT_CONTRACT,
                "predicted_fault_injection_checked": injection_checked,
                "counterfactual_prediction_replayed": True, "uses_oracle_labels": False,
                "uses_full_state_for_rca_score": False, "uses_hidden_injection_manifest_for_score": False,
                "live_reward_calibrated": bool(reward_calibration["eligible"]),
                "reward_calibration": reward_calibration, "decision_threshold": threshold,
                "uncalibrated_reward_override_used": not reward_calibration["eligible"] and not self.config.require_reward_calibration,
                "clean_reproduction_score": clean_score, "counterfactual_evidence_gate": evidence_gate,
                "rca_twin_verified": bool(injection_checked and evidence_gate and score >= threshold),
                "manifestations": manifestations, "baseline": clean["baseline"].to_dict(),
                "clean_workload": clean["workload"].to_dict(),
                "clean_workloads": [w.to_dict() for w in clean["workloads"]],
                "clean_observable_channels": clean["channels"],
                "clean_predicted_root_trace_coverage": clean["coverage"],
                "workload": workload.to_dict(), "workloads": [w.to_dict() for w in capture["workloads"]],
                "twin_namespace_opaque": True, "services_selected": len(scope),
                "selected_service_names": list(scope),
                "service_reduction_percent": self._incident_spec.resource_summary.get("service_reduction_percent"),
                "scope_policy": self._incident_spec.selection_policy, "selected_paths": self.selected_paths,
                "actionable_fault_resources": self._actionable_fault_resources(),
                "measured_resources": capture["collection"]["resources"],
                "clean_measured_resources": clean["collection"]["resources"],
                "telemetry_artifact_path": str(self.work_root) if self.config.artifact_root else None,
            })
            self.before_state, self.before_workload = state, workload
            self.predicted_faults, self.last_rca_result = list(predicted_faults), result
            return result
        except (TwinTelemetryIncomplete, TelemetryCollectionError) as exc:
            self.end_trajectory()
            return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                                     "predicted_fault_injection_checked": False,
                                     "telemetry_incomplete": True, "reason": str(exc), "uses_oracle_labels": False})
        except Exception as exc:
            self.end_trajectory()
            # Infrastructure failures have no usable counterfactual return.
            return _with_live_route({"reproduction_score": 0.0, "rca_twin_verified": False,
                                     "predicted_fault_injection_checked": False,
                                     "telemetry_incomplete": True,
                                     "reason": f"{type(exc).__name__}: {exc}", "uses_oracle_labels": False})

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
        recovery = None
        try:
            recovery = self.session.wait_for_clean_baseline(timeout_seconds=self.config.baseline_timeout_seconds)
            capture = self._capture_phase("post_remediation", require_trace_coverage=True)
            after_workload, after_workloads = capture["workload"], capture["workloads"]
            after_state, after_channels = capture["state"], capture["channels"]
            after_target_coverage = capture["coverage"]
        except Exception as exc:
            return _with_live_route({"resolved": False, "twin_resolved": False,
                "sla_restored": False, "target_sla_restored": False,
                "telemetry_incomplete": True, "reason": str(exc),
                "execution": execution.to_dict(), "recovery": recovery.to_dict() if recovery else None})
        root = self.work_root
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
        # A repaired structural deviation may have preserved an already healthy
        # SLA. Record satisfaction separately from an actual restoration.
        sla_condition_satisfied = after_sla_healthy
        target_restored = bool(
            recovery.ready
            and after_workload.completed
            and not after_workload.failed
            and (after_workload.total_requests or 0) > 0
            and (after_workload.required_ready_endpoints or 0) > 0
            and after_workload.application_failures == 0
            and resolution.get("resolved", False)
            and sla_condition_satisfied
        )
        global_restored = bool(
            resolution.get("resolved", False) and sla_condition_satisfied
        )
        result = _with_live_route({
            "mode": "sparse_live_kubernetes_action_v1",
            "resolved": target_restored,
            "twin_resolved": target_restored,
            "sla_restored": global_restored and sla_transition_restored,
            "target_sla_restored": target_restored and sla_transition_restored,
            "sla_condition_satisfied": global_restored,
            "target_sla_condition_satisfied": target_restored,
            "sla_restoration_applicable": before_sla_violated,
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
            "repair_validation": "independent_frozen_fault_state_v1",
            "measurement_contract": MEASUREMENT_CONTRACT,
            "verified_commands": list(commands) if target_restored else [],
            "verified_namespace": self.session.namespace,
            "measured_resources": capture["collection"]["resources"],
            "telemetry_incomplete": False,
            "observed_channels": after_channels,
            "telemetry_artifact_path": str(root) if self.config.artifact_root else None,
        })
        if target_restored and (self.last_rca_result or {}).get("live_reward_calibrated"):
            from .repair_transfer import export_verified_repair
            try:
                plan = export_verified_repair(self, commands, result)
                result["verified_repair_plan"] = plan
                if self.config.artifact_root:
                    path = root / "verified_repair_plan.json"
                    path.write_text(json.dumps(plan, indent=2))
                    result["verified_repair_plan_path"] = str(path)
            except ValueError as exc:
                result["repair_export_error"] = str(exc)
        return result
