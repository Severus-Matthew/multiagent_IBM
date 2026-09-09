"""Recorder orchestration with the cluster, generator and collector mocked."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset_generation.record_incident_capture import (
    PRIVATE_FILES, RecorderConfig, SourceSession, fingerprint_drift, injection_evidence, record_scenario, run_phase,
    select_workloads, source_fingerprint, spec_targets,
)
from digital_twin_runtime.targeted_telemetry import MEASUREMENT_CONTRACT


def spec(**over):
    base = {"problem_id": "gen_network_delay_hotel_res-detection-frontend-default", "fault_family": "network_delay_hotel_res",
            "task": "detection", "faulty_service": "frontend", "mode": "override_service", "app": "hotel",
            "is_multifault": False, "scenario_spec_sha256": "abc"}
    base.update(over)
    return base


class FakeGenerator:
    def write_json(self, path, obj):
        Path(path).write_text(json.dumps(obj))

    def save_ground_truth(self, problem, spec, out_dir):
        Path(out_dir, "ground_truth.json").write_text(json.dumps({"answer": "Yes", "faulty_service": problem.faulty_service}))

    def save_fault_timing(self, out_dir, inject_time, collect_time):
        Path(out_dir, "fault_timing.json").write_text(json.dumps({"fault_injected_at": inject_time, "telemetry_collected_at": collect_time}))


class FakeProblem:
    def __init__(self, journal, events, manifested=True):
        self.faulty_service = "frontend"; self.journal = journal; self.events = events; self.manifested = manifested

    def get_task_description(self):
        return "desc"

    def inject_fault(self):
        self.events.append("inject")
        self.journal.append({"mechanism": "network_delay", "service": "frontend", "applied": True, "manifested": self.manifested})

    def recover_fault(self):
        self.events.append("recover")


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        self.cfg = RecorderConfig(aiopslab_root=root, output_dir=root / "out", state_abstraction_root=root,
                                  manifestation_settle_seconds=0, telemetry_settle_seconds=0, recovery_timeout_seconds=1)
        self.session = SourceSession("test-hotel-reservation", [
            {"kind": "Deployment", "metadata": {"name": "frontend"}, "spec": {"template": {"metadata": {"labels": {"app": "frontend"}}}}},
            {"kind": "Service", "metadata": {"name": "frontend"}, "spec": {"selector": {"app": "frontend"}, "ports": [{"port": 5000}]}}])
        self.verifier = SimpleNamespace(runtime_profile=SimpleNamespace(frontend_service="frontend", frontend_container=None, frontend_port=5000),
                                        _workload=lambda service: (root / "payload.lua", "http://frontend:5000"))
        (root / "payload.lua").write_text("wrk.method = 'GET'")
        self.events = []; self.journal = []

    def tearDown(self):
        self.temp.cleanup()

    def phase_runner(self, session, profile, workloads, phase, out_dir, *, scrape_interval, cfg):
        self.events.append(phase); out_dir.mkdir(parents=True)
        (out_dir / "collection_metadata.json").write_text("{}")
        return {"phase": phase, "window": {"duration_seconds": 160}, "workload_healthy": True,
                "workload_contract": [], "collection": {"errors": []}}

    def abstractor(self, phase_dir, output_dir, cfg):
        self.events.append("abstract:" + phase_dir.name); output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "state_abstraction_compressed.json").write_text("{}")
        return {"ok": True}

    def _record(self, manifested=True, clean_after=True, s=None):
        problem = FakeProblem(self.journal, self.events, manifested=manifested)
        return record_scenario(s or spec(), cfg=self.cfg, generator=FakeGenerator(), problem_factory=lambda: problem,
                               session=self.session, verifier=self.verifier, journal=self.journal, scrape_interval=60,
                               is_clean=lambda ns: (True, {"reason": "clean"}),
                               wait_clean=lambda ns, t: (clean_after, {"reason": "clean" if clean_after else "chaos_objects_present"}),
                               phase_runner=self.phase_runner, abstractor=self.abstractor, sleep=lambda s_: None)

    def test_phase_order_private_files_and_acceptance(self):
        row = self._record()
        self.assertEqual(self.events, ["clean", "inject", "incident", "recover", "recovered", "abstract:incident", "abstract:clean", "abstract:recovered"])
        scenario = self.cfg.output_dir / "raw" / spec()["problem_id"]
        for name in PRIVATE_FILES:
            self.assertTrue((scenario / name).is_file(), name)
            self.assertTrue((scenario / "incident" / name).is_file(), name)
        self.assertFalse((scenario / "clean" / "spec.json").exists())
        capture = json.loads((scenario / "recorder_capture.json").read_text())
        self.assertEqual(capture["measurement_contract"], MEASUREMENT_CONTRACT)
        self.assertEqual(set(capture["phases"]), {"clean", "incident", "recovered"})
        self.assertTrue(row["accepted"] and row["injection_verified"] and row["recovered_clean"])
        self.assertTrue((self.cfg.output_dir / "processed_states" / spec()["problem_id"] / "state_abstraction_compressed.json").is_file())

    def test_unmanifested_injection_is_recorded_but_not_accepted(self):
        row = self._record(manifested=False)
        self.assertFalse(row["injection_verified"]); self.assertFalse(row["accepted"])
        evidence = json.loads((self.cfg.output_dir / "raw" / spec()["problem_id"] / "injection_evidence.json").read_text())
        self.assertEqual(evidence["format"], "source_injection_evidence_v1"); self.assertFalse(evidence["verified"])

    def test_incident_phase_failure_still_recovers_the_source(self):
        def failing_runner(session, profile, workloads, phase, out_dir, *, scrape_interval, cfg):
            self.events.append(phase); out_dir.mkdir(parents=True)
            if phase == "incident":
                raise RuntimeError("telemetry collection failed: short window")
            return {"phase": phase, "window": {}, "workload_healthy": True, "workload_contract": [], "collection": {}}
        problem = FakeProblem(self.journal, self.events)
        with self.assertRaisesRegex(RuntimeError, "short window"):
            record_scenario(spec(), cfg=self.cfg, generator=FakeGenerator(), problem_factory=lambda: problem,
                            session=self.session, verifier=self.verifier, journal=self.journal, scrape_interval=60,
                            is_clean=lambda ns: (True, {}), wait_clean=lambda ns, t: (True, {"reason": "clean"}),
                            phase_runner=failing_runner, abstractor=self.abstractor, sleep=lambda s_: None)
        self.assertEqual(self.events, ["clean", "inject", "incident", "recover"])

    def test_injection_that_raises_after_a_partial_mutation_is_still_recovered(self):
        problem = FakeProblem(self.journal, self.events)
        def partial_inject():
            self.events.append("inject"); self.journal.append({"mechanism": "network_delay", "service": "frontend", "applied": True, "manifested": False})
            raise RuntimeError("chaos object created but manifestation wait failed")
        problem.inject_fault = partial_inject
        with self.assertRaisesRegex(RuntimeError, "manifestation wait failed"):
            record_scenario(spec(), cfg=self.cfg, generator=FakeGenerator(), problem_factory=lambda: problem,
                            session=self.session, verifier=self.verifier, journal=self.journal, scrape_interval=60,
                            is_clean=lambda ns: (True, {}), wait_clean=lambda ns, t: (True, {"reason": "clean"}),
                            phase_runner=self.phase_runner, abstractor=self.abstractor, sleep=lambda s_: None)
        self.assertEqual(self.events, ["clean", "inject", "recover"])

    def test_failed_recovery_is_fatal_after_recover_attempt(self):
        with self.assertRaisesRegex(RuntimeError, "clean state"):
            self._record(clean_after=False)
        self.assertIn("recover", self.events)

    def test_refuses_existing_capture_and_unsupported_specs(self):
        (self.cfg.output_dir / "raw" / spec()["problem_id"]).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self._record()
        with self.assertRaisesRegex(ValueError, "multifault"):
            spec_targets(spec(is_multifault=True, subproblems=[]))
        with self.assertRaisesRegex(ValueError, "no faulty service"):
            spec_targets(spec(faulty_service=None, mode="constructor_app"))

    def test_unrecovered_spec_mutation_is_fatal_even_when_pods_are_ready(self):
        objects = [{"kind": "Service", "metadata": {"name": "user-service", "resourceVersion": "1"},
                    "spec": {"ports": [{"port": 9090, "targetPort": 9090}]}},
                   {"kind": "Deployment", "metadata": {"name": "user-service"}, "spec": {"replicas": 1}}]
        reference = source_fingerprint(objects)
        same = [dict(o, metadata={**o["metadata"], "resourceVersion": "2"}) for o in objects]
        self.assertEqual(fingerprint_drift(reference, source_fingerprint(same)), {"changed": [], "missing": [], "added": []})
        faulted = json.loads(json.dumps(objects)); faulted[0]["spec"]["ports"][0]["targetPort"] = 65534
        self.assertEqual(fingerprint_drift(reference, source_fingerprint(faulted))["changed"], ["Service/user-service"])
        state = {"objects": objects}
        def load(ns):
            return state["objects"]
        problem = FakeProblem(self.journal, self.events)
        def leaky_recover():
            self.events.append("recover"); state["objects"] = faulted
        problem.recover_fault = leaky_recover
        with self.assertRaisesRegex(RuntimeError, "spec drift"):
            record_scenario(spec(), cfg=self.cfg, generator=FakeGenerator(), problem_factory=lambda: problem,
                            session=self.session, verifier=self.verifier, journal=self.journal, scrape_interval=60,
                            is_clean=lambda ns: (True, {}), wait_clean=lambda ns, t: (True, {"reason": "clean"}),
                            phase_runner=self.phase_runner, abstractor=self.abstractor, sleep=lambda s_: None,
                            reference_fingerprint=reference, load_objects=load)
        self.assertIn("recover", self.events)

    def test_evidence_rule_matches_regeneration(self):
        rows = [{"mechanism": "m", "service": "a", "applied": True, "manifested": True},
                {"mechanism": "m", "service": "b", "applied": True, "manifested": False}]
        self.assertFalse(injection_evidence("p", rows)["verified"])
        self.assertTrue(injection_evidence("p", rows[:1])["verified"])
        self.assertFalse(injection_evidence("p", [])["verified"])

    def test_workloads_deduplicate_like_the_twin(self):
        loads = select_workloads(self.verifier, ["frontend", "geo"])
        self.assertEqual(len(loads), 1); self.assertEqual(loads[0]["service"], "frontend"); self.assertEqual(len(loads[0]["payload_sha256"]), 64)

    def test_run_phase_orders_inventory_workload_window_settle_collect(self):
        calls = []; clock = iter([100.0, 110.0, 250.0])  # workload died after 10s; window held to 150s
        result = SimpleNamespace(completed=True, failed=False, application_failures=0, non_success_responses=0, total_requests=10,
                                 to_dict=lambda: {"total_requests": 10})
        def wrk(session, **kw):
            calls.append(("wrk", kw["duration_seconds"], kw["rate"])); return result
        def collect(session, out_dir, *, window, workload, initial_pod_inventory, scrape_interval_seconds):
            calls.append(("collect", window.phase, round(window.end_unix - window.start_unix), initial_pod_inventory, scrape_interval_seconds))
            return SimpleNamespace(to_dict=lambda: {"errors": []})
        out = Path(self.temp.name) / "phase"; out.mkdir()
        record = run_phase(self.session, self.verifier.runtime_profile, select_workloads(self.verifier, ["frontend"]), "clean", out,
                           scrape_interval=60, cfg=self.cfg, wrk=wrk, collect=collect, inventory=lambda s: {"frontend-1": {}},
                           sleep=lambda s_: calls.append(("sleep", s_)), clock=lambda: next(clock))
        self.assertEqual(calls, [("wrk", 150, 10), ("sleep", 140.0), ("sleep", 0), ("collect", "clean", 150, {"frontend-1": {}}, 60)])
        self.assertTrue(record["workload_healthy"]); self.assertEqual(record["measurement_contract"], MEASUREMENT_CONTRACT)
        self.assertTrue((out / "phase.json").is_file())


if __name__ == "__main__":
    unittest.main()


class PilotControlTests(unittest.TestCase):
    def test_controls_are_label_derived_and_bounded(self):
        from training_pipeline.pilot_score_separation import pilot_controls
        from training_pipeline.schemas import FaultLabel
        label = FaultLabel(service="frontend", fault_type="latency_degradation", fault_mechanism="network_delay")
        controls = dict(pilot_controls([label], ["frontend", "geo"], ["frontend", "geo", "rate"]))
        self.assertEqual(set(controls), {"positive", "wrong_service", "wrong_mechanism"})
        self.assertEqual(controls["wrong_service"][0].service, "geo")
        # single request-path target: prefer a service the incident's traces observed over the alphabetical scope
        only = dict(pilot_controls([label], ["frontend"], ["consul", "frontend", "search"], trace_endpoints=["ROOT", "frontend", "search"]))
        self.assertEqual(only["wrong_service"][0].service, "search")
        self.assertEqual(controls["wrong_mechanism"][0].service, "frontend")
        self.assertNotEqual(controls["wrong_mechanism"][0].fault_mechanism, "network_delay")
        self.assertEqual(controls["positive"], [label])


class SourceHealthTests(unittest.TestCase):
    def test_service_left_at_fault_target_port_is_misrouted(self):
        from dataset_generation.warm_cluster import misrouted_services
        pods = [{"metadata": {"labels": {"service": "user-service"}},
                 "spec": {"containers": [{"ports": [{"containerPort": 9090, "name": "thrift"}]}]}}]
        healthy = [{"metadata": {"name": "user-service"}, "spec": {"selector": {"service": "user-service"}, "ports": [{"port": 9090, "targetPort": 9090}]}}]
        faulted = [{"metadata": {"name": "user-service"}, "spec": {"selector": {"service": "user-service"}, "ports": [{"port": 9090, "targetPort": 65534}]}}]
        named = [{"metadata": {"name": "user-service"}, "spec": {"selector": {"service": "user-service"}, "ports": [{"port": 9090, "targetPort": "thrift"}]}}]
        headless = [{"metadata": {"name": "x"}, "spec": {"ports": [{"port": 1, "targetPort": 65534}]}}]
        declared_mismatch = [{"metadata": {"name": "media-frontend"}, "spec": {"selector": {"service": "media-frontend"}, "ports": [{"port": 8081, "targetPort": 8080}]}}]
        self.assertEqual(misrouted_services(healthy, pods), {})
        self.assertEqual(misrouted_services(named, pods), {})
        self.assertEqual(misrouted_services(declared_mismatch, pods), {})  # declared containerPorts are not ground truth
        self.assertEqual(list(misrouted_services(headless, pods)), ["x"])
        self.assertEqual(list(misrouted_services(faulted, pods)), ["user-service"])
        self.assertEqual(misrouted_services(faulted, pods)["user-service"][0]["targetPort"], 65534)

    def test_target_port_injection_refuses_an_already_faulted_source(self):
        from dataset_generation import injector_fixes
        faulted = {"metadata": {"name": "user-service"}, "spec": {"ports": [{"port": 9090, "targetPort": 65534}]}}
        with patch.object(injector_fixes, "_service", return_value=faulted), patch.object(injector_fixes, "_apply") as apply:
            with self.assertRaisesRegex(injector_fixes.MutationDiscoveryError, "already_misconfigured"):
                injector_fixes._patched_misconfig_k8s(SimpleNamespace(namespace="ns"), ["user-service"])
        apply.assert_not_called()

    def test_target_port_snapshot_is_not_aliased_by_the_mutation(self):
        from dataset_generation import injector_fixes
        healthy = {"metadata": {"name": "user-service", "resourceVersion": "1"}, "spec": {"ports": [{"port": 9090, "targetPort": 9090}]}}
        applied = []
        state = {"svc": healthy}
        def service(ns, name):
            return json.loads(json.dumps(state["svc"]))
        def apply(obj, operation):
            applied.append((operation, obj["spec"]["ports"][0]["targetPort"])); state["svc"] = obj
        injector = SimpleNamespace(namespace="ns")
        with patch.object(injector_fixes, "_service", side_effect=service), patch.object(injector_fixes, "_apply", side_effect=apply), \
             patch.object(injector_fixes, "_wait_for", return_value=True):
            injector_fixes._patched_misconfig_k8s(injector, ["user-service"])
            self.assertEqual(state["svc"]["spec"]["ports"][0]["targetPort"], 65534)
            injector_fixes._patched_recover_misconfig_k8s(injector, ["user-service"])
        self.assertEqual([port for _, port in applied], [65534, 9090])
        self.assertEqual(state["svc"]["spec"]["ports"][0]["targetPort"], 9090)

    def test_target_port_recovery_never_silently_skips(self):
        from dataset_generation import injector_fixes
        injector_fixes._ORIGINAL_SERVICES.pop(("ns", "user-service"), None)
        with self.assertRaisesRegex(RuntimeError, "no recorded original"):
            injector_fixes._patched_recover_misconfig_k8s(SimpleNamespace(namespace="ns"), ["user-service"])
