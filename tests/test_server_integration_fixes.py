"""Regressions found integrating the 2026-09-08 audit on the training host.

All Kubernetes/Prometheus calls are mocked. See training_pipeline/VALIDATION.md.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from digital_twin_runtime.incident_evidence import reference_state_from_objects, resolve_reference_objects
from digital_twin_runtime.telemetry_comparator import canonical_service, compare_symptoms_scoped
from digital_twin_runtime.twin_spec_builder import build_incident_twin_spec
from digital_twin_runtime.targeted_telemetry import (
    MIN_SCRAPES_PER_PHASE, ObservationWindow, TelemetryCollectionError, _prometheus_rows,
    collect_targeted_telemetry, discover_prometheus_scrape_interval, minimum_phase_window_seconds,
    parse_prometheus_duration, require_phase_window_covers_scrapes, scrape_interval_from_config_yaml,
)
from digital_twin_runtime.targeted_workload import WORKLOAD_WAIT_MARGIN_SECONDS, workload_timeout_seconds
from training_pipeline.audit_hf_exact_token_sampler import TinyTokenizer, _build_model
from training_pipeline.hf_exact_token_sampler import (
    ExactTokenGenerationConfig, HFExactTokenPolicySampler, _generation_defaults_owner,
)
from training_pipeline.peft_adapter_control import ROLE_ADAPTERS


def _object(kind, name, namespace):
    if kind == "Service":
        return {"kind": "Service", "metadata": {"name": name, "namespace": namespace},
                "spec": {"selector": {"app": name}, "ports": [{"port": 9090, "targetPort": 9090}]}}
    return {"kind": kind, "metadata": {"name": name, "namespace": namespace},
            "spec": {"replicas": 2, "template": {"spec": {"containers": [{"name": name, "image": name + ":1"}]}}}}


class ReferenceObjectResolutionTests(unittest.TestCase):
    def plan(self):
        return SimpleNamespace(
            controllers=[{"kind": "Deployment", "name": "api", "logical_service": "api", "pod_labels": {"app": "api"}},
                         {"kind": "StatefulSet", "name": "db", "logical_service": "db", "pod_labels": {"app": "db"}}],
            service_objects=[{"name": "api", "type": "ClusterIP", "selector": {"app": "api"}, "ports": []}])

    def test_summaries_are_resolved_with_explicit_kinds(self):
        calls = []
        def fetch(kind, name, namespace):
            calls.append((kind, name, namespace)); return _object(kind, name, namespace)
        objects = resolve_reference_objects(self.plan(), "reference", fetch)
        self.assertEqual(calls, [("Deployment", "api", "reference"), ("StatefulSet", "db", "reference"),
                                 ("Service", "api", "reference")])
        reference = reference_state_from_objects(objects)["system"]
        self.assertEqual(reference["api"]["deployment"]["replicas_desired"], 2)
        self.assertEqual(reference["api"]["service"]["ports"][0]["target_port"], 9090)
        self.assertEqual(reference["db"]["deployment"]["containers"][0]["image"], "db:1")

    def test_summary_rows_themselves_are_rejected(self):
        # The pre-fix code indexed metadata on these rows and crashed on every live incident.
        with self.assertRaises(KeyError):
            reference_state_from_objects_like_prefix(self.plan())
        with self.assertRaisesRegex(ValueError, "complete Kubernetes object"):
            resolve_reference_objects(self.plan(), "reference", lambda kind, name, ns: {"kind": kind, "name": name})
        with self.assertRaisesRegex(ValueError, "complete Kubernetes object"):
            resolve_reference_objects(self.plan(), "reference",
                                      lambda kind, name, ns: {**_object(kind, name, ns), "metadata": {"name": "other"}})
        with self.assertRaisesRegex(ValueError, "kind/name"):
            resolve_reference_objects(SimpleNamespace(controllers=[{"kind": "Job", "name": "x"}], service_objects=[]),
                                      "reference", _object)


def reference_state_from_objects_like_prefix(plan):
    return [{"kind": o["kind"], "name": o["metadata"]["name"]} for o in plan.controllers + plan.service_objects]


HOTEL = ["consul", "frontend", "geo", "jaeger", "jaeger-out", "mongodb-geo", "mongodb-profile", "profile",
         "profile-db", "rate", "search", "user"]


class SymptomAttributionTests(unittest.TestCase):
    def test_container_and_log_names_resolve_to_services(self):
        self.assertEqual(canonical_service("hotel-reserv-geo", HOTEL), "geo")
        self.assertEqual(canonical_service("hotel-reserv-geo-mongo", HOTEL), "mongodb-geo")
        self.assertEqual(canonical_service("hotel-reserv-profile-mongo", HOTEL), "mongodb-profile")
        self.assertEqual(canonical_service("mongodb-geo", HOTEL), "mongodb-geo")
        self.assertEqual(canonical_service("profile-db", HOTEL), "profile-db")
        for artifact in ("unknown", "container-kill", "delay", "utils_mongodb", ""):
            self.assertIsNone(canonical_service(artifact, HOTEL), artifact)

    def incident(self, degraded=("geo",), log_names=("hotel-reserv-geo-mongo", "unknown", "profile-db"), failed_trace=True):
        system = {s: {"health": {"pods_total": 1, "pods_ready": 0 if s in degraded else 1,
                                 "pods_unready": int(s in degraded)},
                      "deployment": {"replicas_desired": 1}} for s in HOTEL}
        system["container-kill"] = {"health": {"pods_total": 1, "pods_ready": 0, "pods_unready": 1}}
        return {"services": HOTEL, "system": system,
                "logs": {name: {"signal": {"error_count": 3}} for name in log_names},
                "graph": {"edges": [["ROOT", "frontend"], ["frontend", "geo"], ["geo", "mongodb-geo"],
                                    ["frontend", "profile"], {"src": "profile", "dst": "mongodb-profile", "startup_required": True}]},
                "traces": {"per_edge": {"frontend->geo": {"source": "frontend", "target": "geo",
                                                          "error_ratio": 1.0 if failed_trace else 0.0}}}}

    def test_incident_scope_resolves_aliases_and_keeps_off_graph_symptoms(self):
        deployable = [s for s in HOTEL if s not in ("profile-db", "jaeger-out")]
        spec = build_incident_twin_spec(self.incident(), deployable_services=deployable)
        summary = spec.resource_summary
        self.assertEqual(summary["incident_request_path_targets"], ["frontend", "geo", "mongodb-geo"])
        self.assertEqual(summary["incident_trace_observable_targets"], ["frontend", "geo"])
        self.assertTrue({"frontend", "geo", "mongodb-geo"}.issubset(spec.services_to_keep))
        self.assertIn("container-kill", summary["unattributed_symptom_names"])
        self.assertIn("unknown", summary["unattributed_symptom_names"])
        self.assertIn("profile-db", summary["undeployable_inventory_names"])
        self.assertNotIn("profile-db", spec.services_to_keep)
        self.assertFalse(spec.target_faults)

    def test_incident_without_request_path_symptoms_fails_closed(self):
        state = self.incident(degraded=(), log_names=("unknown",), failed_trace=False)
        with self.assertRaisesRegex(ValueError, "no observable"):
            build_incident_twin_spec(state, deployable_services=HOTEL)

    def test_unattributed_names_do_not_zero_the_comparison(self):
        state = self.incident()
        scope = ["frontend", "geo", "mongodb-geo"]
        attributable = [s for s in HOTEL if s not in ("profile-db", "jaeger-out")]
        result = compare_symptoms_scoped(state, state, scope, target_services=scope, attributable_services=attributable)
        self.assertTrue(result["incident_scope_coverage_complete"], result)
        self.assertGreater(result["reproduction_score"], 0)
        self.assertIn("unknown", result["unattributed_original_symptom_names"]["top_error_services"])
        # A deployable service with symptoms that is missing from scope still rejects the comparison.
        wider = self.incident(degraded=("geo", "profile"))
        rejected = compare_symptoms_scoped(wider, wider, scope, target_services=scope, attributable_services=attributable)
        self.assertFalse(rejected["incident_scope_coverage_complete"])
        self.assertEqual(rejected["reproduction_score"], 0)


class WorkloadTimeoutTests(unittest.TestCase):
    def test_wait_covers_the_requested_phase(self):
        # A 150s phase at a 1m scrape cadence was abandoned by the old fixed 60s wait.
        self.assertEqual(workload_timeout_seconds(150), 150 + WORKLOAD_WAIT_MARGIN_SECONDS)
        self.assertEqual(workload_timeout_seconds(30, 90), 90)
        with self.assertRaisesRegex(ValueError, "shorter"):
            workload_timeout_seconds(150, 60)


class ScrapeIntervalTests(unittest.TestCase):
    def test_duration_parsing(self):
        self.assertEqual(parse_prometheus_duration("1m"), 60)
        self.assertEqual(parse_prometheus_duration("15s"), 15)
        self.assertEqual(parse_prometheus_duration("1m30s"), 90)
        self.assertEqual(parse_prometheus_duration("500ms"), 0.5)
        for bad in ("", "1", "abc", "1x"):
            with self.assertRaises(ValueError):
                parse_prometheus_duration(bad)

    def test_global_interval_is_read_not_job_intervals(self):
        text = "global:\n  scrape_interval: 1m\n  evaluation_interval: 1m\nscrape_configs:\n- job_name: fast\n  scrape_interval: 5s\n"
        self.assertEqual(scrape_interval_from_config_yaml(text), 60)
        self.assertEqual(scrape_interval_from_config_yaml("scrape_configs:\n- job_name: x\n  scrape_interval: 5s\n"), 60)
        self.assertEqual(scrape_interval_from_config_yaml("global:\n  scrape_interval: '15s'\n"), 15)
        payload = {"status": "success", "data": {"yaml": text}}
        with patch("digital_twin_runtime.targeted_telemetry._read", return_value=json.dumps(payload)) as read:
            self.assertEqual(discover_prometheus_scrape_interval(), 60)
        self.assertIn("/api/v1/status/config", read.call_args.args[0][-1])
        with patch("digital_twin_runtime.targeted_telemetry._read", return_value='{"status":"error"}'):
            with self.assertRaises(RuntimeError):
                discover_prometheus_scrape_interval()

    def test_phase_window_must_contain_two_scrapes(self):
        self.assertEqual(minimum_phase_window_seconds(60), 2 * 60 + 5)
        with self.assertRaisesRegex(ValueError, "twin_workload_duration_seconds"):
            require_phase_window_covers_scrapes(30, 60)
        require_phase_window_covers_scrapes(150, 60)
        require_phase_window_covers_scrapes(40, 15)
        with patch("digital_twin_runtime.targeted_telemetry._read") as read:
            with self.assertRaises(ValueError):
                _prometheus_rows("app", window=ObservationWindow(100, 140, "faulted"), scrape_interval_seconds=60)
        read.assert_not_called()  # a window that cannot hold two scrapes is never widened or queried


class CoverageAccountingTests(unittest.TestCase):
    """Short-of-samples pods invalidate resource measurement without failing the reward channel."""

    window = ObservationWindow(100, 140, "post_injection")

    def session(self):
        objects = [{"kind": "Deployment", "metadata": {"name": "a"},
                    "spec": {"template": {"metadata": {"labels": {"app": "a"}}}}}]
        return SimpleNamespace(namespace="twin", bundle=SimpleNamespace(objects=objects, object_refs=[{"kind": "Deployment", "name": "a"}]))

    def kubectl(self, args):
        if args[0] == "get" and args[1] == "pods":
            return json.dumps({"items": [{"metadata": {"name": "a-1", "uid": "u1", "labels": {"app": "a"}},
                                          "status": {"phase": "Running", "containerStatuses": [{"containerID": "c1", "state": {"running": {"startedAt": "1970-01-01T00:00:00Z"}}}]}}]})
        if args[0] == "get" and args[-1] == "json":
            return json.dumps({"items": []})
        if args[0] == "logs":
            return "1970-01-01T00:02:00Z inside\n"
        raise AssertionError(args)

    def collect(self, samples, temp):
        rows = [{"timestamp": 140, "cmdb_id": "a-1", "kpi_name": "container_cpu_usage_cores", "value": 0.25},
                {"timestamp": 140, "cmdb_id": "a-1", "kpi_name": "container_memory_working_set_bytes", "value": 1024.0}]
        inventory = {"a-1": {"uid": "u1", "containers": ["c1"]}}
        with patch("digital_twin_runtime.targeted_telemetry._read", side_effect=self.kubectl), \
             patch("digital_twin_runtime.targeted_telemetry._prometheus_rows", return_value=rows), \
             patch("digital_twin_runtime.targeted_telemetry._jaeger_rows", return_value=[]), \
             patch("digital_twin_runtime.targeted_telemetry._prometheus_sample_coverage", return_value={"a-1": samples}):
            return collect_targeted_telemetry(self.session(), Path(temp) / "phase", window=self.window,
                                              initial_pod_inventory=inventory, scrape_interval_seconds=15)

    def test_insufficient_samples_invalidate_resources_only(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.collect(MIN_SCRAPES_PER_PHASE - 1, temp)
        self.assertEqual(result.errors, [])
        self.assertTrue(result.channels["metrics"]["query_succeeded"])
        self.assertFalse(result.resources["valid"])
        self.assertIn("insufficient_metric_samples_in_phase", result.resources["invalid_reason"])
        self.assertEqual(result.resources["metric_sample_coverage"]["insufficient_pods"], ["a-1"])
        self.assertEqual(result.window["scrape_interval_seconds"], 15)

    def test_covered_phase_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.collect(MIN_SCRAPES_PER_PHASE, temp)
        self.assertTrue(result.resources["valid"], result.resources)
        self.assertEqual(result.resources["application_cpu_cores_mean"], 0.25)

    def test_failed_coverage_query_is_a_metrics_channel_failure(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch("digital_twin_runtime.targeted_telemetry._read", side_effect=self.kubectl), \
             patch("digital_twin_runtime.targeted_telemetry._prometheus_rows", return_value=[]), \
             patch("digital_twin_runtime.targeted_telemetry._jaeger_rows", return_value=[]), \
             patch("digital_twin_runtime.targeted_telemetry._prometheus_sample_coverage", side_effect=RuntimeError("down")):
            with self.assertRaises(TelemetryCollectionError) as caught:
                collect_targeted_telemetry(self.session(), Path(temp) / "phase", window=self.window, scrape_interval_seconds=15)
        self.assertFalse(caught.exception.result.channels["metrics"]["query_succeeded"])


class InstalledTransformersSamplingTests(unittest.TestCase):
    def test_both_adapters_sample_raw_logits_and_restore_inherited_config(self):
        model = _build_model()
        owner = _generation_defaults_owner(model)
        owner.generation_config.top_k = 1
        owner.generation_config.top_p = 0.1
        owner.generation_config.repetition_penalty = 2.0
        owner.generation_config.suppress_tokens = [0, 3]
        sampler = HFExactTokenPolicySampler(model, TinyTokenizer(), config=ExactTokenGenerationConfig(max_new_tokens=2), device="cpu")
        original_generate = model.generate
        for adapter in ROLE_ADAPTERS:
            observed = []
            def capture(**kwargs):
                with torch.no_grad():
                    raw = model(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"]).logits[:, -1, :]
                generated = original_generate(**kwargs, return_dict_in_generate=True, output_scores=True)
                observed.append((raw, generated.scores[0]))
                return generated.sequences
            with patch.object(model, "generate", side_effect=capture):
                _, info = sampler.generate("distribution check", adapter_name=adapter, sample_index=0, group_id="g")
            self.assertEqual(len(observed), 1, adapter)
            self.assertTrue(torch.allclose(observed[0][0], observed[0][1], atol=1e-6, rtol=1e-5), adapter)
            self.assertEqual(info["sampling_contract"], "raw_softmax_v1")
        # Another caller's decoding defaults are untouched afterwards.
        self.assertEqual(owner.generation_config.top_k, 1)
        self.assertEqual(owner.generation_config.repetition_penalty, 2.0)
        self.assertEqual(owner.generation_config.suppress_tokens, [0, 3])


if __name__ == "__main__":
    unittest.main()
