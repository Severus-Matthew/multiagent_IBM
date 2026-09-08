"""Focused regression checks for the 2026-09 audit corrections.

Run from the repository root:
    python -m unittest discover -s tests -p test_audit_corrections.py -v
No cluster, GPU, LLM, or external API calls are performed.
"""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import mean, quantiles
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The state-abstraction CLI modules currently use sibling absolute imports.
sys.path.insert(0, str(ROOT / "state_abstraction_full"))

from state_abstraction_full.traces_parser import parse_traces, merge_edges
from state_abstraction_full.compress import compress_state
from training_pipeline.agent_input_safety import sanitize_agent_state, agent_input_safety_report
from training_pipeline.bounded_agent_state import BoundedAgentStateConfig, build_bounded_agent_state
from training_pipeline.action_loop import _namespace
from training_pipeline.end_to_end_loop import run_end_to_end_trajectory_group
from training_pipeline.grpo_math import group_relative_advantages
from training_pipeline.schemas import FaultLabel
from digital_twin_runtime.live_capabilities import assess_live_reward_calibration, LIVE_REWARD_CALIBRATION


FIELDS = ("trace_id", "span_id", "parent_span", "service_name",
          "operation_name", "duration", "response", "has_error")


def span(i, duration, *, trace=None, parent="", service="api", error=False):
    return dict(zip(FIELDS, (str(i) if trace is None else trace, str(i), parent,
                             service, "request", duration, 500 if error else 200,
                             str(error).lower())))


def write_trace(root, filename, rows):
    destination = Path(root) / "traces" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TraceAggregationTests(unittest.TestCase):
    def test_single_file_p95_is_raw_sample_p95(self):
        durations = [10] * 99 + [1000]
        with tempfile.TemporaryDirectory() as root:
            write_trace(root, "traces.csv", [span(i, d) for i, d in enumerate(durations)])
            edges, _, _ = parse_traces(root)
        actual = edges["ROOT->api"]
        self.assertAlmostEqual(actual["latency_p95_us"], quantiles(durations, n=100, method="inclusive")[94])
        self.assertAlmostEqual(actual["latency_mean_us"], mean(durations))
        self.assertEqual(actual["request_count"], 100)

    def test_unequal_file_sizes_match_one_export(self):
        rows = [span(i, d, error=i == 100) for i, d in enumerate([10] * 100 + [1000])]
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            write_trace(whole, "traces.csv", rows)
            write_trace(split, "traces-a.csv", rows[:100])
            write_trace(split, "traces-b.csv", rows[100:])
            expected, expected_edges, _ = parse_traces(whole)
            actual, actual_edges, _ = parse_traces(split)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_edges, expected_edges)

    def test_duplicate_exports_count_each_span_once(self):
        rows = [span(i, d) for i, d in enumerate([0, 10, 20, 100])]
        with tempfile.TemporaryDirectory() as root:
            write_trace(root, "traces-a.csv", rows)
            write_trace(root, "traces-b.csv", rows)
            edges, _, meta = parse_traces(root)
        self.assertEqual(edges["ROOT->api"]["request_count"], 4)
        self.assertEqual(meta["num_unique_spans"], 4)
        self.assertEqual(meta["duplicate_span_rows"], 4)

    def test_parent_can_be_in_another_export(self):
        with tempfile.TemporaryDirectory() as root:
            write_trace(root, "traces-a.csv", [span(1, 100, trace="t", service="front")])
            write_trace(root, "traces-b.csv", [span(2, 50, trace="t", parent="1")])
            edges, _, _ = parse_traces(root)
        self.assertIn("front->api", edges)
        self.assertNotIn("ROOT->api", edges)

    def test_conflicting_span_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            write_trace(root, "traces-a.csv", [span(1, 10)])
            write_trace(root, "traces-b.csv", [span(1, 1000)])
            with self.assertRaisesRegex(ValueError, "conflicting observations"):
                parse_traces(root)

    def test_header_only_export_is_not_trace_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            write_trace(root, "traces.csv", [])
            edges, _, meta = parse_traces(root)
        self.assertEqual(edges, {})
        self.assertFalse(meta["trace_signal_present"])

    def test_summary_merge_refuses_fabricated_quantiles(self):
        with self.assertRaisesRegex(ValueError, "raw spans"):
            merge_edges([{"ROOT->api": {"request_count": 1}},
                         {"ROOT->api": {"request_count": 10}}])


class RoutingEvidenceTests(unittest.TestCase):
    def test_service_spec_survives_compression_sanitization_and_projection(self):
        spec = {
            "namespace": "private-source", "cluster_ip": "10.0.0.1",
            "service_type": "ClusterIP", "selector": {"app": "z-api"},
            "ports": [{"name": "rpc", "port": 8080, "target_port": "wrong-port", "protocol": "TCP"}],
        }
        state = {
            "services": ["a", "z-api"],
            "system": {
                "a": {"pods_unready": 1, "infra_issue_flag": True},
                "z-api": {"services": spec, "pods_ready": 1, "pods_total": 1},
            },
            "fault_context": {"faulty_service": "z-api", "target_namespace": "private-source"},
        }
        compressed = compress_state(state)
        safe = sanitize_agent_state(compressed)
        projected = build_bounded_agent_state(safe, config=BoundedAgentStateConfig(max_system_services=1))
        actual = projected["system"]["z-api"]["service"]
        self.assertEqual(actual["ports"], spec["ports"])
        self.assertEqual(actual["selector"], spec["selector"])
        self.assertNotIn("namespace", actual)
        self.assertNotIn("cluster_ip", actual)
        self.assertNotIn("z-api", projected["system"]["__projection_summary__"]["rich_detail_service_names"])

    def test_private_labels_do_not_change_routing_evidence(self):
        state = {"services": ["api"], "system": {"api": {"services": {
            "ports": [{"port": 80, "target_port": 8080}], "selector": {"app": "api"}
        }}}, "fault_context": {"faulty_service": "api", "target_namespace": "source-a"}}
        altered = copy.deepcopy(state)
        altered["fault_context"] = {"faulty_service": "db", "target_namespace": "source-b"}
        self.assertEqual(sanitize_agent_state(compress_state(state)),
                         sanitize_agent_state(compress_state(altered)))


class LeakageBoundaryTests(unittest.TestCase):
    def test_safety_report_detects_unsanitized_fqdn_keys_and_values(self):
        obj = {"dependencies": {"api.original.svc.cluster.local": 2},
               "text": "db.original.svc.cluster.local refused connection"}
        self.assertFalse(agent_input_safety_report(obj)["safe_for_training_agent"])
        self.assertTrue(agent_input_safety_report(sanitize_agent_state(obj))["safe_for_training_agent"])

    def test_key_beyond_first_thousand_list_entries_is_checked(self):
        obj = [{} for _ in range(1001)] + [{"namespace": "private-source"}]
        self.assertFalse(agent_input_safety_report(obj)["safe_for_training_agent"])

    def test_namespace_comes_from_live_verifier(self):
        verifier = SimpleNamespace(is_live=True, action_namespace=lambda: "aiops-twin-owned")
        private = {"fault_context": {"target_namespace": "production"}}
        self.assertEqual(_namespace(private, {"namespace": "production"}, verifier), "aiops-twin-owned")

    def test_missing_live_namespace_never_falls_back(self):
        for verifier in (SimpleNamespace(is_live=True, action_namespace=lambda: None),
                         SimpleNamespace(is_live=True)):
            with self.subTest(verifier=verifier), self.assertRaises(RuntimeError):
                _namespace({"fault_context": {"target_namespace": "production"}},
                           {"namespace": "production"}, verifier)

    def test_offline_namespace_is_constant_and_not_oracle_derived(self):
        self.assertEqual(_namespace({"fault_context": {"namespace": "production"}}, {}),
                         "aiops-twin-debug")


def run_mock_group(eligible, rewards, action_invoked=None):
    if action_invoked is None:
        action_invoked = [True] * len(eligible)
    rca_results, action_results, scored = [], [], []
    for i, admitted in enumerate(eligible):
        rca_results.append({
            "final_prediction": "api::infra_failure::scale_replicas_zero",
            "grpo_samples": [{"stage": "rca", "completion": "diagnose", "metadata": {}}],
            "attempts": [],
        })
        action_results.append({
            "grpo_samples": ([{"stage": "action", "completion": "repair", "metadata": {}}]
                             if action_invoked[i] else []),
            "skipped_action": not action_invoked[i], "attempts": [],
        })
        scored.append({
            "system_reward": rewards[i], "system_quality": 0.0, "success": False,
            "rca_policy_return": rewards[i], "action_policy_return": rewards[i],
            "components": {"optimizer_credit_eligible": admitted, "reward_route": "live"},
        })
    with patch("training_pipeline.end_to_end_loop.run_rca_grpo_episode", side_effect=rca_results), \
         patch("training_pipeline.end_to_end_loop.run_action_prompt_optimizer_loop", side_effect=action_results), \
         patch("training_pipeline.end_to_end_loop.end_to_end_reward", side_effect=scored), \
         patch("training_pipeline.end_to_end_loop.examples_from_group_result", return_value=[]):
        return run_end_to_end_trajectory_group(
            {"scenario_id": "audit-incident"}, {"services": ["api"]},
            rca_instruction_policy=None, rca_solver=None,
            action_prompt_policy=None, action_agent=None,
            twin_verifier=SimpleNamespace(is_live=True),
            trajectory_group_size=len(eligible),
        )


class CalibrationContractTests(unittest.TestCase):
    def test_legacy_control_thresholds_are_not_reused_after_parser_change(self):
        for mechanism in ("scale_replicas_zero", "assign_to_non_existent_node"):
            with self.subTest(mechanism=mechanism):
                fault = FaultLabel(service="api", fault_type="infra_failure",
                                   fault_mechanism=mechanism)
                legacy = {mechanism: {"threshold": 0.4702, "evidence": "legacy-controls"}}
                with patch.dict(LIVE_REWARD_CALIBRATION, legacy, clear=True):
                    result = assess_live_reward_calibration([fault])
                self.assertFalse(result["eligible"])
                self.assertEqual(result["reason"], "reward_controls_require_raw_span_requalification")


class OptimizerAdmissionTests(unittest.TestCase):
    def test_ineligible_trajectory_cannot_shift_baseline_or_replay(self):
        result = run_mock_group([True, True, False], [0.2, 0.8, 0.0])
        expected = group_relative_advantages([0.2, 0.8]).advantages
        for role in ("rca", "action"):
            rows = result[role + "_grpo_samples"]
            self.assertEqual(len(rows), 2)
            for row, advantage in zip(rows, expected):
                self.assertAlmostEqual(row["policy_advantage"], advantage, places=5)
                self.assertNotIn("traj2", row["trajectory_id"])
        self.assertEqual(len(result["optimizer_ineligible_trajectory_ids"]), 1)
        self.assertEqual(result["reward_route_counts"], {"live": 3})

    def test_singleton_after_admission_is_dropped_for_both_roles(self):
        result = run_mock_group([True, False], [0.8, 0.0])
        self.assertEqual(result["joint_grpo_samples"], [])
        self.assertTrue(result["dropped_rca_optimizer_groups"])
        self.assertTrue(result["dropped_action_optimizer_groups"])

    def test_skipped_action_does_not_change_action_baseline(self):
        result = run_mock_group([True, True, True], [0.2, 0.8, -0.5], [True, True, False])
        self.assertEqual(len(result["rca_grpo_samples"]), 3)
        self.assertEqual(len(result["action_grpo_samples"]), 2)
        expected = group_relative_advantages([0.2, 0.8]).advantages
        for row, advantage in zip(result["action_grpo_samples"], expected):
            self.assertAlmostEqual(row["policy_advantage"], advantage, places=5)

    def test_all_ineligible_emits_no_optimizer_rows(self):
        result = run_mock_group([False, False], [0.0, 0.0])
        self.assertEqual(result["joint_grpo_samples"], [])
        self.assertEqual(len(result["trajectories"]), 2)
        self.assertEqual(len(result["optimizer_ineligible_trajectory_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
