from __future__ import annotations

"""Fail-closed audit of leakage, RCA identity, and live-Twin scoring contracts."""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from digital_twin_runtime.telemetry_comparator import compare_symptoms_scoped, score_resolution
from training_pipeline.action_loop import _safe_action_history_entry
from training_pipeline.action_reward import _is_mutating_command
from training_pipeline.agent_input_safety import agent_input_safety_report, sanitize_agent_state
from training_pipeline.command_normalizer import normalize_command
from training_pipeline.command_safety import check_command_safety
from training_pipeline.rca_loop import _hypothesis_key, _public_twin_verified, _safe_history_entry
from training_pipeline.rca_reward import rca_reward
from training_pipeline.schemas import ActionAttempt, FaultLabel, RCAAttempt, parse_fault_lines


def _assert(condition: bool, name: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def run_audit() -> dict[str, Any]:
    checks: list[str] = []

    same_type = parse_fault_lines("user-timeline-service::infra_failure::delete_pod")
    other_mech = parse_fault_lines(
        "user-timeline-service::infra_failure::assign_to_non_existent_node"
    )
    _assert(same_type[0].canonical_key() == other_mech[0].canonical_key(), "legacy_canonical_key_is_type_only", checks)
    _assert(same_type[0].hypothesis_key() != other_mech[0].hypothesis_key(), "hypothesis_key_distinguishes_mechanisms", checks)
    _assert(_hypothesis_key(same_type) != _hypothesis_key(other_mech), "retry_identity_uses_hypothesis_key", checks)

    gt = [
        FaultLabel(
            service="user-timeline-service",
            fault_type="infra_failure",
            fault_family="assign_to_non_existent_node_social_net",
            fault_mechanism="assign_to_non_existent_node",
        )
    ]
    exact = rca_reward({}, gt, other_mech)
    wrong = rca_reward({}, gt, same_type)
    _assert(exact["success"] and not wrong["success"], "exact_set_requires_injectible_mechanism", checks)
    _assert(exact["reward"] > wrong["reward"], "correct_mechanism_outranks_same_service_wrong_mechanism", checks)

    leaky = {
        "scenario_id": "gen_assign_to_non_existent_node_social_net-detection-user-timeline-service-default",
        "timestamp": "2026-09-03T12:34:56.789Z",
        "task": "detection",
        "fault_context": {"faulty_service": "user-timeline-service"},
        "llm_view": {
            "scenario_id": "gen_assign_to_non_existent_node_social_net-detection-user-timeline-service-default",
            "top_log_error_services": [{"service": "user-timeline-service"}],
        },
        "services": ["user-timeline-service", "nginx-thrift"],
        "system": {"user-timeline-service": {"health": {"status": "pending", "pods_unready": 1}}},
    }
    safe = sanitize_agent_state(leaky, mode="training_safe")
    report = agent_input_safety_report(safe)
    _assert("scenario_id" not in safe, "sanitizer_drops_scenario_id_key", checks)
    _assert("timestamp" not in safe, "sanitizer_drops_processing_timestamp", checks)
    _assert("task" not in safe, "sanitizer_drops_fault_context_task", checks)
    _assert("fault_context" not in safe, "sanitizer_drops_fault_context", checks)
    _assert("scenario_id" not in (safe.get("llm_view") or {}), "sanitizer_drops_nested_scenario_id", checks)
    _assert(bool(report.get("safe_for_training_agent")), "sanitized_state_passes_leak_report", checks)
    _assert("[redacted_scenario]" not in json.dumps(safe), "synthetic_state_had_no_path_embedded_ids", checks)

    path_leaky = {
        "services": ["user-timeline-service"],
        "observability_metadata": {
            "logs": {
                "files_seen": [
                    "/tmp/telemetry_outputs/gen_assign_to_non_existent_node_social_net-detection-user-timeline-service-default/logs.txt"
                ]
            }
        },
    }
    path_safe = sanitize_agent_state(path_leaky, mode="training_safe")
    path_blob = json.dumps(path_safe)
    _assert("gen_assign_to_non_existent_node" not in path_blob, "file_paths_do_not_retain_fault_family", checks)
    _assert("user-timeline-service-default" not in path_blob, "file_paths_do_not_retain_hidden_target_suffix", checks)
    _assert("[redacted_scenario]" in path_blob, "file_paths_are_redacted_to_opaque_token", checks)
    _assert(bool(agent_input_safety_report(path_safe).get("safe_for_training_agent")), "redacted_paths_pass_leak_report", checks)

    still_leaky = dict(safe)
    still_leaky["note"] = "gen_assign_to_non_existent_node_social_net-detection-user-timeline-service-default"
    leak_report = agent_input_safety_report(still_leaky)
    _assert(not leak_report.get("safe_for_training_agent"), "descriptive_scenario_id_values_are_rejected", checks)
    _assert(bool(leak_report.get("descriptive_scenario_id_values")), "descriptive_scenario_id_values_are_reported", checks)

    empty = compare_symptoms_scoped(
        {"system": {}, "service_health": {}, "logs": {}, "traces": {}, "llm_view": {}},
        {"system": {}, "service_health": {}, "logs": {}, "traces": {}, "llm_view": {}},
        ["user-timeline-service"],
    )
    _assert(float(empty["reproduction_score"]) == 0.0, "empty_scoped_channels_do_not_score_perfect", checks)
    _assert(empty["score_reason"] == "no_original_symptoms_in_sparse_scope", "empty_scope_fails_closed", checks)

    matched = compare_symptoms_scoped(
        {
            "system": {"user-timeline-service": {"health": {"infra_issue_flag": True, "pods_unready": 1}}},
            "service_health": {},
            "logs": {},
            "traces": {"per_edge": {}},
            "llm_view": {},
        },
        {
            "system": {"user-timeline-service": {"health": {"infra_issue_flag": True, "pods_unready": 1}}},
            "service_health": {},
            "logs": {},
            "traces": {"per_edge": {}},
            "llm_view": {},
        },
        ["user-timeline-service", "nginx-thrift"],
    )
    _assert(float(matched["reproduction_score"]) == 1.0, "matching_degraded_root_scores_one_without_empty_trace_credit", checks)

    rca_attempt = RCAAttempt(
        iteration=0,
        instruction="x",
        prediction_text="user-timeline-service::infra_failure::assign_to_non_existent_node",
        predicted_faults=other_mech,
        reward=3.2,
        reward_components={
            "twin_reproduction_score": 0.8,
            "rca_twin_verified": True,
            "exact_set_match": True,
            "invalid_format": False,
        },
        success=True,
        feedback="Counterfactual twin reproduction is comparatively strong.",
    )
    rca_history = _safe_history_entry(rca_attempt)
    blob = json.dumps(rca_history).lower()
    _assert("exact_set_match" not in blob, "rca_retry_history_omits_private_exact_match", checks)
    _assert("reward" not in rca_history, "rca_retry_history_omits_evaluator_reward", checks)
    _assert("success" not in rca_history, "rca_retry_history_omits_private_success", checks)

    action_history = _safe_action_history_entry(
        ActionAttempt(
            iteration=0,
            instruction_prompt="x",
            commands=["kubectl rollout status deployment/user-timeline-service -n ns"],
            reward=4.2,
            reward_components={"resolved": True, "safe": True, "sla_restored": True},
            success=True,
            feedback="Commands repaired the twin target.",
        )
    )
    _assert("reward" not in action_history, "action_retry_history_omits_evaluator_reward", checks)
    _assert("success" not in action_history, "action_retry_history_omits_success_flag", checks)
    _assert(
        not check_command_safety(
            ["kubectl get secret datastore-credentials -n aiops-twin-123"]
        )["safe"],
        "secret_reads_are_rejected",
        checks,
    )
    from types import SimpleNamespace
    from digital_twin_runtime.live_action_executor import execute_twin_commands

    fake_session = SimpleNamespace(
        namespace="aiops-twin-123",
        bundle=SimpleNamespace(object_refs=[]),
    )
    chaos_command = (
        "kubectl delete networkchaos twin-network-delay-service-a "
        "-n aiops-twin-123"
    )
    normalized_chaos = normalize_command(chaos_command)
    _assert(
        normalized_chaos["action"] == "remove_fault_resource"
        and normalized_chaos["valid"],
        "owned_chaos_delete_is_a_valid_normalized_action",
        checks,
    )
    _assert(
        _is_mutating_command(chaos_command),
        "owned_chaos_delete_is_counted_as_mutation",
        checks,
    )
    with patch(
        "digital_twin_runtime.live_action_executor.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="deleted", stderr=""),
    ):
        chaos_execution = execute_twin_commands(
            fake_session,
            [chaos_command],
            owned_runtime_objects=[
                {"kind": "networkchaos", "name": "twin-network-delay-service-a"}
            ],
        )
    _assert(
        chaos_execution.safe and chaos_execution.executed,
        "exact_owned_chaos_resource_can_be_remediated",
        checks,
    )
    unowned_execution = execute_twin_commands(
        fake_session,
        [chaos_command],
        owned_runtime_objects=[],
    )
    _assert(
        not unowned_execution.safe and not unowned_execution.executed,
        "unowned_chaos_resource_delete_is_rejected",
        checks,
    )

    _assert(
        _public_twin_verified(
            {"predicted_fault_injection_checked": True, "reproduction_score": 0.2, "rca_twin_verified": False},
            0.1,
        ),
        "public_twin_stop_uses_live_injection_and_score",
        checks,
    )
    _assert(
        not _public_twin_verified(
            {"predicted_fault_injection_checked": True, "reproduction_score": 0.05, "rca_twin_verified": True},
            0.1,
        ),
        "verifier_flag_cannot_bypass_reproduction_threshold",
        checks,
    )
    _assert(
        not _public_twin_verified(
            {"predicted_fault_injection_checked": True, "reproduction_score": 0.0, "rca_twin_verified": False},
            0.1,
        ),
        "zero_reproduction_does_not_stop_rca_retry",
        checks,
    )
    _assert(
        not _public_twin_verified(
            {"predicted_fault_injection_checked": False, "reproduction_score": 0.9, "rca_twin_verified": False},
            0.1,
        ),
        "offline_behavioral_score_is_not_a_public_live_stop",
        checks,
    )

    empty_resolution = score_resolution(
        {"system": {}, "service_health": {}, "logs": {}, "traces": {}, "llm_view": {}},
        {"system": {}, "service_health": {}, "logs": {}, "traces": {}, "llm_view": {}},
    )
    _assert(float(empty_resolution["symptom_reduction"]) == 0.0, "empty_action_resolution_is_not_perfect", checks)
    _assert(empty_resolution["resolved"] is False, "empty_action_resolution_is_not_resolved", checks)

    missing_after = score_resolution(
        {"system": {"user-timeline-service": {"ready": False}}, "service_health": {}, "logs": {}, "traces": {}},
        {},
    )
    _assert(missing_after["resolved"] is False, "unobserved_after_state_is_not_resolved", checks)
    _assert(float(missing_after["symptom_reduction"]) == 0.0, "unobserved_after_state_is_not_perfect", checks)

    from inspect import signature

    from dataclasses import fields

    from digital_twin_runtime.sparse_live_verifier import (
        SparseLiveTwinVerifier,
        SparseLiveVerifierConfig,
        TwinTelemetryIncomplete,
    )
    from digital_twin_runtime.live_capabilities import (
        assess_live_reward_calibration,
    )
    from digital_twin_runtime.live_fault_injector import POD_CHAOS_ACTIONS
    from digital_twin_runtime.targeted_workload import WorkloadResult
    from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
    from training_pipeline.end_to_end_loop import (
        _compute_factorized_advantages,
        run_end_to_end_trajectory_group,
    )
    from training_pipeline.grpo_math import drop_undersized_optimizer_groups
    from training_pipeline.rca_loop import run_rca_grpo_episode
    from training_pipeline.train_qwen_live_grpo import _parser as live_trainer_parser

    hops_default = signature(build_sparse_live_twin_spec).parameters["downstream_support_hops"].default
    verifier_hops = next(
        f.default for f in fields(SparseLiveVerifierConfig) if f.name == "downstream_support_hops"
    )
    _assert(int(hops_default) == 1, "sparse_twin_default_support_hops_is_one", checks)
    _assert(int(verifier_hops) == 1, "live_verifier_default_support_hops_is_one", checks)
    e2e_params = signature(run_end_to_end_trajectory_group).parameters
    _assert(int(e2e_params["rca_max_iterations"].default) == 7, "rca_retry_default_is_seven", checks)
    _assert(int(e2e_params["action_max_iterations"].default) == 7, "action_retry_default_is_seven", checks)
    parser = live_trainer_parser()
    _assert(int(parser.get_default("rca_max_iterations")) == 7, "trainer_rca_retry_default_is_seven", checks)
    _assert(int(parser.get_default("action_max_iterations")) == 7, "trainer_action_retry_default_is_seven", checks)
    _assert(
        assess_live_reward_calibration([
            FaultLabel(
                "service-a",
                "infra_failure",
                fault_mechanism="assign_to_non_existent_node",
            )
        ])["eligible"],
        "matched_control_mechanism_is_reward_calibrated",
        checks,
    )
    _assert(
        not assess_live_reward_calibration([
            FaultLabel(
                "service-a",
                "config_error",
                fault_mechanism="target_port_misconfig",
            )
        ])["eligible"],
        "uncalibrated_mechanism_is_not_reward_eligible",
        checks,
    )
    _assert(
        not assess_live_reward_calibration([
            FaultLabel("service-a", "infra_failure", fault_mechanism="assign_to_non_existent_node"),
            FaultLabel("service-b", "infra_failure", fault_mechanism="scale_replicas_zero"),
        ])["eligible"],
        "uncalibrated_multifault_threshold_is_not_reward_eligible",
        checks,
    )
    _assert(
        POD_CHAOS_ACTIONS["pod_kill"] == "pod-kill"
        and POD_CHAOS_ACTIONS["pod_failure"] == "pod-failure",
        "pod_kill_and_pod_failure_use_distinct_chaos_actions",
        checks,
    )

    # The manifest layer resolves Kubernetes objects but may not widen the
    # planner-selected service scope merely because a shared registry ConfigMap
    # happens to name every application service.
    import digital_twin_runtime.sparse_live_manifest as manifest_module
    def controller(name: str) -> dict[str, Any]:
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name},
            "spec": {
                "replicas": 1,
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [{"name": name, "image": f"example/{name}:1"}],
                        "volumes": [{"name": "registry", "configMap": {"name": "service-registry"}}],
                    },
                },
            },
            "status": {"readyReplicas": 1, "availableReplicas": 1},
        }

    fake_objects = {
        "deployments": [controller("service-a"), controller("service-b")],
        "statefulsets": [],
        "services": [
            {"metadata": {"name": name}, "spec": {"selector": {"app": name}, "ports": [{"port": 80}]}}
            for name in ("service-a", "service-b")
        ],
        "configmaps": [{
            "metadata": {"name": "service-registry"},
            "data": {"services.json": '{"service-a":"service-a","service-b":"service-b"}'},
        }],
        "secrets": [],
        "persistentvolumeclaims": [],
        "serviceaccounts": [{"metadata": {"name": "default"}}],
    }
    with patch.object(
        manifest_module,
        "_items",
        side_effect=lambda kind, namespace: deepcopy(fake_objects[kind]),
    ):
        manifest_plan = manifest_module.discover_sparse_manifest_plan(
            "source", ["service-a"]
        )
    _assert(
        manifest_plan.selected_services == ["service-a"],
        "manifest_discovery_does_not_widen_causal_service_scope",
        checks,
    )

    verifier = SparseLiveTwinVerifier(
        SparseLiveVerifierConfig("source", ".", ".")
    )
    uncalibrated_result = verifier.validate_rca_prediction(
        {},
        {"services": ["service-a"]},
        [
            FaultLabel(
                "service-a",
                "config_error",
                fault_mechanism="target_port_misconfig",
            )
        ],
    )
    _assert(
        not uncalibrated_result["rca_twin_verified"]
        and not uncalibrated_result["predicted_fault_injection_checked"],
        "uncalibrated_prediction_fails_before_live_mutation",
        checks,
    )
    workload_rows = {
        service: WorkloadResult(
            name=f"workload-{service}",
            endpoint="http://frontend:80",
            completed=True,
            failed=False,
            elapsed_seconds=1.0,
            requests_per_second=10.0,
            total_requests=10,
            non_success_responses=0,
            application_failures=0,
            probe_http_status=200,
            probe_body="ok",
            required_service=service,
            required_ready_endpoints=1,
            socket_errors={},
            output="10 requests in 1s",
            execution_started=True,
        )
        for service in ("service-a", "service-b")
    }
    with patch.object(
        verifier,
        "_run_workload",
        side_effect=lambda service: workload_rows[service],
    ):
        aggregate_workload, individual_workloads = (
            verifier._run_predicted_root_workloads([
                FaultLabel("service-a", "infra_failure"),
                FaultLabel("service-b", "infra_failure"),
            ])
        )
    _assert(len(individual_workloads) == 2, "multifault_executes_every_predicted_root_workload", checks)
    _assert(aggregate_workload.total_requests == 20, "multifault_workload_aggregates_requests", checks)
    coverage = verifier._require_predicted_roots_observed(
        {
            "traces": {
                "per_edge": {
                    "frontend->service-a": {"source": "frontend", "target": "service-a"},
                    "frontend->service-b": {"source": "frontend", "target": "service-b"},
                }
            }
        },
        [FaultLabel("service-a", "infra_failure"), FaultLabel("service-b", "infra_failure")],
        "audit",
    )
    _assert(not coverage["missing_predicted_root_services"], "multifault_trace_covers_every_predicted_root", checks)
    try:
        verifier._require_predicted_roots_observed(
            {
                "traces": {
                    "per_edge": {
                        "frontend->service-a": {"source": "frontend", "target": "service-a"},
                    }
                }
            },
            [FaultLabel("service-a", "infra_failure"), FaultLabel("service-b", "infra_failure")],
            "audit",
        )
    except TwinTelemetryIncomplete:
        pass
    else:
        raise AssertionError("missing multifault root trace coverage did not fail closed")
    checks.append("missing_multifault_root_trace_coverage_fails_closed")
    try:
        verifier._require_predicted_roots_observed(
            {
                "services": ["rate", "mongodb-rate"],
                "traces": {
                    "per_edge": {
                        "frontend->rate": {"source": "frontend", "target": "rate"},
                    }
                },
            },
            [FaultLabel("mongodb-rate", "auth_failure")],
            "audit",
        )
    except TwinTelemetryIncomplete:
        pass
    else:
        raise AssertionError("colliding service alias satisfied the wrong root")
    checks.append("multifault_trace_coverage_rejects_colliding_service_alias")

    skipped = {
        "trajectory_id": "t0",
        "system_reward": 0.1,
        "rca_policy_return": 0.5,
        "action_policy_return": -0.36,
        "action_stage_invoked": False,
        "_policy_samples": [{"stage": "rca"}],
    }
    acted = {
        "trajectory_id": "t1",
        "system_reward": 0.8,
        "rca_policy_return": 0.4,
        "action_policy_return": 1.0,
        "action_stage_invoked": True,
        "_policy_samples": [{"stage": "rca"}, {"stage": "action"}],
    }
    _compute_factorized_advantages([skipped, acted])
    _assert(skipped["action_policy_advantage"] == 0.0, "skipped_action_is_not_an_action_participant", checks)
    _assert(skipped["action_policy_advantage_participant"] is False, "skipped_action_marked_non_participant", checks)
    _assert(acted["action_policy_advantage"] == 0.0, "singleton_action_group_has_zero_advantage", checks)
    _assert(acted["action_policy_advantage_group_size"] == 1, "action_baseline_excludes_skipped_trajectory", checks)
    _assert(acted["action_group_return_mean"] == 1.0, "action_baseline_mean_ignores_skipped_return", checks)

    kept, dropped = drop_undersized_optimizer_groups(
        [
            {"optimizer_group_id": "g:action_policy", "trajectory_id": "t1", "policy_advantage": 0.0},
            {"optimizer_group_id": "g:rca_policy", "trajectory_id": "t0", "policy_advantage": 0.1},
            {"optimizer_group_id": "g:rca_policy", "trajectory_id": "t1", "policy_advantage": -0.1},
        ]
    )
    _assert(dropped == ["g:action_policy"], "singleton_action_group_is_dropped", checks)
    _assert(len(kept) == 2, "valid_rca_group_is_kept", checks)
    _assert(all(row["optimizer_group_id"] == "g:rca_policy" for row in kept), "dropped_group_rows_are_absent", checks)

    class _CountTwin:
        def __init__(self) -> None:
            self.calls = 0

        def validate_rca_prediction(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            self.calls += 1
            return {
                "mode": "sparse_live_kubernetes_v1",
                "reproduction_score": 0.05,
                "predicted_fault_injection_checked": True,
                "rca_twin_verified": False,
                "reward_route": "live",
            }

    class _ConstPolicy:
        def generate_instruction(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return "retry the same hypothesis"

    class _ConstSolver:
        def solve(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return "user-timeline-service::infra_failure::assign_to_non_existent_node"

    twin = _CountTwin()
    cached = run_rca_grpo_episode(
        {
            "scenario_id": "cache-identical-hypothesis",
            "fault_context": {
                "faulty_service": "compose-post-service",
                "fault_family": "k8s_target_port_misconfig",
            },
        },
        {"scenario_id": "cache-identical-hypothesis", "services": ["user-timeline-service"]},
        _ConstPolicy(),
        _ConstSolver(),
        twin_validator=twin,
        max_iterations=2,
        group_size=1,
        stop_on_local_success=False,
        stop_on_public_twin_verified=False,
    )
    _assert(twin.calls == 1, "identical_hypothesis_does_not_rebuild_twin", checks)
    reused_flags = [
        (attempt.get("reward_components") or {}).get("reused_identical_hypothesis")
        for attempt in cached.get("attempts") or []
    ]
    _assert(reused_flags == [False, True], "second_identical_hypothesis_is_marked_reused", checks)
    routes = {
        (attempt.get("reward_components") or {}).get("reward_route")
        for attempt in cached.get("attempts") or []
    }
    _assert(routes == {"live"}, "rca_attempts_record_live_reward_route", checks)

    leaked = _safe_history_entry(
        RCAAttempt(
            iteration=0,
            instruction="x",
            prediction_text="user-timeline-service::infra_failure::assign_to_non_existent_node",
            predicted_faults=parse_fault_lines(
                "user-timeline-service::infra_failure::assign_to_non_existent_node"
            ),
            reward=0.0,
            reward_components={
                "reward_route": "live",
                "twin_mode": "sparse_live_kubernetes_v1",
                "twin_reproduction_score": 0.05,
            },
            success=False,
            feedback="ok",
        )
    )
    summary = leaked.get("public_verifier_summary") or {}
    _assert("reward_route" not in leaked, "reward_route_is_not_agent_visible", checks)
    _assert("twin_mode" not in leaked, "twin_mode_is_not_agent_visible", checks)
    _assert("reward_route" not in summary, "reward_route_absent_from_public_summary", checks)
    _assert("twin_mode" not in summary, "twin_mode_absent_from_public_summary", checks)

    return {
        "status": "PASS_PIPELINE_CONTRACT",
        "num_checks": len(checks),
        "checks": checks,
    }


def main() -> None:
    print(json.dumps(run_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
