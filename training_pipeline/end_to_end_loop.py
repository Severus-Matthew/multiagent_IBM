from __future__ import annotations

from typing import Any

from .action_loop import run_action_prompt_optimizer_loop
from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .bounded_agent_state import BoundedAgentStateConfig, build_bounded_agent_state
from .end_to_end_reward import end_to_end_reward
from .generation_examples import examples_from_group_result
from .grpo_math import drop_undersized_optimizer_groups, group_relative_advantages
from .rca_loop import run_rca_grpo_episode
from .schemas import parse_fault_lines


def _reward_route_from_episode(rca_result: dict[str, Any], action_result: dict[str, Any]) -> str:
    for source in (action_result, rca_result):
        if not isinstance(source, dict):
            continue
        route = source.get("reward_route")
        if route:
            return str(route)
        for attempt in reversed(source.get("attempts") or []):
            comps = (attempt or {}).get("reward_components") or {}
            if comps.get("reward_route"):
                return str(comps["reward_route"])
    return "unknown"


def _group_normalize(
    trajectories: list[dict[str, Any]],
    *,
    value_key: str,
    mean_key: str,
    std_key: str,
    advantage_key: str,
    zero_variance_key: str,
    participants: list[dict[str, Any]] | None = None,
) -> None:
    """Normalize a role return across the trajectories that role actually acted in.

    ``participants`` restricts the baseline to trajectories that produced a
    decision for this role. A trajectory whose action stage was skipped, for
    example because the live gate rejected the RCA hypothesis, still carries an
    action return, but it is not a sample from the action policy and must not
    shift the mean or standard deviation used to normalize the trajectories that
    did act. Non-participants keep a zero advantage and contribute no rows.
    """
    scored = trajectories if participants is None else participants
    result = group_relative_advantages(
        [float(t.get(value_key, 0.0) or 0.0) for t in scored],
        scale_by_std=True,
    )
    advantages = {id(t): a for t, a in zip(scored, result.advantages)}
    participating = {id(t) for t in scored}
    for trajectory in trajectories:
        member = id(trajectory) in participating
        trajectory[mean_key] = round(result.mean, 6)
        trajectory[std_key] = round(result.std, 6)
        trajectory[advantage_key] = round(float(advantages.get(id(trajectory), 0.0)), 6)
        trajectory[zero_variance_key] = bool(result.zero_variance)
        trajectory[f"{advantage_key}_participant"] = member
        trajectory[f"{advantage_key}_group_size"] = len(scored)
        trajectory[f"{advantage_key}_std_correction"] = result.std_correction
        trajectory[f"{advantage_key}_normalization_epsilon"] = result.normalization_epsilon


def _optimizer_eligible(trajectory: dict[str, Any]) -> bool:
    # A zero placeholder return is not an observed reward. Require the reward
    # boundary's explicit admission decision before baseline or token replay.
    reward = trajectory.get("reward") or {}
    components = reward.get("components") or {}
    return components.get("optimizer_credit_eligible") is True


def _compute_factorized_advantages(trajectories: list[dict[str, Any]]) -> None:
    _group_normalize(
        trajectories,
        value_key="system_reward",
        mean_key="system_group_reward_mean",
        std_key="system_group_reward_std",
        advantage_key="system_advantage",
        zero_variance_key="system_group_zero_variance",
    )
    eligible = [t for t in trajectories if _optimizer_eligible(t)]
    _group_normalize(
        trajectories,
        value_key="rca_policy_return",
        mean_key="rca_group_return_mean",
        std_key="rca_group_return_std",
        advantage_key="rca_policy_advantage",
        zero_variance_key="rca_group_zero_variance",
        participants=[t for t in eligible if t.get("rca_stage_invoked")],
    )
    action_participants = [t for t in eligible if t.get("action_stage_invoked")]
    _group_normalize(
        trajectories,
        value_key="action_policy_return",
        mean_key="action_group_return_mean",
        std_key="action_group_return_std",
        advantage_key="action_policy_advantage",
        zero_variance_key="action_group_zero_variance",
        participants=action_participants,
    )


def _attach_factorized_credit(
    samples: list[dict[str, Any]],
    *,
    trajectory_group_id: str,
    trajectory_id: str,
    trajectory_index: int,
    system_reward: float,
    system_advantage: float,
    rca_policy_return: float,
    rca_policy_advantage: float,
    action_policy_return: float,
    action_policy_advantage: float,
    reward_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rca_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    role_counts = {
        "rca": sum(1 for sample in samples if str(sample.get("stage") or "") == "rca"),
        "action": sum(1 for sample in samples if str(sample.get("stage") or "") == "action"),
    }
    role_seen = {"rca": 0, "action": 0}

    for decision_index, sample in enumerate(samples):
        row = dict(sample)
        stage = str(row.get("stage") or "")
        if stage == "rca":
            policy_reward = rca_policy_return
            policy_advantage = rca_policy_advantage
            buffer_name = "rca_policy_buffer"
            optimizer_role = "rca_policy"
        elif stage == "action":
            policy_reward = action_policy_return
            policy_advantage = action_policy_advantage
            buffer_name = "action_policy_buffer"
            optimizer_role = "action_policy"
        else:
            continue

        role_count = max(1, int(role_counts[stage]))
        role_index = role_seen[stage]
        role_seen[stage] += 1
        decision_weight = 1.0 / float(role_count)

        metadata = dict(row.get("metadata", {}) or {})
        metadata.update({
            "trajectory_group_id": trajectory_group_id,
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "trajectory_decision_index": decision_index,
            "trajectory_role_decision_index": role_index,
            "trajectory_role_decision_count": role_count,
            "trajectory_role_decision_weight": decision_weight,
            "role_local_reward": row.get("reward"),
            "role_local_advantage": row.get("advantage"),
            "system_reward": round(float(system_reward), 6),
            "system_advantage": round(float(system_advantage), 6),
            "factorized_reward_mode": reward_mode,
            "optimizer_role": optimizer_role,
            "optimizer_buffer": buffer_name,
            "optimizer_advantage_field": "policy_advantage",
            "optimizer_sample_weight_field": "optimizer_sample_weight",
            "optimizer_loss_aggregation_contract": (
                "for each role: normalize return across complete trajectories from the same incident; "
                "for each decision average clipped token surrogates over active completion tokens; "
                "weight each decision by 1/num_role_decisions_in_trajectory; average equally across trajectories"
            ),
        })
        row["metadata"] = metadata

        row["policy_reward"] = round(float(policy_reward), 6)
        row["policy_advantage"] = round(float(policy_advantage), 6)
        row["optimizer_sample_weight"] = decision_weight
        row["optimizer_group_id"] = f"{trajectory_group_id}:{optimizer_role}"
        row["trajectory_group_id"] = trajectory_group_id
        row["trajectory_id"] = trajectory_id

        row["system_reward"] = round(float(system_reward), 6)
        row["system_advantage"] = round(float(system_advantage), 6)
        row["joint_reward"] = row["system_reward"]
        row["joint_advantage"] = row["system_advantage"]

        if stage == "rca":
            rca_rows.append(row)
        else:
            action_rows.append(row)

    return rca_rows, action_rows


def run_end_to_end_trajectory_group(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    *,
    rca_instruction_policy,
    rca_solver,
    action_prompt_policy,
    action_agent,
    twin_verifier,
    trajectory_group_size: int = 4,
    rca_max_iterations: int = 7,
    action_max_iterations: int = 7,
    rca_policy_model_name: str = "debug-rca-policy",
    action_policy_model_name: str = "structured-action-policy",
    policy_version: str = "v0",
    agent_input_mode: str = "training_safe",
    reward_mode: str = "factorized_joint_pipeline_v2_no_double_count",
    min_twin_reproduction_score: float = 0.5,
    rca_downstream_credit_weight: float = 0.15,
    action_system_credit_weight: float = 0.25,
    bounded_agent_state_config: BoundedAgentStateConfig | None = None,
    rca_recent_performance: dict[str, Any] | None = None,
    action_recent_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate complete joint trajectories with factorized role-specific credit.

    The group baseline is over complete trajectories sampled from the same initial
    incident. This is a trajectory-level group-relative policy-gradient design;
    later RCA/Action decision prompts can differ because their histories differ.
    We therefore do not claim that every decision row is a vanilla same-prompt
    GRPO completion. The stored trajectory advantage is Monte-Carlo credit applied
    to all decisions of that role in the sampled trajectory.

    ``compressed_state`` remains the verifier/twin input. ``agent_state`` is derived
    independently from that already-redacted state. When ``bounded_agent_state_config``
    is supplied, only the RCA/Action policy and downstream agent-facing view is
    semantically projected; the twin still receives the original compressed state.
    This separation prevents a training-memory optimization from weakening the
    verifier's observable state.
    """
    if agent_input_mode != "training_safe":
        raise ValueError("joint training requires agent_input_mode='training_safe'")
    if int(trajectory_group_size) < 2:
        raise ValueError(
            "trajectory_group_size must be >= 2 for group-relative policy optimization; "
            "a singleton group has zero relative advantage"
        )

    prepare_scenario = getattr(twin_verifier, "prepare_scenario", None)
    if callable(prepare_scenario):
        prepare_scenario(full_state, compressed_state)
    public_agent_state = getattr(twin_verifier, "public_agent_state", None)
    policy_source_state = (
        public_agent_state(compressed_state) if callable(public_agent_state) else compressed_state
    )
    agent_state = sanitize_agent_state(policy_source_state, mode="training_safe")
    if bounded_agent_state_config is not None:
        agent_state = build_bounded_agent_state(
            agent_state,
            config=bounded_agent_state_config,
        )
    safety = agent_input_safety_report(agent_state)
    if not safety.get("safe_for_training_agent"):
        raise ValueError(f"agent-facing state failed safety audit: {safety}")

    scenario_id = str(full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown")
    trajectory_group_id = f"e2e:{scenario_id}"
    trajectories: list[dict[str, Any]] = []
    live_mode = bool(getattr(twin_verifier, "is_live", False))

    for trajectory_index in range(int(trajectory_group_size)):
        trajectory_id = f"{trajectory_group_id}:traj{trajectory_index}"
        begin = getattr(twin_verifier, "begin_trajectory", None)
        end = getattr(twin_verifier, "end_trajectory", None)
        if callable(begin):
            begin(trajectory_id)
        try:
            prepare_incident = getattr(twin_verifier, "prepare_incident_twin", None)
            if callable(prepare_incident):
                try:
                    prepare_incident(compressed_state)
                except Exception as exc:
                    trajectories.append({
                        "trajectory_id": trajectory_id, "trajectory_index": trajectory_index,
                        "system_reward": 0.0, "system_quality": 0.0,
                        "rca_policy_return": 0.0, "action_policy_return": 0.0,
                        "trajectory_success": False,
                        "reward": {"components": {"optimizer_credit_eligible": False,
                                                   "telemetry_incomplete": True}},
                        "reward_route": "live", "action_stage_invoked": False,
                        "skipped_action": True, "rca_result": {}, "action_result": {},
                        "preparation_error": f"{type(exc).__name__}: {exc}", "_policy_samples": [],
                    })
                    continue
            rca_result = run_rca_grpo_episode(
                full_state,
                compressed_state,
                rca_instruction_policy,
                rca_solver,
                twin_validator=twin_verifier,
                max_iterations=rca_max_iterations,
                group_size=1,
                selection_strategy="sample0",
                policy_model_name=rca_policy_model_name,
                policy_version=policy_version,
                agent_state=agent_state,
                agent_input_mode="training_safe",
                agent_input_safety=safety,
                sample_index_offset=trajectory_index,
                stop_on_local_success=False,
                stop_on_public_twin_verified=True,
                min_twin_reproduction_score=min_twin_reproduction_score,
                recent_performance=rca_recent_performance,
            )
            rca_samples = list(rca_result.get("grpo_samples", []) or [])
            rca_faults = parse_fault_lines(rca_result.get("final_prediction", ""))
            active_gate = None
            current_gate = getattr(twin_verifier, "current_rca_gate", None)
            if callable(current_gate):
                active_gate = current_gate(rca_faults)

            action_result = run_action_prompt_optimizer_loop(
                full_state,
                compressed_state,
                rca_result,
                rca_faults,
                action_prompt_policy,
                action_agent,
                twin_verifier,
                max_iterations=action_max_iterations,
                require_rca_twin_verification=live_mode,
                skip_action_if_rca_unverified=live_mode,
                min_twin_reproduction_score=min_twin_reproduction_score,
                rca_twin_gate=active_gate,
                group_size=1,
                selection_strategy="sample0",
                policy_model_name=action_policy_model_name,
                policy_version=policy_version,
                agent_state=agent_state,
                agent_input_mode="training_safe",
                agent_input_safety=safety,
                sample_index_offset=trajectory_index,
                require_upstream_label_success_for_gate=False,
                recent_performance=action_recent_performance,
            )
            action_samples = list(action_result.get("grpo_samples", []) or [])

            reward_obj = end_to_end_reward(
                rca_result,
                action_result,
                reward_mode=reward_mode,
                rca_downstream_credit_weight=rca_downstream_credit_weight,
                action_system_credit_weight=action_system_credit_weight,
            )
            trajectories.append({
                "trajectory_id": trajectory_id,
                "trajectory_index": trajectory_index,
                "system_reward": reward_obj["system_reward"],
                "system_quality": reward_obj["system_quality"],
                "rca_policy_return": reward_obj["rca_policy_return"],
                "action_policy_return": reward_obj["action_policy_return"],
                "trajectory_success": reward_obj["success"],
                "reward": reward_obj,
                "reward_route": reward_obj["components"]["reward_route"],
                "rca_stage_invoked": bool(rca_samples),
                "action_stage_invoked": bool(action_samples) and not bool(action_result.get("skipped_action")),
                "skipped_action": bool(action_result.get("skipped_action")),
                "rca_result": {k: v for k, v in rca_result.items() if k != "grpo_samples"},
                "action_result": {k: v for k, v in action_result.items() if k != "grpo_samples"},
                "_policy_samples": rca_samples + action_samples,
            })
        finally:
            if callable(end):
                end()

    _compute_factorized_advantages(trajectories)

    rca_policy_samples: list[dict[str, Any]] = []
    action_policy_samples: list[dict[str, Any]] = []
    excluded_trajectory_ids = []
    for trajectory in trajectories:
        samples = trajectory.pop("_policy_samples", [])
        if not _optimizer_eligible(trajectory):
            # Retain public/private diagnostic records, but exclude the entire
            # trajectory from optimizer replay, including its KL term.
            excluded_trajectory_ids.append(trajectory["trajectory_id"])
            continue
        rca_rows, action_rows = _attach_factorized_credit(
            samples,
            trajectory_group_id=trajectory_group_id,
            trajectory_id=trajectory["trajectory_id"],
            trajectory_index=int(trajectory["trajectory_index"]),
            system_reward=float(trajectory["system_reward"]),
            system_advantage=float(trajectory.get("system_advantage", 0.0)),
            rca_policy_return=float(trajectory["rca_policy_return"]),
            rca_policy_advantage=float(trajectory.get("rca_policy_advantage", 0.0)),
            action_policy_return=float(trajectory["action_policy_return"]),
            action_policy_advantage=float(trajectory.get("action_policy_advantage", 0.0)),
            reward_mode=reward_mode,
        )
        rca_policy_samples.extend(rca_rows)
        action_policy_samples.extend(action_rows)

    rca_policy_samples, dropped_rca_groups = drop_undersized_optimizer_groups(rca_policy_samples)
    action_policy_samples, dropped_action_groups = drop_undersized_optimizer_groups(action_policy_samples)
    all_policy_samples = rca_policy_samples + action_policy_samples
    projection = agent_state.get("projection") if isinstance(agent_state, dict) else None
    route_counts: dict[str, int] = {}
    for trajectory in trajectories:
        route = str(trajectory.get("reward_route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
    generation_examples = examples_from_group_result(
        {"scenario_id": scenario_id, "trajectories": trajectories},
        scenario_id=scenario_id,
    )

    return {
        "scenario_id": scenario_id,
        "trajectory_group_id": trajectory_group_id,
        "trajectory_group_size": int(trajectory_group_size),
        "agent_input_mode": "training_safe",
        "agent_input_safety": safety,
        "agent_state_projection": projection,
        "bounded_agent_state_enabled": bounded_agent_state_config is not None,
        "reward_mode": reward_mode,
        "credit_assignment_mode": "joint_rollout_factorized_policy_returns_v2",
        "update_schedule": "batch_synchronized_separate_policy_updates",
        "trajectories": trajectories,
        "generation_examples": generation_examples,
        "rca_grpo_samples": rca_policy_samples,
        "action_grpo_samples": action_policy_samples,
        "joint_grpo_samples": all_policy_samples,
        "system_group_reward_mean": trajectories[0].get("system_group_reward_mean") if trajectories else None,
        "system_group_reward_std": trajectories[0].get("system_group_reward_std") if trajectories else None,
        "rca_group_return_mean": trajectories[0].get("rca_group_return_mean") if trajectories else None,
        "rca_group_return_std": trajectories[0].get("rca_group_return_std") if trajectories else None,
        "action_group_return_mean": trajectories[0].get("action_group_return_mean") if trajectories else None,
        "action_group_return_std": trajectories[0].get("action_group_return_std") if trajectories else None,
        "rca_group_zero_variance": trajectories[0].get("rca_group_zero_variance") if trajectories else None,
        "action_group_zero_variance": trajectories[0].get("action_group_zero_variance") if trajectories else None,
        "reward_route_counts": route_counts,
        "dropped_action_optimizer_groups": dropped_action_groups,
        "dropped_rca_optimizer_groups": dropped_rca_groups,
        "optimizer_ineligible_trajectory_ids": excluded_trajectory_ids,
        "num_action_stage_trajectories": sum(1 for t in trajectories if t.get("action_stage_invoked")),
        "num_successful_trajectories": sum(1 for t in trajectories if t.get("trajectory_success")),
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "policy_credit_contract": {
            "rca_optimizer_advantage": "rca_policy_advantage",
            "action_optimizer_advantage": "action_policy_advantage",
            "system_advantage": "diagnostic_only",
            "advantage_normalization": "per_incident_complete_trajectory_group_sample_std_plus_1e-4",
            "trajectory_group_baseline_scope": "same_initial_incident",
            "optimizer_admission": "explicit_reward_eligibility_before_normalization_and_replay",
            "rca_baseline_scope": "eligible_trajectories_that_produced_rca_decisions",
            "action_baseline_scope": "eligible_trajectories_that_produced_action_decisions",
            "undersized_rca_groups_dropped": dropped_rca_groups,
            "undersized_action_groups_dropped": dropped_action_groups,
            "decision_prompt_equivalence": "not_assumed_after_history_diverges",
            "rca_downstream_credit_weight": float(rca_downstream_credit_weight),
            "action_system_credit_weight": float(action_system_credit_weight),
            "verifier_trainable": False,
            "bounded_agent_state_enabled": bounded_agent_state_config is not None,
            "future_loss_aggregation": (
                "per decision: mean clipped surrogate over completion tokens; "
                "per trajectory-role: weighted mean using optimizer_sample_weight=1/D_role; "
                "per optimizer group: equal mean over complete trajectories"
            ),
        },
    }
