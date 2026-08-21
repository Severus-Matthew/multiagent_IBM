from __future__ import annotations

from typing import Any

from .action_loop import run_action_prompt_optimizer_loop
from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .end_to_end_reward import end_to_end_reward
from .grpo_math import group_relative_advantages
from .rca_loop import run_rca_grpo_episode
from .schemas import parse_fault_lines


def _group_normalize(
    trajectories: list[dict[str, Any]],
    *,
    value_key: str,
    mean_key: str,
    std_key: str,
    advantage_key: str,
    zero_variance_key: str,
) -> None:
    result = group_relative_advantages(
        [float(t.get(value_key, 0.0) or 0.0) for t in trajectories],
        scale_by_std=True,
    )
    for trajectory, advantage in zip(trajectories, result.advantages):
        trajectory[mean_key] = round(result.mean, 6)
        trajectory[std_key] = round(result.std, 6)
        trajectory[advantage_key] = round(float(advantage), 6)
        trajectory[zero_variance_key] = bool(result.zero_variance)
        trajectory[f"{advantage_key}_std_correction"] = result.std_correction
        trajectory[f"{advantage_key}_normalization_epsilon"] = result.normalization_epsilon


def _compute_factorized_advantages(trajectories: list[dict[str, Any]]) -> None:
    _group_normalize(
        trajectories,
        value_key="system_reward",
        mean_key="system_group_reward_mean",
        std_key="system_group_reward_std",
        advantage_key="system_advantage",
        zero_variance_key="system_group_zero_variance",
    )
    _group_normalize(
        trajectories,
        value_key="rca_policy_return",
        mean_key="rca_group_return_mean",
        std_key="rca_group_return_std",
        advantage_key="rca_policy_advantage",
        zero_variance_key="rca_group_zero_variance",
    )
    _group_normalize(
        trajectories,
        value_key="action_policy_return",
        mean_key="action_group_return_mean",
        std_key="action_group_return_std",
        advantage_key="action_policy_advantage",
        zero_variance_key="action_group_zero_variance",
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

        role_index = role_seen[stage]
        role_seen[stage] += 1

        metadata = dict(row.get("metadata", {}) or {})
        metadata.update({
            "trajectory_group_id": trajectory_group_id,
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "trajectory_decision_index": decision_index,
            "trajectory_role_decision_index": role_index,
            "trajectory_role_decision_count": role_counts[stage],
            "role_local_reward": row.get("reward"),
            "role_local_advantage": row.get("advantage"),
            "system_reward": round(float(system_reward), 6),
            "system_advantage": round(float(system_advantage), 6),
            "factorized_reward_mode": reward_mode,
            "optimizer_role": optimizer_role,
            "optimizer_buffer": buffer_name,
            "optimizer_advantage_field": "policy_advantage",
            "optimizer_loss_aggregation_contract": (
                "token_level_clipped_surrogate; normalize by total active completion tokens "
                "within each role optimizer update (DAPO-style token normalization)"
            ),
        })
        row["metadata"] = metadata

        row["policy_reward"] = round(float(policy_reward), 6)
        row["policy_advantage"] = round(float(policy_advantage), 6)
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
    rca_max_iterations: int = 3,
    action_max_iterations: int = 3,
    rca_policy_model_name: str = "debug-rca-policy",
    action_policy_model_name: str = "structured-action-policy",
    policy_version: str = "v0",
    agent_input_mode: str = "training_safe",
    reward_mode: str = "factorized_joint_pipeline_v2_no_double_count",
    min_twin_reproduction_score: float = 0.0,
    rca_downstream_credit_weight: float = 0.15,
    action_system_credit_weight: float = 0.25,
) -> dict[str, Any]:
    """Generate complete joint trajectories with factorized role-specific credit.

    The group baseline is over complete trajectories sampled from the same initial
    incident. This is a trajectory-level group-relative policy-gradient design;
    later RCA/Action decision prompts can differ because their histories differ.
    We therefore do not claim that every decision row is a vanilla same-prompt
    GRPO completion. The stored trajectory advantage is the Monte-Carlo credit
    applied to all decisions of that role in the sampled trajectory.
    """
    if agent_input_mode != "training_safe":
        raise ValueError("joint training requires agent_input_mode='training_safe'")
    if int(trajectory_group_size) < 2:
        raise ValueError(
            "trajectory_group_size must be >= 2 for group-relative policy optimization; "
            "a singleton group has zero relative advantage"
        )

    agent_state = sanitize_agent_state(compressed_state, mode="training_safe")
    safety = agent_input_safety_report(agent_state)
    if not safety.get("safe_for_training_agent"):
        raise ValueError(f"agent-facing state failed safety audit: {safety}")

    scenario_id = str(full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown")
    trajectory_group_id = f"e2e:{scenario_id}"
    trajectories: list[dict[str, Any]] = []

    for trajectory_index in range(int(trajectory_group_size)):
        trajectory_id = f"{trajectory_group_id}:traj{trajectory_index}"

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
        )
        rca_samples = list(rca_result.get("grpo_samples", []) or [])
        rca_faults = parse_fault_lines(rca_result.get("final_prediction", ""))

        action_result = run_action_prompt_optimizer_loop(
            full_state,
            compressed_state,
            rca_result,
            rca_faults,
            action_prompt_policy,
            action_agent,
            twin_verifier,
            max_iterations=action_max_iterations,
            require_rca_twin_verification=False,
            skip_action_if_rca_unverified=False,
            min_twin_reproduction_score=min_twin_reproduction_score,
            rca_twin_gate=None,
            group_size=1,
            selection_strategy="sample0",
            policy_model_name=action_policy_model_name,
            policy_version=policy_version,
            agent_state=agent_state,
            agent_input_mode="training_safe",
            agent_input_safety=safety,
            sample_index_offset=trajectory_index,
            require_upstream_label_success_for_gate=False,
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
            "rca_result": {k: v for k, v in rca_result.items() if k != "grpo_samples"},
            "action_result": {k: v for k, v in action_result.items() if k != "grpo_samples"},
            "_policy_samples": rca_samples + action_samples,
        })

    _compute_factorized_advantages(trajectories)

    rca_policy_samples: list[dict[str, Any]] = []
    action_policy_samples: list[dict[str, Any]] = []
    for trajectory in trajectories:
        rca_rows, action_rows = _attach_factorized_credit(
            trajectory.pop("_policy_samples", []),
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

    all_policy_samples = rca_policy_samples + action_policy_samples

    return {
        "scenario_id": scenario_id,
        "trajectory_group_id": trajectory_group_id,
        "trajectory_group_size": int(trajectory_group_size),
        "agent_input_mode": "training_safe",
        "agent_input_safety": safety,
        "reward_mode": reward_mode,
        "credit_assignment_mode": "joint_rollout_factorized_policy_returns_v2",
        "update_schedule": "batch_synchronized_separate_policy_updates",
        "trajectories": trajectories,
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
        "num_successful_trajectories": sum(1 for t in trajectories if t.get("trajectory_success")),
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "policy_credit_contract": {
            "rca_optimizer_advantage": "rca_policy_advantage",
            "action_optimizer_advantage": "action_policy_advantage",
            "system_advantage": "diagnostic_only",
            "advantage_normalization": "per_incident_complete_trajectory_group_sample_std_plus_1e-4",
            "trajectory_group_baseline_scope": "same_initial_incident",
            "decision_prompt_equivalence": "not_assumed_after_history_diverges",
            "rca_downstream_credit_weight": float(rca_downstream_credit_weight),
            "action_system_credit_weight": float(action_system_credit_weight),
            "verifier_trainable": False,
            "future_loss_aggregation": "DAPO-style total-active-token normalization per role optimizer update",
        },
    }
