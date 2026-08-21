from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .action_loop import run_action_prompt_optimizer_loop
from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .end_to_end_reward import end_to_end_reward
from .rca_loop import run_rca_grpo_episode
from .schemas import parse_fault_lines


def _group_normalize(
    trajectories: list[dict[str, Any]],
    *,
    value_key: str,
    mean_key: str,
    std_key: str,
    advantage_key: str,
) -> None:
    values = [float(t.get(value_key, 0.0) or 0.0) for t in trajectories]
    if not values:
        return
    mu = mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    denom = sigma if sigma > 1e-8 else 1.0
    for trajectory in trajectories:
        advantage = (float(trajectory.get(value_key, 0.0) or 0.0) - mu) / denom
        trajectory[mean_key] = round(mu, 6)
        trajectory[std_key] = round(sigma, 6)
        trajectory[advantage_key] = round(advantage, 6)


def _compute_factorized_advantages(trajectories: list[dict[str, Any]]) -> None:
    # System advantage is diagnostic/model-selection only.
    _group_normalize(
        trajectories,
        value_key="system_reward",
        mean_key="system_group_reward_mean",
        std_key="system_group_reward_std",
        advantage_key="system_advantage",
    )
    # These two advantages are the optimizer-facing returns.
    _group_normalize(
        trajectories,
        value_key="rca_policy_return",
        mean_key="rca_group_return_mean",
        std_key="rca_group_return_std",
        advantage_key="rca_policy_advantage",
    )
    _group_normalize(
        trajectories,
        value_key="action_policy_return",
        mean_key="action_group_return_mean",
        std_key="action_group_return_std",
        advantage_key="action_policy_advantage",
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
            # Non-trainable/verifier samples are never placed in an optimizer buffer.
            continue

        metadata = dict(row.get("metadata", {}) or {})
        metadata.update({
            "trajectory_group_id": trajectory_group_id,
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "trajectory_decision_index": decision_index,
            "role_local_reward": row.get("reward"),
            "role_local_advantage": row.get("advantage"),
            "system_reward": round(float(system_reward), 6),
            "system_advantage": round(float(system_advantage), 6),
            "factorized_reward_mode": reward_mode,
            "optimizer_role": optimizer_role,
            "optimizer_buffer": buffer_name,
            "optimizer_advantage_field": "policy_advantage",
        })
        row["metadata"] = metadata

        # Optimizer-facing factorized credit.
        row["policy_reward"] = round(float(policy_reward), 6)
        row["policy_advantage"] = round(float(policy_advantage), 6)
        row["optimizer_group_id"] = f"{trajectory_group_id}:{optimizer_role}"
        row["trajectory_group_id"] = trajectory_group_id
        row["trajectory_id"] = trajectory_id

        # Keep end-to-end quantities for analysis only; do not use these as the
        # direct policy gradient signal.
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
    reward_mode: str = "factorized_joint_pipeline_v1",
    min_twin_reproduction_score: float = 0.0,
    rca_downstream_credit_weight: float = 0.15,
    action_system_credit_weight: float = 0.25,
) -> dict[str, Any]:
    """Generate complete joint trajectories with factorized policy credit.

    Execution is always end-to-end and causally ordered:

        incident -> RCA policy -> RCA solver -> twin -> Action policy
                 -> ActionAgent -> action verifier -> SLA/recovery

    Optimization is factorized:

        RCA decisions    -> RCA-specific return/advantage -> RCA policy buffer
        Action decisions -> Action-specific return/advantage -> Action policy buffer

    The two buffers are intended to be updated separately, using independent LoRA
    adapters/optimizers, after the same rollout batch has completed. This gives us
    joint system learning without forcing both policies to share one noisy scalar
    return. The verifier/twin remains fixed.
    """
    if agent_input_mode != "training_safe":
        raise ValueError("joint training requires agent_input_mode='training_safe'")

    agent_state = sanitize_agent_state(compressed_state, mode="training_safe")
    safety = agent_input_safety_report(agent_state)
    if not safety.get("safe_for_training_agent"):
        raise ValueError(f"agent-facing state failed safety audit: {safety}")

    scenario_id = str(full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown")
    trajectory_group_id = f"e2e:{scenario_id}"
    trajectories: list[dict[str, Any]] = []

    for trajectory_index in range(max(1, int(trajectory_group_size))):
        trajectory_id = f"{trajectory_group_id}:traj{trajectory_index}"

        # One stochastic path through each policy per outer trajectory. Once the
        # trainable Qwen policies are enabled, diversity comes from model sampling.
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
            # Hidden exact-label correctness must not alter the joint state
            # transition. Action still receives the RCA prediction and twin signal.
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
            # Offline twin v3 is diagnostic, not a calibrated strict gate. The
            # action stage still runs so downstream recovery can contribute credit.
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

    # Combined view is retained for debugging/analysis only.
    all_policy_samples = rca_policy_samples + action_policy_samples

    return {
        "scenario_id": scenario_id,
        "trajectory_group_id": trajectory_group_id,
        "trajectory_group_size": max(1, int(trajectory_group_size)),
        "agent_input_mode": "training_safe",
        "agent_input_safety": safety,
        "reward_mode": reward_mode,
        "credit_assignment_mode": "joint_rollout_factorized_policy_returns_v1",
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
        "num_successful_trajectories": sum(1 for t in trajectories if t.get("trajectory_success")),
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "policy_credit_contract": {
            "rca_optimizer_advantage": "rca_policy_advantage",
            "action_optimizer_advantage": "action_policy_advantage",
            "system_advantage": "diagnostic_only",
            "rca_downstream_credit_weight": float(rca_downstream_credit_weight),
            "action_system_credit_weight": float(action_system_credit_weight),
            "verifier_trainable": False,
        },
    }
