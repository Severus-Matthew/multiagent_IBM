from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .action_loop import run_action_prompt_optimizer_loop
from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .end_to_end_reward import end_to_end_reward
from .rca_loop import run_rca_grpo_episode
from .schemas import parse_fault_lines


def _trajectory_advantages(trajectories: list[dict[str, Any]]) -> None:
    rewards = [float(t.get("trajectory_reward", 0.0) or 0.0) for t in trajectories]
    if not rewards:
        return
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    denom = sigma if sigma > 1e-8 else 1.0
    for trajectory in trajectories:
        advantage = (float(trajectory.get("trajectory_reward", 0.0)) - mu) / denom
        trajectory["trajectory_group_reward_mean"] = round(mu, 6)
        trajectory["trajectory_group_reward_std"] = round(sigma, 6)
        trajectory["trajectory_advantage"] = round(advantage, 6)


def _attach_joint_credit(
    samples: list[dict[str, Any]],
    *,
    trajectory_group_id: str,
    trajectory_id: str,
    trajectory_index: int,
    trajectory_reward: float,
    trajectory_advantage: float,
    reward_mode: str,
) -> list[dict[str, Any]]:
    out = []
    for decision_index, sample in enumerate(samples):
        row = dict(sample)
        metadata = dict(row.get("metadata", {}) or {})
        metadata.update({
            "trajectory_group_id": trajectory_group_id,
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "trajectory_decision_index": decision_index,
            "role_local_reward": row.get("reward"),
            "role_local_advantage": row.get("advantage"),
            "joint_reward_mode": reward_mode,
        })
        row["metadata"] = metadata
        # These are intentionally separate from the local stage reward/advantage.
        # The future joint optimizer must use joint_advantage for all trainable
        # decisions belonging to this trajectory.
        row["joint_reward"] = round(float(trajectory_reward), 6)
        row["joint_advantage"] = round(float(trajectory_advantage), 6)
        row["trajectory_group_id"] = trajectory_group_id
        row["trajectory_id"] = trajectory_id
        out.append(row)
    return out


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
    reward_mode: str = "offline_diagnostic_joint_v1",
    min_twin_reproduction_score: float = 0.0,
) -> dict[str, Any]:
    """Generate a GRPO-style group of complete incident-resolution trajectories.

    This is the canonical joint rollout unit. Each outer sample executes the whole
    pipeline:

        sanitized incident state
          -> RCA prompt policy
          -> fixed/safe RCA solver
          -> counterfactual twin feedback
          -> Action prompt policy
          -> fixed/LLM ActionAgent
          -> command safety + twin execution/simulation
          -> SLA/recovery verifier
          -> one end-to-end trajectory reward

    Hidden exact-label RCA success never controls whether the Action stage is
    reached. The complete downstream recovery outcome therefore supplies credit to
    both trainable stages. The verifier/twin is fixed and receives no optimizer
    updates.
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

        # One policy sample per local decision. Diversity is supplied by the outer
        # trajectory index now and by stochastic model sampling once Qwen is live.
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
            # Critical for joint training: private exact-label success must not
            # alter the length/state transition of the trajectory.
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
            # Offline v3 is diagnostic, not a calibrated strict gate. The action
            # stage must still run so end-to-end recovery can provide learning
            # signal. Live twin mode can enable a strict gate later.
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

        reward_obj = end_to_end_reward(rca_result, action_result, reward_mode=reward_mode)
        trajectories.append({
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_index,
            "trajectory_reward": reward_obj["reward"],
            "trajectory_success": reward_obj["success"],
            "reward": reward_obj,
            "rca_result": {k: v for k, v in rca_result.items() if k != "grpo_samples"},
            "action_result": {k: v for k, v in action_result.items() if k != "grpo_samples"},
            "_policy_samples": rca_samples + action_samples,
        })

    _trajectory_advantages(trajectories)

    joint_samples: list[dict[str, Any]] = []
    for trajectory in trajectories:
        joint_samples.extend(_attach_joint_credit(
            trajectory.pop("_policy_samples", []),
            trajectory_group_id=trajectory_group_id,
            trajectory_id=trajectory["trajectory_id"],
            trajectory_index=int(trajectory["trajectory_index"]),
            trajectory_reward=float(trajectory["trajectory_reward"]),
            trajectory_advantage=float(trajectory.get("trajectory_advantage", 0.0)),
            reward_mode=reward_mode,
        ))

    return {
        "scenario_id": scenario_id,
        "trajectory_group_id": trajectory_group_id,
        "trajectory_group_size": max(1, int(trajectory_group_size)),
        "agent_input_mode": "training_safe",
        "agent_input_safety": safety,
        "reward_mode": reward_mode,
        "trajectories": trajectories,
        "joint_grpo_samples": joint_samples,
        "group_reward_mean": trajectories[0].get("trajectory_group_reward_mean") if trajectories else None,
        "group_reward_std": trajectories[0].get("trajectory_group_reward_std") if trajectories else None,
        "num_successful_trajectories": sum(1 for t in trajectories if t.get("trajectory_success")),
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "joint_credit_contract": (
            "All trainable RCA/action policy decisions in one trajectory receive the same joint_advantage. "
            "Verifier/twin components remain fixed."
        ),
    }
