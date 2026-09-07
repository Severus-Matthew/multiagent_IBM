from __future__ import annotations

"""Extract copy-pasteable training examples from each agent and component.

Joint trajectories already store the raw rollouts. This module flattens one
trajectory into a stable record that names who generated what:

* RCA instruction policy (trainable LoRA_RCA completion)
* RCA solver (frozen) prediction
* Twin verifier counterfactual result
* Action instruction policy (trainable LoRA_Action completion)
* Action agent (frozen) commands
* Action Twin/SLA verifier outcome
* factorized rewards, including whether anything improved or fully resolved

JSONL is the source of truth. Markdown is a human preview for reports.
"""

import json
from pathlib import Path
from typing import Any


def _clip(value: Any, limit: int = 4000) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_clip(item, limit) for item in value[:40]]
    if isinstance(value, dict):
        return {str(k): _clip(v, limit) for k, v in list(value.items())[:40]}
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "\n...[truncated]"


def _final_attempt(result: dict[str, Any] | None) -> dict[str, Any]:
    attempts = (result or {}).get("attempts") or []
    return attempts[-1] if attempts else {}


def _twin_from_rca(rca_attempt: dict[str, Any], rca_result: dict[str, Any]) -> dict[str, Any]:
    comps = rca_attempt.get("reward_components") or {}
    gate = rca_result.get("public_rca_twin_gate") or {}
    return {
        "component": "sparse_live_or_offline_twin_verifier",
        "trainable": False,
        "reward_route": comps.get("reward_route") or rca_result.get("reward_route") or gate.get("reward_route"),
        "reproduction_score": comps.get("twin_reproduction_score", gate.get("reproduction_score")),
        "rca_twin_verified": comps.get("rca_twin_verified"),
        "predicted_fault_injection_checked": comps.get("predicted_fault_injection_checked"),
        "reason": comps.get("reason") or gate.get("reason"),
        "reused_identical_hypothesis": comps.get("reused_identical_hypothesis"),
        "telemetry_incomplete": comps.get("telemetry_incomplete") or gate.get("telemetry_incomplete"),
        "twin_mode": comps.get("twin_mode"),
    }


def build_generation_example(
    trajectory: dict[str, Any],
    *,
    scenario_id: str | None = None,
    sync_batch_id: str | None = None,
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Flatten one joint trajectory into a named per-component generation record."""
    rca_result = trajectory.get("rca_result") or {}
    action_result = trajectory.get("action_result") or {}
    rca_attempt = _final_attempt(rca_result)
    action_attempt = _final_attempt(action_result)
    reward = trajectory.get("reward") or {}
    comps = reward.get("components") or {}
    verifier = action_attempt.get("verifier_result") or {}

    rca_instruction = rca_attempt.get("instruction") or ""
    rca_prediction = rca_attempt.get("prediction_text") or rca_result.get("final_prediction") or ""
    action_instruction = action_attempt.get("instruction_prompt") or ""
    action_commands = action_attempt.get("commands") or []

    return {
        "record_type": "joint_training_generation_example_v1",
        "scenario_id": scenario_id or rca_result.get("scenario_id") or action_result.get("scenario_id"),
        "trajectory_id": trajectory.get("trajectory_id"),
        "trajectory_index": trajectory.get("trajectory_index"),
        "sync_batch_id": sync_batch_id,
        "policy_version": policy_version,
        "reward_route": trajectory.get("reward_route"),
        "skipped_action": bool(trajectory.get("skipped_action") or action_result.get("skipped_action")),
        "skip_reason": action_result.get("skip_reason"),
        "success": bool(trajectory.get("trajectory_success") or reward.get("success")),
        "observable_improvement": bool(comps.get("observable_improvement")),
        "full_success": bool(comps.get("full_success")),
        "improvement_credit": comps.get("improvement_credit"),
        "rewards": {
            "system_reward": trajectory.get("system_reward", reward.get("system_reward")),
            "system_quality": trajectory.get("system_quality", reward.get("system_quality")),
            "rca_policy_return": trajectory.get("rca_policy_return", reward.get("rca_policy_return")),
            "action_policy_return": trajectory.get("action_policy_return", reward.get("action_policy_return")),
            "rca_policy_advantage": trajectory.get("rca_policy_advantage"),
            "action_policy_advantage": trajectory.get("action_policy_advantage"),
        },
        "agents": {
            "rca_instruction_policy": {
                "component": "LoRA_RCA_prompt_policy",
                "trainable": True,
                "iteration": rca_attempt.get("iteration"),
                "output": _clip(rca_instruction),
            },
            "rca_solver": {
                "component": "frozen_rca_solver",
                "trainable": False,
                "output": _clip(rca_prediction),
                "parsed": _clip(rca_attempt.get("predicted_faults") or rca_result.get("final_prediction")),
            },
            "twin_verifier": _clip(_twin_from_rca(rca_attempt, rca_result)),
            "action_instruction_policy": {
                "component": "LoRA_Action_prompt_policy",
                "trainable": True,
                "iteration": action_attempt.get("iteration"),
                "output": _clip(action_instruction),
                "invoked": bool(trajectory.get("action_stage_invoked")),
            },
            "action_agent": {
                "component": "frozen_action_agent",
                "trainable": False,
                "output": _clip(action_commands),
                "feedback": _clip(action_attempt.get("feedback")),
            },
            "action_verifier": {
                "component": "twin_action_sla_verifier",
                "trainable": False,
                "resolved": verifier.get("resolved"),
                "sla_restored": verifier.get("sla_restored"),
                "target_sla_restored": verifier.get("target_sla_restored"),
                "target_symptom_reduction": verifier.get("target_symptom_reduction"),
                "global_symptom_reduction": verifier.get("global_symptom_reduction"),
                "reason": verifier.get("reason"),
                "score_reason": (verifier.get("resolution") or {}).get("score_reason"),
            },
        },
    }


def examples_from_group_result(
    result: dict[str, Any],
    *,
    scenario_id: str | None = None,
    sync_batch_id: str | None = None,
    policy_version: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    sid = scenario_id or result.get("scenario_id")
    for trajectory in result.get("trajectories") or []:
        example = build_generation_example(
            trajectory,
            scenario_id=sid,
            sync_batch_id=sync_batch_id,
            policy_version=policy_version,
        )
        rows.append(example)
        trajectory["generation_example"] = example
    return rows


def append_examples_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        for row in examples:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_examples_markdown(path: str | Path, examples: list[dict[str, Any]], *, limit: int = 8) -> None:
    """Write a short human preview of the latest generation examples."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blocks = ["# Training generation examples", ""]
    for example in examples[: max(1, int(limit))]:
        agents = example.get("agents") or {}
        rca_pol = agents.get("rca_instruction_policy") or {}
        rca_sol = agents.get("rca_solver") or {}
        twin = agents.get("twin_verifier") or {}
        act_pol = agents.get("action_instruction_policy") or {}
        act_ag = agents.get("action_agent") or {}
        act_ver = agents.get("action_verifier") or {}
        rewards = example.get("rewards") or {}
        blocks.extend(
            [
                f"## {example.get('scenario_id')} / {example.get('trajectory_id')}",
                "",
                f"- route: `{example.get('reward_route')}`  success: `{example.get('success')}`  "
                f"improved: `{example.get('observable_improvement')}`",
                f"- rewards: system={rewards.get('system_reward')}  "
                f"rca={rewards.get('rca_policy_return')}  action={rewards.get('action_policy_return')}",
                "",
                "### RCA instruction policy (trainable)",
                "",
                "```text",
                str(rca_pol.get("output") or ""),
                "```",
                "",
                "### RCA solver (frozen)",
                "",
                "```text",
                str(rca_sol.get("output") or ""),
                "```",
                "",
                "### Twin verifier (frozen environment)",
                "",
                "```json",
                json.dumps(twin, indent=2, sort_keys=True, default=str),
                "```",
                "",
                "### Action instruction policy (trainable)",
                "",
                "```text",
                str(act_pol.get("output") or "(action stage skipped)"),
                "```",
                "",
                "### Action agent commands (frozen)",
                "",
                "```text",
                "\n".join(str(x) for x in (act_ag.get("output") or []) or ["(none)"]),
                "```",
                "",
                "### Action verifier",
                "",
                "```json",
                json.dumps(act_ver, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        )
    target.write_text("\n".join(blocks), encoding="utf-8")
