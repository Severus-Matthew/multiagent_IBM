from __future__ import annotations

from typing import Any


def action_reward(commands: list[str], safety: dict[str, Any], verifier_result: dict[str, Any],
                  instruction_tokens: int = 0, iteration_index: int = 0) -> dict[str, Any]:
    resolved = 1.0 if verifier_result.get("resolved") else 0.0
    reduction = float(verifier_result.get("symptom_reduction", 0.0) or 0.0)
    safe = 1.0 if safety.get("safe") else 0.0
    has_verify = any("rollout status" in c or "kubectl get" in c for c in commands)
    reward = (2.0 * resolved + 0.8 * reduction + 0.4 * safe + 0.3 * (1.0 if has_verify else 0.0)
              - 0.2 * (0.0 if safety.get("safe") else 1.0) - 0.05 * len(commands)
              - 0.001 * max(0, instruction_tokens) - 0.10 * iteration_index)
    success = bool(resolved and safety.get("safe"))
    if success:
        reward += 1.0
    if not commands:
        reward -= 1.0
    return {"reward": round(float(reward), 4), "success": success,
            "components": {"resolved": bool(resolved), "symptom_reduction": reduction,
                           "safe": safety.get("safe"), "num_commands": len(commands),
                           "has_verification_command": has_verify,
                           "instruction_tokens": instruction_tokens, "iteration_index": iteration_index},
            "feedback": feedback(safety, verifier_result, commands)}


def feedback(safety: dict[str, Any], verifier_result: dict[str, Any], commands: list[str]) -> str:
    if not safety.get("safe"):
        return "One or more commands were unsafe or unsupported. Use scoped kubectl/helm commands only."
    if not commands:
        return "No executable commands were produced."
    if verifier_result.get("resolved"):
        return "Commands resolved twin symptoms and restored SLA."
    red = verifier_result.get("symptom_reduction", 0.0)
    if red > 0:
        return f"Commands were valid but only partially reduced symptoms (reduction={red:.2f})."
    return "Commands were valid but did not improve twin symptoms."


def terminal_action_failure_penalty(num_iterations: int = 5) -> dict[str, Any]:
    return {"reward": -2.0, "success": False,
            "components": {"terminal_failure": True, "num_iterations": num_iterations},
            "feedback": "Action loop failed after the iteration budget. Try a different remediation strategy."}
