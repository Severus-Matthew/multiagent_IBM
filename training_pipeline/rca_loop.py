from __future__ import annotations

import json
from typing import Any, Protocol
from .ground_truth import labels_from_full_state
from .rca_reward import rca_reward, terminal_rca_failure_penalty
from .schemas import RCAAttempt, approx_token_count, parse_fault_lines

class RCAInstructionPolicy(Protocol):
    def generate_instruction(self, compressed_state: dict[str, Any], history: list[dict[str, Any]], iteration: int) -> str: ...

class RCASolver(Protocol):
    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str: ...

class HeuristicRCAInstructionPolicy:
    """Debug baseline. Replace with trainable Qwen/LoRA policy."""
    def generate_instruction(self, compressed_state: dict[str, Any], history: list[dict[str, Any]], iteration: int) -> str:
        retry = " Avoid repeating previous wrong guesses." if history else ""
        return ("Read only the redacted telemetry. Prioritize direct Kubernetes service-health signals "
                "such as Pending pods, no ready endpoints, crashloops, and failed scheduling; then use logs/traces "
                "to separate root cause from downstream cascade. Output only service::fault_type, one root cause per line."
                + retry)

class HeuristicRCASolver:
    """No-LLM smoke-test baseline using redacted state only."""
    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        candidates = []
        for svc, info in (compressed_state.get("system", {}) or {}).items():
            health = (info.get("health", {}) if isinstance(info, dict) else {}) or {}
            score = 0.0
            if health.get("infra_issue_flag"): score += 2.0
            if health.get("pods_unready", 0) > 0: score += 1.0
            status = str(health.get("status", "")).lower()
            if "no_ready" in status or "pending" in status or "crash" in status: score += 1.0
            if score > 0: candidates.append((score, svc, "infra_failure"))
        if candidates:
            candidates.sort(reverse=True)
            return f"{candidates[0][1]}::{candidates[0][2]}"
        top = ((compressed_state.get("llm_view", {}) or {}).get("top_log_error_services") or [{}])[0]
        if top.get("service"):
            return f"{top['service']}::dependency_failure"
        services = compressed_state.get("services", []) or []
        return f"{services[0] if services else 'unknown'}::unknown"


def run_rca_self_prompting_loop(full_state: dict[str, Any], compressed_state: dict[str, Any],
                                instruction_policy: RCAInstructionPolicy, solver: RCASolver,
                                twin_validator=None, max_iterations: int = 5) -> dict[str, Any]:
    gt_labels = labels_from_full_state(full_state)
    attempts: list[RCAAttempt] = []
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    for iteration in range(max_iterations):
        instruction = instruction_policy.generate_instruction(compressed_state, history, iteration)
        prediction_text = solver.solve(compressed_state, instruction)
        pred_labels = parse_fault_lines(prediction_text)
        key = "\n".join(sorted(x.canonical_key() for x in pred_labels))
        repeated = bool(key) and key in seen
        seen.add(key)
        twin_result = None
        if twin_validator is not None and pred_labels:
            twin_result = twin_validator.validate_rca_prediction(full_state, compressed_state, pred_labels)
        reward_obj = rca_reward(full_state, gt_labels, pred_labels,
                                instruction_tokens=approx_token_count(instruction),
                                iteration_index=iteration, twin_result=twin_result,
                                invalid_format=not pred_labels, repeated_wrong_guess=repeated)
        attempt = RCAAttempt(iteration=iteration, instruction=instruction, prediction_text=prediction_text,
                             predicted_faults=pred_labels, reward=reward_obj["reward"],
                             reward_components=reward_obj["components"], success=reward_obj["success"],
                             feedback=reward_obj["feedback"],
                             token_counts={"instruction_tokens": approx_token_count(instruction),
                                           "prediction_tokens": approx_token_count(prediction_text)})
        attempts.append(attempt)
        history.append({"iteration": iteration, "prediction": prediction_text,
                        "reward": reward_obj["reward"], "reward_components": reward_obj["components"],
                        "feedback": reward_obj["feedback"]})
        if attempt.success:
            break
    terminal = None if attempts and attempts[-1].success else terminal_rca_failure_penalty(max_iterations)
    return {"scenario_id": full_state.get("scenario_id") or compressed_state.get("scenario_id"),
            "success": bool(attempts and attempts[-1].success),
            "attempts": [a.to_dict() for a in attempts],
            "final_prediction": attempts[-1].prediction_text if attempts else "",
            "ground_truth_summary": {"num_faults": len(gt_labels), "labels": [x.to_dict() for x in gt_labels]},
            "terminal": terminal}


def main() -> None:
    import argparse
    from .data_loader import iter_scenarios
    ap = argparse.ArgumentParser(description="Smoke-test Stage 1 RCA loop")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rows = []
    policy = HeuristicRCAInstructionPolicy(); solver = HeuristicRCASolver()
    for rec in iter_scenarios(args.processed_states, limit=args.limit):
        rows.append(run_rca_self_prompting_loop(rec.full_state, rec.compressed_state, policy, solver))
    print(json.dumps({"total": len(rows), "passed": sum(r["success"] for r in rows)}, indent=2))
    if args.output:
        with open(args.output, "w") as f: json.dump(rows, f, indent=2)

if __name__ == "__main__":
    main()
