from __future__ import annotations

from typing import Any
from .schemas import FaultLabel, normalize_fault_type


def _neighbors(full_state: dict[str, Any], service: str) -> set[str]:
    out = {service}
    for edge in (full_state.get("graph", {}) or {}).get("edges", []) or []:
        src, dst = edge.get("src"), edge.get("dst")
        if src == service and dst:
            out.add(dst)
        if dst == service and src:
            out.add(src)
    return out


def _pair_score(gt: FaultLabel, pred: FaultLabel, full_state: dict[str, Any]) -> dict[str, Any]:
    service_exact = 1.0 if gt.service == pred.service else 0.0
    fault_exact = 1.0 if normalize_fault_type(gt.fault_type) == normalize_fault_type(pred.fault_type) else 0.0
    neighborhood = 1.0 if pred.service in _neighbors(full_state, gt.service) else 0.0
    score = 0.60 * service_exact + 0.30 * fault_exact + 0.10 * neighborhood
    return {"score": score, "service_exact": service_exact, "fault_type_exact": fault_exact,
            "neighborhood_match": neighborhood, "gt": gt.to_dict(), "pred": pred.to_dict()}


def greedy_match(gt_labels: list[FaultLabel], pred_labels: list[FaultLabel], full_state: dict[str, Any]):
    remaining = set(range(len(pred_labels)))
    matches = []
    for gt in gt_labels:
        best = None; best_j = None
        for j in remaining:
            cand = _pair_score(gt, pred_labels[j], full_state)
            if best is None or cand["score"] > best["score"]:
                best = cand; best_j = j
        if best is not None and best_j is not None:
            remaining.remove(best_j); matches.append(best)
    if not gt_labels:
        return [], 0.0
    matched = sum(m["score"] for m in matches) / len(gt_labels)
    count_penalty = 0.15 * abs(len(pred_labels) - len(gt_labels))
    return matches, max(0.0, matched - count_penalty)


def rca_reward(full_state: dict[str, Any], gt_labels: list[FaultLabel], pred_labels: list[FaultLabel],
               instruction_tokens: int = 0, iteration_index: int = 0,
               twin_result: dict[str, Any] | None = None, invalid_format: bool = False,
               repeated_wrong_guess: bool = False) -> dict[str, Any]:
    matches, pair_score = greedy_match(gt_labels, pred_labels, full_state)
    twin_score = float((twin_result or {}).get("reproduction_score", 0.0) or 0.0)
    exact = {x.canonical_key() for x in gt_labels} == {x.canonical_key() for x in pred_labels}
    reward = 2.0 * pair_score + 0.5 * twin_score - 0.10 * iteration_index - 0.001 * max(0, instruction_tokens)
    if repeated_wrong_guess:
        reward -= 0.05
    if invalid_format:
        reward -= 0.50
    if exact:
        reward += 1.0
    components = {"pair_score": round(pair_score, 4), "twin_reproduction_score": round(twin_score, 4),
                  "exact_set_match": exact, "num_gt": len(gt_labels), "num_pred": len(pred_labels),
                  "instruction_tokens": instruction_tokens, "iteration_index": iteration_index,
                  "invalid_format": invalid_format, "repeated_wrong_guess": repeated_wrong_guess,
                  "matches": matches}
    return {"reward": round(float(reward), 4), "success": exact,
            "components": components, "feedback": non_leaking_feedback(components)}


def non_leaking_feedback(c: dict[str, Any]) -> str:
    if c.get("exact_set_match"):
        return "RCA prediction matched oracle labels and sufficient symptom evidence."
    parts = []
    if c.get("invalid_format"):
        parts.append("Output format invalid; use one service::fault_type per line.")
    if c.get("pair_score", 0.0) < 0.5:
        parts.append("Prediction did not align well with oracle service/fault structure.")
    if c.get("twin_reproduction_score", 0.0) < 0.5:
        parts.append("Injected prediction did not reproduce original symptoms well.")
    return " ".join(parts or ["Prediction was partially aligned but not exact."])


def terminal_rca_failure_penalty(num_iterations: int = 5) -> dict[str, Any]:
    return {"reward": -2.0, "success": False,
            "components": {"terminal_failure": True, "num_iterations": num_iterations},
            "feedback": "RCA failed after the iteration budget. Try a different evidence strategy."}
