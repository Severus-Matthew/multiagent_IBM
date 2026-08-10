from __future__ import annotations

from typing import Any
from .schemas import FaultLabel, normalize_fault_type


def normalize_service_name(service: str | None) -> str:
    return str(service or "").strip().lower().replace("_", "-")


def _neighbors(full_state: dict[str, Any], service: str) -> set[str]:
    target = normalize_service_name(service)
    out = {target}
    for edge in (full_state.get("graph", {}) or {}).get("edges", []) or []:
        src, dst = edge.get("src"), edge.get("dst")
        src_n, dst_n = normalize_service_name(src), normalize_service_name(dst)
        if src_n == target and dst:
            out.add(dst_n)
        if dst_n == target and src:
            out.add(src_n)
    return out


def _canonical_key(label: FaultLabel) -> str:
    return f"{normalize_service_name(label.service)}::{normalize_fault_type(label.fault_type or label.fault_family)}"


def _pair_score(gt: FaultLabel, pred: FaultLabel, full_state: dict[str, Any]) -> dict[str, Any]:
    gt_service = normalize_service_name(gt.service)
    pred_service = normalize_service_name(pred.service)
    service_exact = 1.0 if gt_service == pred_service else 0.0
    fault_exact = 1.0 if normalize_fault_type(gt.fault_type) == normalize_fault_type(pred.fault_type) else 0.0
    neighborhood = 1.0 if pred_service in _neighbors(full_state, gt.service) else 0.0
    score = 0.55 * service_exact + 0.30 * fault_exact + 0.15 * neighborhood
    return {
        "score": score,
        "service_exact": service_exact,
        "fault_type_exact": fault_exact,
        "neighborhood_match": neighborhood,
        "predicted_key": _canonical_key(pred),
    }


def greedy_match(gt_labels: list[FaultLabel], pred_labels: list[FaultLabel], full_state: dict[str, Any]):
    """Return one-to-one greedy match diagnostics without exposing GT names in components."""
    remaining = set(range(len(pred_labels)))
    matches = []
    for gt in gt_labels:
        best = None
        best_j = None
        for j in remaining:
            cand = _pair_score(gt, pred_labels[j], full_state)
            if best is None or cand["score"] > best["score"]:
                best = cand
                best_j = j
        if best is not None and best_j is not None:
            remaining.remove(best_j)
            matches.append(best)
    if not gt_labels:
        return [], 0.0
    pair_score = sum(m["score"] for m in matches) / len(gt_labels)
    return matches, max(0.0, min(1.0, pair_score))


def exact_set_match(gt_labels: list[FaultLabel], pred_labels: list[FaultLabel]) -> bool:
    return {_canonical_key(x) for x in gt_labels} == {_canonical_key(x) for x in pred_labels}


def rca_reward(
    full_state: dict[str, Any],
    gt_labels: list[FaultLabel],
    pred_labels: list[FaultLabel],
    instruction_tokens: int = 0,
    iteration_index: int = 0,
    twin_result: dict[str, Any] | None = None,
    invalid_format: bool = False,
    repeated_wrong_guess: bool = False,
) -> dict[str, Any]:
    """Reward one RCA attempt."""
    matches, pair_score = greedy_match(gt_labels, pred_labels, full_state)
    twin_score = float((twin_result or {}).get("reproduction_score", 0.0) or 0.0)
    twin_score = max(0.0, min(1.0, twin_score))
    exact = exact_set_match(gt_labels, pred_labels)
    count_mismatch = abs(len(pred_labels) - len(gt_labels))

    valid_format = bool(pred_labels) and not invalid_format
    format_reward = 0.20 if valid_format else -1.00
    pair_match_reward = 2.00 * pair_score
    exact_set_bonus = 1.00 if exact else 0.00
    twin_reproduction_reward = 0.00
    count_mismatch_penalty = 0.40 * count_mismatch
    repeated_wrong_guess_penalty = 0.25 if repeated_wrong_guess else 0.00
    iteration_penalty = 0.10 * iteration_index
    token_penalty = 0.001 * max(0, instruction_tokens - 120)

    reward = (
        format_reward
        + pair_match_reward
        + exact_set_bonus
        + twin_reproduction_reward
        - count_mismatch_penalty
        - repeated_wrong_guess_penalty
        - iteration_penalty
        - token_penalty
    )

    soft_success = False
    success = bool(exact)

    components = {
        "format_reward": round(format_reward, 4),
        "pair_score": round(pair_score, 4),
        "pair_match_reward": round(pair_match_reward, 4),
        "exact_set_match": exact,
        "exact_label_required_for_success": True,
        "exact_set_bonus": round(exact_set_bonus, 4),
        "twin_reproduction_score": round(twin_score, 4),
        "twin_reproduction_reward": round(twin_reproduction_reward, 4),
        "twin_score_diagnostic_only": True,
        "count_mismatch": count_mismatch,
        "count_mismatch_penalty": round(count_mismatch_penalty, 4),
        "repeated_wrong_guess": repeated_wrong_guess,
        "repeated_wrong_guess_penalty": round(repeated_wrong_guess_penalty, 4),
        "instruction_tokens": instruction_tokens,
        "iteration_index": iteration_index,
        "iteration_penalty": round(iteration_penalty, 4),
        "token_penalty": round(token_penalty, 4),
        "invalid_format": invalid_format,
        "num_gt": len(gt_labels),
        "num_pred": len(pred_labels),
        "soft_success": soft_success,
        "matches_public": matches,
        "service_normalization": "case_insensitive_underscore_to_dash",
    }
    return {
        "reward": round(float(reward), 4),
        "success": success,
        "components": components,
        "feedback": non_leaking_feedback(components),
    }


def non_leaking_feedback(c: dict[str, Any]) -> str:
    """Feedback visible to the trainable policy. Never reveal the true service/fault."""
    if c.get("exact_set_match"):
        return "RCA prediction explains the observed incident pattern sufficiently."

    parts = []
    if c.get("invalid_format"):
        parts.append("Output format invalid; use one service::fault_type per line.")
    if c.get("count_mismatch", 0) > 0:
        parts.append("Predicted number of root causes does not match the incident structure.")
    if c.get("pair_score", 0.0) < 0.5:
        parts.append("Prediction does not align with the strongest service/fault evidence pattern.")
    elif c.get("pair_score", 0.0) < 0.9:
        parts.append("Prediction is partially aligned but not specific enough.")
    if c.get("twin_reproduction_score", 0.0) < 0.5:
        parts.append("Injected prediction does not reproduce the original symptom pattern well.")
    if c.get("repeated_wrong_guess"):
        parts.append("Avoid repeating the same unsuccessful guess.")
    return " ".join(parts or ["Prediction was close but did not satisfy the verifier."])


def terminal_rca_failure_penalty(num_iterations: int = 5) -> dict[str, Any]:
    return {
        "reward": -2.0,
        "success": False,
        "components": {"terminal_failure": True, "num_iterations": num_iterations},
        "feedback": "RCA failed after the iteration budget. Try a different non-leaking evidence strategy.",
    }
