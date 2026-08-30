from __future__ import annotations

from functools import lru_cache
from typing import Any

from .schemas import FaultLabel, normalize_fault_mechanism, normalize_fault_type


def normalize_service_name(service: str | None) -> str:
    return str(service or "").strip().lower().replace("_", "-")


def _neighbors(full_state: dict[str, Any], service: str) -> set[str]:
    target = normalize_service_name(service)
    out = {target}
    for edge in (full_state.get("graph", {}) or {}).get("edges", []) or []:
        if isinstance(edge, dict):
            src, dst = edge.get("src"), edge.get("dst")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        src_n, dst_n = normalize_service_name(src), normalize_service_name(dst)
        if src_n == target and dst_n:
            out.add(dst_n)
        if dst_n == target and src_n:
            out.add(src_n)
    return out


def _canonical_key(label: FaultLabel) -> str:
    base = f"{normalize_service_name(label.service)}::{normalize_fault_type(label.fault_type or label.fault_family)}"
    mechanism = normalize_fault_mechanism(label.fault_mechanism)
    if mechanism:
        return f"{base}::{mechanism}::{label.variant_name or 'default'}"
    return base


def _pair_score(gt: FaultLabel, pred: FaultLabel, full_state: dict[str, Any]) -> dict[str, Any]:
    gt_service = normalize_service_name(gt.service)
    pred_service = normalize_service_name(pred.service)
    service_exact = 1.0 if gt_service == pred_service else 0.0
    fault_exact = 1.0 if normalize_fault_type(gt.fault_type) == normalize_fault_type(pred.fault_type) else 0.0
    gt_mechanism = normalize_fault_mechanism(gt.fault_mechanism)
    pred_mechanism = normalize_fault_mechanism(pred.fault_mechanism)
    mechanism_available = bool(gt_mechanism)
    mechanism_exact = 1.0 if mechanism_available and gt_mechanism == pred_mechanism else 0.0
    variant_exact = 1.0 if mechanism_exact and (gt.variant_name or "default") == (pred.variant_name or "default") else 0.0
    neighborhood = 1.0 if pred_service in _neighbors(full_state, gt.service) else 0.0
    if mechanism_available:
        score = (
            0.40 * service_exact
            + 0.20 * fault_exact
            + 0.15 * mechanism_exact
            + 0.10 * variant_exact
            + 0.15 * neighborhood
        )
    else:
        # Legacy/unsupported evaluator labels retain dense service/type credit.
        score = 0.45 * service_exact + 0.40 * fault_exact + 0.15 * neighborhood
    return {
        "score": score,
        "service_exact": service_exact,
        "fault_type_exact": fault_exact,
        "fault_mechanism_exact": mechanism_exact,
        "variant_exact": variant_exact,
        "mechanism_available_in_ground_truth": mechanism_available,
        "neighborhood_match": neighborhood,
        "predicted_key": _canonical_key(pred),
    }


def optimal_match(
    gt_labels: list[FaultLabel],
    pred_labels: list[FaultLabel],
    full_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """Maximum-weight one-to-one GT/prediction matching.

    The previous greedy matcher depended on ground-truth ordering and could assign
    a suboptimal pair in multifault incidents. This bitmask dynamic program finds
    the exact maximum-score assignment while allowing unmatched GT labels when
    fewer predictions are present. RCA root-cause sets are small, so the exact
    O(|GT| * |PRED| * 2^|PRED|) solver is inexpensive here.
    """
    if not gt_labels:
        return [], 0.0
    if not pred_labels:
        return [], 0.0

    pair_matrix = [
        [_pair_score(gt, pred, full_state) for pred in pred_labels]
        for gt in gt_labels
    ]

    @lru_cache(maxsize=None)
    def solve(gt_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if gt_index >= len(gt_labels):
            return 0.0, ()

        # Leaving this GT unmatched contributes zero and is necessary when there
        # are fewer predictions than GT labels.
        best_score, best_pairs = solve(gt_index + 1, used_mask)

        for pred_index in range(len(pred_labels)):
            bit = 1 << pred_index
            if used_mask & bit:
                continue
            tail_score, tail_pairs = solve(gt_index + 1, used_mask | bit)
            candidate_score = float(pair_matrix[gt_index][pred_index]["score"]) + tail_score
            candidate_pairs = ((gt_index, pred_index),) + tail_pairs
            if candidate_score > best_score + 1e-12:
                best_score, best_pairs = candidate_score, candidate_pairs
            elif abs(candidate_score - best_score) <= 1e-12 and candidate_pairs < best_pairs:
                # Deterministic tie-break for reproducible audits.
                best_score, best_pairs = candidate_score, candidate_pairs

        return best_score, best_pairs

    total_score, assignment = solve(0, 0)
    matches: list[dict[str, Any]] = []
    for gt_index, pred_index in assignment:
        row = dict(pair_matrix[gt_index][pred_index])
        row["ground_truth_index"] = gt_index
        row["prediction_index"] = pred_index
        matches.append(row)

    pair_score = total_score / len(gt_labels)
    return matches, max(0.0, min(1.0, pair_score))


def greedy_match(gt_labels: list[FaultLabel], pred_labels: list[FaultLabel], full_state: dict[str, Any]):
    """Backward-compatible alias; matching is now exact rather than greedy."""
    return optimal_match(gt_labels, pred_labels, full_state)


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
    """Reward one RCA attempt.

    Hidden labels may shape the evaluator-side scalar reward, but they must never
    appear in policy-visible retry feedback. The only retry feedback emitted by
    this module is based on output validity, repeated self-history, and the
    independent twin reproduction signal.
    """
    matches, pair_score = optimal_match(gt_labels, pred_labels, full_state)
    twin_score = float((twin_result or {}).get("reproduction_score", 0.0) or 0.0)
    twin_score = max(0.0, min(1.0, twin_score))
    exact = exact_set_match(gt_labels, pred_labels)
    count_mismatch = abs(len(pred_labels) - len(gt_labels))

    valid_format = bool(pred_labels) and not invalid_format
    format_reward = 0.20 if valid_format else -1.00
    pair_match_reward = 2.00 * pair_score
    exact_set_bonus = 1.00 if exact else 0.00
    twin_reproduction_reward = 1.00 * twin_score
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

    # Exact label match remains a private evaluator success flag. Joint end-to-end
    # rollouts may still continue into the action stage even when this is false.
    success = bool(exact)

    components = {
        "format_reward": round(format_reward, 4),
        "pair_score": round(pair_score, 4),
        "pair_match_reward": round(pair_match_reward, 4),
        "matching_algorithm": "exact_max_weight_bipartite_dp",
        "exact_set_match": exact,
        "exact_label_required_for_local_rca_success": True,
        "exact_set_bonus": round(exact_set_bonus, 4),
        "twin_reproduction_score": round(twin_score, 4),
        "twin_reproduction_reward": round(twin_reproduction_reward, 4),
        "twin_score_used_as_reward": True,
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
        "matches_private_evaluator": matches,
        "service_normalization": "case_insensitive_underscore_to_dash",
    }
    return {
        "reward": round(float(reward), 4),
        "success": success,
        "components": components,
        "feedback": non_leaking_feedback(components),
    }


def non_leaking_feedback(c: dict[str, Any]) -> str:
    """Policy-visible feedback derived only from public/self/twin signals."""
    parts = []
    if c.get("invalid_format"):
        parts.append("Output format invalid; use one service::fault_type per line.")

    twin_score = float(c.get("twin_reproduction_score", 0.0) or 0.0)
    if twin_score < 0.20:
        parts.append("Counterfactual twin reproduction is very low; choose a different upstream service/mechanism hypothesis.")
    elif twin_score < 0.50:
        parts.append("Counterfactual twin reproduction is weak; refine the service/mechanism using the observed telemetry.")
    else:
        parts.append("Counterfactual twin reproduction is comparatively strong; keep the causal explanation focused and avoid adding unsupported roots.")

    if c.get("repeated_wrong_guess"):
        parts.append("Avoid repeating the same previous hypothesis.")
    return " ".join(parts)


def terminal_rca_failure_penalty(num_iterations: int = 5) -> dict[str, Any]:
    return {
        "reward": -2.0,
        "success": False,
        "components": {"terminal_failure": True, "num_iterations": num_iterations},
        "feedback": "RCA iteration budget exhausted. Change the telemetry-based causal strategy rather than repeating prior hypotheses.",
    }
