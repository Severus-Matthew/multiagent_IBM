from __future__ import annotations

from typing import Any


def _jaccard(a, b) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def symptom_signature(state: dict[str, Any]) -> dict[str, Any]:
    degraded = []
    for svc, info in (state.get("system", {}) or {}).items():
        health = info.get("health", info) if isinstance(info, dict) else {}
        if health.get("infra_issue_flag") or health.get("pods_unready", 0) > 0 or health.get("crashloop_count", 0) > 0:
            degraded.append(svc)
    failed_edges = []
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    for edge, feats in per_edge.items():
        if isinstance(feats, dict) and (feats.get("error_ratio", 0) > 0.2 or feats.get("is_suspicious")):
            failed_edges.append(edge)
    error_services = []
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if item.get("service"):
            error_services.append(item["service"])
    return {"degraded_services": sorted(set(degraded)),
            "failed_edges": sorted(set(failed_edges)),
            "top_error_services": sorted(set(error_services))}


def compare_symptoms(original_state: dict[str, Any], twin_state: dict[str, Any]) -> dict[str, Any]:
    orig = symptom_signature(original_state); twin = symptom_signature(twin_state)
    degraded = _jaccard(orig["degraded_services"], twin["degraded_services"])
    edges = _jaccard(orig["failed_edges"], twin["failed_edges"])
    logs = _jaccard(orig["top_error_services"], twin["top_error_services"])
    score = 0.45 * degraded + 0.35 * edges + 0.20 * logs
    return {"reproduction_score": round(score, 4), "degraded_service_overlap": round(degraded, 4),
            "trace_edge_overlap": round(edges, 4), "log_error_service_overlap": round(logs, 4),
            "original_signature": orig, "twin_signature": twin}


def score_resolution(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    before = symptom_signature(before_state); after = symptom_signature(after_state)
    before_count = len(before["degraded_services"]) + len(before["failed_edges"]) + len(before["top_error_services"])
    after_count = len(after["degraded_services"]) + len(after["failed_edges"]) + len(after["top_error_services"])
    reduction = 1.0 if before_count == 0 and after_count == 0 else max(0.0, min(1.0, (before_count - after_count) / max(before_count, 1)))
    return {"symptom_reduction": round(reduction, 4), "before_signature": before,
            "after_signature": after, "resolved": after_count == 0 or reduction >= 0.95}
