from __future__ import annotations

from typing import Any

from .telemetry_comparator import graph_neighborhood, symptom_signature


def _norm_service(value: Any) -> str:
    return str(value or "").strip()


def _norm_fault_type(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    aliases = {
        "pod_failure": "infra_failure",
        "pod_kill": "infra_failure",
        "container_kill": "infra_failure",
        "assign_non_existent_node": "infra_failure",
        "assign_to_non_existent_node": "infra_failure",
        "scale_pod": "infra_failure",
        "auth": "auth_failure",
        "revoke_auth": "auth_failure",
        "auth_miss_mongodb": "auth_failure",
        "mongo": "dependency_failure",
        "mongodb": "dependency_failure",
        "network_delay": "latency_degradation",
        "delay": "latency_degradation",
        "latency": "latency_degradation",
        "network_loss": "network_failure",
        "loss": "network_failure",
        "k8s_target_port": "config_error",
        "misconfig": "config_error",
        "config": "config_error",
        "cpu": "resource_exhaustion",
        "memory": "resource_exhaustion",
        "oom": "resource_exhaustion",
        "wrong_bin": "unknown",
        "wrong binary": "unknown",
    }
    canonical = {
        "infra_failure", "auth_failure", "dependency_failure", "resource_exhaustion",
        "latency_degradation", "network_failure", "config_error", "unknown",
    }
    if text in canonical:
        return text
    for pattern, mapped in aliases.items():
        if pattern in text:
            return mapped
    return "unknown"


def _graph_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for edge in (state.get("graph", {}) or {}).get("edges", []) or []:
        if isinstance(edge, dict):
            src, dst = edge.get("src"), edge.get("dst")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        if src and dst:
            out.append((str(src), str(dst)))
    return out


def _touching_edges(state: dict[str, Any], service: str) -> list[str]:
    return [f"{src}->{dst}" for src, dst in _graph_edges(state) if src == service or dst == service]


def _callers(state: dict[str, Any], service: str) -> set[str]:
    return {src for src, dst in _graph_edges(state) if dst == service}


def _dependencies(state: dict[str, Any], service: str) -> set[str]:
    return {dst for src, dst in _graph_edges(state) if src == service}


def predict_fault_signature(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a counterfactual symptom footprint from prediction + topology only.

    This deliberately does not inspect ground-truth fault labels or mechanism text in
    observed logs. The mechanism determines which observable channels should change;
    the graph determines where those changes may propagate.
    """
    out = {
        "degraded_services": set(),
        "failed_edges": set(),
        "top_error_services": set(),
        "trace_sources": set(),
        "trace_targets": set(),
        "metric_anomaly_services": set(),
        "affected_services": set(),
    }

    for fault in predicted_faults or []:
        service = _norm_service(fault.get("service"))
        if not service:
            continue
        ft = _norm_fault_type(fault.get("fault_type") or fault.get("fault_family"))
        callers = _callers(state, service)
        deps = _dependencies(state, service)
        neighbors = graph_neighborhood(state, service, radius=1) or {service}
        touching = _touching_edges(state, service)

        if ft == "infra_failure":
            out["degraded_services"].add(service)
            out["top_error_services"].add(service)
            out["failed_edges"].update(touching)
            out["trace_targets"].add(service)
            out["trace_sources"].update(callers)
            out["affected_services"].update(neighbors)
        elif ft == "auth_failure":
            out["top_error_services"].add(service)
            out["top_error_services"].update(callers)
            out["trace_targets"].add(service)
            out["trace_sources"].update(callers)
            out["failed_edges"].update(f"{src}->{service}" for src in callers)
            out["affected_services"].update({service, *callers})
        elif ft == "dependency_failure":
            out["top_error_services"].add(service)
            out["top_error_services"].update(callers)
            out["trace_targets"].add(service)
            out["trace_sources"].update(callers)
            out["failed_edges"].update(f"{src}->{service}" for src in callers)
            out["affected_services"].update({service, *callers})
        elif ft == "latency_degradation":
            out["metric_anomaly_services"].add(service)
            out["trace_targets"].add(service)
            out["trace_sources"].update(callers)
            out["top_error_services"].update(callers)
            out["affected_services"].update({service, *callers})
        elif ft == "network_failure":
            out["failed_edges"].update(touching)
            out["trace_sources"].add(service)
            out["trace_sources"].update(callers)
            out["trace_targets"].add(service)
            out["trace_targets"].update(deps)
            out["top_error_services"].update({service, *callers})
            out["affected_services"].update(neighbors)
        elif ft == "config_error":
            out["degraded_services"].add(service)
            out["top_error_services"].add(service)
            out["top_error_services"].update(callers)
            out["trace_targets"].add(service)
            out["failed_edges"].update(f"{src}->{service}" for src in callers)
            out["affected_services"].update({service, *callers})
        elif ft == "resource_exhaustion":
            out["degraded_services"].add(service)
            out["metric_anomaly_services"].add(service)
            out["top_error_services"].add(service)
            out["trace_targets"].add(service)
            out["affected_services"].update({service, *callers})
        else:
            out["top_error_services"].add(service)
            out["affected_services"].add(service)

    return {key: sorted(value) for key, value in out.items()}


def _set_score(expected: list[str], observed: list[str]) -> float:
    e = set(expected or [])
    o = set(observed or [])
    if not e and not o:
        return 1.0
    if not e:
        return 0.5
    if not o:
        return 0.0
    precision = len(e & o) / len(e)
    recall = len(e & o) / len(o)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _presence_match(expected: list[str], observed: list[str]) -> float:
    e = bool(expected)
    o = bool(observed)
    if e == o:
        return 1.0
    return 0.0


def score_counterfactual_reproduction(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare predicted counterfactual fault footprint with observed incident.

    The score is independent of ground-truth labels. It rewards both location
    overlap and mechanism-specific channel agreement, so a same-service prediction
    with the wrong mechanism should activate the wrong telemetry channels.
    """
    observed = symptom_signature(state)
    predicted = predict_fault_signature(state, predicted_faults)
    if not predicted_faults:
        return {
            "mode": "counterfactual_offline_twin_v1",
            "reproduction_score": 0.0,
            "predicted_signature": predicted,
            "observed_signature": observed,
            "reason": "empty_prediction",
        }

    weights = {
        "degraded_services": 0.16,
        "failed_edges": 0.22,
        "top_error_services": 0.14,
        "trace_sources": 0.10,
        "trace_targets": 0.10,
        "metric_anomaly_services": 0.18,
        "affected_services": 0.10,
    }
    channel_scores: dict[str, float] = {}
    channel_presence: dict[str, float] = {}
    for channel in weights:
        channel_scores[channel] = _set_score(predicted.get(channel, []), observed.get(channel, []))
        channel_presence[channel] = _presence_match(predicted.get(channel, []), observed.get(channel, []))

    overlap_score = sum(weights[c] * channel_scores[c] for c in weights)
    mechanism_presence_score = sum(weights[c] * channel_presence[c] for c in weights)

    predicted_services = {
        _norm_service(f.get("service")) for f in predicted_faults if _norm_service(f.get("service"))
    }
    observed_services = set(observed.get("affected_services", []) or [])
    service_support = 0.0
    if predicted_services:
        service_support = len(predicted_services & observed_services) / len(predicted_services)

    score = 0.58 * overlap_score + 0.30 * mechanism_presence_score + 0.12 * service_support
    return {
        "mode": "counterfactual_offline_twin_v1",
        "reproduction_score": round(max(0.0, min(1.0, score)), 4),
        "counterfactual_overlap_score": round(overlap_score, 4),
        "mechanism_channel_presence_score": round(mechanism_presence_score, 4),
        "predicted_service_support": round(service_support, 4),
        "channel_scores": {k: round(v, 4) for k, v in channel_scores.items()},
        "channel_presence": {k: round(v, 4) for k, v in channel_presence.items()},
        "predicted_signature": predicted,
        "observed_signature": observed,
        "uses_oracle_labels_for_score": False,
        "uses_full_state_for_score": False,
    }
