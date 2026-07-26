from __future__ import annotations

from typing import Any


def _jaccard(a, b) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _coverage(predicted, observed) -> float:
    predicted_set, observed_set = set(predicted or []), set(observed or [])
    if not observed_set:
        return 0.0
    if not predicted_set:
        return 0.0
    return len(predicted_set & observed_set) / len(observed_set)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_fault_type(text: str | None) -> str:
    value = str(text or "unknown").strip().lower()
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
        "wrong_bin": "config_error",
        "cpu": "resource_exhaustion",
        "memory": "resource_exhaustion",
        "oom": "resource_exhaustion",
    }
    canonical = {
        "infra_failure", "auth_failure", "dependency_failure", "resource_exhaustion",
        "latency_degradation", "network_failure", "config_error", "unknown",
    }
    if value in canonical:
        return value
    for pat, mapped in aliases.items():
        if pat in value:
            return mapped
    return "unknown"


def _graph_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    edges = []
    graph_edges = (state.get("graph", {}) or {}).get("edges", []) or []
    for e in graph_edges:
        if isinstance(e, dict):
            src, dst = e.get("src"), e.get("dst")
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            src, dst = e[0], e[1]
        else:
            continue
        if src and dst:
            edges.append((str(src), str(dst)))
    return edges


def graph_neighborhood(state: dict[str, Any], service: str, radius: int = 1) -> set[str]:
    if not service:
        return set()
    neighborhood = {service}
    frontier = {service}
    edges = _graph_edges(state)
    for _ in range(max(0, radius)):
        nxt = set()
        for src, dst in edges:
            if src in frontier:
                nxt.add(dst)
            if dst in frontier:
                nxt.add(src)
        nxt -= neighborhood
        neighborhood |= nxt
        frontier = nxt
        if not frontier:
            break
    return neighborhood


def symptom_signature(state: dict[str, Any]) -> dict[str, Any]:
    degraded = []
    for svc, info in (state.get("system", {}) or {}).items():
        health = info.get("health", info) if isinstance(info, dict) else {}
        if (
            health.get("infra_issue_flag")
            or health.get("pods_unready", 0) > 0
            or health.get("crashloop_count", 0) > 0
            or health.get("oomkilled_count", 0) > 0
            or health.get("restart_count", 0) > 0
        ):
            degraded.append(svc)

    service_health = state.get("service_health", {}) or {}
    for svc, h in service_health.items():
        if isinstance(h, dict) and str(h.get("status", "healthy")).lower() not in {"healthy", "unknown", ""}:
            degraded.append(svc)

    failed_edges = []
    trace_sources = []
    trace_targets = []
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        error_ratio = _safe_float(feats.get("error_ratio"))
        if error_ratio > 0.2 or feats.get("is_suspicious"):
            failed_edges.append(edge)
            src = feats.get("source")
            dst = feats.get("target")
            if (not src or not dst) and "->" in str(edge):
                src, dst = str(edge).split("->", 1)
            if src:
                trace_sources.append(str(src))
            if dst:
                trace_targets.append(str(dst))

    error_services = []
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service"):
            error_services.append(str(item["service"]))
    for svc, item in (state.get("logs", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        sig = item.get("signal", item) if isinstance(item.get("signal", item), dict) else {}
        if _safe_float(sig.get("error_count")) > 0 or _safe_float(sig.get("log_anomaly_score")) > 0.3:
            error_services.append(str(svc))

    metric_services = []
    for svc, item in (state.get("metrics", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        flat = item.get("flat_summary", item) if isinstance(item.get("flat_summary", item), dict) else {}
        if _safe_float(flat.get("latency_ms")) > 500:
            metric_services.append(str(svc))
        if _safe_float(flat.get("cpu_usage_delta")) > 0 and _safe_float(flat.get("memory_working_set_last")) > 0:
            # Weak signal only; it says the service is active under load, not necessarily faulty.
            pass

    affected = set(degraded) | set(error_services) | set(trace_sources) | set(trace_targets) | set(metric_services)
    return {
        "degraded_services": sorted(set(degraded)),
        "failed_edges": sorted(set(failed_edges)),
        "top_error_services": sorted(set(error_services)),
        "trace_sources": sorted(set(trace_sources)),
        "trace_targets": sorted(set(trace_targets)),
        "metric_anomaly_services": sorted(set(metric_services)),
        "affected_services": sorted(affected),
    }


def _service_log_text(state: dict[str, Any], service: str) -> str:
    logs = (state.get("logs", {}) or {}).get(service, {}) or {}
    chunks = []
    sig = logs.get("signal", logs) if isinstance(logs, dict) else {}
    for key in ("dominant_error_type", "error_families", "dependency_error_counts", "evidence_lines_top", "error_templates_top"):
        chunks.append(str(logs.get(key, "")))
        if isinstance(sig, dict):
            chunks.append(str(sig.get(key, "")))
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service") == service:
            chunks.append(str(item))
    return " ".join(chunks).lower()


def _service_trace_text(state: dict[str, Any], service: str) -> str:
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    chunks = []
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        src = feats.get("source")
        dst = feats.get("target")
        if (not src or not dst) and "->" in str(edge):
            src, dst = str(edge).split("->", 1)
        if service in {src, dst} or str(edge).startswith(service + "->") or str(edge).endswith("->" + service):
            chunks.append(str(feats))
    return " ".join(chunks).lower()


def _fault_type_compatibility(state: dict[str, Any], service: str, fault_type: str, sig: dict[str, Any]) -> float:
    ft = _norm_fault_type(fault_type)
    system = (state.get("system", {}) or {}).get(service, {}) or {}
    health = system.get("health", system) if isinstance(system, dict) else {}
    log_text = _service_log_text(state, service)
    trace_text = _service_trace_text(state, service)

    if ft == "infra_failure":
        if service in sig["degraded_services"] or health.get("infra_issue_flag") or _safe_float(health.get("pods_unready")) > 0:
            return 1.0
        if any(x in log_text for x in ["crash", "oom", "pod", "container", "endpoint"]):
            return 0.7
        return 0.25

    if ft == "auth_failure":
        if any(x in log_text for x in ["auth", "unauthorized", "forbidden", "permission", "credential"]):
            return 1.0
        if service in sig["top_error_services"]:
            return 0.45
        return 0.1

    if ft == "dependency_failure":
        if any(x in log_text for x in ["mongo", "mongodb", "database", "dependency", "connection", "refused", "unavailable"]):
            return 1.0
        if service in sig["trace_targets"] or service in sig["top_error_services"]:
            return 0.55
        return 0.15

    if ft == "latency_degradation":
        if any(x in trace_text for x in ["latency", "timeout", "slow", "delay"]):
            return 1.0
        if service in sig["trace_sources"] or service in sig["trace_targets"] or service in sig["metric_anomaly_services"]:
            return 0.55
        return 0.15

    if ft == "network_failure":
        if any(x in trace_text for x in ["network", "loss", "unreachable", "reset", "refused"]):
            return 1.0
        if service in sig["trace_sources"] or service in sig["trace_targets"]:
            return 0.55
        return 0.15

    if ft == "config_error":
        if any(x in log_text for x in ["config", "misconfig", "target port", "port", "wrong", "binary", "bin"]):
            return 1.0
        if service in sig["top_error_services"] or service in sig["degraded_services"]:
            return 0.5
        return 0.15

    if ft == "resource_exhaustion":
        if any(x in log_text for x in ["oom", "memory", "cpu", "resource", "throttle"]):
            return 1.0
        if service in sig["metric_anomaly_services"] or _safe_float(health.get("oomkilled_count")) > 0:
            return 0.75
        return 0.15

    # Unknown predictions can still receive weak credit if the service is visibly affected.
    return 0.35 if service in sig["affected_services"] else 0.05


def score_prediction_reproduction(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Score whether predicted RCA faults explain observable symptoms.

    This is an offline behavioral-twin proxy. It intentionally uses only the
    redacted/compressed observable state plus the predicted RCA labels. It avoids
    starting from the already-faulted full state, because that made symptom overlap
    nearly constant and therefore not useful as a reward component.
    """
    sig = symptom_signature(state)
    observed = set(sig["affected_services"])
    predicted_rows = []
    predicted_support = set()

    if not predicted_faults:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy",
            "reason": "empty_prediction",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    for fault in predicted_faults:
        service = str(fault.get("service") or "")
        fault_type = str(fault.get("fault_type") or fault.get("fault_family") or "unknown")
        if not service:
            continue
        neighborhood = graph_neighborhood(state, service, radius=1)
        direct_hits = {
            "degraded": service in sig["degraded_services"],
            "log": service in sig["top_error_services"],
            "trace_source": service in sig["trace_sources"],
            "trace_target": service in sig["trace_targets"],
            "metric": service in sig["metric_anomaly_services"],
        }
        direct_score = min(1.0, sum([
            0.35 if direct_hits["degraded"] else 0.0,
            0.25 if direct_hits["log"] else 0.0,
            0.20 if direct_hits["trace_source"] else 0.0,
            0.20 if direct_hits["trace_target"] else 0.0,
            0.15 if direct_hits["metric"] else 0.0,
        ]))
        neighborhood_score = _coverage(neighborhood, observed)
        compatibility = _fault_type_compatibility(state, service, fault_type, sig)
        per_fault_score = 0.55 * direct_score + 0.25 * neighborhood_score + 0.20 * compatibility
        predicted_support |= neighborhood
        predicted_rows.append({
            "service": service,
            "fault_type": _norm_fault_type(fault_type),
            "direct_evidence_score": round(direct_score, 4),
            "neighborhood_score": round(neighborhood_score, 4),
            "fault_type_compatibility": round(compatibility, 4),
            "per_fault_score": round(per_fault_score, 4),
            "direct_hits": direct_hits,
            "neighborhood": sorted(neighborhood),
        })

    if not predicted_rows:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy",
            "reason": "no_valid_predicted_services",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    avg_pred = sum(r["per_fault_score"] for r in predicted_rows) / len(predicted_rows)
    coverage = _coverage(predicted_support, observed)
    overprediction_penalty = 0.05 * max(0, len(predicted_rows) - max(1, min(len(observed), 3)))
    score = max(0.0, min(1.0, 0.75 * avg_pred + 0.25 * coverage - overprediction_penalty))

    return {
        "reproduction_score": round(score, 4),
        "mode": "behavioral_offline_proxy",
        "direct_evidence_score": round(sum(r["direct_evidence_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "graph_neighborhood_score": round(sum(r["neighborhood_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "fault_type_compatibility_score": round(sum(r["fault_type_compatibility"] for r in predicted_rows) / len(predicted_rows), 4),
        "symptom_coverage_score": round(coverage, 4),
        "overprediction_penalty": round(overprediction_penalty, 4),
        "evidence_signature": sig,
        "predicted_signature": {
            "services": [r["service"] for r in predicted_rows],
            "fault_types": [r["fault_type"] for r in predicted_rows],
            "neighborhood": sorted(predicted_support),
        },
        "per_fault": predicted_rows,
    }


def compare_symptoms(original_state: dict[str, Any], twin_state: dict[str, Any]) -> dict[str, Any]:
    orig = symptom_signature(original_state)
    twin = symptom_signature(twin_state)
    degraded = _jaccard(orig["degraded_services"], twin["degraded_services"])
    edges = _jaccard(orig["failed_edges"], twin["failed_edges"])
    logs = _jaccard(orig["top_error_services"], twin["top_error_services"])
    score = 0.45 * degraded + 0.35 * edges + 0.20 * logs
    return {
        "reproduction_score": round(score, 4),
        "degraded_service_overlap": round(degraded, 4),
        "trace_edge_overlap": round(edges, 4),
        "log_error_service_overlap": round(logs, 4),
        "original_signature": orig,
        "twin_signature": twin,
    }


def score_resolution(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    before = symptom_signature(before_state)
    after = symptom_signature(after_state)
    before_count = len(before["degraded_services"]) + len(before["failed_edges"]) + len(before["top_error_services"])
    after_count = len(after["degraded_services"]) + len(after["failed_edges"]) + len(after["top_error_services"])
    reduction = 1.0 if before_count == 0 and after_count == 0 else max(
        0.0, min(1.0, (before_count - after_count) / max(before_count, 1))
    )
    return {
        "symptom_reduction": round(reduction, 4),
        "before_signature": before,
        "after_signature": after,
        "resolved": after_count == 0 or reduction >= 0.95,
    }
