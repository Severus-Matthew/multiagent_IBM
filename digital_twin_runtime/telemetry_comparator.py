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
    if not observed_set or not predicted_set:
        return 0.0
    return len(predicted_set & observed_set) / len(observed_set)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_service(service: str | None) -> str:
    return str(service or "").strip()


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
        "wrong_bin": "unknown",
        "wrong-binary": "unknown",
        "wrong binary": "unknown",
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
    service = _norm_service(service)
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
    degraded: list[str] = []
    for svc, info in (state.get("system", {}) or {}).items():
        health = info.get("health", info) if isinstance(info, dict) else {}
        if (
            health.get("infra_issue_flag")
            or _safe_float(health.get("pods_unready")) > 0
            or _safe_float(health.get("crashloop_count")) > 0
            or _safe_float(health.get("oomkilled_count")) > 0
            or _safe_float(health.get("restart_count")) > 0
        ):
            degraded.append(str(svc))

    for svc, h in (state.get("service_health", {}) or {}).items():
        if isinstance(h, dict) and str(h.get("status", "healthy")).lower() not in {"healthy", "unknown", ""}:
            degraded.append(str(svc))

    failed_edges: list[str] = []
    trace_sources: list[str] = []
    trace_targets: list[str] = []
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        error_ratio = _safe_float(feats.get("error_ratio"))
        if error_ratio > 0.2 or feats.get("is_suspicious"):
            edge_s = str(edge)
            failed_edges.append(edge_s)
            src = feats.get("source")
            dst = feats.get("target")
            if (not src or not dst) and "->" in edge_s:
                src, dst = edge_s.split("->", 1)
            if src:
                trace_sources.append(str(src))
            if dst:
                trace_targets.append(str(dst))

    error_services: list[str] = []
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service"):
            error_services.append(str(item["service"]))
    for svc, item in (state.get("logs", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        sig = item.get("signal", item) if isinstance(item.get("signal", item), dict) else {}
        if _safe_float(sig.get("error_count")) > 0 or _safe_float(sig.get("log_anomaly_score")) > 0.3:
            error_services.append(str(svc))

    metric_services: list[str] = []
    for svc, item in (state.get("metrics", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        flat = item.get("flat_summary", item) if isinstance(item.get("flat_summary", item), dict) else {}
        if _safe_float(flat.get("latency_ms")) > 500:
            metric_services.append(str(svc))

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
    service = _norm_service(service)
    logs = (state.get("logs", {}) or {}).get(service, {}) or {}
    chunks = []
    sig = logs.get("signal", logs) if isinstance(logs, dict) else {}
    for key in ("dominant_error_type", "error_families", "dependency_error_counts", "evidence_lines_top", "error_templates_top", "messages", "sample"):
        if isinstance(logs, dict):
            chunks.append(str(logs.get(key, "")))
        if isinstance(sig, dict):
            chunks.append(str(sig.get(key, "")))
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service") == service:
            chunks.append(str(item))
    return " ".join(chunks).lower()


def _service_health_text(state: dict[str, Any], service: str) -> str:
    service = _norm_service(service)
    chunks = []
    system = (state.get("system", {}) or {}).get(service, {}) or {}
    health = system.get("health", system) if isinstance(system, dict) else {}
    svc_health = (state.get("service_health", {}) or {}).get(service, {}) or {}
    chunks.extend([str(system), str(health), str(svc_health)])
    return " ".join(chunks).lower()


def _service_trace_text(state: dict[str, Any], service: str) -> str:
    service = _norm_service(service)
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    chunks = []
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        src = feats.get("source")
        dst = feats.get("target")
        edge_s = str(edge)
        if (not src or not dst) and "->" in edge_s:
            src, dst = edge_s.split("->", 1)
        if service in {src, dst} or edge_s.startswith(service + "->") or edge_s.endswith("->" + service):
            chunks.append(str(feats))
    return " ".join(chunks).lower()


def _has(text: str, tokens: tuple[str, ...]) -> bool:
    return any(tok in text for tok in tokens)


def _fault_type_compatibility(state: dict[str, Any], service: str, fault_type: str, sig: dict[str, Any]) -> float:
    ft = _norm_fault_type(fault_type)
    service = _norm_service(service)
    log_text = _service_log_text(state, service)
    trace_text = _service_trace_text(state, service)
    health_text = _service_health_text(state, service)
    local = " ".join([log_text, trace_text, health_text])

    if ft == "infra_failure":
        if service in sig["degraded_services"] and _has(local, ("pod", "container", "crash", "oom", "endpoint", "replica", "schedule", "node", "unready", "killed")):
            return 1.0
        if service in sig["degraded_services"]:
            return 0.75
        return 0.05
    if ft == "auth_failure":
        if _has(local, ("auth", "unauthorized", "forbidden", "permission", "credential", "login", "denied")):
            return 1.0
        if service in sig["top_error_services"] and _has(local, ("mongo", "mongodb", "database")):
            return 0.45
        return 0.03
    if ft == "dependency_failure":
        if _has(local, ("dependency", "connection refused", "connection", "unavailable", "upstream", "downstream", "database", "mongodb", "redis")):
            return 0.9
        if service in sig["trace_targets"] or service in sig["top_error_services"]:
            return 0.25
        return 0.03
    if ft == "latency_degradation":
        if _has(local, ("latency", "timeout", "timed out", "slow", "delay", "p95", "p99")):
            return 1.0
        if service in sig["trace_sources"] or service in sig["trace_targets"] or service in sig["metric_anomaly_services"]:
            return 0.35
        return 0.03
    if ft == "network_failure":
        if _has(local, ("network", "packet", "loss", "unreachable", "reset", "dns", "no route", "drop")):
            return 1.0
        if service in sig["trace_sources"] or service in sig["trace_targets"]:
            return 0.25
        return 0.03
    if ft == "config_error":
        if _has(local, ("config", "misconfig", "target port", "port", "wrong", "binary", "bin", "env", "environment")):
            return 1.0
        if service in sig["degraded_services"] and service in sig["top_error_services"]:
            return 0.35
        return 0.03
    if ft == "resource_exhaustion":
        if _has(local, ("oom", "memory", "cpu", "resource", "throttle", "quota", "limit")):
            return 1.0
        if service in sig["metric_anomaly_services"]:
            return 0.55
        return 0.03
    if _has(local, ("wrong binary", "wrong-bin", "wrong_bin", "unknown", "invalid executable")):
        return 0.75
    return 0.08 if service in sig["affected_services"] else 0.0


def _service_direct_score(service: str, sig: dict[str, Any]) -> float:
    service = _norm_service(service)
    score = 0.0
    if service in sig["degraded_services"]:
        score += 0.35
    if service in sig["top_error_services"]:
        score += 0.25
    if service in sig["trace_sources"]:
        score += 0.18
    if service in sig["trace_targets"]:
        score += 0.18
    if service in sig["metric_anomaly_services"]:
        score += 0.12
    return min(1.0, score)


def score_prediction_reproduction(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Counterfactual behavioral-twin score for RCA predictions."""
    sig = symptom_signature(state)
    observed = set(sig["affected_services"])
    predicted_rows = []
    predicted_support: set[str] = set()

    if not predicted_faults:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy_v2_strict_local_counterfactual",
            "reason": "empty_prediction",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    for fault in predicted_faults:
        service = _norm_service(str(fault.get("service") or ""))
        fault_type = _norm_fault_type(str(fault.get("fault_type") or fault.get("fault_family") or "unknown"))
        if not service:
            continue
        direct_score = _service_direct_score(service, sig)
        compatibility = _fault_type_compatibility(state, service, fault_type, sig)
        radius = 0 if fault_type in {"config_error", "auth_failure", "resource_exhaustion", "unknown"} else 1
        neighborhood = graph_neighborhood(state, service, radius=radius) or {service}
        neighborhood_score = _coverage(neighborhood, observed)
        local_observed = 1.0 if service in observed else 0.0
        per_fault_score = (0.45 * direct_score + 0.45 * compatibility + 0.10 * neighborhood_score) * (0.35 + 0.65 * local_observed)
        predicted_support |= neighborhood
        predicted_rows.append({
            "service": service,
            "fault_type": fault_type,
            "direct_evidence_score": round(direct_score, 4),
            "neighborhood_score": round(neighborhood_score, 4),
            "fault_type_compatibility": round(compatibility, 4),
            "local_observed": bool(local_observed),
            "per_fault_score": round(max(0.0, min(1.0, per_fault_score)), 4),
            "neighborhood": sorted(neighborhood),
        })

    if not predicted_rows:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy_v2_strict_local_counterfactual",
            "reason": "no_valid_predicted_services",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    avg_pred = sum(r["per_fault_score"] for r in predicted_rows) / len(predicted_rows)
    coverage = _coverage(predicted_support, observed)
    overprediction_penalty = 0.08 * max(0, len(predicted_rows) - 2)
    weak_type_penalty = 0.12 * sum(1 for r in predicted_rows if r["fault_type_compatibility"] < 0.2)
    score = max(0.0, min(1.0, 0.88 * avg_pred + 0.12 * coverage - overprediction_penalty - weak_type_penalty))

    return {
        "reproduction_score": round(score, 4),
        "mode": "behavioral_offline_proxy_v2_strict_local_counterfactual",
        "direct_evidence_score": round(sum(r["direct_evidence_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "graph_neighborhood_score": round(sum(r["neighborhood_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "fault_type_compatibility_score": round(sum(r["fault_type_compatibility"] for r in predicted_rows) / len(predicted_rows), 4),
        "symptom_coverage_score": round(coverage, 4),
        "overprediction_penalty": round(overprediction_penalty, 4),
        "weak_type_penalty": round(weak_type_penalty, 4),
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
    reduction = 1.0 if before_count == 0 and after_count == 0 else max(0.0, min(1.0, (before_count - after_count) / max(before_count, 1)))
    return {
        "symptom_reduction": round(reduction, 4),
        "before_signature": before,
        "after_signature": after,
        "resolved": after_count == 0 or reduction >= 0.95,
    }
