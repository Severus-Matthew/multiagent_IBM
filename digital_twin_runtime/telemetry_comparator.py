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


def _service_aliases(service: str | None) -> set[str]:
    s = _norm_service(service)
    if not s:
        return set()
    low = s.lower()
    out = {s, low}
    if low.startswith("hotel-reserv-"):
        out.add(low[len("hotel-reserv-"):])
    if low.endswith("-mongo"):
        base = low[:-len("-mongo")].split("-")[-1]
        out.add("mongodb-" + base)
        out.add(base)
    if low.startswith("mongodb-"):
        base = low.replace("mongodb-", "")
        out.add(base)
        out.add("hotel-reserv-" + base + "-mongo")
    return {x for x in out if x}


def _same_service(a: str | None, b: str | None) -> bool:
    return bool(_service_aliases(a) & _service_aliases(b))


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
    aliases = _service_aliases(service)
    neighborhood = set(aliases)
    frontier = set(aliases)
    edges = _graph_edges(state)
    for _ in range(max(0, radius)):
        nxt = set()
        for src, dst in edges:
            if src in frontier or bool(_service_aliases(src) & frontier):
                nxt.add(dst)
            if dst in frontier or bool(_service_aliases(dst) & frontier):
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


def _service_in(values: list[str], service: str) -> bool:
    return any(_same_service(service, v) for v in values or [])


def _service_log_text(state: dict[str, Any], service: str) -> str:
    chunks = []
    for alias in _service_aliases(service):
        logs = (state.get("logs", {}) or {}).get(alias, {}) or {}
        sig = logs.get("signal", logs) if isinstance(logs, dict) else {}
        for key in ("dominant_error_type", "error_families", "dependency_error_counts", "evidence_lines_top", "error_templates_top", "messages", "sample"):
            if isinstance(logs, dict):
                chunks.append(str(logs.get(key, "")))
            if isinstance(sig, dict):
                chunks.append(str(sig.get(key, "")))
        for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
            if isinstance(item, dict) and _same_service(item.get("service"), alias):
                chunks.append(str(item))
    return " ".join(chunks).lower()


def _service_health_text(state: dict[str, Any], service: str) -> str:
    chunks = []
    for alias in _service_aliases(service):
        system = (state.get("system", {}) or {}).get(alias, {}) or {}
        health = system.get("health", system) if isinstance(system, dict) else {}
        svc_health = (state.get("service_health", {}) or {}).get(alias, {}) or {}
        chunks.extend([str(system), str(health), str(svc_health)])
    return " ".join(chunks).lower()


def _service_trace_text(state: dict[str, Any], service: str) -> str:
    aliases = _service_aliases(service)
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
        if src in aliases or dst in aliases or any(edge_s.startswith(a + "->") or edge_s.endswith("->" + a) for a in aliases):
            chunks.append(str(feats))
    return " ".join(chunks).lower()


def _has(text: str, tokens: tuple[str, ...]) -> bool:
    return any(tok in text for tok in tokens)


def _iter_candidate_rows(obj: Any):
    if isinstance(obj, dict):
        if "service" in obj and ("fault_type" in obj or "fault_family" in obj):
            yield obj
        for value in obj.values():
            yield from _iter_candidate_rows(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_candidate_rows(item)


def _candidate_evidence_roots(state: dict[str, Any]) -> list[Any]:
    roots: list[Any] = []
    # Only inspect verifier-side telemetry evidence summaries; never inspect
    # ground_truth/fault_context fields even if they accidentally exist.
    for key in ("high_signal_evidence", "rca_agent_structured_evidence", "llm_view", "clusters", "service_health"):
        if isinstance(state.get(key), (dict, list)):
            roots.append(state[key])
    try:
        from training_pipeline.rca_candidate_generator_v4 import compact_state_for_llm_v4
        compact = compact_state_for_llm_v4(state, char_budget=24000)
        if isinstance(compact, dict):
            ev = compact.get("high_signal_evidence")
            if isinstance(ev, dict):
                roots.append(ev)
    except Exception:
        pass
    return roots


def _typed_candidate_support(state: dict[str, Any], service: str, fault_type: str) -> float:
    ft = _norm_fault_type(fault_type)
    best = 0.0
    for root in _candidate_evidence_roots(state):
        for row in _iter_candidate_rows(root):
            if not isinstance(row, dict):
                continue
            row_service = str(row.get("service") or "")
            row_ft = _norm_fault_type(str(row.get("fault_type") or row.get("fault_family") or "unknown"))
            if not _same_service(row_service, service) or row_ft != ft:
                continue
            raw_score = _safe_float(row.get("score"), 0.0)
            support = min(1.0, raw_score / 14.0) if raw_score > 0 else 0.25
            reasons = " ".join(str(x) for x in (row.get("reasons", []) or [])).lower()
            if any(tok in reasons for tok in ("local", "explicit", "typed", "direct", "v4_local")):
                support = max(support, 0.85)
            elif any(tok in reasons for tok in ("backstop", "paired", "family", "v4_")):
                support = max(support, 0.55)
            elif any(tok in reasons for tok in ("generic", "symptom_signature", "cluster")):
                support = min(support, 0.18)
            best = max(best, support)
    return best


def _fault_type_compatibility(state: dict[str, Any], service: str, fault_type: str, sig: dict[str, Any]) -> float:
    ft = _norm_fault_type(fault_type)
    service = _norm_service(service)
    log_text = _service_log_text(state, service)
    trace_text = _service_trace_text(state, service)
    health_text = _service_health_text(state, service)
    local = " ".join([log_text, trace_text, health_text])
    typed_support = _typed_candidate_support(state, service, ft)

    if ft == "infra_failure":
        if _service_in(sig["degraded_services"], service) and _has(local, ("pod", "container", "crash", "oom", "endpoint", "replica", "schedule", "node", "unready", "killed")):
            return max(1.0, typed_support)
        return max(typed_support, 0.10 if _service_in(sig["degraded_services"], service) else 0.03)
    if ft == "auth_failure":
        if _has(local, ("auth", "unauthorized", "forbidden", "permission", "credential", "login", "denied")):
            return max(1.0, typed_support)
        if _service_in(sig["top_error_services"], service) and _has(local, ("mongo", "mongodb", "database")):
            return max(typed_support, 0.18)
        return max(typed_support, 0.03)
    if ft == "dependency_failure":
        if _has(local, ("dependency", "connection refused", "connection", "unavailable", "upstream", "downstream", "database", "mongodb", "redis")):
            return max(0.9, typed_support)
        if _service_in(sig["trace_targets"], service) or _service_in(sig["top_error_services"], service):
            return max(typed_support, 0.12)
        return max(typed_support, 0.03)
    if ft == "latency_degradation":
        if _has(local, ("latency", "timeout", "timed out", "slow", "delay", "p95", "p99")):
            return max(1.0, typed_support)
        if _service_in(sig["trace_sources"], service) or _service_in(sig["trace_targets"], service) or _service_in(sig["metric_anomaly_services"], service):
            return max(typed_support, 0.14)
        return max(typed_support, 0.03)
    if ft == "network_failure":
        if _has(local, ("network", "packet", "loss", "unreachable", "reset", "dns", "no route", "drop")):
            return max(1.0, typed_support)
        if _service_in(sig["trace_sources"], service) or _service_in(sig["trace_targets"], service):
            return max(typed_support, 0.12)
        return max(typed_support, 0.03)
    if ft == "config_error":
        if _has(local, ("config", "misconfig", "target port", "port", "wrong", "binary", "bin", "env", "environment")):
            return max(1.0, typed_support)
        if _service_in(sig["degraded_services"], service) and _service_in(sig["top_error_services"], service):
            return max(typed_support, 0.12)
        return max(typed_support, 0.03)
    if ft == "resource_exhaustion":
        if _has(local, ("oom", "memory", "cpu", "resource", "throttle", "quota", "limit")):
            return max(1.0, typed_support)
        if _service_in(sig["metric_anomaly_services"], service):
            return max(typed_support, 0.24)
        return max(typed_support, 0.03)
    if _has(local, ("wrong binary", "wrong-bin", "wrong_bin", "unknown", "invalid executable")):
        return max(0.75, typed_support)
    return max(typed_support, 0.04 if _service_in(sig["affected_services"], service) else 0.0)


def _service_direct_score(service: str, sig: dict[str, Any]) -> float:
    score = 0.0
    if _service_in(sig["degraded_services"], service):
        score += 0.35
    if _service_in(sig["top_error_services"], service):
        score += 0.25
    if _service_in(sig["trace_sources"], service):
        score += 0.18
    if _service_in(sig["trace_targets"], service):
        score += 0.18
    if _service_in(sig["metric_anomaly_services"], service):
        score += 0.12
    return min(1.0, score)


def score_prediction_reproduction(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Counterfactual behavioral-twin score for RCA predictions.

    v4 is verifier-side only: it may use typed evidence mined from the redacted
    state abstraction, but that evidence is not shown to the RCA agent. A label
    must have both service-local evidence and fault-type-specific support; noisy
    downstream services and wrong fault types should not pass merely by overlap.
    """
    sig = symptom_signature(state)
    observed = set(sig["affected_services"])
    predicted_rows = []
    predicted_support: set[str] = set()

    if not predicted_faults:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy_v4_typed_evidence_counterfactual",
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
        local_observed = 1.0 if _service_in(sig["affected_services"], service) else 0.0
        type_gate = 0.05 + 0.95 * compatibility
        raw = (0.18 * direct_score + 0.74 * compatibility + 0.08 * neighborhood_score) * type_gate
        per_fault_score = raw * (0.20 + 0.80 * local_observed)
        predicted_support |= neighborhood
        predicted_rows.append({
            "service": service,
            "fault_type": fault_type,
            "direct_evidence_score": round(direct_score, 4),
            "neighborhood_score": round(neighborhood_score, 4),
            "fault_type_compatibility": round(compatibility, 4),
            "typed_candidate_support": round(_typed_candidate_support(state, service, fault_type), 4),
            "local_observed": bool(local_observed),
            "per_fault_score": round(max(0.0, min(1.0, per_fault_score)), 4),
            "neighborhood": sorted(neighborhood),
        })

    if not predicted_rows:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy_v4_typed_evidence_counterfactual",
            "reason": "no_valid_predicted_services",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    avg_pred = sum(r["per_fault_score"] for r in predicted_rows) / len(predicted_rows)
    coverage = _coverage(predicted_support, observed)
    overprediction_penalty = 0.08 * max(0, len(predicted_rows) - 2)
    weak_type_penalty = 0.22 * sum(1 for r in predicted_rows if r["fault_type_compatibility"] < 0.25)
    score = max(0.0, min(1.0, 0.94 * avg_pred + 0.06 * coverage - overprediction_penalty - weak_type_penalty))

    return {
        "reproduction_score": round(score, 4),
        "mode": "behavioral_offline_proxy_v4_typed_evidence_counterfactual",
        "direct_evidence_score": round(sum(r["direct_evidence_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "graph_neighborhood_score": round(sum(r["neighborhood_score"] for r in predicted_rows) / len(predicted_rows), 4),
        "fault_type_compatibility_score": round(sum(r["fault_type_compatibility"] for r in predicted_rows) / len(predicted_rows), 4),
        "typed_candidate_support_score": round(sum(r["typed_candidate_support"] for r in predicted_rows) / len(predicted_rows), 4),
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
