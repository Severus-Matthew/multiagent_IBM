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
    ps, os = set(predicted or []), set(observed or [])
    if not ps or not os:
        return 0.0
    return len(ps & os) / len(os)


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
        out.update({base, "mongodb-" + base})
    if low.startswith("mongodb-"):
        base = low.replace("mongodb-", "")
        out.update({base, "hotel-reserv-" + base + "-mongo"})
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


def _fault_tokens(ft: str) -> tuple[str, ...]:
    return {
        "infra_failure": ("pod", "pods_unready", "container", "crash", "oom", "endpoint", "replica", "schedule", "node", "unready", "killed", "pending", "no_ready"),
        "auth_failure": ("auth", "unauthorized", "forbidden", "permission", "credential", "login", "denied", "revoke"),
        "dependency_failure": ("dependency", "connection refused", "connection", "unavailable", "upstream", "downstream", "database", "mongodb", "redis", "mongo"),
        "latency_degradation": ("latency", "timeout", "timed out", "slow", "delay", "p95", "p99"),
        "network_failure": ("network", "packet", "loss", "unreachable", "reset", "dns", "no route", "drop"),
        "config_error": ("config", "misconfig", "target port", "port", "wrong", "binary", "bin", "env", "environment"),
        "resource_exhaustion": ("oom", "memory", "cpu", "resource", "throttle", "quota", "limit"),
        "unknown": ("wrong binary", "wrong-bin", "wrong_bin", "invalid executable", "unknown"),
    }.get(_norm_fault_type(ft), ("unknown",))


def _has(text: str, tokens: tuple[str, ...]) -> bool:
    return any(tok in text for tok in tokens)


def _graph_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    edges = []
    for e in (state.get("graph", {}) or {}).get("edges", []) or []:
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
    aliases = _service_aliases(service)
    if not aliases:
        return set()
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
            bool(health.get("infra_issue_flag"))
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
        if _safe_float(feats.get("error_ratio")) > 0.2 or feats.get("is_suspicious"):
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


def _service_dicts(state: dict[str, Any], service: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for alias in _service_aliases(service):
        for root_key in ("system", "service_health", "metrics", "logs"):
            item = (state.get(root_key, {}) or {}).get(alias)
            if isinstance(item, dict):
                out.append(item)
    return out


def _health_signal(state: dict[str, Any], service: str, fields: tuple[str, ...]) -> bool:
    for item in _service_dicts(state, service):
        h = item.get("health", item) if isinstance(item, dict) else {}
        flat = item.get("flat_summary", item) if isinstance(item, dict) else {}
        for src in (h, flat, item):
            if not isinstance(src, dict):
                continue
            for field in fields:
                value = src.get(field)
                if isinstance(value, bool) and value:
                    return True
                if _safe_float(value, 0.0) > 0:
                    return True
            status = str(src.get("status", "")).lower()
            if any(tok in status for tok in fields):
                return True
    return False


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
    for key in ("high_signal_evidence", "rca_agent_structured_evidence", "llm_view", "clusters", "service_health"):
        if isinstance(state.get(key), (dict, list)):
            roots.append(state[key])
    try:
        from training_pipeline.rca_candidate_generator_v4 import compact_state_for_llm_v4
        compact = compact_state_for_llm_v4(state, char_budget=24000)
        if isinstance(compact, dict) and isinstance(compact.get("high_signal_evidence"), dict):
            roots.append(compact["high_signal_evidence"])
    except Exception:
        pass
    return roots


def _typed_candidate_support(state: dict[str, Any], service: str, fault_type: str) -> float:
    ft = _norm_fault_type(fault_type)
    tokens = _fault_tokens(ft)
    best = 0.0
    for root in _candidate_evidence_roots(state):
        for row in _iter_candidate_rows(root):
            if not isinstance(row, dict):
                continue
            row_service = str(row.get("service") or "")
            row_ft = _norm_fault_type(str(row.get("fault_type") or row.get("fault_family") or "unknown"))
            if not _same_service(row_service, service) or row_ft != ft:
                continue
            evidence = " ".join(str(row.get(k, "")) for k in (
                "reason", "reasons", "evidence", "evidence_text", "template", "templates", "source", "feature"
            )).lower()
            # Critical: a candidate row is not mechanism evidence unless the row
            # itself contains mechanism-specific terms. This prevents service-only
            # candidates from validating same-service wrong-fault controls.
            if not _has(evidence, tokens):
                continue
            raw_score = _safe_float(row.get("score"), 0.0)
            support = 0.70 if raw_score <= 0 else min(0.95, 0.55 + raw_score / 30.0)
            if _has(evidence, ("local", "explicit", "typed", "direct", "v4_local")):
                support = max(support, 0.85)
            best = max(best, support)
    return best


def _fault_type_compatibility(state: dict[str, Any], service: str, fault_type: str, sig: dict[str, Any]) -> float:
    ft = _norm_fault_type(fault_type)
    log_text = _service_log_text(state, service)
    trace_text = _service_trace_text(state, service)
    local_text = " ".join([log_text, trace_text])
    typed_support = _typed_candidate_support(state, service, ft)

    if ft == "infra_failure":
        structured = _health_signal(state, service, ("infra_issue_flag", "pods_unready", "crashloop_count", "oomkilled_count", "restart_count", "pending", "no_ready", "unready", "crash"))
        if structured:
            return max(0.95, typed_support)
        return max(typed_support, 0.04)
    if ft == "auth_failure":
        return max(typed_support, 1.0 if _has(log_text, _fault_tokens(ft)) else 0.03)
    if ft == "dependency_failure":
        if _has(log_text, _fault_tokens(ft)):
            return max(0.90, typed_support)
        return max(typed_support, 0.08 if _service_in(sig["trace_targets"], service) else 0.03)
    if ft == "latency_degradation":
        if _has(local_text, _fault_tokens(ft)):
            return max(0.95, typed_support)
        return max(typed_support, 0.10 if _service_in(sig["metric_anomaly_services"], service) else 0.03)
    if ft == "network_failure":
        if _has(local_text, _fault_tokens(ft)):
            return max(0.95, typed_support)
        return max(typed_support, 0.08 if _service_in(sig["trace_sources"], service) or _service_in(sig["trace_targets"], service) else 0.03)
    if ft == "config_error":
        return max(typed_support, 1.0 if _has(log_text, _fault_tokens(ft)) else 0.03)
    if ft == "resource_exhaustion":
        structured = _health_signal(state, service, ("oomkilled_count", "memory", "cpu", "throttle", "quota", "limit"))
        return max(typed_support, 0.90 if structured or _has(log_text, _fault_tokens(ft)) else 0.03)
    # unknown is accepted only for explicit wrong-binary/unknown mechanism evidence.
    return max(typed_support, 0.75 if _has(log_text, _fault_tokens(ft)) else 0.02)


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
    """Verifier-side behavioral twin score for RCA predictions.

    v6 requires mechanism-specific evidence. A noisy service is insufficient:
    the predicted mechanism must be supported by local logs/traces/structured
    health or by a typed evidence row whose text itself contains mechanism terms.
    """
    sig = symptom_signature(state)
    observed = set(sig["affected_services"])
    predicted_rows = []
    predicted_support: set[str] = set()
    if not predicted_faults:
        return {
            "reproduction_score": 0.0,
            "mode": "behavioral_offline_proxy_v6_mechanism_evidence_gate",
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
        radius = 0 if fault_type in {"config_error", "auth_failure", "resource_exhaustion", "unknown", "infra_failure"} else 1
        neighborhood = graph_neighborhood(state, service, radius=radius) or {service}
        neighborhood_score = _coverage(neighborhood, observed)
        local_observed = 1.0 if _service_in(sig["affected_services"], service) else 0.0
        mechanism_gate = compatibility ** 1.35
        raw = (0.12 * direct_score + 0.82 * compatibility + 0.06 * neighborhood_score) * mechanism_gate
        per_fault_score = raw * (0.15 + 0.85 * local_observed)
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
            "mode": "behavioral_offline_proxy_v6_mechanism_evidence_gate",
            "reason": "no_valid_predicted_services",
            "evidence_signature": sig,
            "predicted_signature": {"services": [], "neighborhood": []},
        }

    avg_pred = sum(r["per_fault_score"] for r in predicted_rows) / len(predicted_rows)
    coverage = _coverage(predicted_support, observed)
    overprediction_penalty = 0.08 * max(0, len(predicted_rows) - 2)
    weak_type_penalty = 0.28 * sum(1 for r in predicted_rows if r["fault_type_compatibility"] < 0.25)
    score = max(0.0, min(1.0, 0.96 * avg_pred + 0.04 * coverage - overprediction_penalty - weak_type_penalty))
    return {
        "reproduction_score": round(score, 4),
        "mode": "behavioral_offline_proxy_v6_mechanism_evidence_gate",
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
