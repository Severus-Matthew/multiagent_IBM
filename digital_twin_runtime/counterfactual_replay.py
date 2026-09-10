from __future__ import annotations

from typing import Any

from .telemetry_comparator import graph_neighborhood, symptom_signature


CANONICAL_FAULT_TYPES = (
    "infra_failure",
    "auth_failure",
    "dependency_failure",
    "resource_exhaustion",
    "latency_degradation",
    "network_failure",
    "config_error",
    "unknown",
)


def _norm_service(value: Any) -> str:
    return str(value or "").strip()


def _service_aliases(service: Any) -> set[str]:
    s = _norm_service(service)
    if not s:
        return set()
    low = s.lower()
    out = {low}
    if low.startswith("hotel-reserv-"):
        out.add(low[len("hotel-reserv-"):])
    if low.endswith("-mongo"):
        base = low[:-len("-mongo")].split("-")[-1]
        out.update({base, "mongodb-" + base})
    if low.startswith("mongodb-"):
        base = low[len("mongodb-"):]
        out.update({base, "hotel-reserv-" + base + "-mongo"})
    return {x for x in out if x}


def _same_service(a: Any, b: Any) -> bool:
    return bool(_service_aliases(a) & _service_aliases(b))


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
    if text in CANONICAL_FAULT_TYPES:
        return text
    for pattern, mapped in aliases.items():
        if pattern in text:
            return mapped
    return "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
    return [
        f"{src}->{dst}"
        for src, dst in _graph_edges(state)
        if _same_service(src, service) or _same_service(dst, service)
    ]


def _callers(state: dict[str, Any], service: str) -> set[str]:
    return {src for src, dst in _graph_edges(state) if _same_service(dst, service)}


def _dependencies(state: dict[str, Any], service: str) -> set[str]:
    return {dst for src, dst in _graph_edges(state) if _same_service(src, service)}


def _service_rows(root: dict[str, Any], service: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in (root or {}).items():
        if _same_service(key, service) and isinstance(value, dict):
            rows.append(value)
    return rows


def _flatten_value_text(obj: Any, max_chars: int = 12000) -> str:
    """Flatten only observed values, never schema/key names.

    This distinction is critical for mechanism verification. Compressed telemetry
    contains keys such as ``pods_unready``, ``oomkilled_count``, ``cpu_usage_delta``
    and ``memory_usage_last`` for every service, including healthy services. Treating
    key names as evidence makes infra/resource mechanisms appear present even when
    the corresponding numeric values are zero. Only actual non-empty/non-zero values
    are textual evidence here; typed numeric fields are handled structurally below.
    """
    chunks: list[str] = []
    total = 0

    def walk(value: Any) -> None:
        nonlocal total
        if total >= max_chars:
            return
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
                if total >= max_chars:
                    break
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
                if total >= max_chars:
                    break
        elif isinstance(value, bool):
            if value:
                chunks.append("true")
                total += 5
        elif isinstance(value, (int, float)):
            # Numeric mechanism evidence is interpreted structurally in the
            # mechanism-specific blocks; raw numbers have no semantic token value.
            return
        elif value is not None:
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "false", "0", "0.0"}:
                return
            chunks.append(text)
            total += len(text) + 1

    walk(obj)
    return " ".join(chunks).lower()[:max_chars]


def _contains(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _family_count(log_row: dict[str, Any], *families: str) -> float:
    total = 0.0
    for fam, count in (log_row.get("error_families", {}) or {}).items():
        if str(fam).lower() in families:
            total += _safe_float(count)
    return total


def _trace_rows_for_service(state: dict[str, Any], service: str) -> list[dict[str, Any]]:
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    rows: list[dict[str, Any]] = []
    for edge, feats in (per_edge or {}).items():
        if not isinstance(feats, dict):
            continue
        src = feats.get("source")
        dst = feats.get("target")
        if (not src or not dst) and "->" in str(edge):
            src, dst = str(edge).split("->", 1)
        if _same_service(src, service) or _same_service(dst, service):
            row = dict(feats)
            row["_edge"] = str(edge)
            rows.append(row)
    return rows


def _observed_mechanism_profile(state: dict[str, Any], service: str) -> dict[str, float]:
    """Extract mechanism-specific support from redacted telemetry only."""
    supports = {ft: 0.0 for ft in CANONICAL_FAULT_TYPES}

    system_rows = _service_rows(state.get("system", {}) or {}, service)
    metric_rows = _service_rows(state.get("metrics", {}) or {}, service)
    log_rows = _service_rows(state.get("logs", {}) or {}, service)
    trace_rows = _trace_rows_for_service(state, service)

    system_text = _flatten_value_text(system_rows)
    log_text = _flatten_value_text(log_rows)
    trace_text = _flatten_value_text(trace_rows)

    infra = 0.0
    for row in system_rows:
        health = row.get("health", row) if isinstance(row, dict) else {}
        pods_total = max(1.0, _safe_float(health.get("pods_total"), 1.0))
        pods_unready = _safe_float(health.get("pods_unready"))
        restart_count = _safe_float(health.get("restart_count"))
        crashloop_count = _safe_float(health.get("crashloop_count"))
        if bool(health.get("infra_issue_flag")):
            infra = max(infra, 0.95)
        if pods_unready > 0:
            infra = max(infra, min(1.0, 0.65 + 0.35 * pods_unready / pods_total))
        if crashloop_count > 0:
            infra = max(infra, min(1.0, 0.75 + 0.08 * crashloop_count))
        if restart_count > 0:
            infra = max(infra, min(0.85, 0.45 + 0.08 * restart_count))
    if _contains(system_text, (
        "unschedulable", "unscheduled", "pending", "no ready replicas",
        "crashloopbackoff", "container killed", "pod killed",
    )):
        infra = max(infra, 0.90)
    supports["infra_failure"] = infra

    auth = 0.0
    if any(_family_count(row, "auth") > 0 for row in log_rows):
        auth = 1.0
    if _contains(log_text, (
        "unauthorized", "forbidden", "authentication", "authorization",
        "auth failed", "permission denied", "credential", "login failed",
        "not authorized", "revoke",
    )):
        auth = max(auth, 0.98)
    supports["auth_failure"] = auth

    dependency = 0.0
    for row in log_rows:
        dep_counts = row.get("dependency_error_counts", {}) or {}
        if sum(_safe_float(v) for v in dep_counts.values()) > 0:
            dependency = max(dependency, 0.95)
        if _family_count(row, "dependency", "database") > 0:
            dependency = max(dependency, 0.90)
    if _contains(log_text, (
        "connection refused", "failed to connect", "connection reset",
        "server selection", "database unavailable", "mongodb unavailable",
        "upstream unavailable", "dependency unavailable", "no reachable server",
    )):
        dependency = max(dependency, 0.98)
    if _contains(trace_text, ("dependency", "connection refused", "unavailable", "upstream failure")):
        dependency = max(dependency, 0.85)
    supports["dependency_failure"] = dependency

    latency = 0.0
    for row in metric_rows:
        flat = row.get("flat_summary", row) if isinstance(row, dict) else {}
        latency_ms = _safe_float(flat.get("latency_ms"))
        if latency_ms > 0:
            latency = max(latency, min(1.0, latency_ms / 1000.0))
    for row in trace_rows:
        p95 = _safe_float(row.get("latency_p95_ms"))
        if p95 >= 100:
            latency = max(latency, min(1.0, 0.45 + p95 / 1800.0))
        failure_type = str(row.get("failure_type") or "").lower()
        if _contains(failure_type, ("timeout", "latency", "slow", "delay")):
            latency = max(latency, 0.95)
    if _contains(log_text, ("timed out", "timeout", "latency", "slow request", "deadline exceeded")):
        latency = max(latency, 0.90)
    supports["latency_degradation"] = latency

    network = 0.0
    for row in metric_rows:
        flat = row.get("flat_summary", row) if isinstance(row, dict) else {}
        tx_errors = _safe_float(flat.get("network_tx_errors_delta"))
        if tx_errors > 0:
            network = max(network, min(1.0, 0.70 + 0.05 * tx_errors))
    for row in trace_rows:
        failure_type = str(row.get("failure_type") or "").lower()
        if _contains(failure_type, ("network", "packet", "loss", "unreachable", "reset", "dns", "no route", "drop")):
            network = max(network, 0.98)
    if _contains(log_text + " " + trace_text, (
        "packet loss", "network unreachable", "connection reset by peer",
        "no route to host", "dns failure", "name resolution", "network drop",
    )):
        network = max(network, 0.95)
    supports["network_failure"] = network

    config = 0.0
    if _contains(log_text + " " + system_text, (
        "misconfigured", "misconfiguration", "configuration error",
        "invalid configuration", "invalid config", "wrong port", "port mismatch",
        "invalid environment", "missing environment", "failed to load config",
    )):
        config = max(config, 0.98)
    if _contains(log_text, ("invalid option", "unknown option", "bad configuration")):
        config = max(config, 0.90)
    supports["config_error"] = config

    resource = 0.0
    for row in system_rows:
        health = row.get("health", row) if isinstance(row, dict) else {}
        oom = _safe_float(health.get("oomkilled_count"))
        if oom > 0:
            resource = max(resource, 1.0)
    for row in log_rows:
        if _family_count(row, "resource") > 0:
            resource = max(resource, 0.90)
    # Do not inspect metric/system schema keys such as cpu_usage_delta,
    # memory_usage_last, or oomkilled_count as text. They exist for all services.
    if _contains(log_text + " " + system_text, (
        "out of memory", "oom killed", "cpu throttled", "cpu throttling",
        "memory limit exceeded", "resource exhausted", "quota exceeded",
        "cannot allocate memory",
    )):
        resource = max(resource, 0.98)
    supports["resource_exhaustion"] = resource

    unknown = 0.0
    if _contains(log_text + " " + system_text, (
        "wrong binary", "wrong-bin", "wrong_bin", "exec format error",
        "invalid executable", "command not found", "cannot execute binary",
    )):
        unknown = 0.98
    supports["unknown"] = unknown

    return {k: round(max(0.0, min(1.0, v)), 4) for k, v in supports.items()}


def predict_fault_signature(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a topology-aware expected symptom footprint from the RCA prediction."""
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
        elif ft in {"auth_failure", "dependency_failure"}:
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


def _set_f1(expected: list[str], observed: list[str]) -> float:
    e = set(expected or [])
    o = set(observed or [])
    if not e or not o:
        return 0.0
    precision = len(e & o) / len(e)
    recall = len(e & o) / len(o)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _service_supported(service: str, observed_services: list[str]) -> bool:
    return any(_same_service(service, observed) for observed in observed_services or [])


def score_counterfactual_reproduction(state: dict[str, Any], predicted_faults: list[dict[str, Any]]) -> dict[str, Any]:
    """Score prediction using a value-grounded mechanism-gated replay model.

    Generic symptom overlap cannot make a wrong mechanism pass. The score uses
    only redacted telemetry + the prediction; hidden labels are never consulted.
    Schema/key names are explicitly excluded from mechanism evidence.
    """
    observed = symptom_signature(state)
    predicted = predict_fault_signature(state, predicted_faults)
    if not predicted_faults:
        return {
            "mode": "counterfactual_offline_twin_v3_value_grounded",
            "reproduction_score": 0.0,
            "predicted_signature": predicted,
            "observed_signature": observed,
            "reason": "empty_prediction",
            "uses_oracle_labels_for_score": False,
            "uses_full_state_for_score": False,
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
    active_weight = 0.0
    weighted_overlap = 0.0
    for channel, weight in weights.items():
        expected = predicted.get(channel, []) or []
        channel_score = _set_f1(expected, observed.get(channel, []) or [])
        channel_scores[channel] = channel_score
        if expected:
            active_weight += weight
            weighted_overlap += weight * channel_score
    footprint_overlap = weighted_overlap / active_weight if active_weight > 0 else 0.0

    per_fault = []
    mechanism_scores: list[float] = []
    location_scores: list[float] = []
    observed_affected = observed.get("affected_services", []) or []

    for fault in predicted_faults:
        service = _norm_service(fault.get("service"))
        ft = _norm_fault_type(fault.get("fault_type") or fault.get("fault_family"))
        profile = _observed_mechanism_profile(state, service)
        support = profile.get(ft, 0.0)
        location = 1.0 if service and _service_supported(service, observed_affected) else 0.0
        mechanism_scores.append(support)
        location_scores.append(location)
        per_fault.append({
            "service": service,
            "fault_type": ft,
            "mechanism_support": round(support, 4),
            "location_support": round(location, 4),
            "observed_mechanism_profile": profile,
        })

    mechanism_support = sum(mechanism_scores) / max(len(mechanism_scores), 1)
    location_support = sum(location_scores) / max(len(location_scores), 1)

    mechanism_gate = mechanism_support ** 1.5
    ungated = (
        0.55 * mechanism_support
        + 0.25 * location_support
        + 0.20 * footprint_overlap
    )
    score = ungated * (0.05 + 0.95 * mechanism_gate)

    return {
        "mode": "counterfactual_offline_twin_v3_value_grounded",
        "reproduction_score": round(max(0.0, min(1.0, score)), 4),
        "mechanism_support_score": round(mechanism_support, 4),
        "mechanism_gate": round(mechanism_gate, 4),
        "predicted_service_support": round(location_support, 4),
        "counterfactual_overlap_score": round(footprint_overlap, 4),
        "channel_scores": {k: round(v, 4) for k, v in channel_scores.items()},
        "per_fault": per_fault,
        "predicted_signature": predicted,
        "observed_signature": observed,
        "uses_oracle_labels_for_score": False,
        "uses_full_state_for_score": False,
        "uses_schema_keys_as_mechanism_evidence": False,
    }
