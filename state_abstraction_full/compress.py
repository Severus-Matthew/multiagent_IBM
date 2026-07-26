import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


STAT_KEYS = ["count", "first", "last", "min", "max", "mean", "delta"]
LEAK_VALUE_MARKERS = (
    "scenario_fault_context",
    "generated_fault_context",
    "oracle_ground_truth",
    "oracle_ground_truth_fault",
    "oracle_neighbor_of_",
    "ranked_by_rca_context_or_weak_signals",
    "suspect_silent_failure",
)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def round_float(x, ndigits=4):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return round(x, ndigits)
    return x


def round_deep(x, ndigits=4):
    if isinstance(x, dict):
        return {k: round_deep(v, ndigits) for k, v in x.items()}
    if isinstance(x, list):
        return [round_deep(v, ndigits) for v in x]
    return round_float(x, ndigits)


def is_stat_dict(x):
    return isinstance(x, dict) and all(k in x for k in STAT_KEYS)


def metric_signal(stats):
    if not is_stat_dict(stats):
        return stats
    count = stats.get("count", 0)
    first = stats.get("first", 0.0)
    last = stats.get("last", 0.0)
    min_v = stats.get("min", 0.0)
    max_v = stats.get("max", 0.0)
    mean = stats.get("mean", 0.0)
    delta = stats.get("delta", 0.0)
    if max_v == 0 and mean == 0 and delta == 0:
        return {"signal": "zero", "count": count}
    if delta == 0 and min_v == max_v:
        return {"signal": "constant", "count": count, "value": last}
    return {"signal": "dynamic", "count": count, "first": first, "last": last, "min": min_v, "max": max_v, "mean": mean, "delta": delta}


def _kpi(raw, *candidates):
    for c in candidates:
        if c in raw and raw[c]:
            return raw[c]
    lc = candidates[-1].lower().replace("kpi_", "").replace("container_", "")
    for k, v in raw.items():
        if lc in k.lower() and v:
            return v
    return {}


def compress_metrics(metrics):
    out = {}
    for svc, m in metrics.items():
        raw = m.get("raw_kpis", {}) or {}
        groups = {}
        for group in ["cpu", "memory", "network", "threads", "spec", "other"]:
            group_obj = m.get(group, {}) or {}
            summarized = {}
            for name, stats in group_obj.items():
                summarized[name] = round_deep(metric_signal(stats))
            if summarized:
                groups[group] = summarized
        out[svc] = round_deep({
            "flat_summary": {
                "cpu_usage_delta": m.get("cpu_usage_delta", _kpi(raw, "kpi_container_cpu_usage_seconds_total", "container_cpu_usage_seconds_total").get("delta", 0.0)),
                "cpu_load_last": m.get("cpu_load_last", _kpi(raw, "kpi_container_cpu_load_average_10s", "container_cpu_load_average_10s").get("last", 0.0)),
                "memory_working_set_last": m.get("memory_working_set_last", _kpi(raw, "kpi_container_memory_working_set_bytes", "container_memory_working_set_bytes").get("last", 0.0)),
                "memory_usage_last": m.get("memory_usage_last", _kpi(raw, "kpi_container_memory_usage_bytes", "container_memory_usage_bytes").get("last", 0.0)),
                "network_rx_bytes_delta": m.get("network_rx_bytes_delta", 0.0),
                "network_tx_bytes_delta": m.get("network_tx_bytes_delta", 0.0),
                "network_tx_errors_delta": m.get("network_tx_errors_delta", 0.0),
                "threads_last": m.get("threads_last", _kpi(raw, "kpi_container_threads", "container_threads").get("last", 0.0)),
                "latency_ms": m.get("latency_ms", m.get("latency", 0.0)),
            },
            "groups": groups,
            "metric_signal_present": bool(m.get("metric_signal_present", False)),
        })
    return out


ERROR_FAMILY_PATTERNS = {
    "dependency": ["dependency", "connection", "connect", "refused", "unavailable", "endpoint"],
    "timeout": ["timeout", "timed out", "serverselectiontimeoutms"],
    "database": ["mongodb", "mongo", "db", "database", "index"],
    "rpc": ["rpc", "grpc", "thrift", "transport"],
    "auth": ["auth", "unauthorized", "forbidden", "permission"],
    "resource": ["oom", "memory", "cpu", "resource"],
    "crash": ["crash", "fatal", "panic", "segfault"],
    "http": ["http", "5xx", "4xx", "status"],
}


def map_error_family(text):
    text = str(text or "").lower()
    for fam, pats in ERROR_FAMILY_PATTERNS.items():
        if any(p in text for p in pats):
            return fam
    return "other"


def compress_logs(logs):
    out = {}
    for svc, l in logs.items():
        family_counts = Counter()
        for err_type, count in (l.get("error_type_counts", {}) or {}).items():
            family_counts[map_error_family(err_type)] += count
        for dep, count in (l.get("dependency_error_counts", {}) or {}).items():
            family_counts["dependency"] += count
            if "mongo" in dep.lower() or "db" in dep.lower():
                family_counts["database"] += count
        out[svc] = round_deep({
            "signal": {
                "line_count": l.get("line_count", 0),
                "error_count": l.get("error_count", 0),
                "warn_count": l.get("warn_count", 0),
                "fatal_count": l.get("fatal_count", 0),
                "log_anomaly_score": l.get("log_anomaly_score", 0.0),
                "log_health_score": l.get("log_health_score", 1.0),
                "dominant_error_type": l.get("dominant_error_type", "none"),
            },
            "severity_counts": l.get("severity_counts", {}) or {},
            "error_families": dict(family_counts),
            "dependency_error_counts": l.get("dependency_error_counts", {}) or {},
            "evidence_lines_top": (l.get("evidence_lines", []) or [])[:5],
            "error_templates_top": (l.get("top_error_templates", []) or [])[:5],
        })
    return out


def compress_traces(traces):
    per_edge = {}
    failed_edges = []
    slow_edges = []
    scores = []
    fan_in = Counter()
    fan_out = Counter()
    for edge, t in traces.items():
        src, dst = edge.split("->", 1) if "->" in edge else ("unknown", edge)
        p95_ms = t.get("latency_p95_us", 0.0) / 1000.0
        item = round_deep({
            "source": src,
            "target": dst,
            "request_count": t.get("request_count", 0),
            "error_count": t.get("error_count", 0),
            "error_ratio": t.get("error_ratio", 0.0),
            "latency_p95_ms": p95_ms,
            "failure_type": t.get("failure_type", "unknown"),
            "is_suspicious": t.get("is_suspicious", False),
            "edge_rank_score": t.get("edge_rank_score", 0.0),
            "top_operations": t.get("top_operations", {}),
            "responses": t.get("responses", {}),
        })
        per_edge[edge] = item
        fan_out[src] += 1
        fan_in[dst] += 1
        if item["error_count"] > 0 or item["error_ratio"] > 0:
            failed_edges.append(edge)
        if p95_ms > 100:
            slow_edges.append(edge)
        scores.append((edge, item["edge_rank_score"]))
    return {"per_edge": per_edge, "summary": round_deep({"num_edges": len(per_edge), "failed_edges": failed_edges, "slow_edges": slow_edges, "top_edges_by_score": [e for e, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:10]], "fan_in": dict(fan_in), "fan_out": dict(fan_out)})}


def compact_list(xs, max_items=10):
    xs = xs or []
    if len(xs) <= max_items:
        return xs
    return {"items": xs[:max_items], "truncated": True, "total_count": len(xs)}


def compress_system(system):
    out = {}
    for svc, s in system.items():
        out[svc] = round_deep({
            "health": {
                "status": s.get("service_health_status", "unknown"),
                "infra_issue_flag": s.get("infra_issue_flag", False),
                "availability_ratio": s.get("availability_ratio", 0.0),
                "pods_total": s.get("pods_total", 0),
                "pods_ready": s.get("pods_ready", 0),
                "pods_unready": s.get("pods_unready", 0),
                "restart_count": s.get("restart_count", 0),
                "crashloop_count": s.get("crashloop_count", 0),
                "oomkilled_count": s.get("oomkilled_count", 0),
                "image_pull_error_count": s.get("image_pull_error_count", 0),
                "warning_event_count": s.get("warning_event_count", 0),
            },
            "pods": compact_list(s.get("pod_names", []), 10),
            "nodes": compact_list(s.get("nodes", []), 5),
            "containers": compact_list(s.get("containers", []), 10),
            "images": compact_list(s.get("images", []), 5),
            "endpoints": s.get("endpoints", {}),
            "deployment": s.get("deployment", {}),
            "events_top": compact_list(s.get("events", []), 5),
        })
    return out


def _leaky_text(value) -> bool:
    low = str(value or "").lower()
    return any(marker in low for marker in LEAK_VALUE_MARKERS)


def sanitize_service_health(service_health):
    """Return agent-safe service-health entries.

    Older full states may contain oracle-derived RCA weak labels such as
    `suspect_silent_failure` or `ranked_by_rca_context_or_weak_signals`.
    Those are removed here before compressed state generation.
    """
    out = {}
    for svc, h in (service_health or {}).items():
        if not isinstance(h, dict):
            continue
        status = h.get("status") or "unknown"
        reasons = [str(r) for r in (h.get("reasons", []) or []) if not _leaky_text(r)]
        if _leaky_text(status):
            status = "healthy"
        if status in {"healthy", "unknown", None, ""} and not reasons:
            continue
        if status == "suspect_silent_failure":
            if not reasons:
                continue
            status = "observable_suspect"
        out[svc] = {"status": status, "reasons": reasons[:8]}
    return out


def build_model_vector(state):
    rows = []
    for svc in state.get("services", []):
        raw = state.get("metrics", {}).get(svc, {}).get("raw_kpis", {})
        log = state.get("logs", {}).get(svc, {})
        sys = state.get("system", {}).get(svc, {})
        rows.append(round_deep({
            "service": svc,
            "cpu_delta": _kpi(raw, "kpi_container_cpu_usage_seconds_total", "container_cpu_usage_seconds_total").get("delta", 0.0),
            "mem_usage_last": _kpi(raw, "kpi_container_memory_usage_bytes", "container_memory_usage_bytes").get("last", 0.0),
            "threads_last": _kpi(raw, "kpi_container_threads", "container_threads").get("last", 0.0),
            "log_errors": log.get("error_count", 0),
            "log_warnings": log.get("warn_count", 0),
            "log_anomaly": log.get("log_anomaly_score", 0.0),
            "restart_count": sys.get("restart_count", 0),
            "pods_unready": sys.get("pods_unready", 0),
            "availability_ratio": sys.get("availability_ratio", 0.0),
            "infra_issue": 1.0 if sys.get("infra_issue_flag", False) else 0.0,
        }))
    return rows


def simple_cluster_rows(rows):
    clusters = defaultdict(list)
    for r in rows:
        svc = r["service"]
        if r.get("log_errors", 0) > 0 or r.get("log_anomaly", 0) >= 0.5:
            bucket = "log_error_or_dependency_failure"
        elif r.get("infra_issue", 0) > 0 or r.get("pods_unready", 0) > 0:
            bucket = "infra_unhealthy"
        elif r.get("cpu_delta", 0) > 0 or r.get("mem_usage_last", 0) > 0:
            bucket = "active_but_no_errors"
        else:
            bucket = "low_signal_or_no_data"
        clusters[bucket].append(svc)
    return dict(clusters)


def build_llm_view(compressed):
    top_log_error_services = []
    for svc, l in compressed.get("logs", {}).items():
        sig = l.get("signal", {})
        if sig.get("error_count", 0) > 0 or sig.get("log_anomaly_score", 0) > 0.3:
            top_log_error_services.append({"service": svc, "error_count": sig.get("error_count", 0), "dominant_error_type": sig.get("dominant_error_type"), "error_families": l.get("error_families", {}), "dependency_error_counts": l.get("dependency_error_counts", {}), "evidence": l.get("evidence_lines_top", [])[:2]})
    top_log_error_services = sorted(top_log_error_services, key=lambda x: x["error_count"], reverse=True)[:15]
    return {"scenario_id": compressed.get("scenario_id"), "top_log_error_services": top_log_error_services, "trace_summary": compressed.get("traces", {}).get("summary", {}), "service_clusters": compressed.get("clusters", {})}


def compress_state(state):
    fault_ctx = state.get("fault_context", {}) or {}
    compressed = {
        "timestamp": state.get("timestamp"),
        "scenario_id": state.get("scenario_id"),
        "state_type": "redacted_compressed_aiops_state_abstraction_v3",
        "source_state_type": state.get("state_type"),
        "redaction": {
            "ground_truth_removed": True,
            "fault_context_removed": True,
            "rca_weak_labels_removed": True,
            "service_health_oracle_markers_removed": True,
            "safe_for_rca_agent": True,
        },
        "namespace": fault_ctx.get("target_namespace"),
        "task": fault_ctx.get("task"),
        "services": state.get("services", []),
    }
    compressed["metrics"] = compress_metrics(state.get("metrics", {}))
    compressed["logs"] = compress_logs(state.get("logs", {}))
    compressed["traces"] = compress_traces(state.get("traces", {}))
    compressed["system"] = compress_system(state.get("system", {}))
    compressed["workload"] = state.get("workload", {})
    compressed["graph"] = state.get("graph", {})
    compressed["sla"] = state.get("sla", {})
    compressed["service_health"] = sanitize_service_health(state.get("service_health", {}))
    compressed["observability_metadata"] = state.get("observability_metadata", {})
    model_rows = build_model_vector(state)
    compressed["model_table"] = model_rows
    compressed["clusters"] = simple_cluster_rows(model_rows)
    compressed["llm_view"] = build_llm_view(compressed)
    return compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to state_abstraction.json")
    ap.add_argument("--output", default=None, help="Path to compressed output json")
    args = ap.parse_args()
    inp = Path(args.input)
    compressed = compress_state(read_json(inp))
    out = Path(args.output) if args.output else inp.with_name("state_abstraction_compressed.json")
    write_json(compressed, out)
    print(f"[OK] wrote redacted compressed state to {out}")
    print(f"[INFO] services: {len(compressed.get('services', []))}")


if __name__ == "__main__":
    main()
