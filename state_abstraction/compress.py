import argparse
import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter


STAT_KEYS = ["count", "first", "last", "min", "max", "mean", "delta"]


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


def stable_json_key(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def group_identical_objects(named_objects):
    """
    Groups only if the full value object is identical.

    Input:
      {
        "ip6gre0": {"signal": "zero", "count": 20},
        "gre0": {"signal": "zero", "count": 20},
        "eth0": {"signal": "dynamic", ...}
      }

    Output:
      [
        {
          "names": ["ip6gre0", "gre0"],
          "value": {"signal": "zero", "count": 20}
        },
        {
          "names": ["eth0"],
          "value": {"signal": "dynamic", ...}
        }
      ]
    """
    grouped = {}

    for name, obj in named_objects.items():
        key = stable_json_key(obj)
        grouped.setdefault(key, {
            "names": [],
            "value": obj,
        })
        grouped[key]["names"].append(name)

    out = list(grouped.values())
    for g in out:
        g["names"] = sorted(g["names"])

    return out


def group_service_profiles(per_service_dict, profile_key="profile"):
    grouped = group_identical_objects(per_service_dict)

    return [
        {
            "services": g["names"],
            profile_key: g["value"],
        }
        for g in grouped
    ]


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
        return {
            "signal": "zero",
            "count": count,
        }

    if delta == 0 and min_v == max_v:
        return {
            "signal": "constant",
            "count": count,
            "value": last,
        }

    return {
        "signal": "dynamic",
        "count": count,
        "first": first,
        "last": last,
        "min": min_v,
        "max": max_v,
        "mean": mean,
        "delta": delta,
    }


def split_metric_name(metric_name):
    metric_name = str(metric_name)

    if "." in metric_name:
        base, suffix = metric_name.split(".", 1)
    else:
        base, suffix = metric_name, "global"

    base = re.sub(r"^kpi_", "", base)
    base = re.sub(r"^container_", "", base)

    return base, suffix


def aggregate_metric_group(items):
    dynamic = []
    zero = []
    constant = []

    for name, stats in items.items():
        sig = stats.get("signal") if isinstance(stats, dict) else None

        if sig == "zero":
            zero.append(name)
        elif sig == "constant":
            constant.append((name, stats.get("value", 0.0)))
        elif sig == "dynamic":
            dynamic.append((name, stats))

    agg = {
        "num_items": len(items),
        "zero_items": sorted(zero),
        "constant_items": [
            {"names": sorted([name for name, val in constant if val == value]), "value": value}
            for value in sorted(set(val for _, val in constant))
        ],
        "dynamic_items": sorted([n for n, _ in dynamic]),
    }

    if dynamic:
        agg["dynamic_summary"] = {
            "max_of_max": max(s.get("max", 0.0) for _, s in dynamic),
            "sum_delta": sum(s.get("delta", 0.0) for _, s in dynamic),
            "mean_of_means": sum(s.get("mean", 0.0) for _, s in dynamic) / len(dynamic),
        }

    return round_deep(agg)


def compress_metrics(metrics):
    compressed = {}

    skip_keys = {
        "raw_kpis",
        "metric_signal_present",
        "cpu_usage_delta",
        "cpu_load_last",
        "memory_working_set_last",
        "memory_usage_last",
        "network_rx_bytes_delta",
        "network_tx_bytes_delta",
        "network_tx_errors_delta",
        "network_rx_drops_delta",
        "network_tx_drops_delta",
        "threads_last",
        "latency_ms",
        "latency",
    }

    for svc, svc_metrics in metrics.items():
        out = {
            "groups": {},
            "flat_summary": {},
        }

        for category, cat_metrics in svc_metrics.items():
            if category in skip_keys:
                continue

            if not isinstance(cat_metrics, dict):
                continue

            out["groups"].setdefault(category, {})

            for metric_name, stats in cat_metrics.items():
                if not is_stat_dict(stats):
                    continue

                base, suffix = split_metric_name(metric_name)
                compressed_stats = metric_signal(stats)

                out["groups"][category].setdefault(base, {})
                out["groups"][category][base][suffix] = round_deep(compressed_stats)

        for category in list(out["groups"].keys()):
            for base, items in list(out["groups"][category].items()):
                out["groups"][category][base] = {
                    "items_grouped": group_identical_objects(items),
                    "summary": aggregate_metric_group(items),
                }

        raw = svc_metrics.get("raw_kpis", {})

        out["flat_summary"] = round_deep({
            "cpu_usage_delta": raw.get("container_cpu_usage_seconds_total", {}).get("delta", 0.0),
            "cpu_user_delta": raw.get("container_cpu_user_seconds_total", {}).get("delta", 0.0),
            "cpu_system_delta": raw.get("container_cpu_system_seconds_total", {}).get("delta", 0.0),
            "cpu_load_last": raw.get("container_cpu_load_average_10s", {}).get("last", 0.0),
            "memory_usage_last": raw.get("container_memory_usage_bytes", {}).get("last", 0.0),
            "memory_working_set_last": raw.get("container_memory_working_set_bytes", {}).get("last", 0.0),
            "memory_rss_last": raw.get("container_memory_rss", {}).get("last", 0.0),
            "threads_last": raw.get("container_threads", {}).get("last", 0.0),
            "latency_ms": svc_metrics.get("latency_ms", svc_metrics.get("latency", 0.0)),
        })

        compressed[svc] = out

    return compressed


ERROR_FAMILY_PATTERNS = {
    "dependency": [
        "dependency",
        "connection",
        "unavailable",
        "no_ready_endpoints",
        "endpoint",
    ],
    "timeout": [
        "timeout",
        "timed out",
        "serverselectiontimeoutms",
    ],
    "database": [
        "mongodb",
        "mongo",
        "db",
        "database",
        "index",
    ],
    "rpc": [
        "rpc",
        "grpc",
        "thrift",
        "transport",
    ],
    "auth": [
        "auth",
        "unauthorized",
        "forbidden",
        "permission",
    ],
    "resource": [
        "oom",
        "memory",
        "cpu",
        "resource",
    ],
    "crash": [
        "crash",
        "fatal",
        "panic",
        "segfault",
    ],
    "http": [
        "http",
        "5xx",
        "4xx",
        "status",
    ],
}


def map_error_family(text):
    text = str(text or "").lower()

    for fam, pats in ERROR_FAMILY_PATTERNS.items():
        if any(p in text for p in pats):
            return fam

    return "other"


def compress_logs_per_service(logs):
    compressed = {}

    for svc, l in logs.items():
        error_type_counts = l.get("error_type_counts", {}) or {}
        dep_counts = l.get("dependency_error_counts", {}) or {}
        severity_counts = l.get("severity_counts", {}) or {}

        family_counts = Counter()

        for err_type, count in error_type_counts.items():
            family_counts[map_error_family(err_type)] += count

        for dep, count in dep_counts.items():
            family_counts["dependency"] += count
            if "mongo" in dep.lower() or "db" in dep.lower():
                family_counts["database"] += count

        compressed[svc] = round_deep({
            "signal": {
                "line_count": l.get("line_count", 0),
                "error_count": l.get("error_count", 0),
                "warn_count": l.get("warn_count", 0),
                "fatal_count": l.get("fatal_count", 0),
                "log_anomaly_score": l.get("log_anomaly_score", 0.0),
                "log_health_score": l.get("log_health_score", 1.0),
                "dominant_error_type": l.get("dominant_error_type", "none"),
            },
            "severity_counts": severity_counts,
            "error_families": dict(family_counts),
            "dependency_error_counts": dep_counts,
            "dependency_errors_top": l.get("dependency_errors", [])[:10],
            "evidence_lines_top": l.get("evidence_lines", [])[:5],
            "error_templates_top": l.get("top_error_templates", [])[:5],
        })

    return compressed


def compress_logs(logs):
    per_service = compress_logs_per_service(logs)

    return {
        "per_service": per_service,
        "grouped_profiles": group_service_profiles(per_service, profile_key="log_profile"),
    }


def compress_traces_per_edge(traces):
    compressed = {}
    global_summary = {
        "num_edges": 0,
        "failed_edges": [],
        "slow_edges": [],
        "top_edges_by_score": [],
        "fan_in": Counter(),
        "fan_out": Counter(),
    }

    edge_scores = []

    for edge, t in traces.items():
        if "->" in edge:
            src, dst = edge.split("->", 1)
        else:
            src, dst = "unknown", edge

        req = t.get("request_count", 0)
        err = t.get("error_count", 0)
        ratio = t.get("error_ratio", 0.0)
        p95 = t.get("latency_p95_us", 0.0)
        score = t.get("edge_rank_score", 0.0)

        item = round_deep({
            "source": src,
            "target": dst,
            "request_count": req,
            "error_count": err,
            "error_ratio": ratio,
            "latency_p95_ms": p95 / 1000.0,
            "failure_type": t.get("failure_type", "unknown"),
            "is_suspicious": t.get("is_suspicious", False),
            "edge_rank_score": score,
            "top_operations": t.get("top_operations", {}),
            "responses": t.get("responses", {}),
        })

        compressed[edge] = item

        global_summary["num_edges"] += 1
        global_summary["fan_out"][src] += 1
        global_summary["fan_in"][dst] += 1

        if item["error_count"] > 0 or item["error_ratio"] > 0:
            global_summary["failed_edges"].append(edge)

        if item["latency_p95_ms"] > 100:
            global_summary["slow_edges"].append(edge)

        edge_scores.append((edge, score))

    global_summary["top_edges_by_score"] = [
        edge for edge, _ in sorted(edge_scores, key=lambda x: x[1], reverse=True)[:10]
    ]
    global_summary["fan_in"] = dict(global_summary["fan_in"])
    global_summary["fan_out"] = dict(global_summary["fan_out"])

    return compressed, round_deep(global_summary)


def compress_traces(traces):
    per_edge, summary = compress_traces_per_edge(traces)

    return {
        "per_edge": per_edge,
        "grouped_edge_profiles": group_service_profiles(per_edge, profile_key="trace_profile"),
        "summary": summary,
    }


def compress_ports(ports):
    """
    Preserves names when present.
    Groups only when full port object is identical.
    """
    port_objects = {}

    for i, p in enumerate(ports or []):
        if not isinstance(p, dict):
            continue

        port = (
            p.get("container_port")
            if p.get("container_port") is not None
            else p.get("port")
            if p.get("port") is not None
            else p.get("target_port")
        )

        protocol = p.get("protocol", "UNKNOWN")
        name = p.get("name")

        obj = {
            "port": port,
            "protocol": protocol,
            "name": name,
        }

        key_name = name if name else f"unnamed_{port}_{protocol}_{i}"
        port_objects[key_name] = obj

    grouped = group_identical_objects(port_objects)

    if all(len(g["names"]) == 1 for g in grouped):
        return [
            {
                "name": g["value"].get("name"),
                "port": g["value"].get("port"),
                "protocol": g["value"].get("protocol"),
            }
            for g in grouped
        ]

    return grouped


def compact_list(xs, max_items=10):
    xs = xs or []
    if len(xs) <= max_items:
        return xs
    return {
        "items": xs[:max_items],
        "truncated": True,
        "total_count": len(xs),
    }


def compress_k8s_service(obj):
    if not isinstance(obj, dict):
        return {}

    ports = obj.get("ports", [])

    return {
        "service_name": obj.get("service_name"),
        "namespace": obj.get("namespace"),
        "service_type": obj.get("service_type"),
        "cluster_ip": obj.get("cluster_ip"),
        "selector": obj.get("selector", {}),
        "ports": compress_ports(ports),
        "session_affinity": obj.get("session_affinity"),
        "internal_traffic_policy": obj.get("internal_traffic_policy"),
    }


def compress_endpoints(obj):
    if not isinstance(obj, dict):
        return {}

    return {
        "ready_endpoint_count": obj.get("ready_endpoint_count", 0),
        "not_ready_endpoint_count": obj.get("not_ready_endpoint_count", 0),
        "has_ready_endpoints": obj.get("has_ready_endpoints", False),
        "ports": compress_ports(obj.get("ports", [])),
        "addresses": compact_list(obj.get("addresses", []), max_items=5),
        "not_ready_addresses": compact_list(obj.get("not_ready_addresses", []), max_items=5),
    }


def compress_deployment(obj):
    if not isinstance(obj, dict):
        return {}

    return {
        "deployment_name": obj.get("deployment_name"),
        "replicas_desired": obj.get("replicas_desired", 0),
        "replicas_current": obj.get("replicas_current", 0),
        "replicas_ready": obj.get("replicas_ready", 0),
        "replicas_available": obj.get("replicas_available", 0),
        "replicas_unavailable": obj.get("replicas_unavailable", 0),
        "conditions": obj.get("conditions", {}),
    }


def compress_system_per_service(system):
    compressed = {}

    for svc, s in system.items():
        ports = s.get("ports", [])

        compressed[svc] = round_deep({
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
            "pods": compact_list(s.get("pod_names", []), max_items=10),
            "ips": compact_list(s.get("pod_ips", []), max_items=10),
            "nodes": compact_list(s.get("nodes", []), max_items=5),
            "containers": compact_list(s.get("containers", []), max_items=10),
            "images": compact_list(s.get("images", []), max_items=5),
            "ports": compress_ports(ports),
            "endpoints": compress_endpoints(s.get("endpoints", {})),
            "k8s_service": compress_k8s_service(s.get("k8s_service", {})),
            "deployment": compress_deployment(s.get("deployment", {})),
            "events_top": compact_list(s.get("events", []), max_items=5),
        })

    return compressed


def compress_system(system):
    per_service = compress_system_per_service(system)

    return {
        "per_service": per_service,
        "grouped_profiles": group_service_profiles(per_service, profile_key="system_profile"),
    }


def build_model_vector(state):
    rows = []

    services = state.get("services", [])
    metrics = state.get("metrics", {})
    logs = state.get("logs", {})
    system = state.get("system", {})

    for svc in services:
        raw = metrics.get(svc, {}).get("raw_kpis", {})
        log = logs.get(svc, {})
        sys = system.get(svc, {})

        row = {
            "service": svc,
            "cpu_delta": raw.get("container_cpu_usage_seconds_total", {}).get("delta", 0.0),
            "cpu_user_delta": raw.get("container_cpu_user_seconds_total", {}).get("delta", 0.0),
            "cpu_system_delta": raw.get("container_cpu_system_seconds_total", {}).get("delta", 0.0),
            "mem_usage_last": raw.get("container_memory_usage_bytes", {}).get("last", 0.0),
            "mem_working_last": raw.get("container_memory_working_set_bytes", {}).get("last", 0.0),
            "threads_last": raw.get("container_threads", {}).get("last", 0.0),
            "log_errors": log.get("error_count", 0),
            "log_warnings": log.get("warn_count", 0),
            "log_anomaly": log.get("log_anomaly_score", 0.0),
            "restart_count": sys.get("restart_count", 0),
            "pods_unready": sys.get("pods_unready", 0),
            "availability_ratio": sys.get("availability_ratio", 0.0),
            "infra_issue": 1.0 if sys.get("infra_issue_flag", False) else 0.0,
        }

        rows.append(row)

    return round_deep(rows)


def zscore_rows(rows):
    if not rows:
        return rows

    keys = [k for k in rows[0] if k != "service"]

    means = {}
    stds = {}

    for k in keys:
        vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
        means[k] = sum(vals) / len(vals)
        var = sum((v - means[k]) ** 2 for v in vals) / len(vals)
        stds[k] = math.sqrt(var) if var > 0 else 1.0

    out = []

    for r in rows:
        nr = {"service": r["service"]}
        for k in keys:
            nr[k] = (float(r.get(k, 0.0) or 0.0) - means[k]) / stds[k]
        out.append(round_deep(nr))

    return out


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

    logs_per_service = compressed.get("logs", {}).get("per_service", {})

    for svc, l in logs_per_service.items():
        sig = l.get("signal", {})
        if sig.get("error_count", 0) > 0 or sig.get("log_anomaly_score", 0) > 0.3:
            top_log_error_services.append({
                "service": svc,
                "error_count": sig.get("error_count", 0),
                "dominant_error_type": sig.get("dominant_error_type"),
                "error_families": l.get("error_families", {}),
                "dependency_error_counts": l.get("dependency_error_counts", {}),
                "evidence": l.get("evidence_lines_top", [])[:2],
            })

    top_log_error_services = sorted(
        top_log_error_services,
        key=lambda x: x["error_count"],
        reverse=True,
    )[:15]

    return {
        "scenario_id": compressed.get("scenario_id"),
        "fault_context": compressed.get("fault_context"),
        "top_log_error_services": top_log_error_services,
        "trace_summary": compressed.get("traces", {}).get("summary", {}),
        "service_clusters": compressed.get("clusters", {}),
    }


def compress_state(state):
    compressed = {
        "timestamp": state.get("timestamp"),
        "scenario_id": state.get("scenario_id"),
        "state_type": "compressed_aiops_state_abstraction_v2",
        "source_state_type": state.get("state_type"),
        "fault_context": state.get("fault_context", {}),
        "services": state.get("services", []),
    }

    compressed["metrics"] = compress_metrics(state.get("metrics", {}))
    compressed["logs"] = compress_logs(state.get("logs", {}))
    compressed["traces"] = compress_traces(state.get("traces", {}))
    compressed["system"] = compress_system(state.get("system", {}))

    compressed["workload"] = state.get("workload", {})
    compressed["graph"] = state.get("graph", {})
    compressed["sla"] = state.get("sla", {})
    compressed["rca_features"] = state.get("rca_features", {})
    compressed["service_health"] = state.get("service_health", {})

    model_rows = build_model_vector(state)
    compressed["model_table"] = model_rows
    compressed["model_table_zscore"] = zscore_rows(model_rows)
    compressed["clusters"] = simple_cluster_rows(model_rows)

    compressed["llm_view"] = build_llm_view(compressed)

    return compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to state_abstraction.json")
    ap.add_argument("--output", default=None, help="Path to compressed output json")
    ap.add_argument("--keep_original", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    state = read_json(inp)

    compressed = compress_state(state)

    if args.keep_original:
        compressed["original_state"] = state

    out = Path(args.output) if args.output else inp.with_name("state_abstraction_compressed_v2.json")
    write_json(compressed, out)

    print(f"[OK] wrote compressed state to {out}")
    print(f"[INFO] services: {len(compressed.get('services', []))}")
    print(f"[INFO] clusters: {json.dumps(compressed.get('clusters', {}), indent=2)}")


if __name__ == "__main__":
    main()