from collections import defaultdict
from pathlib import Path

from config import (
    LOGS_DIR,
    METRICS_DIR,
    SYSTEM_DIR,
    TRACES_DIR,
    OUTPUT_DIR,
    STATE_JSONL,
    STATE_SUMMARY_JSON,
    GRAPH_JSON,
    SNAPSHOT_INDEX_JSON,
    SERVICES,
    DEFAULT_HISTORY,
)
from utils import (
    extract_timestamp_from_name,
    sort_timestamps,
    write_json,
    append_jsonl,
    ensure_dir,
)
from collections import defaultdict
from parsers.logs_parser import parse_logs_snapshot
from parsers.metrics_parser import parse_metrics_snapshot
from parsers.system_parser import parse_system_snapshot
from parsers.traces_parser import parse_traces_snapshot
from parsers.workload_parser import parse_workload_from_traces
from parsers.graph_builder import build_graph


def index_files():
    idx = defaultdict(lambda: {
        "logs": {},
        "metrics": {},
        "system": {},
        "traces": [],
    })

    for f in LOGS_DIR.glob("*.log"):
        ts = extract_timestamp_from_name(f.name)
        if not ts:
            continue
        svc = f.name.split("_")[0]
        idx[ts]["logs"][svc] = f

    for f in METRICS_DIR.glob("*.json"):
        ts = extract_timestamp_from_name(f.name)
        if not ts:
            continue
        metric_type = f.name.split("_")[0]
        if f.name.startswith("network_rx"):
            metric_type = "network_rx"
        elif f.name.startswith("network_tx"):
            metric_type = "network_tx"
        elif f.name.startswith("cpu"):
            metric_type = "cpu"
        elif f.name.startswith("memory"):
            metric_type = "memory"
        elif f.name.startswith("restarts"):
            metric_type = "restarts"

        idx[ts]["metrics"][metric_type] = f

    for f in SYSTEM_DIR.glob("*.txt"):
        ts = extract_timestamp_from_name(f.name)
        if not ts:
            continue
        if f.name.startswith("pods_"):
            idx[ts]["system"]["pods"] = f
        elif f.name.startswith("describe_"):
            idx[ts]["system"]["describe"] = f

    for f in TRACES_DIR.glob("*.json"):
        ts = extract_timestamp_from_name(f.name)
        if not ts:
            continue
        idx[ts]["traces"].append(f)

    return idx


def derive_latency_from_traces(service_metrics, trace_edges):
    incoming = defaultdict(list)
    for edge, feats in trace_edges.items():
        if "->" not in edge:
            continue
        _, dst = edge.split("->", 1)
        incoming[dst].append(feats.get("latency_p95_us", 0.0) / 1000.0)  # ms

    for svc in SERVICES:
        vals = incoming.get(svc, [])
        service_metrics.setdefault(svc, {})
        service_metrics[svc]["latency"] = max(vals) if vals else 0.0

    return service_metrics


def build_state_one_snapshot(ts, files):
    logs = parse_logs_snapshot(files["logs"])
    metrics = parse_metrics_snapshot(files["metrics"])
    system = parse_system_snapshot(files["system"])
    trace_edges, observed_edges = parse_traces_snapshot(files["traces"])
    metrics = derive_latency_from_traces(metrics, trace_edges)
    workload = parse_workload_from_traces(trace_edges)

    graph = build_graph(
        service_metrics=metrics,
        service_logs=logs,
        service_system=system,
        trace_edges=trace_edges,
        observed_edges=observed_edges,
    )

    rca_features = build_rca_features(
        metrics=metrics,
        logs=logs,
        traces=trace_edges,
        system=system,
    )

    state = {
        "timestamp": ts,
        "state_type": "system_health_representation",
        "metrics": metrics,
        "logs": logs,
        "traces": trace_edges,
        "system": system,
        "workload": workload,
        "graph": {
            "services": list(graph["nodes"].keys()),
            "edges": graph["edges"],
        },
        "history": DEFAULT_HISTORY,
        "rca_features": rca_features,
    }

    return state, graph

from collections import defaultdict

def build_rca_features(metrics, logs, traces, system):
    service_scores = defaultdict(float)
    reasons = defaultdict(list)

    logs_available = False
    metrics_available = False
    trace_available = bool(traces)
    system_available = bool(system)

    # logs
    for svc, feats in logs.items():
        if feats.get("line_count", 0) > 0:
            logs_available = True

        score = feats.get("log_anomaly_score", 0.0)
        if score > 0:
            service_scores[svc] += score
            reasons[svc].append(f"log_anomaly_score={score:.3f}")

        det = feats.get("dominant_error_type", "none")
        if det != "none":
            reasons[svc].append(f"dominant_error_type={det}")

    # metrics
    for svc, feats in metrics.items():
        nonzero_resource = any(
            feats.get(k, 0.0) not in [0, 0.0]
            for k in ["cpu", "memory", "network_rx", "network_tx", "restarts"]
        )
        if nonzero_resource:
            metrics_available = True

        lat = feats.get("latency", 0.0)
        if lat > 100:
            service_scores[svc] += 0.25
            reasons[svc].append(f"high_latency_ms={lat:.3f}")

    # system
    for svc, feats in system.items():
        if feats.get("infra_issue_flag", False):
            service_scores[svc] += 0.5
            reasons[svc].append(f"infra_issue={feats.get('service_health_status')}")

    # traces
    most_suspicious_edge = None
    best_edge_score = -1.0
    dominant_error_family = "unknown"

    for edge, feats in traces.items():
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)

        er = feats.get("error_ratio", 0.0)
        es = feats.get("edge_rank_score", 0.0)
        ft = feats.get("failure_type", "healthy_path")

        service_scores[dst] += es
        if es > 0.2:
            reasons[dst].append(f"edge_issue={edge}")
            reasons[dst].append(f"edge_failure_type={ft}")

        # self-loop is especially useful
        if src == dst and er > 0.5:
            service_scores[dst] += 0.4
            reasons[dst].append("self_loop_failure")

        if es > best_edge_score:
            best_edge_score = es
            most_suspicious_edge = edge
            if er >= 0.9:
                dominant_error_family = "dependency_failure"
            elif ft in ["high_latency_path", "degraded_latency_path"]:
                dominant_error_family = "latency_degradation"

    ranked = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)

    candidates = []
    for svc, score in ranked[:5]:
        candidates.append({
            "service": svc,
            "score": round(score, 4),
            "reasons": reasons[svc][:8],
        })

    blind_spots = []
    if not logs_available:
        blind_spots.append("log_pipeline_empty")
    if not metrics_available:
        blind_spots.append("resource_metric_pipeline_empty")
    if not trace_available:
        blind_spots.append("trace_pipeline_empty")

    confidence = 0.85
    if blind_spots:
        confidence -= 0.15 * len(blind_spots)
    confidence = max(0.2, min(0.95, confidence))

    return {
        "candidate_root_causes": candidates,
        "most_suspicious_service": candidates[0]["service"] if candidates else None,
        "most_suspicious_edge": most_suspicious_edge,
        "dominant_error_family": dominant_error_family,
        "observability_gaps": blind_spots,
        "confidence": round(confidence, 3),
        "observability": {
            "logs_available": logs_available,
            "resource_metrics_available": metrics_available,
            "trace_available": trace_available,
            "system_available": system_available,
            "blind_spots": blind_spots,
        }
    }

def build_summary(states):
    summary = {
        "num_snapshots": len(states),
        "services": SERVICES,
        "categories_present": [
            "metrics",
            "logs",
            "traces",
            "system",
            "workload",
            "graph",
            "history",
        ],
        "state_definition": "System Health Representation",
    }
    return summary


def main():
    ensure_dir(OUTPUT_DIR)

    if STATE_JSONL.exists():
        STATE_JSONL.unlink()

    index = index_files()
    timestamps = sort_timestamps(index.keys())

    all_states = []
    last_graph = None

    for ts in timestamps:
        state, graph = build_state_one_snapshot(ts, index[ts])
        append_jsonl(state, STATE_JSONL)
        all_states.append(state)
        last_graph = graph

    write_json(index_to_serializable(index), SNAPSHOT_INDEX_JSON)
    write_json(build_summary(all_states), STATE_SUMMARY_JSON)
    if last_graph is not None:
        write_json(last_graph, GRAPH_JSON)

    print(f"Wrote {len(all_states)} states to {STATE_JSONL}")
    print(f"Wrote summary to {STATE_SUMMARY_JSON}")
    print(f"Wrote graph to {GRAPH_JSON}")


def index_to_serializable(index):
    out = {}
    for ts, item in index.items():
        out[ts] = {
            "logs": {k: str(v) for k, v in item["logs"].items()},
            "metrics": {k: str(v) for k, v in item["metrics"].items()},
            "system": {k: str(v) for k, v in item["system"].items()},
            "traces": [str(x) for x in item["traces"]],
        }
    return out


if __name__ == "__main__":
    main()