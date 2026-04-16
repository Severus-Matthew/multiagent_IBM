import json
from collections import defaultdict
from utils import mean, percentile


def classify_trace_failure(error_ratio, latency_p95_us):
    if error_ratio >= 0.9:
        return "hard_error_path"
    if error_ratio >= 0.3:
        return "partial_error_path"
    if latency_p95_us >= 500000:
        return "high_latency_path"
    if latency_p95_us >= 100000:
        return "degraded_latency_path"
    return "healthy_path"


def edge_rank_score(error_ratio, latency_p95_us):
    score = 0.0
    score += min(error_ratio, 1.0) * 0.8
    score += min(latency_p95_us / 1_000_000.0, 1.0) * 0.2
    return min(score, 1.0)


def parse_trace_file(path):
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}, []

    traces = obj.get("data", [])
    edge_stats = defaultdict(lambda: {
        "request_count": 0,
        "error_count": 0,
        "durations_us": [],
    })
    observed_edges = set()

    for trace in traces:
        spans = trace.get("spans", [])
        processes = trace.get("processes", {})

        span_by_id = {}
        for s in spans:
            sid = s.get("spanID")
            if sid:
                span_by_id[sid] = s

        def span_service(span):
            pid = span.get("processID")
            proc = processes.get(pid, {})
            return proc.get("serviceName", "unknown")

        for span in spans:
            child_service = span_service(span)
            refs = span.get("references", [])
            error_flag = False

            for t in span.get("tags", []):
                if t.get("key") == "error" and str(t.get("value")).lower() == "true":
                    error_flag = True

            for ref in refs:
                if ref.get("refType") != "CHILD_OF":
                    continue
                parent_span = span_by_id.get(ref.get("spanID"))
                if not parent_span:
                    continue
                parent_service = span_service(parent_span)

                edge = f"{parent_service}->{child_service}"
                observed_edges.add((parent_service, child_service))
                edge_stats[edge]["request_count"] += 1
                edge_stats[edge]["durations_us"].append(float(span.get("duration", 0.0)))
                if error_flag:
                    edge_stats[edge]["error_count"] += 1

    final = {}
    for edge, stats in edge_stats.items():
        req = stats["request_count"]
        errs = stats["error_count"]
        durs = stats["durations_us"]

        error_ratio = (errs / req) if req else 0.0
        latency_mean_us = mean(durs)
        latency_p95_us = percentile(durs, 95)
        latency_max_us = max(durs) if durs else 0.0

        final[edge] = {
            "request_count": req,
            "error_ratio": error_ratio,
            "latency_mean_us": latency_mean_us,
            "latency_p95_us": latency_p95_us,
            "latency_max_us": latency_max_us,
            "dependency_failure_score": error_ratio + (latency_p95_us / 1e6),
            "failure_type": classify_trace_failure(error_ratio, latency_p95_us),
            "is_suspicious": (error_ratio > 0.2 or latency_p95_us > 100000),
            "edge_rank_score": edge_rank_score(error_ratio, latency_p95_us),
        }

    return final, sorted(list(observed_edges))


def parse_traces_snapshot(files):
    merged = {}
    all_edges = set()

    for f in files:
        edge_stats, observed_edges = parse_trace_file(f)
        for edge, feats in edge_stats.items():
            if edge not in merged:
                merged[edge] = feats
            else:
                merged[edge]["request_count"] += feats["request_count"]
                merged[edge]["error_ratio"] = max(merged[edge]["error_ratio"], feats["error_ratio"])
                merged[edge]["latency_mean_us"] = max(merged[edge]["latency_mean_us"], feats["latency_mean_us"])
                merged[edge]["latency_p95_us"] = max(merged[edge]["latency_p95_us"], feats["latency_p95_us"])
                merged[edge]["latency_max_us"] = max(merged[edge]["latency_max_us"], feats["latency_max_us"])
                merged[edge]["dependency_failure_score"] = max(
                    merged[edge]["dependency_failure_score"],
                    feats["dependency_failure_score"],
                )
                merged[edge]["edge_rank_score"] = max(
                    merged[edge]["edge_rank_score"],
                    feats["edge_rank_score"],
                )
                merged[edge]["is_suspicious"] = merged[edge]["is_suspicious"] or feats["is_suspicious"]

        all_edges.update(observed_edges)

    return merged, sorted(list(all_edges))