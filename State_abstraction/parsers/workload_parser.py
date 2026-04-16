from collections import defaultdict


def parse_workload_from_traces(trace_edges):
    per_service_incoming = defaultdict(int)
    total_requests = 0

    for edge, feats in trace_edges.items():
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)
        req = feats.get("request_count", 0)
        per_service_incoming[dst] += req
        total_requests += req

    return {
        "estimated_request_rate": total_requests,
        "per_service_incoming_requests": dict(per_service_incoming),
    }