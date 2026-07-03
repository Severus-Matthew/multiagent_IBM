from collections import defaultdict


def parse_workload_from_traces(trace_edges):
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    total = 0
    for edge, feats in trace_edges.items():
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)
        req = int(feats.get("request_count", 0) or 0)
        incoming[dst] += req
        outgoing[src] += req
        total += req
    return {
        "estimated_request_rate": total,
        "per_service_incoming_requests": dict(incoming),
        "per_service_outgoing_requests": dict(outgoing),
        "traffic_signal_present": total > 0,
    }
