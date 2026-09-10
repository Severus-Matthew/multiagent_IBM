from collections import defaultdict


def parse_workload_from_traces(trace_edges, observation_seconds=None):
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    total = 0
    root_requests = 0
    for edge, feats in trace_edges.items():
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)
        req = int(feats.get("request_count", 0) or 0)
        incoming[dst] += req
        outgoing[src] += req
        total += req
        if src.upper() == "ROOT":
            root_requests += req
    return {
        "observed_edge_span_count": total,
        "observed_root_request_count": root_requests,
        "estimated_request_rate": (
            root_requests / observation_seconds
            if observation_seconds is not None and observation_seconds > 0 else None
        ),
        "request_rate_unit": "requests_per_second",
        "request_rate_observed": bool(observation_seconds is not None and observation_seconds > 0),
        "per_service_incoming_requests": dict(incoming),
        "per_service_outgoing_requests": dict(outgoing),
        "traffic_signal_present": total > 0,
    }
