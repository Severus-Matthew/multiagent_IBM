from config import SERVICES


def build_graph(service_metrics, service_logs, service_system, trace_edges, observed_edges):
    nodes = {}
    for svc in SERVICES:
        nodes[svc] = {
            "metrics": service_metrics.get(svc, {}),
            "logs": service_logs.get(svc, {}),
            "system": service_system.get(svc, {}),
        }

    edges = []
    for src, dst in observed_edges:
        key = f"{src}->{dst}"
        edges.append({
            "src": src,
            "dst": dst,
            "features": trace_edges.get(key, {}),
        })

    return {
        "nodes": nodes,
        "edges": edges,
    }