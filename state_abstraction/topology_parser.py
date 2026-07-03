from collections import defaultdict
from pathlib import Path
from utils import read_json


def parse_static_topology(run_dir, services):
    run_dir = Path(run_dir)
    obj = read_json(run_dir / "topology" / "topology.json") or read_json(run_dir / "topology" / "graph.json") or {}
    nodes = set(services)
    edges = []
    if isinstance(obj, dict):
        for s in obj.get("services", []):
            if isinstance(s, str):
                nodes.add(s)
            elif isinstance(s, dict):
                nodes.add(s.get("name") or s.get("service") or s.get("id"))
        raw_edges = obj.get("edges") or obj.get("dependencies") or []
        for e in raw_edges:
            if isinstance(e, dict):
                src = e.get("src") or e.get("source") or e.get("from")
                dst = e.get("dst") or e.get("target") or e.get("to")
                if src and dst:
                    edges.append({"src": src, "dst": dst, "features": e.get("features", {})})
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append({"src": e[0], "dst": e[1], "features": {}})
    return sorted(x for x in nodes if x), edges


def build_graph(services, metrics, logs, system, trace_edges, observed_edges, static_edges=None):
    nodes = {}
    for svc in sorted(set(services)):
        nodes[svc] = {
            "metrics": metrics.get(svc, {}),
            "logs": logs.get(svc, {}),
            "system": system.get(svc, {}),
        }
    edge_map = {}
    for src, dst in observed_edges:
        edge_map[f"{src}->{dst}"] = {"src": src, "dst": dst, "features": trace_edges.get(f"{src}->{dst}", {})}
    for e in static_edges or []:
        src, dst = e.get("src"), e.get("dst")
        if src and dst:
            edge_map.setdefault(f"{src}->{dst}", {"src": src, "dst": dst, "features": e.get("features", {})})
    for edge_id, feats in trace_edges.items():
        if "->" in edge_id:
            src, dst = edge_id.split("->", 1)
            edge_map.setdefault(edge_id, {"src": src, "dst": dst, "features": feats})
    return {"nodes": nodes, "edges": list(edge_map.values())}
