from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_feature_schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES


@dataclass
class GraphState:
    """Tensor-ready graph representation of one redacted AIOps state.

    This file intentionally does not require torch at import time. The lists are
    plain Python values so state encoding can be tested on CPU-only machines and
    inside py_compile. GNN policies can call as_torch() when torch is available.
    """

    node_names: list[str]
    node_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_features: list[list[float]]
    metadata: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index)

    def as_torch(self, device: str | None = None):
        """Return (x, edge_index, edge_attr) tensors. Imports torch lazily."""
        import torch

        x = torch.tensor(self.node_features, dtype=torch.float32, device=device)
        if self.edge_index:
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=device).t().contiguous()
            edge_attr = torch.tensor(self.edge_features, dtype=torch.float32, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32, device=device)
        return x, edge_index, edge_attr

    def summary(self) -> dict[str, Any]:
        nonzero = 0
        total = 0
        for row in self.node_features:
            for val in row:
                total += 1
                nonzero += int(abs(float(val)) > 1e-12)
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_feature_dim": len(NODE_FEATURE_NAMES),
            "edge_feature_dim": len(EDGE_FEATURE_NAMES),
            "node_feature_nonzero_rate": nonzero / max(total, 1),
            "metadata": self.metadata,
        }


def encode_state_graph(compressed_state: dict[str, Any], history: list[dict[str, Any]] | None = None) -> GraphState:
    """Encode redacted telemetry as a service graph for the GNN controller.

    No ground truth fields are read here. Service names come from the compressed
    state, graph/topology state, system records, logs, traces, and clusters.
    """
    history = history or []
    services = _discover_services(compressed_state)
    node_to_idx = {svc: i for i, svc in enumerate(services)}

    trace_stats = _trace_stats(compressed_state)
    top_log = _top_log_services(compressed_state)
    clusters = compressed_state.get("clusters", {}) or {}
    previous = _history_stats(history)

    node_features: list[list[float]] = []
    for svc in services:
        system_info = ((compressed_state.get("system", {}) or {}).get(svc, {}) or {})
        health = system_info.get("health", system_info) if isinstance(system_info, dict) else {}
        service_health = ((compressed_state.get("service_health", {}) or {}).get(svc, {}) or {})
        metrics = ((compressed_state.get("metrics", {}) or {}).get(svc, {}) or {})
        flat_metrics = metrics.get("flat_summary", metrics) if isinstance(metrics, dict) else {}
        status = str(health.get("status", service_health.get("status", ""))).lower()
        service_status = str(service_health.get("status", "")).lower()
        latency_ms = _safe_float(flat_metrics.get("latency_ms") or flat_metrics.get("latency"))
        restarts = _safe_float(health.get("restart_count") or health.get("restarts") or health.get("restart_count_total"))
        crashloops = _safe_float(health.get("crashloop_count"))
        pods_unready = _safe_float(health.get("pods_unready") or health.get("unready_pods"))
        incoming = trace_stats["incoming"].get(svc, {})
        outgoing = trace_stats["outgoing"].get(svc, {})

        node_features.append([
            1.0,
            1.0 if svc in (compressed_state.get("system", {}) or {}) else 0.0,
            1.0 if health.get("infra_issue_flag") else 0.0,
            _cap01(pods_unready / 5.0),
            _cap01(restarts / 10.0),
            _cap01(crashloops / 5.0),
            1.0 if ("no_ready" in status or "endpoint" in status) else 0.0,
            1.0 if service_status not in {"", "healthy", "unknown"} else 0.0,
            1.0 if svc in top_log else 0.0,
            top_log.get(svc, 0.0),
            _cap01(_safe_float(incoming.get("failed_count")) / 5.0),
            _cap01(_safe_float(outgoing.get("failed_count")) / 5.0),
            _cap01(_safe_float(incoming.get("max_error_ratio"))),
            _cap01(_safe_float(outgoing.get("max_error_ratio"))),
            1.0 if latency_ms > 500.0 else 0.0,
            _cap01(latency_ms / 5000.0),
            1.0 if svc in set(clusters.get("infra_unhealthy", []) or []) else 0.0,
            1.0 if svc in set(clusters.get("log_error_or_dependency_failure", []) or []) else 0.0,
            1.0 if svc in previous["predicted_services"] else 0.0,
            1.0 if svc in previous["failed_services"] else 0.0,
            1.0 if svc in previous["successful_services"] else 0.0,
            _cap_signed(previous["mean_reward_by_service"].get(svc, 0.0)),
        ])

    edge_map: dict[tuple[int, int], list[float]] = {}
    for src, dst in _graph_edges(compressed_state):
        if src in node_to_idx and dst in node_to_idx and src != dst:
            _merge_edge(edge_map, (node_to_idx[src], node_to_idx[dst]), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    for src, dst, error_ratio, latency_ms in _trace_edges(compressed_state):
        if src in node_to_idx and dst in node_to_idx and src != dst:
            failed = 1.0 if error_ratio > 0.2 else 0.0
            _merge_edge(
                edge_map,
                (node_to_idx[src], node_to_idx[dst]),
                [1.0, 0.0, 1.0, failed, _cap01(error_ratio), _cap01(latency_ms / 5000.0)],
            )

    # Ensure every nontrivial graph has at least light self-context through a chain fallback.
    if not edge_map and len(services) > 1:
        for i in range(len(services) - 1):
            edge_map[(i, i + 1)] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    edge_index = list(edge_map.keys())
    edge_features = [edge_map[e] for e in edge_index]

    metadata = {
        "services": services,
        "feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "top_log_services": sorted(top_log, key=top_log.get, reverse=True),
        "degraded_services": [svc for svc, row in zip(services, node_features) if row[2] or row[7]],
        "previous_predicted_services": sorted(previous["predicted_services"]),
    }
    return GraphState(services, node_features, edge_index, edge_features, metadata)


def _discover_services(state: dict[str, Any]) -> list[str]:
    services: set[str] = set(str(s) for s in (state.get("services", []) or []) if s)
    services.update(str(s) for s in (state.get("system", {}) or {}).keys())
    services.update(str(s) for s in (state.get("service_health", {}) or {}).keys())
    services.update(str(s) for s in (state.get("metrics", {}) or {}).keys())
    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service"):
            services.add(str(item["service"]))
    for src, dst in _graph_edges(state):
        services.add(src); services.add(dst)
    for src, dst, _, _ in _trace_edges(state):
        services.add(src); services.add(dst)
    for bucket in (state.get("clusters", {}) or {}).values():
        if isinstance(bucket, list):
            services.update(str(x) for x in bucket if x)
    return sorted(s for s in services if s and s != "unknown")


def _top_log_services(state: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    rows = (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []
    for rank, item in enumerate(rows):
        if isinstance(item, dict) and item.get("service"):
            out[str(item["service"])] = max(out.get(str(item["service"]), 0.0), 1.0 / float(rank + 1))
    return out


def _graph_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    graph = state.get("graph", {}) or {}
    raw_edges = graph.get("edges", []) if isinstance(graph, dict) else []
    edges: list[tuple[str, str]] = []
    for e in raw_edges or []:
        src = dst = None
        if isinstance(e, str) and "->" in e:
            src, dst = e.split("->", 1)
        elif isinstance(e, dict):
            src = e.get("source") or e.get("src") or e.get("from") or e.get("from_service")
            dst = e.get("target") or e.get("dst") or e.get("to") or e.get("to_service") or e.get("to_pod_or_ip")
            if isinstance(dst, str) and "." in dst:
                dst = dst.split(".", 1)[0]
        if src and dst:
            edges.append((str(src), str(dst)))
    return edges


def _trace_edges(state: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    edges: list[tuple[str, str, float, float]] = []
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        src = feats.get("source")
        dst = feats.get("target")
        edge_s = str(edge)
        if (not src or not dst) and "->" in edge_s:
            src, dst = edge_s.split("->", 1)
        if src and dst:
            error_ratio = _safe_float(feats.get("error_ratio"))
            latency_ms = _safe_float(feats.get("latency_p95_ms") or feats.get("latency_ms") or _safe_float(feats.get("latency_p95_us")) / 1000.0)
            edges.append((str(src), str(dst), error_ratio, latency_ms))
    return edges


def _trace_stats(state: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    incoming: dict[str, dict[str, float]] = {}
    outgoing: dict[str, dict[str, float]] = {}
    for src, dst, error_ratio, _ in _trace_edges(state):
        for table, svc in [(incoming, dst), (outgoing, src)]:
            row = table.setdefault(svc, {"failed_count": 0.0, "max_error_ratio": 0.0})
            if error_ratio > 0.2:
                row["failed_count"] += 1.0
            row["max_error_ratio"] = max(row["max_error_ratio"], error_ratio)
    return {"incoming": incoming, "outgoing": outgoing}


def _history_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_services: set[str] = set()
    failed_services: set[str] = set()
    successful_services: set[str] = set()
    rewards: dict[str, list[float]] = {}
    for h in history:
        reward = _safe_float(h.get("reward"))
        success = bool(h.get("success"))
        for item in h.get("parsed_prediction", []) or []:
            if not isinstance(item, dict):
                continue
            svc = item.get("service") or item.get("root_cause_service")
            if not svc:
                continue
            svc = str(svc)
            predicted_services.add(svc)
            rewards.setdefault(svc, []).append(reward)
            if success:
                successful_services.add(svc)
            else:
                failed_services.add(svc)
    return {
        "predicted_services": predicted_services,
        "failed_services": failed_services,
        "successful_services": successful_services,
        "mean_reward_by_service": {svc: sum(vals) / max(len(vals), 1) for svc, vals in rewards.items()},
    }


def _merge_edge(edge_map: dict[tuple[int, int], list[float]], key: tuple[int, int], feat: list[float]) -> None:
    if key not in edge_map:
        edge_map[key] = feat
        return
    cur = edge_map[key]
    edge_map[key] = [max(a, b) for a, b in zip(cur, feat)]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _cap01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _cap_signed(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))
