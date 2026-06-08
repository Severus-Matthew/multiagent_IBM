import copy
from collections import defaultdict
from utils import read_json, write_json, read_jsonl, ensure_dir
from sla import evaluate_sla

class AbstractBehavioralSimulator:
    def __init__(self, graph):
        self.graph = graph
        self.services = sorted(graph.get("nodes", {}).keys())
        self.edges = []
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for e in graph.get("edges", []):
            src, dst = e.get("src"), e.get("dst")
            if not src or not dst:
                continue
            eid = f"{src}->{dst}"
            self.edges.append(eid)
            self.out_edges[src].append(dst)
            self.in_edges[dst].append(src)

    def init_latent_from_state(self, state):
        latent = {"services": {}, "edges": {}}
        for svc in self.services:
            sy = state.get("system", {}).get(svc, {})
            lg = state.get("logs", {}).get(svc, {})
            mt = state.get("metrics", {}).get(svc, {})
            latent["services"][svc] = {"latent_mode": "healthy", "fault_type": None, "app_health": max(0.0, 1.0 - lg.get("log_anomaly_score", 0.0)), "infra_health": 0.3 if sy.get("infra_issue_flag") else 1.0, "latency_ms": mt.get("latency", 0.0)}
        for eid in self.edges:
            tf = state.get("traces", {}).get(eid, {})
            latent["edges"][eid] = {"latent_mode": "healthy", "error_ratio": tf.get("error_ratio", 0.0), "latency_p95_us": tf.get("latency_p95_us", 0.0)}
        return latent

    def inject_rca_fault(self, latent, service, fault_type):
        latent = copy.deepcopy(latent)
        if service not in latent["services"]:
            return latent
        s = latent["services"][service]
        s["fault_type"] = fault_type
        if fault_type in ["dependency_failure", "ghost_failure", "config_error", "auth_failure"]:
            s["latent_mode"] = "failed"
            s["app_health"] = 0.1
            s["infra_health"] = 1.0
        elif fault_type == "latency_degradation":
            s["latent_mode"] = "degraded"
            s["latency_ms"] = max(1000.0, s["latency_ms"] * 8)
            s["app_health"] = 0.5
        elif fault_type == "infra_failure":
            s["latent_mode"] = "failed"
            s["infra_health"] = 0.0
            s["app_health"] = 0.2
        return latent

    def propagate(self, latent):
        latent = copy.deepcopy(latent)
        for svc, s in latent["services"].items():
            mode = s["latent_mode"]
            if mode == "healthy":
                continue
            for dst in self.out_edges.get(svc, []):
                eid = f"{svc}->{dst}"
                e = latent["edges"].setdefault(eid, {"latent_mode": "healthy", "error_ratio": 0.0, "latency_p95_us": 0.0})
                if mode == "failed":
                    e["latent_mode"] = "failing"
                    e["error_ratio"] = max(e["error_ratio"], 0.9)
                    e["latency_p95_us"] = max(e["latency_p95_us"], 200000.0)
                elif mode == "degraded":
                    e["latent_mode"] = "degraded"
                    e["error_ratio"] = max(e["error_ratio"], 0.25)
                    e["latency_p95_us"] = max(e["latency_p95_us"], s["latency_ms"] * 1000)
        return latent

    def apply_action(self, latent, action):
        latent = copy.deepcopy(latent)
        if not action:
            return latent
        act = action.get("action")
        svc = action.get("service")
        if act in ["restart_service", "rollback_config"] and svc in latent["services"]:
            latent["services"][svc].update({"latent_mode": "healthy", "fault_type": None, "app_health": 1.0})
            for dst in self.out_edges.get(svc, []):
                eid = f"{svc}->{dst}"
                if eid in latent["edges"]:
                    latent["edges"][eid].update({"latent_mode": "healthy", "error_ratio": 0.0})
        elif act == "scale_service" and svc in latent["services"]:
            latent["services"][svc]["latency_ms"] *= 0.7
        elif act == "disable_edge":
            eid = action.get("edge") or f"{action.get('src')}->{action.get('dst')}"
            if eid in latent["edges"]:
                latent["edges"][eid].update({"latent_mode": "healthy", "error_ratio": 0.0, "latency_p95_us": 0.0})
        return latent

    def latent_to_observation(self, latent, base_state):
        obs = copy.deepcopy(base_state)
        for svc, s in latent["services"].items():
            obs.setdefault("metrics", {}).setdefault(svc, {})["latency"] = s["latency_ms"]
            obs.setdefault("logs", {}).setdefault(svc, {})
            obs.setdefault("system", {}).setdefault(svc, {})
            if s["latent_mode"] == "healthy":
                obs["logs"][svc].update({"dominant_error_type": "none", "log_anomaly_score": 0.0, "log_health_score": 1.0})
                obs["system"][svc]["infra_issue_flag"] = False
            else:
                det = "auth" if s["fault_type"] == "auth_failure" else "config" if s["fault_type"] == "config_error" else "timeout" if s["fault_type"] == "latency_degradation" else "error"
                obs["logs"][svc].update({"dominant_error_type": det, "log_anomaly_score": 0.7, "log_health_score": 0.3, "evidence_lines": [f"synthetic {s['fault_type']} on {svc}"]})
                obs["system"][svc]["infra_issue_flag"] = s["fault_type"] == "infra_failure"
        obs["traces"] = obs.get("traces", {})
        for eid, e in latent["edges"].items():
            obs["traces"].setdefault(eid, {})
            obs["traces"][eid].update({"error_ratio": e["error_ratio"], "latency_p95_us": e["latency_p95_us"], "is_suspicious": e["error_ratio"] > 0.2 or e["latency_p95_us"] > 100000})
            obs["traces"][eid]["failure_type"] = "hard_error_path" if e["error_ratio"] >= 0.9 else "partial_error_path" if e["error_ratio"] >= 0.3 else "high_latency_path" if e["latency_p95_us"] >= 500000 else "healthy_path"
            obs["traces"][eid]["edge_rank_score"] = min(1.0, e["error_ratio"] * 0.8 + min(e["latency_p95_us"] / 1_000_000, 1.0) * 0.2)
        obs["sla"] = evaluate_sla(obs)
        return obs

    def simulate(self, base_state, rca_fault=None, mitigation_action=None):
        latent = self.init_latent_from_state(base_state)
        if rca_fault:
            latent = self.inject_rca_fault(latent, rca_fault.get("service"), rca_fault.get("fault_type", "dependency_failure"))
        latent = self.propagate(latent)
        if mitigation_action:
            latent = self.apply_action(latent, mitigation_action)
            latent = self.propagate(latent)
        obs = self.latent_to_observation(latent, base_state)
        return obs, obs["sla"]
