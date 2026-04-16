import copy
from collections import defaultdict

from twin_utils import read_json
from twin_runtime_config import GRAPH_JSON
from SLA import evaluate_sla


class MinimalDigitalTwin:
    """
    Low-cost abstract-state twin.

    Latent state:
      - per-service latent condition
      - per-edge latent failure

    Observable state:
      - same schema style as states.jsonl
    """

    def __init__(self, graph_path=GRAPH_JSON):
        graph = read_json(graph_path)

        self.graph = graph
        self.services = sorted(graph["nodes"].keys())

        self.edges = []
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

        for e in graph["edges"]:
            src = e["src"]
            dst = e["dst"]
            edge_id = f"{src}->{dst}"
            self.edges.append(edge_id)
            self.out_edges[src].append(dst)
            self.in_edges[dst].append(src)

    # -----------------------------------
    # Build latent state from a real state
    # -----------------------------------
    def init_latent_from_state(self, state):
        latent = {
            "services": {},
            "edges": {},
        }

        for svc in self.services:
            metrics = state.get("metrics", {}).get(svc, {})
            logs = state.get("logs", {}).get(svc, {})
            system = state.get("system", {}).get(svc, {})

            infra_health = 1.0
            if system.get("infra_issue_flag", False):
                infra_health = 0.3

            app_health = 1.0
            if logs.get("log_anomaly_score", 0.0) > 0.4:
                app_health = 0.5

            latent["services"][svc] = {
                "latent_mode": "healthy",       # healthy/degraded/failed
                "fault_type": None,
                "app_health": app_health,
                "infra_health": infra_health,
                "resource_health": 1.0,
                "log_health": logs.get("log_health_score", 1.0),
                "latency_ms": metrics.get("latency", 0.0),
            }

        for edge_id in self.edges:
            traces = state.get("traces", {})
            edge_feats = traces.get(edge_id, {})
            latent["edges"][edge_id] = {
                "latent_mode": "healthy",       # healthy/degraded/failing
                "error_ratio": edge_feats.get("error_ratio", 0.0),
                "latency_p95_us": edge_feats.get("latency_p95_us", 0.0),
            }

        return latent

    # -----------------------------------
    # Inject RCA hypothesis as latent fault
    # -----------------------------------
    def inject_rca_fault(self, latent, service, fault_type):
        latent = copy.deepcopy(latent)

        if service not in latent["services"]:
            return latent

        s = latent["services"][service]
        s["fault_type"] = fault_type

        if fault_type in ["dependency_failure", "ghost_failure", "config_error"]:
            s["latent_mode"] = "failed"
            s["app_health"] = 0.1
            s["infra_health"] = 1.0

        elif fault_type == "latency_degradation":
            s["latent_mode"] = "degraded"
            s["latency_ms"] = max(100.0, s["latency_ms"] * 8.0)
            s["app_health"] = 0.5

        elif fault_type == "infra_failure":
            s["latent_mode"] = "failed"
            s["infra_health"] = 0.0
            s["app_health"] = 0.2

        return latent

    # -----------------------------------
    # Propagate fault to edges and neighbors
    # -----------------------------------
    def propagate(self, latent):
        latent = copy.deepcopy(latent)

        for svc, s in latent["services"].items():
            mode = s["latent_mode"]
            fault_type = s["fault_type"]

            if mode == "healthy":
                continue

            # outgoing effects
            for dst in self.out_edges.get(svc, []):
                edge_id = f"{svc}->{dst}"
                edge = latent["edges"][edge_id]

                if mode == "failed":
                    edge["latent_mode"] = "failing"

                    if fault_type in ["dependency_failure", "ghost_failure", "config_error"]:
                        edge["error_ratio"] = 1.0
                        edge["latency_p95_us"] = max(edge["latency_p95_us"], 100000.0)

                    elif fault_type == "infra_failure":
                        edge["error_ratio"] = max(edge["error_ratio"], 0.8)
                        edge["latency_p95_us"] = max(edge["latency_p95_us"], 200000.0)

                elif mode == "degraded":
                    edge["latent_mode"] = "degraded"
                    edge["error_ratio"] = max(edge["error_ratio"], 0.25)
                    edge["latency_p95_us"] = max(edge["latency_p95_us"], s["latency_ms"] * 1000.0)

            # self loop is powerful RCA signal
            self_edge = f"{svc}->{svc}"
            if self_edge in latent["edges"]:
                if mode == "failed":
                    latent["edges"][self_edge]["latent_mode"] = "failing"
                    latent["edges"][self_edge]["error_ratio"] = 1.0
                    latent["edges"][self_edge]["latency_p95_us"] = max(
                        latent["edges"][self_edge]["latency_p95_us"], 50000.0
                    )

        return latent

    # -----------------------------------
    # Apply mitigation
    # -----------------------------------
    def apply_action(self, latent, action):
        latent = copy.deepcopy(latent)
        if not action:
            return latent

        act = action.get("action")

        if act == "restart_service":
            svc = action["service"]
            if svc in latent["services"]:
                # helps transient app/config/ghost issues more than infra
                latent["services"][svc]["latent_mode"] = "healthy"
                latent["services"][svc]["app_health"] = 1.0
                latent["services"][svc]["fault_type"] = None

                for dst in self.out_edges.get(svc, []):
                    edge_id = f"{svc}->{dst}"
                    latent["edges"][edge_id]["latent_mode"] = "healthy"
                    latent["edges"][edge_id]["error_ratio"] = 0.0

                self_edge = f"{svc}->{svc}"
                if self_edge in latent["edges"]:
                    latent["edges"][self_edge]["latent_mode"] = "healthy"
                    latent["edges"][self_edge]["error_ratio"] = 0.0

        elif act == "scale_service":
            svc = action["service"]
            if svc in latent["services"]:
                # helps degradation but not pure logic failure much
                latent["services"][svc]["resource_health"] = min(
                    1.0, latent["services"][svc]["resource_health"] + 0.3
                )
                if latent["services"][svc]["latent_mode"] == "degraded":
                    latent["services"][svc]["latency_ms"] *= 0.7

        elif act == "rollback_config":
            svc = action["service"]
            if svc in latent["services"]:
                latent["services"][svc]["latent_mode"] = "healthy"
                latent["services"][svc]["fault_type"] = None
                latent["services"][svc]["app_health"] = 1.0
                for dst in self.out_edges.get(svc, []):
                    edge_id = f"{svc}->{dst}"
                    latent["edges"][edge_id]["error_ratio"] = 0.0
                    latent["edges"][edge_id]["latent_mode"] = "healthy"

        elif act == "disable_edge":
            edge_id = action["edge"]
            if edge_id in latent["edges"]:
                latent["edges"][edge_id]["latent_mode"] = "healthy"
                latent["edges"][edge_id]["error_ratio"] = 0.0
                latent["edges"][edge_id]["latency_p95_us"] = 0.0

        elif act == "wait":
            pass

        return latent

    # -----------------------------------
    # Convert latent state -> observable abstracted state
    # -----------------------------------
    def latent_to_observation(self, latent, base_state):
        obs = copy.deepcopy(base_state)

        # ---- service-level observables ----
        for svc, s in latent["services"].items():
            mode = s["latent_mode"]
            fault_type = s["fault_type"]

            # metrics
            obs["metrics"].setdefault(svc, {})
            obs["metrics"][svc]["latency"] = s["latency_ms"]

            # logs
            obs["logs"].setdefault(svc, {})
            if mode == "healthy":
                obs["logs"][svc]["dominant_error_type"] = "none"
                obs["logs"][svc]["log_anomaly_score"] = 0.0
                obs["logs"][svc]["log_health_score"] = 1.0
                obs["logs"][svc]["log_signal_present"] = False
                obs["logs"][svc]["top_error_templates"] = []
                obs["logs"][svc]["evidence_lines"] = []
            else:
                # synthetic but meaningful abstraction
                if fault_type == "config_error":
                    det = "config"
                    template = "invalid configuration or bad upstream endpoint"
                elif fault_type == "latency_degradation":
                    det = "timeout"
                    template = "request latency exceeded expected threshold"
                elif fault_type == "infra_failure":
                    det = "connection"
                    template = "service unavailable due to infrastructure issue"
                else:
                    det = "error"
                    template = "synthetic downstream dependency failure"

                obs["logs"][svc]["dominant_error_type"] = det
                obs["logs"][svc]["log_anomaly_score"] = 0.7 if mode == "failed" else 0.4
                obs["logs"][svc]["log_health_score"] = 1.0 - obs["logs"][svc]["log_anomaly_score"]
                obs["logs"][svc]["log_signal_present"] = True
                obs["logs"][svc]["top_error_templates"] = [template]
                obs["logs"][svc]["evidence_lines"] = [template]

            # system
            obs["system"].setdefault(svc, {})
            if fault_type == "infra_failure":
                obs["system"][svc]["pods_unready"] = 1
                obs["system"][svc]["pods_ready"] = 0
                obs["system"][svc]["infra_issue_flag"] = True
                obs["system"][svc]["service_health_status"] = "unready"
            else:
                obs["system"][svc]["infra_issue_flag"] = False
                if obs["system"][svc].get("pods_ready", 0) == 0:
                    obs["system"][svc]["pods_ready"] = 1
                obs["system"][svc]["service_health_status"] = "healthy"

        # ---- edge-level observables ----
        obs["traces"] = obs.get("traces", {})
        for edge_id, e in latent["edges"].items():
            obs["traces"].setdefault(edge_id, {})
            obs["traces"][edge_id]["error_ratio"] = e["error_ratio"]
            obs["traces"][edge_id]["latency_p95_us"] = e["latency_p95_us"]

            if e["error_ratio"] >= 0.9:
                failure_type = "hard_error_path"
            elif e["error_ratio"] >= 0.3:
                failure_type = "partial_error_path"
            elif e["latency_p95_us"] >= 500000:
                failure_type = "high_latency_path"
            elif e["latency_p95_us"] >= 100000:
                failure_type = "degraded_latency_path"
            else:
                failure_type = "healthy_path"

            obs["traces"][edge_id]["failure_type"] = failure_type
            obs["traces"][edge_id]["is_suspicious"] = (
                e["error_ratio"] > 0.2 or e["latency_p95_us"] > 100000
            )
            obs["traces"][edge_id]["edge_rank_score"] = min(
                1.0, e["error_ratio"] * 0.8 + min(e["latency_p95_us"] / 1_000_000.0, 1.0) * 0.2
            )

        return obs

    # -----------------------------------
    # Simulate one scenario
    # -----------------------------------
    def simulate(self, base_state, rca_fault=None, mitigation_action=None):
        latent = self.init_latent_from_state(base_state)

        if rca_fault:
            latent = self.inject_rca_fault(
                latent,
                service=rca_fault["service"],
                fault_type=rca_fault["fault_type"],
            )

        latent = self.propagate(latent)

        if mitigation_action:
            latent = self.apply_action(latent, mitigation_action)
            latent = self.propagate(latent)

        simulated_state = self.latent_to_observation(latent, base_state)
        sla_result = evaluate_sla(simulated_state)

        return simulated_state, sla_result