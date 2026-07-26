from __future__ import annotations

from typing import Any

from .graph_feature_schema import (
    CANONICAL_FAULT_TYPES,
    EVIDENCE_ORDER_NAMES,
    EVIDENCE_ORDERS,
    NODE_FEATURE_NAMES,
    RCA_OPERATOR_NAMES,
)
from .graph_state_encoder import GraphState, encode_state_graph
from .prompt_operator_policy import RCAPromptPlan, render_rca_prompt_plan


class GNNRCAInstructionPolicy:
    """Graph-conditioned structured RCA prompt controller.

    This is the non-Qwen controller architecture we want for the paper path:

        redacted state -> service graph tensorizer -> GraphSAGE encoder ->
        discrete policy heads -> RCAPromptPlan -> deterministic prompt renderer

    This first version is inference-only. It builds the real GNN module and emits
    prompt plans from its policy heads plus transparent heuristic priors. The next
    patch will store log-probabilities and add a GRPO-style trainer for the same
    discrete action heads.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_focus_services: int = 6,
        seed: int = 7,
        prior_weight: float = 1.0,
        device: str | None = None,
    ):
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_focus_services = max(1, int(max_focus_services))
        self.seed = int(seed)
        self.prior_weight = float(prior_weight)
        self.device = device
        self._model = None
        self._input_dim = None
        self.last_policy_info: dict[str, Any] = {}

    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str:
        graph = encode_state_graph(compressed_state, history)
        plan, info = self._plan_from_graph(graph, sample_index=sample_index, iteration=iteration)
        self.last_policy_info = info
        return render_rca_prompt_plan(plan)

    def _plan_from_graph(self, graph: GraphState, sample_index: int, iteration: int) -> tuple[RCAPromptPlan, dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "--instruction_policy gnn requires PyTorch. Install it in the same environment "
                "that runs training, or use --instruction_policy operator until torch is available."
            ) from exc

        if graph.num_nodes == 0:
            plan = RCAPromptPlan(
                evidence_priority=EVIDENCE_ORDERS[0],
                focus_services=[],
                root_cause_count_hint=1,
                fault_type_bias=["unknown"],
                operators=["ENFORCE_OUTPUT_SCHEMA", "USE_SMALLEST_EXPLANATORY_SET"],
                retry_strategy="none" if iteration == 0 else "avoid repeating previous wrong guesses",
            )
            return plan, {"graph_summary": graph.summary(), "empty_graph": True}

        self._ensure_model(len(NODE_FEATURE_NAMES), torch)
        x, edge_index, edge_attr = graph.as_torch(self.device)

        self._model.eval()
        with torch.no_grad():
            out = self._model(x, edge_index, edge_attr)

            evidence_logits = out["evidence_logits"] + self.prior_weight * _evidence_prior(graph, torch, x.device)
            count_logits = out["count_logits"] + self.prior_weight * _count_prior(graph, torch, x.device)
            operator_logits = out["operator_logits"] + self.prior_weight * _operator_prior(graph, torch, x.device)
            fault_logits = out["fault_logits"] + self.prior_weight * _fault_prior(graph, torch, x.device)
            service_logits = out["service_logits"] + self.prior_weight * _service_prior(graph, torch, x.device)

            evidence_idx = _ranked_choice(evidence_logits, sample_index)
            count_idx = _ranked_choice(count_logits, sample_index // max(1, len(EVIDENCE_ORDERS)))
            root_count = [1, 2, 3][count_idx]

            focus_indices = _topk_indices(service_logits, min(self.max_focus_services, graph.num_nodes))
            focus_services = [graph.node_names[i] for i in focus_indices]

            operator_indices = _topk_indices(operator_logits, 5)
            operators = [RCA_OPERATOR_NAMES[i] for i in operator_indices]
            if "ENFORCE_OUTPUT_SCHEMA" not in operators:
                operators.insert(0, "ENFORCE_OUTPUT_SCHEMA")
            if root_count > 1 and "ASK_FOR_MULTIFAULT" not in operators:
                operators.append("ASK_FOR_MULTIFAULT")
            if iteration > 0 and "AVOID_REPEATED_GUESSES" not in operators:
                operators.append("AVOID_REPEATED_GUESSES")

            fault_indices = _topk_indices(fault_logits, 4)
            fault_bias = [CANONICAL_FAULT_TYPES[i] for i in fault_indices]
            if "unknown" not in fault_bias:
                fault_bias.append("unknown")

            retry_strategy = "none"
            if iteration > 0 or graph.metadata.get("previous_predicted_services"):
                retry_strategy = (
                    "avoid repeating services and fault types that received low public reward; "
                    "shift to another high-probability neighborhood if the same guess already failed"
                )

            plan = RCAPromptPlan(
                evidence_priority=EVIDENCE_ORDERS[evidence_idx],
                focus_services=focus_services,
                root_cause_count_hint=root_count,
                fault_type_bias=fault_bias[:5],
                operators=operators[:7],
                retry_strategy=retry_strategy,
            )

            info = {
                "graph_summary": graph.summary(),
                "sample_index": sample_index,
                "evidence_choice": EVIDENCE_ORDER_NAMES[evidence_idx],
                "root_cause_count_hint": root_count,
                "focus_services": focus_services,
                "operators": operators[:7],
                "fault_type_bias": fault_bias[:5],
                "uses_torch_gnn": True,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "prior_weight": self.prior_weight,
            }
            return plan, info

    def _ensure_model(self, input_dim: int, torch_module) -> None:
        if self._model is not None and self._input_dim == input_dim:
            return
        torch_module.manual_seed(self.seed)
        self._input_dim = input_dim
        self._model = _GraphSAGEPromptController(
            input_dim=input_dim,
            edge_dim=len(__import__("training_pipeline.graph_feature_schema", fromlist=["EDGE_FEATURE_NAMES"]).EDGE_FEATURE_NAMES),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_evidence=len(EVIDENCE_ORDERS),
            num_counts=3,
            num_operators=len(RCA_OPERATOR_NAMES),
            num_fault_types=len(CANONICAL_FAULT_TYPES),
        )
        if self.device:
            self._model.to(self.device)


class _GraphSAGEPromptController:
    """Small torch.nn.Module wrapper kept dependency-lazy through construction."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class Impl(nn.Module):
            def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int, num_layers: int,
                         num_evidence: int, num_counts: int, num_operators: int, num_fault_types: int):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.layers = nn.ModuleList([
                    nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(max(1, num_layers))
                ])
                self.edge_gate = nn.Linear(edge_dim, hidden_dim)
                self.service_head = nn.Linear(hidden_dim, 1)
                self.evidence_head = nn.Linear(hidden_dim, num_evidence)
                self.count_head = nn.Linear(hidden_dim, num_counts)
                self.operator_head = nn.Linear(hidden_dim, num_operators)
                self.fault_head = nn.Linear(hidden_dim, num_fault_types)
                self.activation = nn.ReLU()

            def forward(self, x, edge_index, edge_attr):
                import torch

                h = self.activation(self.input_proj(x))
                for layer in self.layers:
                    agg = torch.zeros_like(h)
                    deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                    if edge_index.numel() > 0:
                        src = edge_index[0]
                        dst = edge_index[1]
                        msg = h[src]
                        if edge_attr is not None and edge_attr.numel() > 0:
                            gate = torch.sigmoid(self.edge_gate(edge_attr))
                            msg = msg * gate
                        agg.index_add_(0, dst, msg)
                        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=h.dtype, device=h.device))
                        agg = agg / deg.clamp_min(1.0)
                    h = self.activation(layer(torch.cat([h, agg], dim=-1)))
                graph_emb = h.mean(dim=0)
                return {
                    "node_embeddings": h,
                    "graph_embedding": graph_emb,
                    "service_logits": self.service_head(h).squeeze(-1),
                    "evidence_logits": self.evidence_head(graph_emb),
                    "count_logits": self.count_head(graph_emb),
                    "operator_logits": self.operator_head(graph_emb),
                    "fault_logits": self.fault_head(graph_emb),
                }

        return Impl(*args, **kwargs)


def _feature(graph: GraphState, name: str) -> list[float]:
    idx = NODE_FEATURE_NAMES.index(name)
    return [float(row[idx]) for row in graph.node_features]


def _service_prior(graph: GraphState, torch, device):
    vals = [
        2.5 * row[NODE_FEATURE_NAMES.index("infra_issue_flag")]
        + 1.8 * row[NODE_FEATURE_NAMES.index("service_health_degraded_flag")]
        + 1.6 * row[NODE_FEATURE_NAMES.index("top_log_error_flag")]
        + 1.2 * row[NODE_FEATURE_NAMES.index("incoming_failed_trace_norm")]
        + 1.0 * row[NODE_FEATURE_NAMES.index("outgoing_failed_trace_norm")]
        + 0.8 * row[NODE_FEATURE_NAMES.index("metric_latency_high_flag")]
        - 0.6 * row[NODE_FEATURE_NAMES.index("previous_failure_flag")]
        for row in graph.node_features
    ]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _evidence_prior(graph: GraphState, torch, device):
    degraded = sum(_feature(graph, "infra_issue_flag")) + sum(_feature(graph, "service_health_degraded_flag"))
    trace = sum(_feature(graph, "incoming_failed_trace_norm")) + sum(_feature(graph, "outgoing_failed_trace_norm"))
    logs = sum(_feature(graph, "top_log_error_flag"))
    metrics = sum(_feature(graph, "metric_latency_high_flag"))
    vals = [degraded, trace, logs, metrics]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _count_prior(graph: GraphState, torch, device):
    independent = len(set(graph.metadata.get("degraded_services", [])) | set(graph.metadata.get("top_log_services", [])))
    if independent >= 4:
        vals = [0.0, 1.2, 0.6]
    elif independent >= 2:
        vals = [0.4, 0.9, 0.0]
    else:
        vals = [1.0, 0.0, -0.4]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _operator_prior(graph: GraphState, torch, device):
    vals = {name: 0.0 for name in RCA_OPERATOR_NAMES}
    vals["ENFORCE_OUTPUT_SCHEMA"] = 3.0
    vals["USE_SMALLEST_EXPLANATORY_SET"] = 1.0
    vals["AVOID_DOWNSTREAM_VICTIMS"] = 1.0
    if sum(_feature(graph, "infra_issue_flag")) or sum(_feature(graph, "pods_unready_norm")):
        vals["PRIORITIZE_SYSTEM_HEALTH"] += 1.5
        vals["FOCUS_ON_K8S_INFRA"] += 1.5
    if sum(_feature(graph, "top_log_error_flag")):
        vals["PRIORITIZE_LOG_ERRORS"] += 1.4
        vals["FOCUS_ON_DATABASE_DEPENDENCIES"] += 0.7
    if sum(_feature(graph, "incoming_failed_trace_norm")) or sum(_feature(graph, "outgoing_failed_trace_norm")):
        vals["PRIORITIZE_TRACE_EDGES"] += 1.4
        vals["FOCUS_ON_TRACE_TARGETS"] += 1.2
    if sum(_feature(graph, "metric_latency_high_flag")):
        vals["PRIORITIZE_METRICS"] += 1.2
    if len(graph.metadata.get("degraded_services", [])) >= 2 and len(graph.metadata.get("top_log_services", [])) >= 2:
        vals["ASK_FOR_MULTIFAULT"] += 1.3
    if graph.metadata.get("previous_predicted_services"):
        vals["AVOID_REPEATED_GUESSES"] += 1.5
    return torch.tensor([vals[name] for name in RCA_OPERATOR_NAMES], dtype=torch.float32, device=device)


def _fault_prior(graph: GraphState, torch, device):
    vals = {name: 0.0 for name in CANONICAL_FAULT_TYPES}
    if sum(_feature(graph, "infra_issue_flag")) or sum(_feature(graph, "pods_unready_norm")):
        vals["infra_failure"] += 1.5
        vals["resource_exhaustion"] += 0.5
    if sum(_feature(graph, "top_log_error_flag")):
        vals["dependency_failure"] += 1.0
        vals["auth_failure"] += 0.7
        vals["config_error"] += 0.5
    if sum(_feature(graph, "incoming_failed_trace_norm")) or sum(_feature(graph, "outgoing_failed_trace_norm")):
        vals["network_failure"] += 0.8
        vals["latency_degradation"] += 0.7
        vals["dependency_failure"] += 0.6
    vals["unknown"] += 0.1
    return torch.tensor([vals[name] for name in CANONICAL_FAULT_TYPES], dtype=torch.float32, device=device)


def _ranked_choice(logits, offset: int) -> int:
    order = logits.detach().cpu().argsort(descending=True).tolist()
    return int(order[offset % len(order)])


def _topk_indices(logits, k: int) -> list[int]:
    if k <= 0:
        return []
    return [int(i) for i in logits.detach().cpu().argsort(descending=True).tolist()[:k]]
