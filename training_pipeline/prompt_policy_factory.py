from __future__ import annotations

from typing import Any

from .gnn_prompt_policy import GNNRCAInstructionPolicy
from .prompt_operator_policy import OperatorRCAInstructionPolicy
from .qwen_prompt_policy import QwenRCAInstructionPolicy
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver


RCA_INSTRUCTION_POLICIES = {"heuristic", "operator", "gnn", "qwen_stub"}
RCA_SOLVERS = {"heuristic"}


def build_rca_instruction_policy(args: Any):
    name = getattr(args, "instruction_policy", "heuristic")
    if name == "heuristic":
        return HeuristicRCAInstructionPolicy()
    if name == "operator":
        return OperatorRCAInstructionPolicy(
            profile=getattr(args, "operator_profile", "auto"),
            max_focus_services=getattr(args, "operator_max_focus_services", 6),
        )
    if name == "gnn":
        return GNNRCAInstructionPolicy(
            hidden_dim=getattr(args, "gnn_hidden_dim", 64),
            num_layers=getattr(args, "gnn_num_layers", 2),
            max_focus_services=getattr(args, "operator_max_focus_services", 6),
            seed=getattr(args, "gnn_seed", 7),
            prior_weight=getattr(args, "gnn_prior_weight", 1.0),
            device=getattr(args, "gnn_device", None),
        )
    if name == "qwen_stub":
        return QwenRCAInstructionPolicy(
            model_name=getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            adapter_path=getattr(args, "qwen_adapter_path", None),
            dry_run=True,
            max_new_tokens=getattr(args, "qwen_max_new_tokens", 256),
            temperature=getattr(args, "qwen_temperature", 0.7),
            top_p=getattr(args, "qwen_top_p", 0.9),
        )
    raise ValueError(f"unknown instruction policy {name!r}; valid={sorted(RCA_INSTRUCTION_POLICIES)}")


def build_rca_solver(args: Any):
    name = getattr(args, "rca_solver", "heuristic")
    if name == "heuristic":
        return HeuristicRCASolver()
    raise ValueError(f"unknown RCA solver {name!r}; valid={sorted(RCA_SOLVERS)}")


def policy_metadata(args: Any) -> dict[str, Any]:
    return {
        "instruction_policy": getattr(args, "instruction_policy", "heuristic"),
        "operator_profile": getattr(args, "operator_profile", "auto"),
        "operator_max_focus_services": getattr(args, "operator_max_focus_services", 6),
        "gnn_hidden_dim": getattr(args, "gnn_hidden_dim", 64),
        "gnn_num_layers": getattr(args, "gnn_num_layers", 2),
        "gnn_seed": getattr(args, "gnn_seed", 7),
        "gnn_prior_weight": getattr(args, "gnn_prior_weight", 1.0),
        "gnn_device": getattr(args, "gnn_device", None),
        "qwen_model": getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "qwen_adapter_path": getattr(args, "qwen_adapter_path", None),
        "qwen_dry_run": getattr(args, "instruction_policy", "heuristic") == "qwen_stub",
        "rca_solver": getattr(args, "rca_solver", "heuristic"),
    }


def default_policy_model_name(args: Any) -> str:
    name = getattr(args, "instruction_policy", "heuristic")
    if name == "qwen_stub":
        return getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct") + ":dry-run"
    if name == "operator":
        return "operator-controller:" + getattr(args, "operator_profile", "auto")
    if name == "gnn":
        return (
            "gnn-operator-controller:"
            f"h{getattr(args, 'gnn_hidden_dim', 64)}:"
            f"l{getattr(args, 'gnn_num_layers', 2)}:"
            f"prior{getattr(args, 'gnn_prior_weight', 1.0)}"
        )
    return "debug-heuristic-policy"
