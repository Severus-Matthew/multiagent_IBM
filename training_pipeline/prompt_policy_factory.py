from __future__ import annotations

from typing import Any

from .rca_candidate_generator_v3 import EvidenceFirstLLMRCASolver
from .prompt_operator_policy import OperatorRCAInstructionPolicy
from .qwen_prompt_policy import QwenRCAInstructionPolicy
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver


RCA_INSTRUCTION_POLICIES = {"heuristic", "operator", "qwen_stub"}
RCA_SOLVERS = {"heuristic", "llm"}


def build_rca_instruction_policy(args: Any):
    name = getattr(args, "instruction_policy", "heuristic")
    if name == "heuristic":
        return HeuristicRCAInstructionPolicy()
    if name == "operator":
        return OperatorRCAInstructionPolicy(
            profile=getattr(args, "operator_profile", "auto"),
            max_focus_services=getattr(args, "operator_max_focus_services", 6),
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
    if name == "llm":
        return EvidenceFirstLLMRCASolver(
            provider=getattr(args, "llm_provider", "openai"),
            model=getattr(args, "llm_model", None),
            max_tokens=getattr(args, "llm_max_tokens", 300),
            temperature=getattr(args, "llm_temperature", 0.0),
            state_char_budget=getattr(args, "llm_state_char_budget", 24000),
            cache_path=getattr(args, "llm_cache_path", None),
            max_root_causes=getattr(args, "llm_max_root_causes", 1),
        )
    raise ValueError(f"unknown RCA solver {name!r}; valid={sorted(RCA_SOLVERS)}")


def policy_metadata(args: Any) -> dict[str, Any]:
    return {
        "instruction_policy": getattr(args, "instruction_policy", "heuristic"),
        "operator_profile": getattr(args, "operator_profile", "auto"),
        "operator_max_focus_services": getattr(args, "operator_max_focus_services", 6),
        "qwen_model": getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "qwen_adapter_path": getattr(args, "qwen_adapter_path", None),
        "qwen_dry_run": getattr(args, "instruction_policy", "heuristic") == "qwen_stub",
        "rca_solver": getattr(args, "rca_solver", "heuristic"),
        "llm_provider": getattr(args, "llm_provider", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_model": getattr(args, "llm_model", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_temperature": getattr(args, "llm_temperature", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_max_tokens": getattr(args, "llm_max_tokens", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_state_char_budget": getattr(args, "llm_state_char_budget", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_max_root_causes": getattr(args, "llm_max_root_causes", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_cache_path": getattr(args, "llm_cache_path", None) if getattr(args, "rca_solver", "heuristic") == "llm" else None,
        "llm_solver_impl": "EvidenceFirstLLMRCASolver:v3" if getattr(args, "rca_solver", "heuristic") == "llm" else None,
    }


def default_policy_model_name(args: Any) -> str:
    name = getattr(args, "instruction_policy", "heuristic")
    if name == "qwen_stub":
        return getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct") + ":dry-run"
    if name == "operator":
        return "operator-controller:" + getattr(args, "operator_profile", "auto")
    return "debug-heuristic-policy"
