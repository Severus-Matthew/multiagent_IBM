from __future__ import annotations

from typing import Any

from .rca_candidate_generator_v5 import LLMSelectorRCASolverV5
from .rca_candidate_sweep_solver import CandidateSweepRCASolver
from .training_safe_llm_rca_solver import TrainingSafeLLMRCASolver
from .prompt_operator_policy import OperatorRCAInstructionPolicy
from .qwen_prompt_policy import QwenRCAInstructionPolicy
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver


RCA_INSTRUCTION_POLICIES = {"heuristic", "operator", "qwen_stub"}
RCA_SOLVERS = {"heuristic", "llm", "safe_llm"}


def build_rca_instruction_policy(args: Any):
    name = getattr(args, "instruction_policy", "heuristic")
    safe_mode = getattr(args, "agent_input_mode", "legacy") == "training_safe"
    if name == "heuristic":
        return HeuristicRCAInstructionPolicy()
    if name == "operator":
        return OperatorRCAInstructionPolicy(
            profile=getattr(args, "operator_profile", "auto"),
            max_focus_services=getattr(args, "operator_max_focus_services", 6),
            safe_mode=safe_mode,
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


def _use_candidate_sweep(args: Any) -> bool:
    return getattr(args, "rca_solver", "heuristic") == "llm" and str(getattr(args, "llm_model", "") or "").lower() in {
        "candidate_sweep",
        "candidate-sweep",
        "sweep",
    }


def _use_training_safe_llm(args: Any) -> bool:
    return getattr(args, "rca_solver", "heuristic") == "safe_llm" or str(getattr(args, "llm_model", "") or "").lower() in {
        "training_safe",
        "training-safe",
        "safe_llm",
        "safe-llm",
    }


def build_rca_solver(args: Any):
    name = getattr(args, "rca_solver", "heuristic")
    if name == "heuristic":
        return HeuristicRCASolver()
    if name in {"llm", "safe_llm"}:
        if _use_candidate_sweep(args):
            return CandidateSweepRCASolver(
                max_root_causes=getattr(args, "llm_max_root_causes", 1),
            )
        if _use_training_safe_llm(args):
            return TrainingSafeLLMRCASolver(
                provider=getattr(args, "llm_provider", "openai"),
                model=None if str(getattr(args, "llm_model", "") or "").lower() in {"training_safe", "training-safe", "safe_llm", "safe-llm"} else getattr(args, "llm_model", None),
                max_tokens=getattr(args, "llm_max_tokens", 300),
                temperature=getattr(args, "llm_temperature", 0.7),
                state_char_budget=getattr(args, "llm_state_char_budget", 24000),
                cache_path=getattr(args, "llm_cache_path", None),
                max_root_causes=getattr(args, "llm_max_root_causes", 1),
            )
        return LLMSelectorRCASolverV5(
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
    use_sweep = _use_candidate_sweep(args)
    use_safe = _use_training_safe_llm(args)
    rca_solver = getattr(args, "rca_solver", "heuristic")
    return {
        "instruction_policy": getattr(args, "instruction_policy", "heuristic"),
        "operator_profile": getattr(args, "operator_profile", "auto"),
        "operator_max_focus_services": getattr(args, "operator_max_focus_services", 6),
        "agent_input_mode": getattr(args, "agent_input_mode", "legacy"),
        "qwen_model": getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "qwen_adapter_path": getattr(args, "qwen_adapter_path", None),
        "qwen_dry_run": getattr(args, "instruction_policy", "heuristic") == "qwen_stub",
        "rca_solver": rca_solver,
        "llm_provider": getattr(args, "llm_provider", None) if rca_solver in {"llm", "safe_llm"} and not use_sweep else None,
        "llm_model": getattr(args, "llm_model", None) if rca_solver in {"llm", "safe_llm"} else None,
        "llm_temperature": getattr(args, "llm_temperature", None) if rca_solver in {"llm", "safe_llm"} and not use_sweep else None,
        "llm_max_tokens": getattr(args, "llm_max_tokens", None) if rca_solver in {"llm", "safe_llm"} and not use_sweep else None,
        "llm_state_char_budget": getattr(args, "llm_state_char_budget", None) if rca_solver in {"llm", "safe_llm"} and not use_sweep else None,
        "llm_max_root_causes": getattr(args, "llm_max_root_causes", None) if rca_solver in {"llm", "safe_llm"} else None,
        "llm_cache_path": getattr(args, "llm_cache_path", None) if rca_solver in {"llm", "safe_llm"} and not use_sweep else None,
        "llm_solver_impl": (
            "CandidateSweepRCASolver:v6:mixed-root-sweep:audit-only" if use_sweep else
            "TrainingSafeLLMRCASolver:v1:no-candidate-menu" if use_safe else
            ("LLMSelectorRCASolverV5:valid-candidate-selector" if rca_solver == "llm" else None)
        ),
        "candidate_sweep_audit_only": bool(use_sweep),
        "training_safe_no_candidate_menu": bool(use_safe),
    }


def default_policy_model_name(args: Any) -> str:
    name = getattr(args, "instruction_policy", "heuristic")
    if name == "qwen_stub":
        return getattr(args, "qwen_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct") + ":dry-run"
    if name == "operator":
        suffix = ":candidate-sweep-audit" if _use_candidate_sweep(args) else (":training-safe" if _use_training_safe_llm(args) else "")
        return "operator-controller:" + getattr(args, "operator_profile", "auto") + suffix
    return "debug-heuristic-policy"
