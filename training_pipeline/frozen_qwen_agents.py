from __future__ import annotations

"""Instruction-sensitive frozen Qwen downstream agents.

The trainable RCA/Action policies optimize *instructions*.  Scientific credit is
only meaningful if those instructions can change the downstream RCA/remediation
outcome.  The old smoke-test HeuristicRCASolver and FixedActionAgent were largely
instruction-insensitive, so distinct policy samples collapsed to identical
trajectories and zero group-relative advantage.

This module reuses the same shared Qwen base already resident on the GPU, but runs
it with every LoRA adapter disabled.  The downstream agents are therefore fixed:
only ``lora_rca`` and ``lora_action`` are trainable.  Downstream decoding is greedy
so trajectory variation is attributable to the sampled trainable-policy
instruction, not extra executor sampling noise.

Only agent-facing redacted/bounded state and public RCA/twin feedback are passed to
these classes.  They never receive evaluator labels, candidate root causes, or the
private full incident state.
"""

import contextlib
import json
import re
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrozenBaseGenerationConfig:
    max_prompt_tokens: int = 22_000
    rca_max_new_tokens: int = 64
    action_max_new_tokens: int = 128
    chat_role: str = "user"

    def validate(self) -> None:
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1")
        if self.rca_max_new_tokens < 1:
            raise ValueError("rca_max_new_tokens must be >= 1")
        if self.action_max_new_tokens < 1:
            raise ValueError("action_max_new_tokens must be >= 1")
        if not self.chat_role.strip():
            raise ValueError("chat_role must be non-empty")


class FrozenBaseQwenGenerator:
    """Greedy chat-template generation with all PEFT adapters disabled."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        config: FrozenBaseGenerationConfig | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or FrozenBaseGenerationConfig()
        self.config.validate()
        self.device = torch.device(device)
        if not hasattr(model, "set_adapter"):
            raise TypeError("frozen shared-base generator requires a PEFT-capable model")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer must expose apply_chat_template()")
        self.last_generation_info: dict[str, Any] = {}

    def _active_adapters_snapshot(self) -> list[str] | None:
        active = getattr(self.model, "active_adapters", None)
        try:
            value = active() if callable(active) else active
        except Exception:
            return None
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
        return None

    @contextlib.contextmanager
    def _frozen_base_context(self):
        """Temporarily disable every adapter and restore prior adapter state."""
        singular = getattr(self.model, "disable_adapter", None)
        if callable(singular):
            cm = singular()
            if not hasattr(cm, "__enter__") or not hasattr(cm, "__exit__"):
                raise TypeError("model.disable_adapter() is not a context manager")
            with cm:
                yield
            return

        disable = getattr(self.model, "disable_adapters", None)
        enable = getattr(self.model, "enable_adapters", None)
        if not callable(disable) or not callable(enable):
            raise TypeError(
                "shared PEFT model must expose disable_adapter() or "
                "disable_adapters()/enable_adapters()"
            )
        active_before = self._active_adapters_snapshot()
        disable()
        try:
            yield
        finally:
            enable()
            if active_before:
                self.model.set_adapter(
                    active_before[0] if len(active_before) == 1 else active_before
                )

    def _encode(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": self.config.chat_role, "content": str(prompt_text)}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = (
            rendered.get("input_ids")
            if isinstance(rendered, dict)
            else getattr(rendered, "input_ids", None)
        )
        if input_ids is None:
            raise TypeError("chat template did not return input_ids")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"unexpected chat-template input shape: {tuple(input_ids.shape)}")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > self.config.max_prompt_tokens:
            raise ValueError(
                "frozen downstream prompt exceeds semantic mechanics budget: "
                f"{prompt_tokens} > {self.config.max_prompt_tokens}; no token truncation is allowed"
            )
        attention_mask = (
            rendered.get("attention_mask")
            if isinstance(rendered, dict)
            else getattr(rendered, "attention_mask", None)
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return input_ids.to(self.device), attention_mask.to(self.device)

    def generate(self, prompt_text: str, *, max_new_tokens: int) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        input_ids, attention_mask = self._encode(prompt_text)
        prompt_tokens = int(input_ids.shape[1])
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with self._frozen_base_context():
                with torch.inference_mode():
                    sequences = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )
        finally:
            self.model.train(was_training)

        if sequences.ndim != 2 or sequences.shape[0] != 1:
            raise RuntimeError(f"unexpected frozen-base generation shape: {tuple(sequences.shape)}")
        completion_ids = sequences[0, prompt_tokens:].detach().cpu().tolist()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        self.last_generation_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(completion_ids),
            "max_new_tokens": int(max_new_tokens),
            "adapters_disabled": True,
            "do_sample": False,
            "decoder": "greedy",
            "prompt_was_truncated": False,
        }
        return text


def _json_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _normalize_rca_lines(text: str) -> str:
    """Normalize formatting only; never invent a root cause."""
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line).strip()
        if "::" in line:
            lines.append(line)
    return "\n".join(lines) if lines else str(text or "").strip()


def _extract_command_lines(text: str, *, max_commands: int) -> list[str]:
    """Keep literal generated command lines only; there is no remediation fallback."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`").strip()
        line = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line).strip()
        if not line.startswith(("kubectl ", "helm ", "mongosh ")):
            continue
        if line not in seen:
            out.append(line)
            seen.add(line)
        if len(out) >= max_commands:
            break
    return out


class FrozenQwenRCASolver:
    """Fixed RCA agent conditioned on the trainable policy's sampled instruction."""

    def __init__(self, generator: FrozenBaseQwenGenerator):
        self.generator = generator
        self.last_generation_info: dict[str, Any] = {}

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        payload = {
            "task": "Perform root-cause analysis from the redacted observability state.",
            "policy_instruction": str(instruction),
            "redacted_state": compressed_state,
            "output_contract": (
                "Output only root-cause lines in component::fault_mechanism format, "
                "one line per root cause. Do not add prose, labels, confidence, or explanations."
            ),
            "reasoning_requirements": [
                "Use only evidence present in redacted_state.",
                "Distinguish a causal root service from downstream victims.",
                "Use the smallest root-cause set supported by the evidence.",
            ],
        }
        raw = self.generator.generate(
            _json_prompt(payload),
            max_new_tokens=self.generator.config.rca_max_new_tokens,
        )
        self.last_generation_info = dict(self.generator.last_generation_info)
        self.last_generation_info["raw_output"] = raw
        return _normalize_rca_lines(raw)


class FrozenQwenActionAgent:
    """Fixed remediation agent conditioned on the trainable Action-policy instruction."""

    def __init__(self, generator: FrozenBaseQwenGenerator, *, max_commands: int = 15):
        self.generator = generator
        self.max_commands = max(1, int(max_commands))
        self.last_generation_info: dict[str, Any] = {}

    def get_commands(self, instruction_prompt: str, context: dict[str, Any]) -> list[str]:
        payload = {
            "task": "Generate a safe minimal Kubernetes remediation for the predicted RCA.",
            "policy_instruction": str(instruction_prompt),
            "namespace": context.get("namespace") or "default",
            "predicted_rca": context.get("rca_result") or {},
            "predicted_root_causes": context.get("rca_faults") or [],
            "counterfactual_twin_feedback": context.get("rca_twin_gate") or {},
            "current_sla": context.get("current_sla") or {},
            "redacted_state": context.get("redacted_state") or {},
            "output_contract": (
                "Output kubectl, helm, or mongosh commands only, one command per line. "
                "No markdown fences and no prose."
            ),
            "safety_requirements": [
                "Use only namespace-scoped commands.",
                "Target predicted root causes rather than downstream victims.",
                "Prefer the smallest mechanism-appropriate repair.",
                "Include a verification command.",
                "Do not use exec, apply, replace, shell pipelines, broad deletes, or cluster-wide flags.",
            ],
        }
        raw = self.generator.generate(
            _json_prompt(payload),
            max_new_tokens=self.generator.config.action_max_new_tokens,
        )
        commands = _extract_command_lines(raw, max_commands=self.max_commands)
        self.last_generation_info = dict(self.generator.last_generation_info)
        self.last_generation_info["raw_output"] = raw
        self.last_generation_info["parsed_command_count"] = len(commands)
        return commands
