"""
agents/prompt_optimizer.py

Prompt Optimization Agent — Qwen2.5-Coder-7B-Instruct with online GRPO training.

Role: Given the RCA result, fault context, and history of previous failed attempts,
generate a detailed instruction prompt for the Action Agent that will produce
the correct kubectl/helm commands to fix the fault.

This agent is TRAINABLE — its weights are updated via GRPO after each scenario
based on whether the Action Agent's commands successfully remediated the fault.

Architecture:
    PromptOptimizerAgent (this file)  ← inference: generate instruction prompts
    GRPOTrainer (training/grpo_trainer.py) ← training: update weights online

Usage:
    from agents.prompt_optimizer import PromptOptimizerAgent

    agent = PromptOptimizerAgent()         # loads model lazily on first call
    agent = PromptOptimizerAgent(          # explicit config
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        device="auto",
        use_lora=True,
    )

    instruction_prompt = agent.generate(opt_context)  # opt_context from build_prompt_optimizer_input

    # After training update (called by pipeline.py via GRPOTrainer):
    trainer = GRPOTrainer(agent.get_model(), agent.get_tokenizer())
    trainer.add_sample(prompt_text, instruction_prompt, reward)
    trainer.step()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SYSTEM_PROMPT = """\
You are a Kubernetes expert prompt engineer specializing in fault remediation.

Given:
- A root cause analysis result (which service failed and why)
- The fault context (namespace, pod states, service health)
- A history of previous fix attempts (if any) and their outcomes

Your task is to write a precise, actionable instruction prompt for an Action Agent.
The Action Agent can only output kubectl and helm commands — no prose, no explanations.
Your instruction prompt must guide it to produce the CORRECT commands in the RIGHT ORDER.

Be specific:
- Name exact services, deployments, and namespaces
- Specify the repair strategy step by step
- For auth failures: explain exactly how to re-establish credentials
- For pod failures: specify restart/scale sequences
- For config errors: specify exact patch commands with field values
- Tell the Action Agent what success looks like (which pods should be Running)

If previous attempts failed, analyze what went wrong and try a DIFFERENT approach.\
"""


class PromptOptimizerAgent:
    """
    Qwen2.5-Coder-7B-Instruct agent that generates instruction prompts for
    the Action Agent. Weights are updated online via GRPO training.

    Model is loaded lazily — call generate() or load() explicitly.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "auto",
        use_lora: bool = True,
        lora_rank: int = 16,
        max_new_tokens: int = 800,
    ):
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._tokenizer = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load Qwen2.5-Coder-7B-Instruct into memory.

        Applies LoRA adapters if use_lora=True (recommended for A100).
        Uses bfloat16 precision — requires GPU with bf16 support (A100, H100).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[PromptOptimizer] Loading {self.model_name} (device={self.device})")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.use_lora:
            self._apply_lora()

        self._model.eval()
        print(f"[PromptOptimizer] Model loaded. LoRA={self.use_lora}")

    def generate(
        self,
        context: dict,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate an instruction prompt for the Action Agent.

        Args:
            context:     dict from build_prompt_optimizer_input()
            temperature: sampling temperature (higher = more varied)
            top_p:       nucleus sampling threshold

        Returns:
            Instruction prompt string (passed directly to ActionAgent).
        """
        if self._model is None:
            self.load()

        import torch

        user_text = _format_context_for_qwen(context)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=3072
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(
            f"[PromptOptimizer] Generated instruction prompt "
            f"({len(result)} chars, iter={context.get('iteration', 0)})"
        )
        return result

    def get_model(self):
        """Return the underlying HuggingFace model (used by GRPOTrainer)."""
        if self._model is None:
            self.load()
        return self._model

    def get_tokenizer(self):
        """Return the tokenizer (used by GRPOTrainer)."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def save(self, path: str) -> None:
        """Save model / LoRA adapter weights to disk."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        print(f"[PromptOptimizer] Saved to {path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_lora(self) -> None:
        """Wrap the model with LoRA adapters for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError(
                "pip install peft  — required for LoRA fine-tuning"
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            lora_dropout=0.05,
            # Target the attention projection layers
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()


# ── Context formatter ─────────────────────────────────────────────────────────


def _format_context_for_qwen(ctx: dict) -> str:
    """
    Format the prompt_optimizer_input dict into a clean text prompt
    for Qwen2.5-Coder-7B.
    """
    lines = [
        f"## Scenario: {ctx.get('scenario_id', 'unknown')}",
        f"Namespace: {ctx.get('namespace', 'unknown')}",
        f"Iteration: {ctx.get('iteration', 0) + 1} / {ctx.get('max_iterations', 7)}",
        "",
    ]

    rca = ctx.get("rca_result", {})
    fault = ctx.get("fault_summary", {})
    lines += [
        "## RCA Result",
        f"Root cause service : {rca.get('root_cause_service', 'unknown')}",
        f"Fault type         : {rca.get('fault_type', 'unknown')}",
        f"Confidence         : {rca.get('confidence', 0.0):.2f}",
        f"Explanation        : {rca.get('explanation', '')}",
        f"Affected services  : {', '.join(rca.get('affected_services', []))}",
        "",
    ]

    pod_details = ctx.get("pod_details", {})
    if pod_details:
        lines.append("## Pod Details (relevant services)")
        for svc, info in pod_details.items():
            pods = info.get("pods", [])
            health = info.get("health_status", "unknown")
            restarts = info.get("restart_count", 0)
            lines.append(
                f"  {svc}: health={health}  restarts={restarts}  "
                f"pods={pods[:3]}"
            )
        lines.append("")

    action_lib = ctx.get("action_library", [])
    if action_lib:
        lines.append("## Available Actions")
        for a in action_lib[:8]:
            lines.append(f"  [{a.get('risk','?')} risk] {a.get('template','')}")
            lines.append(f"    → {a.get('effect','')}")
        lines.append("")

    prev = ctx.get("previous_attempts", [])
    if prev:
        lines.append("## Previous Attempts (learn from these failures)")
        for attempt in prev:
            lines.append(
                f"  Iteration {attempt.get('iteration', '?')}: "
                f"reward={attempt.get('reward', 0.0):.2f}  "
                f"result={attempt.get('result', '?')}"
            )
            cmds = attempt.get("commands", [])
            for cmd in cmds[:3]:
                lines.append(f"    $ {cmd}")
            if attempt.get("failure_reason"):
                lines.append(f"    Failure: {attempt['failure_reason']}")
        lines.append("")

    lines += [
        "## Task",
        ctx.get("task_instruction", "Generate a detailed instruction prompt for the Action Agent."),
    ]

    return "\n".join(lines)
