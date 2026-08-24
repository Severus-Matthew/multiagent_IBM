from __future__ import annotations

"""Real Qwen3-Coder shared-base backend for the factorized RCA/Action policies.

One frozen instruction-tuned causal LM is loaded once on a single CUDA device.
Two independent LoRA adapters are attached:

    lora_rca     -> RCA prompt policy
    lora_action  -> Action prompt policy

The backend intentionally adapts only attention q/v projections for the first
production implementation.  This keeps the LoRA state small and is directly
supported by the Transformers/PEFT integration for Qwen-family models.  The
factorized trainer owns trainability and optimizer state; this loader leaves every
parameter frozen after construction so rollout/replay cannot accidentally build a
training graph.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .peft_adapter_control import ROLE_ADAPTERS, parameter_belongs_to_adapter


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


@dataclass(frozen=True)
class QwenSharedPolicyBackendConfig:
    model_name: str = DEFAULT_QWEN_MODEL
    device: str = "cuda:0"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("real Qwen backend currently requires a CUDA device")
        if self.lora_r < 1:
            raise ValueError("lora_r must be >= 1")
        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be >= 1")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty")


@dataclass
class QwenSharedPolicyBackend:
    model: Any
    tokenizer: Any
    config: QwenSharedPolicyBackendConfig
    adapter_parameter_counts: dict[str, int]


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("expected CUDA device")
    return int(device.index if device.index is not None else torch.cuda.current_device())


def _freeze_all(model: Any) -> None:
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _adapter_parameter_counts(model: Any) -> dict[str, int]:
    counts = {adapter: 0 for adapter in ROLE_ADAPTERS}
    for name, parameter in model.named_parameters():
        for adapter in ROLE_ADAPTERS:
            if parameter_belongs_to_adapter(name, adapter):
                counts[adapter] += int(parameter.numel())
    missing = [adapter for adapter, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(f"Qwen model is missing expected LoRA parameters for {missing}")
    return counts


def load_qwen_shared_policy_backend(
    config: QwenSharedPolicyBackendConfig | None = None,
) -> QwenSharedPolicyBackend:
    cfg = config or QwenSharedPolicyBackendConfig()
    cfg.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot load the real Qwen policy backend")
    device = torch.device(cfg.device)
    index = _device_index(device)

    try:
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Qwen backend requires transformers and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        device_map={"": index},
        low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    model.add_adapter(lora, adapter_name="lora_rca")
    model.add_adapter(lora, adapter_name="lora_action")

    peft_config = getattr(model, "peft_config", {}) or {}
    missing = sorted(ROLE_ADAPTERS - set(peft_config.keys()))
    if missing:
        raise RuntimeError(f"failed to attach expected adapters: {missing}")

    # Rollout and old/ref replay are always no-grad. The synchronized trainer will
    # explicitly reactivate only one role adapter at each optimizer boundary.
    _freeze_all(model)
    model.set_adapter("lora_rca")

    devices = {str(parameter.device) for parameter in model.parameters()}
    expected = str(device)
    normalized_expected = f"cuda:{index}"
    if any(d not in {expected, normalized_expected} for d in devices):
        raise RuntimeError(f"Qwen parameters are unexpectedly split across devices: {sorted(devices)}")

    return QwenSharedPolicyBackend(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        adapter_parameter_counts=_adapter_parameter_counts(model),
    )
