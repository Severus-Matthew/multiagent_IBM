from __future__ import annotations

"""Real Qwen3-Coder shared-base backend for factorized RCA/Action policies.

The 30B-A3B checkpoint cannot leave enough headroom for Qwen3-MoE prefill and
later LoRA backward passes when loaded as full BF16 on a single 96-GiB GPU. The
single-GPU path therefore uses a frozen bitsandbytes NF4 base with BF16 compute
and two independent LoRA adapters:

    lora_rca     -> RCA prompt policy
    lora_action  -> Action prompt policy

A subtle Qwen3-MoE detail matters here: the routed expert weights are stored as
large grouped tensors rather than ordinary ``nn.Linear`` modules. bitsandbytes
therefore does not turn every expert tensor into ``Params4bit``. Current PEFT
``prepare_model_for_kbit_training`` upcasts every remaining BF16/FP16 parameter
that is not a ``Params4bit`` object to FP32. On this 30B MoE that transiently
nearly doubles the expert-memory footprint and OOMs a 96-GiB GPU.

For our custom LoRA-only learner we do not need that global upcast. We implement
the required preparation explicitly: freeze every base parameter, leave frozen
non-quantized tensors in their loaded dtype, attach LoRA only to q/v projections,
and let the synchronized trainer reactivate exactly one adapter at an optimizer
boundary. This preserves the frozen-base contract without the destructive FP32
conversion.

For this memory-constrained single-GPU path we also default the MoE expert
implementation to ``eager``. Transformers' grouped-mm prefill materializes
num_tokens x top_k expert-token workspaces; on the long Action prompt that pushed
runtime allocation from ~56 GiB after load to ~90 GiB before another ~5 GiB
workspace was requested. Eager expert execution processes expert subsets
sequentially and trades throughput for materially lower peak prefill memory. The
expert backend can still be overridden explicitly for larger-memory hosts.

Reference-policy log probabilities remain well defined: they are evaluated on
the exact same frozen partially-quantized base with both LoRA adapters disabled.
Only LoRA parameters are ever optimized.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .peft_adapter_control import ROLE_ADAPTERS, parameter_belongs_to_adapter


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
SUPPORTED_QUANTIZATION_MODES = {"nf4", "bf16"}
SUPPORTED_EXPERTS_IMPLEMENTATIONS = {"eager", "batched_mm", "grouped_mm"}


@dataclass(frozen=True)
class QwenSharedPolicyBackendConfig:
    model_name: str = DEFAULT_QWEN_MODEL
    device: str = "cuda:0"
    quantization: str = "nf4"
    experts_implementation: str = "eager"
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
        if self.quantization not in SUPPORTED_QUANTIZATION_MODES:
            raise ValueError(
                f"quantization must be one of {sorted(SUPPORTED_QUANTIZATION_MODES)}; "
                f"got {self.quantization!r}"
            )
        if self.experts_implementation not in SUPPORTED_EXPERTS_IMPLEMENTATIONS:
            raise ValueError(
                "experts_implementation must be one of "
                f"{sorted(SUPPORTED_EXPERTS_IMPLEMENTATIONS)}; got {self.experts_implementation!r}"
            )
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
    quantization_mode: str
    experts_implementation: str
    model_memory_footprint_gib: float | None
    cuda_allocated_after_base_load_gib: float | None


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


def _memory_footprint_gib(model: Any) -> float | None:
    getter = getattr(model, "get_memory_footprint", None)
    if not callable(getter):
        return None
    try:
        return float(getter()) / 1024**3
    except Exception:
        return None


def _cuda_allocated_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated()) / 1024**3


def _validate_base_frozen(model: Any) -> None:
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        preview = trainable[:8]
        raise RuntimeError(f"base model must be fully frozen before adapter injection; found {preview}")


def _read_experts_implementation(model: Any, fallback: str) -> str:
    getter = getattr(model, "get_experts_implementation", None)
    if not callable(getter):
        return fallback
    try:
        value = getter()
    except Exception:
        return fallback
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        root = value.get("")
        if root is not None:
            return str(root)
        if value:
            return ",".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return fallback


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
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("Qwen backend requires transformers and peft") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    load_kwargs: dict[str, Any] = {
        "device_map": {"": index},
        "low_cpu_mem_usage": True,
        "experts_implementation": cfg.experts_implementation,
    }
    if cfg.quantization == "nf4":
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "NF4 QLoRA backend requires bitsandbytes; install it with "
                "`python -m pip install -U bitsandbytes`"
            ) from exc
        load_kwargs.update(
            {
                "dtype": torch.bfloat16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            }
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = True
    base_model.eval()

    # IMPORTANT: do NOT call prepare_model_for_kbit_training() here. Current PEFT
    # upcasts every BF16/FP16 parameter that is not a bitsandbytes Params4bit
    # object. Qwen3-MoE expert matrices are grouped tensors and many remain BF16,
    # so that preparation path OOMs a 96-GiB GPU. Our custom learner only needs a
    # frozen base plus trainable LoRA parameters, so freeze the base explicitly
    # and preserve its loaded dtypes.
    _freeze_all(base_model)
    _validate_base_frozen(base_model)
    allocated_after_base_load = _cuda_allocated_gib()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=list(cfg.target_modules),
        bias="none",
    )

    # PEFT understands bitsandbytes Linear4bit attention projections. The first
    # adapter wraps the shared base; the second is another named adapter on the
    # same object. The routed MoE experts remain frozen and are never optimizer
    # parameters.
    model = get_peft_model(base_model, lora, adapter_name="lora_rca")
    model.add_adapter("lora_action", lora)

    peft_config = getattr(model, "peft_config", {}) or {}
    missing = sorted(ROLE_ADAPTERS - set(peft_config.keys()))
    if missing:
        raise RuntimeError(f"failed to attach expected adapters: {missing}")

    # Rollout and old/ref replay are no-grad. The synchronized trainer explicitly
    # reactivates only one role adapter at each optimizer boundary.
    _freeze_all(model)
    model.set_adapter("lora_rca")
    _freeze_all(model)

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
        quantization_mode=cfg.quantization,
        experts_implementation=_read_experts_implementation(model, cfg.experts_implementation),
        model_memory_footprint_gib=_memory_footprint_gib(model),
        cuda_allocated_after_base_load_gib=allocated_after_base_load,
    )
