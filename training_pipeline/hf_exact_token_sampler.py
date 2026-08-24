from __future__ import annotations

"""Exact-token rollout sampling for the factorized RCA/Action policies.

The sampler deliberately separates *sampling* from the log-probability contract
used by GRPO. A completion is first generated with the active role adapter. We
then replay the exact prompt+completion token sequence through that same adapter
and store raw-model per-token log probabilities, matching the policy log-probability
computation used by the audited learner. Reference log probabilities are computed
on the same exact tokens with all LoRA adapters disabled.

Generation may use either plain tokenizer encoding (used by the tiny local audits)
or the model tokenizer's chat template (used by instruction-tuned production
backends such as Qwen3-Coder). In both cases, the exact token IDs actually passed
to ``generate()`` are the authoritative rollout prompt representation.
"""

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .factorized_grpo_learner import completion_logprobs_from_causal_lm_logits
from .peft_adapter_control import ROLE_ADAPTERS


@dataclass(frozen=True)
class ExactTokenGenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    max_prompt_tokens: int | None = None
    seed: int = 0
    prompt_format: Literal["plain", "chat_template"] = "plain"
    chat_role: str = "user"

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1 when provided")
        if self.prompt_format not in {"plain", "chat_template"}:
            raise ValueError("prompt_format must be 'plain' or 'chat_template'")
        if not str(self.chat_role).strip():
            raise ValueError("chat_role must be non-empty")


class HFExactTokenPolicySampler:
    """Generate one policy completion and capture exact rollout statistics.

    ``model`` must be one shared causal-LM object containing named ``lora_rca``
    and ``lora_action`` adapters and supporting ``set_adapter``. ``tokenizer``
    must provide ``encode`` and ``decode`` plus optional EOS/PAD token IDs. When
    ``prompt_format='chat_template'``, it must also expose ``apply_chat_template``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        config: ExactTokenGenerationConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExactTokenGenerationConfig()
        self.config.validate()
        if not hasattr(model, "set_adapter"):
            raise TypeError("HFExactTokenPolicySampler requires a PEFT-capable model with set_adapter()")
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError("tokenizer must expose encode() and decode()")
        if self.config.prompt_format == "chat_template" and not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("chat_template prompt format requires tokenizer.apply_chat_template()")
        self.device = torch.device(device) if device is not None else self._infer_device()

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _plain_prompt_ids(self, prompt_text: str) -> list[int]:
        ids = self.tokenizer.encode(str(prompt_text), add_special_tokens=True)
        return [int(x) for x in ids]

    def _chat_template_prompt_ids(self, prompt_text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": str(self.config.chat_role), "content": str(prompt_text)}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(rendered, dict):
            input_ids = rendered.get("input_ids")
        else:
            input_ids = getattr(rendered, "input_ids", None)
        if input_ids is None:
            raise TypeError("apply_chat_template(..., return_dict=True) did not return input_ids")
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 2:
                if input_ids.shape[0] != 1:
                    raise ValueError(f"chat template returned unexpected batch size: {tuple(input_ids.shape)}")
                input_ids = input_ids[0]
            if input_ids.ndim != 1:
                raise ValueError(f"chat template input_ids must be 1-D after unbatching: {tuple(input_ids.shape)}")
            return [int(x) for x in input_ids.detach().cpu().tolist()]
        if isinstance(input_ids, (list, tuple)):
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                if len(input_ids) != 1:
                    raise ValueError("chat template returned more than one prompt sequence")
                input_ids = input_ids[0]
            return [int(x) for x in input_ids]
        raise TypeError(f"unsupported chat-template input_ids type: {type(input_ids)!r}")

    def _encode_prompt(self, prompt_text: str) -> tuple[list[int], bool]:
        if not str(prompt_text).strip():
            raise ValueError("policy prompt must be non-empty")
        if self.config.prompt_format == "chat_template":
            ids = self._chat_template_prompt_ids(prompt_text)
        else:
            ids = self._plain_prompt_ids(prompt_text)
        if not ids:
            raise ValueError("tokenizer produced an empty policy prompt")
        if any(x < 0 for x in ids):
            raise ValueError("prompt token IDs must be non-negative")

        truncated = False
        limit = self.config.max_prompt_tokens
        if limit is not None and len(ids) > limit:
            # Keep the most recent context and log that exact fact. Production Qwen
            # smoke tests intentionally use no truncation so the full chat template
            # is preserved. A later long-context policy may opt into this behavior.
            ids = ids[-int(limit):]
            truncated = True
        return ids, truncated

    def _rng_context(self, seed: int):
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.random.fork_rng(devices=[index], enabled=True)
        return torch.random.fork_rng(devices=[], enabled=True)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    def _active_adapters_snapshot(self) -> list[str] | None:
        """Best-effort snapshot of the currently active adapter names."""
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
    def _reference_context(self):
        """Run the shared frozen base with *all* PEFT adapters disabled.

        Hugging Face exposes two valid PEFT integration surfaces:

        * ``PeftModel.disable_adapter()`` -- singular context manager;
        * Transformers ``PeftAdapterMixin.disable_adapters()`` /
          ``enable_adapters()`` -- plural imperative methods.

        The tiny/local audits use the Transformers mixin, while a production Qwen
        setup may use a wrapped ``PeftModel``. Supporting both is required; silently
        evaluating an active LoRA as the reference policy is never allowed.
        """
        singular = getattr(self.model, "disable_adapter", None)
        if callable(singular):
            cm = singular()
            if not hasattr(cm, "__enter__") or not hasattr(cm, "__exit__"):
                raise TypeError("model.disable_adapter() exists but is not a context manager")
            with cm:
                yield
            return

        disable = getattr(self.model, "disable_adapters", None)
        enable = getattr(self.model, "enable_adapters", None)
        if not callable(disable) or not callable(enable):
            raise TypeError(
                "shared PEFT model must expose either disable_adapter() context manager "
                "or disable_adapters()/enable_adapters() for exact frozen-base reference logprobs"
            )

        active_before = self._active_adapters_snapshot()
        disable()
        try:
            yield
        finally:
            enable()
            # Transformers normally preserves the active adapter across a disable /
            # enable cycle. Restore it explicitly when the API exposes the name so
            # a future implementation change cannot silently alter rollout state.
            if active_before:
                setter = getattr(self.model, "set_adapter", None)
                if not callable(setter):
                    raise TypeError("model lost set_adapter() while restoring reference context")
                setter(active_before[0] if len(active_before) == 1 else active_before)

    def _completion_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            values = completion_logprobs_from_causal_lm_logits(
                logits,
                input_ids,
                prompt_length=len(prompt_ids),
                completion_length=len(completion_ids),
            )
        return [float(x) for x in values.detach().cpu()]

    def generate(
        self,
        prompt_text: str,
        *,
        adapter_name: str,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if adapter_name not in ROLE_ADAPTERS:
            raise ValueError(f"unsupported adapter_name={adapter_name!r}")
        self.config.validate()
        self.model.set_adapter(adapter_name)

        prompt_ids, prompt_truncated = self._encode_prompt(prompt_text)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        seed = int(self.config.seed) + int(sample_index)
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with self._rng_context(seed):
                self._set_seed(seed)
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": int(self.config.max_new_tokens),
                    "do_sample": bool(self.config.do_sample),
                    "eos_token_id": eos_id,
                    "pad_token_id": pad_id,
                    "use_cache": True,
                }
                if self.config.do_sample:
                    generation_kwargs["temperature"] = float(self.config.temperature)
                    generation_kwargs["top_p"] = float(self.config.top_p)
                with torch.no_grad():
                    sequences = self.model.generate(**generation_kwargs)

            if sequences.ndim != 2 or sequences.shape[0] != 1:
                raise RuntimeError(f"expected generate() to return [1, seq], got {tuple(sequences.shape)}")
            generated = sequences[0, len(prompt_ids):].detach().cpu().tolist()
            completion_ids = [int(x) for x in generated]
            if not completion_ids:
                raise RuntimeError("generation produced zero completion tokens")

            # Recompute old policy logprobs from the active rollout adapter on the
            # exact sampled token sequence. Do not use text re-tokenization and do
            # not use transformed sampling scores for the PPO/GRPO ratio.
            self.model.set_adapter(adapter_name)
            old_logprobs = self._completion_logprobs(prompt_ids, completion_ids)

            # Reference policy is the exact same frozen shared base with adapters
            # disabled, evaluated on the same token sequence.
            with self._reference_context():
                ref_logprobs = self._completion_logprobs(prompt_ids, completion_ids)
            self.model.set_adapter(adapter_name)

            if len(old_logprobs) != len(completion_ids):
                raise AssertionError("old_logprobs/completion_token_ids length mismatch")
            if len(ref_logprobs) != len(completion_ids):
                raise AssertionError("ref_logprobs/completion_token_ids length mismatch")

            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            if not str(completion_text).strip():
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
            completion_text = str(completion_text).strip()
            if not completion_text:
                raise RuntimeError("generated completion decoded to empty text")

            info = {
                "adapter_name": adapter_name,
                "group_id": group_id,
                "sample_index": int(sample_index),
                "generation_seed": seed,
                "prompt_text": str(prompt_text),
                "policy_prompt_sha256": hashlib.sha256(str(prompt_text).encode("utf-8")).hexdigest(),
                "prompt_token_ids": prompt_ids,
                "prompt_was_truncated": bool(prompt_truncated),
                "prompt_format": self.config.prompt_format,
                "chat_role": str(self.config.chat_role) if self.config.prompt_format == "chat_template" else None,
                "chat_template_applied": bool(self.config.prompt_format == "chat_template"),
                "completion_token_ids": completion_ids,
                "old_logprobs": old_logprobs,
                "old_logprob_sum": float(sum(old_logprobs)),
                "ref_logprobs": ref_logprobs,
                "old_logprobs_source": "active_adapter_forward_replay_on_exact_sampled_tokens",
                "reference_logprobs_source": "shared_frozen_base_with_adapters_disabled",
                "sampling_temperature": float(self.config.temperature),
                "sampling_top_p": float(self.config.top_p),
                "sampling_do_sample": bool(self.config.do_sample),
                "exact_token_replay_contract": True,
            }
            return completion_text, info
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()
