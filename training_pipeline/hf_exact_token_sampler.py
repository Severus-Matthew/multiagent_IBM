from __future__ import annotations

"""Exact-token rollout sampling for the factorized RCA/Action policies.

The sampler deliberately separates *sampling* from the log-probability contract
used by GRPO.  A completion is first generated with the active role adapter.  We
then replay the exact prompt+completion token sequence through that same adapter
and store raw-model per-token log probabilities, matching the policy log-probability
computation used by the audited learner.  Reference log probabilities are computed
on the same exact tokens with all LoRA adapters disabled.

This mirrors the important behavior of modern GRPO implementations: generation
may use temperature/top-p to obtain diverse samples, while old-policy logprobs are
recomputed from the rollout model on the sampled token sequence and stored for
later importance ratios.
"""

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Any

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

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1 when provided")


class HFExactTokenPolicySampler:
    """Generate one policy completion and capture exact rollout statistics.

    ``model`` must be one shared causal-LM object containing named ``lora_rca``
    and ``lora_action`` adapters and supporting ``set_adapter``.  ``tokenizer``
    must provide ``encode`` and ``decode`` plus optional EOS/PAD token IDs.
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
        self.device = torch.device(device) if device is not None else self._infer_device()

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _encode_prompt(self, prompt_text: str) -> tuple[list[int], bool]:
        if not str(prompt_text).strip():
            raise ValueError("policy prompt must be non-empty")
        ids = self.tokenizer.encode(str(prompt_text), add_special_tokens=True)
        ids = [int(x) for x in ids]
        if not ids:
            raise ValueError("tokenizer produced an empty policy prompt")
        if any(x < 0 for x in ids):
            raise ValueError("prompt token IDs must be non-negative")

        truncated = False
        limit = self.config.max_prompt_tokens
        if limit is not None and len(ids) > limit:
            # Keep the most recent context.  Real Qwen has a large context window,
            # so truncation should be exceptional and is explicitly logged.
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

    @contextlib.contextmanager
    def _reference_context(self):
        if not hasattr(self.model, "disable_adapter"):
            raise TypeError(
                "shared PEFT model must expose disable_adapter() so frozen-base reference logprobs are exact"
            )
        with self.model.disable_adapter():
            yield

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
                with torch.no_grad():
                    sequences = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(self.config.max_new_tokens),
                        do_sample=bool(self.config.do_sample),
                        temperature=float(self.config.temperature) if self.config.do_sample else None,
                        top_p=float(self.config.top_p) if self.config.do_sample else None,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )

            if sequences.ndim != 2 or sequences.shape[0] != 1:
                raise RuntimeError(f"expected generate() to return [1, seq], got {tuple(sequences.shape)}")
            generated = sequences[0, len(prompt_ids):].detach().cpu().tolist()
            completion_ids = [int(x) for x in generated]
            if not completion_ids:
                raise RuntimeError("generation produced zero completion tokens")

            # Recompute old policy logprobs from the active rollout adapter on the
            # exact sampled token sequence.  Do not use text re-tokenization and do
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
