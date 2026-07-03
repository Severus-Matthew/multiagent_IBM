"""
training/grpo_trainer.py

Online GRPO (Group Relative Policy Optimization) trainer for the Prompt
Optimization Agent (Qwen2.5-Coder-7B).

GRPO summary
------------
For each scenario, up to G=7 iterations produce G (prompt, completion, reward)
samples with the SAME input prompt.  GRPO normalizes the rewards within this
group to compute advantages, then does a PPO-style policy gradient update.

This eliminates the need for a critic/value-network — the group mean acts as
the baseline.  Combined with LoRA adapters, the full 7B model is trainable on
a single A100-80GB.

Reference: DeepSeekMath / Proof2Silicon — online GRPO with verifiable rewards.

Reward normalization (per group):
    A_i = (r_i - mean(group)) / (std(group) + ε)

Policy gradient loss (PPO-clip):
    ratio   = exp(logπ_θ(o|q) - logπ_θ_old(o|q))
    L_clip  = -min(ratio * A_i, clip(ratio, 1-ε, 1+ε) * A_i)

Optional KL penalty (β * KL(π_θ || π_ref)) to prevent policy collapse.

Usage:
    from agents.prompt_optimizer import PromptOptimizerAgent
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

    agent = PromptOptimizerAgent()
    agent.load()

    config = GRPOConfig(learning_rate=1e-5, kl_coeff=0.01)
    trainer = GRPOTrainer(agent.get_model(), agent.get_tokenizer(), config)

    # After each iteration in pipeline:
    trainer.add_sample(prompt_text, completion_text, reward)

    # After scenario ends (all iterations done):
    if trainer.should_update():
        loss = trainer.step()

    # Periodically save:
    trainer.save("checkpoints/step_100")
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GRPOConfig:
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # PPO-clip
    clip_eps: float = 0.2

    # KL regularization (set 0.0 to disable)
    kl_coeff: float = 0.01

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Tokenizer truncation
    max_prompt_length: int = 1536
    max_completion_length: int = 512

    # Minimum samples in the pending buffer before a step is triggered
    # In practice this should equal the number of iterations per scenario (7)
    min_batch_size: int = 2

    # Normalise advantages across the full batch (not just per-prompt group)
    # Set True for large batches; False for online per-scenario updates
    normalize_across_batch: bool = False

    # Checkpoint every N steps (0 = never auto-save)
    checkpoint_every: int = 0
    checkpoint_dir: str = "checkpoints"


# ── Trainer ───────────────────────────────────────────────────────────────────


class GRPOTrainer:
    """
    Online GRPO trainer for a causal LM (Qwen2.5-Coder-7B with LoRA).

    Call add_sample() after each iteration, then step() after each scenario
    (or when should_update() returns True).
    """

    def __init__(self, model, tokenizer, config: GRPOConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()

        self._pending: list[dict] = []
        self._step_count: int = 0
        self._total_samples: int = 0
        self._ref_model = None  # frozen copy for KL; created lazily

        self._optimizer = self._build_optimizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_sample(
        self,
        prompt_text: str,
        completion_text: str,
        reward: float,
        group_id: str | None = None,
    ) -> None:
        """
        Buffer one (prompt, completion, reward) tuple.

        Args:
            prompt_text:      the full text input fed to the model
            completion_text:  the model's generated output
            reward:           scalar reward in [0.0, 1.0]
            group_id:         optional group identifier — samples with the same
                              group_id are normalised together (default: use
                              prompt_text as the group key)
        """
        self._pending.append({
            "prompt": prompt_text,
            "completion": completion_text,
            "reward": float(reward),
            "group_id": group_id if group_id is not None else prompt_text[:200],
        })

    def should_update(self) -> bool:
        """True when enough samples have accumulated for a gradient step."""
        return len(self._pending) >= self.config.min_batch_size

    def step(self) -> dict[str, float]:
        """
        Perform one GRPO gradient update on the pending sample buffer.

        Returns a dict with training metrics:
            {"loss": float, "mean_reward": float, "n_samples": int}
        """
        if not self._pending:
            return {"loss": 0.0, "mean_reward": 0.0, "n_samples": 0}

        batch = self._pending.copy()
        self._pending.clear()

        metrics = self._grpo_step(batch)
        self._step_count += 1
        self._total_samples += len(batch)

        if (
            self.config.checkpoint_every > 0
            and self._step_count % self.config.checkpoint_every == 0
        ):
            self.save(
                os.path.join(self.config.checkpoint_dir, f"step_{self._step_count}")
            )

        print(
            f"[GRPO] step={self._step_count}  "
            f"loss={metrics['loss']:.4f}  "
            f"mean_reward={metrics['mean_reward']:.3f}  "
            f"n={metrics['n_samples']}"
        )
        return metrics

    def save(self, path: str) -> None:
        """Save model (LoRA adapter) weights and trainer state."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        state = {
            "step_count": self._step_count,
            "total_samples": self._total_samples,
        }
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)
        print(f"[GRPO] Saved checkpoint to {path}")

    # ── GRPO update ───────────────────────────────────────────────────────────

    def _grpo_step(self, batch: list[dict]) -> dict[str, float]:
        """
        Full GRPO update over a batch of samples.

        Groups samples by group_id, normalizes rewards within each group
        to produce advantages, then performs PPO-clip policy gradient.
        """
        import torch
        from collections import defaultdict

        # ── Group-relative advantage computation ──────────────────────────────
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in batch:
            groups[s["group_id"]].append(s)

        for group_samples in groups.values():
            rewards = [s["reward"] for s in group_samples]
            if len(rewards) > 1:
                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                std_r = max(var_r ** 0.5, 1e-6)
                for s, r in zip(group_samples, rewards):
                    s["advantage"] = (r - mean_r) / std_r
            else:
                # Single sample: no group normalization possible
                group_samples[0]["advantage"] = 0.0

        # Optionally normalize advantages across the full batch
        if self.config.normalize_across_batch and len(batch) > 1:
            all_adv = [s["advantage"] for s in batch]
            adv_mean = sum(all_adv) / len(all_adv)
            adv_std = max(
                (sum((a - adv_mean) ** 2 for a in all_adv) / len(all_adv)) ** 0.5,
                1e-6,
            )
            for s in batch:
                s["advantage"] = (s["advantage"] - adv_mean) / adv_std

        # ── Policy gradient update ─────────────────────────────────────────────
        self.model.train()
        total_loss = torch.tensor(0.0, device=self._device())
        n_valid = 0

        for sample in batch:
            loss = self._sample_loss(
                prompt_text=sample["prompt"],
                completion_text=sample["completion"],
                advantage=sample["advantage"],
            )
            if loss is not None:
                total_loss = total_loss + loss
                n_valid += 1

        if n_valid == 0:
            self.model.eval()
            return {"loss": 0.0, "mean_reward": 0.0, "n_samples": 0}

        mean_loss = total_loss / n_valid
        self._optimizer.zero_grad()
        mean_loss.backward()
        import torch.nn as nn
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self._optimizer.step()
        self.model.eval()

        mean_reward = sum(s["reward"] for s in batch) / len(batch)
        return {
            "loss": mean_loss.item(),
            "mean_reward": mean_reward,
            "n_samples": len(batch),
        }

    def _sample_loss(
        self,
        prompt_text: str,
        completion_text: str,
        advantage: float,
    ):
        """
        PPO-clip policy gradient loss for one (prompt, completion, advantage) tuple.

        Loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + β*KL
        where ratio = exp(log π_θ(completion|prompt) - log π_θ_old(completion|prompt))

        Since we don't maintain π_θ_old explicitly between samples in online mode,
        we use the current policy as both new and old for the first pass (ratio=1).
        The KL term against the frozen reference model provides the regularisation.
        """
        import torch

        # Tokenize full sequence
        full_text = prompt_text + completion_text
        try:
            full_enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_length + self.config.max_completion_length,
            ).to(self._device())
            prompt_enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]
        except Exception:
            return None

        input_ids = full_enc["input_ids"]
        if input_ids.shape[1] <= prompt_len:
            return None  # no completion tokens

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

        # Completion token log-probs (shift by 1 for next-token prediction)
        comp_logits = logits[0, prompt_len - 1:-1, :]  # (comp_len, vocab)
        comp_ids = input_ids[0, prompt_len:]            # (comp_len,)

        log_probs = torch.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(comp_ids), device=self._device()), comp_ids
        ]
        completion_log_prob = token_log_probs.mean()

        # PPO-clip (with ratio=1 in online single-sample mode)
        # In a proper off-policy setup, old_log_prob would come from a saved snapshot.
        # For online single-sample updates, ratio=1 degenerates to vanilla PG.
        adv = torch.tensor(advantage, dtype=torch.float32, device=self._device())
        ratio = torch.exp(completion_log_prob - completion_log_prob.detach())
        clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
        pg_loss = -torch.min(ratio * adv, clipped * adv)

        # KL regularisation against frozen reference model
        kl_loss = torch.tensor(0.0, device=self._device())
        if self.config.kl_coeff > 0:
            ref = self._get_ref_model()
            if ref is not None:
                with torch.no_grad():
                    ref_outputs = ref(input_ids=input_ids)
                    ref_logits = ref_outputs.logits[0, prompt_len - 1:-1, :]
                    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                    ref_token_lp = ref_log_probs[
                        torch.arange(len(comp_ids), device=self._device()), comp_ids
                    ]
                    ref_log_prob = ref_token_lp.mean()

                kl = completion_log_prob - ref_log_prob
                kl_loss = self.config.kl_coeff * kl

        return pg_loss + kl_loss

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_optimizer(self):
        """Build AdamW optimizer for the model's trainable parameters."""
        try:
            import torch
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable:
                # No LoRA / all frozen — return a no-op optimizer
                return None
            return torch.optim.AdamW(
                trainable,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        except Exception as e:
            print(f"[GRPO] Warning: could not build optimizer: {e}")
            return None

    def _get_ref_model(self):
        """
        Return a frozen copy of the initial model for KL divergence computation.
        Created once on first call.
        """
        if self._ref_model is None and self.config.kl_coeff > 0:
            try:
                import torch
                self._ref_model = copy.deepcopy(self.model)
                for p in self._ref_model.parameters():
                    p.requires_grad_(False)
                self._ref_model.eval()
                print("[GRPO] Reference model frozen for KL regularisation")
            except Exception as e:
                print(f"[GRPO] Could not create reference model: {e}")
                self._ref_model = None
        return self._ref_model

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            import torch
            return torch.device("cpu")
