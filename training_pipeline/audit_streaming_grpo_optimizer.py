from __future__ import annotations

"""CPU audit that streaming row backward equals the reference aggregate gradient."""

import copy
import json
import math
from types import SimpleNamespace

import torch
from torch import nn

from .factorized_grpo_learner import FactorizedGRPOConfig, role_buffer_loss
from .streaming_grpo_optimizer import streaming_role_backward


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 5):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(vocab_size), requires_grad=False)
        self.lora_rca = nn.Parameter(torch.tensor([0.0, 0.1, -0.1, 0.05, -0.05]))
        self.lora_action = nn.Parameter(torch.zeros(vocab_size), requires_grad=False)
        self.active_adapter = "lora_rca"

    def set_adapter(self, name: str) -> None:
        self.active_adapter = name

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        adapter = self.lora_rca if self.active_adapter == "lora_rca" else self.lora_action
        logits = (self.base + adapter).view(1, 1, -1).expand(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        return SimpleNamespace(logits=logits)


def _token_logprob(model: TinyPolicy, token: int) -> float:
    logits = (model.base + model.lora_rca).detach()
    return float(torch.log_softmax(logits, dim=-1)[int(token)])


def _rows(model: TinyPolicy) -> list[dict]:
    # Two trajectories in one incident group. t0 has one decision; t1 has two.
    # This directly exercises the 1/D_role streaming scale.
    specs = [
        ("t0", 1.0, 1, 0.8),
        ("t1", 0.5, 2, -0.8),
        ("t1", 0.5, 3, -0.8),
    ]
    rows = []
    for i, (tid, weight, token, advantage) in enumerate(specs):
        old = _token_logprob(model, token)
        rows.append({
            "optimizer_group_id": "g0:rca_policy",
            "trajectory_id": tid,
            "optimizer_sample_weight": weight,
            "prompt_token_ids": [0, 4],
            "completion_token_ids": [token],
            "old_logprobs": [old],
            "ref_logprobs": [old],
            "policy_advantage": advantage,
            "sample_id": f"s{i}",
        })
    return rows


def main() -> None:
    torch.manual_seed(0)
    cfg = FactorizedGRPOConfig(kl_coeff=0.0, max_grad_norm=100.0)

    reference = TinyPolicy()
    streaming = copy.deepcopy(reference)
    rows = _rows(reference)

    reference.zero_grad(set_to_none=True)
    ref_loss, ref_diag = role_buffer_loss(reference, rows, config=cfg, device="cpu")
    ref_loss.backward()
    ref_grad = reference.lora_rca.grad.detach().clone()

    streaming.zero_grad(set_to_none=True)
    stream_diag = streaming_role_backward(streaming, rows, config=cfg, device="cpu")
    stream_grad = streaming.lora_rca.grad.detach().clone()

    max_grad_diff = float((ref_grad - stream_grad).abs().max())
    loss_diff = abs(float(ref_loss.detach()) - float(stream_diag["loss"]))
    if max_grad_diff > 1e-7:
        raise AssertionError(f"streaming/reference gradient mismatch: {max_grad_diff}")
    if loss_diff > 1e-7:
        raise AssertionError(f"streaming/reference aggregate loss mismatch: {loss_diff}")
    if not torch.isfinite(stream_grad).all():
        raise AssertionError("streaming gradient contains NaN/Inf")

    print(json.dumps({
        "status": "PASS",
        "reference_loss": float(ref_loss.detach()),
        "streaming_loss": float(stream_diag["loss"]),
        "absolute_loss_difference": loss_diff,
        "max_absolute_gradient_difference": max_grad_diff,
        "reference_gradient": [float(x) for x in ref_grad],
        "streaming_gradient": [float(x) for x in stream_grad],
        "unequal_decisions_per_trajectory_tested": True,
        "streaming_row_backward": True,
        "retains_all_row_graphs": False,
        "reference_diagnostics": ref_diag,
        "streaming_diagnostics": stream_diag,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
