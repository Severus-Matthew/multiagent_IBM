from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from torch import nn

from .factorized_grpo_learner import (
    FactorizedGRPOConfig,
    model_decision_loss,
    role_buffer_loss,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor([0.2, -0.1, 0.0]), requires_grad=False)
        self.adapter = nn.Parameter(torch.zeros(3))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = (self.base + self.adapter).view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def rollout_logprob(model: TinyPolicy, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    out = model(input_ids=ids).logits
    # For this stationary tiny policy, every position has the same categorical logits.
    logp = torch.log_softmax(out[0, 0].float(), dim=-1)
    return [float(logp[token].detach()) for token in completion_ids]


def make_row(model: TinyPolicy, trajectory: str, advantage: float, weight: float, completion_ids: list[int]):
    prompt_ids = [2]
    old = rollout_logprob(model, prompt_ids, completion_ids)
    return {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "old_logprobs": old,
        "old_logprob_sum": sum(old),
        "ref_logprobs": old,
        "policy_advantage": advantage,
        "optimizer_sample_weight": weight,
        "optimizer_group_id": "incident0:rca_policy",
        "trajectory_id": trajectory,
    }


def main() -> None:
    torch.manual_seed(0)
    model = TinyPolicy()
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)

    row = make_row(model, "t0", 1.0, 1.0, [1, 0])
    d = model_decision_loss(model, row, config=cfg)
    if abs(float(d.ratio_mean) - 1.0) > 1e-6:
        raise AssertionError(f"unchanged rollout policy must have ratio 1, got {float(d.ratio_mean)}")
    if abs(float(d.loss) + 1.0) > 1e-6:
        raise AssertionError(f"ratio=1, A=1, beta=0 should give decision loss -1, got {float(d.loss)}")

    # Two trajectories. t0 has one decision; t1 has two decisions. The role loss
    # must use stored 1/D weights and therefore give both trajectories equal mass.
    rows = [
        make_row(model, "t0", 1.0, 1.0, [1]),
        make_row(model, "t1", -1.0, 0.5, [0]),
        make_row(model, "t1", -1.0, 0.5, [2]),
    ]
    role_loss, diag = role_buffer_loss(model, rows, config=cfg)

    # At ratio 1 the t0 trajectory loss is -1 and t1 is +1, so equal trajectory
    # averaging must produce exactly 0 even though t1 has twice as many rows.
    if abs(float(role_loss.detach())) > 1e-6:
        raise AssertionError(f"exact-token role aggregation has trajectory-length bias: {float(role_loss)}")

    role_loss.backward()
    if model.base.grad is not None:
        raise AssertionError("frozen base received a gradient")
    if model.adapter.grad is None or not torch.isfinite(model.adapter.grad).all():
        raise AssertionError("trainable adapter did not receive finite gradients")

    bad = dict(row)
    bad.pop("prompt_token_ids")
    try:
        model_decision_loss(model, bad, config=cfg)
    except ValueError:
        missing_prompt_rejected = True
    else:
        missing_prompt_rejected = False
    if not missing_prompt_rejected:
        raise AssertionError("learner accepted a row without exact prompt token IDs")

    print(json.dumps({
        "unchanged_policy_ratio_mean": float(d.ratio_mean),
        "ratio_one_positive_advantage_decision_loss": float(d.loss.detach()),
        "equal_trajectory_role_loss": float(role_loss.detach()),
        "frozen_base_gradient": None,
        "adapter_gradient_finite": True,
        "missing_prompt_token_ids_rejected": True,
        "diagnostics": diag,
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
