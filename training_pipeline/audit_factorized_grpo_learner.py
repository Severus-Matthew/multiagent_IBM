from __future__ import annotations

"""CPU mathematical/gradient audit for the reference factorized GRPO learner.

This audit deliberately uses a tiny causal policy so it can run before any GPU or
Qwen dependency is installed.  A failure is a hard blocker for the real learner.
"""

import argparse
import json
import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .factorized_grpo_learner import (
    FactorizedGRPOConfig,
    WeightedDecisionLoss,
    aggregate_role_loss,
    completion_logprobs_from_causal_lm_logits,
    decision_grpo_loss,
    sampled_reverse_kl,
)


class TinyTwoAdapterCausalPolicy(nn.Module):
    """Minimal shared frozen base + two independent trainable adapter biases."""

    def __init__(self, vocab_size: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.base_bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=False)
        self.lora_rca = nn.Parameter(torch.zeros(vocab_size))
        self.lora_action = nn.Parameter(torch.zeros(vocab_size))
        self.active_adapter = "lora_rca"

    def set_adapter(self, name: str) -> None:
        if name not in {"lora_rca", "lora_action"}:
            raise ValueError(name)
        self.active_adapter = name

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        adapter = self.lora_rca if self.active_adapter == "lora_rca" else self.lora_action
        logits_one = self.base_bias + adapter
        logits = logits_one.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def _assert_close(actual: float, expected: float, tol: float = 1e-7, message: str = "") -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
        raise AssertionError(message or f"expected={expected} actual={actual}")


def _sampled_token_logprob(model: TinyTwoAdapterCausalPolicy, token_id: int = 1) -> torch.Tensor:
    input_ids = torch.tensor([[0, token_id]], dtype=torch.long)
    logits = model(input_ids=input_ids).logits
    return completion_logprobs_from_causal_lm_logits(
        logits,
        input_ids,
        prompt_length=1,
        completion_length=1,
    )[0]


def audit_causal_alignment() -> dict[str, Any]:
    # Position 0 predicts the first completion token at input position 1.
    logits = torch.tensor(
        [[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[2, 0, 1]], dtype=torch.long)
    # Prompt token is [2].  Completion is [0, 1].  Correct causal predictors are
    # logits positions 0 and 1, both assigning high probability to the targets.
    gathered = completion_logprobs_from_causal_lm_logits(
        logits, input_ids, prompt_length=1, completion_length=2
    )
    if gathered.shape != (2,):
        raise AssertionError("completion alignment returned the wrong shape")
    if not bool((gathered > -0.01).all()):
        raise AssertionError(f"causal token/logit alignment is wrong: {gathered.tolist()}")
    return {"completion_logprobs": [float(x) for x in gathered]}


def audit_ratio_and_clipping_gradients() -> dict[str, Any]:
    cfg = FactorizedGRPOConfig(clip_epsilon_low=0.2, clip_epsilon_high=0.2, kl_coeff=0.0)

    # At ratio 1, positive advantage should increase the sampled token logprob.
    new = torch.tensor([-1.0], requires_grad=True)
    old = torch.tensor([-1.0])
    loss = decision_grpo_loss(new, old, 1.0, config=cfg).loss
    loss.backward()
    positive_grad = float(new.grad.item())
    if not positive_grad < 0.0:
        raise AssertionError(f"positive advantage must increase token probability; grad={positive_grad}")

    # At ratio 1, negative advantage should decrease the sampled token logprob.
    new2 = torch.tensor([-1.0], requires_grad=True)
    loss2 = decision_grpo_loss(new2, old, -1.0, config=cfg).loss
    loss2.backward()
    negative_grad = float(new2.grad.item())
    if not negative_grad > 0.0:
        raise AssertionError(f"negative advantage must decrease token probability; grad={negative_grad}")

    # Positive A with ratio above upper clip: clipped branch is constant -> zero grad.
    high = torch.tensor([-1.0 + math.log(1.5)], requires_grad=True)
    high_loss = decision_grpo_loss(high, old, 1.0, config=cfg)
    high_loss.loss.backward()
    _assert_close(float(high.grad.item()), 0.0, tol=1e-7, message="upper clipping gradient is wrong")

    # Negative A with ratio below lower clip: clipped branch is constant -> zero grad.
    low = torch.tensor([-1.0 + math.log(0.5)], requires_grad=True)
    low_loss = decision_grpo_loss(low, old, -1.0, config=cfg)
    low_loss.loss.backward()
    _assert_close(float(low.grad.item()), 0.0, tol=1e-7, message="lower clipping gradient is wrong")

    # Zero advantage and no KL must produce exactly zero policy-gradient contribution.
    zero = torch.tensor([-1.0], requires_grad=True)
    zero_loss = decision_grpo_loss(zero, old, 0.0, config=cfg).loss
    zero_loss.backward()
    _assert_close(float(zero.grad.item()), 0.0, tol=1e-8, message="zero advantage produced policy gradient")

    return {
        "positive_advantage_gradient": positive_grad,
        "negative_advantage_gradient": negative_grad,
        "upper_clipped_gradient": float(high.grad.item()),
        "lower_clipped_gradient": float(low.grad.item()),
        "zero_advantage_gradient": float(zero.grad.item()),
    }


def audit_actual_parameter_step_direction() -> dict[str, Any]:
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)

    pos_model = TinyTwoAdapterCausalPolicy()
    pos_model.set_adapter("lora_rca")
    before_pos = float(_sampled_token_logprob(pos_model).detach())
    old_pos = _sampled_token_logprob(pos_model).detach().view(1)
    opt = torch.optim.SGD([pos_model.lora_rca], lr=0.1)
    opt.zero_grad()
    new_pos = _sampled_token_logprob(pos_model).view(1)
    decision_grpo_loss(new_pos, old_pos, 1.0, config=cfg).loss.backward()
    opt.step()
    after_pos = float(_sampled_token_logprob(pos_model).detach())
    if not after_pos > before_pos:
        raise AssertionError("positive-advantage optimizer step did not increase sampled-token logprob")

    neg_model = TinyTwoAdapterCausalPolicy()
    neg_model.set_adapter("lora_rca")
    before_neg = float(_sampled_token_logprob(neg_model).detach())
    old_neg = _sampled_token_logprob(neg_model).detach().view(1)
    opt2 = torch.optim.SGD([neg_model.lora_rca], lr=0.1)
    opt2.zero_grad()
    new_neg = _sampled_token_logprob(neg_model).view(1)
    decision_grpo_loss(new_neg, old_neg, -1.0, config=cfg).loss.backward()
    opt2.step()
    after_neg = float(_sampled_token_logprob(neg_model).detach())
    if not after_neg < before_neg:
        raise AssertionError("negative-advantage optimizer step did not decrease sampled-token logprob")

    return {
        "positive_before": before_pos,
        "positive_after": after_pos,
        "negative_before": before_neg,
        "negative_after": after_neg,
    }


def audit_trajectory_weighting() -> dict[str, Any]:
    # Group has two trajectories. T0 has one decision with loss 2. T1 has four
    # decisions with loss 6 each. Correct aggregate is (2 + 6) / 2 = 4, not the
    # flat-row average (2 + 4*6)/5 = 5.2.
    records = [
        WeightedDecisionLoss(torch.tensor(2.0), "g", "t0", 1.0),
        *[
            WeightedDecisionLoss(torch.tensor(6.0), "g", "t1", 0.25)
            for _ in range(4)
        ],
    ]
    got = aggregate_role_loss(records)
    _assert_close(float(got), 4.0, message="trajectory length bias in role loss")

    # Equal averaging over incident groups as well.
    records2 = [
        WeightedDecisionLoss(torch.tensor(2.0), "g0", "a", 1.0),
        WeightedDecisionLoss(torch.tensor(2.0), "g0", "b", 1.0),
        WeightedDecisionLoss(torch.tensor(10.0), "g1", "a", 1.0),
        WeightedDecisionLoss(torch.tensor(10.0), "g1", "b", 1.0),
    ]
    got2 = aggregate_role_loss(records2)
    _assert_close(float(got2), 6.0, message="incident-group weighting is wrong")
    return {"unequal_decision_count_loss": float(got), "two_incident_group_loss": float(got2)}


def audit_adapter_isolation() -> dict[str, Any]:
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)
    model = TinyTwoAdapterCausalPolicy()

    # RCA backward must touch only RCA adapter. Freeze Action explicitly, matching
    # the contract the real PEFT wrapper must enforce.
    model.lora_rca.requires_grad_(True)
    model.lora_action.requires_grad_(False)
    model.set_adapter("lora_rca")
    old = _sampled_token_logprob(model).detach().view(1)
    loss = decision_grpo_loss(_sampled_token_logprob(model).view(1), old, 1.0, config=cfg).loss
    loss.backward()
    if model.base_bias.grad is not None:
        raise AssertionError("frozen shared base received RCA gradient")
    if model.lora_rca.grad is None or not torch.isfinite(model.lora_rca.grad).all():
        raise AssertionError("RCA adapter did not receive a finite gradient")
    if model.lora_action.grad is not None:
        raise AssertionError("Action adapter received gradient during RCA backward")

    model.zero_grad(set_to_none=True)
    model.lora_rca.requires_grad_(False)
    model.lora_action.requires_grad_(True)
    model.set_adapter("lora_action")
    old2 = _sampled_token_logprob(model).detach().view(1)
    loss2 = decision_grpo_loss(_sampled_token_logprob(model).view(1), old2, 1.0, config=cfg).loss
    loss2.backward()
    if model.base_bias.grad is not None:
        raise AssertionError("frozen shared base received Action gradient")
    if model.lora_action.grad is None or not torch.isfinite(model.lora_action.grad).all():
        raise AssertionError("Action adapter did not receive a finite gradient")
    if model.lora_rca.grad is not None:
        raise AssertionError("RCA adapter received gradient during Action backward")

    return {
        "shared_base_frozen": True,
        "rca_backward_isolated": True,
        "action_backward_isolated": True,
    }


def audit_kl() -> dict[str, Any]:
    p = torch.tensor([-1.0, -2.0], requires_grad=True)
    ref = torch.tensor([-1.0, -1.5])
    kl = sampled_reverse_kl(p, ref)
    if bool((kl < 0.0).any()):
        raise AssertionError("sampled reverse KL became negative")
    equal = sampled_reverse_kl(torch.tensor([-2.0]), torch.tensor([-2.0]))
    _assert_close(float(equal.item()), 0.0, tol=1e-8)
    return {"values": [float(x) for x in kl.detach()], "equal_policy_reference": float(equal.item())}


def audit_old_policy_is_detached() -> dict[str, Any]:
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)
    new = torch.tensor([-1.0], requires_grad=True)
    old = torch.tensor([-1.0], requires_grad=True)
    loss = decision_grpo_loss(new, old, 1.0, config=cfg).loss
    loss.backward()
    if old.grad is not None:
        raise AssertionError("stored old-policy logprob must be detached from gradient graph")
    if new.grad is None:
        raise AssertionError("new policy logprob must receive gradient")
    return {"old_policy_gradient": None, "new_policy_gradient": float(new.grad.item())}


def audit_nonfinite_guards() -> dict[str, Any]:
    cfg = FactorizedGRPOConfig()
    try:
        decision_grpo_loss(torch.tensor([float("nan")]), torch.tensor([-1.0]), 1.0, config=cfg)
    except FloatingPointError:
        nan_blocked = True
    else:
        nan_blocked = False
    if not nan_blocked:
        raise AssertionError("NaN new logprob was not rejected")

    try:
        aggregate_role_loss([
            WeightedDecisionLoss(torch.tensor(float("inf")), "g", "t0", 1.0),
            WeightedDecisionLoss(torch.tensor(0.0), "g", "t1", 1.0),
        ])
    except FloatingPointError:
        inf_blocked = True
    else:
        inf_blocked = False
    if not inf_blocked:
        raise AssertionError("Inf decision loss was not rejected")
    return {"nan_logprob_blocked": True, "inf_loss_blocked": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU gradient audit for factorized trajectory-GRPO learner")
    ap.parse_args()

    torch.manual_seed(0)
    report = {
        "causal_alignment": audit_causal_alignment(),
        "ratio_and_clipping_gradients": audit_ratio_and_clipping_gradients(),
        "actual_parameter_step_direction": audit_actual_parameter_step_direction(),
        "trajectory_weighting": audit_trajectory_weighting(),
        "adapter_isolation": audit_adapter_isolation(),
        "sampled_reverse_kl": audit_kl(),
        "old_policy_detached": audit_old_policy_is_detached(),
        "nonfinite_guards": audit_nonfinite_guards(),
        "status": "PASS",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
