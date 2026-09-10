from __future__ import annotations

"""Lightweight audit for same-policy parallel rollout adapter replication."""

import json

import torch
from torch import nn

from .peft_adapter_control import copy_role_adapter_parameters


class _Replica(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.tensor([99.0 + offset]), requires_grad=False)
        self.block = nn.ModuleDict({
            "lora_rca": nn.Linear(2, 2, bias=False),
            "lora_action": nn.Linear(2, 2, bias=False),
        })
        with torch.no_grad():
            self.block["lora_rca"].weight.fill_(1.0 + offset)
            self.block["lora_action"].weight.fill_(2.0 + offset)


def main() -> None:
    learner = _Replica(0.0)
    worker = _Replica(10.0)
    worker_base_before = worker.base.detach().clone()

    copied = copy_role_adapter_parameters(learner, worker)
    learner_named = dict(learner.named_parameters())
    worker_named = dict(worker.named_parameters())
    adapter_names = [name for name in learner_named if "lora_" in name]

    if copied != len(adapter_names):
        raise AssertionError(f"expected {len(adapter_names)} copied tensors, got {copied}")
    if any(not torch.equal(learner_named[name], worker_named[name]) for name in adapter_names):
        raise AssertionError("rollout replica adapter differs from published learner bundle")
    if not torch.equal(worker.base, worker_base_before):
        raise AssertionError("adapter replication modified the frozen base")
    if any(worker_named[name].requires_grad for name in adapter_names):
        raise AssertionError("rollout replica adapters must remain frozen")

    print(json.dumps({
        "status": "PASS_PARALLEL_ROLLOUT_REPLICATION",
        "copied_adapter_tensors": copied,
        "frozen_base_unchanged": True,
        "replica_adapters_frozen": True,
        "same_published_policy_bundle": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
