from __future__ import annotations

"""Utilities for safely training one LoRA role adapter at a time.

The IBM joint pipeline keeps one frozen causal-LM base with two independent LoRA
adapters: ``lora_rca`` and ``lora_action``. During one role optimizer step the
selected adapter is active and trainable; the shared base and the other role
adapter must remain frozen.

These helpers deliberately enforce that invariant instead of relying on implicit
PEFT ``requires_grad`` state, which can change when adapters are added/switched.
"""

from typing import Any

import torch


ROLE_ADAPTERS = {"lora_rca", "lora_action"}


def parameter_belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
    """Return True when ``adapter_name`` is an explicit path segment."""
    return str(adapter_name) in str(parameter_name).split(".")


def _clear_all_gradients(model: Any) -> None:
    """Remove every stale gradient tensor before changing optimization roles.

    This must clear gradients for *all* parameters, not only the parameters that
    will belong to the next optimizer. Otherwise an RCA gradient can remain
    attached to a now-frozen RCA tensor while the Action role is active, making
    gradient-isolation audits ambiguous and creating a real risk if a later
    optimizer is accidentally constructed over the wrong parameter set.
    """
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    else:
        for parameter in model.parameters():
            parameter.grad = None

    leaked = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if leaked:
        raise AssertionError(f"failed to clear stale gradients before adapter switch: {leaked[:20]}")


def activate_exclusive_adapter(model: Any, adapter_name: str) -> list[str]:
    """Activate exactly one PEFT adapter and make only its tensors trainable.

    Role switching is an optimization boundary. Every gradient from the previous
    role is first cleared globally, then the requested adapter is activated, and
    finally every non-selected tensor (including the shared base and the other
    role adapter) is forced to ``requires_grad=False``.
    """
    if adapter_name not in ROLE_ADAPTERS:
        raise ValueError(f"unsupported role adapter: {adapter_name}")
    if not hasattr(model, "set_adapter"):
        raise TypeError("model does not expose set_adapter(); PEFT adapter model required")

    # Important: optimizer.zero_grad() on the *new* role optimizer cannot clear
    # gradients left on parameters owned by the previous role because those
    # tensors are not in the new optimizer's parameter groups. Clear the complete
    # model here before changing trainability.
    _clear_all_gradients(model)

    model.set_adapter(adapter_name)
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        selected = parameter_belongs_to_adapter(name, adapter_name)
        parameter.requires_grad_(selected)
        if selected:
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(f"no parameters found for adapter {adapter_name!r}")

    leaked_trainables = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not parameter_belongs_to_adapter(name, adapter_name)
    ]
    if leaked_trainables:
        raise AssertionError(f"non-selected parameters remained trainable: {leaked_trainables[:20]}")

    # Switching adapters must leave a clean gradient state. This assertion also
    # catches PEFT/version-specific behavior that could reattach stale gradients.
    leaked_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if leaked_gradients:
        raise AssertionError(f"adapter activation left stale gradients: {leaked_gradients[:20]}")

    return trainable_names


def trainable_parameter_list(model: Any) -> list[torch.nn.Parameter]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("model has no trainable parameters")
    return params


def snapshot_named_parameters(model: Any) -> dict[str, torch.Tensor]:
    """Detached CPU snapshot for exact changed/unchanged audits."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def changed_parameter_names(
    before: dict[str, torch.Tensor],
    model: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> list[str]:
    changed: list[str] = []
    for name, parameter in model.named_parameters():
        if name not in before:
            raise KeyError(f"parameter missing from snapshot: {name}")
        current = parameter.detach().cpu()
        if not torch.allclose(before[name], current, atol=atol, rtol=rtol):
            changed.append(name)
    return changed


def finite_selected_adapter_gradients(model: Any, adapter_name: str) -> dict[str, Any]:
    selected = 0
    with_grad = 0
    nonfinite: list[str] = []
    leaked_gradients: list[str] = []

    for name, parameter in model.named_parameters():
        belongs = parameter_belongs_to_adapter(name, adapter_name)
        if belongs:
            selected += int(parameter.numel())
            if parameter.grad is not None:
                with_grad += int(parameter.numel())
                if not torch.isfinite(parameter.grad).all():
                    nonfinite.append(name)
        elif parameter.grad is not None:
            # Any gradient tensor on an inactive/base parameter is a contract
            # violation for the current role backward pass.
            leaked_gradients.append(name)

    if selected == 0:
        raise AssertionError(f"adapter {adapter_name} has no parameters")
    return {
        "adapter_name": adapter_name,
        "selected_parameter_count": selected,
        "parameters_with_gradient_count": with_grad,
        "all_selected_gradients_finite": not nonfinite,
        "nonfinite_gradient_parameters": nonfinite,
        "leaked_gradient_parameters": leaked_gradients,
        "gradient_isolation_ok": bool(with_grad > 0 and not nonfinite and not leaked_gradients),
    }
