from __future__ import annotations

"""Shared, position-independent parsing for kubectl/helm command strings.

kubectl accepts flags before or after positional arguments interchangeably
(``patch deploy -n ns name`` and ``patch deploy name -n ns`` and
``-n ns patch deploy name`` are all equivalent). Four call sites in this repo
(command_safety.py, action_reward.py, command_normalizer.py, and
digital_twin_runtime/live_action_executor.py) each independently assumed the
verb/kind/name sit at fixed indices right after the program name, and each
broke the same way once the trainable/frozen agents started writing
``-n <namespace>`` before the verb instead of after it. This module is the
one place that logic lives now; the four call sites use it instead of their
own copies.
"""

import shlex

# The only flag observed (so far) placed before the verb/resource in real
# rollouts. Kept narrow deliberately: anything else starting with "-" is
# still skipped as a single self-contained token (covers "=" forms like
# --type=json, -p='...'), it just isn't assumed to consume a following value.
FLAGS_WITH_SEPARATE_VALUE = {"-n", "--namespace"}


def split_command(cmd: str) -> list[str]:
    try:
        return shlex.split(str(cmd or ""))
    except Exception:
        return str(cmd or "").split()


def positional_args(parts: list[str], start: int) -> list[str]:
    """Flag-stripped positional tokens from ``start`` onward, in order."""
    positional: list[str] = []
    index = start
    while index < len(parts):
        token = parts[index]
        if token.startswith("-"):
            if token in FLAGS_WITH_SEPARATE_VALUE:
                index += 2
                continue
            index += 1
            continue
        positional.append(token)
        index += 1
    return positional


def positional_indices(parts: list[str], start: int) -> list[int]:
    """Like ``positional_args`` but returns indices into ``parts`` instead of
    values, so a caller can slice the *original* list (flags included) from
    a positional token onward — needed by callers that scan trailing flags
    themselves (e.g. a `-l`/`--selector` value) rather than just reading the
    verb/kind/name.
    """
    indices: list[int] = []
    index = start
    while index < len(parts):
        token = parts[index]
        if token.startswith("-"):
            if token in FLAGS_WITH_SEPARATE_VALUE:
                index += 2
                continue
            index += 1
            continue
        indices.append(index)
        index += 1
    return indices


def program_verb(parts: list[str]) -> str:
    """The verb for a ``<program> ... <verb> ...`` command, e.g. "patch" for
    kubectl or "rollback" for helm — the first positional token after
    ``parts[0]``, wherever flags placed it."""
    positional = positional_args(parts, 1)
    return positional[0] if positional else ""


def resource_target(positional: list[str]) -> tuple[str, str] | None:
    """Resolve (kind, name) from ``positional`` — the flag-stripped tokens
    after the program name, as returned by ``positional_args(parts, 1)``.
    """
    if not positional:
        return None
    verb = positional[0]
    kind_index = 2 if verb == "rollout" else 1
    if len(positional) <= kind_index:
        return None
    token = positional[kind_index]
    if "/" in token:
        kind, name = token.split("/", 1)
        return kind.lower(), name
    name = positional[kind_index + 1] if len(positional) > kind_index + 1 else ""
    return token.lower(), name


def namespace_flag(parts: list[str]) -> str | None:
    for index, part in enumerate(parts):
        if part in {"-n", "--namespace"} and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith("--namespace="):
            return part.split("=", 1)[1]
    return None
