from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from training_pipeline.command_safety import check_command_safety
from training_pipeline.kubectl_command_shape import (
    namespace_flag as _namespace,
    positional_args as _positional_args,
    resource_target as _resource_target,
)

from .sparse_live_session import SparseLiveTwinSession


@dataclass
class CommandExecution:
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveActionExecutionResult:
    safe: bool
    executed: bool
    namespace: str
    commands: list[CommandExecution] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "executed": self.executed,
            "namespace": self.namespace,
            "commands": [row.to_dict() for row in self.commands],
            "rejection_reasons": self.rejection_reasons,
        }


def execute_twin_commands(
    session: SparseLiveTwinSession,
    commands: list[str],
    *,
    timeout_seconds: float = 150.0,
    owned_runtime_objects: list[dict[str, str]] | None = None,
) -> LiveActionExecutionResult:
    """Execute preflighted commands without a shell in one owned Twin only."""
    safety = check_command_safety(commands)
    reasons = [
        pattern
        for row in safety.get("unsafe", [])
        for pattern in row.get("patterns", [])
    ]
    selected = {
        str(row.get("name")) for row in session.bundle.object_refs
        if row.get("kind") in {"Deployment", "StatefulSet", "Service"}
    }
    runtime_refs = {
        (
            str(row.get("kind") or "").strip().lower(),
            str(row.get("name") or "").strip(),
        )
        for row in (owned_runtime_objects or [])
        if row.get("kind") and row.get("name")
    }
    parsed: list[tuple[str, list[str]]] = []
    for command in commands:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            reasons.append(f"parse_error:{exc}")
            continue
        if not parts or parts[0] != "kubectl":
            reasons.append("live_executor_supports_kubectl_only")
            continue
        if _namespace(parts) != session.namespace:
            reasons.append("command_namespace_must_equal_owned_twin")
        positional = _positional_args(parts, 1)
        target = _resource_target(positional)
        verb = positional[0] if positional else ""
        if verb == "rollout" and (len(positional) < 2 or positional[1] != "status"):
            reasons.append("portable_repairs_require_patch_scale_or_owned_chaos_delete")
        if verb in {"patch", "scale"}:
            kinds = {"deploy", "deployment", "deployments", "sts", "statefulset", "statefulsets", "svc", "service", "services"}
            expected_positions = 2 if len(positional) > 1 and "/" in positional[1] else 3
            if (not target or target[0] not in kinds or not target[1] or target[1] not in selected
                    or len(positional) != expected_positions):
                reasons.append("mutation_target_not_selected_exact_resource")
        if verb == "delete":
            target_kind = str(target[0] if target else "").strip().lower()
            target_name = str(target[1] if target else "").strip()
            if (target_kind, target_name) not in runtime_refs:
                # Selector-based or arbitrary deletion is intentionally not
                # enabled. The only deletions allowed here are exact runtime
                # fault resources registered by the live verifier.
                reasons.append("delete_target_not_owned_runtime_fault_resource")
        parsed.append((command, parts))

    if reasons:
        return LiveActionExecutionResult(
            safe=False, executed=False, namespace=session.namespace,
            rejection_reasons=sorted(set(reasons)),
        )
    rows: list[CommandExecution] = []
    for command, parts in parsed:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                parts, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=timeout_seconds,
            )
            rows.append(CommandExecution(
                command=command, returncode=proc.returncode,
                stdout=proc.stdout[-8000:], stderr=proc.stderr[-8000:],
                elapsed_seconds=round(time.monotonic() - started, 3),
            ))
            if proc.returncode != 0:
                break
        except subprocess.TimeoutExpired as exc:
            rows.append(CommandExecution(
                command=command, returncode=124,
                stdout=str(exc.stdout or "")[-8000:], stderr="command timeout",
                elapsed_seconds=round(time.monotonic() - started, 3),
            ))
            break
    return LiveActionExecutionResult(
        safe=True,
        executed=bool(rows) and all(row.returncode == 0 for row in rows),
        namespace=session.namespace,
        commands=rows,
    )
