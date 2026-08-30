from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from training_pipeline.command_safety import check_command_safety

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


def _namespace(parts: list[str]) -> str | None:
    for index, part in enumerate(parts):
        if part in {"-n", "--namespace"} and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith("--namespace="):
            return part.split("=", 1)[1]
    return None


def _resource_target(parts: list[str]) -> tuple[str, str] | None:
    if len(parts) < 3:
        return None
    verb = parts[1]
    if verb == "rollout":
        if len(parts) < 4:
            return None
        token = parts[3]
    else:
        token = parts[2]
    if "/" in token:
        kind, name = token.split("/", 1)
        return kind.lower(), name
    if len(parts) >= 4 and not parts[3].startswith("-"):
        return token.lower(), parts[3]
    return token.lower(), ""


def execute_twin_commands(
    session: SparseLiveTwinSession,
    commands: list[str],
    *,
    timeout_seconds: float = 150.0,
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
        target = _resource_target(parts)
        verb = parts[1] if len(parts) > 1 else ""
        if verb in {"patch", "scale", "delete"}:
            if not target or not target[1] or target[1] not in selected:
                # Selector-based pod deletion is intentionally not enabled in
                # the first live executor; it requires a separate owner audit.
                reasons.append("mutation_target_not_selected_exact_resource")
            if verb == "delete" and target and target[0] not in {"pod", "pods"}:
                reasons.append("live_delete_restricted_to_selected_pods")
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
