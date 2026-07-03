"""
digital_twin/executor.py

Executes solution commands (kubectl / helm / mongosh-via-exec) produced
by the Action Agent on a live digital twin and measures pod health recovery.

Usage:
    from digital_twin.executor import SolutionExecutor

    executor = SolutionExecutor(twin)
    result = executor.run([
        "kubectl rollout restart deployment/user -n test-social-network",
        "kubectl scale deployment/url-shorten-service --replicas=1 -n test-social-network",
    ])
    print(f"reward={result.reward:.2f}  passed={result.passed}")
    for cr in result.commands:
        print(f"  [{cr.returncode}] {cr.command[:60]}")
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digital_twin.builder import DigitalTwin


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class CommandResult:
    """Outcome of a single command execution."""
    command: str
    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "stdout": self.stdout[:500],
            "stderr": self.stderr[:500],
            "returncode": self.returncode,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "ok": self.ok,
        }


@dataclass
class ExecutionResult:
    """Aggregated result of running all solution commands on the twin."""
    commands: list[CommandResult]
    pre_health: dict[str, bool]
    post_health: dict[str, bool]
    reward: float
    elapsed_seconds: float
    passed: bool
    reward_breakdown: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "reward": round(self.reward, 4),
            "passed": self.passed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "commands_run": len(self.commands),
            "commands_ok": sum(1 for c in self.commands if c.ok),
            "pre_healthy": sum(1 for v in self.pre_health.values() if v),
            "post_healthy": sum(1 for v in self.post_health.values() if v),
            "reward_breakdown": self.reward_breakdown,
            "commands": [c.as_dict() for c in self.commands],
        }


# ── Executor ──────────────────────────────────────────────────────────────────


class SolutionExecutor:
    """
    Runs action-agent-produced commands on a live DigitalTwin and scores
    the result using pod health before / after execution.

    Safety:
        Commands matching _BLOCKED_PATTERNS are rejected without execution.
        This protects the cluster from catastrophic actions like namespace
        deletion or helm uninstall that would break subsequent iterations.
    """

    # Patterns that are never safe to execute on the digital twin.
    # The twin is shared across iterations; destroying it breaks the loop.
    _BLOCKED_PATTERNS: list[str] = [
        r"kubectl\s+delete\s+namespace\b",
        r"kubectl\s+(delete|rm)\s+.*\s+--all\b",
        r"\bhelm\s+uninstall\b",
        r"\bkind\s+delete\b",
        r"\bkubectl\s+delete\s+cluster\b",
        r"\brm\s+-rf\b",
    ]

    def __init__(
        self,
        twin: "DigitalTwin",
        timeout_per_cmd: int = 60,
    ):
        self.twin = twin
        self.timeout_per_cmd = timeout_per_cmd
        self._context: str = self._kubectl_context()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        commands: list[str],
        stabilize_timeout: int = 120,
    ) -> ExecutionResult:
        """
        Execute all solution commands then measure pod health recovery.

        Args:
            commands: command strings — one kubectl/helm/exec command per element.
                      Blank lines and comment lines (# ...) are skipped.
            stabilize_timeout: seconds to wait for pods to reach Ready after
                               all commands complete.

        Returns:
            ExecutionResult with reward in [0.0, 1.0] and per-command logs.
        """
        t0 = time.time()
        pre_health = self.twin.pod_health_snapshot()

        cmd_results: list[CommandResult] = []
        for raw_cmd in commands:
            cmd = raw_cmd.strip()
            if not cmd or cmd.startswith("#"):
                continue
            result = self._execute_one(cmd)
            cmd_results.append(result)
            if not result.ok:
                print(
                    f"[Executor] ✗ rc={result.returncode}: "
                    f"{cmd[:80]}"
                    + (f"\n  stderr: {result.stderr[:120]}" if result.stderr else "")
                )

        self._wait_for_stable(stabilize_timeout)
        post_health = self.twin.pod_health_snapshot()

        from digital_twin.reward import compute_reward, reward_summary
        reward = compute_reward(self.twin, pre_health, post_health)
        breakdown = reward_summary(pre_health, post_health, self.twin.relevant_pod_names())

        return ExecutionResult(
            commands=cmd_results,
            pre_health=pre_health,
            post_health=post_health,
            reward=reward,
            elapsed_seconds=time.time() - t0,
            passed=reward >= 1.0,
            reward_breakdown=breakdown,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_one(self, cmd: str) -> CommandResult:
        """
        Execute a single command with safety checks and timeout.
        Injects the correct kubectl context if not already specified.
        """
        block_reason = self._check_blocked(cmd)
        if block_reason:
            return CommandResult(
                command=cmd,
                stdout="",
                stderr=f"BLOCKED: {block_reason}",
                returncode=1,
                elapsed_ms=0.0,
            )

        final_cmd = self._inject_context(cmd)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                final_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_per_cmd,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CommandResult(
                command=final_cmd,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                returncode=proc.returncode,
                elapsed_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CommandResult(
                command=final_cmd,
                stdout="",
                stderr=f"TIMEOUT after {self.timeout_per_cmd}s",
                returncode=124,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CommandResult(
                command=final_cmd,
                stdout="",
                stderr=str(e),
                returncode=1,
                elapsed_ms=elapsed_ms,
            )

    def _check_blocked(self, cmd: str) -> str:
        """Return a reason string if the command is blocked, else empty string."""
        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return f"matches unsafe pattern '{pattern}'"
        return ""

    def _inject_context(self, cmd: str) -> str:
        """
        Ensure kubectl commands use the correct cluster context.
        If --context is already present, the command is left unchanged.
        Helm and other commands are passed through unchanged.
        """
        if not cmd.startswith("kubectl"):
            return cmd
        if "--context" in cmd:
            return cmd
        return cmd.replace("kubectl ", f"kubectl --context {self._context} ", 1)

    def _wait_for_stable(self, timeout: int) -> None:
        """
        Wait for all pods in the twin's namespace to reach Ready.
        Logs a warning on timeout rather than raising, so the reward can
        still be computed from a partial-recovery snapshot.
        """
        try:
            from aiopslab.service.kubectl import KubeCtl
            KubeCtl().wait_for_ready(self.twin.namespace, max_wait=timeout)
        except Exception as e:
            print(f"[Executor] stabilize timeout: {e}")

    @staticmethod
    def _kubectl_context() -> str:
        cluster = os.environ.get("AIOPSLAB_CLUSTER", "kind")
        return f"kind-{cluster}"
