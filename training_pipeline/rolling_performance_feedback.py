from __future__ import annotations

"""Cross-trajectory performance feedback for the RCA/Action prompt policies.

Per-trajectory retry feedback (``previous_attempts``) only tells a policy what
happened on *this* incident. It has no way to learn, for example, that
``--type=merge`` null-out patches are being rejected by the command-safety
layer across many different incidents, or that a particular injectible
mechanism is rarely verifying. This module aggregates exactly that: a rolling
window of recent, already-computed reward/verifier signals, summarized into a
small, human-readable dict the prompt-building functions can fold into the
next rollout's prompt.

Every field here is derived from information already public to the policy
(mechanism names, safety-check reasons, verification booleans/scores) or
self-referential to the agent's own past outputs. Nothing here reads
``full_state``, ground truth, or any oracle field — the tracker only ever
receives ``reward_components``/``verifier_result`` dicts, which are themselves
already redacted before they reach here.
"""

from collections import Counter, deque
from typing import Any


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


class RollingPerformanceTracker:
    """Bounded-memory rolling summary of recent RCA/Action attempts.

    Not persisted across a process restart; a fresh tracker simply rebuilds
    over the first few batches after a resume, which is an acceptable cost for
    a feedback signal that is inherently approximate.
    """

    def __init__(self, window_size: int = 200, min_samples_for_summary: int = 8) -> None:
        self.window_size = int(window_size)
        self.min_samples = int(min_samples_for_summary)
        self._rca: deque[dict[str, Any]] = deque(maxlen=self.window_size)
        self._action: deque[dict[str, Any]] = deque(maxlen=self.window_size)

    def record_rca_attempt(self, reward_components: dict[str, Any]) -> None:
        # Deliberately reads only fields already safe for agent/prompt use
        # (twin verification outcome, format validity) — never
        # `matches_private_evaluator` or anything else compared against the
        # hidden label, even though a value inside it would be harmless on its
        # own; the tracker only touches keys that are safe by construction.
        c = reward_components or {}
        self._rca.append({
            "verified": bool(c.get("rca_twin_verified")),
            "twin_score": float(c.get("twin_reproduction_score") or 0.0),
            "invalid_format": bool(c.get("invalid_format")),
            "repeated_wrong_guess": bool(c.get("repeated_wrong_guess")),
            "iteration_index": int(c.get("iteration_index") or 0),
        })

    def record_action_attempt(
        self,
        reward: float,
        reward_components: dict[str, Any],
        verifier_result: dict[str, Any] | None = None,
    ) -> None:
        c = reward_components or {}
        v = verifier_result or {}
        rejection = None
        execution = v.get("execution") if isinstance(v, dict) else None
        if isinstance(execution, dict) and execution.get("rejection_reasons"):
            rejection = str(execution["rejection_reasons"][0])
        self._action.append({
            "safe": bool(c.get("safe")),
            "has_mutation": bool(c.get("has_mutating_command")),
            "resolved": bool(v.get("resolved")) if isinstance(v, dict) else False,
            "reason": str(v.get("reason") or "") if isinstance(v, dict) else "",
            "rejection_reason": rejection,
            "reward": float(reward or 0.0),
        })

    def rca_summary(self) -> dict[str, Any] | None:
        rows = list(self._rca)
        if len(rows) < self.min_samples:
            return None
        n = len(rows)
        verified = sum(1 for r in rows if r["verified"])
        return {
            "window_size": n,
            "verified_rate": _rate(verified, n),
            "mean_twin_reproduction_score": round(sum(r["twin_score"] for r in rows) / n, 4),
            "invalid_format_rate": _rate(sum(1 for r in rows if r["invalid_format"]), n),
            "repeated_wrong_guess_rate": _rate(sum(1 for r in rows if r["repeated_wrong_guess"]), n),
            "note": "Aggregate over your own last attempts across many incidents; not evidence about this incident specifically.",
        }

    def action_summary(self) -> dict[str, Any] | None:
        rows = list(self._action)
        if len(rows) < self.min_samples:
            return None
        n = len(rows)
        rejection_counts: Counter[str] = Counter(r["rejection_reason"] for r in rows if r["rejection_reason"])
        reason_counts: Counter[str] = Counter(r["reason"] for r in rows if r["reason"])
        return {
            "window_size": n,
            "resolved_rate": _rate(sum(1 for r in rows if r["resolved"]), n),
            "safe_rate": _rate(sum(1 for r in rows if r["safe"]), n),
            "has_mutation_rate": _rate(sum(1 for r in rows if r["has_mutation"]), n),
            "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
            "top_command_rejection_reasons": dict(rejection_counts.most_common(5)),
            "top_verifier_outcomes": dict(reason_counts.most_common(5)),
            "note": "Aggregate over your own last attempts across many incidents; not evidence about this incident specifically.",
        }
