"""
digital_twin/reward.py

Computes the scalar reward (0.0–1.0) that drives GRPO training of the
Prompt Optimization Agent.

Reward semantics
----------------
  1.0   All tracked relevant pods are Ready.          → FULL PASS, stop iterating
  0.x   Partial recovery: some relevant pods recovered.
  0.0   No recovery, timeout, or post-health snapshot is empty.

The *relevant* pod set comes from digital_twin_filter (services with a fault
signal). Bystander pods — already deleted from the twin — are excluded.

Partial reward formula
----------------------
  base   = (post_ready_relevant / total_relevant)
  bonus  = 0.1 × (newly_recovered / total_relevant)   ← extra credit for progress
  reward = min(1.0, base + bonus)

The bonus rewards iterations that make *forward progress* even if the solution
is incomplete. This gives the GRPO trainer a useful gradient signal rather
than a binary 0/1 on partial fixes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digital_twin.builder import DigitalTwin


def compute_reward(
    twin: "DigitalTwin",
    pre_health: dict[str, bool],
    post_health: dict[str, bool],
) -> float:
    """
    Compute a scalar reward from pod health snapshots taken before and
    after solution execution.

    Args:
        twin:        the DigitalTwin — supplies the relevant pod set
        pre_health:  {pod_name: is_ready} BEFORE commands executed
        post_health: {pod_name: is_ready} AFTER commands + stabilization wait

    Returns:
        float in [0.0, 1.0]
    """
    relevant_pods = set(twin.relevant_pod_names())

    # Fallback: if no pod-level relevance info, score all pods in namespace
    if not relevant_pods:
        if not post_health:
            return 0.0
        ready_count = sum(1 for v in post_health.values() if v)
        return ready_count / len(post_health)

    # Intersect with pods that actually appear in the snapshot
    tracked = relevant_pods & set(post_health.keys())
    if not tracked:
        return 0.0

    # Full pass
    if all(post_health[p] for p in tracked):
        return 1.0

    # Partial credit
    post_ready = sum(1 for p in tracked if post_health[p])
    newly_recovered = sum(
        1 for p in tracked
        if not pre_health.get(p, True) and post_health.get(p, False)
    )

    base = post_ready / len(tracked)
    bonus = 0.1 * (newly_recovered / len(tracked))
    return min(1.0, base + bonus)


def reward_summary(
    pre_health: dict[str, bool],
    post_health: dict[str, bool],
    relevant_pods: list[str],
) -> dict:
    """
    Human-readable breakdown of the reward computation.
    Returned in ExecutionResult.reward_breakdown for logging and debugging.
    """
    rel = set(relevant_pods)
    tracked = rel & set(post_health.keys())

    pre_ready_set = {p for p in tracked if pre_health.get(p, False)}
    post_ready_set = {p for p in tracked if post_health.get(p, False)}
    newly_recovered = post_ready_set - pre_ready_set
    regressed = pre_ready_set - post_ready_set

    return {
        "relevant_pods_tracked": sorted(tracked),
        "pre_ready": sorted(pre_ready_set),
        "post_ready": sorted(post_ready_set),
        "newly_recovered": sorted(newly_recovered),
        "regressed": sorted(regressed),
        "counts": {
            "total_tracked": len(tracked),
            "pre_ready": len(pre_ready_set),
            "post_ready": len(post_ready_set),
            "newly_recovered": len(newly_recovered),
            "regressed": len(regressed),
        },
    }
