from __future__ import annotations

"""Route eligible scenarios to live replay and all others to offline rewards."""

from typing import Any

from training_pipeline.ground_truth import labels_from_full_state

from .live_capabilities import assess_live_capability


class HybridTwinVerifier:
    """Evaluator-only router; eligibility never enters either agent prompt.

    Scenario-level live vs offline routing uses the hidden label only to keep
    the scoring *environment* constant inside one GRPO incident group. The live
    path still injects the agent's predicted mechanism, never the oracle fault.
    """

    def __init__(self, live: Any, offline: Any) -> None:
        self.live = live
        self.offline = offline
        self.is_live = False
        self.route_reason = "scenario_not_prepared"

    def prepare_scenario(self, full_state: dict[str, Any], compressed_state: dict[str, Any]) -> None:
        del compressed_state
        labels = labels_from_full_state(full_state)
        assessments = [assess_live_capability(label) for label in labels]
        self.is_live = bool(labels) and all(row.get("supported") for row in assessments)
        self.route_reason = "all_fault_mechanisms_live_eligible" if self.is_live else "explicitly_excluded_from_live_reward"

    def begin_trajectory(self, trajectory_id: str | None = None) -> None:
        target = self.live if self.is_live else self.offline
        method = getattr(target, "begin_trajectory", None)
        if callable(method):
            method(trajectory_id)

    def end_trajectory(self) -> None:
        target = self.live if self.is_live else self.offline
        method = getattr(target, "end_trajectory", None)
        if callable(method):
            method()

    def validate_rca_prediction(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        target = self.live if self.is_live else self.offline
        result = dict(target.validate_rca_prediction(*args, **kwargs))
        result.update({
            "reward_route": "live" if self.is_live else "offline",
            "live_reward_eligible": self.is_live,
            "live_reward_exclusion_reason": None if self.is_live else self.route_reason,
        })
        return result

    def current_rca_gate(self, faults: list[Any]) -> dict[str, Any] | None:
        if not self.is_live:
            return None
        return self.live.current_rca_gate(faults)

    def action_namespace(self) -> str | None:
        return self.live.action_namespace() if self.is_live else None

    def apply_commands_and_score(
        self, full_state: dict[str, Any], rca_faults: list[Any], mitigation_action: dict[str, Any],
        commands: list[str], compressed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.is_live:
            result = self.live.apply_commands_and_score(
                full_state, rca_faults, mitigation_action, commands,
                compressed_state=compressed_state,
            )
        else:
            result = self.offline.apply_action_and_score(
                full_state, rca_faults, mitigation_action,
                compressed_state=compressed_state,
            )
        result = dict(result)
        result.update({"reward_route": "live" if self.is_live else "offline",
                       "live_reward_eligible": self.is_live})
        return result
