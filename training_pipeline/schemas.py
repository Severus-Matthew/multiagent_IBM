from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

FAULT_TYPE_ALIASES = {
    "assign_non_existent_node": "infra_failure",
    "assign_to_non_existent_node": "infra_failure",
    "pod_failure": "infra_failure",
    "pod_kill": "infra_failure",
    "container_kill": "infra_failure",
    "scale_pod": "infra_failure",
    "cpu": "resource_exhaustion",
    "memory": "resource_exhaustion",
    "oom": "resource_exhaustion",
    "auth": "auth_failure",
    "revoke_auth": "auth_failure",
    "auth_miss_mongodb": "auth_failure",
    "mongodb": "dependency_failure",
    "mongo": "dependency_failure",
    "network_delay": "latency_degradation",
    "latency": "latency_degradation",
    "delay": "latency_degradation",
    "network_loss": "network_failure",
    "k8s_target_port": "config_error",
    "misconfig": "config_error",
    "config": "config_error",
}

CANONICAL_FAULT_TYPES = {
    "auth_failure", "dependency_failure", "infra_failure", "latency_degradation",
    "config_error", "network_failure", "resource_exhaustion", "multifault", "unknown",
}


@dataclass(frozen=True)
class FaultLabel:
    service: str
    fault_type: str
    fault_family: str = ""
    variant_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_key(self) -> str:
        return f"{self.service}::{normalize_fault_type(self.fault_type or self.fault_family)}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RCAAttempt:
    iteration: int
    instruction: str
    prediction_text: str
    predicted_faults: list[FaultLabel]
    reward: float
    reward_components: dict[str, Any]
    success: bool
    feedback: str
    token_counts: dict[str, int] = field(default_factory=dict)
    group_id: str | None = None
    selected_sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["predicted_faults"] = [f.to_dict() for f in self.predicted_faults]
        return out


@dataclass
class ActionAttempt:
    iteration: int
    instruction_prompt: str
    commands: list[str]
    reward: float
    reward_components: dict[str, Any]
    success: bool
    feedback: str
    execution_result: dict[str, Any] = field(default_factory=dict)
    verifier_result: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GRPORolloutSample:
    """One trainable-policy decision sample for trajectory-level GRPO.

    ``completion`` is the trainable policy output (the prompt/instruction), not
    the fixed downstream agent's answer.

    Real optimization must record the *exact rollout-time tokenization* for both
    the model input prompt and generated completion.  Retokenizing stored text at
    update time is not acceptable: chat-template/special-token boundaries can
    differ and would invalidate token alignment and the old/new importance ratio.
    ``old_logprobs`` must contain one value for every ``completion_token_ids``
    entry under the rollout policy that generated the sample.
    """
    stage: str
    scenario_id: str
    group_id: str
    sample_id: str
    sample_index: int
    iteration: int
    policy_role: str
    policy_prompt: str
    completion: str
    completion_tokens: int
    old_logprob_sum: float | None
    old_logprobs: list[float] | None
    reward: float
    reward_components: dict[str, Any]
    advantage: float | None
    group_reward_mean: float | None
    group_reward_std: float | None
    solver_prediction: str
    parsed_prediction: list[dict[str, Any]]
    success: bool
    terminal: bool
    model_name: str
    policy_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    ref_logprobs: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        info = ((out.get("metadata") or {}).get("policy_info") or {})
        if isinstance(info, dict):
            # The policy object is the authoritative source for exact rollout-time
            # tokenization.  Hoist those fields into the canonical optimizer row so
            # the learner never has to inspect implementation-specific metadata.
            for field_name in (
                "prompt_token_ids",
                "completion_token_ids",
                "old_logprobs",
                "old_logprob_sum",
                "ref_logprobs",
            ):
                if out.get(field_name) is None and info.get(field_name) is not None:
                    out[field_name] = info.get(field_name)

            # When an exact-token policy records the literal prompt text, it must
            # match the rollout row's policy_prompt byte-for-byte.  Fail here rather
            # than permit a silently invalid old/new log-probability ratio later.
            prompt_text = info.get("prompt_text")
            if prompt_text is not None and str(prompt_text) != str(out.get("policy_prompt") or ""):
                raise ValueError(
                    "policy_info.prompt_text does not match GRPO policy_prompt; "
                    "exact-token replay would be invalid"
                )
        return out


def normalize_fault_type(text: str | None) -> str:
    value = str(text or "").strip().lower()
    if not value:
        return "unknown"
    if value in CANONICAL_FAULT_TYPES:
        return value
    for pattern, canonical in FAULT_TYPE_ALIASES.items():
        if pattern in value:
            return canonical
    return value if value in CANONICAL_FAULT_TYPES else "unknown"


def parse_fault_lines(text: str) -> list[FaultLabel]:
    """Parse strict RCA output lines: service::fault_type."""
    labels: list[FaultLabel] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`").strip()
        if not line or line.startswith("#") or "::" not in line:
            continue
        service, fault_type = [x.strip() for x in line.split("::", 1)]
        if not service:
            continue
        label = FaultLabel(service=service, fault_type=normalize_fault_type(fault_type))
        key = label.canonical_key()
        if key not in seen:
            labels.append(label)
            seen.add(key)
    return labels


def approx_token_count(text: str) -> int:
    return max(1, int(len(str(text or "").split()) * 1.25))
