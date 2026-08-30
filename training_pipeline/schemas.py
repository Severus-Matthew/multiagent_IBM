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

# Public capabilities of the live Twin. This is a tool schema, not a
# scenario-specific candidate list, and is therefore safe to expose to RCA.
INJECTIBLE_FAULT_MECHANISMS = {
    "assign_to_non_existent_node": "infra_failure",
    "delete_pod": "infra_failure",
    "scale_replicas_zero": "infra_failure",
    "container_kill": "infra_failure",
    "network_delay": "latency_degradation",
    "network_loss": "network_failure",
    "cpu_stress": "resource_exhaustion",
    "memory_stress": "resource_exhaustion",
    "mongodb_auth_missing": "auth_failure",
    "mongodb_auth_revoked": "auth_failure",
    "target_port_misconfig": "config_error",
    "application_config_misconfig": "config_error",
}

_FAULT_FAMILY_MECHANISM_MARKERS = (
    ("assign_to_non_existent_node", "assign_to_non_existent_node"),
    ("assign_non_existent_node", "assign_to_non_existent_node"),
    ("scale_pod_zero", "scale_replicas_zero"),
    ("scale_replicas_zero", "scale_replicas_zero"),
    ("k8s_target_port_misconfig", "target_port_misconfig"),
    ("target_port_misconfig", "target_port_misconfig"),
    ("auth_miss_mongodb", "mongodb_auth_missing"),
    ("revoke_auth_mongodb", "mongodb_auth_revoked"),
    ("mongodb_auth_missing", "mongodb_auth_missing"),
    ("mongodb_auth_revoked", "mongodb_auth_revoked"),
    ("network_delay", "network_delay"),
    ("network_loss", "network_loss"),
    ("container_kill", "container_kill"),
    ("delete_pod", "delete_pod"),
    ("cpu_stress", "cpu_stress"),
    ("memory_stress", "memory_stress"),
    ("misconfig_app", "application_config_misconfig"),
    ("application_config_misconfig", "application_config_misconfig"),
)


def infer_fault_mechanism(fault_family: str | None) -> str:
    """Map evaluator-side AIOpsLab family identifiers to public replay tools."""
    family = str(fault_family or "").strip().lower().replace("-", "_")
    for marker, mechanism in _FAULT_FAMILY_MECHANISM_MARKERS:
        if marker in family:
            return mechanism
    return ""


@dataclass(frozen=True)
class FaultLabel:
    service: str
    fault_type: str
    fault_family: str = ""
    variant_name: str = "default"
    fault_mechanism: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_key(self) -> str:
        return f"{self.service}::{normalize_fault_type(self.fault_type or self.fault_family)}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def injection_key(self) -> str:
        mechanism = normalize_fault_mechanism(self.fault_mechanism)
        return f"{self.service}::{mechanism}::{self.variant_name}"

    def is_injectible(self) -> bool:
        mechanism = normalize_fault_mechanism(self.fault_mechanism)
        return (
            mechanism in INJECTIBLE_FAULT_MECHANISMS
            and INJECTIBLE_FAULT_MECHANISMS[mechanism]
            == normalize_fault_type(self.fault_type)
        )


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


def normalize_fault_mechanism(text: str | None) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in INJECTIBLE_FAULT_MECHANISMS else ""


def parse_fault_lines(text: str) -> list[FaultLabel]:
    """Parse RCA lines.

    The live contract is ``service::fault_type::mechanism[::variant]``. The
    historical two-field form remains parseable for offline audits, but yields
    a deliberately non-injectible label rather than guessing a mechanism.
    """
    labels: list[FaultLabel] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`").strip()
        if not line or line.startswith("#") or "::" not in line:
            continue
        parts = [x.strip() for x in line.split("::")]
        if len(parts) < 2 or len(parts) > 4:
            continue
        service, fault_type = parts[:2]
        mechanism = normalize_fault_mechanism(parts[2]) if len(parts) >= 3 else ""
        variant = parts[3] if len(parts) == 4 and parts[3] else "default"
        if not service:
            continue
        label = FaultLabel(
            service=service,
            fault_type=normalize_fault_type(fault_type),
            fault_mechanism=mechanism,
            variant_name=variant,
        )
        key = label.injection_key() if mechanism else label.canonical_key()
        if key not in seen:
            labels.append(label)
            seen.add(key)
    return labels


def approx_token_count(text: str) -> int:
    return max(1, int(len(str(text or "").split()) * 1.25))
