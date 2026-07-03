"""
agents/rca_agent.py

RCA Agent — fixed LLM (Claude / GPT) that reads a state abstraction and
outputs the root cause of the fault in structured JSON.

The agent uses algorithmic pre-screening from state_abstraction_full as
strong hints but is free to refute them with evidence from logs/traces.

Output schema:
    {
        "root_cause_service": str,
        "fault_type": str,         # auth_failure | dependency_failure | infra_failure
                                   # | latency_degradation | config_error | pod_failure
        "explanation": str,
        "affected_services": list[str],
        "confidence": float        # 0.0 – 1.0
    }

Usage:
    from agents.rca_agent import RCAAgent
    from agents.llm_client import LLMClient

    agent = RCAAgent()                                  # default: Claude
    agent = RCAAgent(client=LLMClient("openai"))        # GPT variant

    rca_result = agent.analyze(state, compressed)
    print(rca_result["root_cause_service"], rca_result["confidence"])
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.llm_client import LLMClient
from state_abstraction_full.agent_input_builder import build_rca_agent_input

_SYSTEM_PROMPT = """\
You are an expert SRE (Site Reliability Engineer) specializing in microservice \
fault diagnosis on Kubernetes.

You will be given telemetry from a faulted microservice deployment: service health \
signals, log error patterns, distributed trace edge anomalies, metric z-scores, \
and an algorithmic pre-screening result.

Your job is to identify the ROOT CAUSE — the single service and fault type that \
initiated the failure cascade. The pre-screening hints are algorithmic — verify \
them against the evidence and override if the data says otherwise.

Respond ONLY with a JSON object matching this exact schema:
{
    "root_cause_service": "<service_name>",
    "fault_type": "<auth_failure|dependency_failure|infra_failure|latency_degradation|config_error|pod_failure>",
    "explanation": "<concise explanation citing specific evidence from the telemetry>",
    "affected_services": ["<downstream_svc1>", "<downstream_svc2>"],
    "confidence": <0.0–1.0>
}

Do not include any prose before or after the JSON.\
"""

_VALID_FAULT_TYPES = {
    "auth_failure",
    "dependency_failure",
    "infra_failure",
    "latency_degradation",
    "config_error",
    "pod_failure",
    "misconfiguration",
    "network_failure",
    "resource_exhaustion",
}


class RCAAgent:
    """
    Root Cause Analysis Agent.

    Uses a fixed LLM to analyze state abstraction telemetry and identify
    the root cause service and fault type.
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient(provider="claude")

    def analyze(
        self,
        state: dict,
        compressed: dict | None = None,
        max_tokens: int = 1500,
    ) -> dict:
        """
        Run RCA on a scenario state.

        Args:
            state:      full state_abstraction.json dict
            compressed: state_abstraction_compressed.json dict (adds cluster context)
            max_tokens: LLM response budget

        Returns:
            Validated RCA result dict:
            {root_cause_service, fault_type, explanation, affected_services, confidence}
        """
        rca_input = build_rca_agent_input(state, compressed)
        prompt = rca_input["prompt"]

        print(
            f"[RCAAgent] Analyzing scenario={state.get('scenario_id')}  "
            f"model={self.client.model}"
        )

        result = self.client.call_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        return self._validate_and_normalize(result, state)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_and_normalize(self, raw: dict, state: dict) -> dict:
        """
        Validate the LLM's JSON output and fill missing fields with
        algorithmic fallbacks so the rest of the pipeline always gets
        a complete dict.
        """
        if not isinstance(raw, dict):
            raw = {}

        # Fallback: use algorithmic hint if LLM omitted root_cause_service
        algo_hint = (
            state.get("rca_features", {}).get("most_suspicious_service", "")
        )
        fault_ctx = state.get("fault_context", {})

        root_cause = (
            raw.get("root_cause_service")
            or algo_hint
            or fault_ctx.get("faulty_service", "unknown")
        )

        fault_type = raw.get("fault_type", "")
        if fault_type not in _VALID_FAULT_TYPES:
            # Try to infer from fault_family
            family = fault_ctx.get("fault_family", "")
            fault_type = _map_family_to_type(family) or "dependency_failure"

        affected = raw.get("affected_services", [])
        if not isinstance(affected, list):
            affected = []

        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        explanation = raw.get("explanation", "No explanation provided.")

        result = {
            "root_cause_service": root_cause,
            "fault_type": fault_type,
            "explanation": explanation,
            "affected_services": affected,
            "confidence": confidence,
            # Metadata for logging
            "_scenario_id": state.get("scenario_id"),
            "_algo_hint": algo_hint,
            "_llm_raw": raw,
        }

        print(
            f"[RCAAgent] root_cause={root_cause}  "
            f"fault_type={fault_type}  "
            f"confidence={confidence:.2f}"
        )
        return result


def _map_family_to_type(family: str) -> str:
    """Map AIOpsLab fault_family strings to our normalized fault_type set."""
    family = (family or "").lower()
    mapping = {
        "auth_miss_mongodb":  "auth_failure",
        "revoke_auth":        "auth_failure",
        "pod_failure":        "pod_failure",
        "pod_kill":           "pod_failure",
        "container_kill":     "pod_failure",
        "network_delay":      "latency_degradation",
        "network_loss":       "network_failure",
        "misconfig_app":      "config_error",
        "k8s_target_port":    "config_error",
        "scale_pod":          "infra_failure",
        "assign_non_existent_node": "infra_failure",
    }
    for key, val in mapping.items():
        if key in family:
            return val
    return ""
