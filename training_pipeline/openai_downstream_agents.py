from __future__ import annotations

"""Fixed OpenAI downstream RCA and remediation agents with audit-grade call logs."""

import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from .frozen_qwen_agents import _extract_command_lines, _normalize_rca_lines
from .schemas import INJECTIBLE_FAULT_MECHANISMS


class _ResponsesAgent:
    def __init__(self, *, model: str, max_output_tokens: int, audit_path: str | Path,
                 timeout_seconds: float = 120.0, max_retries: int = 2, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=timeout_seconds, max_retries=max_retries)
        self.client = client
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.audit_path = Path(audit_path)
        self._lock = threading.Lock()
        self.last_generation_info: dict[str, Any] = {}

    def _call(self, *, role: str, instructions: str, payload: dict[str, Any]) -> str:
        started = time.time()
        request = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(payload, sort_keys=True, default=str),
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        try:
            response = self.client.responses.create(**request)
            text = str(getattr(response, "output_text", "") or "").strip()
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            row = {
                "timestamp_unix": time.time(), "role": role, "status": "ok",
                "request": request, "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "response_status": getattr(response, "status", None),
                "output_text": text, "usage": usage,
                "latency_seconds": time.time() - started,
            }
        except Exception as exc:
            row = {
                "timestamp_unix": time.time(), "role": role, "status": "error",
                "request": request, "error_type": type(exc).__name__,
                "error": str(exc), "latency_seconds": time.time() - started,
            }
            self._log(row)
            raise
        self._log(row)
        self.last_generation_info = row
        return text

    def _log(self, row: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


class OpenAIRCAAgent(_ResponsesAgent):
    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        # This is the evaluator's public tool capability schema, not a
        # scenario-specific candidate list. Unsupported mechanisms must not be
        # offered merely because a parser knows their names.
        from digital_twin_runtime.live_capabilities import LIVE_MECHANISM_CAPABILITIES
        allowed = sorted(
            mechanism for mechanism, capability in LIVE_MECHANISM_CAPABILITIES.items()
            if capability.live_reward_eligible
        )
        type_contract = {
            mechanism: INJECTIBLE_FAULT_MECHANISMS[mechanism]
            for mechanism in allowed
        }
        raw = self._call(
            role="rca",
            instructions=(
                "You are a fixed Kubernetes root-cause agent. Use the policy instruction as search guidance, "
                "but independently inspect all supplied redacted telemetry, especially Kubernetes desired/ready "
                "replicas, scheduling conditions, Services, endpoints, ports, events, logs, metrics, and traces. "
                "Return only service::fault_type::injectible_mechanism[::variant] lines, with no prose. "
                "For scale_replicas_zero use scale_0, scale_2, or scale_3 when evidence supports it; "
                "for network_delay use delay_100ms, delay_300ms, or delay_1000ms; for network_loss "
                "use loss_5pct, loss_20pct, or loss_50pct. Omit the variant only for default. "
                "The second field is always the canonical fault type and the third field is always "
                "the mechanism. Example: svc::infra_failure::scale_replicas_zero::scale_0."
            ),
            payload={"policy_instruction": instruction, "redacted_state": compressed_state,
                     "allowed_mechanisms": allowed,
                     "required_fault_type_for_mechanism": type_contract,
                     "mechanism_semantics": {
                         "assign_to_non_existent_node": (
                             "desired replicas remain above zero but pods are unavailable because scheduling cannot place them"
                         ),
                         "scale_replicas_zero": (
                             "the Deployment desired replica count itself is zero for scale_0/default, or exactly 2/3 for scale_2/scale_3"
                         ),
                         "target_port_misconfig": (
                             "a Service has ready backing pods/endpoints but forwards to an incorrect targetPort"
                         ),
                         "application_config_misconfig": "the selected workload runs the known faulty application configuration/image",
                     },
                     "variant_contract": {
                         "scale_replicas_zero": ["scale_0", "scale_2", "scale_3"],
                         "network_delay": ["delay_100ms", "delay_300ms", "delay_1000ms"],
                         "network_loss": ["loss_5pct", "loss_20pct", "loss_50pct"],
                     }},
        )
        normalized = _normalize_rca_lines(raw)
        self.last_generation_info["normalized_output"] = normalized
        return normalized


class OpenAIActionAgent(_ResponsesAgent):
    def __init__(self, *args: Any, max_commands: int = 15, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_commands = int(max_commands)

    def get_commands(self, instruction_prompt: str, context: dict[str, Any]) -> list[str]:
        raw = self._call(
            role="action",
            instructions=(
                "You are a fixed Kubernetes remediation agent. Return only safe namespace-scoped "
                "kubectl, helm, or mongosh commands, one per line; no markdown or prose. Never use "
                "exec, apply, replace, pipelines, broad deletes, or cluster-wide flags."
            ),
            payload={"policy_instruction": instruction_prompt, "context": context},
        )
        commands = _extract_command_lines(raw, max_commands=self.max_commands)
        self.last_generation_info["parsed_commands"] = commands
        return commands
