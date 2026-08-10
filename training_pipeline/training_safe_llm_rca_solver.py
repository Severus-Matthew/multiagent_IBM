from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .llm_rca_solver import _default_debug_path, _stable_hash, sanitize_rca_prediction


_SYSTEM_PROMPT_TRAINING_SAFE = """\
You are an RCA solver for Kubernetes/AIOps incidents.

You receive only redacted telemetry and public retry feedback. You do not receive
an oracle label, injected fault family, candidate root-cause menu, list of
possible root causes, or generated scenario identifier.

Infer the most likely upstream root cause(s) from the telemetry. Separate root
causes from downstream cascade victims.

Output ONLY lines in this form:
component::fault_mechanism

Rules:
- Use component names that appear in the telemetry evidence.
- Use concise mechanism words such as auth, config, network, latency, infra,
  dependency, resource, crash, scheduling, or unknown when unclear.
- Do not output explanations, markdown, JSON, bullets, or candidate rankings.
- Use fewer lines when the incident appears single-root; use multiple lines only
  when independent evidence supports multiple root causes.
"""


class TrainingSafeLLMRCASolver:
    """LLM RCA solver with candidate/oracle menus removed from the prompt.

    This is the training/evaluation-safe path.  It may still use private parsing
    and normalization after the model emits text, but the prompt itself never
    includes candidate_root_causes, valid_services, scenario_id, ground truth, or
    injected-fault metadata.  This keeps the model transportable to datasets such
    as ITBench where service topology and canonical fault families differ.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
        state_char_budget: int = 24000,
        cache_path: str | None = None,
        max_root_causes: int = 1,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.state_char_budget = int(state_char_budget)
        self.max_root_causes = max(1, int(max_root_causes))
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.debug_path = _default_debug_path(self.cache_path)
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        from agents.llm_client import LLMClient

        self.client = LLMClient(provider=provider, model=model)
        self.model_name = self.client.model

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        agent_state = sanitize_agent_state(compressed_state, mode="training_safe")
        safety = agent_input_safety_report(agent_state)
        prompt_state = _truncate_state(agent_state, self.state_char_budget)
        cache_key = _stable_hash({
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "solver": "training_safe_llm_v1",
            "instruction": instruction,
            "state_hash": _stable_hash(prompt_state),
        })
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = self._build_user_prompt(prompt_state, instruction, safety)
        raw = self.client.call(
            system_prompt=_SYSTEM_PROMPT_TRAINING_SAFE,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # Private normalization is allowed; the model was not shown the candidate
        # universe.  Do not repair to candidate_root_causes or best-ranked menus.
        sanitized_unlimited = sanitize_rca_prediction(raw, compressed_state, max_root_causes=None)
        final = sanitize_rca_prediction(raw, compressed_state, max_root_causes=self.max_root_causes)
        if not final.strip():
            final = "unknown::unknown"

        debug = {
            "key": cache_key,
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "state_hash": _stable_hash(prompt_state),
            "instruction": instruction,
            "agent_input_safety": safety,
            "raw_response": raw,
            "sanitized_unlimited": sanitized_unlimited,
            "final_prediction_after_sanitization": final,
            "postprocess_mode": "training_safe_no_candidate_repair_v1",
            "prompt_contains_candidate_menu": False,
        }
        self._cache[cache_key] = final
        self._append_cache(cache_key, final, raw)
        self._append_debug(debug)
        return final

    def _build_user_prompt(self, agent_state: dict[str, Any], instruction: str, safety: dict[str, Any]) -> str:
        payload = {
            "policy_instruction": instruction,
            "root_cause_count_instruction": f"Return at most {self.max_root_causes} root cause line(s). Use fewer unless independent evidence supports multiple root causes.",
            "agent_input_safety": safety,
            "redacted_telemetry_state": agent_state,
            "output_contract": "Return only component::fault_mechanism lines. No prose.",
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        with self.cache_path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                key = row.get("key")
                pred = row.get("prediction")
                if key and isinstance(pred, str):
                    self._cache[str(key)] = pred

    def _append_cache(self, key: str, prediction: str, raw_response: str) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": key,
            "provider": self.provider,
            "model": self.model_name,
            "max_root_causes": self.max_root_causes,
            "prediction": prediction,
            "raw_response": raw_response,
            "solver": "training_safe_llm_v1",
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _append_debug(self, row: dict[str, Any]) -> None:
        if not self.debug_path:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _truncate_state(state: dict[str, Any], char_budget: int) -> dict[str, Any]:
    text = json.dumps(state, sort_keys=True, default=str)
    if len(text) <= char_budget:
        return state
    # Conservative schema-agnostic fallback: keep the highest-level observable
    # telemetry summaries rather than deep raw blobs.
    keep_keys = [
        "system",
        "service_health",
        "llm_view",
        "traces",
        "metrics",
        "clusters",
        "graph",
        "sla",
        "redaction_note",
    ]
    slim = {k: state.get(k) for k in keep_keys if isinstance(state, dict) and k in state}
    text = json.dumps(slim, sort_keys=True, default=str)
    if len(text) <= char_budget:
        return slim
    return {
        "truncated_redacted_state": text[: max(1000, int(char_budget))],
        "truncation_note": "State exceeded training-safe prompt budget; candidate/oracle fields were already removed before truncation.",
    }
