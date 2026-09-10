from __future__ import annotations

"""Measure where agent-facing prompt tokens are spent before real Qwen training.

This audit is intentionally read-only. It loads one processed scenario, applies the
same training-safe sanitizer used by the joint runner, and reports both exact Qwen
chat-template token counts and structural JSON hotspots. The goal is to design a
bounded semantic projection of telemetry rather than blindly truncating the final
prompt token sequence.
"""

import argparse
import json
from typing import Any

from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .data_loader import iter_scenarios
from .qwen_shared_policy_backend import DEFAULT_QWEN_MODEL
from .rca_loop import build_rca_policy_prompt


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _plain_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(str(text), add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    if ids is None:
        raise TypeError("tokenizer did not return input_ids")
    if hasattr(ids, "ids"):
        ids = ids.ids
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("unexpected tokenizer batch")
        ids = ids[0]
    return len(ids)


def _chat_token_count(tokenizer: Any, text: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(text)}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ids = rendered.get("input_ids") if isinstance(rendered, dict) else getattr(rendered, "input_ids", None)
    if ids is None:
        raise TypeError("chat template did not return input_ids")
    if getattr(ids, "ndim", None) == 2:
        if ids.shape[0] != 1:
            raise ValueError(f"unexpected chat-template batch shape {tuple(ids.shape)}")
        return int(ids.shape[1])
    if getattr(ids, "ndim", None) == 1:
        return int(ids.shape[0])
    if isinstance(ids, (list, tuple)):
        if ids and isinstance(ids[0], (list, tuple)):
            return len(ids[0])
        return len(ids)
    raise TypeError(f"unsupported chat-template input_ids type: {type(ids)!r}")


def _structural_hotspots(obj: Any, *, max_depth: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(value: Any, path: str, depth: int) -> None:
        text = _json(value)
        rows.append(
            {
                "path": path,
                "depth": depth,
                "serialized_chars": len(text),
                "type": type(value).__name__,
                "items": len(value) if isinstance(value, (dict, list)) else None,
            }
        )
        if depth >= max_depth:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}", depth + 1)
        elif isinstance(value, list):
            # Individual list elements are often telemetry samples. Walking every
            # element can produce millions of audit rows, so summarize the list as
            # one node and inspect only a few representative elements.
            for i, child in enumerate(value[:3]):
                walk(child, f"{path}[{i}]", depth + 1)

    walk(obj, "$", 0)
    return sorted(rows, key=lambda x: int(x["serialized_chars"]), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit exact Qwen prompt-token budget by telemetry field")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", default=None)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if args.top < 1:
        raise ValueError("--top must be >= 1")

    rec = None
    for candidate in iter_scenarios(args.processed_states):
        if args.scenario_id is None or candidate.scenario_id == args.scenario_id:
            rec = candidate
            break
    if rec is None:
        raise RuntimeError(f"scenario not found: {args.scenario_id!r}")

    state = sanitize_agent_state(rec.compressed_state, mode="training_safe")
    safety = agent_input_safety_report(state)
    if not safety.get("safe_for_training_agent"):
        raise AssertionError(f"sanitized state failed safety audit: {safety}")

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("audit requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    state_text = _json(state)
    rca_prompt = build_rca_policy_prompt(state, [], 0, 1)

    top_level: list[dict[str, Any]] = []
    for key, value in state.items():
        text = _json(value)
        top_level.append(
            {
                "key": str(key),
                "type": type(value).__name__,
                "items": len(value) if isinstance(value, (dict, list)) else None,
                "serialized_chars": len(text),
                "plain_tokens": _plain_token_count(tokenizer, text),
            }
        )
    top_level.sort(key=lambda x: int(x["plain_tokens"]), reverse=True)

    summary = {
        "status": "PASS",
        "scenario_id": rec.scenario_id,
        "model": args.model,
        "agent_input_safe": True,
        "state_serialized_chars": len(state_text),
        "state_plain_tokens": _plain_token_count(tokenizer, state_text),
        "rca_policy_prompt_chat_tokens": _chat_token_count(tokenizer, rca_prompt),
        "top_level_fields_by_tokens": top_level,
        "structural_hotspots_by_chars": _structural_hotspots(state)[: args.top],
        "note": (
            "Use these measurements to build a deterministic semantic telemetry projection; "
            "do not truncate raw prompt token IDs blindly."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
