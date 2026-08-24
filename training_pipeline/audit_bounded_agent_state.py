from __future__ import annotations

"""Audit semantic agent-state compression with the exact Qwen tokenizer."""

import argparse
import json
from typing import Any

from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .bounded_agent_state import BoundedAgentStateConfig, build_bounded_agent_state
from .data_loader import iter_scenarios
from .qwen_shared_policy_backend import DEFAULT_QWEN_MODEL
from .rca_loop import build_rca_policy_prompt


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _chat_tokens(tokenizer: Any, text: str) -> int:
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
        return int(ids.shape[1])
    if getattr(ids, "ndim", None) == 1:
        return int(ids.shape[0])
    if isinstance(ids, (list, tuple)):
        if ids and isinstance(ids[0], (list, tuple)):
            return len(ids[0])
        return len(ids)
    raise TypeError(f"unsupported input_ids type: {type(ids)!r}")


def _find_scenario(root: str, scenario_id: str | None):
    for rec in iter_scenarios(root):
        if scenario_id is None or rec.scenario_id == scenario_id:
            return rec
    raise RuntimeError(f"scenario not found: {scenario_id!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit bounded training-safe Qwen agent state")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", default=None)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--max_serialized_chars", type=int, default=42_000)
    ap.add_argument("--max_system_services", type=int, default=24)
    ap.add_argument("--max_metric_services", type=int, default=24)
    ap.add_argument("--max_log_services", type=int, default=20)
    ap.add_argument("--target_prompt_tokens", type=int, default=16_000)
    args = ap.parse_args()

    rec = _find_scenario(args.processed_states, args.scenario_id)
    original = sanitize_agent_state(rec.compressed_state, mode="training_safe")
    cfg = BoundedAgentStateConfig(
        max_serialized_chars=args.max_serialized_chars,
        max_system_services=args.max_system_services,
        max_metric_services=args.max_metric_services,
        max_log_services=args.max_log_services,
    )
    bounded = build_bounded_agent_state(original, config=cfg)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    original_prompt = build_rca_policy_prompt(original, [], 0, 1)
    bounded_prompt = build_rca_policy_prompt(bounded, [], 0, 1)
    original_tokens = _chat_tokens(tokenizer, original_prompt)
    bounded_tokens = _chat_tokens(tokenizer, bounded_prompt)

    before_safety = agent_input_safety_report(original)
    after_safety = agent_input_safety_report(bounded)
    if not after_safety.get("safe_for_training_agent"):
        raise AssertionError(after_safety)

    summary = {
        "status": "PASS" if bounded_tokens <= args.target_prompt_tokens else "OVER_TARGET",
        "scenario_id": rec.scenario_id,
        "model": args.model,
        "target_prompt_tokens": args.target_prompt_tokens,
        "original": {
            "serialized_chars": len(_json(original)),
            "rca_chat_prompt_tokens": original_tokens,
            "safe_for_training_agent": bool(before_safety.get("safe_for_training_agent")),
        },
        "bounded": {
            "serialized_chars": len(_json(bounded)),
            "rca_chat_prompt_tokens": bounded_tokens,
            "safe_for_training_agent": bool(after_safety.get("safe_for_training_agent")),
            "compression_ratio_tokens": round(bounded_tokens / original_tokens, 6),
            "tokens_removed": original_tokens - bounded_tokens,
            "projection": bounded.get("projection"),
            "system_detailed_services": len([k for k in (bounded.get("system") or {}) if k != "__projection_summary__"]),
            "metric_detailed_services": len([k for k in (bounded.get("metrics") or {}) if k != "__projection_summary__"]),
            "log_detailed_services": len([k for k in (bounded.get("logs") or {}) if k != "__projection_summary__"]),
        },
        "preserved_top_level_keys": sorted(bounded.keys()),
        "raw_prompt_token_truncation_used": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=False))

    if bounded_tokens > args.target_prompt_tokens:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
