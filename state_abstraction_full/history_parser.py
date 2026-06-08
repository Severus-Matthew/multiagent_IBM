import re
from pathlib import Path
from utils import read_json

ACTION_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)")


def _extract_action(text):
    m = ACTION_RE.search(text or "")
    if not m:
        return None
    return {"api": m.group(1), "args_text": m.group(2)}


def parse_history(run_dir):
    run_dir = Path(run_dir)
    actions = []
    transcript = read_json(run_dir / "agent_transcript.json") or read_json(run_dir / "run_result.json") or {}
    trace = transcript.get("trace") if isinstance(transcript, dict) else None
    if trace is None and isinstance(transcript, dict):
        trace = transcript.get("history")
    if isinstance(trace, list):
        for item in trace:
            content = item.get("content") if isinstance(item, dict) else str(item)
            act = _extract_action(content)
            if act:
                actions.append(act)
    api_plan = read_json(run_dir / "api_action_plan.json") or []
    return {
        "previous_actions": actions,
        "api_action_plan": api_plan,
        "last_action": actions[-1] if actions else None,
        "action_count": len(actions),
    }
