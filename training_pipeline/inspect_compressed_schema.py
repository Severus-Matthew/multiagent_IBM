from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_loader import iter_scenarios
from .split_utils import read_scenario_ids


SERVICE_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,80}(?:-service|-mongodb|-frontend|-backend|-db|-redis|-memcached)?", re.IGNORECASE)
SIGNAL_TERMS = {
    "infra": ["pending", "unready", "crash", "crashloop", "node", "nodeName", "scheduling", "killed", "container", "replica", "unavailable"],
    "auth": ["auth", "unauthorized", "credential", "permission", "forbidden", "not authorized", "authentication"],
    "config": ["target_port", "target port", "port", "misconfig", "config", "wrong bin", "wrong_binary", "service port"],
    "latency": ["latency", "delay", "timeout", "slow", "duration"],
    "network": ["packet", "loss", "network", "unreachable", "connection reset", "dns"],
    "error": ["error", "exception", "fail", "failed", "failure", "refused", "5xx"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect compressed-state schema and recursive evidence tokens")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_values", type=int, default=20)
    args = ap.parse_args()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    rows = []
    total = skipped = 0
    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            skipped += 1
            continue
        if total >= args.limit:
            break
        total += 1
        rows.append(inspect_state(rec.scenario_id, rec.compressed_state, max_values=args.max_values))

    summary = {"total": total, "skipped_filter": skipped, "rows": rows}
    if args.output:
        p = Path(args.output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def inspect_state(scenario_id: str, state: dict[str, Any], max_values: int = 20) -> dict[str, Any]:
    path_types = Counter()
    key_counts = Counter()
    string_hits: list[dict[str, Any]] = []
    service_mentions = Counter()
    signal_counts = defaultdict(Counter)
    candidate_strings = []

    def visit(obj: Any, path: str, depth: int) -> None:
        if depth > 14:
            return
        path_types[f"{path}:{type(obj).__name__}"] += 1
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                key_counts[ks] += 1
                visit(v, f"{path}.{ks}" if path else ks, depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:200]):
                visit(v, f"{path}[]", depth + 1)
        elif isinstance(obj, str):
            text = obj
            lowered = text.lower()
            services = extract_service_like(text)
            for svc in services:
                service_mentions[svc] += 1
            for group, terms in SIGNAL_TERMS.items():
                if any(t.lower() in lowered for t in terms):
                    signal_counts[group][path] += 1
                    if len(string_hits) < max_values:
                        string_hits.append({"path": path, "group": group, "text": text[:500]})
            if services and len(candidate_strings) < max_values:
                candidate_strings.append({"path": path, "services": services[:10], "text": text[:500]})

    visit(state, "", 0)
    return {
        "scenario_id": scenario_id,
        "top_level_keys": list(state.keys()) if isinstance(state, dict) else [],
        "top_key_counts": key_counts.most_common(80),
        "top_path_types": path_types.most_common(80),
        "service_mentions": service_mentions.most_common(80),
        "signal_counts_by_group": {k: v.most_common(30) for k, v in signal_counts.items()},
        "string_hits": string_hits,
        "candidate_service_strings": candidate_strings,
    }


def extract_service_like(text: str) -> list[str]:
    out = []
    seen = set()
    for m in SERVICE_RE.finditer(text):
        token = m.group(0).strip("._:/")
        lower = token.lower()
        if len(lower) < 3:
            continue
        if lower in {"default", "namespace", "service", "mongodb", "container", "network", "target", "analysis", "mitigation", "localization", "detection"}:
            continue
        if any(x in lower for x in ["service", "mongodb", "frontend", "backend", "geo", "rate", "profile", "reservation", "recommendation", "user", "social", "timeline", "compose", "post", "media", "text", "url", "shorten", "unique", "home"]):
            if lower not in seen:
                seen.add(lower)
                out.append(lower)
    return out


if __name__ == "__main__":
    main()
