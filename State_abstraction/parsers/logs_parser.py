from collections import Counter
from pathlib import Path

from config import LOG_KEYWORDS, SERVICES
from utils import safe_read_text


ERROR_FAMILY_PRIORITY = [
    "config",
    "timeout",
    "connection",
    "dns",
    "error",
    "warn",
    "retry",
]


def classify_dominant_error_type(stats):
    family_map = {
        "config": stats["config_count"],
        "timeout": stats["timeout_count"],
        "connection": stats["connection_count"],
        "dns": stats["dns_count"],
        "error": stats["error_count"],
        "warn": stats["warn_count"],
        "retry": stats["retry_count"],
    }

    best = "none"
    best_val = 0
    for fam in ERROR_FAMILY_PRIORITY:
        if family_map[fam] > best_val:
            best = fam
            best_val = family_map[fam]
    return best


def compute_log_anomaly_score(stats):
    score = 0.0
    score += min(stats["error_count"] * 0.05, 0.4)
    score += min(stats["timeout_count"] * 0.07, 0.3)
    score += min(stats["connection_count"] * 0.08, 0.3)
    score += min(stats["config_count"] * 0.1, 0.35)
    score += min(stats["dns_count"] * 0.08, 0.2)
    return min(score, 1.0)


def extract_evidence_lines(lines):
    evidence = []
    for line in lines:
        ll = line.lower()
        if any(
            kw in ll
            for kws in LOG_KEYWORDS.values()
            for kw in kws
        ):
            evidence.append(line.strip()[:220])
    return evidence[:5]


def parse_log_file(path: Path):
    text = safe_read_text(path)
    lines = text.splitlines()
    lower_lines = [x.lower() for x in lines]

    out = {
        "line_count": len(lines),
        "error_count": 0,
        "warn_count": 0,
        "timeout_count": 0,
        "connection_count": 0,
        "config_count": 0,
        "dns_count": 0,
        "retry_count": 0,
        "dominant_error_type": "none",
        "top_error_templates": [],
        "evidence_lines": [],
        "log_anomaly_score": 0.0,
        "log_health_score": 1.0,
    }

    bucket_map = {
        "error_count": LOG_KEYWORDS["error"],
        "warn_count": LOG_KEYWORDS["warn"],
        "timeout_count": LOG_KEYWORDS["timeout"],
        "connection_count": LOG_KEYWORDS["connection"],
        "config_count": LOG_KEYWORDS["config"],
        "dns_count": LOG_KEYWORDS["dns"],
        "retry_count": LOG_KEYWORDS["retry"],
    }

    for line in lower_lines:
        for feature, kws in bucket_map.items():
            if any(kw in line for kw in kws):
                out[feature] += 1

    template_counter = Counter()
    for line in lines:
        ll = line.lower()
        if any(k in ll for k in ["error", "warn", "timeout", "refused", "failed", "config", "exception"]):
            template_counter[line.strip()[:180]] += 1

    out["top_error_templates"] = [x for x, _ in template_counter.most_common(5)]
    out["evidence_lines"] = extract_evidence_lines(lines)
    out["dominant_error_type"] = classify_dominant_error_type(out)
    out["log_anomaly_score"] = compute_log_anomaly_score(out)
    out["log_health_score"] = 1.0 - out["log_anomaly_score"]

    return out


def empty_log_state():
    return {
        "line_count": 0,
        "error_count": 0,
        "warn_count": 0,
        "timeout_count": 0,
        "connection_count": 0,
        "config_count": 0,
        "dns_count": 0,
        "retry_count": 0,
        "dominant_error_type": "none",
        "top_error_templates": [],
        "evidence_lines": [],
        "log_anomaly_score": 0.0,
        "log_health_score": 1.0,
    }


def parse_logs_snapshot(files_by_service):
    result = {}
    for svc in SERVICES:
        f = files_by_service.get(svc)
        result[svc] = parse_log_file(f) if f else empty_log_state()
    return result