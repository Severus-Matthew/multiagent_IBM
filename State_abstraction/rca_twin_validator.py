from collections import defaultdict


def compare_logs(real_logs, sim_logs):
    services = sorted(set(real_logs.keys()) | set(sim_logs.keys()))
    scores = []

    for svc in services:
        r = real_logs.get(svc, {})
        s = sim_logs.get(svc, {})

        score = 0.0

        # dominant error type
        if r.get("dominant_error_type", "none") == s.get("dominant_error_type", "none"):
            score += 0.35

        # anomaly score closeness
        ra = r.get("log_anomaly_score", 0.0)
        sa = s.get("log_anomaly_score", 0.0)
        score += max(0.0, 0.25 - min(abs(ra - sa), 0.25))

        # signal present agreement
        if r.get("log_signal_present", False) == s.get("log_signal_present", False):
            score += 0.15

        scores.append(min(score, 0.75))

    return sum(scores) / len(scores) if scores else 0.0


def compare_metrics(real_metrics, sim_metrics):
    services = sorted(set(real_metrics.keys()) | set(sim_metrics.keys()))
    scores = []

    for svc in services:
        r = real_metrics.get(svc, {})
        s = sim_metrics.get(svc, {})

        score = 0.0

        # latency closeness
        rl = r.get("latency", 0.0)
        sl = s.get("latency", 0.0)
        diff = abs(rl - sl)
        if diff < 5:
            score += 0.45
        elif diff < 20:
            score += 0.25
        elif diff < 100:
            score += 0.1

        # metric signal presence
        if r.get("metric_signal_present", False) == s.get("metric_signal_present", False):
            score += 0.15

        scores.append(min(score, 0.6))

    return sum(scores) / len(scores) if scores else 0.0


def compare_system(real_system, sim_system):
    services = sorted(set(real_system.keys()) | set(sim_system.keys()))
    scores = []

    for svc in services:
        r = real_system.get(svc, {})
        s = sim_system.get(svc, {})

        score = 0.0

        if r.get("infra_issue_flag", False) == s.get("infra_issue_flag", False):
            score += 0.4

        if r.get("service_health_status", "unknown") == s.get("service_health_status", "unknown"):
            score += 0.35

        scores.append(min(score, 0.75))

    return sum(scores) / len(scores) if scores else 0.0


def compare_traces(real_traces, sim_traces):
    edges = sorted(set(real_traces.keys()) | set(sim_traces.keys()))
    scores = []

    for edge in edges:
        r = real_traces.get(edge, {})
        s = sim_traces.get(edge, {})

        score = 0.0

        # error ratio closeness
        rer = r.get("error_ratio", 0.0)
        ser = s.get("error_ratio", 0.0)
        diff_er = abs(rer - ser)
        if diff_er < 0.1:
            score += 0.45
        elif diff_er < 0.3:
            score += 0.25
        elif diff_er < 0.6:
            score += 0.1

        # failure type agreement
        if r.get("failure_type", "healthy_path") == s.get("failure_type", "healthy_path"):
            score += 0.3

        # suspiciousness agreement
        if r.get("is_suspicious", False) == s.get("is_suspicious", False):
            score += 0.15

        scores.append(min(score, 0.9))

    return sum(scores) / len(scores) if scores else 0.0


def aggregate_similarity(real_state, sim_state):
    log_score = compare_logs(real_state.get("logs", {}), sim_state.get("logs", {}))
    metric_score = compare_metrics(real_state.get("metrics", {}), sim_state.get("metrics", {}))
    system_score = compare_system(real_state.get("system", {}), sim_state.get("system", {}))
    trace_score = compare_traces(real_state.get("traces", {}), sim_state.get("traces", {}))

    # traces matter most for your current setup
    total = (
        0.50 * trace_score +
        0.10 * system_score +
        0.15 * log_score +
        0.25 * metric_score
    )

    return {
        "trace_similarity": round(trace_score, 4),
        "system_similarity": round(system_score, 4),
        "log_similarity": round(log_score, 4),
        "metric_similarity": round(metric_score, 4),
        "overall_similarity": round(total, 4),
    }


def explain_match(real_state, sim_state):
    sim = aggregate_similarity(real_state, sim_state)

    reasons = []
    if sim["trace_similarity"] > 0.75:
        reasons.append("trace pattern matches strongly")
    elif sim["trace_similarity"] > 0.4:
        reasons.append("trace pattern partially matches")

    if sim["log_similarity"] < 0.2:
        reasons.append("log pattern does not match strongly")
    if sim["metric_similarity"] < 0.2:
        reasons.append("metric pattern does not match strongly")
    if sim["system_similarity"] > 0.6:
        reasons.append("system health pattern is consistent")

    return reasons


def verdict_from_similarity(overall_similarity):
    if overall_similarity >= 0.75:
        return "likely_correct"
    if overall_similarity >= 0.70:
        return "partly_correct"
    return "likely_incorrect"