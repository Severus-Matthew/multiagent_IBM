# traces_parser.py

import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from utils import safe_float, percentile, normalize_service_name


def is_error(row):
    has_error = str(row.get("has_error", "")).lower().strip()
    response = str(row.get("response", "")).lower().strip()
    if has_error in ["true", "1", "yes", "error", "failed", "failure"]:
        return True
    if response in ["error", "failed", "failure", "timeout", "exception"]:
        return True
    try:
        return int(float(response)) >= 400
    except Exception:
        return False


def classify_failure(error_ratio, latency_p95_us):
    if error_ratio >= 0.9:
        return "hard_error_path"
    if error_ratio >= 0.3:
        return "partial_error_path"
    if latency_p95_us >= 500000:
        return "high_latency_path"
    if latency_p95_us >= 100000:
        return "degraded_latency_path"
    return "healthy_path"


def edge_rank_score(error_ratio, latency_p95_us):
    return min(1.0, 0.8 * min(error_ratio, 1.0) + 0.2 * min(latency_p95_us / 1_000_000.0, 1.0))


def _first_present(row, *keys):
    for key in keys:
        if key in row and row.get(key) not in [None, ""]:
            return row.get(key)
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        val = lowered.get(str(key).lower())
        if val not in [None, ""]:
            return val
    return ""


def normalize_trace_row(row):
    normalized = dict(row)
    normalized["trace_id"] = _first_present(row, "trace_id", "traceID", "traceId", "traceid")
    normalized["span_id"] = _first_present(row, "span_id", "spanID", "spanId", "spanid")
    normalized["parent_span"] = _first_present(row, "parent_span", "parent_span_id", "parentSpanID", "parentSpanId", "parentspanid")
    normalized["service_name"] = _first_present(row, "service_name", "service", "process.serviceName", "process_servicename", "process_service_name")
    normalized["operation_name"] = _first_present(row, "operation_name", "operation", "operationName", "name")
    normalized["duration"] = _first_present(row, "duration", "duration_us", "durationMicros", "duration_micros", "duration_microseconds")
    normalized["response"] = _first_present(row, "response", "status_code", "http.status_code", "statusCode")
    normalized["has_error"] = _first_present(row, "has_error", "error", "error_flag", "otel.status_code")
    return normalized


def parse_trace_csv(path):
    path = Path(path)
    rows = []
    parse_errors = []
    try:
        with open(path, newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(normalize_trace_row(row))
    except Exception as e:
        return {}, set(), {"file": str(path), "rows_seen": 0, "parse_errors": [{"file": str(path), "error": str(e)}]}

    spans_by_trace = defaultdict(dict)
    for row in rows:
        trace_id = str(row.get("trace_id", "")).strip()
        span_id = str(row.get("span_id", "")).strip()
        if trace_id and span_id:
            spans_by_trace[trace_id][span_id] = row

    edge_stats = defaultdict(lambda: {"request_count": 0, "error_count": 0, "durations_us": [], "operations": Counter(), "responses": Counter(), "trace_ids": set(), "example_error_spans": []})
    service_stats = defaultdict(lambda: {"span_count": 0, "error_span_count": 0, "durations_us": [], "operations": Counter(), "responses": Counter()})
    observed_edges = set()

    for trace_id, spans in spans_by_trace.items():
        for span_id, span in spans.items():
            child_service = normalize_service_name(span.get("service_name"))
            parent_span_id = str(span.get("parent_span", "")).strip()
            operation = str(span.get("operation_name", "")).strip()
            response = str(span.get("response", "")).strip()
            duration_us = safe_float(span.get("duration"), 0.0)
            span_error = is_error(span)

            service_stats[child_service]["span_count"] += 1
            service_stats[child_service]["durations_us"].append(duration_us)
            service_stats[child_service]["operations"][operation] += 1
            service_stats[child_service]["responses"][response] += 1
            if span_error:
                service_stats[child_service]["error_span_count"] += 1

            parent_service = normalize_service_name(spans[parent_span_id].get("service_name")) if parent_span_id and parent_span_id in spans else "ROOT"
            edge = f"{parent_service}->{child_service}"
            observed_edges.add((parent_service, child_service))
            edge_stats[edge]["request_count"] += 1
            edge_stats[edge]["durations_us"].append(duration_us)
            edge_stats[edge]["operations"][operation] += 1
            edge_stats[edge]["responses"][response] += 1
            edge_stats[edge]["trace_ids"].add(trace_id)
            if span_error:
                edge_stats[edge]["error_count"] += 1
                if len(edge_stats[edge]["example_error_spans"]) < 5:
                    edge_stats[edge]["example_error_spans"].append({"trace_id": trace_id, "span_id": span_id, "service": child_service, "operation": operation, "response": response, "duration_us": duration_us})

    final_edges = {}
    for edge, stats in edge_stats.items():
        req = stats["request_count"]
        err = stats["error_count"]
        durations = stats["durations_us"]
        error_ratio = err / req if req else 0.0
        latency_p95_us = percentile(durations, 95)
        final_edges[edge] = {
            "request_count": req,
            "error_count": err,
            "error_ratio": error_ratio,
            "latency_mean_us": sum(durations) / len(durations) if durations else 0.0,
            "latency_p50_us": percentile(durations, 50),
            "latency_p95_us": latency_p95_us,
            "latency_p99_us": percentile(durations, 99),
            "latency_max_us": max(durations) if durations else 0.0,
            "failure_type": classify_failure(error_ratio, latency_p95_us),
            "is_suspicious": error_ratio > 0.2 or latency_p95_us > 100000,
            "edge_rank_score": edge_rank_score(error_ratio, latency_p95_us),
            "top_operations": dict(stats["operations"].most_common(10)),
            "responses": dict(stats["responses"].most_common(10)),
            "num_traces": len(stats["trace_ids"]),
            "example_error_spans": stats["example_error_spans"],
        }

    final_services = {}
    for svc, stats in service_stats.items():
        count = stats["span_count"]
        err = stats["error_span_count"]
        durations = stats["durations_us"]
        final_services[svc] = {
            "span_count": count,
            "error_span_count": err,
            "error_ratio": err / count if count else 0.0,
            "latency_mean_us": sum(durations) / len(durations) if durations else 0.0,
            "latency_p95_us": percentile(durations, 95),
            "latency_max_us": max(durations) if durations else 0.0,
            "top_operations": dict(stats["operations"].most_common(10)),
            "responses": dict(stats["responses"].most_common(10)),
        }

    meta = {"file": str(path), "rows_seen": len(rows), "num_traces": len(spans_by_trace), "num_edges": len(final_edges), "num_services": len(final_services), "parse_errors": parse_errors, "service_summary": final_services}
    return final_edges, observed_edges, meta


def _looks_like_trace_csv(path: Path) -> bool:
    name = path.name.lower()
    if "trace" in name or "span" in name or "jaeger" in name:
        return True
    try:
        with open(path, newline="", errors="ignore") as f:
            header = next(csv.reader(f), [])
    except Exception:
        return False
    cols = {str(col).strip().lower() for col in header}
    strong = {"trace_id", "span_id"}
    service_cols = {"service_name", "service", "process.servicename"}
    return strong.issubset(cols) and bool(cols & service_cols)


def _export_paths_from_text_outputs(run_dir: Path):
    trace_text_dir = run_dir / "builtin_api_outputs" / "traces"
    if not trace_text_dir.exists():
        return []
    paths = []
    for text_path in trace_text_dir.rglob("*.txt"):
        try:
            text = text_path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            match = re.search(r"exported\s+(?:traces\s+)?(?:to\s+file|to):\s*(.+)$", line, re.I)
            if not match:
                continue
            candidate = Path(match.group(1).strip().strip("\"'"))
            if candidate.exists():
                paths.append(candidate)
    return paths


def discover_trace_csv_files(run_dir):
    run_dir = Path(run_dir)
    candidate_roots = [run_dir / "builtin_api_outputs" / "traces", run_dir / "traces", run_dir / "trace_output", run_dir / "traces_output"]
    candidate_roots.extend(_export_paths_from_text_outputs(run_dir))
    trace_files = set()
    for root in candidate_roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() == ".csv" and _looks_like_trace_csv(root):
                trace_files.add(root)
            continue
        for path in root.rglob("*.csv"):
            if _looks_like_trace_csv(path):
                trace_files.add(path)
    for path in run_dir.rglob("*.csv"):
        if _looks_like_trace_csv(path):
            trace_files.add(path)
    return sorted(trace_files)


def merge_edges(all_edge_dicts):
    merged_raw = defaultdict(lambda: {"request_count": 0, "error_count": 0, "durations_us": [], "operations": Counter(), "responses": Counter(), "example_error_spans": []})
    for edge_dict in all_edge_dicts:
        for edge, feats in edge_dict.items():
            merged_raw[edge]["request_count"] += feats.get("request_count", 0)
            merged_raw[edge]["error_count"] += feats.get("error_count", 0)
            for key in ["latency_mean_us", "latency_p50_us", "latency_p95_us", "latency_p99_us", "latency_max_us"]:
                val = feats.get(key, 0.0)
                if val:
                    merged_raw[edge]["durations_us"].append(val)
            merged_raw[edge]["operations"].update(feats.get("top_operations", {}))
            merged_raw[edge]["responses"].update(feats.get("responses", {}))
            merged_raw[edge]["example_error_spans"].extend(feats.get("example_error_spans", [])[:5])
    merged = {}
    for edge, stats in merged_raw.items():
        req = stats["request_count"]
        err = stats["error_count"]
        durations = stats["durations_us"]
        error_ratio = err / req if req else 0.0
        latency_p95_us = percentile(durations, 95)
        merged[edge] = {
            "request_count": req,
            "error_count": err,
            "error_ratio": error_ratio,
            "latency_mean_us": sum(durations) / len(durations) if durations else 0.0,
            "latency_p50_us": percentile(durations, 50),
            "latency_p95_us": latency_p95_us,
            "latency_p99_us": percentile(durations, 99),
            "latency_max_us": max(durations) if durations else 0.0,
            "failure_type": classify_failure(error_ratio, latency_p95_us),
            "is_suspicious": error_ratio > 0.2 or latency_p95_us > 100000,
            "edge_rank_score": edge_rank_score(error_ratio, latency_p95_us),
            "top_operations": dict(stats["operations"].most_common(10)),
            "responses": dict(stats["responses"].most_common(10)),
            "example_error_spans": stats["example_error_spans"][:5],
        }
    return merged


def parse_traces(run_dir):
    trace_files = discover_trace_csv_files(Path(run_dir))
    all_edge_dicts = []
    all_observed_edges = set()
    file_metas = []
    for path in trace_files:
        edges, observed_edges, meta = parse_trace_csv(path)
        all_edge_dicts.append(edges)
        all_observed_edges |= observed_edges
        file_metas.append(meta)
    merged_edges = merge_edges(all_edge_dicts)
    meta = {"files_seen": [str(x) for x in trace_files], "num_files": len(trace_files), "num_edges": len(merged_edges), "num_observed_edges": len(all_observed_edges), "file_metas": file_metas, "trace_signal_present": any(m.get("rows_seen", 0) > 0 for m in file_metas)}
    return merged_edges, sorted(all_observed_edges), meta
