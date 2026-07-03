# metrics_parser.py

import csv
import re
from pathlib import Path
from collections import defaultdict


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def service_from_cmdb_id(cmdb_id: str) -> str:
    """
    Example:
      kind-control-plane.compose-post-service-9f655fc76-cvsfn
      -> compose-post-service
    """
    name = str(cmdb_id).strip()

    if "." in name:
        name = name.split(".", 1)[1]

    # remove pod suffix: -9f655fc76-cvsfn
    name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{5}$", "", name)

    # remove possible remaining hash suffix
    name = re.sub(r"-[a-f0-9]{8,10}$", "", name)

    return name


def summarize_values(values):
    values = [safe_float(v) for v in values]

    if not values:
        return {
            "count": 0,
            "first": 0.0,
            "last": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "delta": 0.0,
        }

    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "delta": values[-1] - values[0],
    }


def metric_group(kpi_name: str) -> str:
    name = kpi_name.lower()

    if "cpu" in name:
        return "cpu"

    if "memory" in name:
        return "memory"

    if "network" in name:
        return "network"

    if "threads" in name:
        return "threads"

    if "spec" in name:
        return "spec"

    return "other"


def _first_present(row, *keys):
    for key in keys:
        if key in row and row.get(key) not in [None, ""]:
            return row.get(key)
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        val = lowered.get(str(key).lower())
        if val not in [None, ""]:
            return val
    return None


def parse_one_metric_csv(path: Path):
    """
    Expected CSV schema:
      timestamp, cmdb_id, kpi_name, value
    """

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = _first_present(row, "timestamp", "time", "ts")
            cmdb_id = _first_present(row, "cmdb_id", "pod", "pod_name", "instance", "container")
            kpi_name = _first_present(row, "kpi_name", "metric", "metric_name", "__name__")
            value = _first_present(row, "value", "val")

            if cmdb_id is None or kpi_name is None or value is None:
                continue

            rows.append({
                "timestamp": timestamp,
                "service": service_from_cmdb_id(cmdb_id),
                "cmdb_id": cmdb_id,
                "kpi_name": kpi_name,
                "value": safe_float(value),
            })

    return rows


def _looks_like_metric_csv(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("kpi_") and name.endswith(".csv"):
        return True

    try:
        with open(path, newline="", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        return False

    cols = {str(col).strip().lower() for col in header}
    has_entity = bool(cols & {"cmdb_id", "pod", "pod_name", "instance", "container"})
    has_metric = bool(cols & {"kpi_name", "metric", "metric_name", "__name__"})
    has_value = bool(cols & {"value", "val"})
    return has_entity and has_metric and has_value


def _export_paths_from_text_outputs(run_dir: Path):
    """
    AIOpsLab sometimes writes a text pointer such as:
      Metrics data exported to directory: /path/to/metrics_output/metric_...

    If that exported path is still present on the machine, include it in the
    parser search space. Missing paths are ignored because the scenario may
    have been moved without its external export directory.
    """
    metric_text_dir = run_dir / "builtin_api_outputs" / "metrics"
    if not metric_text_dir.exists():
        return []

    paths = []
    for text_path in metric_text_dir.rglob("*.txt"):
        try:
            text = text_path.read_text(errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            match = re.search(r"exported\s+(?:metrics\s+)?(?:to\s+directory|to):\s*(.+)$", line, re.I)
            if not match:
                continue
            candidate = Path(match.group(1).strip().strip("\"'"))
            if candidate.exists():
                paths.append(candidate)

    return paths


def discover_metric_csv_files(run_dir):
    run_dir = Path(run_dir)

    candidate_roots = [
        run_dir / "builtin_api_outputs" / "metrics",
        run_dir / "metrics",
        run_dir / "metric_output",
        run_dir / "metrics_output",
    ]
    candidate_roots.extend(_export_paths_from_text_outputs(run_dir))

    metric_files = set()

    for root in candidate_roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() == ".csv" and _looks_like_metric_csv(root):
                metric_files.add(root)
            continue
        for path in root.rglob("*.csv"):
            if _looks_like_metric_csv(path):
                metric_files.add(path)

    # Final fallback for copied scenarios where the export folder was placed
    # somewhere unexpected under the run directory.
    for path in run_dir.rglob("*.csv"):
        if _looks_like_metric_csv(path):
            metric_files.add(path)

    return sorted(metric_files)


def parse_metrics_snapshot(metric_csv_files):
    """
    Reads all:
      metrics/metric_*/container/kpi_*.csv

    Returns:
      metrics[service] = {
        cpu: {...},
        memory: {...},
        network: {...},
        threads: {...},
        spec: {...},
        raw_kpis: {...}
      }
    """

    per_service_kpi_values = defaultdict(lambda: defaultdict(list))
    files_seen = []
    parse_errors = []

    for file_path in metric_csv_files:
        path = Path(file_path)
        files_seen.append(str(path))

        try:
            rows = parse_one_metric_csv(path)
        except Exception as e:
            parse_errors.append({
                "file": str(path),
                "error": str(e),
            })
            continue

        for row in rows:
            svc = row["service"]
            kpi = row["kpi_name"]
            val = row["value"]

            per_service_kpi_values[svc][kpi].append(val)

    metrics = {}

    for svc, kpi_values in per_service_kpi_values.items():
        svc_metrics = {
            "cpu": {},
            "memory": {},
            "network": {},
            "threads": {},
            "spec": {},
            "other": {},
            "raw_kpis": {},
            "metric_signal_present": True,
        }

        for kpi, values in kpi_values.items():
            summary = summarize_values(values)
            group = metric_group(kpi)

            svc_metrics[group][kpi] = summary
            svc_metrics["raw_kpis"][kpi] = summary

        # convenient flattened fields for SLA / RCA
        # Use flexible lookup: try canonical kpi_ prefixed name first,
        # then without prefix, then any key containing the substring.
        def _kpi(raw, *candidates):
            for c in candidates:
                if c in raw and raw[c]:
                    return raw[c]
            # fallback: substring match
            lc = candidates[-1].lower().replace("kpi_", "").replace("container_", "")
            for k, v in raw.items():
                if lc in k.lower() and v:
                    return v
            return {}

        raw = svc_metrics["raw_kpis"]
        svc_metrics["cpu_usage_delta"] = _kpi(
            raw, "kpi_container_cpu_usage_seconds_total", "container_cpu_usage_seconds_total"
        ).get("delta", 0.0)

        svc_metrics["cpu_load_last"] = _kpi(
            raw, "kpi_container_cpu_load_average_10s", "container_cpu_load_average_10s"
        ).get("last", 0.0)

        svc_metrics["memory_working_set_last"] = _kpi(
            raw, "kpi_container_memory_working_set_bytes", "container_memory_working_set_bytes"
        ).get("last", 0.0)

        svc_metrics["memory_usage_last"] = _kpi(
            raw, "kpi_container_memory_usage_bytes", "container_memory_usage_bytes"
        ).get("last", 0.0)

        svc_metrics["network_rx_bytes_delta"] = _kpi(
            raw, "kpi_container_network_receive_bytes_total", "container_network_receive_bytes_total"
        ).get("delta", 0.0)

        svc_metrics["network_tx_bytes_delta"] = _kpi(
            raw, "kpi_container_network_transmit_bytes_total", "container_network_transmit_bytes_total"
        ).get("delta", 0.0)

        svc_metrics["network_tx_errors_delta"] = _kpi(
            raw, "kpi_container_network_transmit_errors_total", "container_network_transmit_errors_total"
        ).get("delta", 0.0)

        svc_metrics["network_rx_drops_delta"] = _kpi(
            raw, "kpi_container_network_receive_packets_dropped_total", "container_network_receive_packets_dropped_total"
        ).get("delta", 0.0)

        svc_metrics["network_tx_drops_delta"] = _kpi(
            raw, "kpi_container_network_transmit_packets_dropped_total", "container_network_transmit_packets_dropped_total"
        ).get("delta", 0.0)

        svc_metrics["threads_last"] = _kpi(
            raw, "kpi_container_threads", "container_threads"
        ).get("last", 0.0)

        # latency_ms is filled later from traces via derive_latency()
        svc_metrics["latency_ms"] = 0.0

        metrics[svc] = svc_metrics

    return {
        "per_service": metrics,
        "files_seen": files_seen,
        "parse_errors": parse_errors,
    }


def parse_metrics_from_run_dir(run_dir):
    run_dir = Path(run_dir)

    metric_files = discover_metric_csv_files(run_dir)

    return parse_metrics_snapshot(metric_files)
