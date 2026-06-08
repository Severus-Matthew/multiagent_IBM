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


def parse_one_metric_csv(path: Path):
    """
    Expected CSV schema:
      timestamp, cmdb_id, kpi_name, value
    """

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = row.get("timestamp")
            cmdb_id = row.get("cmdb_id")
            kpi_name = row.get("kpi_name")
            value = row.get("value")

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

    metric_files = sorted((run_dir /"builtin_api_outputs" / "metrics"/"metric"/"container").rglob("kpi_*.csv"))

    return parse_metrics_snapshot(metric_files)