import re
from config import SERVICES
from utils import safe_read_text


def compute_service_health_status(s):
    if s["crashloop_count"] > 0:
        return "crashloop"
    if s["pods_unready"] > 0:
        return "unready"
    if s["pending_count"] > 0:
        return "pending"
    if s["pods_ready"] > 0:
        return "healthy"
    return "unknown"


def parse_pods_file(path):
    out = {
        svc: {
            "pods_total": 0,
            "pods_ready": 0,
            "pods_unready": 0,
            "restart_count": 0,
            "crashloop_count": 0,
            "pending_count": 0,
            "running_count": 0,
            "service_health_status": "unknown",
            "infra_issue_flag": False,
        }
        for svc in SERVICES
    }

    if not path or not path.exists():
        return out

    text = safe_read_text(path)
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lower().startswith("name"):
            continue

        for svc in SERVICES:
            if svc not in line:
                continue

            parts = re.split(r"\s+", line.strip())
            if len(parts) < 4:
                continue

            ready = parts[1]
            status = parts[2].lower()
            restarts = parts[3]

            out[svc]["pods_total"] += 1

            if "/" in ready:
                a, b = ready.split("/")
                if a == b:
                    out[svc]["pods_ready"] += 1
                else:
                    out[svc]["pods_unready"] += 1

            if status == "running":
                out[svc]["running_count"] += 1
            if status == "pending":
                out[svc]["pending_count"] += 1
            if "crashloopbackoff" in status:
                out[svc]["crashloop_count"] += 1

            try:
                out[svc]["restart_count"] += int(restarts)
            except Exception:
                pass

    for svc in SERVICES:
        out[svc]["service_health_status"] = compute_service_health_status(out[svc])
        out[svc]["infra_issue_flag"] = (
            out[svc]["crashloop_count"] > 0
            or out[svc]["pods_unready"] > 0
            or out[svc]["pending_count"] > 0
        )

    return out


def parse_describe_file(path, existing):
    if not path or not path.exists():
        return existing

    text = safe_read_text(path).lower()

    for svc in SERVICES:
        if svc not in text:
            continue
        existing[svc]["crashloop_count"] += text.count("crashloopbackoff")
        existing[svc]["pods_unready"] += text.count("not ready")

    for svc in SERVICES:
        existing[svc]["service_health_status"] = compute_service_health_status(existing[svc])
        existing[svc]["infra_issue_flag"] = (
            existing[svc]["crashloop_count"] > 0
            or existing[svc]["pods_unready"] > 0
            or existing[svc]["pending_count"] > 0
        )

    return existing


def parse_system_snapshot(files_by_type):
    pods = parse_pods_file(files_by_type.get("pods"))
    pods = parse_describe_file(files_by_type.get("describe"), pods)
    return pods