# logs_parser.py
"""
Generalized log parser for AIOpsLab-style outputs.

Supports:
  1. direct_k8s_outputs/pod_logs/*.log
  2. direct_k8s_outputs/pod_logs/*.stderr
  3. logs/step_*_get_logs.txt
  4. fallback *get_logs*.txt files

Handles:
  - Kubernetes pod-prefixed logs:
      [pod/<pod-name>/<container-name>] actual log
  - C++ service logs:
      [timestamp] <error>: (File.cpp:line:function) message
  - JSON logs:
      MongoDB, Jaeger, OpenTelemetry style
  - Redis/plain text logs
  - nginx/openresty logs
  - empty files
  - noisy clone/build logs

Returns:
  logs[service] = feature dictionary
  meta = parser metadata
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# ---------------------------------------------------------------------
# Generic severity / category config
# ---------------------------------------------------------------------

SEVERITY_MAP = {
    "I": "info",
    "W": "warn",
    "E": "error",
    "F": "fatal",
    "D": "debug",
    "info": "info",
    "warn": "warn",
    "warning": "warn",
    "error": "error",
    "err": "error",
    "fatal": "fatal",
    "debug": "debug",
}


NOISE_KEYWORDS = [
    "please take the next action",
    "cloning into",
    "updating files:",
    "receiving objects:",
    "resolving deltas:",
    "checking out files:",
    "remote: enumerating objects",
    "remote: counting objects",
    "remote: compressing objects",
]


ERROR_TYPE_PATTERNS = {
    "dependency_timeout": [
        "serverselectiontimeoutms",
        "no suitable servers found",
        "deadline exceeded",
        "request timed out",
        "read timed out",
        "socket read timed out",
        "operation timed out",
        "i/o timeout",
    ],
    "dependency_connection": [
        "connection refused",
        "connection reset",
        "connection error",
        "broken pipe",
        "unknown connection error",
        "calling ismaster",
        "could not connect",
        "connection failed",
        "connect failed",
    ],
    "dependency_unavailable": [
        "unavailable",
        "service unavailable",
        "no route to host",
        "host unreachable",
        "temporary failure in name resolution",
        "name resolution",
        "dns",
    ],
    "rpc_transport_failure": [
        "ttransportexception",
        "thrift",
        "grpc",
        "rpc failed",
        "transport exception",
        "transport failure",
    ],
    "db_error": [
        "mongodb",
        "mongo",
        "postgres",
        "postgresql",
        "mysql",
        "redis",
        "memcached",
        "database",
        "db error",
    ],
    "index_init_failure": [
        "failed to create mongodb index",
        "createindexes",
        "create index",
        "index build failed",
    ],
    "auth_error": [
        "unauthorized",
        "authentication failed",
        "auth failed",
        "not authorized",
        "invalid credentials",
        "access denied",
    ],
    "permission_error": [
        "permission denied",
        "forbidden",
        "operation not permitted",
    ],
    "config_error": [
        "invalid configuration",
        "bad config",
        "misconfig",
        "config error",
        "configuration error",
        "invalid option",
    ],
    "oom_or_resource": [
        "oomkilled",
        "out of memory",
        "memory limit",
        "cpu throttling",
        "resource exhausted",
    ],
    "crash_or_panic": [
        "panic",
        "segmentation fault",
        "segfault",
        "core dumped",
        "fatal",
        "crash",
        "crashed",
    ],
    "http_error": [
        " 500 ",
        " 502 ",
        " 503 ",
        " 504 ",
        "status=500",
        "status=502",
        "status=503",
        "status=504",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    ],
    "timeout": [
        "timeout",
        "timed out",
    ],
    "generic_error": [
        "error",
        "failed",
        "failure",
        "exception",
    ],
}


ERROR_TYPE_WEIGHTS = {
    "dependency_timeout": 0.35,
    "dependency_connection": 0.35,
    "dependency_unavailable": 0.35,
    "rpc_transport_failure": 0.30,
    "db_error": 0.30,
    "index_init_failure": 0.25,
    "auth_error": 0.35,
    "permission_error": 0.30,
    "config_error": 0.30,
    "oom_or_resource": 0.35,
    "crash_or_panic": 0.40,
    "http_error": 0.25,
    "timeout": 0.25,
    "generic_error": 0.15,
}


DEPENDENCY_PATTERNS = [
    r"ismaster on '([^']+):(\d+)'",
    r"calling [^']* on '([^']+):(\d+)'",
    r"'([a-zA-Z0-9][a-zA-Z0-9\-\.]*):(\d+)'",
    r"host[:=]\s*['\"]?([a-zA-Z0-9][a-zA-Z0-9\-\.]*):(\d+)",
    r"server[:=]\s*['\"]?([a-zA-Z0-9][a-zA-Z0-9\-\.]*):(\d+)",
    r"connect(?:ion)? .*? ([a-zA-Z0-9][a-zA-Z0-9\-\.]*):(\d+)",
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def normalize_service_name(name: str) -> str:
    """
    Converts pod/container/file names into stable service-ish names.

    Examples:
      post-storage-service-59479466b5-9nxxw.log -> post-storage-service
      kind-control-plane.compose-post-service-9f655fc76-cvsfn -> compose-post-service
    """
    name = str(name or "").strip()

    if not name:
        return "unknown"

    name = Path(name).name

    if "." in name and name.startswith("kind-"):
        name = name.split(".", 1)[1]

    name = re.sub(r"\.(log|stderr|txt)$", "", name)

    # Remove common k8s pod suffix: -<hash>-<rand>
    name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{4,6}$", "", name)

    # Remove remaining replicaset hash
    name = re.sub(r"-[a-f0-9]{8,10}$", "", name)

    return name or "unknown"


def camel_to_service_name(class_or_file: str) -> str:
    """
    Examples:
      ComposePostService.cpp -> compose-post-service
      UserTimelineService -> user-timeline-service
      MongoDBAuthMissingAnalysis -> mongo-db-auth-missing-analysis
    """
    name = str(class_or_file or "").strip()
    name = Path(name).name

    name = re.sub(r"\.(cpp|cc|cxx|h|hpp|py|java|go|rs)$", "", name)
    name = re.sub(r"(Handler|Controller|Main)$", "", name)

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)

    return s2.lower() or "unknown"


def infer_client_service_from_dependency_target(target: str) -> str:
    """
    Generic fallback:
      post-storage-mongodb -> post-storage-service
      user-timeline-redis -> user-timeline-service
    """
    target = normalize_service_name(target)

    for suffix in [
        "-mongodb",
        "-mongo",
        "-redis",
        "-memcached",
        "-postgres",
        "-postgresql",
        "-mysql",
        "-db",
        "-database",
        "-cache",
    ]:
        if target.endswith(suffix):
            return target[: -len(suffix)] + "-service"

    return target


def infer_service_from_filename(path: Path) -> str:
    name = path.name

    if name.startswith("step_") and "get_logs" in name:
        return "unknown"

    return normalize_service_name(name)


# ---------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------

def extract_pod_prefix(line: str):
    """
    Parses:
      [pod/post-storage-service-xxx/post-storage-service] actual log

    Returns:
      pod, container, rest
    """
    m = re.match(r"^\[pod/([^/\]]+)/([^/\]]+)\]\s*(.*)$", line)
    if not m:
        return None, None, line

    return m.group(1), m.group(2), m.group(3)


def is_noise(line: str) -> bool:
    low = line.lower().strip()

    if not low:
        return True

    return any(k in low for k in NOISE_KEYWORDS)


def try_parse_json_log(line: str):
    line = line.strip()

    if not line.startswith("{"):
        return None

    try:
        obj = json.loads(line)
    except Exception:
        return None

    severity_raw = str(obj.get("s") or obj.get("level") or "").strip()
    severity = SEVERITY_MAP.get(severity_raw, severity_raw.lower() or "unknown")

    msg = str(obj.get("msg") or obj.get("message") or "")

    component = str(
        obj.get("c")
        or obj.get("caller")
        or obj.get("system")
        or obj.get("logger")
        or ""
    )

    timestamp = None
    if isinstance(obj.get("t"), dict):
        timestamp = obj["t"].get("$date")
    elif obj.get("ts") is not None:
        timestamp = obj.get("ts")
    elif obj.get("time") is not None:
        timestamp = obj.get("time")
    elif obj.get("timestamp") is not None:
        timestamp = obj.get("timestamp")

    return {
        "format": "json",
        "severity": severity,
        "message": msg,
        "component": component,
        "timestamp": timestamp,
        "raw_obj": obj,
    }


def try_parse_cpp_log(line: str):
    """
    Parses:
      [2026-May-01 00:39:43.277265] <error>: (utils_mongodb.h:76:CreateIndex) message
    """
    m = re.match(
        r"^\[(?P<ts>[^\]]+)\]\s+<(?P<sev>[^>]+)>:\s+\((?P<src>[^)]+)\)\s*(?P<msg>.*)$",
        line.strip(),
    )

    if not m:
        return None

    return {
        "format": "cpp",
        "severity": SEVERITY_MAP.get(m.group("sev").lower(), m.group("sev").lower()),
        "message": m.group("msg").strip(),
        "component": m.group("src").strip(),
        "timestamp": m.group("ts").strip(),
        "raw_obj": None,
    }


def try_parse_nginx_log(line: str):
    """
    Parses nginx/openresty style:
      2026/05/01 00:43:44 [error] ...
    """
    m = re.match(
        r"^(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(?P<sev>[^\]]+)\]\s+(?P<msg>.*)$",
        line.strip(),
    )

    if not m:
        return None

    return {
        "format": "nginx",
        "severity": SEVERITY_MAP.get(m.group("sev").lower(), m.group("sev").lower()),
        "message": m.group("msg").strip(),
        "component": "nginx/openresty",
        "timestamp": m.group("ts").strip(),
        "raw_obj": None,
    }


def parse_plain_log(line: str):
    low = line.lower()

    severity = "info"

    if any(p in low for pats in ERROR_TYPE_PATTERNS.values() for p in pats):
        severity = "error"
    elif "warn" in low or "warning" in low:
        severity = "warn"

    return {
        "format": "plain",
        "severity": severity,
        "message": line.strip(),
        "component": "",
        "timestamp": None,
        "raw_obj": None,
    }


def parse_line(line: str):
    pod, container, stripped = extract_pod_prefix(line)

    parsed = try_parse_json_log(stripped)
    if parsed is None:
        parsed = try_parse_cpp_log(stripped)
    if parsed is None:
        parsed = try_parse_nginx_log(stripped)
    if parsed is None:
        parsed = parse_plain_log(stripped)

    parsed["pod"] = pod
    parsed["container"] = container
    parsed["stripped_line"] = stripped

    return parsed


# ---------------------------------------------------------------------
# Error and dependency extraction
# ---------------------------------------------------------------------

def classify_error_type(message: str):
    low = str(message or "").lower()

    for err_type, patterns in ERROR_TYPE_PATTERNS.items():
        if any(p in low for p in patterns):
            return err_type

    return "none"


def extract_dependency_targets(message: str):
    message = str(message or "")
    targets = []

    for pat in DEPENDENCY_PATTERNS:
        for m in re.finditer(pat, message, flags=re.IGNORECASE):
            host = m.group(1)
            port = m.group(2) if len(m.groups()) >= 2 else None

            if not host:
                continue

            targets.append({
                "target": normalize_service_name(host),
                "raw_target": host,
                "port": port,
            })

    # Extra direct DB/cache target patterns
    for m in re.finditer(
        r"\b([a-zA-Z0-9][a-zA-Z0-9\-]*(?:mongodb|mongo|redis|memcached|postgres|mysql|db|cache)):(\d+)\b",
        message,
        flags=re.IGNORECASE,
    ):
        targets.append({
            "target": normalize_service_name(m.group(1)),
            "raw_target": m.group(1),
            "port": m.group(2),
        })

    seen = set()
    out = []

    for t in targets:
        key = (t["target"], t.get("port"))
        if key not in seen:
            out.append(t)
            seen.add(key)

    return out


def extract_http_info(message: str):
    msg = str(message or "")

    out = {}

    m = re.search(r"request:\s*\"([A-Z]+)\s+([^\" ]+)", msg)
    if m:
        out["http_method"] = m.group(1)
        out["http_path"] = m.group(2)

    m = re.search(r"\b(status|status_code|code)[=: ]+(\d{3})\b", msg, flags=re.IGNORECASE)
    if m:
        out["status_code"] = int(m.group(2))

    m = re.search(r"\b(5\d\d|4\d\d)\b", msg)
    if m and "status_code" not in out:
        out["status_code"] = int(m.group(1))

    return out


# ---------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------

def empty_service_log_state():
    return {
        "line_count": 0,
        "info_count": 0,
        "warn_count": 0,
        "error_count": 0,
        "fatal_count": 0,
        "debug_count": 0,

        "json_log_count": 0,
        "cpp_log_count": 0,
        "nginx_log_count": 0,
        "plain_log_count": 0,

        "noise_line_count": 0,
        "log_signal_present": False,

        "dominant_error_type": "none",
        "error_type_counts": {},
        "severity_counts": {},
        "component_counts": {},
        "operation_counts": {},
        "http_status_counts": {},

        "top_error_templates": [],
        "evidence_lines": [],

        "dependency_errors": [],
        "dependency_error_counts": {},
        "dependency_error_edges": [],

        "startup_signal_count": 0,
        "ready_signal_count": 0,

        "log_anomaly_score": 0.0,
        "log_health_score": 1.0,
    }


def update_service_state(state, parsed, original_line):
    state["line_count"] += 1
    state["log_signal_present"] = True

    fmt = parsed.get("format", "plain")
    if fmt == "json":
        state["json_log_count"] += 1
    elif fmt == "cpp":
        state["cpp_log_count"] += 1
    elif fmt == "nginx":
        state["nginx_log_count"] += 1
    else:
        state["plain_log_count"] += 1

    severity = parsed.get("severity", "unknown")
    msg = parsed.get("message", "")
    component = parsed.get("component", "")

    state.setdefault("_severity_counter", Counter())[severity] += 1

    if severity in ["error", "err", "e"]:
        state["error_count"] += 1
    elif severity in ["fatal", "f"]:
        state["fatal_count"] += 1
    elif severity in ["warn", "warning", "w"]:
        state["warn_count"] += 1
    elif severity == "debug":
        state["debug_count"] += 1
    else:
        low = msg.lower()
        err_type_fallback = classify_error_type(low)
        if err_type_fallback != "none":
            state["error_count"] += 1
        else:
            state["info_count"] += 1

    low = msg.lower()

    if any(x in low for x in ["starting", "server initialized", "mongodb starting", "redis is starting"]):
        state["startup_signal_count"] += 1

    if any(x in low for x in ["ready to accept connections", "waiting for connections", "server started", "health-status\":\"ready"]):
        state["ready_signal_count"] += 1

    err_type = classify_error_type(msg)

    if err_type != "none":
        state.setdefault("_error_type_counter", Counter())[err_type] += 1

        if len(state["evidence_lines"]) < 20:
            state["evidence_lines"].append(original_line.strip()[:700])

        if len(state["top_error_templates"]) < 20:
            state["top_error_templates"].append(msg[:500])

    if component:
        state.setdefault("_component_counter", Counter())[component] += 1

    http_info = extract_http_info(msg)
    if "status_code" in http_info:
        state.setdefault("_http_status_counter", Counter())[str(http_info["status_code"])] += 1

    if "http_path" in http_info:
        state.setdefault("_operation_counter", Counter())[http_info["http_path"]] += 1

    deps = extract_dependency_targets(msg)

    for dep in deps:
        dep_key = dep["target"]
        state.setdefault("_dependency_counter", Counter())[dep_key] += 1

        dep_item = {
            "target": dep["target"],
            "raw_target": dep.get("raw_target"),
            "port": dep.get("port"),
            "error_type": err_type,
            "message": msg[:500],
        }

        if len(state["dependency_errors"]) < 50:
            state["dependency_errors"].append(dep_item)


def finalize_service_state(state):
    err_counter = state.pop("_error_type_counter", Counter())
    comp_counter = state.pop("_component_counter", Counter())
    dep_counter = state.pop("_dependency_counter", Counter())
    severity_counter = state.pop("_severity_counter", Counter())
    http_status_counter = state.pop("_http_status_counter", Counter())
    operation_counter = state.pop("_operation_counter", Counter())

    state["error_type_counts"] = dict(err_counter)
    state["component_counts"] = dict(comp_counter.most_common(30))
    state["dependency_error_counts"] = dict(dep_counter)
    state["severity_counts"] = dict(severity_counter)
    state["http_status_counts"] = dict(http_status_counter)
    state["operation_counts"] = dict(operation_counter.most_common(30))

    if err_counter:
        state["dominant_error_type"] = err_counter.most_common(1)[0][0]
    else:
        state["dominant_error_type"] = "none"

    score = 0.0

    score += min(state["error_count"] * 0.04, 0.50)
    score += min(state["fatal_count"] * 0.20, 0.40)
    score += min(state["warn_count"] * 0.008, 0.12)
    score += min(sum(dep_counter.values()) * 0.04, 0.45)

    for err_type, count in err_counter.items():
        weight = ERROR_TYPE_WEIGHTS.get(err_type, 0.10)
        score += min(count * weight * 0.05, weight)

    # Give extra weight to repeated dependency edges.
    if dep_counter:
        score += 0.15

    score = max(0.0, min(score, 1.0))

    state["log_anomaly_score"] = round(score, 4)
    state["log_health_score"] = round(1.0 - score, 4)

    # Deduplicate templates and evidence.
    def dedupe_keep_order(xs, limit):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
            if len(out) >= limit:
                break
        return out

    state["top_error_templates"] = dedupe_keep_order(state["top_error_templates"], 10)
    state["evidence_lines"] = dedupe_keep_order(state["evidence_lines"], 20)

    return state


# ---------------------------------------------------------------------
# File discovery and service inference
# ---------------------------------------------------------------------

def discover_log_files(run_dir: Path):
    run_dir = Path(run_dir)
    candidates = []

    candidates.extend((run_dir / "direct_k8s_outputs" / "pod_logs").glob("*.log"))
    candidates.extend((run_dir / "direct_k8s_outputs" / "pod_logs").glob("*.stderr"))

    # candidates.extend((run_dir / "logs").glob("*get_logs*.txt"))
    candidates.extend((run_dir / "builtin_api_outputs" / "logs").glob("step_*_get_logs.txt"))

    candidates.extend(run_dir.glob("*get_logs*.txt"))

    seen = set()
    out = []

    for p in candidates:
        if not p.exists():
            continue
        key = str(p.resolve())
        if key not in seen:
            out.append(p)
            seen.add(key)

    return sorted(out)


def service_from_step_file(path: Path, content: str):
    """
    Infer service for logs/step_*_get_logs.txt without hardcoded maps.
    """

    # 1. Pod prefix is strongest signal.
    for line in content.splitlines():
        pod, container, _ = extract_pod_prefix(line)
        if container:
            return normalize_service_name(container)

    # 2. C++ source file signal.
    m = re.search(r"\(([A-Za-z0-9_./-]+):\d+:[^)]+\)", content)
    if m:
        return camel_to_service_name(m.group(1))

    # 3. JSON host field.
    for line in content.splitlines():
        parsed = try_parse_json_log(line)
        if parsed and parsed.get("raw_obj"):
            raw = parsed["raw_obj"]
            attr = raw.get("attr", {})
            if isinstance(attr, dict):
                host = attr.get("host")
                if host:
                    return normalize_service_name(host)

                doc = attr.get("doc")
                if isinstance(doc, dict):
                    app = doc.get("application")
                    if isinstance(app, dict) and app.get("name"):
                        return normalize_service_name(app["name"])

    # 4. Dependency target fallback.
    m = re.search(r"'([a-zA-Z0-9\-]+):(\d+)'", content)
    if m:
        return infer_client_service_from_dependency_target(m.group(1))

    # 5. Filename fallback.
    return infer_service_from_filename(path)


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def parse_logs(run_dir, services=None):
    """
    Main entrypoint used by build_state.py.

    Returns:
      logs, meta
    """
    run_dir = Path(run_dir)
    services = list(services or [])

    log_files = discover_log_files(run_dir)
    logs = defaultdict(empty_service_log_state)

    meta = {
        "files_seen": [],
        "files_parsed": 0,
        "empty_files": [],
        "parse_errors": [],
        "source_counts": {
            "pod_logs": 0,
            "get_logs": 0,
            "other": 0,
        },
        "dependency_error_edges_from_logs": [],
        "num_dependency_error_edges_from_logs": 0,
        "global_error_type_counts": {},
        "global_dependency_error_counts": {},
    }

    for path in log_files:
        text = safe_read_text(path)
        meta["files_seen"].append(str(path))

        if not text.strip():
            meta["empty_files"].append(str(path))
            continue

        path_str = str(path)

        if "pod_logs" in path_str:
            meta["source_counts"]["pod_logs"] += 1
            default_svc = infer_service_from_filename(path)
        elif "get_logs" in path.name:
            meta["source_counts"]["get_logs"] += 1
            default_svc = service_from_step_file(path, text)
        else:
            meta["source_counts"]["other"] += 1
            default_svc = infer_service_from_filename(path)

        for line in text.splitlines():
            if is_noise(line):
                logs[default_svc]["noise_line_count"] += 1
                continue

            try:
                parsed = parse_line(line)

                if parsed.get("container"):
                    svc = normalize_service_name(parsed["container"])
                else:
                    svc = default_svc

                if not svc or svc == "unknown":
                    svc = default_svc or "unknown"

                update_service_state(logs[svc], parsed, line)

            except Exception as e:
                meta["parse_errors"].append({
                    "file": str(path),
                    "line": line[:300],
                    "error": str(e),
                })

        meta["files_parsed"] += 1

    # Ensure discovered/known services exist even if their logs are empty.
    for svc in services:
        _ = logs[svc]

    finalized = {
        svc: finalize_service_state(dict(state))
        for svc, state in logs.items()
    }

    global_error_counter = Counter()
    global_dep_counter = Counter()
    dep_edges = []

    for src, feats in finalized.items():
        global_error_counter.update(feats.get("error_type_counts", {}))
        global_dep_counter.update(feats.get("dependency_error_counts", {}))

        for dep in feats.get("dependency_errors", []):
            edge = {
                "source": src,
                "target": dep["target"],
                "port": dep.get("port"),
                "error_type": dep.get("error_type"),
                "message": dep.get("message"),
            }
            dep_edges.append(edge)

    meta["dependency_error_edges_from_logs"] = dep_edges
    meta["num_dependency_error_edges_from_logs"] = len(dep_edges)
    meta["global_error_type_counts"] = dict(global_error_counter)
    meta["global_dependency_error_counts"] = dict(global_dep_counter)

    return finalized, meta