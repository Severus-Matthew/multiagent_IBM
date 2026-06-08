# system_parser.py

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None


def normalize_service_name(name: str) -> str:
    name = str(name or "").strip()
    name = Path(name).name
    name = re.sub(r"\.(json|txt|log)$", "", name)
    name = re.sub(r"-[a-f0-9]{8,10}-[a-z0-9]{4,6}$", "", name)
    name = re.sub(r"-[a-f0-9]{8,10}$", "", name)
    return name or "unknown"


def get_items(obj):
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return obj["items"]
    if isinstance(obj, list):
        return obj
    return []


def bool_condition(status_obj, cond_type):
    for c in status_obj.get("conditions", []) or []:
        if c.get("type") == cond_type:
            return str(c.get("status")).lower() == "true"
    return False


def extract_waiting_terminated_reasons(container_statuses):
    waiting = []
    terminated = []
    running = []

    for cs in container_statuses or []:
        state = cs.get("state", {}) or {}

        if "waiting" in state:
            waiting.append({
                "container": cs.get("name"),
                "reason": state["waiting"].get("reason"),
                "message": state["waiting"].get("message"),
            })

        if "terminated" in state:
            terminated.append({
                "container": cs.get("name"),
                "reason": state["terminated"].get("reason"),
                "exit_code": state["terminated"].get("exitCode"),
                "message": state["terminated"].get("message"),
            })

        if "running" in state:
            running.append({
                "container": cs.get("name"),
                "started_at": state["running"].get("startedAt"),
            })

    return waiting, terminated, running


def parse_pods_json(path: Path):
    obj = safe_read_json(path)
    service_state = defaultdict(lambda: {
        "pods_total": 0,
        "pods_ready": 0,
        "pods_unready": 0,
        "running_count": 0,
        "pending_count": 0,
        "failed_count": 0,
        "succeeded_count": 0,
        "unknown_phase_count": 0,
        "restart_count": 0,
        "crashloop_count": 0,
        "waiting_count": 0,
        "terminated_count": 0,
        "oomkilled_count": 0,
        "image_pull_error_count": 0,
        "pod_names": [],
        "pod_ips": [],
        "nodes": [],
        "containers": [],
        "images": [],
        "commands": [],
        "ports": [],
        "qos_classes": Counter(),
        "condition_counts": Counter(),
        "waiting_reasons": Counter(),
        "terminated_reasons": Counter(),
        "owner_kinds": Counter(),
        "owner_names": Counter(),
        "service_health_status": "unknown",
        "infra_issue_flag": False,
        "pod_details": [],
    })

    pod_to_service = {}

    for pod in get_items(obj):
        meta = pod.get("metadata", {}) or {}
        spec = pod.get("spec", {}) or {}
        status = pod.get("status", {}) or {}

        pod_name = meta.get("name", "unknown")
        labels = meta.get("labels", {}) or {}

        svc = (
            labels.get("service")
            or labels.get("app")
            or spec.get("hostname")
            or normalize_service_name(pod_name)
        )
        svc = normalize_service_name(svc)

        pod_to_service[pod_name] = svc

        s = service_state[svc]
        s["pods_total"] += 1
        s["pod_names"].append(pod_name)

        phase = status.get("phase", "Unknown")
        if phase == "Running":
            s["running_count"] += 1
        elif phase == "Pending":
            s["pending_count"] += 1
        elif phase == "Failed":
            s["failed_count"] += 1
        elif phase == "Succeeded":
            s["succeeded_count"] += 1
        else:
            s["unknown_phase_count"] += 1

        if status.get("podIP"):
            s["pod_ips"].append(status.get("podIP"))

        if spec.get("nodeName"):
            s["nodes"].append(spec.get("nodeName"))

        if status.get("qosClass"):
            s["qos_classes"][status.get("qosClass")] += 1

        ready = bool_condition(status, "Ready")
        containers_ready = bool_condition(status, "ContainersReady")

        if phase == "Running" and ready and containers_ready:
            s["pods_ready"] += 1
        else:
            s["pods_unready"] += 1

        for c in status.get("conditions", []) or []:
            key = f"{c.get('type')}={c.get('status')}"
            s["condition_counts"][key] += 1

        container_statuses = status.get("containerStatuses", []) or []
        waiting, terminated, running = extract_waiting_terminated_reasons(container_statuses)

        s["waiting_count"] += len(waiting)
        s["terminated_count"] += len(terminated)

        for w in waiting:
            reason = w.get("reason") or "unknown"
            s["waiting_reasons"][reason] += 1
            if reason in ["CrashLoopBackOff"]:
                s["crashloop_count"] += 1
            if reason in ["ImagePullBackOff", "ErrImagePull"]:
                s["image_pull_error_count"] += 1

        for t in terminated:
            reason = t.get("reason") or "unknown"
            s["terminated_reasons"][reason] += 1
            if reason == "OOMKilled":
                s["oomkilled_count"] += 1

        for cs in container_statuses:
            s["restart_count"] += int(cs.get("restartCount", 0) or 0)

        for owner in meta.get("ownerReferences", []) or []:
            if owner.get("kind"):
                s["owner_kinds"][owner.get("kind")] += 1
            if owner.get("name"):
                s["owner_names"][owner.get("name")] += 1

        for c in spec.get("containers", []) or []:
            if c.get("name"):
                s["containers"].append(c.get("name"))
            if c.get("image"):
                s["images"].append(c.get("image"))
            if c.get("command"):
                s["commands"].extend(c.get("command"))
            for p in c.get("ports", []) or []:
                s["ports"].append({
                    "container_port": p.get("containerPort"),
                    "protocol": p.get("protocol"),
                    "name": p.get("name"),
                })

        s["pod_details"].append({
            "pod_name": pod_name,
            "namespace": meta.get("namespace"),
            "phase": phase,
            "ready": ready,
            "containers_ready": containers_ready,
            "pod_ip": status.get("podIP"),
            "host_ip": status.get("hostIP"),
            "node": spec.get("nodeName"),
            "start_time": status.get("startTime"),
            "restart_count": sum(int(cs.get("restartCount", 0) or 0) for cs in container_statuses),
            "waiting": waiting,
            "terminated": terminated,
            "running": running,
            "qos_class": status.get("qosClass"),
        })

    return service_state, pod_to_service


def parse_deployments_json(path: Path):
    obj = safe_read_json(path)
    out = {}

    for dep in get_items(obj):
        meta = dep.get("metadata", {}) or {}
        spec = dep.get("spec", {}) or {}
        status = dep.get("status", {}) or {}

        labels = meta.get("labels", {}) or {}
        name = meta.get("name", "unknown")
        svc = normalize_service_name(labels.get("service") or labels.get("app") or name)

        containers = []
        for c in (
            spec.get("template", {})
            .get("spec", {})
            .get("containers", [])
            or []
        ):
            containers.append({
                "name": c.get("name"),
                "image": c.get("image"),
                "command": c.get("command", []),
                "args": c.get("args", []),
                "ports": c.get("ports", []),
                "resources": c.get("resources", {}),
                "volume_mounts": c.get("volumeMounts", []),
            })

        conditions = {}
        for c in status.get("conditions", []) or []:
            conditions[c.get("type")] = {
                "status": c.get("status"),
                "reason": c.get("reason"),
                "message": c.get("message"),
            }

        out[svc] = {
            "deployment_name": name,
            "namespace": meta.get("namespace"),
            "replicas_desired": spec.get("replicas", 0),
            "replicas_current": status.get("replicas", 0),
            "replicas_ready": status.get("readyReplicas", 0),
            "replicas_available": status.get("availableReplicas", 0),
            "replicas_unavailable": status.get("unavailableReplicas", 0),
            "updated_replicas": status.get("updatedReplicas", 0),
            "observed_generation": status.get("observedGeneration"),
            "generation": meta.get("generation"),
            "selector": spec.get("selector", {}),
            "strategy": spec.get("strategy", {}),
            "conditions": conditions,
            "containers": containers,
            "volumes": (
                spec.get("template", {})
                .get("spec", {})
                .get("volumes", [])
                or []
            ),
        }

    return out


def parse_replicasets_json(path: Path):
    obj = safe_read_json(path)
    out = {}

    for rs in get_items(obj):
        meta = rs.get("metadata", {}) or {}
        spec = rs.get("spec", {}) or {}
        status = rs.get("status", {}) or {}
        labels = meta.get("labels", {}) or {}

        svc = normalize_service_name(labels.get("service") or labels.get("app") or meta.get("name"))

        out.setdefault(svc, [])

        out[svc].append({
            "replicaset_name": meta.get("name"),
            "namespace": meta.get("namespace"),
            "replicas_desired": spec.get("replicas", 0),
            "replicas_current": status.get("replicas", 0),
            "replicas_ready": status.get("readyReplicas", 0),
            "replicas_available": status.get("availableReplicas", 0),
            "fully_labeled_replicas": status.get("fullyLabeledReplicas", 0),
            "owner_references": meta.get("ownerReferences", []),
            "selector": spec.get("selector", {}),
        })

    return out


def parse_endpoints_json(path: Path):
    obj = safe_read_json(path)
    out = {}

    for ep in get_items(obj):
        meta = ep.get("metadata", {}) or {}
        svc = normalize_service_name(meta.get("name"))

        addresses = []
        not_ready_addresses = []
        ports = []

        for subset in ep.get("subsets", []) or []:
            for a in subset.get("addresses", []) or []:
                addresses.append({
                    "ip": a.get("ip"),
                    "node": a.get("nodeName"),
                    "pod": (a.get("targetRef") or {}).get("name"),
                    "kind": (a.get("targetRef") or {}).get("kind"),
                })

            for a in subset.get("notReadyAddresses", []) or []:
                not_ready_addresses.append({
                    "ip": a.get("ip"),
                    "node": a.get("nodeName"),
                    "pod": (a.get("targetRef") or {}).get("name"),
                    "kind": (a.get("targetRef") or {}).get("kind"),
                })

            for p in subset.get("ports", []) or []:
                ports.append({
                    "name": p.get("name"),
                    "port": p.get("port"),
                    "protocol": p.get("protocol"),
                })

        out[svc] = {
            "endpoint_name": meta.get("name"),
            "namespace": meta.get("namespace"),
            "ready_endpoint_count": len(addresses),
            "not_ready_endpoint_count": len(not_ready_addresses),
            "addresses": addresses,
            "not_ready_addresses": not_ready_addresses,
            "ports": ports,
            "has_ready_endpoints": len(addresses) > 0,
        }

    return out


def parse_events_txt(path: Path):
    text = safe_read_text(path)
    events_by_service = defaultdict(list)
    global_events = []

    bad_reasons = {
        "Failed", "FailedScheduling", "FailedMount", "FailedCreate",
        "FailedPull", "BackOff", "Unhealthy", "Killing",
        "CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff",
    }

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("LAST SEEN"):
            continue

        parts = re.split(r"\s+", line, maxsplit=4)
        if len(parts) < 5:
            continue

        last_seen, typ, reason, obj, msg = parts
        obj_name = obj.split("/")[-1] if "/" in obj else obj
        svc = normalize_service_name(obj_name)

        ev = {
            "last_seen": last_seen,
            "type": typ,
            "reason": reason,
            "object": obj,
            "message": msg,
            "is_warning": typ.lower() == "warning" or reason in bad_reasons,
        }

        events_by_service[svc].append(ev)
        global_events.append(ev)

    return dict(events_by_service), global_events


def parse_pods_wide_txt(path: Path):
    text = safe_read_text(path)
    out = {}

    lines = [x for x in text.splitlines() if x.strip()]
    if len(lines) < 2:
        return out

    header = re.split(r"\s+", lines[0].strip())

    for line in lines[1:]:
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 5:
            continue

        name = parts[0]
        ready = parts[1]
        status = parts[2]
        restarts = parts[3]
        age = parts[4]
        ip = parts[5] if len(parts) > 5 else None
        node = parts[6] if len(parts) > 6 else None

        svc = normalize_service_name(name)

        ready_num = 0
        ready_den = 0
        if "/" in ready:
            a, b = ready.split("/", 1)
            try:
                ready_num = int(a)
                ready_den = int(b)
            except Exception:
                pass

        out[svc] = {
            "pod_name": name,
            "ready_raw": ready,
            "ready_containers": ready_num,
            "total_containers": ready_den,
            "status": status,
            "restarts": int(restarts) if str(restarts).isdigit() else 0,
            "age": age,
            "ip": ip,
            "node": node,
        }

    return out

def parse_services_json(path: Path):
    obj = safe_read_json(path)
    out = {}

    for svc_obj in get_items(obj):
        meta = svc_obj.get("metadata", {}) or {}
        spec = svc_obj.get("spec", {}) or {}

        name = normalize_service_name(meta.get("name"))

        ports = []
        for p in spec.get("ports", []) or []:
            ports.append({
                "name": p.get("name"),
                "port": p.get("port"),
                "target_port": p.get("targetPort"),
                "protocol": p.get("protocol"),
            })

        out[name] = {
            "service_name": meta.get("name"),
            "namespace": meta.get("namespace"),
            "service_type": spec.get("type"),
            "cluster_ip": spec.get("clusterIP"),
            "cluster_ips": spec.get("clusterIPs", []),
            "selector": spec.get("selector", {}),
            "ports": ports,
            "session_affinity": spec.get("sessionAffinity"),
            "internal_traffic_policy": spec.get("internalTrafficPolicy"),
        }

    return out
def parse_top_table(path: Path):
    text = safe_read_text(path)
    rows = []

    lines = [x for x in text.splitlines() if x.strip()]
    if len(lines) < 2:
        return rows

    header = re.split(r"\s+", lines[0].strip())

    for line in lines[1:]:
        parts = re.split(r"\s+", line.strip())
        if len(parts) < len(header):
            continue

        row = dict(zip(header, parts))
        rows.append(row)

    return rows

def compute_service_health_status(s):
    if s.get("crashloop_count", 0) > 0:
        return "crashloop"
    if s.get("image_pull_error_count", 0) > 0:
        return "image_pull_error"
    if s.get("oomkilled_count", 0) > 0:
        return "oomkilled"
    if s.get("pending_count", 0) > 0:
        return "pending"
    if s.get("failed_count", 0) > 0:
        return "failed"
    if s.get("pods_unready", 0) > 0:
        return "unready"
    if s.get("pods_ready", 0) > 0:
        return "healthy"
    return "unknown"


def finalize_service_state(s):
    s["qos_classes"] = dict(s.get("qos_classes", Counter()))
    s["condition_counts"] = dict(s.get("condition_counts", Counter()))
    s["waiting_reasons"] = dict(s.get("waiting_reasons", Counter()))
    s["terminated_reasons"] = dict(s.get("terminated_reasons", Counter()))
    s["owner_kinds"] = dict(s.get("owner_kinds", Counter()))
    s["owner_names"] = dict(s.get("owner_names", Counter()))

    for key in ["pod_names", "pod_ips", "nodes", "containers", "images", "commands"]:
        s[key] = sorted(set(s.get(key, [])))

    s["service_health_status"] = compute_service_health_status(s)

    s["infra_issue_flag"] = s["service_health_status"] not in ["healthy", "unknown"]

    s["availability_ratio"] = (
        s["pods_ready"] / s["pods_total"]
        if s["pods_total"] else 0.0
    )

    return s


def merge_workload_objects(system, deployments, replicasets, services_json, endpoints, events_by_service):
    all_svcs = (
        set(system)
        | set(deployments)
        | set(replicasets)
        | set(services_json)
        | set(endpoints)
        | set(events_by_service)
    )
    for svc in all_svcs:
        system.setdefault(svc, finalize_service_state(defaultdict(int)))

        if svc in deployments:
            system[svc]["deployment"] = deployments[svc]

        if svc in replicasets:
            system[svc]["replicasets"] = replicasets[svc]
        if svc in services_json:
            system[svc]["services"] = services_json[svc]

        if svc in endpoints:
            system[svc]["endpoints"] = endpoints[svc]

            if endpoints[svc]["not_ready_endpoint_count"] > 0:
                system[svc]["infra_issue_flag"] = True
                system[svc]["service_health_status"] = "endpoint_not_ready"

            if endpoints[svc]["ready_endpoint_count"] == 0:
                system[svc]["infra_issue_flag"] = True
                system[svc]["service_health_status"] = "no_ready_endpoints"

        if svc in events_by_service:
            system[svc]["events"] = events_by_service[svc]
            warning_count = sum(1 for e in events_by_service[svc] if e.get("is_warning"))
            system[svc]["warning_event_count"] = warning_count

            if warning_count > 0:
                system[svc]["infra_issue_flag"] = True

    return system


def parse_collection_status(path: Path):
    obj = safe_read_json(path)
    if not isinstance(obj, dict):
        return {}

    failed = {}
    ok = {}

    for name, info in obj.items():
        if not isinstance(info, dict):
            continue
        if info.get("returncode") == 0:
            ok[name] = info.get("cmd")
        else:
            failed[name] = {
                "cmd": info.get("cmd"),
                "returncode": info.get("returncode"),
                "stderr": info.get("stderr"),
            }

    return {
        "ok_collections": ok,
        "failed_collections": failed,
        "num_ok": len(ok),
        "num_failed": len(failed),
    }


def parse_system(run_dir, services=None):
    run_dir = Path(run_dir)
    services = list(services or [])

    d = run_dir / "direct_k8s_outputs"

    pods_json = d / "pods.json"
    if not pods_json.exists():
        alt = d / "pods(1).json"
        if alt.exists():
            pods_json = alt

    service_state, pod_to_service = parse_pods_json(pods_json)
    deployments = parse_deployments_json(d / "deployments.json")
    replicasets = parse_replicasets_json(d / "replicasets.json")
    endpoints = parse_endpoints_json(d / "endpoints.json")
    events_by_service, global_events = parse_events_txt(d / "events.txt")
    pods_wide = parse_pods_wide_txt(d / "pods_wide.txt")
    collection_status = parse_collection_status(d / "collection_status.json")
    services_json = parse_services_json(d / "services.json")
    top_pods = parse_top_table(d / "top_pods.txt")
    top_nodes = parse_top_table(d / "top_nodes.txt")    

    finalized = {
        svc: finalize_service_state(dict(state))
        for svc, state in service_state.items()
    }

    finalized = merge_workload_objects(
    finalized,
    deployments,
    replicasets,
    services_json,
    endpoints,
    events_by_service,
)

    for svc in services:
        finalized.setdefault(svc, finalize_service_state(defaultdict(int)))

    meta = {
        "files_used": {
            "pods_json": str(pods_json) if pods_json.exists() else None,
            "services_json": str(d / "services.json") if (d / "services.json").exists() else None,
            "deployments_json": str(d / "deployments.json") if (d / "deployments.json").exists() else None,
            "replicasets_json": str(d / "replicasets.json") if (d / "replicasets.json").exists() else None,
            "endpoints_json": str(d / "endpoints.json") if (d / "endpoints.json").exists() else None,
            "events_txt": str(d / "events.txt") if (d / "events.txt").exists() else None,
            "pods_wide_txt": str(d / "pods_wide.txt") if (d / "pods_wide.txt").exists() else None,
            "top_pods_txt": str(d / "top_pods.txt") if (d / "top_pods.txt").exists() else None,
            "top_nodes_txt": str(d / "top_nodes.txt") if (d / "top_nodes.txt").exists() else None,
            "collection_status_json": str(d / "collection_status.json") if (d / "collection_status.json").exists() else None,
        },
        "pod_to_service": pod_to_service,
        "pods_wide": pods_wide,
        "top_pods": top_pods,
        "top_nodes": top_nodes,
        "collection_status": collection_status,
        "global_events_count": len(global_events),
        "warning_events_count": sum(1 for e in global_events if e.get("is_warning")),
        "unhealthy_services": [
            svc for svc, s in finalized.items()
            if s.get("infra_issue_flag")
        ],
    }

    return finalized, meta