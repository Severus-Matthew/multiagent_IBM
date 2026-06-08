from pathlib import Path
from utils import read_json, normalize_service_name

from pathlib import Path

def discover_run_files(run_dir):
    run_dir = Path(run_dir)

    return {
        "spec": run_dir / "spec.json",
        "run_result": run_dir / "run_result.json",
        "agent_transcript": run_dir / "agent_transcript.json",

        "pod_logs_dir": run_dir / "direct_k8s_outputs" / "pod_logs",
        "pods_json": run_dir / "direct_k8s_outputs" / "pods.json",
        "services_json": run_dir / "direct_k8s_outputs" / "services.json",
        "deployments_json": run_dir / "direct_k8s_outputs" / "deployments.json",

        "metrics_csv_files": sorted((run_dir /"builtin_api_outputs" / "metrics"/"metric"/"container").rglob("kpi_*.csv")),
        "trace_csv_files": sorted((run_dir / "builtin_api_outputs" / "traces").glob("*.csv")),

        "topology_json": run_dir / "topology" / "topology.json",
        "graph_json": run_dir / "topology" / "graph.json",
    }


def discover_services(run_dir):
    paths = discover_run_files(run_dir)
    services = set()

    services_obj = read_json(run_dir / "direct_k8s_outputs" / "services.json") or read_json(run_dir / "services.json")
    if isinstance(services_obj, dict):
        if "items" in services_obj:
            for item in services_obj.get("items", []):
                name = item.get("metadata", {}).get("name")
                if name:
                    services.add(name)
        elif "services" in services_obj:
            for item in services_obj.get("services", []):
                name = item.get("name") or item.get("service")
                if name:
                    services.add(name)

    topo = read_json(run_dir/"topology" / "topology.json") or read_json(paths["topology"] / "graph.json") or read_json(run_dir / "topology.json")
    if isinstance(topo, dict):
        for item in topo.get("services", []):
            if isinstance(item, str):
                services.add(item)
            elif isinstance(item, dict) and item.get("name"):
                services.add(item["name"])
        for item in topo.get("nodes", []):
            if isinstance(item, str):
                services.add(item)
            elif isinstance(item, dict):
                services.add(item.get("service") or item.get("name") or item.get("id"))

    pods_obj = read_json(run_dir / "direct_k8s_outputs" / "pods.json") or read_json(run_dir / "pods.json")
    if isinstance(pods_obj, dict):
        for pod in pods_obj.get("items", []):
            labels = pod.get("metadata", {}).get("labels", {}) or {}
            name = labels.get("service") or labels.get("app") or pod.get("spec", {}).get("hostname")
            if name:
                services.add(name)

    for folder in [run_dir / "direct_k8s_outputs" / "pod_logs", run_dir / "builtin_api_outputs" / "logs"]:
        if folder.exists():
            for f in list(folder.glob("*.log")) + list(folder.glob("*.stderr")) + list(folder.glob("*.txt")):
                services.add(normalize_service_name(f.name))

    return sorted(x for x in services if x and x != "unknown")
