#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def stable_key(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def round_float(x, ndigits=4):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return round(x, ndigits)
    return x


def round_deep(x):
    if isinstance(x, dict):
        return {k: round_deep(v) for k, v in x.items()}
    if isinstance(x, list):
        return [round_deep(v) for v in x]
    return round_float(x)


def group_identical_named_objects(named_objects, names_key="names", value_key="value"):
    groups = {}

    for name, obj in named_objects.items():
        obj = round_deep(obj)
        key = stable_key(obj)

        if key not in groups:
            groups[key] = {
                names_key: [],
                value_key: obj,
            }

        groups[key][names_key].append(name)

    out = []
    for g in groups.values():
        g[names_key] = sorted(g[names_key])
        out.append(g)

    return out


def compress_metric_block(metric_block):
    """
    Input:
      {
        "items": {
          "eth0": {"signal": "dynamic", ...},
          "gre0": {"signal": "zero", "count": 20},
          ...
        },
        "summary": {...}
      }

    Output:
      {
        "items_grouped": [
          {"names": ["gre0", "ip6gre0"], "value": {"signal": "zero", "count": 20}},
          {"names": ["eth0"], "value": {"signal": "dynamic", ...}}
        ],
        "summary": {...}   # only when num_items > 1
      }
    """
    items = metric_block.get("items", {})
    summary = metric_block.get("summary", {})

    out = {
        "items_grouped": group_identical_named_objects(items)
    }

    if summary.get("num_items", len(items)) > 1:
        cleaned_summary = dict(summary)

        # Compact these list fields naturally.
        for k in ["zero_items", "dynamic_items"]:
            if k in cleaned_summary and isinstance(cleaned_summary[k], list):
                cleaned_summary[k] = sorted(cleaned_summary[k])

        out["summary"] = cleaned_summary

    return round_deep(out)


def compress_metrics(metrics):
    out = {}

    for service, svc_obj in metrics.items():
        svc_out = {}

        if "groups" in svc_obj:
            svc_out["groups"] = {}

            for category, category_obj in svc_obj.get("groups", {}).items():
                svc_out["groups"][category] = {}

                for metric_name, metric_block in category_obj.items():
                    svc_out["groups"][category][metric_name] = compress_metric_block(metric_block)

        if "flat_summary" in svc_obj:
            svc_out["flat_summary"] = round_deep(svc_obj["flat_summary"])

        out[service] = svc_out

    return out


def compress_logs(logs):
    """
    Groups repeated log profiles like clean services:
      media-memcached + post-storage-memcached + ...
    """
    return {
        "grouped_profiles": [
            {
                "services": g["names"],
                "log_profile": g["value"],
            }
            for g in group_identical_named_objects(logs)
        ]
    }


def compress_ports(ports):
    """
    Groups ports by same protocol + name.
    container_port becomes a list.

    Input:
      [
        {"container_port": 5775, "protocol": "TCP", "name": null},
        {"container_port": 6831, "protocol": "TCP", "name": null}
      ]

    Output:
      [
        {"protocol": "TCP", "name": null, "container_ports": [5775, 6831]}
      ]
    """
    grouped = {}

    for p in ports or []:
        if not isinstance(p, dict):
            continue

        protocol = p.get("protocol")
        name = p.get("name")
        port = p.get("container_port", p.get("port", p.get("target_port")))

        key = stable_key({
            "protocol": protocol,
            "name": name,
        })

        if key not in grouped:
            grouped[key] = {
                "protocol": protocol,
                "name": name,
                "container_ports": [],
            }

        if port is not None:
            grouped[key]["container_ports"].append(port)

    out = []
    for item in grouped.values():
        item["container_ports"] = sorted(set(item["container_ports"]))
        out.append(item)

    return out


def normalize_system_profile(profile):
    """
    Apply local structural compression inside each service profile.
    """
    p = dict(profile)

    if "ports" in p:
        p["ports"] = compress_ports(p.get("ports", []))

    if isinstance(p.get("k8s_service"), dict) and "ports" in p["k8s_service"]:
        p["k8s_service"] = dict(p["k8s_service"])
        p["k8s_service"]["ports"] = compress_ports(p["k8s_service"].get("ports", []))

    if isinstance(p.get("endpoints"), dict) and "ports" in p["endpoints"]:
        p["endpoints"] = dict(p["endpoints"])
        p["endpoints"]["ports"] = compress_ports(p["endpoints"].get("ports", []))

    return round_deep(p)


def compress_system(system):
    normalized = {
        service: normalize_system_profile(profile)
        for service, profile in system.items()
    }

    return {
        "grouped_profiles": [
            {
                "services": g["names"],
                "system_profile": g["value"],
            }
            for g in group_identical_named_objects(normalized)
        ]
    }


# def compress_state(state):
    # compressed = {
    #     "timestamp": state.get("timestamp"),
    #     "scenario_id": state.get("scenario_id"),
    #     "state_type": "compressed_aiops_state_abstraction_v2_clean",
    #     "source_state_type": state.get("source_state_type") or state.get("state_type"),
    #     "fault_context": state.get("fault_context", {}),
    #     "services": state.get("services", []),
    # }

    # compressed["metrics"] = compress_metrics(state.get("metrics", {}))
    # compressed["logs"] = compress_logs(state.get("logs", {}))
    # compressed["system"] = compress_system(state.get("system", {}))

    # for key in [
    #     "traces",
    #     "workload",
    #     "graph",
    #     "sla",
    #     "rca_features",
    #     "service_health",
    #     "clusters",
    #     "llm_view",
    #     "model_table",
    #     "model_table_zscore",
    # ]:
    #     if key in state:
    #         compressed[key] = round_deep(state[key])

    # return compressed
def compress_state(state):
    compressed = dict(state)

    compressed["state_type"] = "compressed_aiops_state_abstraction_v2_clean"

    if "metrics" in state:
        compressed["metrics"] = compress_metrics(state["metrics"])

    if "logs" in state:
        compressed["logs"] = compress_logs(state["logs"])

    if "system" in state:
        compressed["system"] = compress_system(state["system"])

    return compressed

def dumps_compact_lists(obj, indent=2):
    """
    Pretty-print dicts, but keep scalar lists on one line.

    Example:
      "services": ["a", "b", "c"]
    instead of one item per line.
    """
    def is_scalar(x):
        return x is None or isinstance(x, (str, int, float, bool))

    def render(x, level=0):
        pad = " " * (indent * level)
        next_pad = " " * (indent * (level + 1))

        if isinstance(x, dict):
            if not x:
                return "{}"

            parts = []
            for k, v in x.items():
                parts.append(
                    f'{next_pad}{json.dumps(k)}: {render(v, level + 1)}'
                )

            return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

        if isinstance(x, list):
            if not x:
                return "[]"

            if all(is_scalar(i) for i in x):
                return json.dumps(x, separators=(", ", ": "))

            parts = [f"{next_pad}{render(i, level + 1)}" for i in x]
            return "[\n" + ",\n".join(parts) + "\n" + pad + "]"

        return json.dumps(x)

    return render(obj, 0)


def write_json_compact_lists(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(dumps_compact_lists(obj))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name("state_abstraction_compressed_clean.json")

    state = read_json(inp)
    compressed = compress_state(state)

    write_json_compact_lists(compressed, out)

    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()