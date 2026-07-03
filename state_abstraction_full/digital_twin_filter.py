"""
digital_twin_filter.py

Given a state_abstraction.json (full or compressed), determine which services
are RELEVANT to the fault (keep in the digital twin) vs BYSTANDERS (delete
after full-stack deploy to save resources).

This feeds directly into the digital twin builder which:
  1. Deploys the full app stack (via existing Helm charts)
  2. Calls get_digital_twin_services() to get the bystander list
  3. Deletes bystander pods: kubectl delete pod <pod> -n <namespace>

Usage:
    from digital_twin_filter import get_digital_twin_services

    state = read_json("processed_state/state_abstraction.json")
    compressed = read_json("processed_state/state_abstraction_compressed.json")
    twin_info = get_digital_twin_services(state, compressed)

    # twin_info["bystander_pods"] → delete these from the live cluster
    # twin_info["relevant_services"] → keep these running
"""

from __future__ import annotations


def get_digital_twin_services(state: dict, compressed: dict | None = None) -> dict:
    """
    Classify every service as RELEVANT (fault-affected) or BYSTANDER (idle).

    Classification uses four signals, in decreasing authority:
      1. Known faulty service from the scenario spec (strongest)
      2. Top-ranked RCA candidates with score > 0.2
      3. service_health status is not "healthy"
      4. Service is in an error/infra cluster (from compressed state)
      5. Has a suspicious incoming trace edge (cascade effects — relevant but secondary)

    Services with NONE of these signals → bystander (safe to delete).

    Returns:
        {
            "namespace": str,
            "relevant_services": list[str],
            "bystander_services": list[str],
            "relevant_pods": list[str],
            "bystander_pods": list[str],
            "kubectl_delete_commands": list[str],  # ready-to-run commands
            "reasoning": dict[str, dict],
            "summary": dict,
        }
    """
    services = state.get("services", [])
    service_health = state.get("service_health", {})
    rca = state.get("rca_features", {})
    fault_ctx = state.get("fault_context", {})
    system = state.get("system", {})
    traces = state.get("traces", {})

    clusters: dict = {}
    if compressed:
        clusters = compressed.get("clusters", {})

    namespace = fault_ctx.get("target_namespace", "")

    # ── Signal sets ───────────────────────────────────────────────────────────

    known_faulty: set[str] = set()
    if fault_ctx.get("faulty_service"):
        known_faulty.add(fault_ctx["faulty_service"])

    rca_suspects: set[str] = set()
    for c in rca.get("candidate_root_causes", [])[:5]:
        if c.get("score", 0.0) > 0.2:
            rca_suspects.add(c["service"])
    top = rca.get("most_suspicious_service")
    if top:
        rca_suspects.add(top)

    unhealthy: set[str] = {
        svc for svc, h in service_health.items()
        if h.get("status", "healthy") != "healthy"
    }

    error_cluster: set[str] = set(clusters.get("log_error_or_dependency_failure", []))
    infra_cluster: set[str] = set(clusters.get("infra_unhealthy", []))

    # Services with suspicious incoming trace edges (cascade victims)
    cascade_affected: set[str] = set()
    for edge, feats in traces.items():
        if "->" not in edge:
            continue
        _, dst = edge.split("->", 1)
        if feats.get("is_suspicious") or feats.get("error_ratio", 0.0) > 0.3:
            cascade_affected.add(dst)

    # ── Classify ──────────────────────────────────────────────────────────────

    relevant: set[str] = set()
    bystanders: set[str] = set()
    reasoning: dict[str, dict] = {}

    for svc in services:
        reasons: list[str] = []

        if svc in known_faulty:
            reasons.append("known_faulty_service_from_spec")
        if svc in rca_suspects:
            reasons.append("top_rca_candidate")
        if svc in unhealthy:
            status = service_health.get(svc, {}).get("status", "unknown")
            reasons.append(f"service_health={status}")
        if svc in error_cluster:
            reasons.append("in_log_error_cluster")
        if svc in infra_cluster:
            reasons.append("in_infra_unhealthy_cluster")
        if svc in cascade_affected and not reasons:
            # Only count cascade as relevant if nothing else flagged it
            # (avoid pulling in truly healthy services hit by one bad edge)
            reasons.append("suspicious_incoming_trace_edge")

        if reasons:
            relevant.add(svc)
            reasoning[svc] = {"classification": "relevant", "reasons": reasons}
        else:
            bystanders.add(svc)
            reasoning[svc] = {"classification": "bystander", "reasons": ["no_fault_signal"]}

    # ── Pod-level breakdown from system state ─────────────────────────────────

    relevant_pods: list[str] = []
    bystander_pods: list[str] = []

    for svc in services:
        pod_names = system.get(svc, {}).get("pod_names", [])
        if svc in relevant:
            relevant_pods.extend(pod_names)
        else:
            bystander_pods.extend(pod_names)

    # ── Ready-to-run kubectl delete commands ──────────────────────────────────

    kubectl_cmds: list[str] = []
    if namespace and bystander_pods:
        for pod in sorted(bystander_pods):
            kubectl_cmds.append(
                f"kubectl --context kind-kind delete pod {pod} -n {namespace} --ignore-not-found"
            )

    return {
        "namespace": namespace,
        "relevant_services": sorted(relevant),
        "bystander_services": sorted(bystanders),
        "relevant_pods": sorted(relevant_pods),
        "bystander_pods": sorted(bystander_pods),
        "kubectl_delete_commands": kubectl_cmds,
        "reasoning": reasoning,
        "summary": {
            "total_services": len(services),
            "relevant_count": len(relevant),
            "bystander_count": len(bystanders),
            "known_faulty": sorted(known_faulty),
            "rca_suspects": sorted(rca_suspects),
            "unhealthy_services": sorted(unhealthy),
        },
    }


def print_twin_summary(twin_info: dict) -> None:
    """Pretty-print the digital twin classification for debugging."""
    print(f"\n[Digital Twin Filter] namespace={twin_info['namespace']}")
    print(f"  Relevant ({twin_info['summary']['relevant_count']}): "
          f"{', '.join(twin_info['relevant_services'])}")
    print(f"  Bystanders ({twin_info['summary']['bystander_count']}): "
          f"{', '.join(twin_info['bystander_services'])}")
    print(f"  Known faulty: {twin_info['summary']['known_faulty']}")
    print(f"  RCA suspects: {twin_info['summary']['rca_suspects']}")
    if twin_info["kubectl_delete_commands"]:
        print(f"  kubectl commands to trim twin ({len(twin_info['kubectl_delete_commands'])}):")
        for cmd in twin_info["kubectl_delete_commands"][:5]:
            print(f"    {cmd}")
        if len(twin_info["kubectl_delete_commands"]) > 5:
            print(f"    ... and {len(twin_info['kubectl_delete_commands']) - 5} more")


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Identify digital twin relevant vs bystander services")
    ap.add_argument("--state", required=True, help="Path to state_abstraction.json")
    ap.add_argument("--compressed", default=None, help="Path to state_abstraction_compressed.json")
    ap.add_argument("--output", default=None, help="Write result JSON to this path")
    args = ap.parse_args()

    with open(args.state) as f:
        state = json.load(f)
    compressed = None
    if args.compressed:
        with open(args.compressed) as f:
            compressed = json.load(f)

    result = get_digital_twin_services(state, compressed)
    print_twin_summary(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] wrote twin filter result to {args.output}")
