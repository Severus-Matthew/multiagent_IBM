import argparse
from pathlib import Path
from utils import read_json, read_jsonl, write_json, ensure_dir
# from digital_twin_runtime import MinimalDigitalTwin
from behavioral_simulator import AbstractBehavioralSimulator


def run_check(processed_dir, service=None, fault_type=None, action="rollback_config"):
    processed_dir = Path(processed_dir)
    states = read_jsonl(processed_dir / "states.jsonl")
    graph = read_json(processed_dir / "graph.json")
    if not states:
        raise RuntimeError("No states found")
    base = states[0]
    hyp = base.get("rca_features", {}).get("hypothesis", {})
    rca = {"service": service or hyp.get("service") or base.get("fault_context", {}).get("faulty_service"), "fault_type": fault_type or hyp.get("fault_type") or "dependency_failure"}
    mitigation = {"action": action, "service": rca["service"]} if action not in [None, "none"] else None
    simulator = AbstractBehavioralSimulator(graph)
    sim, sim_sla = simulator.simulate(base, rca_fault=rca)
    mitigated, mitigated_sla = simulator.simulate(base, rca_fault=rca, mitigation_action=mitigation)
    out = processed_dir / "simulator_runtime_outputs"
    ensure_dir(out)
    write_json(sim, out / "simulated_state.json")
    write_json({"rca_hypothesis": rca, "simulated_sla": sim_sla}, out / "rca_validation.json")
    write_json({"rca_hypothesis": rca, "mitigation": mitigation, "sla_after_mitigation": mitigated_sla}, out / "mitigation_result.json")
    print(f"[OK] wrote simulator runtime outputs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--service")
    ap.add_argument("--fault_type")
    ap.add_argument("--action", default="rollback_config")
    args = ap.parse_args()
    run_check(args.processed_dir, args.service, args.fault_type, args.action)

if __name__ == "__main__":
    main()
