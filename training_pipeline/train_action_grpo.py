from __future__ import annotations

import argparse, json
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .rollout_logger import RolloutLogger
from .action_loop import run_action_prompt_optimizer_loop
from .split_utils import read_scenario_ids
from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

class DebugPromptPolicy:
    def generate(self, context):
        svc = context.get("rca_result", {}).get("root_cause_service") or "<service>"
        fault_type = context.get("rca_result", {}).get("fault_type") or "unknown"
        ns = context.get("namespace") or "default"
        if fault_type == "infra_failure":
            return (
                f"Repair Kubernetes scheduling for deployment/{svc} in namespace {ns}. "
                "Remove invalid nodeName/nodeSelector/affinity constraints if present, then restart the deployment and verify rollout. "
                "Output only kubectl commands."
            )
        return f"Restart deployment/{svc} in namespace {ns}, then verify rollout status. Output only kubectl commands."

class DebugActionAgent:
    def get_commands(self, instruction_prompt, context):
        svc = context.get("rca_result", {}).get("root_cause_service") or "<service>"
        fault_type = context.get("rca_result", {}).get("fault_type") or "unknown"
        ns = context.get("namespace") or "default"
        if fault_type == "infra_failure":
            patch = "'[{\"op\":\"remove\",\"path\":\"/spec/template/spec/nodeName\"}]'"
            return [
                f"kubectl patch deployment/{svc} -n {ns} --type=json -p={patch}",
                f"kubectl rollout restart deployment/{svc} -n {ns}",
                f"kubectl rollout status deployment/{svc} -n {ns} --timeout=120s",
            ]
        return [f"kubectl rollout restart deployment/{svc} -n {ns}",
                f"kubectl rollout status deployment/{svc} -n {ns} --timeout=120s"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 action-loop rollout smoke-test")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Maximum selected labeled scenarios to run after filtering.")
    ap.add_argument("--scenario_ids", default=None, help="Optional file with one allowed scenario_id per line.")
    args = ap.parse_args()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    logger = RolloutLogger(args.output_dir); twin = BehavioralTwinVerifier()
    total = passed = skipped_unlabeled = skipped_filter = 0

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            skipped_filter += 1
            continue
        gt = labels_from_full_state(rec.full_state)
        if not gt:
            skipped_unlabeled += 1
            continue
        if args.limit is not None and total >= args.limit:
            break

        total += 1
        rca_result = {"root_cause_service": gt[0].service, "fault_type": gt[0].fault_type, "affected_services": []}
        result = run_action_prompt_optimizer_loop(rec.full_state, rec.compressed_state, rca_result, gt,
                                                  DebugPromptPolicy(), DebugActionAgent(), twin)
        passed += int(result["success"])
        logger.log({"stage": "action", **result})
        print(f"[ACTION] {total} {rec.scenario_id} success={result['success']}")

    summary = {
        "stage": "action",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_filter": skipped_filter,
        "scenario_ids_file": args.scenario_ids,
    }
    logger.write_summary(summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
