import argparse
from pathlib import Path
from build_state import build_and_write
from build_simulation_specs import build_specs
from run_rca_simulation_check import run_check
from sla import evaluate_states_file
from compress import compress_state
from utils import write_json, read_json


def main():
    ap = argparse.ArgumentParser(
        description="Full AIOps state abstraction + SLA + simulator + compression pipeline"
    )
    ap.add_argument("--run_dir", required=True, help="Scenario output folder from gen_and_telmetry.py")
    ap.add_argument("--output_dir", default=None, help="Defaults to <run_dir>/processed_state")
    ap.add_argument("--skip_simulator", action="store_true", help="Skip behavioral simulator step")
    ap.add_argument("--skip_compress", action="store_true", help="Skip state compression step")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "processed_state"

    # ── Step 1: Build full state ──────────────────────────────────────────────
    print(f"[1/4] Building state from {run_dir} ...")
    state = build_and_write(run_dir, output_dir)

    # ── Step 2: SLA evaluation ────────────────────────────────────────────────
    print("[2/4] Evaluating SLA ...")
    evaluate_states_file(output_dir / "states.jsonl", output_dir / "sla_results.json")
    print(f"[OK]  SLA results → {output_dir / 'sla_results.json'}")

    # ── Step 3: Build simulator specs ─────────────────────────────────────────
    print("[3/4] Building simulator specs ...")
    build_specs(output_dir)

    # ── Step 3b: Run behavioral simulator ─────────────────────────────────────
    if not args.skip_simulator:
        rca = state.get("rca_features", {})
        hyp = rca.get("hypothesis", {})
        fault_ctx = state.get("fault_context", {})

        # Guard: fall back to fault_context if hypothesis is empty
        service    = hyp.get("service")    or fault_ctx.get("faulty_service") or "unknown"
        fault_type = hyp.get("fault_type") or "dependency_failure"

        print(f"[3b]  Running simulator: service={service}, fault_type={fault_type} ...")
        run_check(output_dir, service=service, fault_type=fault_type, action="rollback_config")

    # ── Step 4: Compress state for LLM / downstream agents ───────────────────
    if not args.skip_compress:
        print("[4/4] Compressing state ...")
        compressed = compress_state(state)
        out_path = output_dir / "state_abstraction_compressed.json"
        write_json(compressed, out_path)
        print(f"[OK]  Compressed state → {out_path}")

    print("[DONE] Full pipeline complete.")
    print(f"       Output dir: {output_dir}")


if __name__ == "__main__":
    main()
