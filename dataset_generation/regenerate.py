from __future__ import annotations

"""Regenerate AIOpsLab captures with target-honouring injectors and verified faults.

The corpus contains captures whose injector silently mutated nothing, so the
recorded telemetry shows a healthy system under a fault label. This runner exists
to replace those captures and, more importantly, to make the failure mode
impossible to reintroduce: a scenario is only accepted when the injection layer
reports that it actually mutated the requested target.

Parallelism is expressed as shards rather than threads. One AIOpsLab namespace can
host exactly one incident at a time, so shards are meant to be run against
separate namespaces or separate clusters, each with its own ``--shard_index``.

Example, three shards on three clusters::

    python -m dataset_generation.regenerate \\
        --scenario_ids configs/regenerate_ids.txt \\
        --output_dir /mnt/aiops-data/regen \\
        --shard_index 0 --shard_count 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _load_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    return {
        line.strip()
        for line in Path(path).expanduser().read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def _hydrate_scenario_telemetry(scenario_dir: Path, aiopslab_root: Path) -> dict[str, bool]:
    """Copy this scenario's exported traces/metrics into its own directory.

    Mirrors ``hydrate_telemetry_artifacts.py`` exactly (same pointer parsing,
    same source resolution, same copy semantics) but scoped to one scenario so
    it can run immediately after that scenario finishes, rather than as a
    separate batch pass over the whole shard.
    """
    import hydrate_telemetry_artifacts as hydrate  # AIOpsLab root is on sys.path

    result = {"traces": False, "metrics": False}
    for kind in ("traces", "metrics"):
        for pointer in sorted(scenario_dir.glob(f"builtin_api_outputs/{kind}/*.txt")):
            try:
                text = pointer.read_text(errors="ignore")
            except OSError:
                continue
            raw = hydrate.parse_pointer(kind, text)
            if not raw:
                continue
            src = hydrate.resolve_source(raw, aiopslab_root, kind)
            if src is None:
                continue
            dst = scenario_dir / "builtin_api_outputs" / kind
            ok = hydrate.copy_trace_file(src, dst) if kind == "traces" else hydrate.copy_metric_contents(src, dst)
            result[kind] = result[kind] or ok
    return result


def _trace_rows(scenario_dir: Path) -> int:
    """Number of span rows across the capture's Jaeger CSV exports (header excluded)."""
    total = 0
    for path in scenario_dir.glob("builtin_api_outputs/traces/traces_*.csv"):
        try:
            lines = [line for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        except OSError:
            continue
        total += max(0, len(lines) - 1)
    return total


def _shard(items: list[Any], index: int, count: int) -> list[Any]:
    if count <= 1:
        return items
    return [row for position, row in enumerate(items) if position % count == index]


def _spec_apps(spec: dict[str, Any]) -> set[str]:
    apps = {str(spec.get("app") or "").strip()}
    for row in spec.get("subproblems", []) or []:
        if isinstance(row, dict):
            apps.add(str(row.get("app") or "").strip())
    return {app for app in apps if app}


def _import_generator(aiopslab_root: Path, generated_output: Path):
    root = str(aiopslab_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    repo = str(aiopslab_root.resolve().parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # The generator resolves its output directories relative to the process
    # working directory, so it must run from the AIOpsLab checkout.
    os.chdir(root)
    import gen_and_telmetry  # noqa: E402
    if not hasattr(gen_and_telmetry, "attach_scenario_identity"):
        raise RuntimeError("Apply the reviewed generator fix first: python scripts/regen/apply_aiopslab_patches.py")

    # The upstream module binds output paths at import time. Redirect every one
    # before calling run_one so regeneration can never overwrite the historical
    # corpus that is being audited.
    generated_output = generated_output.resolve()
    gen_and_telmetry.OUT_DIR = generated_output
    gen_and_telmetry.PASSED_DIR = generated_output / "passed"
    gen_and_telmetry.FAILED_DIR = generated_output / "failed"
    gen_and_telmetry.SPECS_DIR = generated_output / "specs"
    gen_and_telmetry.LOG_FILE = generated_output / "log.jsonl"
    gen_and_telmetry.TELEMETRY_DIR = generated_output / "telemetry_outputs"

    return gen_and_telmetry


async def _run_shard(args: argparse.Namespace) -> dict[str, Any]:
    from dataset_generation.injector_fixes import INJECTION_JOURNAL, apply_injector_fixes

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aiopslab_root = Path(args.aiopslab_root).expanduser().resolve()
    generator = _import_generator(Path(args.aiopslab_root), out_dir / "generated")
    patched = apply_injector_fixes()
    warm_patched: dict[str, Any] = {}
    if args.jaeger_local_port:
        from dataset_generation.warm_cluster import install_jaeger_port_isolation
        warm_patched["jaeger_port_isolation"] = install_jaeger_port_isolation(args.jaeger_local_port)
    if args.reuse_deployment:
        from dataset_generation.warm_cluster import WARM_JOURNAL, install_warm_application_mode
        warm_patched = install_warm_application_mode(recovery_timeout=args.recovery_timeout_seconds)
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    accepted_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)
    journal_path = out_dir / f"shard_{args.shard_index}_journal.jsonl"

    wanted = _load_ids(args.scenario_ids)
    if args.spec_source == "telemetry_dir":
        # Re-record exactly the specs that produced the historical captures. The
        # enumerator re-derives multi-fault pairs from a bounded combination
        # walk, so a re-enumerated corpus need not contain the same problem ids.
        if not wanted:
            raise SystemExit("--spec_source telemetry_dir requires --scenario_ids")
        telemetry_root = Path(args.telemetry_dir).expanduser().resolve()
        specs = []
        missing = []
        for problem_id in sorted(wanted):
            spec_path = telemetry_root / problem_id / "spec.json"
            if not spec_path.is_file():
                missing.append(problem_id)
                continue
            specs.append(json.loads(spec_path.read_text()))
        if missing:
            raise SystemExit(f"{len(missing)} requested ids have no recorded spec.json: {missing[:5]}")
        specs = [s for s in specs if not args.apps or bool(_spec_apps(s) & set(args.apps))]
    else:
        specs = [
            spec for spec in generator.generate_specs()
            if (not wanted or str(spec.get("problem_id")) in wanted)
            and (not args.apps or bool(_spec_apps(spec) & set(args.apps)))
        ]
    specs = _shard(sorted(specs, key=lambda s: str(s.get("problem_id"))),
                   args.shard_index, args.shard_count)
    if args.resume:
        done = {p.stem for p in accepted_dir.glob("*.json")}
        skipped = [s for s in specs if str(s.get("problem_id")) in done]
        specs = [s for s in specs if str(s.get("problem_id")) not in done]
        print(json.dumps({"event": "resume_skip_accepted", "skipped": len(skipped)}), flush=True)
    if args.limit:
        specs = specs[: args.limit]

    summary = {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "patched_injectors": patched, "warm_application_mode": warm_patched,
        "requested": len(specs),
        "accepted": 0, "rejected": 0, "errors": 0,
    }
    print(json.dumps({"event": "shard_start", **summary}, sort_keys=True), flush=True)

    for spec in specs:
        problem_id = str(spec.get("problem_id"))
        journal_start = len(INJECTION_JOURNAL)
        warm_start = len(WARM_JOURNAL) if args.reuse_deployment else 0
        started = time.monotonic()
        row: dict[str, Any] = {"problem_id": problem_id}
        try:
            result = await generator.run_one(spec)
            row["generator_ok"] = bool(result.get("ok"))
            row["generator_reason"] = result.get("reason")
        except Exception as exc:  # noqa: BLE001 - one scenario must not kill a shard
            summary["errors"] += 1
            row.update({"generator_ok": False, "error": f"{type(exc).__name__}: {exc}"})
            result = {"ok": False}

        mutations = INJECTION_JOURNAL[journal_start:]
        applied = [m for m in mutations if m.get("applied")]
        # A capture is only usable if the fault was really applied to the
        # requested target. Without this gate the corpus silently refills with
        # healthy systems carrying fault labels.
        injection_verified = bool(applied) and all(
            m.get("applied") and m.get("manifested") for m in mutations
        )
        evidence = {
            "format": "source_injection_evidence_v1",
            "problem_id": problem_id,
            "verified": injection_verified,
            "faults": [
                {
                    **mutation,
                    "mutated": bool(mutation.get("applied")),
                    "manifested": bool(mutation.get("manifested")),
                }
                for mutation in mutations
            ],
        }
        scenario_dir = Path(str(result.get("scenario_dir") or ""))
        # get_traces/get_metrics print a pointer to a file under AIOpsLab's cwd
        # (trace_output/, metrics_output/); the CSV never lands under the
        # scenario's own builtin_api_outputs on its own. The historical corpus
        # only has traces because hydrate_telemetry_artifacts.py copied them in
        # after the fact; reuse that exact logic per scenario, immediately,
        # before the shared trace_output/ directory accumulates further files.
        hydrated = _hydrate_scenario_telemetry(scenario_dir, aiopslab_root) if scenario_dir.is_dir() else {}
        row["hydrated"] = hydrated
        # A capture whose trace export came back header-only is exactly the
        # defect the historical corpus suffers from; never accept it again.
        trace_rows = _trace_rows(scenario_dir) if scenario_dir.is_dir() else -1
        traces_collected = trace_rows > 0
        row["trace_rows"] = trace_rows
        if scenario_dir.is_dir():
            (scenario_dir / "injection_evidence.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n"
            )
        if args.reuse_deployment:
            row["warm_lifecycle"] = WARM_JOURNAL[warm_start:]
        row.update({
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "injection_mutations": mutations,
            "injection_verified": injection_verified,
            "traces_collected": traces_collected,
        })

        if row.get("generator_ok") and injection_verified and traces_collected:
            summary["accepted"] += 1
            (accepted_dir / f"{problem_id}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
            verdict = "accepted"
        else:
            summary["rejected"] += 1
            (rejected_dir / f"{problem_id}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
            verdict = "rejected"

        with journal_path.open("a") as stream:
            stream.write(json.dumps({**row, "verdict": verdict}, sort_keys=True, default=str) + "\n")
        print(json.dumps({
            "event": "scenario_done", "problem_id": problem_id, "verdict": verdict,
            "injection_verified": injection_verified,
            "elapsed_seconds": row["elapsed_seconds"],
        }, sort_keys=True), flush=True)

    (out_dir / f"shard_{args.shard_index}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "shard_done", **summary}, sort_keys=True), flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario_ids", default=None,
                    help="file of problem_ids to regenerate; empty means every spec")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--aiopslab_root", default="AIOpsLab")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--spec_source", choices=["generate", "telemetry_dir"], default="generate",
                    help="generate: enumerate specs; telemetry_dir: load each id's recorded spec.json")
    ap.add_argument("--telemetry_dir", default=None,
                    help="historical telemetry_outputs root holding <id>/spec.json (for --spec_source telemetry_dir)")
    ap.add_argument("--resume", action="store_true", help="skip ids already accepted in the output dir")
    ap.add_argument("--jaeger_local_port", type=int, default=None,
                    help="local port for this shard's Jaeger port-forward (distinct per parallel shard)")
    ap.add_argument(
        "--reuse_deployment", action="store_true",
        help="Reuse a healthy application namespace between scenarios instead of a full "
             "uninstall/reinstall; every reuse is gated by a fail-closed cleanliness check.",
    )
    ap.add_argument("--recovery_timeout_seconds", type=float, default=300.0,
                    help="How long recovery may take to return the namespace to a clean state before falling back to full teardown.")
    ap.add_argument(
        "--apps", nargs="*", default=None,
        help="Application partition for namespace-safe parallel workers (for example: social hotel).",
    )
    args = ap.parse_args()
    if not 0 <= args.shard_index < max(1, args.shard_count):
        raise SystemExit("shard_index must be within [0, shard_count)")

    summary = asyncio.run(_run_shard(args))
    return 0 if summary["rejected"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
