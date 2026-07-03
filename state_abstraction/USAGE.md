# State Abstraction Bundle

This folder contains the corrected state abstraction pipeline for generated AIOpsLab/SIOPLabs scenarios.

## Main command

```bash
python run_pipeline.py \
  --run_dir /path/to/generated_scenario \
  --output_dir /path/to/generated_scenario/processed_state
```

For batches:

```bash
python run_batch.py \
  --telemetry_dir /path/to/scenario_root \
  --output_base /path/to/processed_states \
  --resume
```

## Important corrected files

- `fault_parser.py`: supports generated single-fault, multi-fault, and variant metadata.
- `discovery.py`: avoids treating `step_*_get_logs.txt` files as fake services.
- `metrics_parser.py`: discovers metric CSVs in nested folders and follows exported-path pointers from `builtin_api_outputs/metrics/*.txt`.
- `traces_parser.py`: discovers trace CSVs in nested folders, follows exported-path pointers from `builtin_api_outputs/traces/*.txt`, and normalizes common trace column aliases.
- `rca_features.py`: uses generated fault hypotheses without over-trusting weak log-only evidence; maps scheduling/pod/container faults to infrastructure failure.
- `build_state.py`: keeps expected faulty services in the service set and emits fault/multifault metadata in the summary.
- `build_simulation_specs.py`: builds simulator specs from all generated fault instances and avoids warning-only false positives.
- `sla.py`: marks global SLA unhealthy when any unhealthy service exists.
- `digital_twin_filter.py`: selects relevant versus bystander services/pods for scaled-down digital twin construction.

## Notes

Metrics and traces may still be absent if the exported paths in the scenario text files point to files that were not copied with the scenario. In that case the pipeline records the observability gap and falls back to Kubernetes state, logs, topology, and generated fault metadata.
