# Full-scale AIOps State Abstraction + SLA + Digital Twin

Run:

```bash
python run_pipeline.py --run_dir /path/to/scenario_folder
```

Outputs:

```text
processed_state/state_abstraction.json
processed_state/states.jsonl
processed_state/state_summary.json
processed_state/graph.json
processed_state/sla_results.json
processed_state/twin_inputs/*.json
processed_state/twin_runtime_outputs/*.json
```

Expected scenario folder can contain:

```text
spec.json
run_result.json
agent_transcript.json
direct_k8s_outputs/pods.json
direct_k8s_outputs/services.json
direct_k8s_outputs/pod_logs/*.log
metrics/metric_*/container/*.csv
traces/*.json or traces/*.csv
topology/topology.json or topology/graph.json
```
