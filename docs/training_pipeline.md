# IBM AIOpsLab Training Pipeline

This branch begins the staged RL pipeline for the IBM/AIOpsLab multiagent remediation system.

## Stage 0: Dataset organization

Every processed scenario has two views:

- `state_abstraction.json`: full oracle/evaluation state. It can contain `fault_context`, `fault_instances`, and ground truth.
- `state_abstraction_compressed.json`: redacted agent-facing state. RCA and action agents must consume this file only.

`training_pipeline/data_loader.py` loads paired files and fails if the compressed state leaks direct fault labels.

## Stage 1: RCA self-prompting loop

Implemented in `training_pipeline/rca_loop.py`.

For each scenario, the RCA instruction policy receives the redacted state and previous non-leaking feedback. It writes a concise RCA instruction. The solver then outputs only canonical lines:

```text
service::fault_type
```

Multifault scenarios output one line per root cause. Rewards in `training_pipeline/rca_reward.py` combine service match, fault-type match, graph-neighborhood match, optional digital-twin symptom reproduction, iteration penalty, and token penalty.

## Stage 2: Digital twin construction and RCA validation

Implemented as an offline behavioral first pass under `digital_twin_runtime/`.

- `twin_spec_builder.py`: builds oracle and RCA-predicted minimal twin specs.
- `telemetry_comparator.py`: extracts compact symptom signatures and computes reproduction/resolution scores.
- `twin_verifier.py`: wraps the existing `state_abstraction_full/behavioral_simulator.py`.

The real scaled Kubernetes twin should later implement the same verifier interface.

## Stage 3: Action prompt optimizer loop

Implemented in `training_pipeline/action_loop.py`.

After RCA passes, the prompt optimizer receives redacted state, RCA result, digital-twin feedback, and previous action attempts. It generates instructions for a fixed ActionAgent. The ActionAgent returns commands only. Commands are checked by `command_safety.py`, normalized by `command_normalizer.py`, and scored by `action_reward.py`.

## Stage 4: Training method

This first patch produces RL-ready rollout logs. The next patch should connect these rollouts to `training/grpo_trainer.py` and replace the debug heuristic policies with trainable Qwen/LoRA policies.

Smoke commands:

```bash
python -m training_pipeline.data_loader \
  --processed_states ~/multiagent_IBM/AIOpsLab/processed_states

python -m training_pipeline.train_rca_grpo \
  --processed_states ~/multiagent_IBM/AIOpsLab/processed_states \
  --output_dir ~/multiagent_IBM/AIOpsLab/rollouts/rca_debug \
  --limit 10 \
  --use_behavioral_twin

python -m training_pipeline.train_action_grpo \
  --processed_states ~/multiagent_IBM/AIOpsLab/processed_states \
  --output_dir ~/multiagent_IBM/AIOpsLab/rollouts/action_debug \
  --limit 10
```
