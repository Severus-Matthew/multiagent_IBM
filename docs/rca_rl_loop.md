# RCA RL Loop Design

This document describes the first RL loop implemented in `training-pipeline-v0`.

## Scope

This patch only covers the RCA loop. It does not add GPT-5.1, Qwen/LoRA loading, the action-agent RL loop, or real Kubernetes digital-twin execution.

## Training target

The trainable policy is the RCA instruction policy:

```text
redacted state + non-leaking attempt history -> instruction prompt
```

The RCA solver is fixed:

```text
redacted state + instruction prompt -> service::fault_type lines
```

Later, the trainable policy can be Qwen2.5-Coder-7B-Instruct with LoRA and the fixed solver can be GPT-5.1.

## Solver output schema

The solver must output one root cause per line:

```text
service_name::fault_type
```

For multifault cases:

```text
service_a::fault_type_a
service_b::fault_type_b
```

Canonical fault types:

```text
infra_failure
auth_failure
dependency_failure
resource_exhaustion
latency_degradation
network_failure
config_error
unknown
```

## Episode structure

One scenario is one episode. Each episode has up to five RCA iterations.

At each iteration:

1. Build a policy prompt from redacted state and previous non-leaking feedback.
2. Sample a group of candidate RCA instructions.
3. Run the fixed RCA solver once per candidate instruction.
4. Parse solver output into `FaultLabel` objects.
5. Score each candidate against oracle labels and optional behavioral-twin symptom reproduction.
6. Compute group-normalized advantages for GRPO.
7. Select one candidate to advance episode history.
8. Stop if the selected candidate succeeds, otherwise continue.

The default selection strategy is `best`, which is useful for offline verifier-guided data generation. `sample0` is available for stricter on-policy debugging.

## Reward

The RCA reward is:

```text
R_RCA =
  format_reward
  + pair_match_reward
  + exact_set_bonus
  + twin_reproduction_reward
  - count_mismatch_penalty
  - repeated_wrong_guess_penalty
  - iteration_penalty
  - token_penalty
```

The pair score uses:

```text
0.55 * service_exact
+ 0.30 * fault_type_exact
+ 0.15 * neighborhood_match
```

Terminal failure after the iteration budget receives a separate terminal penalty.

## Leakage control

The policy prompt and history must never include ground-truth service names, fault labels, `fault_context`, or oracle match details.

The episode log may contain `ground_truth_summary` for offline evaluation, but that field must not be used as model input.

## GRPO samples

`train_rca_grpo.py` writes two files:

```text
rollouts.jsonl
  episode-level logs

grpo_samples.jsonl
  trainable-policy samples
```

Each GRPO sample stores:

```text
scenario_id
group_id
sample_id
policy_prompt
completion              # generated instruction prompt
reward
reward_components
advantage
solver_prediction
parsed_prediction
old_logprob_sum         # None for debug rollouts; required for real GRPO updates later
old_logprobs            # None for debug rollouts; required for token-level GRPO later
```

## Current limitations

The current debug implementation uses heuristic instruction policies and a heuristic RCA solver. Therefore the debug pass/fail rate is not a scientific model-performance result.

Before actual training, replace:

```text
HeuristicRCAInstructionPolicy -> Qwen/LoRA policy
HeuristicRCASolver            -> GPT-5.1 fixed solver
```

and populate `old_logprob_sum` / `old_logprobs` during policy sampling.
