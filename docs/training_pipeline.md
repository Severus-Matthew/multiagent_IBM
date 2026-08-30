# IBM AIOpsLab Joint Training Pipeline

The canonical training unit is one complete incident-resolution trajectory. RCA and Action execute together inside the same incident trajectory, but they do **not** share one optimizer return. Each trainable role has its own policy, LoRA adapter, optimizer buffer, and policy-specific reward/advantage.

## Data views

Every processed scenario has two views:

- `state_abstraction.json`: private evaluator state. It may contain `fault_context`, `fault_instances`, and ground truth.
- `state_abstraction_compressed.json`: redacted telemetry state.

Before a trainable policy sees a state, `training_pipeline/agent_input_safety.py` applies a second schema-agnostic sanitizer that removes oracle/candidate-root-cause fields and scenario IDs.

## Canonical trajectory

```text
redacted incident state
  -> RCA policy (shared frozen base + LoRA_RCA)
  -> fixed/safe RCA solver
  -> predicted RCA
  -> counterfactual digital-twin feedback
  -> Action policy (shared frozen base + LoRA_Action)
  -> fixed/LLM ActionAgent
  -> command safety + normalization
  -> twin action execution/simulation
  -> SLA/recovery verification
```

The full pipeline is always executed together. After the complete trajectory finishes, credit is factorized:

```text
RCA decisions    -> RCA-specific return    -> RCA GRPO buffer
Action decisions -> Action-specific return -> Action GRPO buffer
whole trajectory -> system reward          -> evaluation/model selection
```

This prevents a noisy downstream Action failure from completely erasing a strong RCA decision, while still giving RCA a small amount of downstream recovery credit so the policies remain coupled.

The canonical orchestration modules are:

- `training_pipeline/end_to_end_loop.py`: complete trajectory-group rollout and factorized advantage assignment.
- `training_pipeline/end_to_end_reward.py`: system reward plus separate RCA/Action policy returns.
- `training_pipeline/train_end_to_end_grpo.py`: canonical joint rollout entry point and separate policy-buffer writer.

`train_rca_grpo.py` and `train_action_grpo.py` remain useful component/debug runners but are not the final training architecture.

## Factorized credit assignment

For each complete trajectory, the reward layer produces three quantities:

```text
system_reward
rca_policy_return
action_policy_return
```

The RCA return is dominated by RCA-local service/fault quality and counterfactual-twin reproduction, with a smaller downstream recovery contribution. The Action return is dominated by action-local safety/repair/recovery quality, with a moderate system-level contribution.

For a GRPO trajectory group, normalization is performed separately:

```text
A_RCA    = normalize(rca_policy_return across trajectories)
A_Action = normalize(action_policy_return across trajectories)
A_System = normalize(system_reward across trajectories)   # diagnostic only
```

Optimizer-facing samples use `policy_advantage`. `system_advantage` is logged for analysis and must not be blindly substituted as the policy gradient signal.

## Update schedule

Execution is sequential and causal:

```text
RCA -> twin -> Action -> recovery
```

Learning uses synchronized separate-policy updates rather than fully asynchronous policy publication:

```text
collect N complete incident groups with policy versions k
        -> RCA buffer + Action buffer
        -> update LoRA_RCA with RCA policy_advantage
        -> update LoRA_Action with Action policy_advantage
        -> publish both adapter versions k+1 together
        -> collect next batch
```

This limits policy non-stationarity: Action does not train against an RCA policy that changes in the middle of the same rollout batch.

The rollout driver writes:

```text
joint_trajectories.jsonl
rca_policy_samples.jsonl
action_policy_samples.jsonl
all_policy_samples_diagnostic.jsonl
policy_update_batches.jsonl
```

`policy_update_batches.jsonl` records the synchronization contract for the future real GRPO learners.

## RCA path

`training_pipeline/rca_loop.py` operates on an agent-safe state and public retry history. Hidden exact-label correctness can shape evaluator-side reward, but exact match, pair score, hidden root-count mismatch, and oracle names are not exposed to later policy decisions.

The safe solver contract is:

```text
component::fault_mechanism
```

one line per predicted root cause.

## Digital twin

The frozen offline verifier is an independent counterfactual replay proxy. It does not use hidden labels to compute its RCA reproduction score. It is retained for cheap diagnostic/shaping feedback only because calibration showed that a static compressed incident snapshot cannot reliably distinguish all mechanisms.

The final strict verifier must use live counterfactual execution:

```text
predicted RCA
  -> clean/replayable Kubernetes twin
  -> inject predicted fault
  -> run workload
  -> recollect metrics/logs/traces/system telemetry
  -> run the same state compressor
  -> compare reproduced incident against observed incident
```

The verifier/twin is an environment component and is never optimized by GRPO.

## Action path

`training_pipeline/action_loop.py` receives only:

- sanitized telemetry,
- the predicted RCA service/mechanism,
- prediction-derived twin feedback,
- current observable SLA state,
- public previous action outcomes.

It does not receive upstream exact-label success or oracle RCA metadata.

Commands are passed through:

1. `command_safety.py`
2. `command_normalizer.py`
3. twin execution/simulation
4. `sla_verifier.py`
5. `action_reward.py`

## Policy architecture

The intended final model layout is:

```text
                 shared frozen Qwen base
                    /             \
                   /               \
              LoRA_RCA          LoRA_Action
                 |                  |
             RCA policy         Action policy
                 |                  |
          separate optimizer  separate optimizer
```

The base model can be shared in memory, but the adapters and optimizer states are independent.

## Current status before GPU training

The factorized joint rollout/credit path is implemented, but real parameter updates are not yet enabled. Before the final training run we still need:

- real Qwen sampling for both trainable policy roles,
- separate `LoRA_RCA` and `LoRA_Action` adapters,
- old token log-probabilities for both buffers,
- two GRPO optimizer steps using `policy_advantage`, synchronized at rollout-batch boundaries,
- live Kubernetes counterfactual twin execution/recollection for the final strict reward.

The next phase is component and full CPU end-to-end smoke testing of this exact architecture before enabling GPU training.
