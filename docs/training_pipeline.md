# IBM AIOpsLab Joint Training Pipeline

The canonical training unit is one complete incident-resolution trajectory. RCA and Action are not trained as independent stages and then frozen. They execute inside the same trajectory and receive credit from the same downstream outcome.

## Data views

Every processed scenario has two views:

- `state_abstraction.json`: private evaluator state. It may contain `fault_context`, `fault_instances`, and ground truth.
- `state_abstraction_compressed.json`: redacted telemetry state.

Before a trainable policy sees a state, `training_pipeline/agent_input_safety.py` applies a second schema-agnostic sanitizer that removes oracle/candidate-root-cause fields and scenario IDs.

## Canonical trajectory

```text
redacted incident state
  -> RCA instruction policy
  -> fixed/safe RCA solver
  -> predicted RCA
  -> counterfactual digital-twin feedback
  -> Action prompt policy
  -> fixed/LLM ActionAgent
  -> command safety + normalization
  -> twin action execution/simulation
  -> SLA/recovery verification
  -> one end-to-end trajectory return
```

A group contains multiple complete trajectories for the same incident. The group-normalized trajectory advantage is attached to every trainable policy decision in that trajectory. This is the joint-credit signal used by the future optimizer.

The canonical orchestration modules are:

- `training_pipeline/end_to_end_loop.py`: complete trajectory-group rollout.
- `training_pipeline/end_to_end_reward.py`: recovery-centric joint reward.
- `training_pipeline/train_end_to_end_grpo.py`: canonical joint rollout entry point.

`train_rca_grpo.py` and `train_action_grpo.py` remain useful component/debug runners but are not the final training architecture.

## RCA path

`training_pipeline/rca_loop.py` operates on an agent-safe state and public retry history. Hidden exact-label correctness can shape evaluator-side scalar reward, but exact match, pair score, hidden root-count mismatch, and oracle names are not exposed to later policy decisions.

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

## Joint reward and success

The complete trajectory reward is recovery-centric. Dense local RCA/twin/action signals can provide shaping, but final trajectory success is based on a safe remediation resolving the verifier environment and restoring the target/global SLA. It does not require the private RCA exact-label flag.

During current CPU plumbing work the reward mode is explicitly labeled `offline_diagnostic_joint_v1`; it must not be reported as a live-twin scientific result.

## Trainable versus fixed components

Trainable in the final joint run:

- RCA prompt/controller policy
- Action prompt/controller policy

These may use separate LoRA adapters or a shared policy with role conditioning, but their updates occur in the same optimizer iteration using the complete-trajectory advantage.

Fixed environment/agent components:

- fixed RCA solver/foundation agent
- fixed ActionAgent/foundation agent
- command safety gate
- digital twin
- telemetry compressor
- SLA verifier

## Remaining before real GPU training

The joint rollout/credit path is now present, but real parameter updates are not yet enabled. Before the final training run we still need:

- real Qwen/LoRA sampling for the trainable prompt policies,
- old token log-probabilities for both trainable roles,
- a joint optimizer that consumes `joint_advantage`,
- live Kubernetes counterfactual twin execution and recollection for the final strict reward.

The next phase is component and end-to-end smoke testing before enabling those expensive training paths.
