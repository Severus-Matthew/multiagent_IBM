# Full trajectory walkthrough: one real training example

This is one complete, real trajectory pulled directly from the currently-running
training run (`live-grpo-stage1`), not a synthetic illustration. Every prompt,
completion, command, and Twin measurement below is copied verbatim from the
actual log files:

- `/mnt/aiops-training/runs/live-grpo-stage1/joint_trajectories.jsonl` (row 12,
  trajectory index 2)
- `/mnt/aiops-training/runs/live-grpo-stage1/openai_calls_worker_0.jsonl`
  (lines 156–162)

Raw companion files are saved alongside this document in
`docs/examples/`: `full_trajectory_raw.json` (the complete trajectory record),
`rca_call_0_raw_request.json` / `rca_call_1_raw_request.json` (the two exact
API requests sent to the frozen RCA agent), and `action_call_0_raw_request.json`
/ `action_call_4_raw_request.json` (the first and last exact API requests sent
to the frozen Action agent).

## Scenario

- **Scenario ID:** `gen_assign_to_non_existent_node_social_net-detection-post-storage-service-default`
- **Policy version at rollout:** `qwen-live-grpo-stage1-v1@u000006` (i.e. this
  rollout happened using the adapters as they stood right after synchronized
  update 6, before update 7)
- **Private ground truth** (read directly from
  `state_abstraction.json.fault_context`, never shown to any agent):
  `fault_family: assign_to_non_existent_node_social_net`,
  `faulty_service: post-storage-service`, single-fault, variant `default`.
- **Twin route for this whole trajectory:** `live` — every verification below
  is a real sparse Kubernetes Twin, not the offline proxy.

---

## Part 1 — RCA

Two roles are involved on the RCA side: the **trainable RCA prompt policy**
(the Qwen LoRA adapter `lora_rca`, sampled with temperature/top_p, currently
being trained) and the **frozen RCA reasoner** (GPT-5.2, greedy, never
trained). The trainable policy writes free-text guidance; the frozen reasoner
reads that guidance plus the redacted telemetry and outputs one structured
prediction line.

### What the frozen RCA reasoner is given, every call

The exact system instructions sent with every RCA call (`instructions` field
of the API request):

> "You are a fixed Kubernetes root-cause agent. Use the policy instruction as
> search guidance, but independently inspect all supplied redacted telemetry,
> especially Kubernetes desired/ready replicas, scheduling conditions,
> Services, endpoints, ports, events, logs, metrics, and traces. Return only
> `service::fault_type::injectible_mechanism[::variant]` lines, with no prose.
> For scale_replicas_zero use scale_0, scale_2, or scale_3 when evidence
> supports it; for network_delay use delay_100ms, delay_300ms, or
> delay_1000ms; for network_loss use loss_5pct, loss_20pct, or loss_50pct.
> Omit the variant only for default. The second field is always the canonical
> fault type and the third field is always the mechanism. Example:
> `svc::infra_failure::scale_replicas_zero::scale_0`."

The `input` field (a single JSON string, **69,942 characters** for the second
call — about 4% the size of the raw `state_abstraction_compressed.json` for
this scenario, and less than 1/20th the size of the full private file with
ground truth) contains:

- `allowed_mechanisms`: the 14 mechanisms the live Twin can actually inject —
  never which one is correct here, just the vocabulary.
- `mechanism_semantics`: one sentence per mechanism describing its observable
  signature (e.g. `"assign_to_non_existent_node": "desired replicas remain
  above zero but pods are unavailable because scheduling cannot place them"`).
- `policy_instruction`: whatever the trainable RCA policy wrote this attempt
  (shown per-attempt below).
- `redacted_state`: the bounded, sanitized telemetry projection — `clusters`,
  `graph`, `logs`, `metrics`, `system`, `traces`, `sla`, `service_health`,
  `observability_metadata`, `workload`, `model_table`. No `scenario_id`, no
  `fault_context`, no `ground_truth`, no descriptive fault-family string
  anywhere in this text — checked directly against the actual bytes sent.

### Attempt 0 (iteration 0)

**Trainable RCA policy's instruction** (its actual sampled completion, passed
into the frozen agent's prompt as `policy_instruction`):

> "To identify the root cause among multiple faulty services, focus on
> services with high log error counts and dependency connection failures. The
> root cause is typically the service that initiates the cascade — look for a
> service with high dependency connection errors and a large number of log
> entries, such as `compose-post-service`, which may be failing to connect to
> `post-storage-service`. Follow the trace from the source of the primary
> error to isolate the initial failure point."

**Frozen RCA reasoner's output** (the entire model response, 13 output
tokens):

```
post-storage-service::infra_failure::pod_failure
```

**Live Twin's reaction:** built a sparse Kubernetes replica containing
`post-storage-service` and its causal neighborhood, injected `pod_failure`
against it for real, ran the targeted workload, collected fresh telemetry, and
compared it against the original incident's scoped symptom signature.

| Measurement | Value |
|---|---|
| `predicted_fault_injection_checked` | `true` |
| `twin_reproduction_score` | **0.256** |
| `rca_twin_verified` | **false** (below the 0.4702 threshold) |
| `pair_score` vs. private ground truth | 0.75 (service exact, fault type exact, mechanism **not** exact) |
| `exact_set_match` | false |

**Reward for this attempt:** 1.956 — format (+0.20) + pair match (+1.50 = 2.00×0.75)
+ Twin reproduction (+0.256) − iteration cost (−0.0, first attempt).

**Feedback returned to the RCA policy for the next iteration:**

> "Counterfactual twin reproduction is weak; refine the service/mechanism
> using the observed telemetry."

Note what the feedback does *not* say: it never says which mechanism was
wrong, never names `assign_to_non_existent_node`, never says "not
pod_failure." It reports only the Twin's own reproduction fidelity, which is
public/self-referential information, not oracle leakage.

### Attempt 1 (iteration 1)

**Trainable RCA policy's revised instruction:**

> "**RCA Instruction Prompt**
>
> You are given a system exhibiting cascading failures. Your task is to
> identify the root cause of the outage by analyzing the provided telemetry.
>
> Use only the redacted telemetry to reason through the failure chain:
>
> 1. **Observe Symptoms**: Identify services with high log error counts,
>    dependency connection failures, or timeouts.
> 2. **Trace Dependencies**: Follow the service graph to understand how
>    failures propagate.
> 3. **Distinguish Cause from Effect**:"

(The policy's own completion was cut here by its 64-token generation budget —
worth knowing if you want to raise `--policy_max_new_tokens` for later stages.)

**Frozen RCA reasoner's output:**

```
post-storage-service::infra_failure::assign_to_non_existent_node
```

**Live Twin's reaction:** tore down the previous fault, injected
`assign_to_non_existent_node` against `post-storage-service` for real (this is
exactly the real, hidden fault — the Twin has no way to know that; it is
purely re-running whatever the policy predicted), re-ran the workload,
recompared.

| Measurement | Value |
|---|---|
| `predicted_fault_injection_checked` | `true` |
| `twin_reproduction_score` | **0.6449** |
| `rca_twin_verified` | **true** |
| `pair_score` vs. private ground truth | **1.0** (service, fault type, mechanism, variant all exact) |
| `exact_set_match` | **true** |

**Reward for this attempt:** 3.7449 = format (0.20) + pair match (2.00×1.0) +
exact-set bonus (1.00) + Twin reproduction (0.6449) − iteration penalty (0.10,
this was iteration index 1). I confirmed this arithmetic term-by-term against
`rca_reward.py`'s formula and it reproduces the logged value exactly.

**Feedback:**

> "Counterfactual twin reproduction is comparatively strong; keep the causal
> explanation focused and avoid adding unsupported roots."

**RCA side concludes here.** `final_prediction:
post-storage-service::infra_failure::assign_to_non_existent_node`,
`success: true`. This prediction is what gets handed to the Action side next
— the trajectory now proceeds to remediation using this exact (correct, in
this case) hypothesis.

---

## Part 2 — Action

Same two-role structure: the **trainable Action prompt policy** (`lora_action`)
writes remediation guidance; the **frozen Action agent** (GPT-5.2) turns that
into real `kubectl` commands executed against the live Twin.

### What the frozen Action agent is given, every call

System instructions (fixed, every call):

> "You are a fixed Kubernetes remediation agent. Return only safe
> namespace-scoped kubectl, helm, or mongosh commands, one per line; no
> markdown or prose. Never use exec, apply, replace, pipelines, broad
> deletes, or cluster-wide flags."

The `input` (**70,845 characters** for the first call) is structured as:

- `action_requirements`: use scoped commands only, prefer minimal remediation
  matching the predicted fault type, include a verification command, no
  exec/apply/replace/pipelines/broad-deletes/cluster-wide flags.
- `current_sla`: the baseline SLA snapshot measured once at the start of the
  Action stage — **27 of the sparse Twin's services showing degraded
  service_health, 16 with log errors, 43 hard violations.** This stays fixed
  across all 5 iterations in this trajectory (it's the "before" baseline the
  agent is working against, not a live re-measurement — the live
  re-measurement per attempt shows up in `previous_attempts[i]`, see below).
- `rca_faults` / `rca_result` / `rca_twin_gate`: the upstream RCA prediction
  and its Twin verification score (0.6449) — the Action side gets the RCA
  policy's *conclusion*, not the RCA policy's raw prompt/instruction.
- `namespace`: the opaque Twin namespace, `aiops-twin-7d6ad5ef93d3` here.
  Never the real `test-social-network`.
- `previous_attempts`: empty on iteration 0, then accumulates the full
  commands + feedback + public reward summary of every prior attempt in this
  same trajectory (verified: by iteration 4, `previous_attempts` held all 4
  earlier attempts in full).
- `redacted_state`: same bounded telemetry family as the RCA side. For the
  actual target, the `system.post-storage-service` entry read:

```json
{
  "deployment_signals": {
    "conditions.Available.reason": "MinimumReplicasUnavailable",
    "conditions.Available.status": "False",
    "replicas_available": 0,
    "replicas_ready": 0,
    "replicas_unavailable": 1
  },
  "endpoint_signals": {
    "has_ready_endpoints": false,
    "ready_endpoint_count": 0
  },
  "health": {
    "pods_ready": 0,
    "pods_total": 1,
    "pods_unready": 1,
    "status": "no_ready_endpoints",
    "warning_event_count": 5
  }
}
```

This is the actual, real symptom of `assign_to_non_existent_node` — a pod that
can't be scheduled — visible in plain structural fields, with no mention of
"node," "scheduling," or the fault name anywhere in this record.

### Attempt 0 (iteration 0)

**Trainable Action policy's instruction:**

```
kubectl get deploy -n test-social-network post-storage-service
kubectl get pods -n test-social-network -l app=post-storage-service
kubectl describe deploy -n test-social-network post-storage-service
kubectl get svc -n test-social-network post-storage-service
kubectl get endpoints -n test-social-network post-storage-service
kubectl logs -n test-social-network -l app=post-storage-service --tail=50
kubectl get pods -n test-social-network -o wide | grep post
```

Note it names the **real** namespace (`test-social-network`) — this is the
trainable policy's raw output; it doesn't automatically know the opaque Twin
namespace, that substitution is the frozen agent's job using the `namespace`
field it's given.

**Frozen Action agent's output (the actual commands executed):**

```
kubectl get deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service
kubectl get pods -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
kubectl describe pod -n aiops-twin-7d6ad5ef93d3 post-storage-service-755d4d45f-7hqbq
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=merge -p '{"spec":{"template":{"spec":{"nodeSelector":null,"affinity":null}}}}'
kubectl rollout status -n aiops-twin-7d6ad5ef93d3 deploy/post-storage-service
kubectl get endpoints -n aiops-twin-7d6ad5ef93d3 post-storage-service
```

Correctly translated to the opaque namespace, and the actual repair attempt
(the `kubectl patch`) correctly targets the real injected field
(`nodeSelector`).

**Live Twin's reaction — rejected before execution:**

```json
{
  "commands": [],
  "executed": false,
  "namespace": "aiops-twin-7d6ad5ef93d3",
  "rejection_reasons": ["mutation_target_not_selected_exact_resource"],
  "safe": false
}
```

`resolved: false`, `reason: no_safe_valid_mitigation_action`. Reward: −1.74.

**Feedback:**

> "One or more commands were unsafe or unsupported. Use scoped kubectl/helm/mongosh
> commands only."

### Attempt 1 (iteration 1)

**Trainable policy's instruction** (still names the real namespace):

```
kubectl get deploy -n test-social-network post-storage-service
kubectl get pods -n test-social-network -l app=post-storage-service -o wide
kubectl describe pod -n test-social-network post-storage-service-755d4d45f-7hqbq
kubectl patch deploy -n test-social-network post-storage-service --type=merge -p '{"spec":{"template":{"spec":{"nodeSelector":null,"affinity":null}}}}'
kubectl rollout status
```

**Frozen Action agent's output:**

```
kubectl get pod -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
kubectl get deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service -o jsonpath='{.spec.template.spec.nodeSelector}{"\n"}{.spec.template.spec.affinity}{"\n"}'
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"},{"op":"remove","path":"/spec/template/spec/affinity"}]'
kubectl rollout status -n aiops-twin-7d6ad5ef93d3 deploy/post-storage-service
kubectl get endpoints -n aiops-twin-7d6ad5ef93d3 post-storage
```

This switched from a `--type=merge` null-out to a `--type=json remove` patch —
a more surgical fix for exactly the injected field.

**Live Twin's reaction:**

```json
{
  "action_repairs_fault_type": false,
  "reason": "live_command_execution_failed",
  "resolved": false,
  "reward_route": "live",
  "sla_restored": false,
  "target_sla_restored": false,
  "twin_resolved": false
}
```

Reward: −0.30. **Feedback:** "Commands were valid but did not improve the
behavioral twin/SLA symptoms."

### Attempt 2 (iteration 2)

**Trainable policy's instruction** (now correctly using the Twin namespace,
learned from the accumulated feedback):

```
kubectl get deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service -o jsonpath='{.spec.template.spec.nodeSelector}{"\n"}{.spec.template.spec.affinity}{"\n"}'
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"},{"
```

(cut off by the 128-token completion budget)

**Frozen Action agent's output:**

```
kubectl get pods -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
kubectl get events -n aiops-twin-7d6ad5ef93d3 --sort-by=.lastTimestamp
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
kubectl rollout status -n aiops-twin-7d6ad5ef93d3 deploy/post-storage-service
kubectl get endpoints -n aiops-twin-7d6ad5ef93d3 post-storage-service
```

Here the frozen agent drifted to patching `nodeName` instead of
`nodeSelector` — `nodeName` is a scheduler-assigned status field, not the
field the fault actually set, so this cannot fix anything.

**Live Twin's reaction:** same shape, `resolved: false`,
`reason: live_command_execution_failed`, reward −0.40. Feedback unchanged in
substance.

### Attempt 3 (iteration 3)

**Frozen Action agent's output:**

```
kubectl get deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service -o jsonpath='{.spec.template.spec.nodeSelector}{"\n"}{.spec.template.spec.affinity}{"\n"}{.spec.template.spec.nodeName}{"\n"}'
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
kubectl rollout status -n aiops-twin-7d6ad5ef93d3 deploy/post-storage-service
kubectl get pods -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
kubectl get endpoints -n aiops-twin-7d6ad5ef93d3 post-storage-service -
```

Still targeting `nodeName`. **Live Twin's reaction:** unresolved again, reward
−0.50.

### Attempt 4 (iteration 4, final)

**Frozen Action agent's output:**

```
kubectl get pod -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
kubectl describe pod -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service
kubectl patch deploy -n aiops-twin-7d6ad5ef93d3 post-storage-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
kubectl rollout status -n aiops-twin-7d6ad5ef93d3 deploy/post-storage-service
kubectl get endpoints -n aiops-twin-7d6ad5ef93d3 post-storage-service
kubectl get pods -n aiops-twin-7d6ad5ef93d3 -l app=post-storage-service -o wide
```

**Live Twin's reaction:** unresolved, reward −2.64 (the largest single-attempt
penalty of the five — see below).

**Terminal outcome:**

```json
{
  "components": {"num_iterations": 5, "terminal_failure": true},
  "feedback": "Action loop failed after the iteration budget. Try a different remediation strategy.",
  "reward": -2.0,
  "success": false
}
```

The loop exhausted its 5-attempt budget without a single accepted mutation, so
it terminates with an explicit failure penalty on top of the per-attempt
rewards.

---

## What this one trajectory contributes to training

- **RCA policy return:** positive (the correct, Twin-verified diagnosis on the
  second try) — this trajectory pushes the RCA adapter toward the behavior
  that produced attempt 1's instruction style.
- **Action policy return:** negative across the board — none of the five
  attempts executed a real mutation. Four were flagged
  `mutation_target_not_selected_exact_resource` / `live_command_execution_failed`,
  and the fifth exhausted the budget. Because `end_to_end_reward.py` requires
  a safe *and mutating* command before granting any positive Action credit,
  this trajectory correctly contributes negative signal to the Action adapter
  — exactly the intended behavior for a remediation that never actually
  fixed anything, not a scoring artifact.
- **System-level:** `system_quality = 0` for this trajectory (no safe
  mutation occurred), which also caps how much system-level credit flows back
  into either policy's return.

This is genuinely representative of very early training (bundle update 6):
the RCA side is already showing real signal, and the Action side is still
finding its footing on the specific mechanics of *which* command gets past
the safety layer for a scheduling fault. That's exactly the kind of gap
worth watching for as training progresses — if `mutation_target_not_selected_exact_resource`
keeps recurring past many more updates, it's worth checking whether the
safety layer's exact-resource match is too strict for this fault family
specifically, since the command shape here (`patch deploy ... nodeSelector`)
is a textbook-correct fix.
