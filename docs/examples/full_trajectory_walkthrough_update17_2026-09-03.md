# Full trajectory walkthrough: bundle update 17 — first real resolution

A fourth real, complete trajectory, same sourcing standard as the first
three. This one is the first documented in this series where the Action
side actually **resolves the incident** — the live Twin measurably recovers,
not just executes safely. Every prompt, completion, command, and Twin
measurement below is copied verbatim from the actual log files:

- `/mnt/aiops-training/runs/live-grpo-stage1/joint_trajectories.jsonl` (line
  36, trajectory index 0)
- `/mnt/aiops-training/runs/live-grpo-stage1/rca_policy_samples.jsonl`
  (matching entry, iteration 0)
- `/mnt/aiops-training/runs/live-grpo-stage1/action_policy_samples.jsonl`
  (matching entries, iterations 0–3)
- `/mnt/aiops-training/runs/live-grpo-stage1/openai_calls_worker_0.jsonl`
  (line 537 for RCA, lines 538–541 for Action)

Raw companion files in `docs/examples/`: `full_trajectory_raw_u17.json`,
`rca_call_raw_request_u17.json`, `action_call_0_raw_request_u17.json`,
`action_call_3_raw_request_u17.json`.

## Scenario

- **Scenario ID:** `gen_assign_to_non_existent_node_social_net-mitigation-text-service-default`
- **Policy version at rollout:** `qwen-live-grpo-stage1-v1@u000016` — the
  first batch computed after the command-shape parser consolidation fix.
- **Private ground truth:** `fault_family:
  assign_to_non_existent_node_social_net`, `fault_mechanism:
  assign_to_non_existent_node`, `fault_type: infra_failure`, `service:
  text-service`, single-fault, variant `default`.
- **Twin route:** `live` throughout.
- **Downstream models:** GPT-5.2, `max_output_tokens: 96` (RCA) / `320`
  (Action) — confirmed from the raw request files.
- **Trainable policy budget:** `--max_new_tokens 224` at this point (the 800
  bump came one update later, in response to what this same trajectory
  helped illustrate — most Action instructions below still get cut off).

---

## Part 1 — RCA

Only one iteration needed this time — the trainable policy's very first
instruction led the frozen reasoner straight to the correct answer.

**Trainable RCA policy's instruction** (205 tokens):

> "To identify the root cause of the observed system degradation while
> distinguishing it from downstream cascades:
>
> 1. **Analyze error signals**:
>    - Focus on services with high `log_errors` counts.
>    - Prioritize logs showing `dependency_connection` or
>      `dependency_timeout` errors.
> 2. **Trace dependency failures**:
>    - Identify which services are reporting connection or timeout errors
>      toward other services.
>    - Look for patterns where multiple services report failures to the
>      same target.
> 3. **Distinguish root cause from cascade**:
>    - A root cause will typically be a service that reports direct
>      dependency errors to another component.
>    - Downstream cascades are usually services that fail due to their
>      dependencies failing — they show indirect symptoms rather than
>      originating causes.
> 4. **Verify health status and stability**:
>    - Services with no log errors but stable metrics may indicate a
>      downstream effect.
>    - Services with continuous `dependency_timeout` or
>      `dependency_connection` errors likely contribute directly to the
>      issue.
>
> Based on these principles, determine the single source of failure and its
> type.
>
> Example output format:
> ```
> compose-post-service::dependency_connection::connection_refused
> ```"

**Frozen RCA reasoner's output:**
`text-service::infra_failure::assign_to_non_existent_node` — correct on the
first try.

**Live Twin's reaction:** injected the real fault, re-measured.
`twin_reproduction_score: 0.6802`, `rca_twin_verified: true`, `pair_score:
1.0`, `exact_set_match: true`. **Reward:** 3.7952 = 0.20 (format) + 2.00
(pair match) + 1.00 (exact-set bonus) + 0.6802 (twin) − 0.085 (token-length
penalty — its instruction was fairly long for a one-shot success).
`final_prediction: text-service::infra_failure::assign_to_non_existent_node`,
`success: true`.

---

## Part 2 — Action: the real fix landed on attempt 1, but wasn't credited until attempt 3

This is the most instructive part of this trajectory: the correct mutation
was actually applied on **attempt 1**, two iterations before the reward
pipeline credited it. Here's exactly why, with the raw evidence.

### Attempt 0 (iteration 0)

**Trainable policy's instruction** (578 tokens — the 3-part structure fully
present, unlike update 16, though this still exceeds the 224-token ceiling
in effect at the time and got recorded in full here because the completion
finished before hitting it):

> "### Part 1: Restate the Predicted Fault
>
> The predicted fault is:
> - **Service:** `text-service`
> - **Fault Type:** `infra_failure`
> - **Fault Mechanism:** `assign_to_non_existent_node`
>
> This prediction is supported by the specific entry in the `redacted_state`
> under the `system` field, specifically within the `text-service`'s
> inventory: `"nodes_count": 0` — indicating that the `text-service`
> deployment has no associated node assigned to its pod(s).
>
> ### Part 2: Technical Reasoning
>
> Check the current deployment configuration using `kubectl get deployment
> text-service -o yaml`. Look at the `.spec.template.spec.nodeName` field —
> if set to a non-existent node, it needs correction. If `nodeName` is not
> specified but `nodeSelector`, `tolerations`, or `affinity` might be
> causing issues, those fields should also be reviewed.
>
> ### Part 3: Explicit Handoff Instructions for ActionAgent
>
> Please proceed with issuing commands to check the deployment spec, edit it
> to correct any invalid node assignment (`nodeName`, `nodeSelector`,
> `affinity`, `tolerations`), and verify the pod starts running."

**Frozen Action agent's output** (7 commands):

```
kubectl -n aiops-twin-7d3d5f0a608f get deploy text-service -o yaml
kubectl -n aiops-twin-7d3d5f0a608f get pod text-service-786cc8868f-n4xdq -o wide
kubectl -n aiops-twin-7d3d5f0a608f describe pod text-service-786cc8868f-n4xdq
kubectl -n aiops-twin-7d3d5f0a608f patch deploy text-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
kubectl -n aiops-twin-7d3d5f0a608f rollout status deploy/text-service
kubectl -n aiops-twin-7d3d5f0a608f get pods -l app=text-service -o wide
kubectl -n aiops-twin-7d3d5f0a608f get endpoints text-service -o yaml
```

The `get deploy ... -o yaml` command's real output revealed the *actual*
problem, verbatim from the Twin:

```yaml
nodeSelector:
  kubernetes.io/hostname: aiops-twin-non-existent-node
```

There was no `nodeName` field at all — attempt 0's mutation targeted the
wrong field (`nodeName`, which didn't exist) based on a reasonable but
unconfirmed guess. Command 2 (`get pod ... -786cc8868f-n4xdq`) also failed
with `NotFound` — that pod name was stale, from before the twin relabeled a
new ReplicaSet. **Reward:** −0.738. **Feedback:** *"Commands were valid but
did not improve the behavioral twin/SLA symptoms."*

### Attempt 1 (iteration 1) — the actual fix

**Trainable policy's instruction** (506 tokens): correctly reasons that
"previous iterations didn't resolve the problem, suggesting other fields
might still contain problematic scheduling constraints" and tells the
ActionAgent to inspect and remove `nodeSelector` specifically this time.

**Frozen Action agent's output** (7 commands):

```
kubectl -n aiops-twin-7d3d5f0a608f get pods -l app=text-service -o wide
kubectl -n aiops-twin-7d3d5f0a608f describe pod -l app=text-service
kubectl -n aiops-twin-7d3d5f0a608f get deploy text-service -o yaml
kubectl -n aiops-twin-7d3d5f0a608f patch deploy text-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"}]'
kubectl -n aiops-twin-7d3d5f0a608f patch deploy text-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/affinity"}]'
kubectl -n aiops-twin-7d3d5f0a608f rollout status deploy/text-service
kubectl -n aiops-twin-7d3d5f0a608f get endpoints text-service -o yaml
```

**What actually happened, command by command:**
1. `get pods` — pod `Pending`, `NODE: <none>`.
2. `describe pod` — confirms `Node-Selectors: kubernetes.io/hostname=aiops-twin-non-existent-node`, event: `FailedScheduling ... didn't match Pod's node affinity/selector`.
3. `get deploy -o yaml` — full spec, confirming the `nodeSelector`.
4. **`patch ... remove nodeSelector` → `returncode: 0`, `"deployment.apps/text-service patched"` — the actual fix, applied successfully, right here.**
5. `patch ... remove affinity` → **`returncode: 1`, `"The request is invalid: the server rejected our request due to an error in our request"`** — `affinity` was never set, so removing a nonexistent JSON-patch path is rejected by the K8s API.
6–7. Never ran — `execute_twin_commands()` stops at the first nonzero-exit command in a batch, so `rollout status` and `get endpoints` were never reached, and no post-fix workload measurement happened this iteration.

**Reward:** −0.766, `verifier_reason: live_command_execution_failed`,
`resolved: false` — **despite the real fix already being live in the
cluster.** Feedback: the same generic *"Commands were valid but did not
improve the behavioral twin/SLA symptoms"* — which undersells what actually
happened, since the improvement genuinely occurred but was never observed
because the verification commands never ran.

### Attempt 2 (iteration 2)

By this point the pod is **already `Running`** — visible directly in this
attempt's own `get pods` output, which the policy issued as a diagnostic
step: `text-service-79df5d74b4-996xm 1/1 Running`. The policy doesn't
recognize this as success (its instruction still frames the fault as
unresolved) and tries removing `topologySpreadConstraints` — a field that
also was never set, so the same rejection happens again
(`returncode: 1`, same K8s error), again halting the batch before the
verification commands run. **Reward:** −0.825, same `verifier_reason` and
`resolved: false`, for a fault that has been fixed in the cluster for one
full iteration already.

### Attempt 3 (iteration 3, final) — credited

**Trainable policy's instruction** (527 tokens) reasons through the same
scheduling-field checklist once more, and the ActionAgent's chosen mutation
this time — remove `schedulerName` — happens to be a **genuine no-op**
(`schedulerName` was already `"default-scheduler"`, its default value):

```
kubectl -n aiops-twin-7d3d5f0a608f get pod -l app=text-service -o wide
kubectl -n aiops-twin-7d3d5f0a608f describe pod -l app=text-service
kubectl -n aiops-twin-7d3d5f0a608f get deploy text-service -o jsonpath='{.spec.template.spec.schedulerName}{"\n"}{.spec.template.spec.tolerations}{"\n"}{.spec.template.spec.affinity}{"\n"}{.spec.template.spec.nodeSelector}{"\n"}{.spec.template.spec.nodeName}{"\n"}'
kubectl -n aiops-twin-7d3d5f0a608f patch deploy text-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/schedulerName"}]'
kubectl -n aiops-twin-7d3d5f0a608f rollout status deploy/text-service
kubectl -n aiops-twin-7d3d5f0a608f get endpoints text-service -o yaml
```

Critically, **every command this time returns 0** — the `schedulerName`
patch is accepted (`"deployment.apps/text-service patched (no change)"`
— K8s accepts removing a field that's absent, unlike removing a
never-set field like `affinity`/`topologySpreadConstraints`, since
`schedulerName` had a real, present value to remove). So the batch runs all
the way through, and the reward pipeline's real workload probe finally
executes:

| | Before | After |
|---|---|---|
| `required_ready_endpoints` (text-service) | 0 | 1 |
| Pod status | — | `Running`, `Ready: True`, `Node: kind-control-plane` |
| Endpoint | none | `10.244.0.155:9090`, real pod backing it |

```json
"recovery": {"ready": true, "stable_seconds": 15.059, "condition": "all_selected_controllers_available", ...all 27 deployments in the twin ready...},
"resolution": {"target_service": "text-service", "resolved": true, "target_symptom_reduction": 1.0, "after_state_observed": true}
```

**Reward:** **+4.303** = the first positive Action reward documented in this
series. `verifier_reason: live_target_recovered`, `resolved: true`,
`twin_resolved: true`, `action_repairs_fault_type: true`,
`positive_credit_eligible: true`. **Feedback:** *"Commands repaired the RCA
target in the twin, but global cascade symptoms remain in the offline
abstraction"* — accurate: `text-service` itself fully recovered
(`target_symptom_reduction: 1.0`), but the broader multi-service SLA
snapshot (`sla_restored: false`) still shows other, unrelated degraded
services (`jaeger`, `post-storage-service`, `social-graph-service` — each
presumably their own separate injected-fault scenarios' concern, not this
one's).

---

## What this trajectory reveals

**The credit-assignment dynamics are real and worth knowing about.** The
correct fix was applied in attempt 1's command 4 and held for the rest of
the trajectory — but two full reward cycles (attempts 1 and 2) scored
strongly negative for an incident that was, in Kubernetes' own state,
already fixed. The cause is a specific interaction: `execute_twin_commands()`
halts a whole batch at its first nonzero-exit command, and both attempts 1
and 2 paired their (successful) real fix with an *additional*,
unrelated removal of a field that was never set, which the K8s API
correctly rejects — with the side effect of preventing the verification
commands later in the same batch from ever running. This isn't unsafe or
incorrect on the API's part, and it isn't obviously wrong on the policy's
part either (methodically checking multiple plausible scheduling fields is
reasonable behavior) — but it means a genuinely correct fix can go
uncredited for one or more retries purely because of what else was bundled
into the same command batch. Worth watching whether this becomes a training
signal problem at scale (systematically delaying credit for a specific
family of fixes) as more trajectories accumulate.

**The token-budget-800 decision, retroactively justified.** Every one of
this trajectory's Action instructions is written in the full 3-part
structure and routinely runs 500–580 tokens — already brushing the 224-token
ceiling in effect at the time. This trajectory is direct evidence for why
the very next change (raising `--max_new_tokens` to 800) was necessary, not
just anecdotal.

**RCA and Action are now both producing real, usable signal.** RCA solved
this scenario in one shot with a twin-verified reward of 3.80. Action, after
three iterations of legitimate exploration, achieved the first documented
full resolution with a reward of +4.30 — both adapters had `updated: True`
for this bundle update, the first time in this session's documented history
that neither role was skipped for zero policy-advantage signal.
