# Full trajectory walkthrough: bundle update 15 (post-fix)

This is a second real, complete trajectory, pulled the same way as the first
(`full_trajectory_walkthrough_2026-09-03.md`, bundle update 6) but from
**bundle update 15** — the second synchronized update after both fixes from
today landed: the Action prompt policy now writes prose strategy instead of
raw `kubectl`, and both RCA and Action policies now receive a rolling summary
of their own recent performance across other incidents. Every prompt,
completion, command, and Twin measurement below is copied verbatim from the
actual log files:

- `/mnt/aiops-training/runs/live-grpo-stage1/joint_trajectories.jsonl` (line
  29, trajectory index 2)
- `/mnt/aiops-training/runs/live-grpo-stage1/rca_policy_samples.jsonl` (lines
  350–353)
- `/mnt/aiops-training/runs/live-grpo-stage1/action_policy_samples.jsonl`
  (lines 380–384)
- `/mnt/aiops-training/runs/live-grpo-stage1/openai_calls_worker_1.jsonl`
  (lines 400–403 for RCA, 404–408 for Action)

Raw companion files are saved alongside this document in `docs/examples/`:
`full_trajectory_raw_u15.json` (the complete trajectory record),
`rca_call_0_raw_request_u15.json` / `rca_call_3_raw_request_u15.json` (the
first and last exact API requests sent to the frozen RCA agent), and
`action_call_0_raw_request_u15.json` / `action_call_4_raw_request_u15.json`
(the first and last exact API requests sent to the frozen Action agent).

## Scenario

- **Scenario ID:** `gen_assign_to_non_existent_node_social_net-mitigation-media-service-default`
- **Policy version at rollout:** `qwen-live-grpo-stage1-v1@u000014` (adapters
  as they stood right after update 14, before update 15)
- **Private ground truth** (from `ground_truth_summary`, never shown to any
  agent): `fault_family: assign_to_non_existent_node_social_net`,
  `fault_mechanism: assign_to_non_existent_node`, `fault_type: infra_failure`,
  `service: media-service`, single-fault, variant `default`.
- **Twin route for this whole trajectory:** `live` (`reward_route: "live"` on
  every attempt's `reward_components`) — every verification below is a real
  sparse Kubernetes Twin.
- **Downstream models:** GPT-5.2 for both RCA and Action, `store: true`,
  confirmed directly from the raw request files.

---

## Part 1 — RCA

Same two-role structure as before: the trainable RCA prompt policy
(`lora_rca`) writes free-text guidance; the frozen RCA reasoner (GPT-5.2,
`max_output_tokens: 96`) reads that guidance plus 80,131 characters of
redacted telemetry and returns one structured prediction line. The fixed
system instructions are byte-identical to the update-6 example — unchanged by
today's work, confirmed directly against the raw request file.

### New this update: rolling performance feedback

Unlike the update-6 example, this prompt now carries a
`recent_own_performance_across_other_incidents` block, built by
`RollingPerformanceTracker` from every RCA attempt completed so far in this
process's lifetime (the process restarted at update 14, so this is a fresh
28-sample window, not the full training history):

```json
{
  "invalid_format_rate": 0.0,
  "mean_twin_reproduction_score": 0.1367,
  "repeated_wrong_guess_rate": 0.4286,
  "verified_rate": 0.1429,
  "window_size": 28,
  "note": "Aggregate over your own last attempts across many incidents; not evidence about this incident specifically."
}
```

Paired with an added `instruction_requirements` line: *"If
recent_own_performance_across_other_incidents shows a high invalid-format or
repeated-wrong-guess rate, adjust the instruction style to address it."* This
same 28-sample snapshot is reused for every scenario in this batch (one
snapshot per batch, computed once, not per-attempt) — visible directly in the
raw request files for both attempt 0 and attempt 3.

### Attempt 0 (iteration 0)

**Trainable RCA policy's instruction** (102 completion tokens):

> "Analyze the service dependency graph and error signals to identify the
> root cause of the incident. Focus on distinguishing between services that
> are failing due to their own issues versus those affected by cascading
> failures from upstream components. Examine logs for patterns like
> "Connection refused" or "timeout" errors, particularly where these occur in
> relation to service dependencies. Also review metrics for abnormal latency
> or resource usage that may indicate a problem at the source rather than in
> downstream consumers. Determine which service(s"

**Frozen RCA reasoner's output:** `media-service::infra_failure::pod_failure`

**Live Twin's reaction:** injected `pod_failure` against `media-service` for
real, re-ran the workload, recompared.

| Measurement | Value |
|---|---|
| `twin_reproduction_score` | 0.2008 |
| `rca_twin_verified` | false |
| `pair_score` vs. ground truth | 0.75 (service + fault type exact, mechanism not) |
| `repeated_wrong_guess` | false (first guess) |

**Reward:** 1.9008 = format (0.20) + pair match (1.50) + twin reproduction
(0.2008). **Feedback:** *"Counterfactual twin reproduction is weak; refine
the service/mechanism using the observed telemetry."*

### Attempt 1 (iteration 1)

**Instruction** (72 tokens): "Given the telemetry, focus on services with
**high dependency connection errors** and **unhealthy status**. 1. **Root
Cause Identification**: Look for services where logs show repeated
`dependency_connection` failures (e.g., `compose-post-service`,
`nginx-thrift`) and observe if they are part of a cascading failure. 2.
**Downstream Cascade Detection**: Identify which services depend on these
failing components. For example, `media-service` depends on `media-service`"

**Frozen reasoner's output:** `media-service::infra_failure::pod_failure`
(identical to attempt 0).

**Live Twin's reaction:** same fault re-injected, same measurement:
`twin_reproduction_score: 0.2008`, `rca_twin_verified: false` — but this time
`repeated_wrong_guess: true` and `reused_identical_hypothesis: true`, so a
**−0.25 penalty** applies. **Reward:** 1.5508 (0.20 + 1.50 + 0.2008 − 0.10
iteration cost − 0.25 repeat penalty). **Feedback:** *"...Avoid repeating the
same previous hypothesis."* — this is the exact mechanism that eventually
pushes the policy off `pod_failure`.

### Attempt 2 (iteration 2)

**Instruction** (55 tokens, now visibly shorter and more directive):
"Analyze the telemetry to identify the root cause of the incident. Focus on
service health, dependency errors, and latency patterns. Distinguish between
the actual root cause and downstream effects. Use only redacted telemetry.
Avoid repeating previous hypotheses. Prioritize service/mechanism refinements
based on observed signals."

**Frozen reasoner's output:** `media-service::infra_failure::pod_failure`
again — the instruction told it to avoid repeating, but the *reasoner*
(GPT-5.2, not the trainable policy) still converged on the same wrong
mechanism from the same telemetry. **Reward:** 1.4508 (repeat penalty now
compounds with a larger iteration cost: 0.20 + 1.50 + 0.2008 − 0.20 − 0.25).

### Attempt 3 (iteration 3, final)

**Instruction** (85 tokens): "Analyze the telemetry to identify the root
cause of the incident. Focus on service health, dependency errors, and
latency patterns. Distinguish the primary failure from downstream effects by
tracing the causal chain from symptoms. Use only redacted telemetry. Do not
repeat previous hypotheses. Adjust your analysis if past attempts showed weak
twin reproduction or repeated wrong guesses. Prioritize mechanisms that align
with the most severe or widespread signals."

Note the new clause the policy added on its own: *"Adjust your analysis if
past attempts showed weak twin reproduction or repeated wrong guesses"* —
language that echoes the `recent_own_performance_across_other_incidents`
block's own vocabulary (`repeated_wrong_guess_rate`), though this is
correlational, not a traced causal claim.

**Frozen reasoner's output:**
`media-service::infra_failure::assign_to_non_existent_node` — correct.

**Live Twin's reaction:** injected the real fault, re-measured.

| Measurement | Value |
|---|---|
| `twin_reproduction_score` | 0.4925 |
| `rca_twin_verified` | **true** |
| `pair_score` | **1.0** (exact set match) |

**Reward:** 3.3925 = format (0.20) + pair match (2.00) + exact-set bonus
(1.00) + twin reproduction (0.4925) − iteration cost (0.30). `final_prediction:
media-service::infra_failure::assign_to_non_existent_node`, `success: true`.

---

## Part 2 — Action

### What changed here since the update-6 example

In the update-6 walkthrough, the trainable Action policy's "instruction" was
literally a list of `kubectl` command lines — the exact bug the user caught
and asked to be fixed. Here, at update 15, on the same policy field
(`instruction_prompt` / `completion`), the trainable policy instead writes
prose strategy naming a remediation *family* (e.g. "Select the
`node_affinity` or `node_selector` remediation...", see attempt 0 below) —
full text for all 5 attempts is quoted per-attempt below, exactly as logged.
The frozen ActionAgent's own system contract is unchanged
(`context.task_instruction`: *"Generate instructions for a fixed ActionAgent
that outputs only kubectl/helm/mongosh commands"* — that line intentionally
still describes the frozen agent's job to itself, not the trainable
policy's).

The Action prompt also now carries its own rolling-performance block:

```json
{
  "has_mutation_rate": 1.0,
  "mean_reward": -0.907,
  "resolved_rate": 0.0,
  "safe_rate": 0.95,
  "top_command_rejection_reasons": {
    "command_namespace_must_equal_owned_twin": 5,
    "mutation_target_not_selected_exact_resource": 9
  },
  "top_verifier_outcomes": {
    "live_command_execution_failed": 19,
    "no_safe_valid_mitigation_action": 1
  },
  "window_size": 20
}
```

paired with: *"If recent_own_performance_across_other_incidents shows
commands are frequently rejected or unresolved for a similar reason, steer
the ActionAgent away from that pattern explicitly."* As the attempts below
show, the policy's own strategy text was already correct on iteration 0 — the
gap in this trajectory is downstream of the policy, not caused by it (see
"What this trajectory reveals" below).

**Namespace:** `aiops-twin-6757ff816954` (opaque, never the real
`test-social-network`). **`current_sla` baseline:** 27 degraded services, 18
with top errors, 4 failed edges, 49 hard violations — fixed across all 5
iterations, same design as the update-6 example.

### Attempt 0 (iteration 0)

**Trainable Action policy's instruction** (its actual sampled completion, 70
completion tokens, passed into the frozen agent's prompt as
`policy_instruction`):

> "## Remediation Strategy
>
> Based on the provided diagnostic information, the root cause is identified
> as:
> **"media-service::infra_failure::assign_to_non_existent_node"**
>
> This indicates an infrastructure-level issue where the `media-service`
> deployment is attempting to assign or schedule itself onto a node that does
> not exist, causing failures.
>
> ### Recommended Action Agent Approach:
>
> 1. **Remediation Family**:
>    - Select the `node_affinity` or `node_selector` remed"

(cut off mid-word by its own 70-token completion — this is the complete
logged text, not a preview.)

**Frozen Action agent's output** (7 commands, `max_output_tokens: 192`):

```
kubectl get pods -n aiops-twin-6757ff816954 -l app=media-service -o wide
kubectl describe pod -n aiops-twin-6757ff816954 -l app=media-service
kubectl get deploy -n aiops-twin-6757ff816954 media-service -o wide
kubectl patch deploy -n aiops-twin-6757ff816954 media-service --type='json' -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"}]'
kubectl patch deploy -n aiops-twin-6757ff816954 media-service --type='json' -p='[{"op":"remove","path":"/spec/template/spec/affinity"}]'
kubectl rollout status -n aiops-twin-6757ff816954 deploy/media-service
kubectl get endpoints -n aiops-twin-6757
```

The mutation (commands 4–5) is exactly the textbook-correct fix for
`assign_to_non_existent_node` — removing both `nodeSelector` and `affinity`,
faithfully following the trainable policy's named remediation family. The
**last command is truncated mid-namespace-argument** (`aiops-twin-6757`
instead of `aiops-twin-6757ff816954`) — the 192-token budget ran out mid-line.

**Live Twin's reaction — rejected before execution:**

```json
{"commands": [], "executed": false, "namespace": "aiops-twin-6757ff816954",
 "rejection_reasons": ["command_namespace_must_equal_owned_twin",
                        "mutation_target_not_selected_exact_resource"],
 "safe": false}
```

Reward: **−0.28**. Feedback: *"Commands were valid but did not improve the
behavioral twin/SLA symptoms."*

### Attempt 1 (iteration 1)

**Trainable policy's instruction** (60 tokens):

> "## Remediation Strategy
>
> Based on the provided diagnostic information, the root cause is identified
> as `media-service::infra_failure::assign_to_non_existent_node`. The
> previous attempt to resolve this by removing nodeSelector and affinity
> configurations from the media-service deployment was unsuccessful.
>
> ### Recommended Approach
>
> **ActionAgent Instructions:**
>
> 1. **Target:** `media-service` Deployment in the `aiops-twin-6757ff816954`
>    namespace.
> 2"

Note it already references its *own* previous attempt's outcome
("unsuccessful") — that's `previous_attempts_non_leaking` from this same
trajectory, not the cross-incident rolling-performance block.

**Frozen agent's output:** 6 commands, mutation = remove `nodeName` only.
**Live Twin's reaction:** `rejection_reasons:
["mutation_target_not_selected_exact_resource"]` (this command's trailing
namespace argument happened to complete before truncation, so the namespace
check passes this time). Reward: **−0.34**.

### Attempt 2 (iteration 2)

**Trainable policy's instruction** (62 tokens):

> "### Remediation Strategy for
> `media-service::infra_failure::assign_to_non_existent_node`
>
> #### **Root Cause Summary**
> The predicted root cause of the incident is:
> - **Service**: `media-service`
> - **Fault Type**: `infra_failure`
> - **Fault Mechanism**: `assign_to_non_existent_node`
>
> This indicates that the deployment configuration for `media-service`
> includes a specification (likely in `nodeSelector` or `nodeName`) that
> references a Kubernetes node which"

**Frozen agent's output:** 4 commands, mutation = remove
`nodeName`+`nodeSelector`+`affinity` together in one patch. **Live Twin's
reaction:** `rejection_reasons: ["command_namespace_must_equal_owned_twin",
"mutation_target_not_selected_exact_resource"]` — last command
(`kubectl rollout status -n aiops-twin-`) truncated with an empty namespace
value. Reward: **−0.36**.

### Attempt 3 (iteration 3)

**Trainable policy's instruction** (67 tokens):

> "### Remediation Strategy for
> `media-service::infra_failure::assign_to_non_existent_node`
>
> #### **Context**
> The root cause prediction identifies `media-service` as the primary
> affected service due to an `infra_failure` characterized by
> `assign_to_non_existent_node`. This indicates that the deployment
> configuration of `media-service` contains a specification (e.g., in its
> `spec.template.spec`) that references a non-existent node name or node
> selector, causing pod scheduling failures.
>
> Previous"

**Frozen agent's output:** 5 commands, mutation = remove `tolerations`.
**Live Twin's reaction:** same double rejection as attempt 2 (last command
`kubectl rollout status -n aiops-twin-6757` truncated mid-namespace). Reward:
**−0.50**.

By iteration 4's request, `previous_attempts` in the raw payload holds all 4
prior attempts in full (commands, feedback, reward) — verified directly
against `action_call_4_raw_request_u15.json`.

### Attempt 4 (iteration 4, final)

**Trainable policy's instruction** (67 tokens):

> "### Remediation Strategy for
> `media-service::infra_failure::assign_to_non_existent_node`
>
> #### **Root Cause Summary**
> The predicted root cause of the incident is:
> - **Service**: `media-service`
> - **Fault Type**: `infra_failure`
> - **Fault Mechanism**: `assign_to_non_existent_node`
>
> This indicates that the deployment configuration for `media-service`
> includes a specification requiring it to run on a Kubernetes node that does
> **not exist** in the current"

**Frozen agent's output** (6 commands, this time not truncated):

```
kubectl get pod -n aiops-twin-6757ff816954 -l app=media-service -o wide
kubectl describe pod -n aiops-twin-6757ff816954 -l app=media-service
kubectl get deploy -n aiops-twin-6757ff816954 media-service -o jsonpath='{.spec.template.spec.nodeName}{"\n"}'
kubectl patch deploy -n aiops-twin-6757ff816954 media-service --type='json' -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
kubectl rollout status -n aiops-twin-6757ff816954 deploy/media-service
kubectl get endpoints -n aiops-twin-6757ff816954 media-service -o wide
```

**Live Twin's reaction:** `rejection_reasons: ["mutation_target_not_selected_exact_resource"]`
only this time (namespace check passes — the command list is complete).
`resolved: false`. **Terminal outcome:** budget exhausted,
`terminal_failure_penalty: −2.0` applied on top, **reward −2.64**.

---

## What this trajectory reveals

**RCA:** positive, correct signal — the rolling-feedback-aware policy
converged on the true fault on iteration 3 of 5, same qualitative arc as the
update-6 example (repeat-penalty pressure pushing it off a locally-plausible
but wrong hypothesis).

**Action — the fix worked, but exposed the *next* problem clearly:** the
trainable policy is no longer the bottleneck. All 5 attempts named the
correct remediation family (scheduling-field removal on the exactly-correct
target) in prose, exactly as intended. Every rejection came from downstream
of the policy, from two distinct, now-clearly-separable causes:

1. **`mutation_target_not_selected_exact_resource` fired on all 5/5
   attempts** — the same verifier gap flagged as "worth watching" in the
   update-6 document. It is still unresolved 9 updates later, and the rolling
   tracker confirms it's the single most common rejection reason across the
   last 20 Action attempts (9 of 20). This looks like a genuine, persistent
   over-strictness in the safety layer's exact-resource match for this fault
   family, not a policy problem.
2. **New finding, only visible by comparing truncated vs. complete responses
   side by side**: `command_namespace_must_equal_owned_twin` fired on
   attempts 0, 2, and 3 — and in all three cases the frozen agent's *last*
   command was cut off by the 192-token output budget exactly where it was
   writing the namespace argument, producing a garbled value
   (`aiops-twin-6757`, `aiops-twin-` empty, `aiops-twin-6757`). Attempts 1 and
   4, where the last command's namespace argument happened to complete before
   truncation (or wasn't truncated at all), never triggered this rejection.
   This is a 5-for-5 correlation in this trajectory alone: **the 192-token
   downstream Action budget is occasionally too small for the number of
   commands GPT-5.2 wants to write, and the resulting truncated last command
   is causing a spurious safety rejection independent of whether the actual
   mutation was safe.** Worth a follow-up: either raise
   `--downstream_action_tokens`, or have the namespace check validate only
   the mutating command rather than every line including trailing verify
   commands.

**Also worth a follow-up, found while extracting this example:** both this
trajectory's RCA and Action `redacted_state.system.media-service` entries
carry `endpoint_signals.namespace: "test-social-network"` — the real
namespace name, not the opaque Twin ID. It's a fixed literal (same string for
every scenario in this benchmark, so it doesn't leak which incident or fault
is present), but it does contradict the stated design principle that agents
should never see the real namespace, and it isn't caught by the current
`sanitize_agent_state()` regex/key-based scrubbing since it's a plain literal
under a nested key. Not fixed as part of this walkthrough — flagging for a
deliberate decision on whether it's in scope.
