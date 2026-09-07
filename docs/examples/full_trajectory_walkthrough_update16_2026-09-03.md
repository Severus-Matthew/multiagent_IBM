# Full trajectory walkthrough: bundle update 16

A third real, complete trajectory, same sourcing standard as the first two
(`full_trajectory_walkthrough_2026-09-03.md` for update 6,
`full_trajectory_walkthrough_update15_2026-09-03.md` for update 15) — every
prompt, completion, command, and Twin measurement below is copied verbatim
from the actual log files:

- `/mnt/aiops-training/runs/live-grpo-stage1/joint_trajectories.jsonl` (line
  35, trajectory index 1)
- `/mnt/aiops-training/runs/live-grpo-stage1/rca_policy_samples.jsonl` (lines
  402–406)
- `/mnt/aiops-training/runs/live-grpo-stage1/action_policy_samples.jsonl`
  (lines 475–479)
- `/mnt/aiops-training/runs/live-grpo-stage1/openai_calls_worker_1.jsonl`
  (line 503 for RCA, lines 504–508 for Action)

Raw companion files in `docs/examples/`: `full_trajectory_raw_u16.json` (the
complete trajectory record), `rca_call_final_raw_request_u16.json`, and
`action_call_0_raw_request_u16.json` / `action_call_4_raw_request_u16.json`.

This trajectory is historically important, not just illustrative: inspecting
its Action side directly — before trusting the fixes already in place —
is what surfaced the multi-file command-parsing bug documented in
`docs/dataset_admission_2026-09-03.md`'s "the same bug existed independently
in three more files" addendum. Every Action attempt below failed for that
exact reason, live, in production, in this exact trajectory.

## Scenario

- **Scenario ID:** `gen_assign_to_non_existent_node_social_net-mitigation-social-graph-service-default`
- **Policy version at rollout:** `qwen-live-grpo-stage1-v1@u000015` (adapters
  as they stood right after update 15, before update 16)
- **Private ground truth:** `fault_family:
  assign_to_non_existent_node_social_net`, `fault_mechanism:
  assign_to_non_existent_node`, `fault_type: infra_failure`, `service:
  social-graph-service`, single-fault, variant `default`.
- **Twin route:** `live` throughout (`reward_route: "live"` on every RCA
  attempt's `reward_components`; the Action side's route shows `"unknown"` —
  see the Action section below for why, which is itself a direct symptom of
  the bug this trajectory exposed).
- **Downstream models:** GPT-5.2 for both roles. RCA `max_output_tokens: 96`;
  Action `max_output_tokens: 320` (the raised budget — confirmed directly
  from the raw request files) — no truncation on any of the 5 Action
  responses this time.

---

## Part 1 — RCA

Full 5-iteration arc: four wrong guesses (the same mechanism, `pod_failure`,
repeated three times after the first), then correct on the last possible
attempt.

### Attempt 0 (iteration 0)

**Trainable RCA policy's instruction** (151 tokens):

> "To identify the root cause among multiple failures:
>
> 1. **Focus on dependency connections**: Look for services with high counts
>    of `dependency_connection` errors, especially those that are upstream
>    dependencies.
> 2. **Check for cascading timeouts**: Services like `post-storage-service`
>    and `user-timeline-service` show timeout errors, which may stem from
>    database issues.
> 3. **Examine service health**: Identify services marked as
>    `infra_degraded` or `app_degraded`, particularly in critical paths.
> 4. **Analyze direct dependencies**: Services such as `nginx-thrift` and
>    `home-timeline-service` appear to be directly connected to failing
>    components.
> 5. **Prioritize by error type**: Root causes often manifest through
>    specific error types like `dependency_timeout` or
>    `dependency_connection`.
>
> Identify one root cause service, its fault type, and an injectible
> mechanism. Only report one root cause per response.
>
> Example format:
> ```
> service_name::fault_type::injectible_mechanism
> ```"

**Frozen RCA reasoner's output:** `social-graph-service::infra_failure::pod_failure`

**Live Twin's reaction:** injected `pod_failure`, re-measured.
`twin_reproduction_score: 0.2704`, `rca_twin_verified: false`, `pair_score:
0.75` (service + fault type exact, mechanism not). **Reward:** 1.9394 = 0.20
(format) + 1.50 (pair match) + 0.2704 (twin) − 0.031 (token-length penalty on
a 151-token instruction — this is the first RCA example this session where
that penalty is visible; it's proportional to instruction length, penalizing
verbosity). **Feedback:** *"Counterfactual twin reproduction is weak; refine
the service/mechanism using the observed telemetry."*

### Attempts 1–3

Same wrong mechanism, `pod_failure`, three more times — each attempt shifting
its narrative (dependency cascades, then `nginx-thrift` as an upstream
culprit) but never changing the actual predicted mechanism, so each is
flagged `repeated_wrong_guess: true` / `reused_identical_hypothesis: true`
and penalized an additional 0.25 on top of a growing iteration cost:

| Attempt | Instruction (abridged) | Prediction | Reward |
|---|---|---|---|
| 1 | "trace dependency failures... start with `home-timeline-service`" | `pod_failure` | 1.5984 |
| 2 | "Root cause: nginx-thrift::infra_failure::pod_failure... avoid repeating the same mechanism" | `pod_failure` | 1.5104 |
| 3 | "Use only redacted telemetry... avoid repeating previous guesses." (46 tokens — its shortest instruction yet) | `pod_failure` | 1.4204 |

Notice the policy's own instruction at attempt 2 explicitly tells the
reasoner *"avoid repeating the same mechanism"* while still being fed a
hardcoded example ending in `pod_failure` in its own draft — and the frozen
reasoner reproduces `pod_failure` anyway. Feedback-following is real but
imperfect at this stage of training.

### Attempt 4 (iteration 4, final)

**Instruction** (165 tokens):

> "Based on the telemetry and feedback:
>
> 1. **Focus on service health signals:** The `service_health` section shows
>    that `social-graph-service` is in `infra_degraded` status, while other
>    services like `home-timeline-service`, `nginx-thrift`, and `jaeger` are
>    marked as `app_degraded`. This indicates an infrastructure-level issue
>    affecting core functionality.
> 2. **Identify cascading failures:** The graph reveals that
>    `home-timeline-service` fails due to a dependency on
>    `social-graph-service` (`social-graph-service` being unready or
>    unhealthy). This suggests that the root cause lies in the
>    infrastructure supporting `social-graph-service`, which then affects
>    its dependents.
> 3. **Reproduce counterfactual twin behavior:** To improve the twin
>    reproduction score, observe the metrics and logs around
>    `social-graph-service`:
>    - It has no log errors but is marked as infra degraded.
>    - Its deployment status shows `"conditions.Available.reason":
>      "MinimumReplicasUnavailable"` and `"status": "False"`.
>    - Logs do not show specific error messages, but the health signal"

(cut off mid-sentence — its own 224-token budget, not the frozen agent's.)

**Frozen RCA reasoner's output:**
`social-graph-service::infra_failure::assign_to_non_existent_node` — correct.

**Live Twin's reaction:** `twin_reproduction_score: 0.6593`,
`rca_twin_verified: true`, `pair_score: 1.0`, `exact_set_match: true`.
**Reward:** 3.4143 = 0.20 + 2.00 (pair match) + 1.00 (exact-set bonus) +
0.6593 (twin) − 0.4 (iteration cost) − 0.045 (token penalty, its longest
instruction of the five). `final_prediction:
social-graph-service::infra_failure::assign_to_non_existent_node`,
`success: true`.

`recent_own_performance_across_other_incidents` was **not present** in any
of this trajectory's RCA prompts — confirmed directly against the raw
prompt JSON. This was the very first batch this process completed after
that day's third restart (the tracker resets on every process restart and
needs 8 accumulated samples before it reports anything), so its absence here
is expected, not a defect.

---

## Part 2 — Action

### The new 3-part template, working as designed

Every one of this trajectory's 5 Action instructions follows the requested
structure precisely — restate the fault and its evidence, then reason
technically, in that order:

**Attempt 0** (100 tokens):

> "### **Remediation Strategy**
>
> #### **1. Restate the Predicted Fault**
> The predicted root cause is:
> > **Service:** `social-graph-service`
> > **Fault Type:** `infra_failure`
> > **Fault Mechanism:** `assign_to_non_existent_node`
>
> This prediction is supported by the `redacted_state.system` field's health
> report for `social-graph-service`, which shows:
> ```json
> "health": {
>   "crashloop_count": 0,
>   "image_pull_error_count": 0,
>   "pods_ready": 0,
>   "pods_total": 1,
>   "pods_unready": 1,
>   "restart_count": 0,
>   "status": "no_ready_endpoints",
>   "warning_event_count": 5
> },
> ```
> Additionally, its deployment signals show:
> ```json
> "conditions.Available.reason": "MinimumReplicasUnavailable",
> "conditions.Available.status": "False"
> ```
> And its endpoint signals are:
> ```json
> "ready_endpoint_count": 0,
> "has_ready_endpoints":"

Cut off mid-JSON by its budget before reaching part (2)/(3) — still a
substantial improvement in specificity over update 15's shorter, vaguer
guidance, and every field it cites is real and correctly quoted from
`redacted_state`. `recent_own_performance_across_other_incidents` was also
absent here, same restart-reset reason as the RCA side.

**Frozen Action agent's output** (6 commands, none truncated —
`max_output_tokens: 320`):

```
kubectl -n aiops-twin-a1ecf5657612 get pod social-graph-service-7fc5d7d577-xg6hj -o wide
kubectl -n aiops-twin-a1ecf5657612 describe pod social-graph-service-7fc5d7d577-xg6hj
kubectl -n aiops-twin-a1ecf5657612 rollout restart deployment/social-graph-service
kubectl -n aiops-twin-a1ecf5657612 rollout status deployment/social-graph-service
kubectl -n aiops-twin-a1ecf5657612 get endpoints social-graph-service
kubectl -n aiops-twin-a1ecf5657612 get pods -l app=social-graph-service -o wide
```

### What actually happened — the bug, live

**Every single command in every one of this trajectory's 5 Action attempts
was rejected with exactly one reason: `kubectl_unsupported_verb:-n`.**

```json
{"num_commands": 6, "safe": false, "unsafe": [
  {"command": "kubectl -n aiops-twin-a1ecf5657612 get pod social-graph-service-7fc5d7d577-xg6hj -o wide",
   "patterns": ["kubectl_unsupported_verb:-n"]},
  {"command": "kubectl -n aiops-twin-a1ecf5657612 describe pod social-graph-service-7fc5d7d577-xg6hj",
   "patterns": ["kubectl_unsupported_verb:-n"]},
  {"command": "kubectl -n aiops-twin-a1ecf5657612 rollout restart deployment/social-graph-service",
   "patterns": ["kubectl_unsupported_verb:-n"]},
  "... (all 6 commands, same pattern)"
]}
```

`command_safety.py`'s `_kubectl_safety()` read `verb = parts[1]` directly.
Every command here puts the namespace flag immediately after `kubectl`
(`kubectl -n <ns> <verb> ...`), so `parts[1]` was always `"-n"` — not a real
verb, not in `ALLOWED_KUBECTL_VERBS`, so every command was rejected before
ever reaching mutation-target or execution logic. `verifier_result.reason:
"no_safe_valid_mitigation_action"`, `has_mutating_command: false` for all 5
attempts (a second, independent copy of the same bug in
`action_reward.py`'s `_is_mutating_command`, which did a raw `"kubectl
patch" in text.lower()` substring check — also defeated by the flag
appearing between the two words). **Reward route: `"unknown"`** rather than
`"live"` for every Action attempt — a direct, visible consequence: the twin
verifier never got a chance to execute anything real, so it couldn't attest
to a live route either.

Rewards across the 5 attempts: −3.19, −3.341, −3.421, −3.227, and −5.316
(terminal, budget-exhausted: `terminal_failure_penalty: −2.0` added on top).
Feedback each time: *"One or more commands were unsafe or unsupported. Use
scoped kubectl/helm/mongosh commands only."* — technically accurate per the
(buggy) safety check, but misleading about the actual cause: nothing about
these commands was actually unsafe.

### Attempts 1–4, briefly

The trainable policy kept refining its technical reasoning attempt over
attempt — citing `inventory.nodes_count: 0` as the smoking-gun evidence by
attempt 3, and the frozen agent kept escalating its remediation precision
correspondingly (`--type=json` targeted removal of just `nodeSelector`/
`affinity` by attempt 1, a broader `--type=merge` null-out of
`nodeName`+`nodeSelector`+`affinity` together by attempt 2, `nodeName`
specifically isolated by attempts 3–4) — every one of these would very
plausibly have fixed the fault. All five were rejected by the same
`-n`-before-verb misparse, never reaching a real execution attempt.

---

## What this trajectory contributes to training, and to this session's fixes

- **RCA:** positive signal, same qualitative shape as the other two
  trajectories — a real repeated-wrong-guess penalty eventually pushing the
  policy toward the correct, twin-verified hypothesis.
- **Action:** strongly negative signal, but for a reason that has nothing to
  do with the trainable policy's guidance quality or the frozen agent's
  command quality — both were good. This trajectory is the direct evidence
  that led to finding and fixing the four-file `kubectl -n <ns> <verb>`
  parsing bug (see `docs/dataset_admission_2026-09-03.md`). Training was
  stopped again after this update specifically because of what this
  trajectory showed.
- **A concrete before/after, in one document:** update 6's Action policy
  wrote raw commands itself; update 15's Action policy wrote correct
  strategy that got rejected by a resource-name parsing bug; this update's
  Action policy writes even more specific, evidence-grounded strategy that
  got rejected by a *different* instance of the same underlying parsing
  defect, one layer earlier in the pipeline (the safety gate, before
  resource-name matching was ever reached). Each fix has been verified
  against real production data before being trusted, and the next
  checkpoint's trajectories are the actual test of whether this one held.
