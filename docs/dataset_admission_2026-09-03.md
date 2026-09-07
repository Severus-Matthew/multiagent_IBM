# Dataset admission for live-Twin reward — 2026-09-03

Split under audit: `configs/dataset_split_624_50` (622 train / 49 test unique
records). Before this work 248 train and 12 test records were admissible for
live reward. This note records why the rest were rejected, what was changed,
and what still requires re-recording.

## Why records were rejected

Three independent causes, established by reading the upstream injector source
and the captured telemetry rather than the admission report alone.

1. **Target-blind upstream injectors (257 records).** `inject_app.py` and
   `inject_virtual.py` in AIOpsLab ignore the requested service for five
   mechanisms: `auth_miss_mongodb` always flips TLS on `url-shorten-mongodb`
   through Helm; `revoke_auth` and `user_unregistered` act only on
   `mongodb-rate`/`mongodb-geo` and are a no-op otherwise; `misconfig_app`
   installs the `yinfangchen/geo:app3` image, whose only difference from the
   normal image is `GeoMongoAddress: mongodb-geo:27777` in the baked-in
   `config.json` (so it is a functional no-op on every service but `geo`);
   `wrong_bin_usage` only rewrites a container whose command contains
   `profile`. A capture produced by a no-op injection is a healthy system under
   a fault label. Relabeling to the "effective" target is not supported by the
   telemetry: the symptom flags on `url-shorten-mongodb` and the other
   datastores appear identically in unrelated control captures (72 of 72
   faithful social-network captures flag `url-shorten-mongodb`; 141 of 141
   hotel captures flag most hotel services), so they are background noise, not
   evidence of the fault.
2. **Header-only trace exports (180 records).** The Jaeger CSV in these
   captures is 96 bytes (the header line); the step transcript shows the
   export was written to a macOS path on the generating machine. Trace-less
   captures are spread evenly across families and target services, which
   rules out a fault-specific cause. Re-processing raw telemetry cannot recover
   spans that were never exported. 152 of the 157 single-reason trace-less
   captures still show the fault at the labeled service in Deployment/endpoint
   state or logs.
3. **One adapter pending live audit (5 records).** `application_config_misconfig`
   on hotel `geo`: the generic adapter looks for a ConfigMap, environment or
   argument endpoint, and the hotel services carry their configuration inside
   the image.

## What changed

* **Channel-aware comparison.** `telemetry_comparator.observed_channels`
  distinguishes a collected trace channel (at least the workload's own request
  edges are present) from one that was never exported. `compare_symptoms_scoped`
  scores the trace channel only when the original capture observed it and
  reports `channels_unobserved_in_original` and `trace_channel_scored`. A
  traced original is scored exactly as before; a Twin that reproduces a
  structural fault faithfully is no longer penalized for request edges the
  incident never recorded. The fail-closed rule for originals with no scoped
  symptoms is unchanged.
* **Evidence-based admission.** `live_dataset_admission` replaces the
  "collected trace edges" gate with `comparable_symptom_evidence`: strong when
  traces were collected or the labeled mechanism is structural (registry field
  `evidence_channels`) and the target's Deployment state is recorded and
  anomalous (replica-count variants count by their recorded count); weak when
  only service-health flags or log error counts exist; none otherwise. Weak
  evidence is admitted only with `--admit_weak_evidence`.
* **Opt-in label corrections** (`training_pipeline.label_corrections`). For a
  multi-fault capture whose one component was a provable injector no-op and
  whose other component was faithful, the private evaluator labels are reduced
  to the faithful component. The agent-visible state is untouched; the
  correction is recorded in `full_state.label_correction`. Records whose every
  component was a no-op are listed as uncorrectable. The manifest is applied
  only when passed explicitly (`--label_corrections`) to the trainer, the live
  matrix audit, the adapter audit, or the dataset freezer.
* **Image-embedded configuration adapter.**
  `fault_mutation_discovery.discover_image_config_corruption` locates the
  configuration file in the fault-independent application source tree,
  chooses the endpoint key naming the target service whose host is a Service in
  the Twin (datastores before caches), resolves the in-container path from the
  Dockerfile, verifies the running container serves that exact file and value,
  and overlays a corrupted copy through a ConfigMap `subPath` mount. The Twin
  injector and the dataset generator (`injector_fixes._patched_misconfig_app`)
  share this implementation. For hotel `geo` it selects `GeoMongoAddress`, the
  same key the upstream buggy image changes. It fails closed for services with
  no service-scoped endpoint (`search`, `frontend`) and for the social network.
* **Warm-application regeneration** (`dataset_generation.warm_cluster`,
  `regenerate.py --reuse_deployment`): reuses a healthy namespace between
  scenarios, skipping OpenEBS and Prometheus rebuilds and the Helm
  uninstall/reinstall, gated by a fail-closed cleanliness check before reuse
  and after recovery. Not yet exercised against the cluster.

Offline audit: `python -m training_pipeline.audit_comparable_evidence_admission`
(31 checks). `audit_pipeline_contract` still passes.

## Admission after the change

`artifacts/dataset-live-admission-622-49-v4-*.json`; id lists under
`configs/curriculum_622_49_v2/admission_v4/`.

| Policy | Train admissible | Test admissible |
|---|---|---|
| Previous contract (trace edges required) | 248 | 12 |
| Strict evidence, no label change | 355 | 24 |
| Strict evidence + label corrections | 454 | 29 |
| Strict evidence + label corrections + config adapter verified live | 459 | 29 |
| Weak evidence admitted + label corrections | 500 | 37 |

Remaining after the strict corrected policy: 117 train and 12 test records
must be re-recorded (`regeneration_queue_v3.json`: single-fault captures of the
five target-blind mechanisms on non-effective targets, multi-fault captures
whose every component was a no-op, and one trace-less `network_loss` capture
with no target evidence); 46 train and 8 test are weak-evidence only (mostly
trace-less `target_port_misconfig` and `mongodb_auth_missing`); 5 train records
waited on the config adapter's live audit, which passed 2 of 2 full-lifecycle
probes on the bootable hotel Twin (`config-adapter-audit-20260903-v2`,
reproduction score 0.7042 with the trace channel scored, clean recovery).

Re-recording cost with the historical lifecycle is about 23 minutes per
scenario (median gap between consecutive `DONE.json` files); the warm mode
exists to remove the rebuild share of that but has not been measured.
Regeneration mutates the source namespaces the Twin renders from, so it must
not run concurrently with live-Twin audits or training on the same cluster.

The RCA and Action retry defaults are seven attempts.

## Addendum: hotel Twins were not bootable

The first live probes of the config adapter injected and manifested the fault
but failed the recovery gate ("control did not recover"). Inspection of the
Twin namespaces showed the cause is independent of the adapter: the planner's
one-hop support rule selects the frontend's direct dependencies (profile,
rate, recommendation, reservation, user) but not *their* datastores, and the
hotel Go services dial their MongoDB/memcached targets in `main` and exit when
they are absent. The bystanders therefore crash-loop, the clean baseline only
passes during restart windows, and recovery can never stabilize. In today's
live matrices hotel cases passed 14 of 33 and 14 of 31 versus social 9 of 14.

Fix: `application_topology` now reports `startup_edges` (targets referenced in
`cmd/<service>/main.go` plus the consul registry), the verifier carries them
into the planner state as `startup_required`, and `build_sparse_live_twin_spec`
closes every kept service over its startup-required dependencies (reason
`startup_required_dependency_of_<caller>`). The social network has no Go
startup edges and its Twins are unchanged. Hotel Twins grow (geo: 11 to 19 of
24 services; user: 9 to 14) but boot deterministically. Earlier hotel adapter
verifications and matrix rows were collected on the flaky Twins and should be
re-qualified on the bootable ones.


## Addendum: review findings and fixes (post-startup-closure)

An adversarial review of the day's changes (partially completed before hitting
usage limits) surfaced three real defects, each verified directly and fixed:

1. **Trace hydration was missing from regeneration.** AIOpsLab's `get_traces`/
   `get_metrics` actions print a pointer to a file under the AIOpsLab checkout's
   `trace_output/`/`metrics_output/` directories; the CSV never lands in the
   scenario's own `builtin_api_outputs/` on its own. The historical corpus only
   has traces because a separate `hydrate_telemetry_artifacts.py` script copied
   them in after the fact. `dataset_generation/regenerate.py` now calls the same
   hydration logic per scenario, immediately after `run_one()`, before checking
   `_trace_rows`. Without this fix every regenerated capture would have been
   rejected by the trace-presence gate regardless of injection correctness,
   which is exactly what the first two smoke captures showed (`trace_rows: 0`
   despite `injection_verified: true`).
2. **Alias collision in evidence lookup.** `_system_entry` matched on alias
   overlap alone, and `_service_aliases("mongodb-rate")` includes `rate`, so a
   `system` dict containing both entries could return the datastore's structural
   signature for an app-service label (or vice versa). Fixed to require an exact
   name match before falling back to alias overlap. Verified by direct
   construction (`rate` returns the `rate` entry even when `mongodb-rate` is
   listed first). The corpus-wide admission counts are unchanged after the fix
   (459/29), so this was a correctness defect, not one that had inflated the
   reported admissible set.
3. **Scale-variant evidence was too loose.** `count_variant` marked any
   `scale_replicas_zero` capture as strong evidence whenever the target had any
   recorded Deployment tokens, without checking the recorded replica count
   against the labeled variant (`scale_2`, `scale_3`). Fixed to require
   `replicas_desired` to equal the number parsed from the variant name. Verified
   directly: a healthy target (desired=1) under a `scale_2` label is now
   correctly `none`; a target with desired=2 is `strong`.

A fourth finding — `apply_label_correction` had no defense against being
applied to a regenerated record that already carries generator-recorded
injection evidence — is not currently reachable (the freezer's
`--override_processed_states` composition already gates corrections to records
from the original source), but a `ValueError` guard was added to
`apply_label_correction` itself so misuse fails loudly rather than silently
mislabeling a newly-faithful regenerated capture.

One finding was not fixed and is an open item: the reproduction threshold
`tau=0.4702` was calibrated exclusively on originals with a collected trace
channel (four four-channel matched triplets). For originals without traces, the
channel-aware comparator now renormalizes across three channels, which raises
the achievable ceiling from 0.75 to 1.0. Traced originals reproduce their
recorded scores exactly (verified against all threshold-control and balanced-
matrix rows). For untraced originals — mostly the structural mechanisms
(`assign_to_non_existent_node`, `scale_replicas_zero`) — the deployment-state
channel (0.35 of 0.75, i.e. renormalized to ~0.47) still carries a hard,
discrete signal (exact replica-count/readiness tokens) that a wrong-service
Twin should not match, so the risk of a wrong reproduction crossing tau is
believed low but has not been validated with dedicated negative controls under
the untraced/3-channel path specifically. Treat live-reward results for the
untraced-structural subset as provisional until that recalibration is done;
this does not block using them as training signal.


## Addendum: weak-evidence records wired into the trainer

`--admit_weak_evidence` is now available on `training_pipeline.train_qwen_live_grpo`
(threaded through `digital_twin_runtime.live_capabilities.audit_live_training_records`).
Without it, the trainer's own live-training preflight rejects weak-evidence
records at the strict default even if `--scenario_ids` includes them, so
passing the "with weak evidence" selection alone was not sufficient to
actually train on that bucket.

## Addendum: pre-launch correctness sweep (2026-09-03, before requested launch)

Before starting real training, per an explicit request to verify "logically and
mathematically correct" and have "everything perfectly in place," the following
was run and checked directly (not assumed from prior documentation):

**Redaction/leak safety, empirically re-verified.** The raw `state_abstraction_compressed.json`
files do carry a top-level `scenario_id` and, inside `observability_metadata`,
file paths that embed the descriptive scenario id. This is expected: those
files are not the agent's prompt. `sanitize_agent_state()` recursively drops
`scenario_id`/`task`/every `BANNED_AGENT_KEYS` entry and regex-scrubs the
descriptive-scenario-id pattern out of every string value, including inside
paths. `end_to_end_loop.py` hard-fails (`raise ValueError`) if joint training is
invoked with anything other than `agent_input_mode="training_safe"`, so there is
no reachable bypass from the real training entry point. Verified directly: ran
`sanitize_agent_state` + `agent_input_safety_report` + `build_bounded_agent_state`
(at the trainer's real default budget, 100k chars) across 60 randomly sampled
records spanning both apps and multi-fault cases — zero leaks, zero budget
violations, max bounded size 81,652 of 100,000 chars.

**GRPO/RL math audits re-run** (all files involved are unchanged from the
already-audited baseline this session): `audit_grpo_objective`,
`audit_factorized_grpo_learner`, `audit_exact_token_grpo_replay`,
`audit_hf_exact_token_sampler`, `audit_hf_peft_two_adapter`,
`audit_streaming_grpo_optimizer`, `audit_injectible_rca_contract`,
`audit_parallel_rollout_replication`, `audit_pipeline_contract`,
`audit_fault_mutation_discovery` — all PASS.

**One stale test found and fixed.** `audit_live_rollout_interface.py` failed:
its synthetic mock verifier predates the `reward_route`/`after_state_observed`
credit-eligibility gate added to `action_reward.py`/`end_to_end_reward.py`
(a safe-no-op-cannot-earn-reward safeguard) and never populated either field.
Traced end to end: the real `SparseLiveTwinVerifier.apply_commands_and_score()`
does correctly set `reward_route="live"` (via `_with_live_route`) and embeds
`resolution.after_state_observed` (from `score_resolution`). Confirmed by
constructing a realistic verifier result and feeding it through `action_reward`
and `end_to_end_reward` directly: a genuine successful remediation scores
`success=True`, `action_policy_return=1.0`; a safe no-op scores `success=False`,
negative reward. The mock was updated to match the real contract; the audit now
passes. This was a test-maintenance gap, not a production bug — but it could
have masked a real future regression in this exact gate, so it was worth
finding and fixing before relying on the audit suite as a spec.

**Trainer's own live-training preflight, verified against the real launch
configuration.** `audit_live_training_records` (called by
`train_qwen_live_grpo.py` before any rollout) was run directly against both
candidate training sets. Without `--label_corrections` it correctly rejects the
129 corrected multi-fault records (it re-derives faithfulness from the
uncorrected labels), which only confirms the flag is load-bearing. With
`--label_corrections` applied exactly as the trainer applies it: strict set
459/459 supported, weak-evidence set 505/505 and 37/37 supported. Zero
rejections in the actual launch configuration.

**Infrastructure drift found and fixed.** `HF_HOME`/`HF_HUB_CACHE`/
`TRANSFORMERS_CACHE` in `~/.bashrc` pointed at `/mnt/hf-cache/huggingface`,
which no longer exists on this host — a live GPU preflight
(`audit_qwen_policy_sensitive_joint_gpu`) failed immediately trying to resolve
it. The actual complete model cache (61G, all 16 safetensor shards present) is
under `/mnt/aiops-training/cache/huggingface`, the same volume as the training
artifacts. `~/.bashrc` updated to point at the correct path. This would have
failed the real training launch identically had it not been caught here.

## Addendum: real-hardware policy-sensitivity check (background, non-blocking)

`audit_qwen_policy_sensitive_joint_gpu` ran to completion after the HF cache fix
(peak 59.6 GiB, model loads in ~10s once cached). Result: 4/4 sampled RCA and
Action policy completions were textually unique, and their downstream
commands were correspondingly unique (`unique_downstream_outputs: 4`) — the
temperature=0.9/64-token sampling config produces genuine policy diversity,
not the degenerate identical-completion failure mode Section 9 of the original
handoff warned about. Exact-token replay ratios were exactly 1.0 pre-update
(`action_exact_replay.all_ratio_one: true`).

The run reported `status: ZERO_POLICY_SIGNAL` and uniform zero policy
reward/advantage across all 4 samples. Traced to the cause: this audit
hardcodes `BehavioralTwinVerifier` (the offline proxy), whose result never
carries `reward_route="live"`. `end_to_end_reward.py`'s
`optimizer_credit_eligible = live_reward and not telemetry_incomplete` forces
both `rca_policy_return` and `action_policy_return` to exactly `0.0` whenever
the route isn't live — by design ("Only measurements produced by the live
Kubernetes verifier may affect an optimizer return"). This audit is a
mechanics/plumbing check, not a live-reward check, and a uniform zero here is
the correct, intended output of that gate, not a defect. The downstream
Action agent's sampled commands were also all read-only (`kubectl get ...`,
no mutation) for this particular scenario/config, which independently would
have zeroed action credit regardless of route. Neither observation affects the
real `--twin_mode live` launch path, whose reward composition was separately
verified directly (see the credit-eligibility trace earlier in this document).

## Addendum: Action prompt policy was writing commands instead of guidance

Caught by inspection of a real trajectory: the trainable Action prompt policy
(`lora_action`) was outputting raw `kubectl` command lines instead of
strategic guidance for the frozen ActionAgent, unlike the RCA side which
correctly produces prose. Root cause, confirmed by comparing the two prompt
builders side by side (`rca_loop.build_rca_policy_prompt` vs
`action_loop._build_action_policy_prompt`): RCA's `instruction_requirements`
explicitly frame the task as reasoning guidance ("tell the solver how to
distinguish..."), while Action's read like direct command-authoring
instructions and never told the policy that a separate agent, not itself, is
responsible for the actual commands. Combined with the Qwen3-Coder base
model's coding bias and this being extremely early in training (the adapter
barely trained), the policy resolved that ambiguity by writing commands.

Fixed `_build_action_policy_prompt` to explicitly state the policy's own role
("write natural-language remediation STRATEGY guidance... The ActionAgent
outputs the actual commands, not you") and reworded `instruction_requirements`
to match RCA's reasoning-guidance framing. Verified the downstream ActionAgent's
own context (`context["task_instruction"]`/`context["action_requirements"]`,
which correctly does ask *it* for commands) was not touched — that field is
consumed only by the frozen agent, never by the trainable policy, confirmed by
grepping for its usage.

Also added cross-trajectory performance feedback for both roles
(`training_pipeline/rolling_performance_feedback.py`): a bounded rolling
window over each policy's own recent attempts (verification rate, mean twin
reproduction score, format-validity rate for RCA; resolved/safe/mutation
rates, mean reward, and the most common command-safety rejection reasons for
Action), recomputed once per batch and folded into the next batch's prompt as
`recent_own_performance_across_other_incidents`. Every field is either public
policy-vocabulary information or self-referential to the policy's own past
outputs — never anything derived from a hidden label. Omitted entirely (not
even an empty key) until the tracker has at least 8 samples, and omitted from
the prompt whenever `None`, so every existing golden-prompt audit that calls
these functions without the new argument is byte-for-byte unaffected.

The one place this took real care: the outer `policy_prompt` computed directly
in `run_rca_grpo_episode`/`run_action_prompt_optimizer_loop` for the training
record must stay byte-identical to what the trainable policy wrapper
internally reconstructs and actually feeds to the tokenizer — that's the
invariant the whole exact-token GRPO audit suite protects. Verified directly
(not just via the existing None-only audits): built both prompts with a
non-None, identical `recent_performance` value from both call sites and
confirmed exact string equality, and confirmed a differing value actually
changes the prompt (the field is really wired in, not silently dropped). Full
audit suite (grpo objective, factorized learner, exact-token replay/sampler,
dual-adapter isolation, streaming optimizer, pipeline contract, live rollout
interface) re-run clean after these changes.

Training resumed from `checkpoints/update-00000013.pt` with these fixes in
place, same dataset (`full-622-49-v1`, 557 admissible train records) and same
launch configuration otherwise.

## Addendum: W&B run fragmentation on resume, and the fix

That resume above did not continue the original Weights & Biases run. The
launch command passed `--wandb_run_name live-grpo-stage1-20260903-resumed`
only, and `wandb.init()` was being called with just `name=`/`project=`/
`entity=`/`config=`/`tags=` — no `id=`, no `resume=`. By W&B's design, `name`
is purely a cosmetic display label; without an explicit `id`, `wandb.init()`
always creates a brand-new run. That resume created run `0jqeczau`, disconnected
from the original run `wsxhzf27` (the run all updates 1-13 were logged into).
Confirmed via WebSearch that W&B has no supported way to merge two
already-created, separately-ID'd runs after the fact after asking about it
directly (open GitHub feature request, not implemented); the only supported
continuity mechanism is `wandb.init(id=<run_id>, resume="must")` at the time
the run is (re)started.

Fix, in `training_pipeline/wandb_logger.py`: `WandbRunLogger.__init__` now
accepts an optional `run_id`; when set, `start()` passes `id=run_id,
resume="must"` into `wandb.init(...)`. After `wandb.init()` returns,
`self.run_id` is reset to `self._run.id` (the actual resolved id, whether
newly created or resumed), so callers can read back and persist it. In
`training_pipeline/train_qwen_live_grpo.py`: added a `--wandb_run_id` CLI
flag; when omitted and `--resume` is set, the code peeks the checkpoint file
directly (`torch.load` on the checkpoint path, before the trainer/model is
constructed) for `last_update.wandb_run_id` and uses that as the fallback, so
future resumes recover run continuity automatically without needing the flag
again. Every training update now writes `update["wandb_run_id"] =
wandb_logger.run_id` into the checkpoint's `last_update` payload to make that
fallback possible starting from this run onward.

Process: stopped the fragmented-run process (`SIGTERM`), found and cleaned up
two `aiops-twin-*` Kubernetes namespaces left in `Active` state (the SIGTERM
landed mid-rollout in both parallel workers, before their `end_trajectory()`/
`destroy()` cleanup path could run) via `kubectl delete ns ... --wait=false`,
confirmed the cluster was clean of twin namespaces, then relaunched with the
identical command plus `--wandb_run_id wsxhzf27`. No training progress was
lost — `bundle_update_step` was still 13 (no update had landed on the
fragmented run before it was stopped). Verified the fix worked directly from
the process output: the run directory changed to
`wandb/run-20260903_175410-wsxhzf27`, W&B printed `Resuming run
live-grpo-stage1-20260903-resumed` (not `Syncing run`, which is what a new-run
creation prints), and the app's own log line confirmed
`run_id=wsxhzf27` before model loading proceeded.

## Addendum: W&B history stuck at step 12 despite the run being correctly attached

Even after the fix above, the W&B *dashboard* itself stayed frozen at step 12
for updates 13 and 14, despite the run being genuinely, correctly attached
(confirmed by the user's own screenshot and independently by querying
`wandb.Api().run(...).scan_history()`, which is authoritative server-side
state, not a UI cache). Ran a 3-way parallel investigation (external W&B SDK
semantics via WebSearch/source reading, local forensics across every
`wandb/run-*` directory and process on the machine, and a from-scratch audit
of the GRPO loss formula, since "why is the loss starting so low" was asked in
the same breath and is a related but separate correctness question) rather
than guessing.

Root cause, found independently by two agents and confirmed against the
installed wandb 0.29.0 source directly: `wandb_logger.py`'s
`log_training_update()` called `self._wandb.log(row, step=int(step))` with
**no `commit=` argument**. Per `Run.log`'s own docstring: `commit` defaults to
`True` only when `step=None`; whenever an explicit `step=` is passed (which
this code always does, since `step` is the trainer's `bundle_update_step`),
`commit` defaults to **`False`**. A `commit=False` row is held as the run's
"pending" row and only becomes visible/durable once a **later** call advances
the step past it, or `commit=True` is passed explicitly, or the run calls
`finish()`. This exactly explains every observed symptom:
- Steps 1-12 appeared because each was flushed as a side effect of the *next*
  update's log call advancing the step.
- Step 13 never appeared because the process was `SIGTERM`'d mid-rollout for
  update 14, before any later call could flush it — this point is genuinely,
  permanently lost from the dashboard (confirmed: attempting the standard
  `wandb sync --no-skip-online` recovery against that dead run's local
  directory failed with `transactionlog: error reading: unexpected EOF`, i.e.
  the local backup file itself was truncated by the ungraceful kill). The real
  numbers for that step are not lost — they are in `training_events.jsonl`
  (`rca_loss=0.0003928`, `action_loss=0.0007678`).
- Step 14 was simply still "pending" (not lost) — it surfaced automatically
  once update 15's log call fired, with no code change needed for that
  specific point.

A matching, near-identical community report (wandb/wandb GitHub issue #12126)
was found describing the same symptom with the same maintainer explanation,
corroborating this is documented, current behavior rather than a version-
specific regression.

Fix: added `commit=True` to that one `self._wandb.log(...)` call in
`training_pipeline/wandb_logger.py`. `log_training_update()` makes exactly one
`log()` call per training update, so there is no legitimate case here for
deferring/accumulating a row — every future update will now appear on the
dashboard immediately instead of one step behind, and will no longer be
losable to an ungraceful kill landing between one update's log call and the
next. This fix is loaded on the *next* resume; it cannot retroactively affect
the already-running process (Python has the old bytecode loaded in memory),
so the currently running process keeps the one-step-lag behavior for the rest
of its life — that lag does not lose data on its own, it only would in
combination with another ungraceful kill, which is not planned again for this
run.

Local forensics also surfaced two purely cosmetic findings, left as-is:
`0jqeczau` (the accidentally-fragmented run from the first resume attempt)
never completed a single training update under it and holds zero real data —
safe to delete later if a clean run list is wanted. Bare `wandb sync` reports
"Skipped 6 online run(s)" because none of today's dead run attempts ever
called `wandb.finish()`, so they likely still show as "running" on the
dashboard indefinitely; harmless, but explains why the run list may look busier
than the training history actually is.

## Addendum: why RCA/action loss values are so small (~1e-4)

Investigated end-to-end (formula from `factorized_grpo_learner.py`, cross-
checked against real per-update diagnostics in `training_events.jsonl` for
updates 9-14) rather than assuming. Verdict: **expected, not a bug.**

The per-token loss is a standard clipped PPO surrogate
(`-min(ratio*adv, clip(ratio,0.8,1.2)*adv)`) plus `kl_coeff * per-token KL`,
averaged per decision, then weighted-mean over decisions/trajectories/groups —
there is no batch-wide division by total token count that would mechanically
shrink an otherwise-normal-sized loss.

Two things combine to make the *reported scalar value* tiny even though
learning is real:
1. Advantages are group-mean-subtracted (standard GRPO), so they are exactly
   zero-mean within each optimizer group by construction.
2. `mean_ratio` is ~1.0 to 8-9 decimal places and `mean_clip_fraction` is 0.0
   for essentially every update — expected, since this is a single on-policy
   gradient step per rollout (no PPO multi-epoch reuse), so the sampling
   policy and the policy being scored are the same at forward-pass time.

With ratio≈1 and no clipping, the surrogate collapses to ≈`mean(advantage)`
over the group, which is ≈0 by construction (1). What's left in the reported
loss is essentially just `kl_coeff × mean_sampled_kl` — verified arithmetically
against the actual JSONL numbers for several updates (e.g. update 14/action:
`0.01 × 0.0153417 = 0.00015342` vs reported `0.00015317`; update 13/action:
`0.01 × 0.076778 = 0.00076778`, an exact match) — confirming the small loss
*value* is the expected algebraic identity of this specific combination, not a
sign of a dead learning signal.

Confirmed separately that the actual gradient signal is healthy and not
vanishing: `grad_norm_before_clip` sits in a stable 0.36-0.71 band across
these updates (nowhere near zero), and `nonzero_advantage_groups`/
`nonzero_advantage_trajectories` are indeed nonzero for every update that
wasn't legitimately skipped (one exception: update 10/action skipped cleanly
via the existing `zero_policy_advantage_signal` gate, `updated: false`, which
is correct, intended behavior for a genuinely uninformative batch). A near-
zero *loss value* alongside a healthy, non-vanishing gradient norm is a known
property of PPO/GRPO-style objectives at ratio≈1 — this is exactly why the
existing wandb dashboards already prioritize `grad_norm`, `clip_fraction`, and
`mean_sampled_kl` over the raw loss scalar for these roles. One follow-up
worth doing later: `ratio_min`/`ratio_max` are already computed per update but
not currently written into `training_events.jsonl`/wandb — adding them would
make ratio dispersion (masked by the mean) visible too, though this isn't
blocking anything today.

## Addendum: rollout to load the W&B `commit=True` fix

Given the fix only affects future launches, not an already-running process,
asked the user whether to interrupt training once more to load it or leave it
running with the one-step-lag dashboard behavior; the user chose to restart.
Sequence: confirmed checkpoint 15 was the latest on disk, `kill -TERM` the
running process (PID 3618399), confirmed it exited, found (again) an orphaned
`aiops-twin-dd173edb892e` namespace stuck `Active` (same ungraceful-kill-mid-
rollout pattern as the previous stop) and deleted it (`kubectl delete ns ...
--wait=false`), confirmed it transitioned to `Terminating`, then relaunched
with `--resume checkpoints/update-00000015.pt --wandb_run_id wsxhzf27` (same
launch command otherwise, tags gained `wandb-commit-fix`). Verified via the
tmux/log output: `wandb: Resuming run live-grpo-stage1-20260903-resumed`
printed again (correct run, not a new one), weights loaded on both workers,
cluster confirmed fully clean of `aiops-twin-*` namespaces immediately after.
Next update (bundle_update_step 16) should land in ~30-45 minutes and should
now appear on the W&B dashboard immediately rather than one step behind.

## Addendum: real bugs found while writing the update-15 trajectory doc, all fixed

Building `docs/examples/full_trajectory_walkthrough_update15_2026-09-03.md`
(a second full real trajectory, requested to show the Action-prompt fix and
rolling-feedback in a real example) surfaced three concrete, previously
undiagnosed defects. All three are fixed and verified below; training was
stopped once more to load them, from checkpoint 15.

**1. `mutation_target_not_selected_exact_resource` was a parser bug, not a
policy or safety-strictness problem.** `digital_twin_runtime
/live_action_executor.py`'s `_resource_target()` assumed a fixed positional
layout (`kubectl patch deploy NAME -n ns ...`). Every command either agent
naturally writes puts the namespace flag *before* the name
(`kubectl patch deploy -n ns NAME ...`), which is standard kubectl style.
Reproduced directly:
`_resource_target(['kubectl','patch','deploy','-n','aiops-twin-...','media-service',...])`
returned `('deploy', '')` — an empty resource name, which always fails the
"is this an owned resource" check regardless of how correct the mutation
actually was. This has likely been silently rejecting most or all
correctly-targeted mutations in both trajectory walkthroughs. Fixed by
rewriting `_resource_target()`/adding `_positional_args()` to scan for
positional tokens while skipping recognized flags (namespace, specifically,
since that's the one observed in both flag-value orderings), instead of
reading fixed indices. Verified against every real command shape pulled from
both this trajectory and the earlier update-6 one (`-n` before the name, `-n`
after the name, `kind/name` combined form for `rollout status`, `scale`,
`delete`) — all now resolve to the correct `(kind, name)`. Did not run the
existing live-cluster integration audit (`audit_live_action_remediation.py`)
against this change since it spins up its own Twin session and training was
actively using the cluster at the time; relying on the direct reproduction
above plus the next real training update's Action success rate as
confirmation.

**2. The frozen Action agent's 192-token output budget was too small,
occasionally corrupting the final command's namespace argument and causing a
second, separate rejection independent of (1).** In four of five attempts in
the update-15 trajectory, the last of 4-7 commands was cut off by the token
budget; three of those four cuts landed mid-namespace-argument (e.g.
`aiops-twin-6757` instead of `aiops-twin-6757ff816954`), and all three
correlated exactly with a `command_namespace_must_equal_owned_twin`
rejection that the other two (uncut, or cut after the namespace already
completed) did not get. Fixed by raising `--downstream_action_tokens` from
192 to 320.

**3. A real namespace leak, worse than first estimated.** The initial
scan (single scenario) found `redacted_state.system.<service>
.endpoint_signals.namespace` carrying the literal real namespace
(`test-social-network`) instead of being stripped. Added `"namespace"` to
`BANNED_AGENT_KEYS` in `agent_input_safety.py` and re-swept all 671 dataset
records for the literal string — found two more, structurally different leak
paths the single-scenario check had missed entirely:
  - `observability_metadata.system.collection_status.{ok_collections,
    failed_collections}` stores `{collected_file: literal shell command}`,
    and the recorded command text (e.g. `"kubectl get pods -n
    test-social-network -o json"`) embeds the real namespace. This is pure
    collection provenance, not diagnostic signal, so `bounded_agent_state.py`
    now collapses it to `{num_ok, num_failed, ok_collection_count,
    failed_collection_count}` instead of passing the raw command map through.
  - In-cluster K8s service DNS names (`<service>.test-social-network
    .svc.cluster.local`), used as **dict keys** (not values) in
    `dependency_error_counts`/`global_dependency_error_counts` maps derived
    from logs. `sanitize_agent_state()`'s regex scrubbing previously only
    ever ran on string *values* — dict keys passed through unscrubbed. Added
    a second regex (`_K8S_NAMESPACE_FQDN_RE`) that redacts just the namespace
    label while preserving the service name and the "this is a K8s service
    reference" shape (`compose-post-service.[redacted_namespace]
    .svc.cluster.local`), and applied both regexes to keys as well as values
    in `_sanitize()`.
  Verified with a full sweep of all 671 admissible records before and after:
  16 scenarios leaked the real namespace string somewhere in their bounded
  state before this fix; 0 leak after. Re-ran `audit_pipeline_contract.py`
  afterward (50/50 checks pass) specifically because sanitizing dict keys is
  a more invasive change than the original value-only scrubbing — confirmed
  no downstream code relies on any literal key name this touches.

**4. Separately, the user asked for the trainable Action policy's own prompt
template to be more explicit and technical** — restate the fault and its
supporting evidence, reason step by step about the fix, then explicitly hand
off to the frozen agent to "give the exact command." Rewrote
`_build_action_policy_prompt`'s `task`/`instruction_requirements` in
`action_loop.py` into that 3-part structure. Also raised the trainable
RCA/Action policy's own completion budget (`--max_new_tokens`) from 96 to
224, since both roles' instructions were routinely cut off mid-sentence at 96
in the real trajectories, and the new Action template asks for more content.
Verified the exact-token invariant still holds after the prompt rewrite
(`audit_hf_exact_token_sampler.py`, `canonical_action_prompt_binding: true`)
— the function's signature didn't change, so both call sites (the direct
build in `run_action_prompt_optimizer_loop` and the trainable policy
wrapper's internal rebuild) stay byte-identical by construction.

Training restarted from checkpoint 15 with all four fixes loaded, same run
(`wsxhzf27`), tags gained `action-parser-fix,namespace-redaction-fix,
action-prompt-restructure,bigger-token-budgets`.

## Addendum: the parser fix above was incomplete — caught live, in the very first post-restart batch

Checked the first real Action samples after the restart above before update
16 even landed (watching GPU utilization + fresh OpenAI call timestamps to
confirm the process wasn't stuck, then inspecting `action_policy_samples
.jsonl` directly). Found a `social-graph-service` attempt with
`has_mutating_command: false, verifier_reason: no_safe_valid_mitigation_action`
despite the frozen agent's actual command list containing a clearly-correct
patch:

```
kubectl -n aiops-twin-56649fa58547 patch deploy social-graph-service --type=json -p='[{"op":"remove","path":"/spec/template/spec/nodeName"}]'
```

This puts the global `-n <namespace>` flag *before the verb itself*
(`kubectl -n ns patch ...`), a shape neither the earlier `_resource_target()`
fix nor the original code anticipated. `execute_twin_commands()` read the
verb as `parts[1]` directly (no flag-skipping at all) — with `-n` at index 1,
`verb` resolved to `"-n"`, which isn't in `{"patch","scale","delete"}`, so
the mutation branch never ran and the command was silently never recognized
as a mutation attempt. This is the same root defect as before (fixed-position
parsing that assumes flags never precede the token being read), just hitting
verb detection instead of resource-name detection, and it's the second time
in one day GPT-5.2 produced a valid-but-differently-ordered kubectl
invocation that fixed-position parsing didn't anticipate.

Fixed properly this time by deriving the verb the same way as the resource
target: `_positional_args(parts, 1)` walks every token after `kubectl`,
skipping flags (including a leading `-n`) wherever they appear, so
`positional[0]` is reliably the verb regardless of flag placement.
`_resource_target()` now takes this already-flag-stripped positional list
directly instead of re-deriving indices from raw `parts`, removing the
possibility of the two functions disagreeing about where the verb is.
Verified against 10 real/adversarial command shapes covering every
flag-ordering combination seen in production so far (`-n` before/after verb,
before/after resource name; `rollout status kind/name` with `-n` in both
positions; non-mutating `get` with `-l`) — all resolve correctly.
`audit_pipeline_contract.py` re-run clean (50/50).

Stopped and relaunched a second time from checkpoint 15 (no progress lost —
update 16 still hadn't landed), tags gained `action-parser-fix-v2`
(superseding the incomplete `action-parser-fix`).

## Addendum: the same bug existed independently in three more files — consolidated into one shared module

Update 16 landed under `action-parser-fix-v2` and was checked immediately
(before trusting it): a `social-graph-service` attempt still showed
`has_mutating_command: false, verifier_reason: no_safe_valid_mitigation_action`
despite the frozen agent writing a textbook-correct patch:
`kubectl -n aiops-twin-1475fe97436e patch deployment social-graph-service
--type=json -p='[...]'`. Traced this to the SAME root defect
(fixed-position parsing assuming the verb immediately follows the program
name) existing **independently, in three more files** that were never
touched by the `live_action_executor.py` fix:

- `training_pipeline/command_safety.py` — `_kubectl_safety`/`_helm_safety`
  read `verb = parts[1]` directly. For this command, `parts[1] == "-n"`,
  which is in neither `DENY_KUBECTL_VERBS` nor `ALLOWED_KUBECTL_VERBS`, so it
  was flagged `kubectl_unsupported_verb:-n` and the entire batch was marked
  unsafe — *before* `live_action_executor.py`'s own (already-fixed) checks
  ever ran, since `check_command_safety()` is called first.
- `training_pipeline/action_reward.py` — `_is_mutating_command()` did a
  naive substring check for the literal text `"kubectl patch"`. With a flag
  between "kubectl" and "patch", that substring never appears, so
  `has_mutating_command` was `false` for a command that plainly mutates.
- `training_pipeline/command_normalizer.py` — `normalize_command()` dispatched
  on a fixed `parts[:2] == ["kubectl", "patch"]` prefix (same failure mode),
  and two of its own helpers (`_deployment`, `_pod_owner_hint`) had the
  identical fixed-index assumption one level deeper, for the case where a
  flag appears *between* the verb and the resource kind.

Four independently-written copies of the same parsing assumption, all broken
by the same command shape, is a sign the logic itself — not any one call
site — needed to be owned in one place. Extracted the fix into a new shared
module, `training_pipeline/kubectl_command_shape.py`
(`positional_args`/`positional_indices`/`resource_target`/`namespace_flag`/
`program_verb`), and rewired all four files to use it instead of their own
copies:
- `digital_twin_runtime/live_action_executor.py`: now imports from the shared
  module instead of defining its own `_namespace`/`_positional_args`/
  `_resource_target`.
- `command_safety.py`: `_kubectl_safety`/`_helm_safety` use
  `positional_args`/`resource_target`.
- `action_reward.py`: `_is_mutating_command` now parses the command (verb +,
  for `delete`, resource kind) instead of substring-matching; `mongosh`/
  `helm rollback` stay simple checks since mongosh has no comparable verb
  position and no flag-before-verb helm command has been observed.
- `command_normalizer.py`: dispatch now locates the verb's actual index via
  `positional_indices` before slicing, and `_deployment`/`_pod_owner_hint`
  flag-strip their own slice before checking for "deployment"/"pod" as the
  first token.

Verified two ways:
1. Unit-level, all 4 files, against every real flag-ordering variant seen in
   production so far (`-n` before/after the verb; before/after the resource
   name; combined with `rollout status kind/name`).
2. Empirically, against **every real Action command GPT-5.2 has produced
   this run so far** (3,372 commands pulled directly from
   `openai_calls_worker_{0,1}.jsonl`): zero commands now trigger
   `kubectl_unsupported_verb` (previously nonzero for any `-n`-before-verb
   command); of 586 real `patch`/`scale`/`delete` commands, only 6 resolve to
   an empty resource name, and all 6 are pre-existing truncation artifacts
   from before the `--downstream_action_tokens` budget increase (the frozen
   agent's response was cut off before it ever wrote a resource name at
   all) — not a new parsing gap. The exact command that surfaced this bug
   now round-trips correctly end-to-end:
   `normalize_command` → `fix_infra_scheduling`/`social-graph-service`,
   `check_command_safety` → `safe: true`, `_is_mutating_command` → `true`.
   `audit_pipeline_contract.py` (50/50) and `audit_hf_exact_token_sampler.py`
   re-run clean.

Stopped and relaunched a third time, this time from checkpoint 16 (it landed
while this fix was in progress; no progress lost), tags gained
`command-shape-parser-consolidation`.

## Addendum: trainable policy instructions still truncating at 224 tokens

The update-15 and update-16 trajectory walkthroughs made the pattern
impossible to miss: essentially every trainable RCA/Action instruction ended
mid-sentence, including the freshly-lengthened 3-part Action template (e.g.
attempt 0 of the update-16 trajectory cut off mid-JSON-quote before ever
reaching parts 2–3 of the requested structure). Per the user's direct
request, raised `--max_new_tokens` from 224 to **800**. This is a real jump
(previous history: 96 → 224 → 800) — flagged the expected cost directly:
longer generation time per rollout and higher GPU memory during the
backward pass, since gradients flow over every completion token, not just
GPU decode time. Stopped and relaunched a fourth time from checkpoint 16 (no
progress lost — one stuck `aiops-twin-*` namespace from the SIGTERM found and
cleaned up, same pattern as prior stops), tags gained `policy-tokens-800`.

## Addendum: `unique-id-service` scenarios have never gotten a live RCA twin check — found and fixed without stopping training

While reviewing update 17 (asked "aren't there supposed to be two scenarios
per update" — yes, confirmed: one per GPU worker; update 17's other scenario
was `unique-id-service`), noticed all 4 of its trajectories had
`action_stage_invoked: false` despite 2 of them reporting RCA `success:
true`. Investigated without touching the live training process (explicit
request) — everything below is read/probe-only against a temporary,
self-cleaning Twin namespace, never the training run's own.

**Scope, checked first:** every `unique-id-service` attempt in this run's
own logs, across every task-phase variant (analysis/detection/localization/
mitigation) and every iteration — `rca_policy_samples.jsonl` shows **100%**
of them (dozens of attempts) with `predicted_fault_injection_checked:
false`, including *first-attempt* guesses that were already the correct
mechanism. Not a flaky/intermittent issue — fully reproducible, and it
affects roughly 23 single-fault dataset scenarios directly (plus more in
`multifault` combinations), each contributing zero real RCA twin-reward
signal for the entire run so far, and never reaching the Action phase since
`rca_twin_verified` gates it.

**Root cause, confirmed by direct reproduction** (temporary namespace,
deleted immediately after): `SparseLiveTwinVerifier._require_observable_channels()`
correctly fails closed with `reason: "twin_telemetry_incomplete:traces"`
when the clean, pre-fault-injection baseline probe collects zero trace
edges — this check exists specifically so a missing telemetry channel is
never silently scored as a real result. The actual defect is one layer
down, in `_workload()`'s targeted-request generation: it discovers the real
API handler for the causal path to `unique-id-service`
(`/wrk2-api/post/compose`) and auto-generates a POST body by guessing a
placeholder value for every captured form field name — literally
`"media_ids=1&media_types=2&post_type=3&text=4&user_id=5&username=6"`.
compose-post validates these fields (a real user_id/username shape, a JSON
array for media_ids), rejects every one of the placeholder requests, and the
probe's own captured response body confirms it — a login-page redirect, not
a successful post. Since the request never actually completes,
`unique-id-service` is never called and never receives a trace span, so
`collected_trace_edges: 0` regardless of whether a fault is injected.

The critical detail: `AIOpsLab/aiopslab-applications/socialNetwork/wrk2/
scripts/social-network/compose-post.lua` — a real, correctly-parameterized
workload script the DeathStarBench benchmark ships for exactly this
endpoint — already existed and was already being discovered by `_workload()`'s
second scoring mechanism, but the auto-generated placeholder script was
checked *first* and always took priority whenever any matching raw handler
was found, so the correct pre-built script was never actually used for this
service.

**Fix:** reordered `_workload()` in `digital_twin_runtime/sparse_live_verifier.py`
to score and prefer an existing purpose-built `wrk2/**/*.lua` script first,
falling back to the handler-derived placeholder generator only when no
pre-built script references the target service/path. Verified with a second
direct reproduction (temporary namespace, deleted immediately after): same
scenario, same fault — `predicted_fault_injection_checked: true`,
`rca_twin_verified: true`, `reproduction_score: 0.6449`,
`collected_trace_edges: 25` (was 0). `audit_pipeline_contract.py` re-run
clean (50/50) afterward. Not deployed to the live run per explicit
instruction not to interrupt training for this — takes effect on whatever
restart comes next.
