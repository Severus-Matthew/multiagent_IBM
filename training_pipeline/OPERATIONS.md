# Corrected live training and repair workflow

These changes target `training-pipeline-v0`. A running Python process keeps its
already imported code. Updating the checkout does not repair earlier trajectories,
replace old abstractions, qualify thresholds, or change an active training process.
Use a new dataset version and experiment for the changed measurement and sampling
contracts. Resume is supported only when dataset, selected IDs, calibration, and
sampling contract agree with the checkpoint's run manifest. Existing artifacts
are not rewritten by the new launcher.

## Implemented contracts

| Boundary | Current behavior |
|---|---|
| Scenario identity | The pinned AIOpsLab generator, after applying the bundled dependency patch, includes all ordered subfault specifications and variant parameters in multifault IDs; conflicting IDs fail before execution. Existing colliding captures must be regenerated. |
| Abstraction | Quantiles come from deduplicated raw spans, including cross-file parent reconstruction. Service selectors/ports survive compression and bounded projection. CPU rates have a cores unit; replica metrics are aggregated at common timestamps and counter resets are computed within pod series. Unknown rates are not presented as request counts. |
| Public input | The standalone compressed file is sanitized. Scenario IDs, timestamps, namespaces and private fault labels belong to private provenance, not policy input. Healthy reference capacity/routing and incident deviations remain observable inputs. |
| Incident Twin | Scope includes every observably affected service plus request paths and startup dependencies, and is frozen before the first RCA prompt. Rendering preserves reference replica counts. Every RCA/action reset uses the same frozen manifests. A dependency closure may legitimately require the full application; reduction is not forced. |
| RCA output | One physical line: `service::fault_type::mechanism[::variant]`, with multiple roots separated by ` | `. Historical newline records remain parseable. |
| Measurements | Workloads define nonoverlapping phase windows. Jaeger queries use explicit start/end times, Prometheus queries use the phase end time and window, and logs are filtered by timestamps. Any failed required query makes a trajectory unscorable. Successful empty faulted traces are distinguishable from a failed query. Clean/recovered workloads must reach all affected services in observed traces. |
| Verifier | A healthy/healthy structural match earns zero reproduction. Original symptoms outside the Twin scope reject the comparison. Faulted evidence must improve over a clean control, all injected components must remain manifested simultaneously, and calibrated thresholds govern acceptance. |
| Action | Each retry resets and requalifies the predicted fault state. All observed symptoms must clear. SLA satisfaction is required; an actual violated-to-healthy transition is reported separately, so an already healthy SLA is never claimed as restored. |
| GRPO | Raw softmax sampling (`temperature=1`, `top_p=1`, `top_k=0`) matches old/new/reference likelihood replay. Ineligible trajectories are excluded before normalization and KL replay. Exact tokens, token PPO clipping, sampled KL, per-role decision weighting, adapter isolation, and atomic update publication are retained. |
| Splits | Explicit train IDs and a verified frozen manifest are mandatory. Train/calibration/test sets are disjoint; identical public captures and service/mechanism variants cannot cross sides. Calibration evidence is bound to its frozen dataset. |
| Real repair | A calibrated, independently successful action exports a plan. Application requires an explicit real context/namespace and `--execute`. Ordinary resource names stay within the qualified scope. Chaos deletion additionally needs the exact real name/UID binding. Atomic UID/resourceVersion preconditions prevent unnoticed replacement; failed postchecks trigger a rollback attempt that refuses concurrent changes. |

This is trajectory-level group-relative policy optimization: complete incident
trajectories share a group, while later decision prompts can differ with history.
It is not a claim that all retry completions were sampled from one identical prompt.

## Rebuild and freeze

Run commands from the repository root, in the existing AIOps/training environment.
Initialize the pinned submodules and apply the reviewed local dependency patch first.
The patch and identity helper are versioned in this repository; no publication to
the separate AIOpsLab remote is required:

```bash
git submodule update --init --recursive   # fresh clones only; never on the training host (see below)
python scripts/regen/apply_aiopslab_patches.py
python state_abstraction_full/run_batch.py \
  --telemetry_dir /path/to/all/raw/telemetry_outputs \
  --output_base /path/to/new/processed_states --workers 8 --skip_simulator
```

Use a fresh output directory. Do not use `--resume` to rebuild obsolete compressed
files. Retain private full states for evaluator labels; agents consume the public
compressed views. Historical captures that lost raw spans, Service evidence, or
collided under the same scenario ID need recollection, not invented reconstruction.

Create **three** grouped splits. `create_labeled_split` uses `--val_size` for the
calibration carve-out and writes `train_N.txt`, `val_N.txt`, `test_N.txt`. Choose
sizes that respect complete service/mechanism groups; exact requested sizes can be
impossible. Keep the test split untouched during calibration, threshold tuning and
checkpoint selection. Apply any label corrections before freezing.

```bash
python -m training_pipeline.create_labeled_split \
  --processed_states /path/to/new/processed_states --output_dir /path/to/new/splits \
  --test_size TEST_COUNT --val_size CALIBRATION_COUNT
python -m training_pipeline.audit_dataset_live_admission \
  --processed_states /path/to/new/processed_states \
  --train /path/to/new/splits/train_N.txt --test /path/to/new/splits/test_N.txt \
  --output /path/to/admission.json
python -m training_pipeline.freeze_dataset_version \
  --processed_states /path/to/new/processed_states \
  --train_ids /path/to/new/splits/train_N.txt \
  --calibration_ids /path/to/new/splits/val_N.txt \
  --test_ids /path/to/new/splits/test_N.txt \
  --admission_report /path/to/admission.json \
  --output_dir /path/to/frozen-v2 --version frozen-v2
```

`TEST_COUNT`, `CALIBRATION_COUNT`, `N`, and paths are placeholders. The freeze
validator checks content hashes, public-input safety, grouping and membership;
an old compressed artifact cannot be admitted merely by copying it into a new directory.

## Collect actual live controls

The source namespace must contain a healthy reference application with the intended
replica capacity, images and configuration. The collector uses **only calibration
IDs** and creates isolated Twins; no optimizer runs. It tests positives, clean
controls, wrong services/mechanisms/variants, extra roots, and missing roots for joint faults.

```bash
python -m training_pipeline.collect_live_calibration \
  --processed_states /path/to/frozen-v2/processed_states \
  --dataset_manifest /path/to/frozen-v2/manifest.json \
  --scenario_ids /path/to/frozen-v2/calibration_ids.txt \
  --source_namespace HEALTHY_REFERENCE_NAMESPACE \
  --application_source_root /path/to/application/source \
  --output_dir /path/to/controls-v2 \
  --workload_duration_seconds 150
```

There is no hardcoded two-mechanism exemption. An application/mechanism/variant or
joint entry needs at least three matched incidents, complete lifecycle telemetry,
and strict positive/negative score separation. Every measured negative for an
application constrains its thresholds, including cross-mechanism negatives.
Failed controls remain failures. If scores do not separate, that entry remains
ineligible: further observations or a better comparator are necessary.

The controls bind reference controller/Service specifications, ConfigMap contents,
Lua payloads, workload settings and SLA definition. Changing these requires new
qualification. The manifest records the raw evidence and recomputes its thresholds
when loaded. Synthetic regression fixtures are never production calibration.

## Launch a new experiment

Select only training IDs whose mechanisms/variants have qualified controls. The
trainer checks eligibility before model loading. Preserve the settings relevant to
your hardware from your current launcher; the essential new arguments are:

```bash
python -m training_pipeline.train_qwen_live_grpo \
  --processed_states /path/to/frozen-v2/processed_states \
  --dataset_manifest /path/to/frozen-v2/manifest.json \
  --scenario_ids /path/to/qualified_train_ids.txt \
  --reward_calibration /path/to/controls-v2/reward_calibration.json \
  --source_namespace HEALTHY_REFERENCE_NAMESPACE \
  --application_source_root /path/to/application/source \
  --output_dir /path/to/new-run --twin_mode live \
  --temperature 1 --top_p 1 --rca_max_iterations 7 --action_max_iterations 7 \
  --twin_workload_duration_seconds 150 \
  --retain_twin_artifacts --wandb --wandb_project aiops-rl
```

`--twin_workload_duration_seconds` (and `--twin_workload_rate`) are part of the
calibration contract and must match the values used by `collect_live_calibration`;
the trainer refuses phases shorter than two Prometheus scrapes before loading the
model (150s on a 1m scrape cadence).

The downstream provider/model remains independently configurable. Production
training never edits frozen labels. `--allow_uncalibrated_live_reward` and offline
routing are exploratory modes and do not qualify experiments or repair plans.
The compatibility launcher `scripts/regen/wait_and_launch_training.sh` now requires
an already rebuilt, calibrated corpus through explicit environment variables.
It no longer silently mixes historical and corrected states or resumes the old run.

Monitor eligible trajectory counts, positive/negative separation, action invocation,
full trajectory recovery, nonzero role advantages, KL/clipping, and actual adapter
updates. High raw Twin acceptance alone does not establish RCA accuracy. Evaluate
scientific accuracy and generalization on the untouched test split after selection.

## Apply an independently verified repair

With `--retain_twin_artifacts`, successful calibrated actions write
`verified_repair_plan.json` inside their Twin artifact directory. The same plan is
included in the trajectory's action verifier result. Training never calls this CLI.

```bash
python -m digital_twin_runtime.repair_transfer \
  --plan /path/to/verified_repair_plan.json \
  --context REAL_CONTEXT --namespace REAL_INCIDENT_NAMESPACE \
  --application_source_root /path/to/application/source \
  --output_dir /path/to/prepared-real-repair
```

This prepares exact targets without changing application controllers. Review the
plan and use a new output directory with `--execute` to apply and verify it. For
Chaos deletion, add `--bindings bindings.json`; keys are exported `kind/name`
identities and values are `{"name":"real-name","uid":"real-uid","allow_delete":true}`.
The tool never guesses a real Chaos object's identity. Application uses the bound
context without changing the user's default context. Rollback is best effort and
explicitly reports conflicts or deletion/finalizer failures; no universal recovery
guarantee is claimed.

## Measure savings

Capture the full healthy reference using a verified incident's identical probes:

```bash
python -m digital_twin_runtime.capture_reference_resources \
  --plan /path/to/verified_repair_plan.json --context REFERENCE_CONTEXT \
  --namespace HEALTHY_REFERENCE_NAMESPACE \
  --application_source_root /path/to/application/source \
  --output_dir /path/to/full-reference-measurement
python -m digital_twin_runtime.compare_resources \
  --full_capture /path/to/full/collection_metadata.json \
  --twin_capture /path/to/same/incident/clean_baseline/collection_metadata.json \
  --output /path/to/resource-comparison.json
```

The comparison requires matching payloads, targets, endpoints, requested rates and
durations, healthy workloads, and unchanged observed pod populations. CPU/memory
are measured application resources. Observer, workload-generator, control-plane and
shared-node overhead are excluded and must be measured separately for total-cost
claims. Small service counts alone are not a resource-savings result.

## Local validation and remaining live gates

See [VALIDATION.md](VALIDATION.md) for the executed checks and their limits.

```bash
python -m unittest discover -s tests -v
python -m training_pipeline.audit_grpo_objective
python -m training_pipeline.audit_injectible_rca_contract
python -m training_pipeline.audit_exact_token_grpo_replay
python -m training_pipeline.audit_factorized_grpo_learner
python -m training_pipeline.audit_streaming_grpo_optimizer
python -m training_pipeline.audit_hf_exact_token_sampler
```

The distribution regression uses a real tiny HF/PEFT model and checks generated
scores against raw model logits even when the inherited model configuration asks
for top-k filtering, repetition penalties and suppressed tokens. Cluster calls in
unit tests are mocked. CPU tests do not establish Qwen GPU throughput, a live Twin's
fidelity, threshold separability, real resource savings, or real-cluster repair
success. Run the live control/recovery workflow and held-out evaluation on the
patched branch before treating a new experiment as reportable.

## Integration on the training host (9 September 2026)

The audit branch was integrated into `training-pipeline-v0` on the host that runs
training. Several assumptions in this document did not hold there; the
qualifications below take precedence on that host and are recorded in
[VALIDATION.md](VALIDATION.md).

### The pre-correction run is preserved, not migrated

W&B run `wsxhzf27` (`/mnt/aiops-training/runs/live-grpo-stage1`) was stopped at
bundle update 132 immediately after its checkpoint was written. The corrected
pipeline intentionally rejects that configuration (tempered sampling, load-time
label corrections, an unfrozen abstraction contract, uncalibrated live reward); no
bypass flag was added. The run resumes from a frozen copy of its own execution
environment instead:

- `/mnt/aiops-training/legacy/wsxhzf27-live-grpo-stage1/` holds the last complete
  checkpoint with both optimizer states (outside checkpoint pruning, hash-verified),
  run manifests, exact launch arguments, dataset/selection references, the
  `.venv-training` package list, and the AIOpsLab submodule state including the
  dirty nested `aiopslab-applications` submodule.
- `/home/ubuntu/multiagent_IBM-legacy-wsxhzf27` is a git worktree pinned to
  `e885485` (the last pre-audit commit) with `AIOpsLab` symlinked to the live
  checkout; `resume_legacy_wsxhzf27.sh` in the backup directory resumes the run
  from there into the same W&B run id. It refuses to start while `aiops-twin-*`
  namespaces exist. Do not upgrade `.venv-training` if that run must stay
  reproducible; snapshot it first.

Results from that run remain an exploratory, pre-correction experiment and cannot
be relabeled as having used the corrected pipeline.

### Stop the trainer before touching the checkout

`SparseLiveTwinVerifier._abstract` runs `state_abstraction_full/run_pipeline.py`
as a subprocess for every Twin phase, so a running trainer reads files from the
checkout even though its own Python modules are already imported. Stop the
trainer, confirm no `run_pipeline.py` children remain, and delete any leftover
`aiops-twin-*` namespaces (Kubernetes workloads do not stop with the parent)
before switching branches or editing the tree.

### AIOpsLab submodule on this host

The gitlink pins `3538780`, but the host checkout is `b56eda8` plus local edits
and an untracked corpus; `git submodule update --init --recursive` must **not** be
run there (it would reset that work). The scenario identity fix was ported by
hand into `AIOpsLab/gen_and_telmetry.py` (multifault specs pass through
`attach_scenario_identity`; `main()` enumerates `unique_scenarios`) and the helper
was installed as `AIOpsLab/scenario_identity.py`. `scripts/regen/apply_aiopslab_patches.py`
recognizes a ported generator and leaves it alone. The pre-port file is kept in
the legacy backup under `source/`.

Multifault ids now carry a `--<24 hex>` parameter hash. `dataset_generation/regenerate.py`
attaches the identity to every spec it runs, matches queue files written with the
legacy ids, and journals `legacy_problem_id` next to `problem_id` in the shard log;
new split/selection files must use the new ids.

### Prometheus scrape cadence and phase length

The cluster's Prometheus scrapes every **1m**. A phase-bounded `rate()` needs two
samples inside the phase, so the default 30s workload observed zero pods. The
trainer, the verifier and the collector now read the global `scrape_interval`
from the Prometheus API and refuse any phase shorter than
`2 * scrape_interval + 5s`; the lookback is never widened into an earlier phase.
On this cluster use `--twin_workload_duration_seconds 150` for training and
`--workload_duration_seconds 150` for `collect_live_calibration`, or lower the
Prometheus global `scrape_interval` (15s makes 40s phases sufficient). The scrape
interval is part of the reference environment fingerprint, so controls must be
recollected after changing it. Per-pod sample coverage is recorded in
`collection_metadata.json` (`resources.metric_sample_coverage`); pods with fewer
than two samples in the phase invalidate the CPU/memory measurement but do not
fail the reward channel.

Phase workloads are deduplicated by payload/endpoint, so several symptomatic
services on the same request path cost one workload run. A phase is a fixed
measurement interval: if the workload dies under the fault (connection refused,
target scaled to zero) the observation window is still held open until
``start + workload_duration_seconds`` before collection, in the Twin and in the
recorder alike, so phases stay comparable and always cover the scrape cadence.

### Incident scope on real captures

Captured symptom signatures name pods, containers (`hotel-reserv-geo-mongo`),
log artifacts (`unknown`), Chaos objects (`container-kill`, `delay`) and volumes
(`profile-db`) as well as services. The incident scope resolves those names onto
the deployable service inventory through the comparator aliases, plans entry
paths for symptomatic services on the request graph, keeps symptomatic datastores
and infrastructure directly with their startup closure, and requires trace
coverage only for services the incident's own traces observed. Names that resolve
to nothing are reported as `unattributed_symptom_names`; inventory names without a
controller in the healthy reference are `undeployable_inventory_names`. The
comparator rejects a comparison only when a *deployable* service with symptoms is
outside scope.

The incident scope also follows observed call edges forward from every kept
service to a fixpoint (`incident_runtime_closure_added`). Without it the first
fresh HotelReservation capture kept 15 controllers and pruned `geo`/`rate`,
which `search` calls on every request: the Twin's consul had no `srv-geo` or
`srv-rate`, 883 of 1507 clean-baseline requests failed, and positive, clean and
wrong-service Twins tied. `prepare_incident_twin` now refuses a clean baseline
with non-2xx responses or in-scope error edges (`clean_baseline_defects`), so a
broken reference fails closed instead of crediting its own defects.

Because every capture in the 622/49 corpus carries background log errors on most
datastores, incident scopes are large: on sampled records they contain every
deployable controller (25 of 27 inventory names for SocialNetwork, 19 of 24 for
HotelReservation; the remaining names have no controller), i.e. **0% deployment
reduction**. That is the corrected contract's expected outcome ("reduction is
not forced"), and it means resource-saving claims need the matched
full-versus-Twin measurement, not the service count.

Open scoring item: matching healthy deployment tokens earns 0.35 of the comparator
weight without any symptom overlap, so a Twin that reproduces nothing can score
0.70 when the log channel is incomparable (observed live on 9 September, see
VALIDATION.md). Decide with fresh controls whether structural agreement should
count without symptom agreement before deriving any threshold.

Open item: the corrected comparator counts restarts as a symptom only while a pod
is still unready, so incidents whose only surviving evidence is a restart counter
(recovered `container_kill` captures) have no request-path symptom and fail closed
as unverifiable. Decide whether restart evidence should re-enter the incident-side
signature before rebuilding the corpus.

### transformers 5

`.venv-training` runs transformers 5.16.1, which removed `use_model_defaults` and
merges `model.generation_config` into unset generation fields. The sampler pins the
raw-softmax contract by neutralizing the owning module's generation defaults for
the duration of each call and restores them afterwards;
`tests/test_server_integration_fixes.py` checks generated scores against raw
logits for both adapters on the installed stack.

### Recording measurement-compatible incidents

`MEASUREMENT_CONTRACT` applies to both sides of a comparison. The generator's
collection rules (Jaeger `lookback`, `kubectl logs --tail`, port-forwarded
`get_metrics(duration=5)`, approximate symptom window) are not the Twin's, and
the historical corpus was captured that way. Before recording a pilot or a new
corpus, record incidents with the same collector the Twin uses:
`collect_targeted_telemetry` around a `run_targeted_wrk` phase with the same
rate, duration and payloads, injecting the fault with the AIOpsLab problem
definition rather than the Twin adapter, and keeping the private label in
`state_abstraction.json` only. Calibration then needs at least three matched
incidents per calibration key with all required controls (see
`digital_twin_runtime/reward_calibration.py`); one incident per mechanism is a
diagnostic, not a qualification.

`dataset_generation/record_incident_capture.py` implements that recorder (run it
from the generation environment, `.venv-aiops312`, with no `aiops-twin-*`
namespaces present and a clean source application):

```bash
.venv-aiops312/bin/python dataset_generation/record_incident_capture.py \
  --scenario_ids ids.txt --output_dir /mnt/aiops-training/datasets/<new-version> \
  --workload_duration_seconds 150
```

Per scenario it records `clean`, `incident` and `recovered` phases under
`raw/<id>/`, each with the Twin's `collection_metadata.json` (window, channels,
scrape coverage) and a `phase.json` (workload contract, payload hashes), writes
the generator's private files (`spec.json`, `ground_truth.json`,
`fault_timing.json`, `injection_evidence.json`) into the scenario and incident
directories only, abstracts the incident phase into `processed_states/<id>/` and
the other phases into `processed_phases/<id>/`, and journals every scenario in
`log.jsonl` (accepted captures also under `accepted/`). It stops if the source
application does not return to a clean state after the problem's own recovery.
Multifault and application-level specs are not supported yet.

`python -m training_pipeline.pilot_score_separation --processed_states
<new-version>/processed_states --processed_phases <new-version>/processed_phases
--output_dir <report>` then runs the positive, one wrong-service and one
wrong-mechanism control per incident on the live verifier and tabulates the
scores with their per-channel overlaps; it also scores the capture's own clean
phase against its incident phase offline. Inspect that table before spending
the full calibration budget.

Accepted captures can then be assembled into one `processed_states` root,
audited (`audit_dataset_live_admission`) and frozen with disjoint
train/calibration/test lists (`freeze_dataset_version`, which validates the
manifest). The frozen manifest satisfies the trainer's strict gate; with
`--allow_uncalibrated_live_reward` the trainer can run a short exploratory
update on it (`--max_updates 1`, small groups) to exercise the automated
RCA -> Action path and checkpointing before controls are collected. That run
qualifies nothing and must use a new output directory.

### Lineage when reusing adapter weights

`update-00000132.pt` from the legacy run may initialize a new experiment only with
its lineage recorded in the new run manifest, and only after checking that the new
calibration and test incidents never appeared in that run's training selection
(`/mnt/aiops-training/legacy/wsxhzf27-live-grpo-stage1/dataset_refs/train_fully_live_reward_admissible.txt`,
557 ids). The trainer currently has no warm-start mode: `--resume` restores the
optimizer states and data cursor as well, so a warm start needs a dedicated flag
before it is used for a reportable experiment.
