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
git submodule update --init --recursive
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
  --output_dir /path/to/controls-v2
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
  --retain_twin_artifacts --wandb --wandb_project aiops-rl
```

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
