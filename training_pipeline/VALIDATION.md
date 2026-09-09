# Validation record — 9 September 2026

Target branch: `training-pipeline-v0`. The audit branch
`codex/correctness-audit-20260908` (`c1e7355`, PR #3) was integrated on the training
host and followed by `fdc113a`, which fixes defects found only on that host. The
8 September audit document is historical; this record supersedes its validation
section. No running process was updated in place: the pre-correction trainer was
stopped after its checkpoint 132 before the checkout changed (see OPERATIONS.md).

## Environment

Training host: 2x RTX PRO 6000 Blackwell, kind cluster `kind-kind`, Prometheus in
`observe` with a global `scrape_interval` of 1m, Jaeger per application namespace.
`.venv-training`: Python 3.12.13, torch 2.11.0+cu128, transformers 5.16.1,
peft 0.20.0. The audit branch was validated by its author on transformers 4.57.6
with Kubernetes and Prometheus mocked; those results did not establish
compatibility with this host.

## Failures discovered on the host, and their fixes

| Failure on this host | Evidence | Fix (`fdc113a`) |
|---|---|---|
| `HFExactTokenPolicySampler.generate` raised `ValueError: model_kwargs not used: ['use_model_defaults']` | `tests/test_raw_sampling_distribution.py` and `audit_hf_exact_token_sampler` failed on transformers 5.16.1 | Version-compatible pinning: the owning module's `generation_config` is replaced by the neutral raw-softmax configuration for the call and restored afterwards; the keyword is passed only where accepted. Scores equal raw logits for both adapters despite inherited `top_k`, `top_p`, `repetition_penalty` and `suppress_tokens`. |
| `prepare_scenario` raised `KeyError: 'metadata'` on every real record | 16/16 sampled train records failed before any Twin was created | `discover_sparse_manifest_plan` returns summaries; controller and Service references are now resolved to real objects (explicit `Service` kind) before the reference state and environment fingerprint are built. The mocked fixture that supplied full objects was corrected. |
| Phase-bounded Prometheus `rate()` returned no series | Live query over `test-social-network`: 0/27 pods at 35s and 61s lookback, 27/27 at 125s; scrape interval 1m | Scrape interval discovered from `/api/v1/status/config`; trainer preflight, verifier and collector refuse phases shorter than two scrapes plus 5s; per-pod `count_over_time` coverage recorded; lookback never widened beyond the phase. Launch with `--twin_workload_duration_seconds 150` on this cluster. |
| Incident scope raised `incident scope cannot cover all observable affected services` on every real record | Affected sets were dominated by log-error noise on datastores, container names, Chaos objects and volume names, which the request-path planner cannot reach | Alias-resolved attribution onto the deployable inventory; request-path targets planned with entry paths; off-graph symptomatic services kept directly with startup closure; trace coverage required only for services the incident's traces observed. 16/16 sampled records now plan a scope (25/27 and 19/24 services) whose policy state passes the public-input sanitizer. |
| Comparator zeroed every real comparison (`unexplained_incident_symptoms_outside_scope`) | Retained Twin captures from the legacy run scored 0 against their incidents because names like `unknown`, `hotel-reserv-*` never matched a service | Outside-scope rejection now applies to attributable application services only; unattributed names are reported separately. |
| Bundled AIOpsLab patch did not apply | Host submodule is `b56eda8` plus local edits, not the pinned `3538780` | Identity fix ported by hand into the generator; helper installed; apply script recognizes the port. The ported generator enumerates 1588 specs (1000 multifault) with unique ids. |

## Executed on the host after integration

- `python -m unittest discover -s tests`: **75 tests passed** (62 from the audit
  branch with corrected fixtures, 13 new in `tests/test_server_integration_fixes.py`).
- Audits passed: `audit_grpo_objective`, `audit_injectible_rca_contract`,
  `audit_exact_token_grpo_replay`, `audit_factorized_grpo_learner`,
  `audit_streaming_grpo_optimizer`, `audit_hf_exact_token_sampler` (CPU, real tiny
  HF/PEFT model on transformers 5.16.1).
- Corrected `state_abstraction_full/run_pipeline.py` executed on 8 retained Twin
  captures from the legacy run (clean, post-injection, post-remediation phases):
  all succeeded; service and edge counts and SLA verdicts matched the legacy
  abstraction; the compressed view now carries `abstraction_contract` and no
  `namespace`, `scenario_id` or `timestamp`.
- Read-only cluster checks: incident-scope planning on 16 real records (above);
  Jaeger accepts explicit `start`/`end`; `kubectl logs --timestamps` stamps parse
  at nanosecond precision.
- `scripts/regen/apply_aiopslab_patches.py` reports the ported generator;
  `git diff --check` clean.

## Live lifecycle evidence (merged checkout, this cluster)

Incident `gen_network_delay_hotel_res-detection-frontend-default` (HotelReservation,
`network_delay` on `frontend`), evaluator-only controls built from the private
label exactly as `collect_live_calibration` does, `require_reward_calibration=False`,
`reproduction_threshold=0.0`, 150s phase workloads at 10 req/s. Scripts and
JSONL logs are in `artifacts/host_validation_2026-09-09/`.

- **Run 1** failed closed in the clean phase: the phase window was 60.7s because
  `run_targeted_wrk` waited a fixed 60s regardless of the requested duration and
  reported the 150s workload as failed. Fixed in `ac21faf` (wait derived from the
  duration). Even in that run traces (7410 rows in-window), logs (18 pods) and
  system state were collected; only the metrics guard rejected the short window.
- **Run 2, scope**: all 19 deployable controllers deployed; the record's
  24-name inventory also lists 5 names without a controller (`jaeger-out`,
  `profile-db`, `recommendation-db`, `reservation-db`, `user-db`). The result
  reports `service_reduction_percent: 0.0`: **no deployment reduction** on this
  incident. Request-path targets on 15 services; trace-observable targets `geo`,
  `rate`; unattributed symptom names `profile-db`, `recommendation-db`,
  `reservation-db`, `user-db`, `unknown`. Scrape interval read as 60s;
  environment fingerprint recorded.
- **Run 2, clean phase** (192s): workload completed, 1507 requests, 0
  application failures, 13 trace edges, trace coverage of both trace-observable
  targets, all four channels observed, resource measurement valid (18 running
  pods, 2 CPU samples per pod inside the window, stable population).
- **Run 2, positive control** (162s): the delay manifested, injection checked,
  scope coverage complete, `reproduction_score` **0.4667** versus
  `clean_reproduction_score` **0.7000**, so `counterfactual_evidence_gate`
  is false and the true hypothesis is **not** verified.
- **Run 2, wrong-service control** (`network_delay` on `consul`, 358s including
  a fresh Twin): manifested, `reproduction_score` **0.7000** = clean.
- **Run 3, recovery machinery only** (the RCA evidence gate was bypassed on
  purpose and the evaluator supplied the exact repair; documented in the
  script): `kubectl delete networkchaos twin-network-delay-frontend` executed
  in the Twin, recovery ready, symptom reduction 1.0, SLA violated before
  (1 unhealthy service, 2 dependency violations) and healthy after,
  `sla_transition_restored` true, post-remediation resources valid, no repair
  plan exported (correct: the hypothesis was not calibrated). This shows the
  repair/recovery machinery works; it does **not** show that the automated
  RCA -> Action path succeeds, because no RCA hypothesis was admitted.

Per-channel breakdown of run 2 from the retained phase states: the historical
capture of this incident has **no degraded service and no failed trace edge**;
its only in-scope symptoms are 22 background log-error names. Every Twin phase
matches the healthy deployment tokens (overlap 1.0) and shares no log-error
names with the historical capture (overlap 0), which yields
0.35 / (0.35 + 0.15) = 0.70 for the clean and the wrong-service Twins. The
injected true fault adds failed edges `ROOT->frontend` and `frontend->frontend`,
activating the trace channel with zero overlap, hence
0.35 / (0.35 + 0.25 + 0.15) = 0.4667. The measurement lifecycle therefore works
on this cluster and the comparator does its arithmetic as designed, but the
result exposes two separate problems:

1. The **legacy corpus is not comparable with the corrected measurement
   contract**: its trace statistics come from the pre-audit aggregation the
   audit itself corrected, and its logs were captured over whole pod lifetimes
   rather than phase windows. Reprocessing the old files cannot recover
   observations or timestamps that were never recorded, so a rebuild of the
   existing raw captures is necessary but is not a demonstrated fix; admission
   and scoring have to be checked together on freshly recorded incidents.
2. A **scoring concern independent of the corpus**: matching healthy
   deployment tokens earns 0.35 of the weight with zero symptom overlap, so a
   Twin that reproduces nothing scores 0.70 whenever the log channel is
   incomparable. Whether structural-state agreement should count without any
   symptom agreement must be decided with fresh controls before any threshold
   is derived.

Positive/negative separation on this incident is inverted, so no threshold could
be qualified from it. Rejecting both hypotheses was the correct outcome.

## Compatible recorder pilot (9 September, later)

`dataset_generation/record_incident_capture.py` records incidents in the source
application under `MEASUREMENT_CONTRACT` (AIOpsLab problem injection with the
reviewed injector fixes; the Twin's payload/endpoint selection, rate, duration,
observation windows and collector). New captures live under
`/mnt/aiops-training/datasets/pilot-recorder-v1/` (the 622/49 dataset and the
legacy run are untouched). Evidence: `artifacts/host_validation_2026-09-09/pilot/`.

**Run 1** (`gen_network_delay_hotel_res-detection-frontend-default`): clean phase
154s, 1507 requests, 0 non-2xx; incident phase held to 165s while the upstream
delay injector made the frontend unreachable after 8 requests; recovered phase
1507 requests, 0 non-2xx; all four channels observed in every phase with valid
resource coverage; injection verified; source clean afterwards; abstraction
`raw_spans_public_state_metric_units_v1`, sanitizer-safe. Under the corrected
contract the fresh incident carries failed edges `ROOT->frontend` and
`frontend->frontend`, an SLA violation and **no** background log-error names;
offline it scores 1.0 against itself and 0.583 against its own clean phase.

Two defects surfaced before any Twin control could be trusted, both fixed and
tested (`fe5cb04`, `5760f19`): the wrk2 wait ended a phase early when the
workload died under the fault (windows are now held to the configured length
on both sides), and the noise-free incident scope pruned `geo`/`rate`, which
`search` calls on every request, so the Twin's clean baseline answered 883 of
1507 requests with non-2xx and positive/clean/wrong-service tied at 0.7917
(observed runtime call closure now kept; unhealthy clean baselines fail closed).

**Run-1 controls on the corrected verifier** (Twin of 19 controllers, consul
catalog complete, clean baseline 0 non-2xx):

| Control | Hypothesis | Score | Clean | Gate | dep | edges |
|---|---|---:|---:|---|---:|---:|
| positive | frontend / network_delay | **1.000** | 0.583 | pass | 1.0 | 1.0 |
| wrong service | consul / network_delay | 0.583 | 0.583 | reject | 1.0 | 0.0 |
| wrong mechanism | frontend / scale_replicas_zero | 0.376 | 0.583 | reject | 0.914 | 0.0 |

This is the first positive/negative separation on a consistently measured
capture. It is one incident of one mechanism; it qualifies no threshold
(calibration needs three matched incidents per key with the full control set).
The deployment-state floor (0.583 for any healthy-looking Twin) remains an open
scoring question; here it did not prevent separation because the incident's
evidence lives in the trace channel.

## Legacy run resumability (verified, not only preflighted)

`/mnt/aiops-training/legacy/wsxhzf27-live-grpo-stage1/` now carries its own copy
of `.venv-training` (base interpreter tarball included) and the frozen worktree
holds its own copy of `AIOpsLab/aiopslab-applications`, so the legacy path no
longer depends on the live checkout or on the live environment. A zero-update
resume (`--max_updates 0`, no W&B, scratch output directory) run from that
worktree with the snapshot venv loaded the base model on both GPUs, restored
`update-00000132.pt` strictly (adapters, both optimizers, data cursor 270,
policy `qwen-live-grpo-stage1-v1@u000132`, git commit e885485), created no Twin
namespace, and left every original run artifact byte-identical (checksums
recorded before and after). A CPU check of the checkpoint found 192 finite
adapter tensors per role with trained B matrices and Adam moments for every
parameter at steps 131/114. What this does not show: a full update on the
legacy code after the pause, which would require the OpenAI key and the cluster.

## Not established by these checks

No corpus was rebuilt, no calibration controls were collected, no GPU training
update ran on the corrected code, no W&B assessment and no real-incident repair
was performed. Rebuilding the corpus with the corrected abstraction, collecting
matched controls with `--workload_duration_seconds 150`, freezing a dataset with
disjoint train/calibration/test splits, and a short frozen-configuration training
run remain prerequisites for any reportable experiment. The live evidence above
qualifies no hypothesis and no threshold: on the one incident exercised, the true
hypothesis scored below the clean control, and the automated RCA -> Action path
was never exercised. Incident scopes on this corpus deploy the full deployable
application (0% reduction on the exercised incident); resource-saving claims
require the matched full-application measurement described in OPERATIONS.md.

Recommended order of the remaining work (corrected 9 September, later):

1. **Make incident recording measurement-compatible first.** The ported
   AIOpsLab generator fixes scenario identity only. Its captures still come
   from `get_traces(duration=5)` (Jaeger `lookback=` from a start time, no
   explicit end, no span deduplication rule), `kubectl logs --tail=1500`
   without `--timestamps`/`--since-time`, `get_metrics(duration=5)` through a
   port-forward, and only `approx_symptom_window_start/end` as provenance. The
   corrected Twin measures each phase inside an explicit `ObservationWindow`
   (Jaeger `start`/`end`, log lines filtered by timestamp, Prometheus rates
   evaluated at phase end with scrape coverage) and records it in
   `collection_metadata.json`. Fresh incidents recorded by the unmodified
   generator would reproduce the mismatch seen above. A recorder that injects
   the AIOpsLab fault into the source application but collects with
   `collect_targeted_telemetry` around a `run_targeted_wrk` phase (same rate,
   duration, payloads and contract) is required before the pilot set.
2. **Size the pilot set for calibration, not just diagnosis.** One incident per
   mechanism diagnoses separation; qualification needs at least
   `MIN_MATCHED_INCIDENTS = 3` incidents per calibration key (application plus
   the sorted mechanism/variant list), each with the required controls
   (`positive`, `no_fault`, `wrong_service`, `wrong_mechanism` for every other
   implemented mechanism, `extra_root`, plus `wrong_variant` for scale/delay/loss
   and `missing_root` for joint faults). With 14 implemented mechanisms that is
   roughly 17-19 Twin lifecycles per single-fault incident; at 150s phases a
   lifecycle is about 8-10 minutes, so budget about 8 hours per key on one
   worker, or lower the Prometheus scrape interval to 15s first (40s phases).
3. **Exercise the live agent path only with a live harness.**
   `audit_end_to_end_hf_exact_rollout` constructs `BehavioralTwinVerifier` and
   cannot show live RCA -> Action success.
   `training_pipeline.audit_sparse_live_joint_rollout` drives the real joint
   RCA -> Action loop on the sparse live Twin but with scripted policies and a
   scripted solver (an interface test; it now accepts `--reward_calibration`,
   `--allow_uncalibrated_live_reward`, `--twin_workload_duration_seconds`).
   The actual LLM agent path is the corrected trainer itself, run with
   `--reward_calibration` once controls separate.
4. A short GPU training/checkpoint test on the corrected trainer into a new run
   directory and W&B run.

The learning signal, not the machinery, is the blocker. Continued W&B logging
into `wsxhzf27` was not exercised by the restore test (W&B was disabled); the
run id and its resume arguments are preserved.
