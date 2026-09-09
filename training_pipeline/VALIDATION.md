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

## Live lifecycle evidence

See the section appended below after the run on the merged checkout.

## Not established by these checks

No corpus was rebuilt, no calibration controls were collected, no GPU training
update ran on the corrected code, no W&B assessment and no real-incident repair
was performed. Rebuilding the corpus with the corrected abstraction, collecting
matched controls with `--workload_duration_seconds 150`, freezing a dataset with
disjoint train/calibration/test splits, and a short frozen-configuration training
run remain prerequisites for any reportable experiment. Incident scopes on this
corpus are close to the full application; resource-saving claims require the
matched measurement described in OPERATIONS.md.
