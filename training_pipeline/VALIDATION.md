# Validation record — 9 September 2026

Target branch: `training-pipeline-v0`. This change builds on the abstraction and
GRPO admission corrections already in PR #3. It does not update a running Python
process or rewrite existing training artifacts.

## Executed locally

- `python -m unittest discover -s tests -v`: **62 tests passed**. Coverage includes
  raw-span quantiles, routing evidence, public-state sanitization, split hashes,
  group admission, incident scope, phase isolation, failed-query handling,
  joint-fault controls, independent action retries, repair transfer, resource
  accounting, multifault identity, and raw sampling probabilities.
- Six executable audits passed: `audit_grpo_objective`,
  `audit_injectible_rca_contract`, `audit_exact_token_grpo_replay`,
  `audit_factorized_grpo_learner`, `audit_streaming_grpo_optimizer`, and
  `audit_hf_exact_token_sampler` (modules under `training_pipeline`).
- `audit_end_to_end_hf_exact_rollout` with a synthetic offline fixture passed its
  **offline admission** checks. Offline trajectories were excluded from production
  optimization; this result is not live reward qualification.
- An integrated mocked test exercises the actual sparse verifier methods through
  clean preparation before RCA, joint injection, recovery, portable plan export,
  and a fresh session for a subsequent action attempt.
- The bundled AIOpsLab patch was applied to a clean copy of the pinned published
  generator (`3538780622eb45cf3bd2f91973eab8ff1300f7cf`). Reapplication was
  idempotent. The patched generator and helper matched the tested working copy.
- Python compilation, modified regeneration shell syntax, and `git diff --check`
  passed. The parent repository's AIOpsLab gitlink remains unchanged.

Model tests used CPU PyTorch 2.14.0, Transformers 4.57.6, and PEFT 0.18.1.
The sampling regression uses a real tiny model and compares generation scores
with raw logits despite conflicting inherited generation settings. Kubernetes
operations in regression tests are mocked.

## Not established by these checks

No GPU training, corpus regeneration, live calibration, Kubernetes fault injection,
real repair, W&B run assessment, or measured live resource comparison was performed
in this environment. No production calibration threshold was fabricated.

Follow [OPERATIONS.md](OPERATIONS.md) to rebuild and freeze the corpus, collect
matched positive and negative controls, and launch a new qualified experiment.
Mechanisms or joint faults whose controls fail or cannot separate remain
ineligible. Live fidelity, resource savings, held-out RCA accuracy, and real repair
success require those measurements; local tests alone do not prove them.
