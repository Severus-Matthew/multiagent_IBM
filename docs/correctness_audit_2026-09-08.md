# Correctness audit and draft corrections — 8 September 2026

Audited source: `training-pipeline-v0` at `e885485c38023605e54d4315bff07b100481ef81`.

**Verdict: the intended architecture is recognizable, but this snapshot is not qualified for a reportable end-to-end experiment.** The changes in this branch are a draft. They have been inspected as source, but the execution environment became unavailable before the new regression suite could run. No live deployment, running training process, dataset rebuild, or GPU update was performed by this audit.

The review covers the active `training_pipeline/`, `digital_twin_runtime/`, and `state_abstraction_full/` code. The older `agents/`, `digital_twin/`, and `training/grpo_trainer.py` are not interchangeable with that implementation.

## Corrections proposed in this branch

| Defect | Consequence | Draft correction |
| --- | --- | --- |
| Trace aggregation treats mean, p50, p95, p99, and maximum as five duration samples per file | Wrong latency statistics even with one file; incorrect weighting across unequal file sizes | Combine unique raw spans before calculating statistics; resolve parents across files; reject conflicting observations of one span |
| Service specs are stored as `system[svc]["services"]` but dropped by compression | The RCA agent loses actual selector and target-port evidence | Carry the routing fields into compressed and bounded state for every service |
| Ineligible trajectories receive zero returns but still participate in GRPO normalization and replay | Missing observations can shift valid trajectories' advantages and receive optimizer/KL updates | Require explicit reward eligibility before both normalization and replay; drop groups with fewer than two participating trajectories for either role |
| Empty RCA optimizer batches raise after eligibility filtering | A fully unscorable batch can crash training | Save the advanced dataset cursor, record a skipped batch, and continue without an adapter update |
| Action namespace falls back to private fault context | A missing live session can expose or direct prompts toward the source namespace | Obtain a live namespace from the verifier; fail if a live verifier cannot supply one; use a constant namespace for offline diagnostics |
| Safety report does not inspect namespace FQDNs and truncates key inspection at 1,000 list entries | Some payloads are reported safe despite containing fields the sanitizer is supposed to remove | Detect unredacted service FQDN namespaces and scan every list entry |
| Reward-route logging independently guesses the route | Logged route can disagree with the actual reward route | Log the route selected by the reward boundary |
| Existing calibration has no link to corrected trace aggregation | Old thresholds could silently authorize a changed measurement pipeline | Reject old control records until requalified with `unique_raw_spans_v1` |

The calibrated-mechanism registry remains intact as historical information. **This draft intentionally prevents its old controls from authorizing production reward.** Do not add the new aggregation tag to old evidence or use the uncalibrated override to label a scientific run qualified. Collect and inspect new controls first.

The trace merger now deduplicates by trace ID plus span ID. This handles overlapping exports and allows a parent and child to occur in different files. Conflicting duplicate observations stop abstraction rather than selecting one arbitrarily. The compatibility `merge_edges` helper permits one summary and rejects multiple summaries because pooled quantiles cannot be reconstructed from summary quantiles alone. Callers must supply raw captures through `parse_traces`.

An arithmetic counterexample demonstrates the original defect. For 99 spans of 10 microseconds and one span of 1,000 microseconds, the true p95 is 10 microseconds and the mean is 19.9 microseconds. The old single-file merge reports approximately 803.98 microseconds for p95 and 211.96 for the mean. These values are calculated from the original algorithm; they are not measurements of a live incident.

The eligibility correction enforces the existing reward function's admission decision. It does not redefine penalties, clipping, KL, adapter ownership, or trajectory weighting. The eligible population is now explicit. Distinguishing unscorable infrastructure failures from invalid policy outputs deserves a separate reward-contract audit: an invalid command and a failed collector should not automatically share a learning treatment.

## Important findings that remain open

**1. Sampling probabilities and replay probabilities differ.**

`HFExactTokenPolicySampler.generate` uses temperature and nucleus sampling, then stores raw-model log probabilities from replay. The production CLI defaults are temperature 0.8 and top-p 0.95. The learner also uses raw logits. Consequently, exact token identity and a replay ratio of one do not establish that the denominator is the distribution that generated the tokens.

Let `p_old` be the raw softmax distribution and `q_old` the actual temperature/top-p sampling distribution. In general `q_old != p_old`. The implemented ratio `p_new/p_old` over samples drawn from `q_old` is not an exact on-policy importance ratio for either distribution. Top-k or other inherited generation processors must also be included in this review.

Choose and test an explicit policy contract: sample the unmodified distribution and use it for all likelihoods, or consistently define the transformed distribution and its support/correction. A naive division by a top-p behavior probability does not recover probability mass outside its truncated support. Do not silently change the objective while making this repair.

The [DeepSeekMath GRPO definition](https://arxiv.org/html/2402.03300v3) samples from the old policy used in the objective. [TRL's documentation](https://huggingface.co/docs/trl/grpo_trainer) distinguishes rollout/learner mismatch and importance correction. This finding is an inference from those definitions and the inspected local implementation; no new GPU experiment was run.

**2. Clean, injected, and recovered telemetry are not separated by observation windows.**

`targeted_telemetry._jaeger_rows` queries a five-minute lookback without a phase start/end bound. The same Twin survives the clean, injected, and recovered workloads. Therefore the later collection can include earlier phases. Pod logs similarly use a tail count without a phase start time. This can hide injected failures behind clean traffic, or make a recovered system retain historical errors.

Add explicit workload-phase timestamps or correlation IDs; filter the collected spans and logs to the appropriate observation window, account for asynchronous export completion, and record the window in provenance. Verify this with a deliberately slow collector and phase-tagged requests. A fresh output directory alone does not separate the data queried from the backend.

**3. Failed trace collection can masquerade as observed silence.**

The Jaeger collector silently continues after a failed service query. The post-injection verifier permits zero trace edges when collection succeeded in the earlier clean phase. An earlier successful request does not prove the later request succeeded. Propagate query failures and coverage explicitly; distinguish a successful, empty current response from an unavailable observation channel.

This is separate from the already-fixed empty after-state resolution check.

**4. RCA consistency is not proof of a unique root cause.**

The comparator uses coarse structural and symptom overlap within the hypothesis-selected scope. Different mechanisms can yield similar features. Symptoms outside that scope are reported but do not by themselves prevent verification. A partial multifault hypothesis can therefore remain a concern even if its selected subgraph reproduces well.

Evaluate matched positive, wrong-service, wrong-mechanism, missing-root, extra-root, and multifault controls using the corrected collector and abstraction. Freeze the decision rule on calibration data that is separate from final evaluation. Keep exact-label diagnostics separate from optimized reward. Do not claim that a threshold alone proves exact root-cause identity.

**5. Sparse scope is enforced in source, but actual resource savings are unverified.**

The inspected source defaults to one downstream support hop and rejects manifest discovery that changes the planner-selected service set. That is a meaningful implementation repair. It does not prove that the running process imported the repair, that every selected dependency is necessary, or that the live deployment achieves the predicted reduction.

For every live qualification case, compare planned services, rendered workloads, and actual running workloads. Record pod counts, requested resources, measured CPU/memory, infrastructure overhead, and total lifecycle time against an equivalent full deployment. Service-count reduction is a topology-size metric, not a CPU/memory saving measurement.

Do not force an arbitrary percentage reduction at the cost of reproduction fidelity. A valid workload path may need services that are not themselves faulty.

**6. Recovery and SLA gates need renewed live qualification.**

The current source requires symptom clearance, a clean post-action workload, and a violated-to-healthy SLA transition. That improves on the older symptom-only proxy. However, the SLA calculation consumes the same affected trace statistics and phase-mixed observations discussed above. The current code also retains cumulative signals such as restarts; confirm that historical evidence is distinguished from current failure.

Use identical workloads and comparable windows before and after an action. An originally healthy SLA should be identified explicitly rather than counted as a violated-to-healthy transition. Separately test clean-control/no-op behavior, unsafe commands, partial recovery, and exact verifier-owned Chaos deletion.

**7. The full dataset and real-incident application path are not verified here.**

Committed ID lists and reports do not substitute for the actual frozen corpus, its content hashes, regeneration outputs, or split construction evidence. The service-routing and trace-statistic changes require new abstractions and a new frozen dataset version. Recheck observable prompt payloads, train/calibration/test separation, and near-duplicate capture lineage on that version.

The inspected training path ends at Twin verification. It does not establish a production application path with target binding, preconditions, rollback, and post-application verification. A successful Twin action must not be described as already applied to the real incident.

## What the source already supports

- Joint RCA-to-Action trajectories with separate trainable prompt policies and frozen downstream agents.
- Mechanism-level fault hypotheses and prediction-based live injection, without substituting hidden labels in the live validator.
- Public Twin-based stopping in the joint loop and seven-attempt defaults.
- One structured fault line per root; multifault output consists of multiple such lines, which differs from a strict one-line total response.
- Separate role returns, per-incident group-relative normalization, exact completion token storage, token-level clipping, and per-trajectory role decision weights summing to one.
- Private exact-label/root-count statistics retained as diagnostics in the end-to-end return.
- Sparse manifest scope equality enforcement and a live workload/recovery path.
- Default rejection of uncalibrated mechanisms and multifault thresholds, rather than silently treating all implemented adapters as scientifically calibrated.

These are source-level findings. They do not certify the running binary, complete leakage absence, scientific novelty relative to all prior work, or live experimental validity.

The `1/D_role` weighting implements the project's chosen average-over-decisions objective. It should not be described as a general proof of an unbiased multi-step policy gradient. Similarly, averaging completion-token losses has its own normalization semantics. The [TRL loss discussion](https://huggingface.co/docs/trl/grpo_trainer) explains why different length normalizations lead to different objectives; this draft preserves the existing choice.

## Validation and restart requirements

Before the workspace became unavailable, these existing audits ran on the unmodified source and returned PASS:

- `python -m training_pipeline.audit_grpo_objective`
- `python -m training_pipeline.audit_injectible_rca_contract`

The log-summary tools found no local rollout files; their empty outputs are not training-health evidence. Torch/GPU integration and live Kubernetes checks were not completed. The 19 new regression tests in this branch have **not been executed**.

Start review with:

```bash
python -m unittest discover -s tests -p test_audit_corrections.py -v
python -m training_pipeline.audit_grpo_objective
python -m training_pipeline.audit_injectible_rca_contract
python -m training_pipeline.audit_exact_token_grpo_replay
python -m training_pipeline.audit_factorized_grpo_learner
python -m training_pipeline.audit_streaming_grpo_optimizer
```

The last three require the project's Torch environment. After resolving sampling and phase isolation, rebuild the corpus, requalify live controls and thresholds, verify actual sparse deployments, and perform a short frozen-configuration training run before a long experiment.

To assess an existing run, retain its run manifest/configuration, exact source commit and dirty-tree state, trajectory and training-event logs, representative redacted policy payloads, and per-phase Twin/collector artifacts. W&B scalar curves alone cannot establish source version, sparse deployment membership, or absence of leakage. A process started before a patch does not automatically load it.

A continued checkpoint can remain useful for development, but a new scientific experiment should have an explicit new provenance boundary after these corrections. Existing results cannot be relabeled as having used the corrected pipeline.
