# Live Twin validation status — 2026-09-03

## Threshold controls

Four matched, lifecycle-complete positive triplets were run for the verified
scheduling/scale mechanisms. Positive scores were `0.5400`, `0.6104`, `0.8417`,
and `0.5400`; eight wrong-service/wrong-mechanism scores ranged from `0.2417` to
`0.4004`. The midpoint of the observed margin is `tau = 0.4702`.

Evidence: `/mnt/aiops-training/artifacts/live-threshold-controls-matched-v3/live_threshold_controls.json`.

This is not evidence that one global threshold transfers to every mechanism.
The balanced matrix below disproved that stronger claim.

## Adapter audit

Two full-lifecycle probes passed for each of: Mongo auth missing, Mongo role
revocation, Mongo user deletion, container kill, container stop, network delay,
network loss, pod failure, pod kill, and wrong binary. These adapters are marked
`verified_live`. Application config misconfiguration remains
`pending_live_audit`: the faithful Hotel `geo` bundle exposes no mutable config
endpoint, and the generic adapter fails closed.

Evidence:

- `/mnt/aiops-training/artifacts/mongo-adapter-repair-validation-20260903/pending_adapter_audit.json`
- `/mnt/aiops-training/artifacts/nonmongo-adapter-validation-20260903/pending_adapter_audit.json`

## Balanced 56-case matrix

All 56 selected cases completed and all Twin namespaces cleaned. 28 passed all
gates. Of 28 failures, 21 completed the live lifecycle but scored below `0.4702`,
five lacked required trace evidence, and two failed baseline stabilization.

Evidence: `/mnt/aiops-training/artifacts/balanced-live-matrix-56-20260903-v1/summary.json`
and its sibling `results.jsonl`.

Conclusion: adapter operability is substantially improved, but the current
historical records do not support one cross-mechanism reward threshold.
Mechanism-class controls and regenerated source captures are required before
these records are optimizer-safe.

## Dataset admission

Superseded by `docs/dataset_admission_2026-09-03.md`. Under the previous
contract (source-faithful capture, collected trace edges, verified-live
adapter) 248 train and 12 test records were admissible. The rejections were
traced to target-blind upstream injectors, header-only Jaeger exports, and one
pending adapter; with channel-aware comparison and evidence-based admission the
strict count is 355/24, and 454/29 with the opt-in label corrections. 117 train
and 12 test records must be re-recorded.

The RCA and Action retry defaults are seven attempts.
