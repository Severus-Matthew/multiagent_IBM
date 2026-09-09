# multiagent_IBM

The canonical live training implementation is on **`training-pipeline-v0`**:
`python -m training_pipeline.train_qwen_live_grpo`.

The pipeline trains separate RCA and Action prompt adapters on a frozen base model.
The downstream RCA and Action agents are fixed. Public, compressed incident
telemetry determines a live Twin's service scope before RCA generation. Each
hypothesis is injected into a clean instance of that frozen scope, compared with
the incident and a clean control, and admitted only under current matched live
calibration. Both stages allow seven attempts. Every Action retry is qualified
from a fresh fault state; recovery requires symptom clearance and satisfaction of
the incident's SLA definition.

Successful, calibrated repairs export a portable plan. The explicit
`digital_twin_runtime.repair_transfer` CLI binds that plan to a real Kubernetes
context and namespace, verifies incident preconditions, applies the repair,
checks recovery, and attempts a concurrency-safe rollback if verification fails.
Training itself never applies repairs to the source application.

See [training_pipeline/OPERATIONS.md](training_pipeline/OPERATIONS.md) for migration,
rebuilding the corpus, calibration, launch commands, repair application, resource
measurement, and validation limits. `configs/training_pipeline.json` documents
contracts; executable settings come from the CLI. The older `agents/`, `training/`,
and standalone debug paths are retained for historical experiments and are not
the canonical training path.

After initializing submodules, run `python scripts/regen/apply_aiopslab_patches.py`
to apply the bundled generator fix. The AIOpsLab gitlink remains pinned to its
existing published commit; the script preserves unrelated local edits.
