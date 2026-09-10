#!/usr/bin/env bash
# Rebuild every selected capture; never mix corrected and legacy abstractions.
set -euo pipefail
: "${REPO:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${PY:=$REPO/.venv-aiops312/bin/python}"
: "${RAW_TELEMETRY_ROOT:?Set the complete raw telemetry root, including all selected incidents}"
: "${PROCESSED:?Set a new processed-state output directory}"
: "${TRAIN_IDS:?Set grouped train membership}"
: "${CALIBRATION_IDS:?Set disjoint grouped calibration membership}"
: "${TEST_IDS:?Set disjoint grouped test membership}"
: "${FROZEN:?Set a new frozen dataset directory}"
: "${VERSION:?Set a new dataset version name}"
cd "$REPO"
if [[ -e "$PROCESSED" || -e "$FROZEN" ]]; then
  echo 'Use new output directories: stale abstractions must not be resumed.' >&2
  exit 1
fi
"$PY" state_abstraction_full/run_batch.py --telemetry_dir "$RAW_TELEMETRY_ROOT" \
  --output_base "$PROCESSED" --workers 8 --skip_simulator
"$PY" -m training_pipeline.audit_dataset_live_admission --processed_states "$PROCESSED" \
  --train "$TRAIN_IDS" --test "$TEST_IDS" --output "$PROCESSED/admission.json"
correction_args=()
if [[ -n "${LABEL_CORRECTIONS:-}" ]]; then
  correction_args=(--label_corrections "$LABEL_CORRECTIONS")
fi
"$PY" -m training_pipeline.freeze_dataset_version --processed_states "$PROCESSED" \
  --train_ids "$TRAIN_IDS" --calibration_ids "$CALIBRATION_IDS" --test_ids "$TEST_IDS" \
  --admission_report "$PROCESSED/admission.json" --output_dir "$FROZEN" --version "$VERSION" \
  "${correction_args[@]}"
echo 'Frozen dataset validated. Collect matched calibration controls before launching training.'
