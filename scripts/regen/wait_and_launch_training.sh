#!/usr/bin/env bash
# Compatibility entrypoint: launch only an already rebuilt and qualified corpus.
# Regeneration/calibration are separate, explicit stages; see training_pipeline/OPERATIONS.md.
set -euo pipefail
: "${REPO:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${TRAINING_PY:=$REPO/.venv-training/bin/python}"
: "${FROZEN:?Set the corrected frozen dataset directory}"
: "${CALIBRATION:?Set the current reward_calibration.json path}"
: "${RUN_DIR:?Set a new output directory for this experiment}"
: "${TRAIN_IDS:=$FROZEN/train_ids.txt}"
: "${SOURCE_NAMESPACE:?Set the healthy reference namespace}"
: "${APPLICATION_SOURCE_ROOT:?Set the application source directory}"
cd "$REPO"
if [[ -e "$RUN_DIR" ]]; then
  echo 'Use a new run directory, or invoke the trainer directly for a compatible --resume.' >&2
  exit 1
fi
exec "$TRAINING_PY" -m training_pipeline.train_qwen_live_grpo \
  --processed_states "$FROZEN/processed_states" --dataset_manifest "$FROZEN/manifest.json" \
  --scenario_ids "$TRAIN_IDS" --reward_calibration "$CALIBRATION" \
  --source_namespace "$SOURCE_NAMESPACE" --application_source_root "$APPLICATION_SOURCE_ROOT" \
  --output_dir "$RUN_DIR" --twin_mode live --temperature 1 --top_p 1 \
  --rca_max_iterations 7 --action_max_iterations 7 --retain_twin_artifacts "$@"
