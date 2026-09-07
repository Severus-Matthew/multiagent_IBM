#!/usr/bin/env bash
# Turn accepted regenerated captures into processed states, then compose a
# frozen dataset version in which regenerated records replace the historical
# ones, and audit admission on the result.
set -euo pipefail
REPO=/home/ubuntu/multiagent_IBM
RAW=${RAW:-/mnt/aiops-training/datasets/regenerated-v1/raw}
PROCESSED=${PROCESSED:-/mnt/aiops-training/datasets/regenerated-v1/processed_states}
VERSION=${VERSION:-full-622-49-v1}
FROZEN=${FROZEN:-/mnt/aiops-training/datasets/$VERSION}
PY=$REPO/.venv-aiops312/bin/python
Q=$REPO/configs/curriculum_622_49_v2/admission_v4
cd "$REPO"
# accepted ids only: the generator wrote accepted/<id>.json for captures whose
# injection was verified and whose trace export carried spans.
ls "$RAW/accepted" | sed 's/\.json$//' | sort > "$RAW/accepted_ids.txt"
echo "accepted captures: $(wc -l < "$RAW/accepted_ids.txt")"
cd state_abstraction_full
"$PY" run_batch.py --telemetry_dir "$RAW/generated/telemetry_outputs" --output_base "$PROCESSED" \
  --scenario_ids "$RAW/accepted_ids.txt" --workers 8 --skip_simulator --resume
cd "$REPO"
# Sanity check: how do the freshly regenerated captures classify on their own,
# before composing them with the historical corpus. Diagnostic only.
cat "$Q/regen_must_social.txt" "$Q/regen_must_hotel.txt" > "$RAW/regenerated_ids_checked.txt"
"$PY" -m training_pipeline.audit_dataset_live_admission \
  --processed_states "$PROCESSED" --train "$RAW/regenerated_ids_checked.txt" --test "$RAW/regenerated_ids_checked.txt" \
  --output "$RAW/admission_regenerated_only.json" || true
"$PY" -m training_pipeline.freeze_dataset_version \
  --processed_states AIOpsLab/processed_states --override_processed_states "$PROCESSED" \
  --train_ids configs/dataset_split_624_50/train_622.txt --test_ids configs/dataset_split_624_50/test_49.txt \
  --label_corrections "$Q/label_corrections.json" \
  --admission_report artifacts/dataset-live-admission-622-49-v5-corrected-strict.json \
  --output_dir "$FROZEN" --version "$VERSION"
# Final selection over the composed corpus (regenerated where available, else
# historical): strict evidence, then a weak-evidence pass for the "workable,
# not perfect" records the user explicitly accepted.
"$PY" -m training_pipeline.audit_dataset_live_admission \
  --processed_states "$FROZEN/processed_states" --train "$FROZEN/train_ids.txt" --test "$FROZEN/test_ids.txt" \
  --label_corrections "$Q/label_corrections.json" \
  --output "$FROZEN/admission_final_strict.json" --selection_dir "$FROZEN/selection_strict"
"$PY" -m training_pipeline.audit_dataset_live_admission \
  --processed_states "$FROZEN/processed_states" --train "$FROZEN/train_ids.txt" --test "$FROZEN/test_ids.txt" \
  --label_corrections "$Q/label_corrections.json" --admit_weak_evidence \
  --output "$FROZEN/admission_final_with_weak.json" --selection_dir "$FROZEN/selection_with_weak"
