#!/usr/bin/env bash
# Re-record the captures that cannot support live reward, one shard per
# application namespace (they never share a namespace, so they run in parallel).
# Must not run while any live-Twin audit/training uses test-social-network or
# test-hotel-reservation as a Twin source: regeneration mutates those namespaces.
set -euo pipefail
REPO=/home/ubuntu/multiagent_IBM
OUT=${OUT:-/mnt/aiops-training/datasets/regenerated-v1/raw}
Q=$REPO/configs/curriculum_622_49_v2/admission_v4
TELEMETRY=$REPO/AIOpsLab/dynamic_generated_scenario_results_all_new/telemetry_outputs
PY=$REPO/.venv-aiops312/bin/python
STAGE=${1:-must}            # must | weak | corrected_multifault
LIMIT=${LIMIT:-}
mkdir -p "$OUT"
cd "$REPO"
if kubectl get ns --no-headers | grep -q '^aiops-twin-'; then
  echo "refusing to start: live Twin namespaces exist (a Twin audit or training is running)" >&2
  exit 2
fi
declare -A JPORT=( [social]=16686 [hotel]=16687 )
for app in ${APPS:-social hotel}; do
  ids="$Q/regen_${STAGE}_${app}.txt"
  [ -s "$ids" ] || { echo "no ids for $STAGE/$app"; continue; }
  log="$OUT/${STAGE}_${app}.log"
  echo "launching $STAGE/$app ($(wc -l < "$ids") ids) -> $log"
  nohup "$PY" -m dataset_generation.regenerate \
    --scenario_ids "$ids" --spec_source telemetry_dir --telemetry_dir "$TELEMETRY" \
    --output_dir "$OUT" --apps "$app" --reuse_deployment --resume --jaeger_local_port "${JPORT[$app]}" \
    ${LIMIT:+--limit "$LIMIT"} \
    >> "$log" 2>&1 &
  echo "$!" > "$OUT/${STAGE}_${app}.pid"
  sleep 2
done
echo "done launching; monitor with: tail -f $OUT/${STAGE}_*.log"
