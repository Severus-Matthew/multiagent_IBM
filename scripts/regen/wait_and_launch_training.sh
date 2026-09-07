#!/usr/bin/env bash
# Waits for both regeneration shards to finish, composes the frozen dataset,
# runs the final admission audit, then launches live GRPO training.
# Run this INSIDE tmux so it survives a lost connection:
#   tmux new -s aiops-train
#   bash scripts/regen/wait_and_launch_training.sh
#   # detach with Ctrl-b d ; reattach later with: tmux attach -t aiops-train
set -euo pipefail
REPO=/home/ubuntu/multiagent_IBM
cd "$REPO"

RAW=/mnt/aiops-training/datasets/regenerated-v1/raw
VERSION=full-622-49-v1
FROZEN=/mnt/aiops-training/datasets/$VERSION
RUN_DIR=/mnt/aiops-training/runs/live-grpo-stage1
LOG=/mnt/aiops-training/logs/wait_and_launch_$(date -u +%Y%m%d_%H%M%S).log
mkdir -p /mnt/aiops-training/logs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%H:%M:%S)] checking preconditions"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "FATAL: OPENAI_API_KEY is not set in this shell. Export it before running this script:"
  echo "  export OPENAI_API_KEY=sk-..."
  exit 1
fi

echo "[$(date -u +%H:%M:%S)] waiting for both regeneration shards to finish"
for f in must_social must_hotel; do
  log="$RAW/$f.log"
  while true; do
    [ -f "$log" ] && grep -q '"event": "shard_done"' "$log" && break
    sleep 30
  done
  echo "[$(date -u +%H:%M:%S)] $f done"
done

# Give the two shards a moment to fully release their kubectl port-forwards
# and any in-flight cleanup before touching the cluster/source namespaces again.
sleep 15
for i in $(seq 1 20); do
  ns_count=$(kubectl get ns --no-headers 2>/dev/null | grep -c '^aiops-twin-' || true)
  [ "$ns_count" = "0" ] && break
  echo "[$(date -u +%H:%M:%S)] waiting for $ns_count leftover aiops-twin- namespace(s) to clear"
  sleep 15
done

echo "[$(date -u +%H:%M:%S)] regeneration complete; composing frozen dataset"
bash scripts/regen/postprocess_regenerated.sh

echo "[$(date -u +%H:%M:%S)] dataset ready at $FROZEN; launching training"
mkdir -p "$RUN_DIR"
export HF_HOME=/mnt/aiops-training/cache/huggingface
export HF_HUB_CACHE=/mnt/aiops-training/cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/aiops-training/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv-training/bin/python -m training_pipeline.train_qwen_live_grpo \
  --processed_states "$FROZEN/processed_states" \
  --scenario_ids "$FROZEN/selection_with_weak/train_fully_live_reward_admissible.txt" \
  --label_corrections "$REPO/configs/curriculum_622_49_v2/admission_v4/label_corrections.json" \
  --admit_weak_evidence \
  --output_dir "$RUN_DIR" \
  --twin_mode live \
  --downstream_provider openai --openai_rca_model gpt-5.2 --openai_action_model gpt-5.2 \
  --policy_version qwen-live-grpo-stage1-v1 \
  --checkpoint_every_updates 1 --checkpoint_keep_last 50 \
  --wandb --wandb_project aiops-rl --wandb_run_name live-grpo-stage1-$(date -u +%Y%m%d) \
  --wandb_tags stage1,full-622-49,admit-weak-evidence,downstream-gpt-5.2
