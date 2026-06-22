#!/usr/bin/env bash
set -euo pipefail

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /mnt/data/lck/code/moflow/moflow_venv/bin/python ]]; then
    PY=/mnt/data/lck/code/moflow/moflow_venv/bin/python
  else
    PY=python
  fi
fi

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_guidecot_timing_seed0_epoch1}"
GUIDE_ROOT="${GUIDE_ROOT:-$MAIN/参考/开源基线模型/GUIDE-CoT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/guidecot/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTPUT_ROOT/configs}"
CACHE_ROOT="${CACHE_ROOT:-$OUTPUT_ROOT/cache}"

DATASETS="${DATASETS:-zara1}"
SEED="${SEED:-0}"
GOAL_EPOCHS="${GOAL_EPOCHS:-1}"
LLM_EPOCHS="${LLM_EPOCHS:-1}"
TARGET_GOAL_EPOCHS="${TARGET_GOAL_EPOCHS:-50}"
TARGET_LLM_EPOCHS="${TARGET_LLM_EPOCHS:-3}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.2}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
GOAL_BATCH_SIZE="${GOAL_BATCH_SIZE:-64}"
GOAL_START_VALIDATION="${GOAL_START_VALIDATION:-0}"
LLM_PREPROCESS_WORKERS="${LLM_PREPROCESS_WORKERS:-8}"
LLM_TRAIN_BATCH_SIZE="${LLM_TRAIN_BATCH_SIZE:-64}"
LLM_EVAL_BATCH_SIZE="${LLM_EVAL_BATCH_SIZE:-64}"
LLM_INFERENCE_BATCH_SIZE="${LLM_INFERENCE_BATCH_SIZE:-1024}"
K="${K:-20}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT/llm" "$CONFIG_ROOT" "$CACHE_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

llm_tag_for_dataset() {
  local dataset="$1"
  echo "GUIDE-CoT-${dataset}-pixel-seed${SEED}-goal${GOAL_EPOCHS}-llm${LLM_EPOCHS}"
}

llm_config_for_dataset() {
  local dataset="$1"
  echo "$CONFIG_ROOT/guidecot-${dataset}-seed${SEED}-llm${LLM_EPOCHS}.json"
}

estimate_seconds() {
  "$PY" - "$1" "$GOAL_EPOCHS" "$LLM_EPOCHS" "$TARGET_GOAL_EPOCHS" "$TARGET_LLM_EPOCHS" "$SAFETY_FACTOR" <<'PY'
import math
import sys

elapsed = float(sys.argv[1])
goal_epochs = max(float(sys.argv[2]), 1.0)
llm_epochs = max(float(sys.argv[3]), 1.0)
target_goal = float(sys.argv[4])
target_llm = float(sys.argv[5])
safety = float(sys.argv[6])
scale = (target_goal + target_llm) / (goal_epochs + llm_epochs)
print(int(math.ceil(elapsed * scale * 5.0 * safety)))
PY
}

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV GPU=$GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} DEVICE=$DEVICE"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV GUIDE_ROOT=$GUIDE_ROOT"
  log "ENV CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$GUIDE_ROOT" || { echo "[ERR] missing GUIDE_ROOT: $GUIDE_ROOT" >&2; exit 1; }
  test -f "$GUIDE_ROOT/goal_module/main.py" || { echo "[ERR] missing GUIDE-CoT goal main.py" >&2; exit 1; }
  test -f "$GUIDE_ROOT/llm_module/trainval.py" || { echo "[ERR] missing GUIDE-CoT llm trainval.py" >&2; exit 1; }
  "$PY" - <<'PY'
import importlib
import numpy as np
import torch

modules = [
    "accelerate",
    "albumentations",
    "clip",
    "cv2",
    "datasets",
    "evaluate",
    "sentencepiece",
    "sklearn",
    "transformers",
    "wandb",
]
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}={version}")
print(f"numpy={np.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  "$PY" -m py_compile \
    trustmoe_traj/scripts/prepare_guidecot_config.py \
    trustmoe_traj/scripts/export_guidecot_predictions.py \
    trustmoe_traj/scripts/evaluate_guidecot_afc.py \
    trustmoe_traj/scripts/summarize_guidecot_afc.py \
    "$GUIDE_ROOT/goal_module/main.py" \
    "$GUIDE_ROOT/llm_module/trainval.py"
}

prepare_llm_config() {
  local dataset="$1"
  local cfg
  cfg="$(llm_config_for_dataset "$dataset")"
  "$PY" -m trustmoe_traj.scripts.prepare_guidecot_config \
    --template "$GUIDE_ROOT/llm_module/config/config-pixel-g2p.json" \
    --output "$cfg" \
    --dataset "$dataset" \
    --epochs "$LLM_EPOCHS" \
    --seed "$SEED" \
    --checkpoint-path "$CHECKPOINT_ROOT/llm" \
    --cache-dir "$CACHE_ROOT/hf" \
    --preprocessing-num-workers "$LLM_PREPROCESS_WORKERS" \
    --train-batch-size "$LLM_TRAIN_BATCH_SIZE" \
    --eval-batch-size "$LLM_EVAL_BATCH_SIZE" \
    --inference-batch-size "$LLM_INFERENCE_BATCH_SIZE" \
    --num-samples "$K" \
    --save-every 1 \
    --overwrite-cache >/dev/null
  echo "$cfg"
}

preprocess_llm_dataset() {
  local dataset="$1"
  local phase
  for phase in train val test; do
    local out="$GUIDE_ROOT/llm_module/datasets/preprocessed/${dataset}-${phase}-8-12-pixel-g2p.json"
    if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$out" ]]; then
      log "PREPROCESS GUIDE-CoT LLM dataset=$dataset phase=$phase"
      (
        cd "$GUIDE_ROOT/llm_module"
        "$PY" utils/preprocessor.py --dataset "$dataset" --phase "$phase"
      ) 2>&1 | tee "$LOG_ROOT/preprocess_${dataset}_${phase}.log"
    else
      log "SKIP preprocess existing=$out"
    fi
  done
}

train_goal_module() {
  local dataset="$1"
  log "TRAIN GUIDE-CoT goal_module dataset=$dataset seed=$SEED epochs=$GOAL_EPOCHS"
  (
    cd "$GUIDE_ROOT/goal_module"
    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
      --model_name VisSem \
      --phase train_test \
      --dataset eth5 \
      --test_set "$dataset" \
      --num_epochs "$GOAL_EPOCHS" \
      --validate_every 1 \
      --start_validation "$GOAL_START_VALIDATION" \
      --shuffle_test_batches False \
      --num_test_runs 1 \
      --down_factor 4 \
      --batch_size "$GOAL_BATCH_SIZE" \
      --sampler_temperature 1.2 \
      --prompt_type arrow \
      --prompt_color red \
      --scheduler ExponentialLR \
      --device cuda:0 \
      --use_wandb False
  ) 2>&1 | tee "$LOG_ROOT/train_goal_${dataset}_seed${SEED}_epoch${GOAL_EPOCHS}.log"
  test -f "$GUIDE_ROOT/goal_module/output/$dataset/VisSem/${dataset}-test-goals-20.npy" || {
    echo "[ERR] expected GUIDE-CoT goal output not found for $dataset" >&2
    exit 1
  }
}

train_llm_module() {
  local dataset="$1"
  local cfg
  local tag
  cfg="$(prepare_llm_config "$dataset")"
  tag="$(llm_tag_for_dataset "$dataset")"
  log "TRAIN GUIDE-CoT llm_module dataset=$dataset seed=$SEED epochs=$LLM_EPOCHS tag=$tag"
  (
    cd "$GUIDE_ROOT/llm_module"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m accelerate.commands.launch \
      --num_processes 1 \
      --mixed_precision no \
      trainval.py \
      --cfg "$cfg" \
      --dataset "$dataset" \
      --tag "$tag"
  ) 2>&1 | tee "$LOG_ROOT/train_llm_${dataset}_seed${SEED}_epoch${LLM_EPOCHS}.log"
  test -d "$CHECKPOINT_ROOT/llm/${tag}-g2p" || {
    echo "[ERR] expected GUIDE-CoT LLM checkpoint not found: $CHECKPOINT_ROOT/llm/${tag}-g2p" >&2
    exit 1
  }
}

run_one() {
  local dataset="$1"
  local start_ts
  local end_ts
  local elapsed_sec
  local estimate
  local tag
  test -d "$GUIDE_ROOT/data/eth5/$dataset/train" || { echo "[ERR] missing GUIDE-CoT goal train dir for $dataset" >&2; exit 1; }
  test -d "$GUIDE_ROOT/llm_module/datasets/$dataset/train" || { echo "[ERR] missing GUIDE-CoT LLM train dir for $dataset" >&2; exit 1; }

  start_ts="$(date +%s)"
  train_goal_module "$dataset"
  preprocess_llm_dataset "$dataset"
  train_llm_module "$dataset"
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"
  estimate="$(estimate_seconds "$elapsed_sec")"
  tag="$(llm_tag_for_dataset "$dataset")"
  {
    echo "dataset=$dataset"
    echo "seed=$SEED"
    echo "goal_epochs=$GOAL_EPOCHS"
    echo "llm_epochs=$LLM_EPOCHS"
    echo "target_goal_epochs=$TARGET_GOAL_EPOCHS"
    echo "target_llm_epochs=$TARGET_LLM_EPOCHS"
    echo "safety_factor=$SAFETY_FACTOR"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed_sec"
    echo "estimated_5dataset_full_sec=$estimate"
    echo "goal_output=$GUIDE_ROOT/goal_module/output/$dataset/VisSem/${dataset}-test-goals-20.npy"
    echo "llm_checkpoint=$CHECKPOINT_ROOT/llm/${tag}-g2p"
    echo "llm_checkpoint_name=${tag}-g2p"
  } | tee "$OUTPUT_ROOT/timing_${dataset}_seed${SEED}_goal${GOAL_EPOCHS}_llm${LLM_EPOCHS}.txt"
}

check_env
log "START GUIDE-CoT subset training DATASETS=$DATASETS SEED=$SEED GOAL_EPOCHS=$GOAL_EPOCHS LLM_EPOCHS=$LLM_EPOCHS"
for dataset in $DATASETS; do
  run_one "$dataset"
done
log "DONE GUIDE-CoT subset training OUTPUT_ROOT=$OUTPUT_ROOT"
