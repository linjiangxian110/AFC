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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_guidecot_afc_exp1_seed0}"
GUIDE_ROOT="${GUIDE_ROOT:-$MAIN/参考/开源基线模型/GUIDE-CoT}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/guidecot/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTPUT_ROOT/configs}"
CACHE_ROOT="${CACHE_ROOT:-$OUTPUT_ROOT/cache}"

DATASETS="${DATASETS:-zara1}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
GOAL_EPOCHS="${GOAL_EPOCHS:-1}"
LLM_EPOCHS="${LLM_EPOCHS:-1}"
LLM_PREPROCESS_WORKERS="${LLM_PREPROCESS_WORKERS:-8}"
LLM_INFERENCE_BATCH_SIZE="${LLM_INFERENCE_BATCH_SIZE:-1024}"
MAX_SCENES="${MAX_SCENES:-}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
MISS_THRESHOLD="${MISS_THRESHOLD:-2.0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CONFIG_ROOT" "$CACHE_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

optional_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    printf '%s\n%s\n' "$flag" "$value"
  fi
}

llm_tag_for_dataset() {
  local dataset="$1"
  local seed="$2"
  echo "GUIDE-CoT-${dataset}-pixel-seed${seed}-goal${GOAL_EPOCHS}-llm${LLM_EPOCHS}"
}

config_for_dataset() {
  local dataset="$1"
  local seed="$2"
  echo "$CONFIG_ROOT/guidecot-${dataset}-seed${seed}-afc.json"
}

prepare_config() {
  local dataset="$1"
  local seed="$2"
  local cfg
  cfg="$(config_for_dataset "$dataset" "$seed")"
  "$PY" -m trustmoe_traj.scripts.prepare_guidecot_config \
    --template "$GUIDE_ROOT/llm_module/config/config-pixel-g2p.json" \
    --output "$cfg" \
    --dataset "$dataset" \
    --epochs "$LLM_EPOCHS" \
    --seed "$seed" \
    --checkpoint-path "$CHECKPOINT_ROOT/llm" \
    --cache-dir "$CACHE_ROOT/hf" \
    --preprocessing-num-workers "$LLM_PREPROCESS_WORKERS" \
    --inference-batch-size "$LLM_INFERENCE_BATCH_SIZE" \
    --num-samples "$K" \
    --overwrite-cache >/dev/null
  echo "$cfg"
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
  test -d "$DATA_ROOT" || { echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
  "$PY" - <<'PY'
import importlib
import numpy as np
import torch

for name in ["accelerate", "datasets", "evaluate", "sklearn", "transformers", "sentencepiece"]:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')}")
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
    trustmoe_traj/scripts/summarize_guidecot_afc.py
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local cfg
  local tag
  local checkpoint_name
  local checkpoint_dir
  local goals_file
  local run_dir
  local bundle
  local output_json
  local max_scene_args=()
  local afc_max_args=()
  local branch="guidecot${K}_pred"

  test "$split" = "test" || { echo "[ERR] GUIDE-CoT only supports split=test, got: $split" >&2; exit 1; }
  cfg="$(prepare_config "$dataset" "$seed")"
  tag="$(llm_tag_for_dataset "$dataset" "$seed")"
  checkpoint_name="${tag}-g2p"
  checkpoint_dir="$CHECKPOINT_ROOT/llm/$checkpoint_name"
  goals_file="$GUIDE_ROOT/goal_module/output/$dataset/VisSem/${dataset}-test-goals-20.npy"
  run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  bundle="$run_dir/${dataset}_${split}_guidecot_k${K}.pt"
  output_json="$run_dir/${dataset}_${split}_guidecot_afc.json"

  test -f "$goals_file" || { echo "[ERR] missing GUIDE-CoT goal output: $goals_file" >&2; exit 1; }
  test -d "$checkpoint_dir" || { echo "[ERR] missing GUIDE-CoT LLM checkpoint: $checkpoint_dir" >&2; exit 1; }
  test -f "$GUIDE_ROOT/llm_module/datasets/preprocessed/${dataset}-test-8-12-pixel-g2p.json" || {
    echo "[ERR] missing GUIDE-CoT preprocessed test JSON for $dataset" >&2
    exit 1
  }
  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT GUIDE-CoT dataset=$dataset seed=$seed split=$split checkpoint=$checkpoint_name bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_guidecot_predictions \
      --guide-root "$GUIDE_ROOT" \
      --cfg "$cfg" \
      --dataset "$dataset" \
      --split "$split" \
      --checkpoint-name "$checkpoint_name" \
      --k "$K" \
      --seed "$seed" \
      --preprocessing-num-workers "$LLM_PREPROCESS_WORKERS" \
      --inference-batch-size "$LLM_INFERENCE_BATCH_SIZE" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC GUIDE-CoT dataset=$dataset seed=$seed split=$split output=$output_json"
    "$PY" -m trustmoe_traj.scripts.evaluate_guidecot_afc \
      --bundle "$bundle" \
      --dataset "$dataset" \
      --split "$split" \
      --data-root "$DATA_ROOT" \
      --branch-name "$branch" \
      --miss-threshold "$MISS_THRESHOLD" \
      --afc-top-m "$AFC_TOP_M" \
      --afc-eps "$AFC_EPS" \
      --afc-batch-scenes "$AFC_BATCH_SCENES" \
      "${afc_max_args[@]}" \
      --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/eval_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP eval existing json=$output_json"
  fi
}

check_env
log "START GUIDE-CoT AFC experiment"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K GOAL_EPOCHS=$GOAL_EPOCHS LLM_EPOCHS=$LLM_EPOCHS"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_guidecot_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "guidecot${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE GUIDE-CoT AFC experiment OUTPUT_ROOT=$OUTPUT_ROOT"
