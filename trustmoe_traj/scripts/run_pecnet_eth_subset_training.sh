#!/usr/bin/env bash
set -euo pipefail

# Train PECNet checkpoints separately for ETH-UCY subsets using the official
# PECNet architecture/loss and TrustMoE ETH-UCY train/val splits.

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
PECNET_ROOT="${PECNET_ROOT:-$MAIN/参考/开源基线模型/PECNet}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
LOG_ROOT="${LOG_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/pecnet/pecnet_eth_subset_training_logs}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
CONFIG_FILE="${CONFIG_FILE:-optimal.yaml}"
SAVE_FILE_TEMPLATE="${SAVE_FILE_TEMPLATE:-PECNET_DATASET_officialcfg_seedSEED.pt}"
EPOCHS="${EPOCHS:-650}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_EVERY="${EVAL_EVERY:-5}"
BEST_OF_N="${BEST_OF_N:-20}"
MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-}"
MAX_EVAL_SCENES="${MAX_EVAL_SCENES:-}"
MIN_AGENTS="${MIN_AGENTS:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

mkdir -p "$LOG_ROOT"

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

render_save_file() {
  local template="$1"
  local dataset="$2"
  local seed="$3"
  template="${template//\{dataset\}/$dataset}"
  template="${template//DATASET/$dataset}"
  template="${template//\{seed\}/$seed}"
  template="${template//SEED/$seed}"
  printf '%s\n' "$template"
}

train_one() {
  local dataset="$1"
  local seed="$2"
  local save_file
  local save_path
  local max_train_args=()
  local max_eval_args=()

  save_file="$(render_save_file "$SAVE_FILE_TEMPLATE" "$dataset" "$seed")"
  save_path="$PECNET_ROOT/saved_models/$save_file"
  mapfile -t max_train_args < <(optional_arg "--max-train-scenes" "$MAX_TRAIN_SCENES")
  mapfile -t max_eval_args < <(optional_arg "--max-eval-scenes" "$MAX_EVAL_SCENES")

  if [[ "$FORCE_TRAIN" != "1" && -f "$save_path" ]]; then
    log "SKIP existing checkpoint dataset=$dataset seed=$seed path=$save_path"
    return
  fi

  log "TRAIN PECNet subset=$dataset seed=$seed save_file=$save_file epochs=$EPOCHS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.train_pecnet_eth_subset \
    --pecnet-root "$PECNET_ROOT" \
    --data-root "$DATA_ROOT" \
    --subset "$dataset" \
    --config-file "$CONFIG_FILE" \
    --save-file "$save_file" \
    --seed "$seed" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --eval-split "$EVAL_SPLIT" \
    --eval-every "$EVAL_EVERY" \
    --best-of-n "$BEST_OF_N" \
    --min-agents "$MIN_AGENTS" \
    "${max_train_args[@]}" \
    "${max_eval_args[@]}" 2>&1 | tee "$LOG_ROOT/train_${dataset}_seed${seed}.log"
}

log "START PECNet ETH subset training"
log "MAIN=$MAIN"
log "PECNET_ROOT=$PECNET_ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "DATASETS=$DATASETS SEEDS=$SEEDS EPOCHS=$EPOCHS EVAL_SPLIT=$EVAL_SPLIT EVAL_EVERY=$EVAL_EVERY"
log "SAVE_FILE_TEMPLATE=$SAVE_FILE_TEMPLATE"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    train_one "$dataset" "$seed"
  done
done

log "DONE PECNet ETH subset training"
