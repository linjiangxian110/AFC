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

GPU="${GPU:-1}"
MID_ROOT="${MID_ROOT:-$MAIN/参考/开源基线模型/MID}"
CONFIG_FILE="${CONFIG_FILE:-$MID_ROOT/configs/trustmoe_afc_seed0.yaml}"
DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
LOG_ROOT="${LOG_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/mid/training_logs}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

mkdir -p "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/mid_training_manifest.log" >&2
}

maybe_preprocess() {
  if [[ "$RUN_PREPROCESS" != "1" ]]; then
    log "SKIP preprocess RUN_PREPROCESS=$RUN_PREPROCESS"
    return
  fi
  if [[ "$FORCE_PREPROCESS" != "1" && -f "$MID_ROOT/processed_data_noise/eth_train.pkl" && -f "$MID_ROOT/processed_data_noise/zara2_test.pkl" ]]; then
    log "SKIP preprocess existing processed_data_noise"
    return
  fi
  log "PREPROCESS MID ETH-UCY raw_data -> processed_data_noise"
  cd "$MID_ROOT"
  "$PY" process_data.py 2>&1 | tee "$LOG_ROOT/process_data.log"
}

train_one() {
  local dataset="$1"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing MID config: $CONFIG_FILE" >&2
    return 1
  fi
  if [[ ! -f "$MID_ROOT/processed_data_noise/${dataset}_train.pkl" ]]; then
    echo "Missing processed train data: $MID_ROOT/processed_data_noise/${dataset}_train.pkl" >&2
    return 1
  fi
  if [[ ! -f "$MID_ROOT/processed_data_noise/${dataset}_test.pkl" ]]; then
    echo "Missing processed test data: $MID_ROOT/processed_data_noise/${dataset}_test.pkl" >&2
    return 1
  fi

  log "TRAIN MID dataset=$dataset gpu=$GPU config=$CONFIG_FILE"
  cd "$MID_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
    --config "$CONFIG_FILE" \
    --dataset "$dataset" 2>&1 | tee "$LOG_ROOT/train_${dataset}.log"
}

log "START MID ETH-UCY subset training"
log "MAIN=$MAIN"
log "MID_ROOT=$MID_ROOT"
log "CONFIG_FILE=$CONFIG_FILE"
log "DATASETS=$DATASETS GPU=$GPU"

maybe_preprocess
for dataset in $DATASETS; do
  train_one "$dataset"
done

log "DONE MID ETH-UCY subset training"
