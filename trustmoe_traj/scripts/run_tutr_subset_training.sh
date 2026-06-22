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

find_baseline_root() {
  local name="$1"
  local candidate
  for candidate in \
    "$MAIN/参考/开源基线模型/$name" \
    "$MAIN/baselines/$name" \
    "$MAIN/$name"
  do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  candidate="$(find "$MAIN" -maxdepth 5 -type d -name "$name" -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" && -d "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  printf '%s\n' "$MAIN/参考/开源基线模型/$name"
}

GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_tutr_timing_seed0_epoch1}"
TUTR_ROOT="${TUTR_ROOT:-$(find_baseline_root TUTR)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/tutr/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"

DATASETS="${DATASETS:-zara1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

map_tutr_data_dir() {
  case "$1" in
    zara1) echo "zara01" ;;
    zara2) echo "zara02" ;;
    *) echo "$1" ;;
  esac
}

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV GPU=$GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} DEVICE=$DEVICE"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV TUTR_ROOT=$TUTR_ROOT"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$TUTR_ROOT" || { echo "[ERR] missing TUTR_ROOT: $TUTR_ROOT" >&2; exit 1; }
  test -f "$TUTR_ROOT/train.py" || { echo "[ERR] missing TUTR train.py" >&2; exit 1; }
  test -f "$TUTR_ROOT/get_data_pkl.py" || { echo "[ERR] missing TUTR get_data_pkl.py" >&2; exit 1; }
  "$PY" - <<'PY'
import matplotlib
import numpy as np
import sklearn
import torch
print(f"numpy={np.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"matplotlib={matplotlib.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
}

prepare_dataset() {
  local dataset="$1"
  local data_dir_name
  data_dir_name="$(map_tutr_data_dir "$dataset")"
  local hp_config="$TUTR_ROOT/config/${dataset}.py"
  local train_dir="$TUTR_ROOT/data/${data_dir_name}/train"
  local test_dir="$TUTR_ROOT/data/${data_dir_name}/test"
  local train_pkl="$TUTR_ROOT/dataset/${dataset}_train.pkl"
  local test_pkl="$TUTR_ROOT/dataset/${dataset}_test.pkl"

  test -f "$hp_config" || { echo "[ERR] missing TUTR config: $hp_config" >&2; exit 1; }
  test -d "$train_dir" || { echo "[ERR] missing TUTR train dir: $train_dir" >&2; exit 1; }
  test -d "$test_dir" || { echo "[ERR] missing TUTR test dir: $test_dir" >&2; exit 1; }

  if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$train_pkl" || ! -f "$test_pkl" ]]; then
    log "PREPROCESS TUTR dataset=$dataset source=$data_dir_name"
    (
      cd "$TUTR_ROOT"
      "$PY" get_data_pkl.py \
        --train "data/${data_dir_name}/train" \
        --test "data/${data_dir_name}/test" \
        --config "config/${dataset}.py" \
        --device cpu \
        --seed "$SEED" \
        --dataset_name "$dataset"
    ) 2>&1 | tee "$LOG_ROOT/preprocess_${dataset}.log"
  else
    log "SKIP preprocess existing pkl dataset=$dataset train=$train_pkl test=$test_pkl"
  fi
}

train_one() {
  local dataset="$1"
  local hp_config="$TUTR_ROOT/config/${dataset}.py"
  local dataset_path="$TUTR_ROOT/dataset/"
  local checkpoint_dir="$CHECKPOINT_ROOT/"
  local start_ts
  local end_ts
  local elapsed_sec
  local estimate

  prepare_dataset "$dataset"
  log "TRAIN TUTR dataset=$dataset seed=$SEED epochs=$EPOCHS checkpoint_root=$CHECKPOINT_ROOT"
  start_ts="$(date +%s)"
  (
    cd "$TUTR_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --dataset_path "$dataset_path" \
      --dataset_name "$dataset" \
      --hp_config "$hp_config" \
      --gpu "$GPU" \
      --seed "$SEED" \
      --checkpoint "$checkpoint_dir" \
      --epochs "$EPOCHS" \
      --num_works "$NUM_WORKERS"
  ) 2>&1 | tee "$LOG_ROOT/train_${dataset}_seed${SEED}_epoch${EPOCHS}.log"
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"
  estimate="$("$PY" - "$elapsed_sec" "$EPOCHS" "$TARGET_EPOCHS" "$DATASETS" <<'PY'
import sys
elapsed = float(sys.argv[1])
epochs = max(float(sys.argv[2]), 1.0)
target_epochs = float(sys.argv[3])
dataset_count = max(len(str(sys.argv[4]).split()), 1)
full = elapsed / epochs * target_epochs * float(dataset_count) * 1.2
print(int(round(full)))
PY
)"
  {
    echo "dataset=$dataset"
    echo "seed=$SEED"
    echo "epochs=$EPOCHS"
    echo "target_epochs=$TARGET_EPOCHS"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed_sec"
    echo "estimated_full_sec=$estimate"
    echo "checkpoint=$CHECKPOINT_ROOT/${dataset}/best.pth"
  } | tee "$OUTPUT_ROOT/timing_${dataset}_seed${SEED}_epoch${EPOCHS}.txt"
}

check_env
log "START TUTR subset training DATASETS=$DATASETS SEED=$SEED EPOCHS=$EPOCHS"
for dataset in $DATASETS; do
  train_one "$dataset"
done
log "DONE TUTR subset training OUTPUT_ROOT=$OUTPUT_ROOT"
