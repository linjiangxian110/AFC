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
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_graphtern_timing_seed0_epoch1}"
SDD_DATA_ROOT="${SDD_DATA_ROOT:-$MAIN/MoFlow/data/sdd}"
GRAPH_TERN_ROOT="${GRAPH_TERN_ROOT:-$MAIN/参考/开源基线模型/GraphTERN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/graphtern/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"

DATASETS="${DATASETS:-zara1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.0}"
TAG_PREFIX="${TAG_PREFIX:-graphtern_seed${SEED}_}"
TAG_SUFFIX="${TAG_SUFFIX:-_epoch${EPOCHS}}"
PREPARE_SDD_DATA="${PREPARE_SDD_DATA:-1}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

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

tag_for_dataset() {
  local dataset="$1"
  echo "${TAG_PREFIX}${dataset}${TAG_SUFFIX}"
}

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV GPU=$GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} DEVICE=$DEVICE"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV GRAPH_TERN_ROOT=$GRAPH_TERN_ROOT"
  log "ENV SDD_DATA_ROOT=$SDD_DATA_ROOT"
  log "ENV CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$GRAPH_TERN_ROOT" || { echo "[ERR] missing GRAPH_TERN_ROOT: $GRAPH_TERN_ROOT" >&2; exit 1; }
  test -f "$GRAPH_TERN_ROOT/train.py" || { echo "[ERR] missing GraphTERN train.py" >&2; exit 1; }
  "$PY" - <<'PY'
import numpy as np
import torch
import tqdm
print(f"numpy={np.__version__}")
print(f"torch={torch.__version__}")
print(f"tqdm={tqdm.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  "$PY" -m py_compile "$GRAPH_TERN_ROOT/train.py" trustmoe_traj/scripts/prepare_sdd_external_text_dataset.py
}

prepare_sdd_data() {
  local force_args=()
  local max_train_args=()
  local max_test_args=()
  [[ "$FORCE_PREPARE" == "1" ]] && force_args=(--force)
  mapfile -t max_train_args < <(optional_arg "--max-train-records" "$MAX_TRAIN_RECORDS")
  mapfile -t max_test_args < <(optional_arg "--max-test-records" "$MAX_TEST_RECORDS")
  log "PREPARE GraphTERN SDD text dataset"
  "$PY" -m trustmoe_traj.scripts.prepare_sdd_external_text_dataset \
    --sdd-data-root "$SDD_DATA_ROOT" \
    --output-dataset-root "$GRAPH_TERN_ROOT/datasets" \
    --dataset-name sdd \
    --val-fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    "${max_train_args[@]}" \
    "${max_test_args[@]}" \
    "${force_args[@]}" \
    --summary-json "$OUTPUT_ROOT/sdd_prepare_summary.json" 2>&1 | tee "$LOG_ROOT/prepare_sdd_dataset.log"
}

estimate_seconds() {
  "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import math
import sys

elapsed = float(sys.argv[1])
epochs = max(float(sys.argv[2]), 1.0)
target = float(sys.argv[3])
safety = float(sys.argv[4])
print(int(math.ceil(elapsed / epochs * target * 5.0 * safety)))
PY
}

run_one() {
  local dataset="$1"
  local tag
  tag="$(tag_for_dataset "$dataset")"
  local start_ts
  local end_ts
  local elapsed
  local estimated
  local checkpoint="$CHECKPOINT_ROOT/$tag/${dataset}_best.pth"

  test -d "$GRAPH_TERN_ROOT/datasets/$dataset/train" || { echo "[ERR] missing dataset train dir: $GRAPH_TERN_ROOT/datasets/$dataset/train" >&2; exit 1; }
  test -d "$GRAPH_TERN_ROOT/datasets/$dataset/val" || { echo "[ERR] missing dataset val dir: $GRAPH_TERN_ROOT/datasets/$dataset/val" >&2; exit 1; }

  log "TRAIN GraphTERN dataset=$dataset seed=$SEED epochs=$EPOCHS checkpoint_root=$CHECKPOINT_ROOT tag=$tag"
  start_ts="$(date +%s)"
  (
    cd "$GRAPH_TERN_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --dataset "$dataset" \
      --num_epochs "$EPOCHS" \
      --seed "$SEED" \
      --num_workers "$NUM_WORKERS" \
      --checkpoint_root "$CHECKPOINT_ROOT" \
      --tag "$tag"
  ) 2>&1 | tee "$LOG_ROOT/train_${dataset}_seed${SEED}.log"
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  estimated="$(estimate_seconds "$elapsed" "$EPOCHS" "$TARGET_EPOCHS" "$SAFETY_FACTOR")"

  {
    echo "dataset=$dataset"
    echo "seed=$SEED"
    echo "epochs=$EPOCHS"
    echo "target_epochs=$TARGET_EPOCHS"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed"
    echo "safety_factor=$SAFETY_FACTOR"
    echo "estimated_5dataset_full_sec=$estimated"
    echo "checkpoint=$checkpoint"
  } | tee "$OUTPUT_ROOT/timing_${dataset}_seed${SEED}.txt"
}

check_env
log "START GraphTERN subset training DATASETS=$DATASETS SEED=$SEED EPOCHS=$EPOCHS"

if [[ "$PREPARE_SDD_DATA" == "1" ]] && [[ " $DATASETS " == *" sdd "* ]]; then
  prepare_sdd_data
fi

for dataset in $DATASETS; do
  run_one "$dataset"
done

log "DONE GraphTERN subset training OUTPUT_ROOT=$OUTPUT_ROOT"
