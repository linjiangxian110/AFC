#!/usr/bin/env bash
set -euo pipefail

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-/mnt/data/lck/code/moflow/moflow_venv/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_social_stgcnn_sdd_training_seed0}"
SOCIAL_ROOT="${SOCIAL_ROOT:-$MAIN/参考/开源基线模型/Social-STGCNN}"
SDD_DATA_ROOT="${SDD_DATA_ROOT:-$MAIN/MoFlow/data/sdd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/sdd_external_baselines/social_stgcnn/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-250}"
TRAIN_LR="${TRAIN_LR:-0.01}"
TRAIN_TAG="${TRAIN_TAG:-social-stgcnn-sdd}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.2}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

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

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV GPU=$GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV SOCIAL_ROOT=$SOCIAL_ROOT"
  log "ENV SDD_DATA_ROOT=$SDD_DATA_ROOT"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$SOCIAL_ROOT" || { echo "[ERR] missing SOCIAL_ROOT: $SOCIAL_ROOT" >&2; exit 1; }
  test -f "$SOCIAL_ROOT/train.py" || { echo "[ERR] missing Social-STGCNN train.py" >&2; exit 1; }
  test -f "$SDD_DATA_ROOT/original/sdd_train.pkl" || { echo "[ERR] missing sdd_train.pkl under $SDD_DATA_ROOT/original" >&2; exit 1; }
  test -f "$SDD_DATA_ROOT/original/sdd_test.pkl" || { echo "[ERR] missing sdd_test.pkl under $SDD_DATA_ROOT/original" >&2; exit 1; }
  "$PY" -m py_compile \
    trustmoe_traj/scripts/prepare_sdd_external_text_dataset.py \
    "$SOCIAL_ROOT/train.py" \
    "$SOCIAL_ROOT/utils.py"
}

prepare_data() {
  local force_args=()
  local max_train_args=()
  local max_test_args=()
  [[ "$FORCE_PREPARE" == "1" ]] && force_args=(--force)
  mapfile -t max_train_args < <(optional_arg "--max-train-records" "$MAX_TRAIN_RECORDS")
  mapfile -t max_test_args < <(optional_arg "--max-test-records" "$MAX_TEST_RECORDS")
  log "PREPARE Social-STGCNN SDD text dataset"
  "$PY" -m trustmoe_traj.scripts.prepare_sdd_external_text_dataset \
    --sdd-data-root "$SDD_DATA_ROOT" \
    --output-dataset-root "$SOCIAL_ROOT/datasets" \
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
print(int(math.ceil(elapsed / epochs * target * safety)))
PY
}

train_model() {
  local start_ts
  local end_ts
  local elapsed
  local estimate
  local checkpoint="$SOCIAL_ROOT/checkpoint/$TRAIN_TAG/val_best.pth"
  log "TRAIN Social-STGCNN SDD tag=$TRAIN_TAG seed=$SEED epochs=$EPOCHS"
  start_ts="$(date +%s)"
  (
    cd "$SOCIAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --lr "$TRAIN_LR" \
      --n_stgcnn 1 \
      --n_txpcnn 5 \
      --dataset sdd \
      --tag "$TRAIN_TAG" \
      --seed "$SEED" \
      --use_lrschd \
      --num_epochs "$EPOCHS"
  ) 2>&1 | tee "$LOG_ROOT/train_sdd_seed${SEED}_epoch${EPOCHS}.log"
  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  estimate="$(estimate_seconds "$elapsed" "$EPOCHS" "$TARGET_EPOCHS" "$SAFETY_FACTOR")"
  test -f "$checkpoint" || { echo "[ERR] expected checkpoint not found: $checkpoint" >&2; exit 1; }
  {
    echo "dataset=sdd"
    echo "seed=$SEED"
    echo "epochs=$EPOCHS"
    echo "target_epochs=$TARGET_EPOCHS"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed"
    echo "safety_factor=$SAFETY_FACTOR"
    echo "estimated_target_sec=$estimate"
    echo "checkpoint=$checkpoint"
    echo "args=$SOCIAL_ROOT/checkpoint/$TRAIN_TAG/args.pkl"
  } | tee "$OUTPUT_ROOT/timing_sdd_seed${SEED}_epoch${EPOCHS}.txt"
}

check_env
log "START Social-STGCNN SDD training"
prepare_data
train_model
log "DONE Social-STGCNN SDD training OUTPUT_ROOT=$OUTPUT_ROOT"
