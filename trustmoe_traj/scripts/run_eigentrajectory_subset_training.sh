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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_eigentrajectory_timing_seed0_epoch1}"
SDD_DATA_ROOT="${SDD_DATA_ROOT:-$MAIN/MoFlow/data/sdd}"
EIGEN_ROOT="${EIGEN_ROOT:-$MAIN/参考/开源基线模型/EigenTrajectory}"
EIGEN_BASELINE="${EIGEN_BASELINE:-lbebm}"
TAG="${TAG:-EigenTrajectory-${EIGEN_BASELINE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/eigentrajectory/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTPUT_ROOT/configs}"

DATASETS="${DATASETS:-zara1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-256}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.2}"
PREPARE_SDD_DATA="${PREPARE_SDD_DATA:-1}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$CONFIG_ROOT"

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

config_for_dataset() {
  local dataset="$1"
  echo "$CONFIG_ROOT/eigentrajectory-${EIGEN_BASELINE}-${dataset}.json"
}

prepare_config() {
  local dataset="$1"
  local template="$EIGEN_ROOT/config/eigentrajectory-{baseline}-${dataset}.json"
  local cfg
  cfg="$(config_for_dataset "$dataset")"
  if [[ ! -f "$template" && "$dataset" == "sdd" ]]; then
    template="$EIGEN_ROOT/config/eigentrajectory-{baseline}-zara2.json"
  fi
  test -f "$template" || { echo "[ERR] missing EigenTrajectory config template: $template" >&2; exit 1; }
  "$PY" -m trustmoe_traj.scripts.prepare_eigentrajectory_config \
    --template "$template" \
    --output "$cfg" \
    --dataset "$dataset" \
    --baseline "$EIGEN_BASELINE" \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --dataset-dir "$EIGEN_ROOT/datasets" \
    --epochs "$EPOCHS" >/dev/null
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
  log "ENV EIGEN_ROOT=$EIGEN_ROOT"
  log "ENV SDD_DATA_ROOT=$SDD_DATA_ROOT"
  log "ENV EIGEN_BASELINE=$EIGEN_BASELINE TAG=$TAG"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$EIGEN_ROOT" || { echo "[ERR] missing EIGEN_ROOT: $EIGEN_ROOT" >&2; exit 1; }
  test -f "$EIGEN_ROOT/trainval.py" || { echo "[ERR] missing trainval.py" >&2; exit 1; }
  "$PY" - <<'PY'
import numpy as np
import sklearn
import torch
import tqdm
print(f"numpy={np.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"torch={torch.__version__}")
print(f"tqdm={tqdm.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  (
    cd "$EIGEN_ROOT"
    "$PY" - "$EIGEN_BASELINE" <<'PY'
import sys

baseline_name = sys.argv[1]
import baseline

baseline_module = getattr(baseline, baseline_name)
from EigenTrajectory import EigenTrajectory
from utils import DotDict, get_exp_config
import utils.trainer as trainer

print(f"baseline_import={baseline_name}")
print(f"baseline_predictor={baseline_module.TrajectoryPredictor.__name__}")
print(f"eigentraj_model={EigenTrajectory.__name__}")
print("trainer_import=ok")
PY
  )
  "$PY" -m py_compile trustmoe_traj/scripts/prepare_sdd_external_text_dataset.py
}

prepare_sdd_data() {
  local force_args=()
  local max_train_args=()
  local max_test_args=()
  [[ "$FORCE_PREPARE" == "1" ]] && force_args=(--force)
  mapfile -t max_train_args < <(optional_arg "--max-train-records" "$MAX_TRAIN_RECORDS")
  mapfile -t max_test_args < <(optional_arg "--max-test-records" "$MAX_TEST_RECORDS")
  log "PREPARE EigenTrajectory SDD text dataset"
  "$PY" -m trustmoe_traj.scripts.prepare_sdd_external_text_dataset \
    --sdd-data-root "$SDD_DATA_ROOT" \
    --output-dataset-root "$EIGEN_ROOT/datasets" \
    --dataset-name sdd \
    --val-fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    "${max_train_args[@]}" \
    "${max_test_args[@]}" \
    "${force_args[@]}" \
    --summary-json "$OUTPUT_ROOT/sdd_prepare_summary.json" 2>&1 | tee "$LOG_ROOT/prepare_sdd_dataset.log"
}

train_one() {
  local dataset="$1"
  local cfg
  local checkpoint
  local start_ts
  local end_ts
  local elapsed_sec
  local estimate

  test -d "$EIGEN_ROOT/datasets/$dataset/train" || { echo "[ERR] missing dataset train dir: $EIGEN_ROOT/datasets/$dataset/train" >&2; exit 1; }
  test -d "$EIGEN_ROOT/datasets/$dataset/val" || { echo "[ERR] missing dataset val dir: $EIGEN_ROOT/datasets/$dataset/val" >&2; exit 1; }
  test -d "$EIGEN_ROOT/datasets/$dataset/test" || { echo "[ERR] missing dataset test dir: $EIGEN_ROOT/datasets/$dataset/test" >&2; exit 1; }
  cfg="$(prepare_config "$dataset")"
  checkpoint="$CHECKPOINT_ROOT/$TAG/$dataset/model_best.pth"

  log "TRAIN EigenTrajectory dataset=$dataset baseline=$EIGEN_BASELINE seed=$SEED epochs=$EPOCHS cfg=$cfg"
  start_ts="$(date +%s)"
  (
    cd "$EIGEN_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" trainval.py \
      --cfg "$cfg" \
      --tag "$TAG" \
      --gpu_id "$GPU" \
      --seed "$SEED" \
      --epochs "$EPOCHS"
  ) 2>&1 | tee "$LOG_ROOT/train_${EIGEN_BASELINE}_${dataset}_seed${SEED}_epoch${EPOCHS}.log"
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"
  estimate="$("$PY" - "$elapsed_sec" "$EPOCHS" "$TARGET_EPOCHS" "$SAFETY_FACTOR" <<'PY'
import sys
elapsed = float(sys.argv[1])
epochs = max(float(sys.argv[2]), 1.0)
target_epochs = float(sys.argv[3])
safety = float(sys.argv[4])
full = elapsed / epochs * target_epochs * 5.0 * safety
print(int(round(full)))
PY
)"
  {
    echo "dataset=$dataset"
    echo "baseline=$EIGEN_BASELINE"
    echo "tag=$TAG"
    echo "seed=$SEED"
    echo "epochs=$EPOCHS"
    echo "target_epochs=$TARGET_EPOCHS"
    echo "safety_factor=$SAFETY_FACTOR"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed_sec"
    echo "estimated_5dataset_full_sec=$estimate"
    echo "config=$cfg"
    echo "checkpoint=$checkpoint"
  } | tee "$OUTPUT_ROOT/timing_${EIGEN_BASELINE}_${dataset}_seed${SEED}_epoch${EPOCHS}.txt"
  test -f "$checkpoint" || { echo "[ERR] expected checkpoint not found: $checkpoint" >&2; exit 1; }
}

check_env
log "START EigenTrajectory subset training DATASETS=$DATASETS BASELINE=$EIGEN_BASELINE SEED=$SEED EPOCHS=$EPOCHS"
if [[ "$PREPARE_SDD_DATA" == "1" ]] && [[ " $DATASETS " == *" sdd "* ]]; then
  prepare_sdd_data
fi
for dataset in $DATASETS; do
  train_one "$dataset"
done
log "DONE EigenTrajectory subset training OUTPUT_ROOT=$OUTPUT_ROOT"
