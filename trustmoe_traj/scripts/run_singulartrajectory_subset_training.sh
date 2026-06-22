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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_singulartrajectory_timing_seed0_epoch1}"
SINGULAR_ROOT="${SINGULAR_ROOT:-$MAIN/参考/开源基线模型/SingularTrajectory}"
SINGULAR_TASK="${SINGULAR_TASK:-stochastic}"
SINGULAR_BASELINE="${SINGULAR_BASELINE:-transformerdiffusion}"
TAG="${TAG:-SingularTrajectory-${SINGULAR_TASK}-seed${SEED:-0}-epoch${EPOCHS:-1}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/singulartrajectory/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUTPUT_ROOT/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTPUT_ROOT/configs}"

DATASETS="${DATASETS:-zara1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-256}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$CONFIG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

config_for_dataset() {
  local dataset="$1"
  echo "$CONFIG_ROOT/singulartrajectory-${SINGULAR_TASK}-${SINGULAR_BASELINE}-${dataset}.json"
}

prepare_config() {
  local dataset="$1"
  local template="$SINGULAR_ROOT/config/config_example.json"
  local cfg
  cfg="$(config_for_dataset "$dataset")"
  test -f "$template" || { echo "[ERR] missing SingularTrajectory config template: $template" >&2; exit 1; }
  "$PY" -m trustmoe_traj.scripts.prepare_singulartrajectory_config \
    --template "$template" \
    --output "$cfg" \
    --dataset "$dataset" \
    --task "$SINGULAR_TASK" \
    --baseline "$SINGULAR_BASELINE" \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --dataset-dir "$SINGULAR_ROOT/datasets" \
    --num-epochs "$EPOCHS" >/dev/null
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
  log "ENV SINGULAR_ROOT=$SINGULAR_ROOT"
  log "ENV CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  log "ENV SINGULAR_TASK=$SINGULAR_TASK SINGULAR_BASELINE=$SINGULAR_BASELINE TAG=$TAG"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$SINGULAR_ROOT" || { echo "[ERR] missing SINGULAR_ROOT: $SINGULAR_ROOT" >&2; exit 1; }
  test -f "$SINGULAR_ROOT/trainval.py" || { echo "[ERR] missing SingularTrajectory trainval.py" >&2; exit 1; }
  "$PY" - <<'PY'
import numpy as np
import scipy
import sklearn
import torch
import tqdm
print(f"numpy={np.__version__}")
print(f"scipy={scipy.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"torch={torch.__version__}")
print(f"tqdm={tqdm.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  "$PY" -m py_compile "$SINGULAR_ROOT/trainval.py" trustmoe_traj/scripts/prepare_singulartrajectory_config.py
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
  local cfg
  local start_ts
  local end_ts
  local elapsed
  local estimated
  local checkpoint="$CHECKPOINT_ROOT/$TAG/$dataset/model_best.pth"

  cfg="$(prepare_config "$dataset")"
  test -d "$SINGULAR_ROOT/datasets/$dataset/train" || { echo "[ERR] missing dataset train dir: $SINGULAR_ROOT/datasets/$dataset/train" >&2; exit 1; }
  test -d "$SINGULAR_ROOT/datasets/$dataset/val" || { echo "[ERR] missing dataset val dir: $SINGULAR_ROOT/datasets/$dataset/val" >&2; exit 1; }
  test -d "$SINGULAR_ROOT/datasets/$dataset/homography" || { echo "[ERR] missing dataset homography dir: $SINGULAR_ROOT/datasets/$dataset/homography" >&2; exit 1; }
  test -d "$SINGULAR_ROOT/datasets/$dataset/vectorfield" || { echo "[ERR] missing dataset vectorfield dir: $SINGULAR_ROOT/datasets/$dataset/vectorfield" >&2; exit 1; }

  log "TRAIN SingularTrajectory dataset=$dataset seed=$SEED epochs=$EPOCHS checkpoint_root=$CHECKPOINT_ROOT tag=$TAG"
  start_ts="$(date +%s)"
  (
    cd "$SINGULAR_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" trainval.py \
      --cfg "$cfg" \
      --tag "$TAG" \
      --gpu_id "$GPU" \
      --seed "$SEED" \
      --epochs "$EPOCHS"
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
log "START SingularTrajectory subset training DATASETS=$DATASETS SEED=$SEED EPOCHS=$EPOCHS"

for dataset in $DATASETS; do
  run_one "$dataset"
done

log "DONE SingularTrajectory subset training OUTPUT_ROOT=$OUTPUT_ROOT"
