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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_eigentrajectory_afc_exp1_seed0}"
EIGEN_ROOT="${EIGEN_ROOT:-$MAIN/参考/开源基线模型/EigenTrajectory}"
EIGEN_BASELINE="${EIGEN_BASELINE:-lbebm}"
TAG="${TAG:-EigenTrajectory-${EIGEN_BASELINE}}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/eigentrajectory/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$EIGEN_ROOT/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTPUT_ROOT/configs}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
MAX_SCENES="${MAX_SCENES:-}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
MISS_THRESHOLD="${MISS_THRESHOLD:-2.0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CONFIG_ROOT"

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
    --dataset-dir "$EIGEN_ROOT/datasets" >/dev/null
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
  log "ENV CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  log "ENV EIGEN_BASELINE=$EIGEN_BASELINE TAG=$TAG"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$EIGEN_ROOT" || { echo "[ERR] missing EIGEN_ROOT: $EIGEN_ROOT" >&2; exit 1; }
  test -d "$DATA_ROOT" || { echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
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
  "$PY" -m py_compile \
    trustmoe_traj/scripts/prepare_eigentrajectory_config.py \
    trustmoe_traj/scripts/export_eigentrajectory_predictions.py \
    trustmoe_traj/scripts/evaluate_eigentrajectory_afc.py \
    trustmoe_traj/scripts/summarize_eigentrajectory_afc.py
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
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local cfg
  local checkpoint="$CHECKPOINT_ROOT/$TAG/$dataset/model_best.pth"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_eigentrajectory_${EIGEN_BASELINE}_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_eigentrajectory_afc.json"
  local max_scene_args=()
  local afc_max_args=()
  local branch="eigentrajectory_${EIGEN_BASELINE}${K}_pred"

  cfg="$(prepare_config "$dataset")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing EigenTrajectory checkpoint: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT EigenTrajectory dataset=$dataset seed=$seed split=$split checkpoint=$checkpoint bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_eigentrajectory_predictions \
      --eigen-root "$EIGEN_ROOT" \
      --cfg "$cfg" \
      --tag "$TAG" \
      --dataset "$dataset" \
      --split "$split" \
      --baseline "$EIGEN_BASELINE" \
      --k "$K" \
      --seed "$seed" \
      --gpu-id "$GPU" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${EIGEN_BASELINE}_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC EigenTrajectory dataset=$dataset seed=$seed split=$split output=$output_json"
    "$PY" -m trustmoe_traj.scripts.evaluate_eigentrajectory_afc \
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
      --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/eval_${EIGEN_BASELINE}_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP eval existing json=$output_json"
  fi
}

check_env
log "START EigenTrajectory AFC experiment"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K BASELINE=$EIGEN_BASELINE GPU=$GPU"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_eigentrajectory_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "eigentrajectory_${EIGEN_BASELINE}${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE EigenTrajectory AFC experiment OUTPUT_ROOT=$OUTPUT_ROOT"
