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

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_trajevo_afc_exp1_seed0}"
TRAJEVO_ROOT="${TRAJEVO_ROOT:-$(find_baseline_root TrajEvo)}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/trajevo/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
TRAJEVO_HEURISTIC_DATASET="${TRAJEVO_HEURISTIC_DATASET:-}"
TRAJEVO_SEED_MODE="${TRAJEVO_SEED_MODE:-scene}"
TRAJEVO_SDD_SCALE_FACTOR="${TRAJEVO_SDD_SCALE_FACTOR:-100.0}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-}"
MAX_SCENES="${MAX_SCENES:-}"
MAX_RECORDS="${MAX_RECORDS:-}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
MISS_THRESHOLD="${MISS_THRESHOLD:-2.0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

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
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV TRAJEVO_ROOT=$TRAJEVO_ROOT"
  log "ENV DATA_ROOT=$DATA_ROOT"
  log "ENV DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K SEED_MODE=$TRAJEVO_SEED_MODE"
  log "ENV TRAJEVO_SDD_SCALE_FACTOR=$TRAJEVO_SDD_SCALE_FACTOR"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$TRAJEVO_ROOT" || { echo "[ERR] missing TRAJEVO_ROOT: $TRAJEVO_ROOT" >&2; exit 1; }
  test -d "$DATA_ROOT" || { echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
  test -d "$TRAJEVO_ROOT/trajectory_prediction/datasets" || { echo "[ERR] missing TrajEvo datasets dir" >&2; exit 1; }
  test -d "$TRAJEVO_ROOT/trajectory_prediction/trajevo" || { echo "[ERR] missing TrajEvo heuristic dir" >&2; exit 1; }
  TRAJEVO_ROOT="$TRAJEVO_ROOT" "$PY" - <<'PY'
import os
import sys
import numpy as np
import torch
root = os.environ["TRAJEVO_ROOT"]
sys.path.insert(0, root)
from trajectory_prediction import utils as trajevo_utils
print(f"numpy={np.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"trajevo_utils={trajevo_utils.__file__}")
PY
  "$PY" -m py_compile \
    trustmoe_traj/scripts/export_trajevo_predictions.py \
    trustmoe_traj/scripts/evaluate_trajevo_afc.py \
    trustmoe_traj/scripts/summarize_trajevo_afc.py
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_trajevo_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_trajevo_afc.json"
  local max_scene_args=()
  local sample_args=()
  local max_record_args=()
  local afc_max_args=()
  local branch="trajevo${K}_pred"
  local heuristic_dataset="$dataset"
  if [[ -n "$TRAJEVO_HEURISTIC_DATASET" ]]; then
    heuristic_dataset="$TRAJEVO_HEURISTIC_DATASET"
  elif [[ "$dataset" == "sdd" ]]; then
    heuristic_dataset="eth"
  fi

  test -f "$TRAJEVO_ROOT/trajectory_prediction/trajevo/${heuristic_dataset}.py" || {
    echo "[ERR] missing TrajEvo heuristic: $TRAJEVO_ROOT/trajectory_prediction/trajevo/${heuristic_dataset}.py" >&2
    return 1
  }
  test -d "$TRAJEVO_ROOT/trajectory_prediction/datasets/${dataset}/${split}" || {
    echo "[ERR] missing TrajEvo split: $TRAJEVO_ROOT/trajectory_prediction/datasets/${dataset}/${split}" >&2
    return 1
  }
  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t sample_args < <(optional_arg "--samples-per-scene" "$SAMPLES_PER_SCENE")
  mapfile -t max_record_args < <(optional_arg "--max-records" "$MAX_RECORDS")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT TrajEvo dataset=$dataset seed=$seed split=$split bundle=$bundle"
    "$PY" -m trustmoe_traj.scripts.export_trajevo_predictions \
      --trajevo-root "$TRAJEVO_ROOT" \
      --dataset "$dataset" \
      --heuristic-dataset "$heuristic_dataset" \
      --split "$split" \
      --k "$K" \
      --seed "$seed" \
      --seed-mode "$TRAJEVO_SEED_MODE" \
      --sdd-scale-factor "$TRAJEVO_SDD_SCALE_FACTOR" \
      "${sample_args[@]}" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC TrajEvo dataset=$dataset seed=$seed split=$split output=$output_json"
    "$PY" -m trustmoe_traj.scripts.evaluate_trajevo_afc \
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
      "${max_record_args[@]}" \
      --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/eval_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP eval existing json=$output_json"
  fi
}

check_env
log "START TrajEvo AFC experiment"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_trajevo_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "trajevo${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE TrajEvo AFC experiment OUTPUT_ROOT=$OUTPUT_ROOT"
