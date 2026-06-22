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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_tutr_afc_exp1_seed0}"
TUTR_ROOT="${TUTR_ROOT:-$(find_baseline_root TUTR)}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/tutr/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$TUTR_ROOT/checkpoint}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
TUTR_SDD_SCALE_FACTOR="${TUTR_SDD_SCALE_FACTOR:-1.0}"
MAX_RECORDS="${MAX_RECORDS:-}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
MISS_THRESHOLD="${MISS_THRESHOLD:-2.0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

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
  log "ENV CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
  log "ENV TUTR_SDD_SCALE_FACTOR=$TUTR_SDD_SCALE_FACTOR"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$TUTR_ROOT" || { echo "[ERR] missing TUTR_ROOT: $TUTR_ROOT" >&2; exit 1; }
  test -d "$DATA_ROOT" || { echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
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
  "$PY" -m py_compile \
    trustmoe_traj/scripts/export_tutr_predictions.py \
    trustmoe_traj/scripts/evaluate_tutr_afc.py \
    trustmoe_traj/scripts/summarize_tutr_afc.py
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
        --seed "0" \
        --dataset_name "$dataset"
    ) 2>&1 | tee "$LOG_ROOT/preprocess_${dataset}.log"
  else
    log "SKIP preprocess existing pkl dataset=$dataset train=$train_pkl test=$test_pkl"
  fi
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local checkpoint="$CHECKPOINT_ROOT/${dataset}/best.pth"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_tutr_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_tutr_afc.json"
  local max_record_args=()
  local afc_max_args=()

  prepare_dataset "$dataset"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing TUTR checkpoint: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  mapfile -t max_record_args < <(optional_arg "--max-records" "$MAX_RECORDS")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT TUTR dataset=$dataset seed=$seed split=$split checkpoint=$checkpoint bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_tutr_predictions \
      --tutr-root "$TUTR_ROOT" \
      --dataset-path "$TUTR_ROOT/dataset" \
      --dataset-name "$dataset" \
      --split "$split" \
      --hp-config "$TUTR_ROOT/config/${dataset}.py" \
      --checkpoint "$checkpoint" \
      --k "$K" \
      --seed "$seed" \
      --device "$DEVICE" \
      --sdd-scale-factor "$TUTR_SDD_SCALE_FACTOR" \
      "${max_record_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC TUTR dataset=$dataset seed=$seed split=$split output=$output_json"
    "$PY" -m trustmoe_traj.scripts.evaluate_tutr_afc \
      --bundle "$bundle" \
      --dataset "$dataset" \
      --split "$split" \
      --data-root "$DATA_ROOT" \
      --branch-name "tutr${K}_pred" \
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
log "START TUTR AFC experiment"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K GPU=$GPU"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_tutr_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "tutr${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE TUTR AFC experiment OUTPUT_ROOT=$OUTPUT_ROOT"
