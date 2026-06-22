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

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_lmtrajectory_zero_afc_exp1_seed0}"
LM_ROOT="${LM_ROOT:-$MAIN/参考/开源基线模型/LMTrajectory}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/lmtrajectory_zero/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

LMT_MODEL_NAME="${LMT_MODEL_NAME:-gpt-3.5-turbo-0301}"
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

model_tag() {
  echo "$LMT_MODEL_NAME" | tr '/ :' '___'
}

dump_for_dataset() {
  local dataset="$1"
  echo "$LM_ROOT/zero-shot/output_dump/$LMT_MODEL_NAME/${dataset}_chatgpt_api_dump.json"
}

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV LM_ROOT=$LM_ROOT"
  log "ENV DATA_ROOT=$DATA_ROOT"
  log "ENV LMT_MODEL_NAME=$LMT_MODEL_NAME"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$LM_ROOT" || { echo "[ERR] missing LM_ROOT: $LM_ROOT" >&2; exit 1; }
  test -d "$DATA_ROOT" || { echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
  test -d "$LM_ROOT/zero-shot" || { echo "[ERR] missing LMTrajectory zero-shot dir: $LM_ROOT/zero-shot" >&2; exit 1; }
  "$PY" - <<'PY'
import numpy as np
import sklearn
import torch
print(f"numpy={np.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  "$PY" -m py_compile \
    trustmoe_traj/scripts/export_lmtrajectory_zero_predictions.py \
    trustmoe_traj/scripts/evaluate_lmtrajectory_zero_afc.py \
    trustmoe_traj/scripts/summarize_lmtrajectory_zero_afc.py
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local tag
  local dump_json
  local run_dir
  local bundle
  local output_json
  local max_scene_args=()
  local afc_max_args=()
  local branch="lmtrajectory_zero${K}_pred"

  tag="$(model_tag)"
  dump_json="$(dump_for_dataset "$dataset")"
  run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  bundle="$run_dir/${dataset}_${split}_lmtrajectory_zero_${tag}_k${K}.pt"
  output_json="$run_dir/${dataset}_${split}_lmtrajectory_zero_afc.json"

  test "$split" = "test" || { echo "[ERR] LMTrajectory-ZERO only supports split=test, got: $split" >&2; exit 1; }
  if [[ ! -f "$dump_json" ]]; then
    echo "[ERR] missing LMTrajectory zero-shot dump: $dump_json" >&2
    echo "Hint: download LMTraj-ZERO_output_trajectory.zip and extract it into $LM_ROOT." >&2
    return 1
  fi
  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT LMTrajectory-ZERO dataset=$dataset seed=$seed split=$split model=$LMT_MODEL_NAME bundle=$bundle"
    "$PY" -m trustmoe_traj.scripts.export_lmtrajectory_zero_predictions \
      --lm-root "$LM_ROOT" \
      --dump-json "$dump_json" \
      --dataset "$dataset" \
      --split "$split" \
      --model-name "$LMT_MODEL_NAME" \
      --k "$K" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC LMTrajectory-ZERO dataset=$dataset seed=$seed split=$split output=$output_json"
    "$PY" -m trustmoe_traj.scripts.evaluate_lmtrajectory_zero_afc \
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
      --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/eval_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP eval existing json=$output_json"
  fi
}

check_env
log "START LMTrajectory-ZERO AFC experiment"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K MODEL=$LMT_MODEL_NAME"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_lmtrajectory_zero_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "lmtrajectory_zero${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE LMTrajectory-ZERO AFC experiment OUTPUT_ROOT=$OUTPUT_ROOT"
