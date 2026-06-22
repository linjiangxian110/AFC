#!/usr/bin/env bash
set -euo pipefail

# Export and evaluate PECNet on ETH-UCY subsets with the standard AFC protocol.

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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_pecnet_eth_afc_exp1}"
PECNET_ROOT="${PECNET_ROOT:-$MAIN/参考/开源基线模型/PECNet}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/pecnet/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

LOAD_FILE="${LOAD_FILE:-PECNET_social_model1.pt}"
LOAD_FILE_TEMPLATE="${LOAD_FILE_TEMPLATE:-}"
DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
MAX_SCENES="${MAX_SCENES:-}"
MIN_AGENTS="${MIN_AGENTS:-1}"
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

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local load_file="$LOAD_FILE"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_pecnet_eth_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_pecnet_afc.json"
  local max_scene_args=()
  local afc_max_args=()

  mkdir -p "$run_dir"
  if [[ -n "$LOAD_FILE_TEMPLATE" ]]; then
    load_file="${LOAD_FILE_TEMPLATE//\{dataset\}/$dataset}"
    load_file="${load_file//DATASET/$dataset}"
  fi
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT dataset=$dataset seed=$seed split=$split load_file=$load_file bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_pecnet_eth_predictions \
      --pecnet-root "$PECNET_ROOT" \
      --load-file "$load_file" \
      --data-root "$DATA_ROOT" \
      --subset "$dataset" \
      --split "$split" \
      --k "$K" \
      --seed "$seed" \
      --device "$DEVICE" \
      --min-agents "$MIN_AGENTS" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC dataset=$dataset seed=$seed split=$split output=$output_json"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.evaluate_pecnet_eth_afc \
      --bundle "$bundle" \
      --subset "$dataset" \
      --split "$split" \
      --data-root "$DATA_ROOT" \
      --branch-name "pecnet${K}_pred" \
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

log "START PECNet ETH AFC experiment"
log "MAIN=$MAIN"
log "PECNET_ROOT=$PECNET_ROOT"
log "DATA_ROOT=$DATA_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOAD_FILE=$LOAD_FILE"
log "LOAD_FILE_TEMPLATE=${LOAD_FILE_TEMPLATE:-}"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K AFC_EPS=$AFC_EPS"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_pecnet_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --dataset-labels "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "pecnet${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE PECNet ETH AFC experiment"
