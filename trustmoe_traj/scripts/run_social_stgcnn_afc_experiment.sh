#!/usr/bin/env bash
set -euo pipefail

# Export and evaluate Social-STGCNN predictions with the AFC protocol.
# By default this uses the provided pretrained checkpoints. Set RUN_TRAIN=1 to
# retrain selected datasets before export.

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
DATASETS="${DATASETS:-eth hotel zara1}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_social_stgcnn_afc_exp1}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
SOCIAL_ROOT="${SOCIAL_ROOT:-$MAIN/参考/开源基线模型/Social-STGCNN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/social_stgcnn/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

K="${K:-20}"
RUN_TRAIN="${RUN_TRAIN:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-250}"
TRAIN_LR="${TRAIN_LR:-0.01}"
TRAIN_TAG_TEMPLATE="${TRAIN_TAG_TEMPLATE:-social-stgcnn-{dataset}}"

AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
MAX_SCENES="${MAX_SCENES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
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

render_template() {
  local template="$1"
  local dataset="$2"
  printf '%s\n' "${template//\{dataset\}/$dataset}"
}

train_one() {
  local dataset="$1"
  local tag
  tag="$(render_template "$TRAIN_TAG_TEMPLATE" "$dataset")"
  log "TRAIN Social-STGCNN dataset=$dataset tag=$tag epochs=$TRAIN_EPOCHS"
  (
    cd "$SOCIAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --lr "$TRAIN_LR" \
      --n_stgcnn 1 \
      --n_txpcnn 5 \
      --dataset "$dataset" \
      --tag "$tag" \
      --use_lrschd \
      --num_epochs "$TRAIN_EPOCHS"
  ) 2>&1 | tee "$LOG_ROOT/train_${dataset}.log"
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_social_stgcnn_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_social_stgcnn_afc.json"
  local max_scene_args=()
  local afc_max_args=()

  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT dataset=$dataset seed=$seed split=$split bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_social_stgcnn_predictions \
      --social-root "$SOCIAL_ROOT" \
      --dataset "$dataset" \
      --split "$split" \
      --k "$K" \
      --seed "$seed" \
      --device "$DEVICE" \
      --num-workers "$NUM_WORKERS" \
      "${max_scene_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC dataset=$dataset seed=$seed split=$split output=$output_json"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.evaluate_social_stgcnn_afc \
      --social-root "$SOCIAL_ROOT" \
      --bundle "$bundle" \
      --dataset "$dataset" \
      --split "$split" \
      --data-root "$DATA_ROOT" \
      --branch-name "social_stgcnn${K}_pred" \
      --miss-threshold "$MISS_THRESHOLD" \
      --afc-top-m "$AFC_TOP_M" \
      --afc-eps "$AFC_EPS" \
      "${afc_max_args[@]}" \
      --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/eval_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP eval existing json=$output_json"
  fi
}

log "START Social-STGCNN AFC experiment"
log "MAIN=$MAIN"
log "DATA_ROOT=$DATA_ROOT"
log "SOCIAL_ROOT=$SOCIAL_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "DATASETS=$DATASETS"
log "SEEDS=$SEEDS"
log "SPLITS=$SPLITS"
log "K=$K RUN_TRAIN=$RUN_TRAIN AFC_EPS=$AFC_EPS"

if [[ "$RUN_TRAIN" == "1" ]]; then
  for dataset in $DATASETS; do
    train_one "$dataset"
  done
fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_social_stgcnn_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "social_stgcnn${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE Social-STGCNN AFC experiment"
