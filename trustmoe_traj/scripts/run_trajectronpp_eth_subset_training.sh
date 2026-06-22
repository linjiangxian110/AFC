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
TRAJECTRONPP_ROOT="${TRAJECTRONPP_ROOT:-$MAIN/参考/开源基线模型/Trajectron-plus-plus}"
DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
VIS_EVERY="${VIS_EVERY:-999999}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LOG_ROOT="${LOG_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/trajectronpp/training_logs}"
MODEL_ROOT="${MODEL_ROOT:-$TRAJECTRONPP_ROOT/experiments/pedestrians/models}"
CONF_TEMPLATE="${CONF_TEMPLATE:-$MODEL_ROOT/DATASET_vel/config.json}"
RUN_PREPROCESS="${RUN_PREPROCESS:-0}"
OFFLINE_SCENE_GRAPH="${OFFLINE_SCENE_GRAPH:-yes}"
AUGMENT="${AUGMENT:-1}"

mkdir -p "$LOG_ROOT" "$MODEL_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/trajectronpp_training_manifest.log" >&2
}

render_template() {
  local template="$1"
  local dataset="$2"
  local seed="${3:-}"
  template="${template//\{dataset\}/$dataset}"
  template="${template//\{seed\}/$seed}"
  template="${template//DATASET/$dataset}"
  template="${template//SEED/$seed}"
  printf '%s\n' "$template"
}

maybe_preprocess() {
  local processed_dir="$TRAJECTRONPP_ROOT/experiments/processed"
  if [[ "$RUN_PREPROCESS" == "1" || ! -f "$processed_dir/eth_train.pkl" || ! -f "$processed_dir/zara2_test.pkl" ]]; then
    log "PREPROCESS Trajectron++ pedestrian data"
    cd "$TRAJECTRONPP_ROOT/experiments/pedestrians"
    "$PY" process_data.py 2>&1 | tee "$LOG_ROOT/process_data.log"
  else
    log "SKIP preprocess existing processed data at $processed_dir"
  fi
}

train_one() {
  local dataset="$1"
  local seed="$2"
  local conf
  local train_pkl="${dataset}_train.pkl"
  local val_pkl="${dataset}_val.pkl"
  local log_tag="_${dataset}_vel_afc_seed${seed}"
  local augment_args=()

  conf="$(render_template "$CONF_TEMPLATE" "$dataset" "$seed")"
  if [[ ! -f "$conf" ]]; then
    echo "Missing Trajectron++ config: $conf" >&2
    return 1
  fi
  if [[ ! -f "$TRAJECTRONPP_ROOT/experiments/processed/$train_pkl" ]]; then
    echo "Missing processed train data: $TRAJECTRONPP_ROOT/experiments/processed/$train_pkl" >&2
    return 1
  fi
  if [[ ! -f "$TRAJECTRONPP_ROOT/experiments/processed/$val_pkl" ]]; then
    echo "Missing processed val data: $TRAJECTRONPP_ROOT/experiments/processed/$val_pkl" >&2
    return 1
  fi
  if [[ "$AUGMENT" == "1" ]]; then
    augment_args=(--augment)
  fi

  log "TRAIN dataset=$dataset seed=$seed epochs=$TRAIN_EPOCHS gpu=$GPU conf=$conf"
  cd "$TRAJECTRONPP_ROOT/trajectron"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
    --eval_every "$EVAL_EVERY" \
    --vis_every "$VIS_EVERY" \
    --save_every "$SAVE_EVERY" \
    --train_data_dict "$train_pkl" \
    --eval_data_dict "$val_pkl" \
    --offline_scene_graph "$OFFLINE_SCENE_GRAPH" \
    --preprocess_workers "$PREPROCESS_WORKERS" \
    --log_dir "$MODEL_ROOT" \
    --log_tag "$log_tag" \
    --train_epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --seed "$seed" \
    "${augment_args[@]}" \
    --conf "$conf" 2>&1 | tee "$LOG_ROOT/train_${dataset}_seed${seed}.log"
}

log "START Trajectron++ subset training"
log "TRAJECTRONPP_ROOT=$TRAJECTRONPP_ROOT"
log "DATASETS=$DATASETS SEEDS=$SEEDS TRAIN_EPOCHS=$TRAIN_EPOCHS GPU=$GPU"

maybe_preprocess
for dataset in $DATASETS; do
  for seed in $SEEDS; do
    train_one "$dataset" "$seed"
  done
done

log "DONE Trajectron++ subset training"
