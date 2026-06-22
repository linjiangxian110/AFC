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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_trajectronpp_afc_exp1}"
TRAJECTRONPP_ROOT="${TRAJECTRONPP_ROOT:-$MAIN/参考/开源基线模型/Trajectron-plus-plus}"
DATA_ROOT="${DATA_ROOT:-$MAIN/MoFlow/data/eth_ucy/original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/external_baselines/trajectronpp/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
K="${K:-20}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-}"
MODEL_GLOB_TEMPLATE="${MODEL_GLOB_TEMPLATE:-$TRAJECTRONPP_ROOT/experiments/pedestrians/models/models_*_DATASET_vel_afc_seedSEED}"
CHECKPOINT="${CHECKPOINT:-auto}"
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

resolve_model_dir() {
  local dataset="$1"
  local seed="$2"
  local model_dir=""
  local glob_pattern=""
  local matches=()

  if [[ -n "$MODEL_TEMPLATE" ]]; then
    model_dir="$(render_template "$MODEL_TEMPLATE" "$dataset" "$seed")"
    if [[ -d "$model_dir" ]]; then
      printf '%s\n' "$model_dir"
      return 0
    fi
  fi

  glob_pattern="$(render_template "$MODEL_GLOB_TEMPLATE" "$dataset" "$seed")"
  mapfile -t matches < <(compgen -G "$glob_pattern" || true)
  if [[ "${#matches[@]}" -gt 0 ]]; then
    ls -td "${matches[@]}" | head -1
    return 0
  fi

  model_dir="$TRAJECTRONPP_ROOT/experiments/pedestrians/models/${dataset}_vel"
  if [[ -d "$model_dir" ]]; then
    printf '%s\n' "$model_dir"
    return 0
  fi

  echo "Could not resolve Trajectron++ model dir for dataset=$dataset seed=$seed" >&2
  echo "MODEL_TEMPLATE=$MODEL_TEMPLATE" >&2
  echo "MODEL_GLOB_TEMPLATE=$MODEL_GLOB_TEMPLATE" >&2
  return 1
}

resolve_checkpoint() {
  local model_dir="$1"
  local checkpoint="$2"
  local latest=""
  if [[ "$checkpoint" != "auto" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  latest="$(
    find "$model_dir" -maxdepth 1 -type f -name 'model_registrar-*.pt' -printf '%f\n' 2>/dev/null \
      | sed -E 's/model_registrar-([0-9]+)\.pt/\1/' \
      | sort -n \
      | tail -1
  )"
  if [[ -z "$latest" ]]; then
    echo "No model_registrar-*.pt checkpoint found in $model_dir" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local model_dir
  local checkpoint_resolved
  local data_pkl="$TRAJECTRONPP_ROOT/experiments/processed/${dataset}_${split}.pkl"
  local run_dir="$OUTPUT_ROOT/${RUN_ID}_${dataset}_seed${seed}"
  local bundle="$run_dir/${dataset}_${split}_trajectronpp_k${K}.pt"
  local output_json="$run_dir/${dataset}_${split}_trajectronpp_afc.json"
  local max_scene_args=()
  local max_record_args=()
  local afc_max_args=()

  model_dir="$(resolve_model_dir "$dataset" "$seed")"
  checkpoint_resolved="$(resolve_checkpoint "$model_dir" "$CHECKPOINT")"
  mkdir -p "$run_dir"
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t max_record_args < <(optional_arg "--max-records" "$MAX_RECORDS")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")

  if [[ "$FORCE" == "1" || ! -f "$bundle" ]]; then
    log "EXPORT dataset=$dataset seed=$seed split=$split model=$model_dir checkpoint=$checkpoint_resolved bundle=$bundle"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.export_trajectronpp_predictions \
      --trajectron-root "$TRAJECTRONPP_ROOT" \
      --model-dir "$model_dir" \
      --checkpoint "$checkpoint_resolved" \
      --data "$data_pkl" \
      --subset "$dataset" \
      --split "$split" \
      --k "$K" \
      --seed "$seed" \
      --device "$DEVICE" \
      "${max_scene_args[@]}" \
      "${max_record_args[@]}" \
      --output-bundle "$bundle" 2>&1 | tee "$LOG_ROOT/export_${dataset}_seed${seed}_${split}.log"
  else
    log "SKIP export existing bundle=$bundle"
  fi

  if [[ "$FORCE" == "1" || ! -f "$output_json" ]]; then
    log "EVAL_AFC dataset=$dataset seed=$seed split=$split output=$output_json"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.evaluate_trajectronpp_afc \
      --bundle "$bundle" \
      --subset "$dataset" \
      --split "$split" \
      --data-root "$DATA_ROOT" \
      --branch-name "trajectronpp${K}_pred" \
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

log "START Trajectron++ AFC experiment"
log "TRAJECTRONPP_ROOT=$TRAJECTRONPP_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS K=$K CHECKPOINT=$CHECKPOINT"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  log "SUMMARY"
  "$PY" -m trustmoe_traj.scripts.summarize_trajectronpp_afc \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$DATASETS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --branch-name "trajectronpp${K}_pred" 2>&1 | tee "$LOG_ROOT/summary.log"
fi

log "DONE Trajectron++ AFC experiment"
