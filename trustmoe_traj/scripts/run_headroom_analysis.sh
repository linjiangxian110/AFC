#!/usr/bin/env bash
set -euo pipefail

# Run AFC/QD headroom diagnostics. No model is trained here.
# The script resolves slow/data paths from existing eval JSONs and optionally
# attaches an existing residual refiner checkpoint if one can be found.

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
SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-test}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_headroom_analysis}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/headroom_analysis/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
SOURCE_JSON_ROOT="${SOURCE_JSON_ROOT:-$MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models}"
SOURCE_JSON_ROOTS="${SOURCE_JSON_ROOTS:-$SOURCE_JSON_ROOT $MAIN/trustmoe_traj/analysis/eval_results $MAIN/trustmoe_traj/analysis}"
SOURCE_ALLOW_SEED0_FALLBACK="${SOURCE_ALLOW_SEED0_FALLBACK:-1}"
SOURCE_ALLOW_GENERIC_FALLBACK="${SOURCE_ALLOW_GENERIC_FALLBACK:-1}"
REFINER_SEARCH_ROOTS="${REFINER_SEARCH_ROOTS:-$MAIN/trustmoe_traj/analysis/v60_role_afc_models $MAIN/trustmoe_traj/analysis/v59_anchor_afc_models $MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models}"

KEEP_K="${KEEP_K:-20}"
SLOW_POOL_KS="${SLOW_POOL_KS:-20 50 100}"
RESIDUAL_SLOTS="${RESIDUAL_SLOTS:-8}"
USE_REFINER="${USE_REFINER:-1}"
ROTATE_TIME_FRAME="${ROTATE_TIME_FRAME:-6}"
BATCH_SCENES="${BATCH_SCENES:-8}"
LATENCY_RUNS="${LATENCY_RUNS:-1}"
LOG_EVERY="${LOG_EVERY:-20}"
MAX_SCENES="${MAX_SCENES:-}"
ORACLE_SELECT_METRIC="${ORACLE_SELECT_METRIC:-ade_fde}"
DISABLE_RANDOM_POOL_SELECTION="${DISABLE_RANDOM_POOL_SELECTION:-0}"
RANDOM_POOL_TRIALS="${RANDOM_POOL_TRIALS:-1}"
RANDOM_POOL_EMIT_TRIALS="${RANDOM_POOL_EMIT_TRIALS:-0}"
DISABLE_CV_LINEAR="${DISABLE_CV_LINEAR:-0}"
DISABLE_RANDOM_SPREAD="${DISABLE_RANDOM_SPREAD:-0}"
RANDOM_SPREAD_SOURCE="${RANDOM_SPREAD_SOURCE:-slow_radial}"
RANDOM_SPREAD_ENDPOINT_SCALE="${RANDOM_SPREAD_ENDPOINT_SCALE:-2.0}"
RANDOM_SPREAD_ENDPOINT_SCALES="${RANDOM_SPREAD_ENDPOINT_SCALES:-}"
RANDOM_SPREAD_NOISE_SCALE="${RANDOM_SPREAD_NOISE_SCALE:-0.05}"

AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.5,1.0}"
AFC_SELECTION_TAU="${AFC_SELECTION_TAU:-1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"

RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

space_to_comma() {
  local raw="$1"
  raw="${raw// /,}"
  printf '%s\n' "$raw"
}

optional_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    printf '%s\n%s\n' "$flag" "$value"
  fi
}

find_source_json() {
  local dataset="$1"
  local seed="$2"
  local root
  local found
  for root in $SOURCE_JSON_ROOTS; do
    [[ -d "$root" ]] || continue
    found="$(find "$root" -type f \
      \( -path "*${dataset}*seed${seed}/*p95_test.json" -o -path "*${dataset}*seed${seed}/*p95_val.json" \) \
      -print 2>/dev/null | sort | tail -1)"
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
    if [[ "$SOURCE_ALLOW_SEED0_FALLBACK" == "1" && "$seed" != "0" ]]; then
      found="$(find "$root" -type f \
        \( -path "*${dataset}*seed0/*p95_test.json" -o -path "*${dataset}*seed0/*p95_val.json" \) \
        -print 2>/dev/null | sort | tail -1)"
      if [[ -n "$found" ]]; then
        log "WARN: using seed0 source eval JSON fallback for dataset=$dataset requested_seed=$seed source=$found"
        printf '%s\n' "$found"
        return 0
      fi
    fi
    if [[ "$SOURCE_ALLOW_GENERIC_FALLBACK" == "1" ]]; then
      found="$(find "$root" -type f \
        \( -path "*/official_eval/${dataset}_slow_fast_test.json" \
        -o -path "*/official_eval/${dataset}_slow_fast_val.json" \
        -o -path "*${dataset}*slow_fast_test.json" \
        -o -path "*${dataset}*slow_fast_val.json" \) \
        -print 2>/dev/null | sort | tail -1)"
      if [[ -n "$found" ]]; then
        log "WARN: using generic slow/base source eval JSON for dataset=$dataset seed=$seed source=$found"
        printf '%s\n' "$found"
        return 0
      fi
    fi
  done
  printf '%s\n' ""
}

load_eval_paths() {
  local source_json="$1"
  "$PY" - "$source_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
args = payload.get("args", {})
dataset = payload.get("dataset", {})

def first(*values):
    for value in values:
        if value:
            return str(value)
    return ""

print(first(args.get("slow_cfg_path")))
print(first(payload.get("slow_checkpoint"), args.get("slow_checkpoint")))
print(first(args.get("data_root"), dataset.get("data_root")))
PY
}

find_refiner_ckpt() {
  local dataset="$1"
  local seed="$2"
  local root
  local found
  for root in $REFINER_SEARCH_ROOTS; do
    [[ -d "$root" ]] || continue
    found="$(find "$root" -type f \
      \( -path "*${dataset}*seed${seed}*/refiner/*refiner_best.pt" -o -path "*${dataset}*seed${seed}*/*refiner_best.pt" \) \
      -print 2>/dev/null | sort | tail -1)"
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
  printf '%s\n' ""
}

run_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local run_prefix="${RUN_ID}_${dataset}"
  local run_dir="$OUTPUT_ROOT/${run_prefix}_seed${seed}"
  local output_json="$run_dir/${dataset}_${split}_headroom.json"
  local source_json
  local paths
  local slow_cfg
  local slow_ckpt
  local data_root
  local refiner_ckpt

  mkdir -p "$run_dir"
  if [[ "$FORCE" != "1" && -f "$output_json" ]]; then
    log "SKIP dataset=$dataset seed=$seed split=$split json=$output_json"
    return 0
  fi

  source_json="$(find_source_json "$dataset" "$seed")"
  if [[ -z "$source_json" ]]; then
    log "ERROR: cannot find p95 source eval JSON for dataset=$dataset seed=$seed under roots: $SOURCE_JSON_ROOTS"
    return 1
  fi
  mapfile -t paths < <(load_eval_paths "$source_json")
  slow_cfg="${paths[0]}"
  slow_ckpt="${paths[1]}"
  data_root="${paths[2]}"
  if [[ -z "$slow_cfg" || -z "$slow_ckpt" || -z "$data_root" ]]; then
    log "ERROR: incomplete slow/data paths from $source_json"
    return 1
  fi
  refiner_ckpt=""
  if [[ "$USE_REFINER" == "1" ]]; then
    refiner_ckpt="$(find_refiner_ckpt "$dataset" "$seed")"
    if [[ -z "$refiner_ckpt" ]]; then
      log "WARN: no refiner checkpoint found for dataset=$dataset seed=$seed; running slow-pool headroom only"
    fi
  fi

  local max_scene_args=()
  local afc_max_args=()
  local refiner_args=()
  local experiment1_args=()
  local random_spread_scales_arg=()
  mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")
  if [[ -n "$refiner_ckpt" ]]; then
    refiner_args=(--refiner-checkpoint "$refiner_ckpt" --residual-slots "$RESIDUAL_SLOTS")
  fi
  if [[ "$DISABLE_RANDOM_POOL_SELECTION" == "1" ]]; then
    experiment1_args+=(--disable-random-pool-selection)
  fi
  if [[ "$DISABLE_CV_LINEAR" == "1" ]]; then
    experiment1_args+=(--disable-cv-linear)
  fi
  if [[ "$DISABLE_RANDOM_SPREAD" == "1" ]]; then
    experiment1_args+=(--disable-random-spread)
  fi
  if [[ "$RANDOM_POOL_EMIT_TRIALS" == "1" ]]; then
    experiment1_args+=(--random-pool-emit-trials)
  fi
  mapfile -t random_spread_scales_arg < <(optional_arg "--random-spread-endpoint-scales" "$RANDOM_SPREAD_ENDPOINT_SCALES")

  log "HEADROOM dataset=$dataset seed=$seed split=$split source_json=$source_json"
  log "slow_cfg=$slow_cfg"
  log "slow_ckpt=$slow_ckpt"
  log "data_root=$data_root"
  log "refiner=${refiner_ckpt:-none}"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.diagnose_headroom_analysis \
    --protocol official_align \
    --subset "$dataset" \
    --split "$split" \
    --data-root "$data_root" \
    --sample-mode per_agent \
    --batch-scenes "$BATCH_SCENES" \
    "${max_scene_args[@]}" \
    --device "$DEVICE" \
    --seed "$seed" \
    --rotate \
    --rotate-time-frame "$ROTATE_TIME_FRAME" \
    --num-to-gen 1 \
    --latency-runs "$LATENCY_RUNS" \
    --slow-cfg-path "$slow_cfg" \
    --slow-checkpoint "$slow_ckpt" \
    --keep-k "$KEEP_K" \
    --slow-pool-ks "$(space_to_comma "$SLOW_POOL_KS")" \
    --oracle-select-metric "$ORACLE_SELECT_METRIC" \
    --afc-selection-tau "$AFC_SELECTION_TAU" \
    --random-pool-trials "$RANDOM_POOL_TRIALS" \
    --random-spread-source "$RANDOM_SPREAD_SOURCE" \
    --random-spread-endpoint-scale "$RANDOM_SPREAD_ENDPOINT_SCALE" \
    "${random_spread_scales_arg[@]}" \
    --random-spread-noise-scale "$RANDOM_SPREAD_NOISE_SCALE" \
    --afc-top-m "$AFC_TOP_M" \
    --afc-eps "$AFC_EPS" \
    --afc-batch-scenes "$AFC_BATCH_SCENES" \
    "${afc_max_args[@]}" \
    "${experiment1_args[@]}" \
    "${refiner_args[@]}" \
    --log-every "$LOG_EVERY" \
    --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/${run_prefix}_seed${seed}_${split}_headroom.log"
}

log "START headroom analysis"
log "MAIN=$MAIN"
log "PY=$PY"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "SOURCE_JSON_ROOTS=$SOURCE_JSON_ROOTS"
log "SOURCE_ALLOW_SEED0_FALLBACK=$SOURCE_ALLOW_SEED0_FALLBACK"
log "SOURCE_ALLOW_GENERIC_FALLBACK=$SOURCE_ALLOW_GENERIC_FALLBACK"
log "REFINER_SEARCH_ROOTS=$REFINER_SEARCH_ROOTS"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS SLOW_POOL_KS=$SLOW_POOL_KS GPU=$GPU"
log "AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS AFC_SELECTION_TAU=$AFC_SELECTION_TAU"
log "EXPERIMENT1 controls random_pool_disabled=$DISABLE_RANDOM_POOL_SELECTION random_pool_trials=$RANDOM_POOL_TRIALS random_pool_emit_trials=$RANDOM_POOL_EMIT_TRIALS cv_disabled=$DISABLE_CV_LINEAR random_spread_disabled=$DISABLE_RANDOM_SPREAD random_spread_source=$RANDOM_SPREAD_SOURCE random_spread_endpoint_scale=$RANDOM_SPREAD_ENDPOINT_SCALE random_spread_endpoint_scales=$RANDOM_SPREAD_ENDPOINT_SCALES"

cd "$MAIN"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for split in $SPLITS; do
      run_one "$dataset" "$seed" "$split"
    done
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  mkdir -p "$OUTPUT_ROOT/analysis"
  log "SUMMARIZE headroom analysis"
  "$PY" -m trustmoe_traj.scripts.summarize_headroom_analysis \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "$(space_to_comma "$DATASETS")" \
    --seeds "$(space_to_comma "$SEEDS")" \
    --splits "$(space_to_comma "$SPLITS")" 2>&1 | tee "$LOG_ROOT/summary_headroom.log"
fi

log "DONE. OUTPUT_ROOT=$OUTPUT_ROOT"
