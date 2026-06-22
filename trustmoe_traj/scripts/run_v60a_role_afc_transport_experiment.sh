#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate V60A GT-decoupled role-aware AFC residual transport.
# The script intentionally resolves slow/data paths from existing V58M eval JSONs,
# while the new V60A refiner and quality scorer checkpoints are passed explicitly.

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
DATASETS="${DATASETS:-eth}"
SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-val test}"
THRESHOLDS="${THRESHOLDS:-0.95 0.97 0.99}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_v60a_role_afc_transport}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/v60_role_afc_models/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
CACHE_ROOT="${CACHE_ROOT:-$MAIN/trustmoe_traj/analysis/teacher_student_cache}"
SOURCE_JSON_ROOT="${SOURCE_JSON_ROOT:-$MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models}"
SOURCE_JSON_ROOTS="${SOURCE_JSON_ROOTS:-$SOURCE_JSON_ROOT $MAIN/trustmoe_traj/analysis/eval_results $MAIN/trustmoe_traj/analysis}"

RESIDUAL_SLOTS="${RESIDUAL_SLOTS:-8}"
CANDIDATE_SLOTS="${CANDIDATE_SLOTS:-0,1,2,3,4,5,6,7}"
TRAIN_SLOTS="${TRAIN_SLOTS:-1,2,3,4,5,6,7}"
KEEP_K="${KEEP_K:-20}"
ROTATE_TIME_FRAME="${ROTATE_TIME_FRAME:-6}"
BATCH_SCENES="${BATCH_SCENES:-8}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-8192}"
LATENCY_RUNS="${LATENCY_RUNS:-1}"
LOG_EVERY="${LOG_EVERY:-20}"

EPOCHS="${EPOCHS:-80}"
MAX_ITEMS="${MAX_ITEMS:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
QUALITY_EPOCHS="${QUALITY_EPOCHS:-40}"
MAX_SCENES="${MAX_SCENES:-}"
MAX_VAL_SCENES="${MAX_VAL_SCENES:-}"

V60_ANCHOR_MODES="${V60_ANCHOR_MODES:-4}"
V60_AFC_TOP_M="${V60_AFC_TOP_M:-20}"
V60_AFC_CLUSTERS="${V60_AFC_CLUSTERS:-8}"
V60_AFC_MAX_BANK_ITEMS="${V60_AFC_MAX_BANK_ITEMS:-0}"
V60_ROLE_TEMPERATURE="${V60_ROLE_TEMPERATURE:-0.05}"
V60_DYNAMIC_SLOT_OFFSET_SCALE="${V60_DYNAMIC_SLOT_OFFSET_SCALE:-0.75}"
V60_DYNAMIC_SLOT_HIDDEN_DIM="${V60_DYNAMIC_SLOT_HIDDEN_DIM:-128}"
LAMBDA_V60_ANCHOR_KEEP="${LAMBDA_V60_ANCHOR_KEEP:-2.0}"
LAMBDA_V60_AFC_ROLE="${LAMBDA_V60_AFC_ROLE:-1.5}"
LAMBDA_V60_BASE_IDENTITY="${LAMBDA_V60_BASE_IDENTITY:-0.03}"
LAMBDA_V60_SPREAD_FLOOR="${LAMBDA_V60_SPREAD_FLOOR:-3.0}"
LAMBDA_V60_RISK="${LAMBDA_V60_RISK:-0.1}"
LAMBDA_V60_RESIDUAL="${LAMBDA_V60_RESIDUAL:-0.002}"
V60_BASE_IDENTITY_MARGIN="${V60_BASE_IDENTITY_MARGIN:-1.0}"
V60_SPREAD_FLOOR_ENDPOINT_RATIO="${V60_SPREAD_FLOOR_ENDPOINT_RATIO:-1.02}"
V60_SPREAD_FLOOR_TRAJECTORY_RATIO="${V60_SPREAD_FLOOR_TRAJECTORY_RATIO:-1.02}"
V60_VELOCITY_DELTA_MAX="${V60_VELOCITY_DELTA_MAX:-0.45}"
V60_ACCEL_DELTA_MAX="${V60_ACCEL_DELTA_MAX:-0.45}"

AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
ANCHOR_QD_SELECTION_MODE="${ANCHOR_QD_SELECTION_MODE:-role_transport}"
ANCHOR_QD_ALPHA="${ANCHOR_QD_ALPHA:-0.7}"
ANCHOR_QD_BETA="${ANCHOR_QD_BETA:-1.0}"
ANCHOR_QD_COVERAGE_WEIGHT="${ANCHOR_QD_COVERAGE_WEIGHT:-0.8}"
ANCHOR_QD_COVERAGE_CLUSTERS="${ANCHOR_QD_COVERAGE_CLUSTERS:-6}"
ANCHOR_QD_RESIDUAL_PENALTY="${ANCHOR_QD_RESIDUAL_PENALTY:-0.05}"
ANCHOR_QD_MARGIN="${ANCHOR_QD_MARGIN:-0.0}"
ANCHOR_QD_TAU="${ANCHOR_QD_TAU:-1.0}"
ANCHOR_QD_ANCHOR_K="${ANCHOR_QD_ANCHOR_K:-4}"
ANCHOR_QD_DIVERSITY_MIN_PROB="${ANCHOR_QD_DIVERSITY_MIN_PROB:-0.25}"
ANCHOR_QD_BASE_QUALITY="${ANCHOR_QD_BASE_QUALITY:-0.5}"
ANCHOR_QD_MAX_RESIDUAL_L2="${ANCHOR_QD_MAX_RESIDUAL_L2:-0.0}"
ANCHOR_QD_SPREAD_FLOOR_ENDPOINT_RATIO="${ANCHOR_QD_SPREAD_FLOOR_ENDPOINT_RATIO:-0.98}"
ANCHOR_QD_SPREAD_FLOOR_TRAJECTORY_RATIO="${ANCHOR_QD_SPREAD_FLOOR_TRAJECTORY_RATIO:-0.98}"

TRAIN_REFINER="${TRAIN_REFINER:-1}"
TRAIN_QUALITY="${TRAIN_QUALITY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

threshold_tag() {
  "$PY" - "$1" <<'PY'
import sys
print(f"p{int(round(float(sys.argv[1]) * 100.0)):02d}")
PY
}

threshold_tags_csv() {
  local tags=()
  local threshold
  for threshold in $THRESHOLDS; do
    tags+=("$(threshold_tag "$threshold")")
  done
  local joined
  joined="$(printf ',%s' "${tags[@]}")"
  printf '%s\n' "${joined:1}"
}

space_to_comma() {
  local raw="$1"
  raw="${raw// /,}"
  printf '%s\n' "$raw"
}

find_cache() {
  local dataset="$1"
  local found
  found="$(find "$CACHE_ROOT" -type f -name "*${dataset}*train*temporal*.pt" -print 2>/dev/null | sort | tail -1)"
  if [[ -z "$found" ]]; then
    found="$(find "$CACHE_ROOT" -type f -name "*${dataset}*train*teacher_student_predictions*.pt" -print 2>/dev/null | sort | tail -1)"
  fi
  printf '%s\n' "$found"
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
    found="$(find "$root" -type f \
      \( -path "*${dataset}*seed0/*p95_test.json" -o -path "*${dataset}*seed0/*p95_val.json" \) \
      -print 2>/dev/null | sort | tail -1)"
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
  printf '%s\n' "$found"
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

optional_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    printf '%s\n%s\n' "$flag" "$value"
  fi
}

run_dataset_seed() {
  local dataset="$1"
  local seed="$2"
  local run_prefix="${RUN_ID}_${dataset}"
  local run_dir="$OUTPUT_ROOT/${run_prefix}_seed${seed}"
  local refiner_dir="$run_dir/refiner"
  local refiner_ckpt="$refiner_dir/v60a_refiner_best.pt"
  local quality_ckpt="$run_dir/quality/v58_slot_quality_scorer_best.pt"
  local cache_path
  local source_json
  local paths
  local slow_cfg
  local slow_ckpt
  local data_root

  mkdir -p "$run_dir" "$refiner_dir"
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

  cache_path="$(find_cache "$dataset")"
  if [[ -z "$cache_path" ]]; then
    log "ERROR: cannot find train cache for dataset=$dataset under $CACHE_ROOT"
    return 1
  fi

  log "dataset=$dataset seed=$seed source_json=$source_json"
  log "cache=$cache_path"
  log "slow_cfg=$slow_cfg"
  log "slow_ckpt=$slow_ckpt"
  log "data_root=$data_root"

  if [[ "$TRAIN_REFINER" == "1" && ( "$FORCE" == "1" || ! -f "$refiner_ckpt" ) ]]; then
    local max_items_args=()
    mapfile -t max_items_args < <(optional_arg "--max-items" "$MAX_ITEMS")
    log "TRAIN V60A refiner dataset=$dataset seed=$seed"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.train_social_cvae_refiner \
      --variant v60a \
      --cache-path "$cache_path" \
      --output-dir "$refiner_dir" \
      --run-name v60a_refiner \
      --device "$DEVICE" \
      --seed "$seed" \
      "${max_items_args[@]}" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --use-energy-risk-map \
      --use-temporal-energy-encoder \
      --decoder-layers 3 \
      --set-residual-slots "$RESIDUAL_SLOTS" \
      --eval-z-mode slots \
      --dynamic-slot-hidden-dim "$V60_DYNAMIC_SLOT_HIDDEN_DIM" \
      --dynamic-slot-offset-scale "$V60_DYNAMIC_SLOT_OFFSET_SCALE" \
      --v60-anchor-modes "$V60_ANCHOR_MODES" \
      --v60-afc-top-m "$V60_AFC_TOP_M" \
      --v60-afc-clusters "$V60_AFC_CLUSTERS" \
      --v60-afc-max-bank-items "$V60_AFC_MAX_BANK_ITEMS" \
      --v60-role-temperature "$V60_ROLE_TEMPERATURE" \
      --lambda-v60-anchor-keep "$LAMBDA_V60_ANCHOR_KEEP" \
      --lambda-v60-afc-role "$LAMBDA_V60_AFC_ROLE" \
      --lambda-v60-base-identity "$LAMBDA_V60_BASE_IDENTITY" \
      --lambda-v60-spread-floor "$LAMBDA_V60_SPREAD_FLOOR" \
      --lambda-v60-risk "$LAMBDA_V60_RISK" \
      --lambda-v60-residual "$LAMBDA_V60_RESIDUAL" \
      --v60-base-identity-margin "$V60_BASE_IDENTITY_MARGIN" \
      --v60-spread-floor-endpoint-ratio "$V60_SPREAD_FLOOR_ENDPOINT_RATIO" \
      --v60-spread-floor-trajectory-ratio "$V60_SPREAD_FLOOR_TRAJECTORY_RATIO" \
      --v59-velocity-delta-max "$V60_VELOCITY_DELTA_MAX" \
      --v59-accel-delta-max "$V60_ACCEL_DELTA_MAX" \
      --log-every "$LOG_EVERY" 2>&1 | tee "$LOG_ROOT/${run_prefix}_seed${seed}_refiner.log"
  else
    log "SKIP refiner train dataset=$dataset seed=$seed ckpt=$refiner_ckpt"
  fi

  if [[ "$TRAIN_QUALITY" == "1" && ( "$FORCE" == "1" || ! -f "$quality_ckpt" ) ]]; then
    local max_scene_args=()
    local max_val_scene_args=()
    mapfile -t max_scene_args < <(optional_arg "--max-scenes" "$MAX_SCENES")
    mapfile -t max_val_scene_args < <(optional_arg "--max-val-scenes" "$MAX_VAL_SCENES")
    log "TRAIN V60A quality scorer dataset=$dataset seed=$seed"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.train_v58_slot_quality_scorer \
      --protocol official_align \
      --subset "$dataset" \
      --train-split train \
      --val-split val \
      --data-root "$data_root" \
      --sample-mode per_agent \
      --batch-scenes "$BATCH_SCENES" \
      "${max_scene_args[@]}" \
      "${max_val_scene_args[@]}" \
      --device "$DEVICE" \
      --seed "$seed" \
      --rotate \
      --rotate-time-frame "$ROTATE_TIME_FRAME" \
      --num-to-gen 1 \
      --slow-cfg-path "$slow_cfg" \
      --slow-checkpoint "$slow_ckpt" \
      --refiner-checkpoint "$refiner_ckpt" \
      --residual-slots "$RESIDUAL_SLOTS" \
      --train-slots "$TRAIN_SLOTS" \
      --training-mode two_stage_replacement \
      --rank-label-metric ade_fde \
      --accept-improve-mode pareto \
      --accept-require-improve-slow \
      --include-index-features \
      --epochs "$QUALITY_EPOCHS" \
      --batch-size "$QUALITY_BATCH_SIZE" \
      --output-dir "$run_dir" \
      --run-name quality \
      --log-every "$LOG_EVERY" 2>&1 | tee "$LOG_ROOT/${run_prefix}_seed${seed}_quality.log"
  else
    log "SKIP quality train dataset=$dataset seed=$seed ckpt=$quality_ckpt"
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    local split
    local threshold
    for threshold in $THRESHOLDS; do
      local tag
      local eval_prefix
      tag="$(threshold_tag "$threshold")"
      eval_prefix="v60a_${dataset}_role_afc_transport_${tag}"
      for split in $SPLITS; do
        local eval_json="$run_dir/${eval_prefix}_${split}.json"
        if [[ "$FORCE" != "1" && -f "$eval_json" ]]; then
          log "SKIP eval dataset=$dataset seed=$seed split=$split threshold=$threshold json=$eval_json"
          continue
        fi
        local afc_max_args=()
        mapfile -t afc_max_args < <(optional_arg "--afc-max-train-scenes" "$AFC_MAX_TRAIN_SCENES")
        log "EVAL V60A dataset=$dataset seed=$seed split=$split threshold=$threshold"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.eval_v58_slot_quality_scorer \
          --protocol official_align \
          --subset "$dataset" \
          --split "$split" \
          --data-root "$data_root" \
          --sample-mode per_agent \
          --batch-scenes "$BATCH_SCENES" \
          --device "$DEVICE" \
          --seed "$seed" \
          --rotate \
          --rotate-time-frame "$ROTATE_TIME_FRAME" \
          --num-to-gen 1 \
          --latency-runs "$LATENCY_RUNS" \
          --slow-cfg-path "$slow_cfg" \
          --slow-checkpoint "$slow_ckpt" \
          --refiner-checkpoint "$refiner_ckpt" \
          --quality-checkpoint "$quality_ckpt" \
          --residual-slots "$RESIDUAL_SLOTS" \
          --keep-k "$KEEP_K" \
          --candidate-slots "$CANDIDATE_SLOTS" \
          --diagnostic-prefix "$eval_prefix" \
          --branch-name "${eval_prefix}_20_pred" \
          --selection-mode two_stage_replacement \
          --accept-prob-threshold "$threshold" \
          --oracle-select-metric ade_fde \
          --output-json "$eval_json" \
          --enable-afc \
          --afc-top-m "$AFC_TOP_M" \
          --afc-eps "$AFC_EPS" \
          --afc-batch-scenes "$AFC_BATCH_SCENES" \
          "${afc_max_args[@]}" \
          --enable-anchor-qd \
          --anchor-qd-selection-mode "$ANCHOR_QD_SELECTION_MODE" \
          --anchor-qd-alpha "$ANCHOR_QD_ALPHA" \
          --anchor-qd-beta "$ANCHOR_QD_BETA" \
          --anchor-qd-coverage-weight "$ANCHOR_QD_COVERAGE_WEIGHT" \
          --anchor-qd-coverage-clusters "$ANCHOR_QD_COVERAGE_CLUSTERS" \
          --anchor-qd-residual-penalty "$ANCHOR_QD_RESIDUAL_PENALTY" \
          --anchor-qd-margin "$ANCHOR_QD_MARGIN" \
          --anchor-qd-tau "$ANCHOR_QD_TAU" \
          --anchor-qd-anchor-k "$ANCHOR_QD_ANCHOR_K" \
          --anchor-qd-diversity-min-prob "$ANCHOR_QD_DIVERSITY_MIN_PROB" \
          --anchor-qd-base-quality "$ANCHOR_QD_BASE_QUALITY" \
          --anchor-qd-max-residual-l2 "$ANCHOR_QD_MAX_RESIDUAL_L2" \
          --anchor-qd-spread-floor-endpoint-ratio "$ANCHOR_QD_SPREAD_FLOOR_ENDPOINT_RATIO" \
          --anchor-qd-spread-floor-trajectory-ratio "$ANCHOR_QD_SPREAD_FLOOR_TRAJECTORY_RATIO" 2>&1 | tee "$LOG_ROOT/${run_prefix}_seed${seed}_${eval_prefix}_${split}.log"
      done
    done
  fi
}

summarize_dataset() {
  local dataset="$1"
  local run_prefix="${RUN_ID}_${dataset}"
  local threshold
  for threshold in $THRESHOLDS; do
    local tag
    local eval_prefix
    tag="$(threshold_tag "$threshold")"
    eval_prefix="v60a_${dataset}_role_afc_transport_${tag}"
    log "SUMMARY dataset=$dataset tag=$tag"
    "$PY" -m trustmoe_traj.scripts.summarize_v58_slot_quality_scorer \
      --project-root "$MAIN" \
      --input-root "$OUTPUT_ROOT" \
      --run-prefix "$run_prefix" \
      --eval-file-prefix "$eval_prefix" \
      --seeds "$(space_to_comma "$SEEDS")" \
      --splits "$(space_to_comma "$SPLITS")" \
      --output-json "$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.json" \
      --output-txt "$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.txt" 2>&1 | tee "$LOG_ROOT/${run_prefix}_${eval_prefix}_summary.log"
  done
}

log "START V60A role-aware AFC transport experiment"
log "MAIN=$MAIN"
log "PY=$PY"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "SOURCE_JSON_ROOTS=$SOURCE_JSON_ROOTS"
log "DATASETS=$DATASETS SEEDS=$SEEDS SPLITS=$SPLITS THRESHOLDS=$THRESHOLDS GPU=$GPU"

cd "$MAIN"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    run_dataset_seed "$dataset" "$seed"
  done
  if [[ "$RUN_SUMMARY" == "1" ]]; then
    summarize_dataset "$dataset"
  fi
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  mkdir -p "$OUTPUT_ROOT/analysis"
  log "ANALYZE AFC/QD summaries"
  "$PY" -m trustmoe_traj.scripts.analyze_v58m_afc_mvp \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --datasets "$(space_to_comma "$DATASETS")" \
    --threshold-tags "$(threshold_tags_csv)" \
    --run-prefix-template "${RUN_ID}_{dataset}" \
    --eval-prefix-template "v60a_{dataset}_role_afc_transport_{tag}" 2>&1 | tee "$LOG_ROOT/analysis_afc_mvp.log"
  "$PY" -m trustmoe_traj.scripts.analyze_v58m_qd_supplement \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --datasets "$(space_to_comma "$DATASETS")" \
    --threshold-tags "$(threshold_tags_csv)" \
    --run-prefix-template "${RUN_ID}_{dataset}" \
    --eval-prefix-template "v60a_{dataset}_role_afc_transport_{tag}" 2>&1 | tee "$LOG_ROOT/analysis_qd_supplement.log"
fi

log "DONE. OUTPUT_ROOT=$OUTPUT_ROOT"
