#!/usr/bin/env bash
set -euo pipefail

# Run SDD Exp1 controlled diagnostic.  This script does not train a model; it
# evaluates an existing MoFlow SDD slow checkpoint under the AFC protocol.

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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_afc_exp1_seed0}"
SDD_DATA_ROOT="${SDD_DATA_ROOT:-$MAIN/MoFlow/data/sdd}"
OLD_SDD_DATA_ROOT="${OLD_SDD_DATA_ROOT:-/mnt/data/lck/code/moflow/MoFlow/data/sdd}"

DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_exp1/$RUN_ID"
if [[ -n "${SDD_EXP1_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_EXP1_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_exp1/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD Exp1; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi
DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_EXP1_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_EXP1_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD Exp1; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
KEEP_K="${KEEP_K:-20}"
SLOW_POOL_KS="${SLOW_POOL_KS:-20 50 100 200}"
BATCH_RECORDS="${BATCH_RECORDS:-64}"
MAX_RECORDS="${MAX_RECORDS:-}"
NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-auto}"
NORMALIZATION_MAX_TRAIN_RECORDS="${NORMALIZATION_MAX_TRAIN_RECORDS:-}"
ROTATE="${ROTATE:-1}"
ROTATE_TIME_FRAME="${ROTATE_TIME_FRAME:-6}"
LATENCY_RUNS="${LATENCY_RUNS:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
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
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
AFC_FEATURE_VARIANT="${AFC_FEATURE_VARIANT:-full_past_social}"
AFC_SELECTION_TAU="${AFC_SELECTION_TAU:-1.0}"
AFC_MAX_TRAIN_RECORDS="${AFC_MAX_TRAIN_RECORDS:-}"
AFC_BATCH_RECORDS="${AFC_BATCH_RECORDS:-256}"
AFC_USE_SOURCE_METADATA="${AFC_USE_SOURCE_METADATA:-0}"
AFC_SOURCE_ID_FIELD="${AFC_SOURCE_ID_FIELD:-source_file}"
AFC_FILTER_SAME_SOURCE="${AFC_FILTER_SAME_SOURCE:-0}"
AFC_TEMPORAL_GAP_FRAMES="${AFC_TEMPORAL_GAP_FRAMES:-0}"
AFC_RANDOMIZE_BANK_SEED="${AFC_RANDOMIZE_BANK_SEED:-}"
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

maybe_link_sdd_data() {
  local target_dir="$SDD_DATA_ROOT/original"
  local old_train="$OLD_SDD_DATA_ROOT/original/sdd_train.pkl"
  local old_test="$OLD_SDD_DATA_ROOT/original/sdd_test.pkl"
  mkdir -p "$target_dir"
  if [[ ! -e "$target_dir/sdd_train.pkl" && -f "$old_train" ]]; then
    ln -s "$old_train" "$target_dir/sdd_train.pkl"
    log "LINKED $target_dir/sdd_train.pkl -> $old_train"
  fi
  if [[ ! -e "$target_dir/sdd_test.pkl" && -f "$old_test" ]]; then
    ln -s "$old_test" "$target_dir/sdd_test.pkl"
    log "LINKED $target_dir/sdd_test.pkl -> $old_test"
  fi
}

find_slow_dir() {
  if [[ -n "${SLOW_DIR:-}" ]]; then
    printf '%s\n' "$SLOW_DIR"
    return 0
  fi
  local root
  local found
  for root in "$MAIN/MoFlow/results_sdd/cor_fm" "/mnt/data/lck/code/moflow/MoFlow/results_sdd/cor_fm"; do
    [[ -d "$root" ]] || continue
    found="$(find "$root" -maxdepth 1 -type d -name "*sdd*rot_6*" -print 2>/dev/null | sort | tail -1)"
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
  printf '%s\n' ""
}

resolve_slow_paths() {
  local slow_dir="$1"
  local ckpt="${SLOW_CHECKPOINT:-}"
  local cfg="${SLOW_CFG_PATH:-}"
  if [[ -z "$ckpt" ]]; then
    for candidate in \
      "$slow_dir/models/checkpoint_best.pt" \
      "$slow_dir/models/checkpoint_last.pt" \
      "$slow_dir/models/checkpoint_150.pt"; do
      if [[ -f "$candidate" ]]; then
        ckpt="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$ckpt" && -d "$slow_dir/models" ]]; then
    ckpt="$(find "$slow_dir/models" -maxdepth 1 -type f -name "checkpoint*.pt" -print 2>/dev/null | sort | tail -1)"
  fi
  if [[ -z "$cfg" ]]; then
    cfg="$(find "$slow_dir" -maxdepth 1 -type f -name "*_updated.yml" -print 2>/dev/null | sort | tail -1)"
  fi
  if [[ -z "$cfg" ]]; then
    cfg="$(find "$slow_dir" -maxdepth 1 -type f -name "*.yml" -print 2>/dev/null | sort | tail -1)"
  fi
  printf '%s\n%s\n' "$cfg" "$ckpt"
}

run_one() {
  local seed="$1"
  local split="$2"
  local run_prefix="${RUN_ID}_sdd"
  local run_dir="$OUTPUT_ROOT/${run_prefix}_seed${seed}"
  local output_json="$run_dir/sdd_${split}_headroom.json"
  local slow_dir
  local slow_cfg
  local slow_ckpt
  local paths

  mkdir -p "$run_dir"
  if [[ "$FORCE" != "1" && -f "$output_json" ]]; then
    log "SKIP seed=$seed split=$split json=$output_json"
    return 0
  fi

  slow_dir="$(find_slow_dir)"
  if [[ -z "$slow_dir" || ! -d "$slow_dir" ]]; then
    log "ERROR: cannot find SDD slow checkpoint directory. Set SLOW_DIR or SLOW_CHECKPOINT/SLOW_CFG_PATH explicitly."
    return 1
  fi
  mapfile -t paths < <(resolve_slow_paths "$slow_dir")
  slow_cfg="${paths[0]}"
  slow_ckpt="${paths[1]}"
  if [[ -z "$slow_cfg" || ! -f "$slow_cfg" ]]; then
    log "ERROR: cannot resolve SDD slow cfg under $slow_dir. Set SLOW_CFG_PATH."
    return 1
  fi
  if [[ -z "$slow_ckpt" || ! -f "$slow_ckpt" ]]; then
    log "ERROR: cannot resolve SDD slow checkpoint under $slow_dir. Set SLOW_CHECKPOINT."
    return 1
  fi

  local max_record_args=()
  local norm_max_args=()
  local afc_max_args=()
  local afc_randomize_args=()
  local control_args=()
  local random_spread_scales_arg=()
  local rotate_args=()

  mapfile -t max_record_args < <(optional_arg "--max-records" "$MAX_RECORDS")
  mapfile -t norm_max_args < <(optional_arg "--normalization-max-train-records" "$NORMALIZATION_MAX_TRAIN_RECORDS")
  mapfile -t afc_max_args < <(optional_arg "--afc-max-train-records" "$AFC_MAX_TRAIN_RECORDS")
  mapfile -t afc_randomize_args < <(optional_arg "--afc-randomize-bank-seed" "$AFC_RANDOMIZE_BANK_SEED")
  mapfile -t random_spread_scales_arg < <(optional_arg "--random-spread-endpoint-scales" "$RANDOM_SPREAD_ENDPOINT_SCALES")

  if [[ "$ROTATE" == "1" ]]; then
    rotate_args=(--rotate)
  fi
  if [[ "$DISABLE_RANDOM_POOL_SELECTION" == "1" ]]; then
    control_args+=(--disable-random-pool-selection)
  fi
  if [[ "$DISABLE_CV_LINEAR" == "1" ]]; then
    control_args+=(--disable-cv-linear)
  fi
  if [[ "$DISABLE_RANDOM_SPREAD" == "1" ]]; then
    control_args+=(--disable-random-spread)
  fi
  if [[ "$RANDOM_POOL_EMIT_TRIALS" == "1" ]]; then
    control_args+=(--random-pool-emit-trials)
  fi
  if [[ "$AFC_USE_SOURCE_METADATA" == "1" ]]; then
    control_args+=(--afc-use-source-metadata)
  fi
  if [[ "$AFC_FILTER_SAME_SOURCE" == "1" ]]; then
    control_args+=(--afc-filter-same-source)
  fi

  log "SDD_EXP1 seed=$seed split=$split"
  log "slow_dir=$slow_dir"
  log "slow_cfg=$slow_cfg"
  log "slow_ckpt=$slow_ckpt"
  log "output_json=$output_json"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m trustmoe_traj.scripts.diagnose_sdd_headroom_analysis \
    --split "$split" \
    --data-root "$SDD_DATA_ROOT" \
    --sample-mode per_scene \
    --agents 1 \
    --data-norm min_max \
    --normalization-source "$NORMALIZATION_SOURCE" \
    "${norm_max_args[@]}" \
    --batch-records "$BATCH_RECORDS" \
    "${max_record_args[@]}" \
    --device "$DEVICE" \
    --seed "$seed" \
    "${rotate_args[@]}" \
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
    --afc-feature-variant "$AFC_FEATURE_VARIANT" \
    --afc-batch-records "$AFC_BATCH_RECORDS" \
    "${afc_max_args[@]}" \
    --afc-source-id-field "$AFC_SOURCE_ID_FIELD" \
    --afc-temporal-gap-frames "$AFC_TEMPORAL_GAP_FRAMES" \
    "${afc_randomize_args[@]}" \
    "${control_args[@]}" \
    --log-every "$LOG_EVERY" \
    --output-json "$output_json" 2>&1 | tee "$LOG_ROOT/${run_prefix}_seed${seed}_${split}_headroom.log"
}

log "START SDD Exp1 AFC controlled diagnostic"
log "MAIN=$MAIN"
log "PY=$PY"
log "GPU=$GPU"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "DEVICE=$DEVICE"
log "RUN_ID=$RUN_ID"
log "SDD_DATA_ROOT=$SDD_DATA_ROOT"
log "OLD_SDD_DATA_ROOT=$OLD_SDD_DATA_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_ROOT=$LOG_ROOT"
log "SEEDS=$SEEDS SPLITS=$SPLITS KEEP_K=$KEEP_K SLOW_POOL_KS=$SLOW_POOL_KS"
log "MAX_RECORDS=${MAX_RECORDS:-all} AFC_MAX_TRAIN_RECORDS=${AFC_MAX_TRAIN_RECORDS:-all}"
log "AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS AFC_FEATURE_VARIANT=$AFC_FEATURE_VARIANT AFC_SELECTION_TAU=$AFC_SELECTION_TAU"
log "AFC_USE_SOURCE_METADATA=$AFC_USE_SOURCE_METADATA AFC_SOURCE_ID_FIELD=$AFC_SOURCE_ID_FIELD AFC_FILTER_SAME_SOURCE=$AFC_FILTER_SAME_SOURCE AFC_TEMPORAL_GAP_FRAMES=$AFC_TEMPORAL_GAP_FRAMES AFC_RANDOMIZE_BANK_SEED=${AFC_RANDOMIZE_BANK_SEED:-none}"
log "NORMALIZATION_SOURCE=$NORMALIZATION_SOURCE ROTATE=$ROTATE ROTATE_TIME_FRAME=$ROTATE_TIME_FRAME"

cd "$MAIN"
pwd | tee -a "$LOG_ROOT/manifest.log"
"$PY" - <<'PY' | tee -a "$LOG_ROOT/manifest.log"
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch_import_error", repr(exc))
PY

maybe_link_sdd_data

for seed in $SEEDS; do
  for split in $SPLITS; do
    run_one "$seed" "$split"
  done
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  mkdir -p "$OUTPUT_ROOT/analysis"
  log "SUMMARIZE SDD Exp1"
  "$PY" -m trustmoe_traj.scripts.summarize_headroom_analysis \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --run-id "$RUN_ID" \
    --datasets "sdd" \
    --seeds "$(space_to_comma "$SEEDS")" \
    --splits "$(space_to_comma "$SPLITS")" 2>&1 | tee "$LOG_ROOT/summary_headroom.log"
fi

log "DONE SDD Exp1. OUTPUT_ROOT=$OUTPUT_ROOT"
