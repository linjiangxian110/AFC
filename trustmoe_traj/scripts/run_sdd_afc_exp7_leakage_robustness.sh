#!/usr/bin/env bash
set -euo pipefail

# Run SDD AFC Experiment 7 leakage/robustness checks.  Each setting reuses the
# SDD Exp1 diagnostic with a different AFC-bank filtering/randomization option.

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-/mnt/data/lck/code/moflow/moflow_venv/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_exp7_leakage_robustness_seed0}"

DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_exp7_leakage_robustness/$RUN_ID"
if [[ -n "${SDD_EXP7_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_EXP7_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_exp7_leakage_robustness/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD Exp7; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi

DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_EXP7_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_EXP7_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD Exp7; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

SETTINGS="${SETTINGS:-default_train_bank same_source_filtered temporal_gap_filtered same_source_temporal_filtered scene_exclusion randomized_bank}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
MAX_RECORDS="${MAX_RECORDS:-1000}"
AFC_MAX_TRAIN_RECORDS="${AFC_MAX_TRAIN_RECORDS:-3000}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
TEMPORAL_GAP_FRAMES="${TEMPORAL_GAP_FRAMES:-80}"
RANDOMIZE_BANK_SEED="${RANDOMIZE_BANK_SEED:-12345}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest_exp7.log" >&2
}

space_to_comma() {
  local raw="$1"
  raw="${raw// /,}"
  printf '%s\n' "$raw"
}

run_setting() {
  local setting="$1"
  local source_metadata=0
  local source_field="source_file"
  local filter_same_source=0
  local temporal_gap=0
  local randomize_seed=""

  case "$setting" in
    default_train_bank)
      ;;
    same_source_filtered)
      source_metadata=1
      filter_same_source=1
      ;;
    temporal_gap_filtered)
      source_metadata=1
      temporal_gap="$TEMPORAL_GAP_FRAMES"
      ;;
    same_source_temporal_filtered)
      source_metadata=1
      filter_same_source=1
      temporal_gap="$TEMPORAL_GAP_FRAMES"
      ;;
    scene_exclusion)
      source_metadata=1
      source_field="scene_id"
      filter_same_source=1
      ;;
    randomized_bank)
      randomize_seed="$RANDOMIZE_BANK_SEED"
      ;;
    *)
      echo "[ERR] unsupported Exp7 setting: $setting" >&2
      return 1
      ;;
  esac

  log "RUN setting=$setting"
  (
    export RUN_ID="${RUN_ID}_${setting}"
    export SDD_EXP1_OUTPUT_ROOT="$OUTPUT_ROOT"
    export SDD_EXP1_LOG_ROOT="$LOG_ROOT"
    export OUTPUT_ROOT="$OUTPUT_ROOT"
    export LOG_ROOT="$LOG_ROOT"
    export MAIN PY GPU SEEDS SPLITS MAX_RECORDS AFC_MAX_TRAIN_RECORDS AFC_TOP_M AFC_EPS FORCE
    export NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-train_split}"
    export SLOW_POOL_KS="${SLOW_POOL_KS:-20 100}"
    export KEEP_K="${KEEP_K:-20}"
    export DISABLE_RANDOM_POOL_SELECTION=1
    export DISABLE_RANDOM_SPREAD=1
    export DISABLE_CV_LINEAR=0
    export AFC_FEATURE_VARIANT="${AFC_FEATURE_VARIANT:-full_past_social}"
    export AFC_USE_SOURCE_METADATA="$source_metadata"
    export AFC_SOURCE_ID_FIELD="$source_field"
    export AFC_FILTER_SAME_SOURCE="$filter_same_source"
    export AFC_TEMPORAL_GAP_FRAMES="$temporal_gap"
    export AFC_RANDOMIZE_BANK_SEED="$randomize_seed"
    bash "$MAIN/trustmoe_traj/scripts/run_sdd_afc_exp1.sh"
  ) 2>&1 | tee "$LOG_ROOT/${RUN_ID}_${setting}.log"
}

log "START SDD Exp7 leakage/robustness"
log "MAIN=$MAIN"
log "PY=$PY"
log "GPU=$GPU"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "RUN_ID=$RUN_ID"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_ROOT=$LOG_ROOT"
log "SETTINGS=$SETTINGS"
log "SEEDS=$SEEDS SPLITS=$SPLITS MAX_RECORDS=$MAX_RECORDS AFC_MAX_TRAIN_RECORDS=$AFC_MAX_TRAIN_RECORDS"
log "AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS TEMPORAL_GAP_FRAMES=$TEMPORAL_GAP_FRAMES RANDOMIZE_BANK_SEED=$RANDOMIZE_BANK_SEED"

cd "$MAIN"
pwd | tee -a "$LOG_ROOT/manifest_exp7.log"

for setting in $SETTINGS; do
  run_setting "$setting"
done

mkdir -p "$OUTPUT_ROOT/analysis"
"$PY" -m trustmoe_traj.scripts.summarize_afc_exp7_leakage_robustness \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/analysis" \
  --run-id "$RUN_ID" \
  --datasets sdd \
  --seeds "$(space_to_comma "$SEEDS")" \
  --splits "$(space_to_comma "$SPLITS")" \
  --settings "$(space_to_comma "$SETTINGS")" \
  --file-template "{input_root}/{run_id}_{setting}_sdd_seed{seed}/sdd_{split}_headroom.json" \
  2>&1 | tee "$LOG_ROOT/summary_exp7.log"

sed -n '1,220p' "$OUTPUT_ROOT/analysis/afc_exp7_leakage_robustness_summary.md"
log "DONE SDD Exp7 leakage/robustness. OUTPUT_ROOT=$OUTPUT_ROOT"
echo "===== HAVE_FINISHED ${RUN_ID} status=success have finished! ====="
