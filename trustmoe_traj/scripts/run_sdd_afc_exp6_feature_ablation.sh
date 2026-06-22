#!/usr/bin/env bash
set -euo pipefail

# Run SDD AFC Experiment 6 feature ablation from an existing MoFlow SDD slow
# checkpoint.  The output layout is fixed for the Exp6 summarizer.

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-/mnt/data/lck/code/moflow/moflow_venv/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_exp6_feature_ablation_seed0}"

DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_exp6_feature_ablation/$RUN_ID"
if [[ -n "${SDD_EXP6_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_EXP6_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_exp6_feature_ablation/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD Exp6; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi

DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_EXP6_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_EXP6_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD Exp6; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

FEATURE_VARIANTS="${FEATURE_VARIANTS:-past_shape past_velocity past_velocity_accel past_velocity_social full_past_social}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-test}"
MAX_RECORDS="${MAX_RECORDS:-1000}"
AFC_MAX_TRAIN_RECORDS="${AFC_MAX_TRAIN_RECORDS:-3000}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest_exp6.log" >&2
}

space_to_comma() {
  local raw="$1"
  raw="${raw// /,}"
  printf '%s\n' "$raw"
}

log "START SDD Exp6 feature ablation"
log "MAIN=$MAIN"
log "PY=$PY"
log "GPU=$GPU"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "RUN_ID=$RUN_ID"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_ROOT=$LOG_ROOT"
log "FEATURE_VARIANTS=$FEATURE_VARIANTS"
log "SEEDS=$SEEDS SPLITS=$SPLITS MAX_RECORDS=$MAX_RECORDS AFC_MAX_TRAIN_RECORDS=$AFC_MAX_TRAIN_RECORDS"
log "AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS"

cd "$MAIN"
pwd | tee -a "$LOG_ROOT/manifest_exp6.log"

for feature in $FEATURE_VARIANTS; do
  log "RUN feature_variant=$feature"
  (
    export RUN_ID="${RUN_ID}_${feature}"
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
    export AFC_FEATURE_VARIANT="$feature"
    bash "$MAIN/trustmoe_traj/scripts/run_sdd_afc_exp1.sh"
  ) 2>&1 | tee "$LOG_ROOT/${RUN_ID}_${feature}.log"
done

mkdir -p "$OUTPUT_ROOT/analysis"
"$PY" -m trustmoe_traj.scripts.summarize_afc_exp6_feature_ablation \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/analysis" \
  --run-id "$RUN_ID" \
  --datasets sdd \
  --seeds "$(space_to_comma "$SEEDS")" \
  --splits "$(space_to_comma "$SPLITS")" \
  --feature-variants "$(space_to_comma "$FEATURE_VARIANTS")" \
  --file-template "{input_root}/{run_id}_{feature_variant}_sdd_seed{seed}/sdd_{split}_headroom.json" \
  2>&1 | tee "$LOG_ROOT/summary_exp6.log"

sed -n '1,220p' "$OUTPUT_ROOT/analysis/afc_exp6_feature_ablation_summary.md"
log "DONE SDD Exp6 feature ablation. OUTPUT_ROOT=$OUTPUT_ROOT"
echo "===== HAVE_FINISHED ${RUN_ID} status=success have finished! ====="
