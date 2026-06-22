#!/usr/bin/env bash
set -euo pipefail

# Run SDD Exp3 dense-K sampling headroom analysis.
# This wrapper reuses the SDD headroom diagnostic but fixes the output family,
# dense slow-pool K grid, and summary lookup paths for Exp3.

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
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_exp3_densek_seed0}"

DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_exp3_densek/$RUN_ID"
if [[ -n "${SDD_EXP3_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_EXP3_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_exp3_densek/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD Exp3; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi

DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_EXP3_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_EXP3_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD Exp3; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

export MAIN PY GPU OUTPUT_ROOT LOG_ROOT
export SDD_EXP1_OUTPUT_ROOT="$OUTPUT_ROOT"
export SDD_EXP1_LOG_ROOT="$LOG_ROOT"
export SEEDS="${SEEDS:-0}"
export SPLITS="${SPLITS:-test}"
export KEEP_K="${KEEP_K:-20}"
export SLOW_POOL_KS="${SLOW_POOL_KS:-20 40 60 80 100 150 200}"
export AFC_TOP_M="${AFC_TOP_M:-20}"
export AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
export NORMALIZATION_SOURCE="${NORMALIZATION_SOURCE:-train_split}"
export BATCH_RECORDS="${BATCH_RECORDS:-64}"
export AFC_BATCH_RECORDS="${AFC_BATCH_RECORDS:-256}"
export RANDOM_POOL_TRIALS="${RANDOM_POOL_TRIALS:-1}"
export RANDOM_POOL_EMIT_TRIALS="${RANDOM_POOL_EMIT_TRIALS:-0}"
export RANDOM_SPREAD_ENDPOINT_SCALES="${RANDOM_SPREAD_ENDPOINT_SCALES:-}"
export DISABLE_RANDOM_POOL_SELECTION="${DISABLE_RANDOM_POOL_SELECTION:-1}"
export DISABLE_RANDOM_SPREAD="${DISABLE_RANDOM_SPREAD:-1}"
export DISABLE_CV_LINEAR="${DISABLE_CV_LINEAR:-1}"
export RUN_SUMMARY="${RUN_SUMMARY:-1}"
export FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest_exp3.log" >&2
}

log "START SDD Exp3 dense-K sampling headroom"
log "MAIN=$MAIN"
log "PY=$PY"
log "GPU=$GPU"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "RUN_ID=$RUN_ID"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_ROOT=$LOG_ROOT"
log "SEEDS=$SEEDS SPLITS=$SPLITS KEEP_K=$KEEP_K SLOW_POOL_KS=$SLOW_POOL_KS"
log "MAX_RECORDS=${MAX_RECORDS:-all} AFC_MAX_TRAIN_RECORDS=${AFC_MAX_TRAIN_RECORDS:-all}"
log "AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS"
log "DISABLE_RANDOM_POOL_SELECTION=$DISABLE_RANDOM_POOL_SELECTION DISABLE_RANDOM_SPREAD=$DISABLE_RANDOM_SPREAD DISABLE_CV_LINEAR=$DISABLE_CV_LINEAR"

cd "$MAIN"
pwd | tee -a "$LOG_ROOT/manifest_exp3.log"

bash "$MAIN/trustmoe_traj/scripts/run_sdd_afc_exp1.sh" 2>&1 | tee "$LOG_ROOT/run_sdd_exp3_densek.log"

summary="$OUTPUT_ROOT/analysis/headroom_summary.md"
if [[ -f "$summary" ]]; then
  sed -n '1,220p' "$summary"
else
  log "ERROR: missing summary: $summary"
  exit 1
fi

log "DONE SDD Exp3 dense-K sampling headroom. OUTPUT_ROOT=$OUTPUT_ROOT"
echo "===== HAVE_FINISHED ${RUN_ID} status=success ====="
