#!/usr/bin/env bash
set -euo pipefail

# Run SDD Exp2 complementarity analysis from an existing SDD headroom summary.
# This is a second-stage analysis: it does not train or evaluate a model.

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /mnt/data/lck/code/moflow/moflow_venv/bin/python ]]; then
    PY=/mnt/data/lck/code/moflow/moflow_venv/bin/python
  else
    PY=python
  fi
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_exp2_complementarity_seed0}"
INPUT_CSV="${INPUT_CSV:-$MAIN/trustmoe_traj/analysis/sdd_exp1/20260617_sdd_afc_exp1_seed0/analysis/headroom_summary.csv}"
PLOT_NAME_SUFFIX="${PLOT_NAME_SUFFIX:-sdd_seed0}"
MIN_POINTS="${MIN_POINTS:-3}"

DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_exp2/$RUN_ID"
if [[ -n "${SDD_EXP2_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_EXP2_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_exp2/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD Exp2; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi

DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_EXP2_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_EXP2_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD Exp2; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

mkdir -p "$OUTPUT_ROOT/analysis" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

log "START SDD Exp2 complementarity analysis"
log "MAIN=$MAIN"
log "PY=$PY"
log "RUN_ID=$RUN_ID"
log "INPUT_CSV=$INPUT_CSV"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_ROOT=$LOG_ROOT"

cd "$MAIN"
pwd | tee -a "$LOG_ROOT/manifest.log"

if [[ ! -f "$INPUT_CSV" ]]; then
  log "ERROR: INPUT_CSV does not exist: $INPUT_CSV"
  exit 1
fi

"$PY" -m trustmoe_traj.scripts.analyze_afc_exp2_complementarity \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_ROOT/analysis" \
  --run-id "$RUN_ID" \
  --plot-name-suffix "$PLOT_NAME_SUFFIX" \
  --min-points "$MIN_POINTS" 2>&1 | tee "$LOG_ROOT/analyze.log"

summary="$OUTPUT_ROOT/analysis/afc_exp2_complementarity_summary.md"
if [[ -f "$summary" ]]; then
  sed -n '1,220p' "$summary"
else
  log "ERROR: missing summary: $summary"
  exit 1
fi

log "DONE SDD Exp2 complementarity analysis. OUTPUT_ROOT=$OUTPUT_ROOT"
echo "===== HAVE_FINISHED ${RUN_ID} status=success ====="
