#!/usr/bin/env bash
set -euo pipefail

# P0 SDD AFC infrastructure check. This script does not train a model.

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /mnt/data/lck/code/moflow/moflow_venv/bin/python ]]; then
    PY=/mnt/data/lck/code/moflow/moflow_venv/bin/python
  else
    PY=python
  fi
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_sdd_afc_p0_check}"
SDD_DATA_ROOT="${SDD_DATA_ROOT:-$MAIN/MoFlow/data/sdd}"
OLD_SDD_DATA_ROOT="${OLD_SDD_DATA_ROOT:-/mnt/data/lck/code/moflow/MoFlow/data/sdd}"
DEFAULT_OUTPUT_ROOT="$MAIN/trustmoe_traj/analysis/sdd_p0/$RUN_ID"
if [[ -n "${SDD_P0_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$SDD_P0_OUTPUT_ROOT"
elif [[ -n "${OUTPUT_ROOT:-}" && "$OUTPUT_ROOT" != *"/sdd_p0/$RUN_ID"* ]]; then
  echo "[WARN] Ignoring stale OUTPUT_ROOT=$OUTPUT_ROOT for SDD P0; using $DEFAULT_OUTPUT_ROOT" >&2
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
fi
DEFAULT_LOG_ROOT="$OUTPUT_ROOT/logs"
if [[ -n "${SDD_P0_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SDD_P0_LOG_ROOT"
elif [[ -n "${LOG_ROOT:-}" && "$LOG_ROOT" != "$DEFAULT_LOG_ROOT" ]]; then
  echo "[WARN] Ignoring stale LOG_ROOT=$LOG_ROOT for SDD P0; using $DEFAULT_LOG_ROOT" >&2
  LOG_ROOT="$DEFAULT_LOG_ROOT"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi

MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-200}"
MAX_RECORDS="${MAX_RECORDS:-50}"
BATCH_SCENES="${BATCH_SCENES:-64}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-256}"
K="${K:-20}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.3,0.5,1.0}"
REQUIRE_SMOKE="${REQUIRE_SMOKE:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
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

require_args=()
if [[ "$REQUIRE_SMOKE" == "1" ]]; then
  require_args+=(--require-smoke)
fi

log "START SDD AFC P0 check"
log "MAIN=$MAIN"
log "PY=$PY"
log "RUN_ID=$RUN_ID"
log "SDD_DATA_ROOT=$SDD_DATA_ROOT"
log "OLD_SDD_DATA_ROOT=$OLD_SDD_DATA_ROOT"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "MAX_TRAIN_SCENES=$MAX_TRAIN_SCENES MAX_RECORDS=$MAX_RECORDS AFC_TOP_M=$AFC_TOP_M AFC_EPS=$AFC_EPS"

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

"$PY" -m trustmoe_traj.scripts.check_sdd_afc_p0 \
  --data-root "$SDD_DATA_ROOT" \
  --run-id "$RUN_ID" \
  --output-json "$OUTPUT_ROOT/sdd_afc_p0_check.json" \
  --max-train-scenes "$MAX_TRAIN_SCENES" \
  --max-records "$MAX_RECORDS" \
  --batch-scenes "$BATCH_SCENES" \
  --afc-batch-scenes "$AFC_BATCH_SCENES" \
  --k "$K" \
  --afc-top-m "$AFC_TOP_M" \
  --afc-eps "$AFC_EPS" \
  "${require_args[@]}" \
  2>&1 | tee "$LOG_ROOT/sdd_afc_p0_check.log"

log "DONE SDD AFC P0 check output=$OUTPUT_ROOT/sdd_afc_p0_check.json"
