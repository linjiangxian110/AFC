#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate V58M front3 pareto slot-quality scorers with stricter accept
# thresholds. The script bootstraps slow/refiner/quality paths from existing
# p95 eval JSON files, so it does not retrain any model.
#
# Typical usage on ciisr:
#   export MAIN=/mnt/data/lck/code/TrustMoE-Traj-v38
#   export GPU=0
#   bash "$MAIN/trustmoe_traj/scripts/run_v58m_threshold_sweep.sh"
#
# Common overrides:
#   DATASETS="hotel zara1" THRESHOLDS="0.97 0.99" bash ...
#   ALLOW_MISSING=1 bash ...

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"

DATASETS="${DATASETS:-eth hotel univ zara1 zara2}"
THRESHOLDS="${THRESHOLDS:-0.95 0.97 0.99}"
SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-val test}"

SOURCE_ROOT="${SOURCE_ROOT:-$MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models/20260601_cross_subset_v58m_front3_pareto_adefde_rot6}"
SEARCH_ROOT="${SEARCH_ROOT:-$MAIN/trustmoe_traj/analysis}"
SOURCE_JSON_TEMPLATE="${SOURCE_JSON_TEMPLATE:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_v58m_threshold_sweep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

if [[ ! -v RUN_PREFIX_TEMPLATE || -z "$RUN_PREFIX_TEMPLATE" ]]; then
  RUN_PREFIX_TEMPLATE='20260601_ciisr_{dataset}_v58m_front3_pareto_adefde_rot6'
fi
if [[ ! -v SOURCE_EVAL_PREFIX_TEMPLATE || -z "$SOURCE_EVAL_PREFIX_TEMPLATE" ]]; then
  SOURCE_EVAL_PREFIX_TEMPLATE='v58m_{dataset}_front3_pareto_rot6_p95'
fi
if [[ ! -v EVAL_PREFIX_TEMPLATE || -z "$EVAL_PREFIX_TEMPLATE" ]]; then
  EVAL_PREFIX_TEMPLATE='v58m_{dataset}_front3_pareto_rot6_{tag}'
fi

CANDIDATE_SLOTS="${CANDIDATE_SLOTS:-0,1,2}"
RESIDUAL_SLOTS="${RESIDUAL_SLOTS:-8}"
KEEP_K="${KEEP_K:-20}"
BATCH_SCENES="${BATCH_SCENES:-8}"
LATENCY_RUNS="${LATENCY_RUNS:-1}"
LOG_EVERY="${LOG_EVERY:-20}"
ORACLE_SELECT_METRIC="${ORACLE_SELECT_METRIC:-ade_fde}"
ROTATE_TIME_FRAME="${ROTATE_TIME_FRAME:-6}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

render_template() {
  local template="$1"
  local dataset="$2"
  local tag="$3"
  python - "$template" "$dataset" "$tag" <<'PY'
import sys

template, dataset, tag = sys.argv[1:4]
print(
    template
    .replace("{dataset}", dataset)
    .replace("{ds}", dataset)
    .replace("{tag}", tag)
)
PY
}

render_source_path() {
  local template="$1"
  local dataset="$2"
  local seed="$3"
  local split="$4"
  local run_prefix="$5"
  local eval_prefix="$6"
  python - "$template" "$dataset" "$seed" "$split" "$run_prefix" "$eval_prefix" <<'PY'
import sys

template, dataset, seed, split, run_prefix, eval_prefix = sys.argv[1:7]
print(
    template
    .replace("{dataset}", dataset)
    .replace("{ds}", dataset)
    .replace("{seed}", seed)
    .replace("{split}", split)
    .replace("{run_prefix}", run_prefix)
    .replace("{eval_prefix}", eval_prefix)
)
PY
}

threshold_tag() {
  python - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"p{int(round(value * 100.0)):02d}")
PY
}

space_to_comma() {
  local raw="$1"
  raw="${raw// /,}"
  printf '%s\n' "$raw"
}

find_source_json() {
  local dataset="$1"
  local seed="$2"
  local source_tag="p95"
  local run_prefix
  local eval_prefix
  local split
  local direct
  local found

  run_prefix="$(render_template "$RUN_PREFIX_TEMPLATE" "$dataset" "$source_tag")"
  eval_prefix="$(render_template "$SOURCE_EVAL_PREFIX_TEMPLATE" "$dataset" "$source_tag")"

  if [[ -n "$SOURCE_JSON_TEMPLATE" ]]; then
    for split in val test; do
      direct="$(render_source_path "$SOURCE_JSON_TEMPLATE" "$dataset" "$seed" "$split" "$run_prefix" "$eval_prefix")"
      if [[ "${DEBUG_SOURCE:-0}" == "1" ]]; then
        log "DEBUG source candidate dataset=$dataset seed=$seed split=$split path=$direct"
      fi
      if [[ -f "$direct" ]]; then
        printf '%s\n' "$direct"
        return 0
      fi
    done
  fi

  for split in val test; do
    direct="$SOURCE_ROOT/${run_prefix}_seed${seed}/${eval_prefix}_${split}.json"
    if [[ -f "$direct" ]]; then
      printf '%s\n' "$direct"
      return 0
    fi
  done

  found="$(find "$SOURCE_ROOT" "$SEARCH_ROOT" -type f \
    \( -path "*${run_prefix}_seed${seed}/*${eval_prefix}_*.json" \
       -o -name "*${dataset}*${eval_prefix}*seed${seed}*.json" \
       -o -path "*${dataset}*seed${seed}*/*${eval_prefix}_*.json" \) \
    -print 2>/dev/null | sort -u | head -1)"
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  python - "$SOURCE_ROOT" "$SEARCH_ROOT" "$run_prefix" "$eval_prefix" "$dataset" "$seed" <<'PY'
import sys
from pathlib import Path

roots = [Path(sys.argv[1]), Path(sys.argv[2])]
run_prefix = sys.argv[3]
eval_prefix = sys.argv[4]
dataset = sys.argv[5]
seed = sys.argv[6]
seed_token = f"seed{seed}"

candidates = []
seen = set()
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*.json"):
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        name = path.name
        parent = path.parent.name
        full = key
        if not (name.endswith("_val.json") or name.endswith("_test.json")):
            continue
        if eval_prefix not in name:
            continue
        exact_parent = f"{run_prefix}_{seed_token}" in parent
        loose_parent = dataset in full and seed_token in full
        if exact_parent or loose_parent:
            candidates.append(path)

if candidates:
    print(sorted(candidates, key=lambda p: p.as_posix())[0].as_posix())
PY
}

load_eval_paths() {
  local source_json="$1"
  python - "$source_json" <<'PY'
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
print(first(payload.get("refiner_checkpoint"), args.get("refiner_checkpoint")))
print(first(payload.get("quality_checkpoint"), args.get("quality_checkpoint")))
print(first(args.get("data_root"), dataset.get("data_root")))
PY
}

run_eval_one() {
  local dataset="$1"
  local seed="$2"
  local split="$3"
  local threshold="$4"
  local tag="$5"
  local source_json="$6"
  local run_prefix="$7"
  local eval_prefix="$8"
  local branch_name="${eval_prefix}_20_pred"
  local outdir="$OUTPUT_ROOT/${run_prefix}_seed${seed}"
  local output_json="$outdir/${eval_prefix}_${split}.json"
  local log_file="$LOG_ROOT/${run_prefix}_seed${seed}_${eval_prefix}_${split}.log"
  local paths
  local slow_cfg
  local slow_ckpt
  local refiner_ckpt
  local quality_ckpt
  local data_root

  mapfile -t paths < <(load_eval_paths "$source_json")
  slow_cfg="${paths[0]}"
  slow_ckpt="${paths[1]}"
  refiner_ckpt="${paths[2]}"
  quality_ckpt="${paths[3]}"
  data_root="${paths[4]}"

  if [[ -z "$slow_cfg" || -z "$slow_ckpt" || -z "$refiner_ckpt" || -z "$quality_ckpt" || -z "$data_root" ]]; then
    log "ERROR: incomplete paths from $source_json"
    return 1
  fi
  if [[ ! -f "$slow_cfg" || ! -f "$slow_ckpt" || ! -f "$refiner_ckpt" || ! -f "$quality_ckpt" ]]; then
    log "ERROR: missing checkpoint/cfg for dataset=$dataset seed=$seed source=$source_json"
    log "  slow_cfg=$slow_cfg"
    log "  slow_ckpt=$slow_ckpt"
    log "  refiner_ckpt=$refiner_ckpt"
    log "  quality_ckpt=$quality_ckpt"
    return 1
  fi

  mkdir -p "$outdir"
  log "EVAL dataset=$dataset seed=$seed split=$split threshold=$threshold tag=$tag source=$source_json"
  env CUDA_VISIBLE_DEVICES="$GPU" python -m trustmoe_traj.scripts.eval_v58_slot_quality_scorer \
    --protocol official_align \
    --subset "$dataset" \
    --split "$split" \
    --data-root "$data_root" \
    --sample-mode per_agent \
    --slow-cfg-path "$slow_cfg" \
    --slow-checkpoint "$slow_ckpt" \
    --refiner-checkpoint "$refiner_ckpt" \
    --quality-checkpoint "$quality_ckpt" \
    --residual-slots "$RESIDUAL_SLOTS" \
    --keep-k "$KEEP_K" \
    --candidate-slots "$CANDIDATE_SLOTS" \
    --diagnostic-prefix "$eval_prefix" \
    --branch-name "$branch_name" \
    --selection-mode auto \
    --accept-prob-threshold "$threshold" \
    --oracle-select-metric "$ORACLE_SELECT_METRIC" \
    --batch-scenes "$BATCH_SCENES" \
    --device "$DEVICE" \
    --seed "$seed" \
    --latency-runs "$LATENCY_RUNS" \
    --log-every "$LOG_EVERY" \
    --rotate \
    --rotate-time-frame "$ROTATE_TIME_FRAME" \
    --output-json "$output_json" \
    2>&1 | tee "$log_file"
}

summarize_one() {
  local dataset="$1"
  local tag="$2"
  local run_prefix="$3"
  local eval_prefix="$4"
  local seeds_csv
  local splits_csv
  local summary_json="$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.json"
  local summary_txt="$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.txt"
  local log_file="$LOG_ROOT/${run_prefix}_${eval_prefix}_summary.log"

  seeds_csv="$(space_to_comma "$SEEDS")"
  splits_csv="$(space_to_comma "$SPLITS")"

  log "SUMMARY dataset=$dataset tag=$tag run_prefix=$run_prefix eval_prefix=$eval_prefix"
  python -m trustmoe_traj.scripts.summarize_v58_slot_quality_scorer \
    --input-root "$OUTPUT_ROOT" \
    --run-prefix "$run_prefix" \
    --eval-file-prefix "$eval_prefix" \
    --seeds "$seeds_csv" \
    --splits "$splits_csv" \
    --branches "${eval_prefix}_20_pred" \
    --output-json "$summary_json" \
    --output-txt "$summary_txt" \
    2>&1 | tee "$log_file"
}

main() {
  cd "$MAIN"
  log "START V58M threshold sweep"
  log "MAIN=$MAIN"
  log "SOURCE_ROOT=$SOURCE_ROOT"
  log "SEARCH_ROOT=$SEARCH_ROOT"
  if [[ -n "$SOURCE_JSON_TEMPLATE" ]]; then
    log "SOURCE_JSON_TEMPLATE=$SOURCE_JSON_TEMPLATE"
  fi
  log "OUTPUT_ROOT=$OUTPUT_ROOT"
  log "DATASETS=$DATASETS"
  log "THRESHOLDS=$THRESHOLDS"
  log "SEEDS=$SEEDS"
  log "SPLITS=$SPLITS"

  local dataset
  local seed
  local split
  local threshold
  local tag
  local source_json
  local run_prefix
  local eval_prefix
  local missing_inputs=0

  log "PREFLIGHT source p95 eval JSONs"
  for dataset in $DATASETS; do
    for seed in $SEEDS; do
      source_json="$(find_source_json "$dataset" "$seed" || true)"
      if [[ -z "$source_json" || ! -f "$source_json" ]]; then
        log "MISSING source p95 eval JSON: dataset=$dataset seed=$seed under $SOURCE_ROOT"
        missing_inputs=1
      else
        log "FOUND source dataset=$dataset seed=$seed json=$source_json"
      fi
    done
  done

  if [[ "$missing_inputs" != "0" && "$ALLOW_MISSING" != "1" ]]; then
    log "ERROR: missing source inputs. Set ALLOW_MISSING=1 to skip unavailable datasets/seeds."
    exit 1
  fi

  for dataset in $DATASETS; do
    for seed in $SEEDS; do
      source_json="$(find_source_json "$dataset" "$seed" || true)"
      if [[ -z "$source_json" || ! -f "$source_json" ]]; then
        continue
      fi
      for threshold in $THRESHOLDS; do
        tag="$(threshold_tag "$threshold")"
        run_prefix="$(render_template "$RUN_PREFIX_TEMPLATE" "$dataset" "$tag")"
        eval_prefix="$(render_template "$EVAL_PREFIX_TEMPLATE" "$dataset" "$tag")"
        for split in $SPLITS; do
          run_eval_one "$dataset" "$seed" "$split" "$threshold" "$tag" "$source_json" "$run_prefix" "$eval_prefix"
        done
      done
    done

    for threshold in $THRESHOLDS; do
      tag="$(threshold_tag "$threshold")"
      run_prefix="$(render_template "$RUN_PREFIX_TEMPLATE" "$dataset" "$tag")"
      eval_prefix="$(render_template "$EVAL_PREFIX_TEMPLATE" "$dataset" "$tag")"
      summarize_one "$dataset" "$tag" "$run_prefix" "$eval_prefix"
    done
  done

  log "ANALYZE threshold sweep"
  python -m trustmoe_traj.scripts.analyze_v58m_threshold_sweep \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --datasets "$DATASETS" \
    --threshold-tags "$(for threshold in $THRESHOLDS; do threshold_tag "$threshold"; done | tr '\n' ' ')" \
    --splits "$SPLITS" \
    --run-prefix-template "$RUN_PREFIX_TEMPLATE" \
    --eval-prefix-template "$EVAL_PREFIX_TEMPLATE" \
    2>&1 | tee "$LOG_ROOT/analysis.log"

  log "DONE V58M threshold sweep"
  log "analysis_md=$OUTPUT_ROOT/analysis/v58m_threshold_sweep_summary.md"
  log "analysis_csv=$OUTPUT_ROOT/analysis/v58m_threshold_sweep_summary.csv"
}

main "$@"
