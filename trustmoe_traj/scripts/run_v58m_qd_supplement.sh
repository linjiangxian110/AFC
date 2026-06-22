#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate V58M front3 quality-diversity supplement branches. This script
# does not retrain models; it bootstraps slow/refiner/quality paths from
# existing p95 V58M eval JSONs and refreshes eval JSONs with cluster metrics,
# raw quality-only branch, and GT-oriented oracle/collapsed references.

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"

DATASETS="${DATASETS:-eth hotel zara1}"
THRESHOLDS="${THRESHOLDS:-0.95 0.97 0.99}"
SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-val test}"

SOURCE_ROOT="${SOURCE_ROOT:-$MAIN/trustmoe_traj/analysis/v58_slot_quality_scorer_models/20260601_cross_subset_v58m_front3_pareto_adefde_rot6}"
SEARCH_ROOT="${SEARCH_ROOT:-$MAIN/trustmoe_traj/analysis}"
SOURCE_JSON_TEMPLATE="${SOURCE_JSON_TEMPLATE:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_v58m_qd_supplement}"
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
ENABLE_AFC="${ENABLE_AFC:-0}"
AFC_TRAIN_SPLIT="${AFC_TRAIN_SPLIT:-train}"
AFC_TOP_M="${AFC_TOP_M:-20}"
AFC_EPS="${AFC_EPS:-0.5,1.0}"
AFC_MAX_TRAIN_SCENES="${AFC_MAX_TRAIN_SCENES:-}"
AFC_BATCH_SCENES="${AFC_BATCH_SCENES:-64}"
ENABLE_ANCHOR_QD="${ENABLE_ANCHOR_QD:-0}"
ANCHOR_QD_ALPHA="${ANCHOR_QD_ALPHA:-1.0}"
ANCHOR_QD_BETA="${ANCHOR_QD_BETA:-0.5}"
ANCHOR_QD_RESIDUAL_PENALTY="${ANCHOR_QD_RESIDUAL_PENALTY:-0.05}"
ANCHOR_QD_MARGIN="${ANCHOR_QD_MARGIN:-0.0}"
ANCHOR_QD_TAU="${ANCHOR_QD_TAU:-1.0}"
ANCHOR_QD_ANCHOR_K="${ANCHOR_QD_ANCHOR_K:-4}"
ANCHOR_QD_ANCHOR_MIN_PROB="${ANCHOR_QD_ANCHOR_MIN_PROB:-}"
ANCHOR_QD_DIVERSITY_MIN_PROB="${ANCHOR_QD_DIVERSITY_MIN_PROB:-0.35}"
ANCHOR_QD_BASE_QUALITY="${ANCHOR_QD_BASE_QUALITY:-0.5}"
ANCHOR_QD_MAX_RESIDUAL_L2="${ANCHOR_QD_MAX_RESIDUAL_L2:-0.0}"

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
print(template.replace("{dataset}", dataset).replace("{ds}", dataset).replace("{tag}", tag))
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
  local afc_args=()
  local anchor_qd_args=()

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

  if [[ "$ENABLE_AFC" == "1" ]]; then
    afc_args=(
      --enable-afc
      --afc-train-split "$AFC_TRAIN_SPLIT"
      --afc-top-m "$AFC_TOP_M"
      --afc-eps "$AFC_EPS"
      --afc-batch-scenes "$AFC_BATCH_SCENES"
    )
    if [[ -n "$AFC_MAX_TRAIN_SCENES" ]]; then
      afc_args+=(--afc-max-train-scenes "$AFC_MAX_TRAIN_SCENES")
    fi
  fi
  if [[ "$ENABLE_ANCHOR_QD" == "1" ]]; then
    anchor_qd_args=(
      --enable-anchor-qd
      --anchor-qd-alpha "$ANCHOR_QD_ALPHA"
      --anchor-qd-beta "$ANCHOR_QD_BETA"
      --anchor-qd-residual-penalty "$ANCHOR_QD_RESIDUAL_PENALTY"
      --anchor-qd-margin "$ANCHOR_QD_MARGIN"
      --anchor-qd-tau "$ANCHOR_QD_TAU"
      --anchor-qd-anchor-k "$ANCHOR_QD_ANCHOR_K"
      --anchor-qd-diversity-min-prob "$ANCHOR_QD_DIVERSITY_MIN_PROB"
      --anchor-qd-base-quality "$ANCHOR_QD_BASE_QUALITY"
      --anchor-qd-max-residual-l2 "$ANCHOR_QD_MAX_RESIDUAL_L2"
    )
    if [[ -n "$ANCHOR_QD_ANCHOR_MIN_PROB" ]]; then
      anchor_qd_args+=(--anchor-qd-anchor-min-prob "$ANCHOR_QD_ANCHOR_MIN_PROB")
    fi
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
    "${afc_args[@]}" \
    "${anchor_qd_args[@]}" \
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
  local branches
  local summary_json="$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.json"
  local summary_txt="$OUTPUT_ROOT/${run_prefix}_${eval_prefix}_summary.txt"
  local log_file="$LOG_ROOT/${run_prefix}_${eval_prefix}_summary.log"

  seeds_csv="$(space_to_comma "$SEEDS")"
  splits_csv="$(space_to_comma "$SPLITS")"
  branches="slow_pred ${eval_prefix}_20_pred ${eval_prefix}_raw_quality20_pred ${eval_prefix}_raw_quality_global20_pred ${eval_prefix}_slots0to2_oracle20_pred ${eval_prefix}_slots0to2_global_oracle20_pred ${eval_prefix}_per_base_oracle20_pred ${eval_prefix}_global_oracle20_pred ${eval_prefix}_full160_pred"

  log "SUMMARY dataset=$dataset tag=$tag run_prefix=$run_prefix eval_prefix=$eval_prefix"
  python -m trustmoe_traj.scripts.summarize_v58_slot_quality_scorer \
    --input-root "$OUTPUT_ROOT" \
    --run-prefix "$run_prefix" \
    --eval-file-prefix "$eval_prefix" \
    --seeds "$seeds_csv" \
    --splits "$splits_csv" \
    --branches "$branches" \
    --output-json "$summary_json" \
    --output-txt "$summary_txt" \
    2>&1 | tee "$log_file"
}

main() {
  cd "$MAIN"
  log "START V58M quality-diversity supplement"
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
  log "ENABLE_AFC=$ENABLE_AFC"
  if [[ "$ENABLE_AFC" == "1" ]]; then
    log "AFC_TRAIN_SPLIT=$AFC_TRAIN_SPLIT"
    log "AFC_TOP_M=$AFC_TOP_M"
    log "AFC_EPS=$AFC_EPS"
    log "AFC_MAX_TRAIN_SCENES=$AFC_MAX_TRAIN_SCENES"
    log "AFC_BATCH_SCENES=$AFC_BATCH_SCENES"
  fi
  log "ENABLE_ANCHOR_QD=$ENABLE_ANCHOR_QD"
  if [[ "$ENABLE_ANCHOR_QD" == "1" ]]; then
    log "ANCHOR_QD_ALPHA=$ANCHOR_QD_ALPHA"
    log "ANCHOR_QD_BETA=$ANCHOR_QD_BETA"
    log "ANCHOR_QD_RESIDUAL_PENALTY=$ANCHOR_QD_RESIDUAL_PENALTY"
    log "ANCHOR_QD_MARGIN=$ANCHOR_QD_MARGIN"
    log "ANCHOR_QD_TAU=$ANCHOR_QD_TAU"
    log "ANCHOR_QD_ANCHOR_K=$ANCHOR_QD_ANCHOR_K"
    log "ANCHOR_QD_ANCHOR_MIN_PROB=$ANCHOR_QD_ANCHOR_MIN_PROB"
    log "ANCHOR_QD_DIVERSITY_MIN_PROB=$ANCHOR_QD_DIVERSITY_MIN_PROB"
    log "ANCHOR_QD_BASE_QUALITY=$ANCHOR_QD_BASE_QUALITY"
    log "ANCHOR_QD_MAX_RESIDUAL_L2=$ANCHOR_QD_MAX_RESIDUAL_L2"
  fi

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

  log "ANALYZE quality-diversity supplement"
  python -m trustmoe_traj.scripts.analyze_v58m_qd_supplement \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/analysis" \
    --datasets "$DATASETS" \
    --threshold-tags "$(for threshold in $THRESHOLDS; do threshold_tag "$threshold"; done | tr '\n' ' ')" \
    --split test \
    --run-prefix-template "$RUN_PREFIX_TEMPLATE" \
    --eval-prefix-template "$EVAL_PREFIX_TEMPLATE" \
    2>&1 | tee "$LOG_ROOT/analysis_qd_supplement.log"

  if [[ "$ENABLE_AFC" == "1" ]]; then
    log "ANALYZE AFC MVP supplement"
    python -m trustmoe_traj.scripts.analyze_v58m_afc_mvp \
      --input-root "$OUTPUT_ROOT" \
      --output-dir "$OUTPUT_ROOT/analysis" \
      --datasets "$DATASETS" \
      --threshold-tags "$(for threshold in $THRESHOLDS; do threshold_tag "$threshold"; done | tr '\n' ' ')" \
      --split test \
      --run-prefix-template "$RUN_PREFIX_TEMPLATE" \
      --eval-prefix-template "$EVAL_PREFIX_TEMPLATE" \
      2>&1 | tee "$LOG_ROOT/analysis_afc_mvp.log"
  fi

  log "DONE V58M quality-diversity supplement"
  log "analysis_md=$OUTPUT_ROOT/analysis/v58m_qd_supplement_summary.md"
  log "analysis_csv=$OUTPUT_ROOT/analysis/v58m_qd_supplement_summary.csv"
  if [[ "$ENABLE_AFC" == "1" ]]; then
    log "afc_analysis_md=$OUTPUT_ROOT/analysis/v58m_afc_mvp_summary.md"
    log "afc_analysis_csv=$OUTPUT_ROOT/analysis/v58m_afc_mvp_summary.csv"
  fi
}

main "$@"
