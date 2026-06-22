#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj}"
CACHE="${CACHE:-$MAIN/trustmoe_traj/analysis/teacher_student_cache/official_align_eth_train_teacher_student_predictions_teacher_temporal.pt}"
DATA_ROOT="${DATA_ROOT:-$MAIN/trustmoe_traj/data/ETH}"
SLOW_RUN_DIR="${SLOW_RUN_DIR:-$MAIN/MoFlow/results_eth_ucy/cor_fm/trustmoe_slow_eth_v1_retry_FM_S10_log_m-0.5_s1.5_dire_drop_emb_m0.5_k20.0_IS_TN_NN_A_REG_S_eth_rot_6_min_max_LR0.0001_WD0.01_CLS_1.0_BS32_EP150}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d)_ciisr}"
SEEDS="${SEEDS:-0,1,2}"
SPLITS="${SPLITS:-val,test}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BATCH_SCENES="${BATCH_SCENES:-2}"
SLOTS="${SLOTS:-4}"
RUN_DIAGNOSIS="${RUN_DIAGNOSIS:-1}"
RANDOM_TRIALS="${RANDOM_TRIALS:-50}"
LOG_ROOT="${LOG_ROOT:-trustmoe_traj/analysis/logs/v56c_sequential/$RUN_STAMP}"

mkdir -p "$LOG_ROOT"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a SPLIT_ARRAY <<< "$SPLITS"

run_suffix() {
  case "$1" in
    v56c1) echo "v56c1_qwta_slots${SLOTS}" ;;
    v56c2) echo "v56c2_qwta_slot0_slots${SLOTS}" ;;
    v56c3) echo "v56c3_qwta_gain_norm_slots${SLOTS}" ;;
    *) echo "unknown_${1}" ;;
  esac
}

train_variant() {
  local variant="$1"
  local suffix="$2"
  local run_prefix="${RUN_STAMP}_${suffix}"
  local -a variant_args

  case "$variant" in
    v56c1)
      variant_args=(
        --variant v56c1
        --lambda-elite-soft-wta 0.6
        --elite-soft-temperature 0.08
        --elite-base-topk 1
      )
      ;;
    v56c2)
      variant_args=(
        --variant v56c2
        --lambda-elite-soft-wta 0.6
        --elite-soft-temperature 0.08
        --elite-base-topk 1
        --lambda-slot0-preserve 0.25
      )
      ;;
    v56c3)
      variant_args=(
        --variant v56c3
        --lambda-elite-soft-wta 0.6
        --elite-soft-temperature 0.08
        --elite-base-topk 1
        --lambda-elite-improvement 0.8
        --elite-improvement-margin 0.02
        --lambda-residual-norm-band 0.05
        --residual-endpoint-norm-max 0.6
        --residual-trajectory-norm-max 0.35
      )
      ;;
    *)
      echo "Unsupported variant: $variant" >&2
      exit 2
      ;;
  esac

  echo "===== TRAIN $variant run_prefix=$run_prefix ====="
  for seed in "${SEED_ARRAY[@]}"; do
    local run="${run_prefix}_seed${seed}"
    python -m trustmoe_traj.scripts.train_social_cvae_refiner \
      "${variant_args[@]}" \
      --cache-path "$CACHE" \
      --output-dir "trustmoe_traj/analysis/social_cvae_refiner_models/$run" \
      --run-name social_cvae_refiner \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --hidden-dim 128 \
      --latent-dim 16 \
      --use-energy-risk-map \
      --use-temporal-energy-encoder \
      --decoder-layers 3 \
      --set-residual-slots "$SLOTS" \
      --eval-z-mode slots \
      --selection-metric fde_min \
      --lambda-recon-best 1.0 \
      --lambda-set-coverage 1.0 \
      --set-coverage-temperature 0.1 \
      --lambda-energy-recon 0.2 \
      --lambda-base-best-nohurt 2.0 \
      --lambda-good-nohurt 1.0 \
      --good-nohurt-frac 0.25 \
      --lambda-diversity-preserve 0.5 \
      --lambda-keep 0.05 \
      --lambda-delta-l2 0.01 \
      --lambda-slot-spread 0.05 \
      2>&1 | tee "$LOG_ROOT/${run}_train.log"
  done
}

eval_variant() {
  local variant="$1"
  local suffix="$2"
  local run_prefix="${RUN_STAMP}_${suffix}"
  local eval_prefix="${variant}_official"

  echo "===== EVAL $variant run_prefix=$run_prefix ====="
  for seed in "${SEED_ARRAY[@]}"; do
    local run="${run_prefix}_seed${seed}"
    local ref="trustmoe_traj/analysis/social_cvae_refiner_models/$run/social_cvae_refiner_best.pt"
    local outdir="trustmoe_traj/analysis/eval_results/$run"
    mkdir -p "$outdir"

    for split in "${SPLIT_ARRAY[@]}"; do
      python -m trustmoe_traj.scripts.eval_social_cvae_refiner \
        --protocol official_align \
        --subset eth \
        --split "$split" \
        --data-root "$DATA_ROOT" \
        --sample-mode per_agent \
        --slow-cfg-path "$SLOW_RUN_DIR/cor_fm_updated.yml" \
        --slow-checkpoint "$SLOW_RUN_DIR/models/checkpoint_best.pt" \
        --refiner-checkpoint "$ref" \
        --num-residual-samples "$SLOTS" \
        --z-mode slots \
        --batch-scenes "$BATCH_SCENES" \
        --device "$DEVICE" \
        --rotate \
        --rotate-time-frame 6 \
        --output-json "$outdir/${eval_prefix}_${split}.json" \
        2>&1 | tee "$LOG_ROOT/${run}_${split}_eval.log"
    done
  done
}

summarize_variant() {
  local variant="$1"
  local suffix="$2"
  local run_prefix="${RUN_STAMP}_${suffix}"
  local eval_prefix="${variant}_official"

  echo "===== SUMMARY $variant run_prefix=$run_prefix ====="
  python -m trustmoe_traj.scripts.summarize_social_cvae_refiner \
    --run-prefix "$run_prefix" \
    --run-name social_cvae_refiner \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --eval-file-prefix "$eval_prefix" \
    2>&1 | tee "$LOG_ROOT/${run_prefix}_summary.log"
}

diagnose_variant() {
  local variant="$1"
  local suffix="$2"
  local run_prefix="${RUN_STAMP}_${suffix}"
  local out_root="trustmoe_traj/analysis/${run_prefix}_adaptive_base_budget"

  echo "===== DIAGNOSE $variant run_prefix=$run_prefix ====="
  for seed in "${SEED_ARRAY[@]}"; do
    local run="${run_prefix}_seed${seed}"
    local ref="trustmoe_traj/analysis/social_cvae_refiner_models/$run/social_cvae_refiner_best.pt"
    for split in "${SPLIT_ARRAY[@]}"; do
      python -m trustmoe_traj.scripts.diagnose_v55_adaptive_base_budget \
        --protocol official_align \
        --subset eth \
        --split "$split" \
        --data-root "$DATA_ROOT" \
        --sample-mode per_agent \
        --slow-cfg-path "$SLOW_RUN_DIR/cor_fm_updated.yml" \
        --slow-checkpoint "$SLOW_RUN_DIR/models/checkpoint_best.pt" \
        --refiner-checkpoint "$ref" \
        --residual-slots "$SLOTS" \
        --keep-k 20 \
        --random-trials "$RANDOM_TRIALS" \
        --batch-scenes "$BATCH_SCENES" \
        --device "$DEVICE" \
        --rotate \
        --rotate-time-frame 6 \
        --output-json "$out_root/seed${seed}_${split}.json" \
        2>&1 | tee "$LOG_ROOT/${run}_${split}_adaptive_budget.log"
    done
  done

  python -m trustmoe_traj.scripts.summarize_v55_adaptive_base_budget \
    --input-root "$out_root" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    2>&1 | tee "$LOG_ROOT/${run_prefix}_adaptive_budget_summary.log"
}

for variant in v56c1 v56c2 v56c3; do
  suffix="$(run_suffix "$variant")"
  train_variant "$variant" "$suffix"
  eval_variant "$variant" "$suffix"
  summarize_variant "$variant" "$suffix"
  if [[ "$RUN_DIAGNOSIS" == "1" ]]; then
    diagnose_variant "$variant" "$suffix"
  fi
done

echo "===== V56-C sequential pipeline complete ====="
