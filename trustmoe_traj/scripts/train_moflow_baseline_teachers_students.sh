#!/usr/bin/env bash

# Train/reuse MoFlow slow teachers, export train IMLE targets, and train
# original IMLE students for ETH-UCY subsets and SDD.
#
# Typical usage:
#   export MAIN=/mnt/data/lck/code/TrustMoE-Traj-v38
#   export GPU=0
#   bash "$MAIN/trustmoe_traj/scripts/train_moflow_baseline_teachers_students.sh"
#
# Useful overrides:
#   ETH_SUBSETS="univ zara2" RUN_SDD=0 bash ...
#   SLOW_EPOCHS=150 IMLE_EPOCHS=150 ETH_IMLE_BATCH_SIZE=24 bash ...

set -uo pipefail

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_moflow_baseline}"
SEED="${SEED:-0}"

RUN_ETH="${RUN_ETH:-1}"
RUN_SDD="${RUN_SDD:-1}"
ETH_SUBSETS="${ETH_SUBSETS:-eth hotel zara1 univ zara2}"

SLOW_EPOCHS="${SLOW_EPOCHS:-150}"
IMLE_EPOCHS="${IMLE_EPOCHS:-150}"
SLOW_BATCH_SIZE="${SLOW_BATCH_SIZE:-32}"
ETH_IMLE_BATCH_SIZE="${ETH_IMLE_BATCH_SIZE:-24}"
SDD_IMLE_BATCH_SIZE="${SDD_IMLE_BATCH_SIZE:-64}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-64}"
LR="${LR:-1e-4}"

ETH_DATA_DIR="${ETH_DATA_DIR:-$MAIN/MoFlow/data/eth_ucy}"
SDD_DATA_DIR="${SDD_DATA_DIR:-$MAIN/MoFlow/data/sdd}"
MANIFEST_DIR="$MAIN/trustmoe_traj/analysis/moflow_baseline_training/$RUN_ID"
MANIFEST="$MANIFEST_DIR/manifest.txt"

mkdir -p "$MANIFEST_DIR"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$MANIFEST" >&2
}

run_cmd() {
  log "RUN: $*"
  "$@" >&2
}

find_latest_dir() {
  local pattern="$1"
  shift
  local root
  local candidates=()
  for root in "$@"; do
    if [[ -d "$root" ]]; then
      while IFS= read -r item; do
        candidates+=("$item")
      done < <(find "$root" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort)
    fi
  done
  if [[ ${#candidates[@]} -gt 0 ]]; then
    printf '%s\n' "${candidates[@]}" | sort | tail -1
  fi
}

copy_new_train_samples() {
  local sample_dir="$1"
  local stamp_file="$2"
  local target_dir="$3"
  local label="$4"
  local samples=()

  mkdir -p "$target_dir"
  rm -f "$target_dir"/*train*.pkl 2>/dev/null || true

  while IFS= read -r item; do
    samples+=("$item")
  done < <(find "$sample_dir" -type f -name "*train*.pkl" -newer "$stamp_file" 2>/dev/null | sort)

  if [[ ${#samples[@]} -eq 0 ]]; then
    log "ERROR: no freshly exported train samples found for $label under $sample_dir"
    return 1
  fi

  cp -f "${samples[@]}" "$target_dir"/
  log "$label exported_train_samples=${#samples[@]} target_dir=$target_dir"
}

find_eth_slow_run() {
  local subset="$1"
  find_latest_dir \
    "*${subset}_rot_6*" \
    "$MAIN/MoFlow/results_eth_ucy/cor_fm" \
    "/mnt/data/lck/code/TrustMoE-Traj/MoFlow/results_eth_ucy/cor_fm" \
    "/mnt/data/lck/code/moflow/MoFlow/results_eth_ucy/cor_fm"
}

ensure_eth_pickles() {
  local subset="$1"
  local missing=0
  local split

  for split in train test; do
    if [[ ! -f "$ETH_DATA_DIR/original/$subset/${subset}_${split}.pkl" ]]; then
      missing=1
    fi
  done

  if [[ "$missing" == "0" ]]; then
    return 0
  fi

  log "ETH-UCY $subset pickle files missing; generate ETH-UCY original pickles"
  cd "$MAIN/MoFlow" || return 1
  run_cmd python -m data.store_pickle_eth_files --data_dir ./data/eth_ucy/original

  for split in train test; do
    if [[ ! -f "$ETH_DATA_DIR/original/$subset/${subset}_${split}.pkl" ]]; then
      log "ERROR: still missing $ETH_DATA_DIR/original/$subset/${subset}_${split}.pkl after pickle generation"
      return 1
    fi
  done
}

ensure_eth_slow() {
  local subset="$1"
  local slow_run
  slow_run="$(find_eth_slow_run "$subset")"
  if [[ -n "$slow_run" && -f "$slow_run/cor_fm_updated.yml" && -f "$slow_run/models/checkpoint_best.pt" ]]; then
    log "ETH-UCY $subset slow exists: $slow_run"
    printf '%s\n' "$slow_run"
    return 0
  fi

  ensure_eth_pickles "$subset" || return 1

  log "ETH-UCY $subset slow missing; start training"
  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python fm_eth.py \
    --cfg cfg/eth_ucy/cor_fm.yml \
    --exp "trustmoe_slow_${subset}_baseline_${RUN_ID}" \
    --data_source original \
    --data_dir ./data/eth_ucy \
    --subset "$subset" \
    --rotate \
    --rotate_time_frame 6 \
    --tied_noise \
    --fm_in_scaling \
    --checkpt_freq 1 \
    --epochs "$SLOW_EPOCHS" \
    --batch_size "$SLOW_BATCH_SIZE" \
    --init_lr "$LR"

  slow_run="$(find_latest_dir "*trustmoe_slow_${subset}_baseline_${RUN_ID}*${subset}_rot_6*" "$MAIN/MoFlow/results_eth_ucy/cor_fm")"
  if [[ -z "$slow_run" || ! -f "$slow_run/models/checkpoint_best.pt" ]]; then
    log "ERROR: ETH-UCY $subset slow training finished but checkpoint was not found"
    return 1
  fi
  printf '%s\n' "$slow_run"
}

export_eth_imle_targets() {
  local subset="$1"
  local slow_run="$2"
  local slow_cfg="$slow_run/cor_fm_updated.yml"
  local slow_ckpt="$slow_run/models/checkpoint_best.pt"
  local imle_dir_name="imle_baseline_${subset}_${RUN_ID}"
  local target_dir="$ETH_DATA_DIR/$imle_dir_name/$subset"
  local stamp_file

  stamp_file="$(mktemp)"
  touch "$stamp_file"

  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python eval_eth.py \
    --ckpt_path "$slow_ckpt" \
    --cfg "$slow_cfg" \
    --exp "export_imle_${subset}_${RUN_ID}" \
    --save_samples \
    --eval_on_train \
    --data_source original \
    --data_dir ./data/eth_ucy \
    --subset "$subset" \
    --rotate \
    --rotate_time_frame 6 \
    --batch_size "$EXPORT_BATCH_SIZE"

  copy_new_train_samples "$slow_run/samples" "$stamp_file" "$target_dir" "eth_ucy/$subset" || return 1
  rm -f "$stamp_file"
  printf '%s\n' "$imle_dir_name"
}

train_eth_imle() {
  local subset="$1"
  local slow_run="$2"
  local imle_dir_name="$3"
  local slow_ckpt="$slow_run/models/checkpoint_best.pt"
  local existing

  existing="$(find_latest_dir "*trustmoe_imle_${subset}_${RUN_ID}*${imle_dir_name}*" "$MAIN/MoFlow/results_eth_ucy/imle")"
  if [[ -n "$existing" && -f "$existing/models/checkpoint_best.pt" ]]; then
    log "ETH-UCY $subset IMLE exists: $existing"
    return 0
  fi

  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python imle_eth.py \
    --cfg cfg/eth_ucy/imle.yml \
    --exp "trustmoe_imle_${subset}_${RUN_ID}" \
    --data_source original \
    --data_dir ./data/eth_ucy \
    --subset "$subset" \
    --imle_dir_name "$imle_dir_name" \
    --rotate \
    --rotate_time_frame 6 \
    --checkpt_freq 1 \
    --epochs "$IMLE_EPOCHS" \
    --batch_size "$ETH_IMLE_BATCH_SIZE" \
    --init_lr "$LR" \
    --num_to_gen 20 \
    --load_pretrained \
    --ckpt_path "$slow_ckpt" \
    --fix_random_seed \
    --seed "$SEED"
}

find_sdd_slow_run() {
  find_latest_dir \
    "*sdd*rot_6*" \
    "$MAIN/MoFlow/results_sdd/cor_fm" \
    "/mnt/data/lck/code/TrustMoE-Traj/MoFlow/results_sdd/cor_fm" \
    "/mnt/data/lck/code/moflow/MoFlow/results_sdd/cor_fm"
}

ensure_sdd_slow() {
  local slow_run
  slow_run="$(find_sdd_slow_run)"
  if [[ -n "$slow_run" && -f "$slow_run/cor_fm_updated.yml" && -f "$slow_run/models/checkpoint_best.pt" ]]; then
    log "SDD slow exists: $slow_run"
    printf '%s\n' "$slow_run"
    return 0
  fi

  if [[ ! -f "$SDD_DATA_DIR/original/sdd_train.pkl" || ! -f "$SDD_DATA_DIR/original/sdd_test.pkl" ]]; then
    log "SKIP SDD: missing $SDD_DATA_DIR/original/sdd_train.pkl or sdd_test.pkl"
    log "Hint: find /mnt/data/lck/code -type f \\( -name 'sdd_train.pkl' -o -name 'sdd_test.pkl' \\) | sort"
    return 1
  fi

  log "SDD slow missing; start training"
  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python fm_sdd.py \
    --cfg cfg/sdd/cor_fm.yml \
    --exp "trustmoe_slow_sdd_baseline_${RUN_ID}" \
    --data_dir ./data/sdd \
    --rotate \
    --rotate_time_frame 6 \
    --tied_noise \
    --fm_in_scaling \
    --checkpt_freq 1 \
    --epochs "$SLOW_EPOCHS" \
    --batch_size "$SLOW_BATCH_SIZE" \
    --init_lr "$LR"

  slow_run="$(find_latest_dir "*trustmoe_slow_sdd_baseline_${RUN_ID}*rot_6*" "$MAIN/MoFlow/results_sdd/cor_fm")"
  if [[ -z "$slow_run" || ! -f "$slow_run/models/checkpoint_best.pt" ]]; then
    log "ERROR: SDD slow training finished but checkpoint was not found"
    return 1
  fi
  printf '%s\n' "$slow_run"
}

export_sdd_imle_targets() {
  local slow_run="$1"
  local slow_cfg="$slow_run/cor_fm_updated.yml"
  local slow_ckpt="$slow_run/models/checkpoint_best.pt"
  local imle_dir_name="imle_baseline_sdd_${RUN_ID}"
  local target_dir="$SDD_DATA_DIR/$imle_dir_name"
  local stamp_file

  stamp_file="$(mktemp)"
  touch "$stamp_file"

  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python eval_sdd.py \
    --ckpt_path "$slow_ckpt" \
    --cfg "$slow_cfg" \
    --exp "export_imle_sdd_${RUN_ID}" \
    --save_samples \
    --eval_on_train \
    --data_dir ./data/sdd \
    --rotate \
    --rotate_time_frame 6 \
    --batch_size "$EXPORT_BATCH_SIZE"

  copy_new_train_samples "$slow_run/samples" "$stamp_file" "$target_dir" "sdd" || return 1
  rm -f "$stamp_file"
  printf '%s\n' "$imle_dir_name"
}

train_sdd_imle() {
  local slow_run="$1"
  local imle_dir_name="$2"
  local slow_ckpt="$slow_run/models/checkpoint_best.pt"
  local existing

  existing="$(find_latest_dir "*trustmoe_imle_sdd_${RUN_ID}*${imle_dir_name}*" "$MAIN/MoFlow/results_sdd/imle")"
  if [[ -n "$existing" && -f "$existing/models/checkpoint_best.pt" ]]; then
    log "SDD IMLE exists: $existing"
    return 0
  fi

  cd "$MAIN/MoFlow" || return 1
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" python imle_sdd.py \
    --cfg cfg/sdd/imle.yml \
    --exp "trustmoe_imle_sdd_${RUN_ID}" \
    --data_dir ./data/sdd \
    --imle_dir_name "$imle_dir_name" \
    --rotate \
    --rotate_time_frame 6 \
    --checkpt_freq 1 \
    --epochs "$IMLE_EPOCHS" \
    --batch_size "$SDD_IMLE_BATCH_SIZE" \
    --init_lr "$LR" \
    --num_to_gen 20 \
    --load_pretrained \
    --ckpt_path "$slow_ckpt" \
    --fix_random_seed \
    --seed "$SEED"
}

log "MAIN=$MAIN"
log "RUN_ID=$RUN_ID GPU=$GPU SEED=$SEED"
log "SLOW_EPOCHS=$SLOW_EPOCHS IMLE_EPOCHS=$IMLE_EPOCHS"
log "ETH_SUBSETS=$ETH_SUBSETS RUN_ETH=$RUN_ETH RUN_SDD=$RUN_SDD"

if [[ "$RUN_ETH" == "1" ]]; then
  if [[ ! -d "$ETH_DATA_DIR/original" ]]; then
    log "SKIP ETH-UCY: missing $ETH_DATA_DIR/original"
  else
    for subset in $ETH_SUBSETS; do
      log "===== ETH-UCY $subset ====="
      slow_run="$(ensure_eth_slow "$subset")" || {
        log "SKIP ETH-UCY $subset student because slow teacher is unavailable"
        continue
      }
      imle_dir_name="$(export_eth_imle_targets "$subset" "$slow_run")" || {
        log "SKIP ETH-UCY $subset student because target export failed"
        continue
      }
      train_eth_imle "$subset" "$slow_run" "$imle_dir_name" || {
        log "ERROR: ETH-UCY $subset IMLE training failed"
        continue
      }
    done
  fi
fi

if [[ "$RUN_SDD" == "1" ]]; then
  log "===== SDD ====="
  slow_run="$(ensure_sdd_slow)" || {
    log "SKIP SDD student because slow teacher is unavailable"
  }
  if [[ -n "${slow_run:-}" ]]; then
    imle_dir_name="$(export_sdd_imle_targets "$slow_run")" || {
      log "SKIP SDD student because target export failed"
      imle_dir_name=""
    }
    if [[ -n "${imle_dir_name:-}" ]]; then
      train_sdd_imle "$slow_run" "$imle_dir_name" || log "ERROR: SDD IMLE training failed"
    fi
  fi
fi

log "Done. Manifest: $MANIFEST"
