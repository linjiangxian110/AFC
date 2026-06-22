#!/usr/bin/env bash
set -euo pipefail

MAIN="${MAIN:-/mnt/data/lck/code/TrustMoE-Traj-v38}"
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /mnt/data/lck/code/moflow/moflow_venv/bin/python ]]; then
    PY=/mnt/data/lck/code/moflow/moflow_venv/bin/python
  else
    PY=python
  fi
fi

find_baseline_root() {
  local name="$1"
  local candidate
  for candidate in \
    "$MAIN/参考/开源基线模型/$name" \
    "$MAIN/baselines/$name" \
    "$MAIN/$name"
  do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  candidate="$(find "$MAIN" -maxdepth 5 -type d -name "$name" -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" && -d "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  printf '%s\n' "$MAIN/参考/开源基线模型/$name"
}

GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_mid_sdd_timing_seed0_epoch1}"
MID_ROOT="${MID_ROOT:-$(find_baseline_root MID)}"
CONFIG_FILE="${CONFIG_FILE:-$MID_ROOT/configs/trustmoe_afc_seed0.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN/trustmoe_traj/analysis/sdd_external_baselines/mid/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
PROCESSED_DIR="${PROCESSED_DIR:-$MID_ROOT/processed_data}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
EPOCHS="${EPOCHS:-1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-90}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$PROCESSED_DIR"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  echo "$line" | tee -a "$LOG_ROOT/manifest.log" >&2
}

check_env() {
  log "ENV pwd=$(pwd)"
  log "ENV MAIN=$MAIN"
  log "ENV PY=$PY"
  log "ENV GPU=$GPU CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  log "ENV RUN_ID=$RUN_ID"
  log "ENV OUTPUT_ROOT=$OUTPUT_ROOT"
  log "ENV LOG_ROOT=$LOG_ROOT"
  log "ENV MID_ROOT=$MID_ROOT"
  log "ENV CONFIG_FILE=$CONFIG_FILE"
  log "ENV PROCESSED_DIR=$PROCESSED_DIR"
  test "$(pwd)" = "$MAIN" || { echo "[ERR] not in MAIN: $(pwd)" >&2; exit 1; }
  test -x "$PY" || { echo "[ERR] PY not executable: $PY" >&2; exit 1; }
  test -d "$MID_ROOT" || { echo "[ERR] missing MID_ROOT: $MID_ROOT" >&2; exit 1; }
  test -f "$CONFIG_FILE" || { echo "[ERR] missing CONFIG_FILE: $CONFIG_FILE" >&2; exit 1; }
  test -f "$MID_ROOT/raw_data/stanford/train_trajnet.pkl" || { echo "[ERR] missing MID SDD raw train_trajnet.pkl" >&2; exit 1; }
  test -f "$MID_ROOT/raw_data/stanford/test_trajnet.pkl" || { echo "[ERR] missing MID SDD raw test_trajnet.pkl" >&2; exit 1; }
}

preprocess_sdd() {
  if [[ "$RUN_PREPROCESS" != "1" ]]; then
    log "SKIP preprocess RUN_PREPROCESS=$RUN_PREPROCESS"
    return
  fi
  if [[ "$FORCE_PREPROCESS" != "1" && -f "$PROCESSED_DIR/sdd_train.pkl" && -f "$PROCESSED_DIR/sdd_test.pkl" ]]; then
    log "SKIP preprocess existing SDD processed data at $PROCESSED_DIR"
    return
  fi
  log "PREPROCESS MID SDD raw_data/stanford -> $PROCESSED_DIR"
  (
    cd "$MID_ROOT"
    PROCESSED_DIR="$PROCESSED_DIR" "$PY" - <<'PY'
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import dill
except Exception:
    dill = pickle

from environment import Environment, Scene, Node, derivative_of

out_path = Path(os.environ["PROCESSED_DIR"])
out_path.mkdir(parents=True, exist_ok=True)
raw_path = Path("raw_data/stanford")
dt = 0.4
standardization = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}
data_columns = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return matrix @ pc
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)
    alpha = angle * np.pi / 180.0
    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()
        x, y = rotate_pc(np.array([x, y]), alpha)
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)
        node_data = pd.DataFrame(
            {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            },
            columns=data_columns,
        )
        scene_aug.nodes.append(Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep))
    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug

for data_class in ("train", "test"):
    print(f"Processing SDD {data_class}")
    frame = pickle.load(open(raw_path / f"{data_class}_trajnet.pkl", "rb"))
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0}
    scenes = []
    for _scene_name, data in frame.groupby("sceneId"):
        data = data.copy()
        data["frame"] = pd.to_numeric(data["frame"], downcast="integer")
        data["trackId"] = pd.to_numeric(data["trackId"], downcast="integer")
        data["frame"] = data["frame"] // 12
        data["frame"] -= data["frame"].min()
        data["node_id"] = data["trackId"].astype(str)
        data["x"] = data["x"] / 50.0
        data["y"] = data["y"] / 50.0
        data["x"] = data["x"] - data["x"].mean()
        data["y"] = data["y"] - data["y"].mean()
        if len(data) <= 0:
            continue
        scene = Scene(
            timesteps=int(data["frame"].max()) + 1,
            dt=dt,
            name=f"sdd_{data_class}",
            aug_func=augment if data_class == "train" else None,
        )
        for node_id in pd.unique(data["node_id"]):
            node_df = data[data["node_id"] == node_id]
            if len(node_df) <= 1:
                continue
            if not np.all(np.diff(node_df["frame"]) == 1):
                continue
            node_values = node_df[["x", "y"]].values
            if node_values.shape[0] < 2:
                continue
            first_idx = int(node_df["frame"].iloc[0])
            x = node_values[:, 0]
            y = node_values[:, 1]
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)
            node_data = pd.DataFrame(
                {
                    ("position", "x"): x,
                    ("position", "y"): y,
                    ("velocity", "x"): vx,
                    ("velocity", "y"): vy,
                    ("acceleration", "x"): ax,
                    ("acceleration", "y"): ay,
                },
                columns=data_columns,
            )
            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
            node.first_timestep = first_idx
            scene.nodes.append(node)
        if data_class == "train":
            scene.augmented = [augment_scene(scene, angle) for angle in np.arange(0, 360, 15)]
        scenes.append(scene)
    env.scenes = scenes
    print(f"Processed {len(scenes)} SDD {data_class} scenes")
    with open(out_path / f"sdd_{data_class}.pkl", "wb") as handle:
        dill.dump(env, handle, protocol=dill.HIGHEST_PROTOCOL)
PY
  ) 2>&1 | tee "$LOG_ROOT/process_sdd_data.log"
}

train_sdd() {
  test -f "$PROCESSED_DIR/sdd_train.pkl" || { echo "[ERR] missing $PROCESSED_DIR/sdd_train.pkl" >&2; exit 1; }
  test -f "$PROCESSED_DIR/sdd_test.pkl" || { echo "[ERR] missing $PROCESSED_DIR/sdd_test.pkl" >&2; exit 1; }
  log "TRAIN MID SDD epochs=$EPOCHS target_epochs=$TARGET_EPOCHS gpu=$GPU"
  local tmp_config="$OUTPUT_ROOT/${RUN_ID}_config_sdd_epoch${EPOCHS}.yaml"
  "$PY" - "$CONFIG_FILE" "$tmp_config" "$PROCESSED_DIR" "$EPOCHS" <<'PY'
import sys
import yaml
src, dst, processed_dir, epochs = sys.argv[1:5]
payload = yaml.safe_load(open(src, "r", encoding="utf-8"))
payload["data_dir"] = processed_dir
payload["epochs"] = int(epochs)
payload["eval_every"] = max(1, int(epochs))
payload["eval_at"] = int(epochs)
payload["eval_mode"] = False
payload["preprocess_workers"] = int(payload.get("preprocess_workers", 0))
open(dst, "w", encoding="utf-8").write(yaml.safe_dump(payload, sort_keys=False))
PY
  local exp_name
  local model_dir
  exp_name="$(basename "$tmp_config" .yaml)"
  model_dir="$MID_ROOT/experiments/$exp_name"
  local start_ts
  local end_ts
  local elapsed_sec
  local estimate
  start_ts="$(date +%s)"
  (
    cd "$MID_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py --config "$tmp_config" --dataset sdd
  ) 2>&1 | tee "$LOG_ROOT/train_sdd_epoch${EPOCHS}.log"
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"
  estimate="$("$PY" - "$elapsed_sec" "$EPOCHS" "$TARGET_EPOCHS" <<'PY'
import sys
elapsed = float(sys.argv[1])
epochs = max(float(sys.argv[2]), 1.0)
target_epochs = float(sys.argv[3])
print(int(round(elapsed / epochs * target_epochs * 1.2)))
PY
)"
  {
    echo "dataset=sdd"
    echo "seed=0"
    echo "epochs=$EPOCHS"
    echo "target_epochs=$TARGET_EPOCHS"
    echo "start_ts=$start_ts"
    echo "end_ts=$end_ts"
    echo "elapsed_sec=$elapsed_sec"
    echo "estimated_full_sec=$estimate"
    echo "model_dir=$model_dir"
    echo "checkpoint=$model_dir/sdd_epoch${EPOCHS}.pt"
    echo "config=$tmp_config"
    echo "processed_dir=$PROCESSED_DIR"
  } | tee "$OUTPUT_ROOT/timing_sdd_seed0_epoch${EPOCHS}.txt"
}

check_env
log "START MID SDD training"
preprocess_sdd
train_sdd
log "DONE MID SDD training OUTPUT_ROOT=$OUTPUT_ROOT"
