"""Export Trajectron++ prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import dill
import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Trajectron++ K-sample predictions.")
    parser.add_argument("--trajectron-root", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _trajectron_import_context(trajectron_root: Path) -> Iterator[None]:
    root = trajectron_root.resolve()
    paths = [root / "trajectron", root / "experiments" / "pedestrians"]
    inserted: List[str] = []
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(text)
    try:
        yield
    finally:
        for text in inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is unavailable")
    return device


def _load_modules() -> Dict[str, Any]:
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    return {"ModelRegistrar": ModelRegistrar, "Trajectron": Trajectron}


def _load_model(*, modules: Mapping[str, Any], model_dir: Path, checkpoint: int, env: Any, device: torch.device):
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / f"model_registrar-{int(checkpoint)}.pt"
    if not config_path.exists():
        raise SystemExit(f"Missing Trajectron++ config: {config_path.as_posix()}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing Trajectron++ checkpoint: {checkpoint_path.as_posix()}")
    hyperparams = json.loads(config_path.read_text(encoding="utf-8"))
    registrar = modules["ModelRegistrar"](model_dir.as_posix(), device)
    registrar.load_models(int(checkpoint))
    trajectron = modules["Trajectron"](registrar, hyperparams, None, device)
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams, checkpoint_path


def _prediction_array(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    # Trajectron++ stores per-node prediction as [1,K,T,2].
    if arr.ndim == 4 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected Trajectron++ node prediction shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _record_from_group(
    *,
    scene_index: int,
    scene_name: str,
    timestep: int,
    node_items: List[tuple[Any, np.ndarray]],
    state: Mapping[str, Any],
    max_hl: int,
    ph: int,
) -> Dict[str, Any]:
    past_items: List[np.ndarray] = []
    future_items: List[np.ndarray] = []
    pred_items: List[np.ndarray] = []
    node_ids: List[str] = []
    position_state = {"position": ["x", "y"]}
    for node, pred in node_items:
        past = node.get(np.asarray([int(timestep) - int(max_hl), int(timestep)]), state[node.type])
        # Trajectron++ may be trained with velocity targets, but predict() returns
        # absolute positions after dynamics integration. AFC/ADE/FDE therefore
        # need the position future, not hyperparams["pred_state"].
        future = node.get(np.asarray([int(timestep) + 1, int(timestep) + int(ph)]), position_state)
        if np.isnan(past).any() or np.isnan(future).any():
            continue
        if past.shape[0] != int(max_hl) + 1 or future.shape[0] != int(ph):
            continue
        if pred.shape[0] <= 0 or pred.shape[1] != int(ph):
            continue
        past_items.append(np.asarray(past[:, :2], dtype=np.float32))
        future_items.append(np.asarray(future[:, :2], dtype=np.float32))
        pred_items.append(np.asarray(pred[:, :, :2], dtype=np.float32))
        node_ids.append(str(node.id))

    if not past_items:
        raise ValueError("No valid Trajectron++ nodes in prediction group")
    past_abs = np.stack(past_items, axis=0)
    future_abs = np.stack(future_items, axis=0)
    pred_abs = np.stack(pred_items, axis=1)  # [K,A,T,2]
    last_obs = past_abs[:, -1:, :]
    pred_rel = (pred_abs - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    features = build_moflow_eth_feature_arrays(past_abs, future_abs, rotate=False)
    agent_mask = np.ones((past_abs.shape[0],), dtype=np.int64)
    social = compute_past_social_risk_features(past_abs, agent_mask)
    return {
        "scene_index": int(scene_index),
        "scene_name": str(scene_name),
        "timestep": int(timestep),
        "node_ids": node_ids,
        "obs_abs": torch.from_numpy(past_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(pred_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(pred_rel),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
    }


@torch.no_grad()
def _predict_records(
    *,
    trajectron: Any,
    env: Any,
    hyperparams: Mapping[str, Any],
    k: int,
    max_scenes: Optional[int],
    max_records: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    ph = int(hyperparams["prediction_horizon"])
    max_hl = int(hyperparams["maximum_history_length"])
    for override in hyperparams.get("override_attention_radius", []):
        node_type1, node_type2, radius = str(override).split(" ")
        env.attention_radius[(node_type1, node_type2)] = float(radius)
    for scene_index, scene in enumerate(env.scenes):
        if max_scenes is not None and scene_index >= int(max_scenes):
            break
        scene.calculate_scene_graph(
            env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )
        timesteps = np.arange(scene.timesteps)
        predictions = trajectron.predict(
            scene,
            timesteps,
            ph,
            num_samples=int(k),
            min_history_timesteps=max_hl,
            min_future_timesteps=ph,
            z_mode=False,
            gmm_mode=False,
            full_dist=False,
        )
        for timestep in sorted(predictions.keys()):
            node_items: List[tuple[Any, np.ndarray]] = []
            for node, raw_pred in predictions[timestep].items():
                node_items.append((node, _prediction_array(raw_pred)))
            try:
                record = _record_from_group(
                    scene_index=scene_index,
                    scene_name=str(scene.name),
                    timestep=int(timestep),
                    node_items=node_items,
                    state=hyperparams["state"],
                    max_hl=max_hl,
                    ph=ph,
                )
            except ValueError:
                continue
            records.append(record)
            if max_records is not None and len(records) >= int(max_records):
                return records
        print(f"[export_trajectronpp_predictions] scene={scene_index + 1}/{len(env.scenes)} records={len(records)}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    root = Path(args.trajectron_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Missing Trajectron++ root: {root.as_posix()}")
    if not data_path.exists():
        raise SystemExit(f"Missing Trajectron++ processed data: {data_path.as_posix()}")
    device = _resolve_device(str(args.device))

    with _trajectron_import_context(root):
        modules = _load_modules()
        with data_path.open("rb") as handle:
            env = dill.load(handle, encoding="latin1")
        trajectron, hyperparams, checkpoint_path = _load_model(
            modules=modules,
            model_dir=model_dir,
            checkpoint=int(args.checkpoint),
            env=env,
            device=device,
        )
        records = _predict_records(
            trajectron=trajectron,
            env=env,
            hyperparams=hyperparams,
            k=int(args.k),
            max_scenes=args.max_scenes,
            max_records=args.max_records,
        )

    if not records:
        raise SystemExit("No Trajectron++ records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_trajectronpp_predictions",
            "baseline": "Trajectron++",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trajectron_root": root.as_posix(),
            "model_dir": model_dir.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_epoch": int(args.checkpoint),
            "data": data_path.as_posix(),
            "subset": str(args.subset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "num_records": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "trajectron_prediction_space": "absolute_position_after_dynamics",
            "ground_truth_future_source": "position",
            "evaluation_prediction_space": "relative_to_last_observed_position",
            "evaluation_ground_truth_space": "relative_to_last_observed_position",
        },
        "records": records,
    }
    output_path = Path(args.output_bundle).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
