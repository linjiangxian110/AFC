"""Export TrajEvo heuristic prediction sets for AFC evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd")
SPLITS = ("test", "val")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TrajEvo K=20 heuristic prediction bundles.")
    parser.add_argument("--trajevo-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=SPLITS)
    parser.add_argument(
        "--heuristic-dataset",
        type=str,
        default=None,
        choices=DATASETS,
        help=(
            "Heuristic file under trajectory_prediction/trajevo to use. "
            "Defaults to eth for dataset=sdd, matching the TrajEvo README SDD generalization protocol."
        ),
    )
    parser.add_argument(
        "--heuristic-path",
        type=str,
        default=None,
        help="Explicit TrajEvo heuristic .py file. Use this for a trained SDD heuristic stored outside the default folder.",
    )
    parser.add_argument(
        "--allow-cross-dataset-heuristic",
        action="store_true",
        help="Record an explicitly non-default cross-dataset diagnostic heuristic use.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-mode", type=str, default="scene", choices=["scene", "constant"])
    parser.add_argument(
        "--sdd-scale-factor",
        type=float,
        default=None,
        help=(
            "Scale TrajEvo SDD dataset coordinates back to the MoFlow SDD original coordinate convention. "
            "Defaults to 100.0 for dataset=sdd, matching the current TrajEvo README-style adapter."
        ),
    )
    parser.add_argument("--samples-per-scene", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


def _load_predict_fn_from_path(module_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    if not module_path.exists():
        raise SystemExit(f"Missing TrajEvo heuristic file: {module_path.as_posix()}")
    spec = importlib.util.spec_from_file_location(f"trajevo_heuristic_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import TrajEvo heuristic file: {module_path.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "predict_trajectory", None)
    if not callable(fn):
        raise SystemExit(f"TrajEvo heuristic has no callable predict_trajectory: {module_path.as_posix()}")
    return fn


def _heuristic_path_from_dataset(trajevo_root: Path, dataset: str) -> Path:
    return trajevo_root / "trajectory_prediction" / "trajevo" / f"{dataset}.py"


def _load_data(
    *,
    trajevo_root: Path,
    dataset: str,
    split: str,
    max_scenes: Optional[int],
    samples_per_scene: Optional[int],
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    sys.path.insert(0, trajevo_root.as_posix())
    from trajectory_prediction.utils import load_limited_data_per_scene

    dataset_dir = trajevo_root / "trajectory_prediction" / "datasets" / dataset
    if not dataset_dir.exists():
        raise SystemExit(f"Missing TrajEvo dataset directory: {dataset_dir.as_posix()}")
    phase_dir = dataset_dir / split
    if not phase_dir.exists():
        raise SystemExit(f"Missing TrajEvo split directory: {phase_dir.as_posix()}")
    per_scene = samples_per_scene
    if per_scene is None:
        per_scene = int(max_scenes) if max_scenes is not None else 1_000_000_000
    inputs, targets = load_limited_data_per_scene(
        dataset_dir.as_posix(),
        split,
        obs_len=8,
        pred_len=12,
        samples_per_scene=int(per_scene),
        seed=42,
    )
    if max_scenes is not None:
        inputs = inputs[: int(max_scenes)]
        targets = targets[: int(max_scenes)]
    return inputs, targets


def _as_agent_time_xy(value: Any, *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise RuntimeError(f"Invalid {name} shape: {arr.shape}; expected [agents,{expected_len},2]")
    if arr.shape[1] != expected_len:
        raise RuntimeError(f"Invalid {name} length: {arr.shape}; expected trajectory length {expected_len}")
    return arr.astype(np.float32, copy=False)


def _as_prediction(value: Any, *, k: int, num_agents: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4 or arr.shape[-2:] != (12, 2):
        raise RuntimeError(f"Invalid TrajEvo prediction shape: {arr.shape}; expected [K,agents,12,2]")
    if arr.shape[0] < int(k):
        raise RuntimeError(f"TrajEvo prediction has only {arr.shape[0]} samples, need K={k}")
    if arr.shape[1] != int(num_agents):
        raise RuntimeError(f"TrajEvo agent count mismatch: prediction={arr.shape[1]} obs={num_agents}")
    return arr[: int(k)].astype(np.float32, copy=False)


def _record_from_scene(
    *,
    dataset: str,
    split: str,
    scene_index: int,
    obs_abs: np.ndarray,
    future_abs: np.ndarray,
    prediction_abs: np.ndarray,
    scale_factor: float = 1.0,
) -> Dict[str, Any]:
    scale = float(scale_factor)
    if scale != 1.0:
        obs_abs = obs_abs * scale
        future_abs = future_abs * scale
        prediction_abs = prediction_abs * scale
    agent_mask = np.ones((obs_abs.shape[0],), dtype=np.int64)
    prediction_rel = prediction_abs - obs_abs[None, :, -1:, :]
    features = build_moflow_eth_feature_arrays(obs_abs, future_abs, rotate=False)
    social = compute_past_social_risk_features(obs_abs, agent_mask)
    return {
        "dataset": str(dataset),
        "split": str(split),
        "scene_index": int(scene_index),
        "source_scene_id": f"{dataset}_{split}_{scene_index:06d}",
        "obs_abs": torch.from_numpy(obs_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(prediction_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(prediction_rel.astype(np.float32, copy=False)),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
    }


def main() -> None:
    args = build_parser().parse_args()
    trajevo_root = Path(args.trajevo_root).expanduser().resolve()
    output_path = Path(args.output_bundle).expanduser().resolve()
    heuristic_dataset = str(args.heuristic_dataset or ("eth" if str(args.dataset) == "sdd" else args.dataset))
    heuristic_path = (
        Path(args.heuristic_path).expanduser().resolve()
        if args.heuristic_path
        else _heuristic_path_from_dataset(trajevo_root, heuristic_dataset)
    )
    protocol = "dataset_specific"
    if str(args.dataset) == "sdd" and heuristic_dataset == "eth":
        protocol = "official_sdd_generalization"
    elif heuristic_dataset != str(args.dataset):
        protocol = "cross_dataset_diagnostic" if bool(args.allow_cross_dataset_heuristic) else "cross_dataset_unspecified"
    scale_factor = 1.0
    if str(args.dataset) == "sdd":
        scale_factor = float(args.sdd_scale_factor) if args.sdd_scale_factor is not None else 100.0
    if not trajevo_root.exists():
        raise SystemExit(f"Missing TrajEvo root: {trajevo_root.as_posix()}")
    if int(args.k) != 20:
        raise SystemExit("TrajEvo official heuristics return K=20; keep --k 20 for this adapter.")

    predict_trajectory = _load_predict_fn_from_path(heuristic_path)
    inputs, targets = _load_data(
        trajevo_root=trajevo_root,
        dataset=str(args.dataset),
        split=str(args.split),
        max_scenes=args.max_scenes,
        samples_per_scene=args.samples_per_scene,
    )
    if not inputs:
        raise SystemExit("No TrajEvo records were loaded")

    records: List[Dict[str, Any]] = []
    for scene_index, (obs_raw, future_raw) in enumerate(zip(inputs, targets)):
        scene_seed = int(args.seed) if str(args.seed_mode) == "constant" else int(args.seed) + int(scene_index)
        np.random.seed(scene_seed)
        torch.manual_seed(scene_seed)
        obs_abs = _as_agent_time_xy(obs_raw, name="obs", expected_len=8)
        future_abs = _as_agent_time_xy(future_raw, name="future", expected_len=12)
        prediction_abs = _as_prediction(predict_trajectory(obs_abs.copy()), k=int(args.k), num_agents=obs_abs.shape[0])
        records.append(
            _record_from_scene(
                dataset=str(args.dataset),
                split=str(args.split),
                scene_index=scene_index,
                obs_abs=obs_abs,
                future_abs=future_abs,
                prediction_abs=prediction_abs,
                scale_factor=scale_factor,
            )
        )
        if len(records) % 100 == 0:
            print(f"[export_trajevo_predictions] scenes={len(records)}")

    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_trajevo_predictions",
            "baseline": "TrajEvo",
            "model_name": "TrajEvo-official-heuristic",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trajevo_root": trajevo_root.as_posix(),
            "heuristic_file": heuristic_path.as_posix(),
            "heuristic_dataset": heuristic_dataset,
            "allow_cross_dataset_heuristic": bool(args.allow_cross_dataset_heuristic),
            "trajevo_protocol": protocol,
            "dataset": str(args.dataset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "seed_mode": str(args.seed_mode),
            "samples_per_scene": args.samples_per_scene,
            "max_scenes": args.max_scenes,
            "scale_factor": float(scale_factor),
            "coordinate_contract": "MoFlow SDD relative future convention: prediction_rel and fut_traj_original_scale are relative to the last observed position.",
            "num_records": int(len(records)),
            "num_valid_agents": int(valid_agents),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
