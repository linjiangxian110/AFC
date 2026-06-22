"""Export TUTR prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TUTR K=20 prediction bundles.")
    parser.add_argument("--tutr-root", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--hp-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument(
        "--sdd-scale-factor",
        type=float,
        default=1.0,
        help="Optional scale from TUTR SDD pkl coordinates to the MoFlow SDD original coordinate convention.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _tutr_import_context(tutr_root: Path) -> Iterator[None]:
    root = tutr_root.resolve()
    old_cwd = Path.cwd()
    inserted: List[str] = []
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted.append(text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for item in inserted:
            try:
                sys.path.remove(item)
            except ValueError:
                pass


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_python_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("tutr_hp_config", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load TUTR config: {path.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _rotation_matrix(angle: float) -> np.ndarray:
    return np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )


def _prepare_single_item(
    item: Sequence[Any],
    *,
    obs_len: int,
    dist_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    ped_obs_traj = np.asarray(item[0], dtype=np.float32)
    ped_pred_traj = np.asarray(item[1], dtype=np.float32)
    raw_neis_traj = np.asarray(item[2], dtype=np.float32)

    obs_abs = ped_obs_traj[:, :2].astype(np.float32, copy=False)
    future_abs = ped_pred_traj[:, :2].astype(np.float32, copy=False)
    ped_traj_abs = np.concatenate((obs_abs, future_abs), axis=0)

    if raw_neis_traj.size == 0:
        neis_traj = np.zeros((0, ped_traj_abs.shape[0], 2), dtype=np.float32)
    else:
        neis_traj = raw_neis_traj[:, :, :2].transpose(1, 0, 2).astype(np.float32, copy=False)
    neis_traj = np.concatenate((np.expand_dims(ped_traj_abs, axis=0), neis_traj), axis=0)
    distance = np.linalg.norm(np.expand_dims(ped_traj_abs, axis=0) - neis_traj, axis=-1)
    distance = np.mean(distance[:, :obs_len], axis=-1)
    neis_traj = neis_traj[distance < float(dist_threshold)]
    if neis_traj.shape[0] <= 0:
        neis_traj = np.expand_dims(ped_traj_abs, axis=0)

    origin = ped_traj_abs[obs_len - 1 : obs_len].astype(np.float32, copy=True)
    ped_traj = ped_traj_abs - origin
    neis_traj = neis_traj - np.expand_dims(origin, axis=0)

    ref_point = ped_traj[0]
    angle = float(np.arctan2(ref_point[1], ref_point[0]))
    rot_mat = _rotation_matrix(angle)
    ped_traj = np.matmul(ped_traj, rot_mat)
    neis_traj = np.matmul(neis_traj, np.expand_dims(rot_mat, axis=0))

    meta = {
        "obs_abs": obs_abs,
        "future_abs": future_abs,
        "origin": origin.reshape(2),
        "rot_mat": rot_mat,
    }
    return ped_traj.astype(np.float32, copy=False), neis_traj.astype(np.float32, copy=False), origin.reshape(2), rot_mat, meta


def _collate_items(
    items: Sequence[Sequence[Any]],
    *,
    obs_len: int,
    dist_threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, np.ndarray]]]:
    ped_items: List[np.ndarray] = []
    nei_items: List[np.ndarray] = []
    meta_items: List[Dict[str, np.ndarray]] = []
    n_neighbors: List[int] = []

    for item in items:
        ped_traj, neis_traj, _origin, _rot_mat, meta = _prepare_single_item(
            item,
            obs_len=int(obs_len),
            dist_threshold=float(dist_threshold),
        )
        ped_items.append(ped_traj)
        nei_items.append(neis_traj)
        meta_items.append(meta)
        n_neighbors.append(int(neis_traj.shape[0]))

    max_neighbors = max(n_neighbors)
    neis_pad: List[np.ndarray] = []
    neis_mask: List[np.ndarray] = []
    for neighbor, n in zip(nei_items, n_neighbors):
        neis_pad.append(np.pad(neighbor, ((0, max_neighbors - n), (0, 0), (0, 0)), "constant"))
        mask = np.zeros((max_neighbors, max_neighbors), dtype=np.int32)
        mask[:n, :n] = 1
        neis_mask.append(mask)

    ped = torch.tensor(np.stack(ped_items, axis=0), dtype=torch.float32, device=device)
    neis = torch.tensor(np.stack(neis_pad, axis=0), dtype=torch.float32, device=device)
    mask = torch.tensor(np.stack(neis_mask, axis=0), dtype=torch.int32, device=device)
    return ped, neis, mask, meta_items


def _record_from_prediction(
    *,
    dataset: str,
    split: str,
    record_index: int,
    prediction_abs: np.ndarray,
    top_scores: Optional[np.ndarray],
    meta: Mapping[str, np.ndarray],
    sdd_scale_factor: float,
) -> Dict[str, Any]:
    obs_abs = np.asarray(meta["obs_abs"], dtype=np.float32)[None, :, :]
    future_abs = np.asarray(meta["future_abs"], dtype=np.float32)[None, :, :]
    pred_abs = np.asarray(prediction_abs, dtype=np.float32)[:, None, :, :]
    coordinate_scale_factor = float(sdd_scale_factor) if str(dataset) == "sdd" else 1.0
    if coordinate_scale_factor != 1.0:
        obs_abs = obs_abs * coordinate_scale_factor
        future_abs = future_abs * coordinate_scale_factor
        pred_abs = pred_abs * coordinate_scale_factor
    pred_rel = pred_abs - obs_abs[None, :, -1:, :]
    features = build_moflow_eth_feature_arrays(obs_abs, future_abs, rotate=False)
    agent_mask = np.ones((1,), dtype=np.int64)
    social = compute_past_social_risk_features(obs_abs, agent_mask)
    record: Dict[str, Any] = {
        "dataset": str(dataset),
        "split": str(split),
        "record_index": int(record_index),
        "obs_abs": torch.from_numpy(obs_abs[0]),
        "future_abs": torch.from_numpy(future_abs[0]),
        "prediction_abs": torch.from_numpy(pred_abs),
        "prediction_rel": torch.from_numpy(pred_rel),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
        "coordinate_scale_factor": float(coordinate_scale_factor),
    }
    if top_scores is not None:
        record["top_scores"] = torch.from_numpy(np.asarray(top_scores, dtype=np.float32))
    return record


@torch.no_grad()
def _predict_records(
    *,
    model: torch.nn.Module,
    scenarios: Sequence[Any],
    motion_modes: torch.Tensor,
    dataset: str,
    split: str,
    k: int,
    obs_len: int,
    pred_len: int,
    dist_threshold: float,
    batch_size: int,
    max_records: Optional[int],
    device: torch.device,
    sdd_scale_factor: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    limit = len(scenarios) if max_records is None else min(int(max_records), len(scenarios))
    for start in range(0, limit, int(batch_size)):
        batch_items = scenarios[start : min(start + int(batch_size), limit)]
        ped, neis, mask, metas = _collate_items(
            batch_items,
            obs_len=int(obs_len),
            dist_threshold=float(dist_threshold),
            device=device,
        )
        ped_obs = ped[:, :obs_len]
        neis_obs = neis[:, :, :obs_len]
        pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True, num_k=int(k))
        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], int(k), int(pred_len), 2)

        rot = torch.tensor(np.stack([meta["rot_mat"] for meta in metas], axis=0), dtype=torch.float32, device=device)
        origin = torch.tensor(np.stack([meta["origin"] for meta in metas], axis=0), dtype=torch.float32, device=device)
        pred_abs = torch.einsum("bktd,bde->bkte", pred_trajs, rot.transpose(1, 2)) + origin[:, None, None, :]

        score_tensor = scores if scores.ndim == 2 else scores.unsqueeze(0)
        top_score_values = torch.topk(score_tensor, k=int(k), dim=-1).values
        top_probabilities = torch.softmax(top_score_values, dim=-1)

        pred_abs_np = pred_abs.detach().cpu().numpy()
        top_probs_np = top_probabilities.detach().cpu().numpy()
        for offset, meta in enumerate(metas):
            records.append(
                _record_from_prediction(
                    dataset=dataset,
                    split=split,
                    record_index=start + offset,
                    prediction_abs=pred_abs_np[offset],
                    top_scores=top_probs_np[offset],
                    meta=meta,
                    sdd_scale_factor=float(sdd_scale_factor),
                )
            )
        if (start + len(batch_items)) % 500 == 0 or start + len(batch_items) >= limit:
            print(f"[export_tutr_predictions] records={start + len(batch_items)}/{limit}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    tutr_root = Path(args.tutr_root).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else tutr_root / "dataset"
    hp_config_path = Path(args.hp_config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output_bundle).expanduser().resolve()

    if not tutr_root.exists():
        raise SystemExit(f"Missing TUTR root: {tutr_root.as_posix()}")
    if not hp_config_path.exists():
        raise SystemExit(f"Missing TUTR config: {hp_config_path.as_posix()}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing TUTR checkpoint: {checkpoint_path.as_posix()}")
    dataset_file = dataset_path / f"{args.dataset_name}_{args.split}.pkl"
    motion_modes_file = dataset_path / f"{args.dataset_name}_motion_modes.pkl"
    if not dataset_file.exists():
        raise SystemExit(f"Missing TUTR pkl data: {dataset_file.as_posix()}")
    if not motion_modes_file.exists():
        raise SystemExit(f"Missing TUTR motion modes: {motion_modes_file.as_posix()}")

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    hp_config = _load_python_config(hp_config_path)
    batch_size = int(args.batch_size or getattr(hp_config, "batch_size", 128))

    with _tutr_import_context(tutr_root):
        from model import TrajectoryModel

        scenarios = list(_load_pickle(dataset_file))
        motion_modes_np = np.asarray(_load_pickle(motion_modes_file), dtype=np.float32)
        motion_modes = torch.tensor(motion_modes_np, dtype=torch.float32, device=device)
        model = TrajectoryModel(
            in_size=2,
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            embed_size=int(hp_config.model_hidden_dim),
            enc_num_layers=2,
            int_num_layers_list=[1, 1],
            heads=4,
            forward_expansion=2,
        ).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        records = _predict_records(
            model=model,
            scenarios=scenarios,
            motion_modes=motion_modes,
            dataset=str(args.dataset_name),
            split=str(args.split),
            k=int(args.k),
            obs_len=int(args.obs_len),
            pred_len=int(args.pred_len),
            dist_threshold=float(getattr(hp_config, "dist_threshold", 2.0)),
            batch_size=batch_size,
            max_records=args.max_records,
            device=device,
            sdd_scale_factor=float(args.sdd_scale_factor),
        )

    if not records:
        raise SystemExit("No TUTR records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_tutr_predictions",
            "baseline": "TUTR",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tutr_root": tutr_root.as_posix(),
            "dataset_path": dataset_path.as_posix(),
            "dataset": str(args.dataset_name),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "num_records": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "hp_config": hp_config_path.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "coordinate_contract": "MoFlow SDD relative future convention: prediction_rel and fut_traj_original_scale are relative to the last observed position.",
            "sdd_scale_factor": float(args.sdd_scale_factor),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
