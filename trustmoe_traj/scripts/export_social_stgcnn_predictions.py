"""Export Social-STGCNN prediction bundles for AFC evaluation.

The upstream Social-STGCNN ``test.py`` only prints ADE/FDE. This adapter keeps
its original inference logic, samples K futures, and stores a compact bundle
with observed past, GT future, and predictions in both absolute and
last-observation-relative coordinates.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist
from torch.utils.data import DataLoader

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Social-STGCNN K-sample predictions.")
    parser.add_argument("--social-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["eth", "hotel", "univ", "zara1", "zara2", "sdd"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_social_modules(social_root: Path) -> Dict[str, Any]:
    root = social_root.resolve()
    sys.path.insert(0, str(root))
    try:
        utils = importlib.import_module("utils")
        metrics = importlib.import_module("metrics")
        model_mod = importlib.import_module("model")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return {
        "TrajectoryDataset": utils.TrajectoryDataset,
        "seq_to_nodes": metrics.seq_to_nodes,
        "nodes_rel_to_nodes_abs": metrics.nodes_rel_to_nodes_abs,
        "social_stgcnn": model_mod.social_stgcnn,
    }


def _to_record(
    *,
    scene_index: int,
    obs_abs_time_major: np.ndarray,
    future_abs_time_major: np.ndarray,
    pred_abs_modes_time_major: np.ndarray,
) -> Dict[str, Any]:
    # Upstream arrays are [T,A,2] and [K,T,A,2]. TrustMoE evaluators use
    # [A,T,2] and [K,A,T,2].
    obs_abs = np.asarray(obs_abs_time_major, dtype=np.float32).transpose(1, 0, 2)
    future_abs = np.asarray(future_abs_time_major, dtype=np.float32).transpose(1, 0, 2)
    pred_abs = np.asarray(pred_abs_modes_time_major, dtype=np.float32).transpose(0, 2, 1, 3)

    if obs_abs.ndim != 3 or future_abs.ndim != 3 or pred_abs.ndim != 4:
        raise ValueError(
            "Unexpected Social-STGCNN record shapes: "
            f"obs={obs_abs.shape}, future={future_abs.shape}, pred={pred_abs.shape}"
        )
    if int(obs_abs.shape[0]) != int(future_abs.shape[0]) or int(obs_abs.shape[0]) != int(pred_abs.shape[1]):
        raise ValueError(
            "Agent count mismatch: "
            f"obs={obs_abs.shape}, future={future_abs.shape}, pred={pred_abs.shape}"
        )

    features = build_moflow_eth_feature_arrays(obs_abs, future_abs, rotate=False)
    social_features = compute_past_social_risk_features(obs_abs, np.ones((obs_abs.shape[0],), dtype=np.int64))
    last_obs = obs_abs[:, -1:, :]
    pred_rel = (pred_abs - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    agent_mask = np.ones((obs_abs.shape[0],), dtype=np.int64)

    return {
        "scene_index": int(scene_index),
        "obs_abs": torch.from_numpy(obs_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(pred_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(pred_rel),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social_features.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
    }


@torch.no_grad()
def _predict_records(
    *,
    modules: Dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int,
    max_scenes: int | None,
) -> List[Dict[str, Any]]:
    seq_to_nodes = modules["seq_to_nodes"]
    nodes_rel_to_nodes_abs = modules["nodes_rel_to_nodes_abs"]
    records: List[Dict[str, Any]] = []

    model.eval()
    for scene_index, batch in enumerate(loader):
        if max_scenes is not None and scene_index >= int(max_scenes):
            break

        batch = [tensor.to(device) for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            _pred_traj_gt_rel,
            _non_linear_ped,
            _loss_mask,
            V_obs,
            A_obs,
            V_tr,
            _A_tr,
        ) = batch

        num_objects = int(obs_traj_rel.shape[1])
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1).squeeze()
        V_tr = V_tr.squeeze()
        V_pred = V_pred[:, :num_objects, :]
        V_tr = V_tr[:, :num_objects, :]

        sx = torch.exp(V_pred[:, :, 2])
        sy = torch.exp(V_pred[:, :, 3])
        corr = torch.tanh(V_pred[:, :, 4])
        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2, device=device, dtype=V_pred.dtype)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]
        mvnormal = torchdist.MultivariateNormal(mean, cov)

        V_x = seq_to_nodes(obs_traj.detach().cpu().numpy().copy())
        obs_abs = nodes_rel_to_nodes_abs(
            V_obs.detach().cpu().numpy().squeeze().copy(),
            V_x[0, :, :].copy(),
        )
        future_abs = nodes_rel_to_nodes_abs(
            V_tr.detach().cpu().numpy().squeeze().copy(),
            V_x[-1, :, :].copy(),
        )

        pred_modes: List[np.ndarray] = []
        for _sample_index in range(int(k)):
            pred_rel = mvnormal.sample()
            pred_abs = nodes_rel_to_nodes_abs(
                pred_rel.detach().cpu().numpy().squeeze().copy(),
                V_x[-1, :, :].copy(),
            )
            pred_modes.append(np.asarray(pred_abs, dtype=np.float32))

        records.append(
            _to_record(
                scene_index=scene_index,
                obs_abs_time_major=np.asarray(obs_abs, dtype=np.float32),
                future_abs_time_major=np.asarray(future_abs, dtype=np.float32),
                pred_abs_modes_time_major=np.stack(pred_modes, axis=0),
            )
        )

        if (scene_index + 1) % 50 == 0:
            print(f"[export_social_stgcnn_predictions] processed_scenes={scene_index + 1}")

    return records


def main() -> None:
    args = build_parser().parse_args()
    if int(args.k) <= 0:
        raise SystemExit("--k must be positive")

    _set_seed(int(args.seed))
    social_root = Path(args.social_root).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else social_root / "checkpoint" / f"social-stgcnn-{args.dataset}"
    )
    args_path = checkpoint_dir / "args.pkl"
    model_path = checkpoint_dir / "val_best.pth"
    if not args_path.exists():
        raise SystemExit(f"Missing args.pkl: {args_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing val_best.pth: {model_path}")

    modules = _load_social_modules(social_root)
    with args_path.open("rb") as handle:
        model_args = pickle.load(handle)
    if str(model_args.dataset) != str(args.dataset):
        raise SystemExit(f"Checkpoint dataset={model_args.dataset!r} does not match --dataset={args.dataset!r}")

    data_dir = social_root / "datasets" / str(model_args.dataset) / str(args.split)
    if not data_dir.exists():
        raise SystemExit(f"Missing Social-STGCNN data split: {data_dir}")

    dataset = modules["TrajectoryDataset"](
        str(data_dir) + "/",
        obs_len=int(model_args.obs_seq_len),
        pred_len=int(model_args.pred_seq_len),
        skip=1,
        norm_lap_matr=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers))

    device = _resolve_device(str(args.device))
    model = modules["social_stgcnn"](
        n_stgcnn=int(model_args.n_stgcnn),
        n_txpcnn=int(model_args.n_txpcnn),
        output_feat=int(model_args.output_size),
        seq_len=int(model_args.obs_seq_len),
        kernel_size=int(model_args.kernel_size),
        pred_seq_len=int(model_args.pred_seq_len),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(
        "[export_social_stgcnn_predictions] "
        f"dataset={args.dataset} split={args.split} scenes={len(dataset)} k={args.k} device={device} "
        f"checkpoint={checkpoint_dir.as_posix()}"
    )
    records = _predict_records(
        modules=modules,
        model=model,
        loader=loader,
        device=device,
        k=int(args.k),
        max_scenes=args.max_scenes,
    )
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_social_stgcnn_predictions",
            "baseline": "Social-STGCNN",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "social_root": social_root.as_posix(),
            "checkpoint_dir": checkpoint_dir.as_posix(),
            "dataset": str(args.dataset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "obs_len": int(model_args.obs_seq_len),
            "pred_len": int(model_args.pred_seq_len),
            "num_scenes": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "max_scenes": None if args.max_scenes is None else int(args.max_scenes),
        },
        "records": records,
    }
    output_path = Path(args.output_bundle).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
