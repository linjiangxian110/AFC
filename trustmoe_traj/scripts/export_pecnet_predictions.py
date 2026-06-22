"""Export PECNet K-sample prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PECNet K-sample predictions.")
    parser.add_argument("--pecnet-root", type=str, required=True)
    parser.add_argument("--load-file", type=str, default="PECNET_social_model1.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--dataset-label", type=str, default="all")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


def _resolve_device(raw: str, gpu_index: int = 0) -> torch.device:
    if raw == "auto":
        raw = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
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


def _load_pecnet_modules(pecnet_root: Path) -> Dict[str, Any]:
    utils_root = pecnet_root.resolve() / "utils"
    sys.path.insert(0, str(utils_root))
    try:
        models = importlib.import_module("models")
        social_utils = importlib.import_module("social_utils")
    finally:
        try:
            sys.path.remove(str(utils_root))
        except ValueError:
            pass
    return {
        "PECNet": models.PECNet,
        "SocialDataset": social_utils.SocialDataset,
    }


def _normalise_pecnet_dataset(dataset: Any, *, data_scale: float) -> None:
    """Match PECNet's official test preprocessing in-place."""

    for traj in dataset.trajectory_batches:
        traj -= traj[:, :1, :]
        traj *= float(data_scale)


def _record_from_arrays(
    *,
    batch_index: int,
    traj_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    data_scale: float,
) -> Dict[str, Any]:
    # PECNet evaluates in the shifted coordinate system divided by data_scale.
    # This keeps ADE/FDE comparable with the official script while still giving
    # AFC a consistent observed-past/future geometry.
    scale = float(data_scale)
    traj = np.asarray(traj_scaled, dtype=np.float32) / scale
    pred_abs = np.asarray(prediction_scaled, dtype=np.float32) / scale
    past_abs = traj[:, :8, :]
    future_abs = traj[:, 8:, :]
    last_obs = past_abs[:, -1:, :]
    pred_rel = (pred_abs - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    features = build_moflow_eth_feature_arrays(past_abs, future_abs, rotate=False)
    agent_mask = np.ones((past_abs.shape[0],), dtype=np.int64)
    social = compute_past_social_risk_features(past_abs, agent_mask)
    return {
        "batch_index": int(batch_index),
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
def _predict_bundle_records(
    *,
    model: torch.nn.Module,
    dataset: Any,
    hyper_params: Mapping[str, Any],
    device: torch.device,
    k: int,
    max_batches: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    past_length = int(hyper_params["past_length"])
    future_length = int(hyper_params["future_length"])
    data_scale = float(hyper_params["data_scale"])
    model.eval()

    iterator = zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches)
    for batch_index, (traj_np, mask_np, initial_pos_np) in enumerate(iterator):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        traj = torch.as_tensor(traj_np, dtype=torch.float64, device=device)
        mask = torch.as_tensor(mask_np, dtype=torch.float64, device=device)
        initial_pos = torch.as_tensor(initial_pos_np, dtype=torch.float64, device=device)
        x = traj[:, :past_length, :]
        x_flat = x.contiguous().view(-1, x.shape[1] * x.shape[2])

        pred_modes: List[np.ndarray] = []
        for _sample_index in range(int(k)):
            dest_recon = model.forward(x_flat, initial_pos, device=device)
            interpolated_future = model.predict(x_flat, dest_recon, mask, initial_pos)
            predicted_future = torch.cat((interpolated_future, dest_recon), dim=1)
            predicted_future = predicted_future.reshape(-1, future_length, 2)
            pred_modes.append(predicted_future.detach().cpu().numpy().astype(np.float32, copy=False))

        records.append(
            _record_from_arrays(
                batch_index=batch_index,
                traj_scaled=np.asarray(traj_np, dtype=np.float32),
                prediction_scaled=np.stack(pred_modes, axis=0),
                data_scale=data_scale,
            )
        )
        if (batch_index + 1) % 20 == 0:
            print(f"[export_pecnet_predictions] processed_batches={batch_index + 1}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    if int(args.k) <= 0:
        raise SystemExit("--k must be positive")
    _set_seed(int(args.seed))
    pecnet_root = Path(args.pecnet_root).expanduser().resolve()
    modules = _load_pecnet_modules(pecnet_root)
    checkpoint_path = pecnet_root / "saved_models" / str(args.load_file)
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing PECNet checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hyper_params = checkpoint["hyper_params"]
    device = _resolve_device(str(args.device), int(hyper_params.get("gpu_index", 0)))
    model = modules["PECNet"](
        hyper_params["enc_past_size"],
        hyper_params["enc_dest_size"],
        hyper_params["enc_latent_size"],
        hyper_params["dec_size"],
        hyper_params["predictor_hidden_size"],
        hyper_params["non_local_theta_size"],
        hyper_params["non_local_phi_size"],
        hyper_params["non_local_g_size"],
        hyper_params["fdim"],
        hyper_params["zdim"],
        hyper_params["nonlocal_pools"],
        hyper_params["non_local_dim"],
        hyper_params["sigma"],
        hyper_params["past_length"],
        hyper_params["future_length"],
        False,
    ).double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cwd = Path.cwd()
    try:
        # PECNet SocialDataset uses relative ../social_pool_data paths.
        scripts_root = pecnet_root / "scripts"
        scripts_root.mkdir(exist_ok=True)
        import os

        os.chdir(scripts_root)
        dataset = modules["SocialDataset"](
            set_name=str(args.split),
            b_size=int(hyper_params["test_b_size"] if args.split == "test" else hyper_params["train_b_size"]),
            t_tresh=int(hyper_params["time_thresh"]),
            d_tresh=int(hyper_params["dist_thresh"]),
            verbose=False,
        )
    finally:
        import os

        os.chdir(cwd)

    _normalise_pecnet_dataset(dataset, data_scale=float(hyper_params["data_scale"]))
    records = _predict_bundle_records(
        model=model,
        dataset=dataset,
        hyper_params=hyper_params,
        device=device,
        k=int(args.k),
        max_batches=args.max_batches,
    )
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_pecnet_predictions",
            "baseline": "PECNet",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pecnet_root": pecnet_root.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "load_file": str(args.load_file),
            "dataset_label": str(args.dataset_label),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "num_batches": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "max_batches": None if args.max_batches is None else int(args.max_batches),
        },
        "hyper_params": dict(hyper_params),
        "records": records,
    }
    output_path = Path(args.output_bundle).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
