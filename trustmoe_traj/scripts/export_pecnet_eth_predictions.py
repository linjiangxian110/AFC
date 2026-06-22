"""Export PECNet predictions on TrustMoE ETH-UCY subsets."""

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

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)
from trustmoe_traj.scripts.run_eval import DEFAULT_DATA_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PECNet K-sample predictions on ETH-UCY subsets.")
    parser.add_argument("--pecnet-root", type=str, required=True)
    parser.add_argument("--load-file", type=str, default="PECNET_social_model1.pt")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--subset", type=str, required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=1)
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


def _sample_to_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    raise TypeError(f"Unsupported ETH sample type: {type(sample)!r}")


def _load_pecnet_modules(pecnet_root: Path) -> Dict[str, Any]:
    utils_root = pecnet_root.resolve() / "utils"
    sys.path.insert(0, str(utils_root))
    try:
        models = importlib.import_module("models")
    finally:
        try:
            sys.path.remove(str(utils_root))
        except ValueError:
            pass
    return {"PECNet": models.PECNet}


def _record_from_prediction(
    *,
    scene_index: int,
    past_abs: np.ndarray,
    future_abs: np.ndarray,
    pred_shifted_modes: np.ndarray,
) -> Dict[str, Any]:
    origin = past_abs[:, :1, :]
    past_shifted = (past_abs - origin).astype(np.float32, copy=False)
    future_shifted = (future_abs - origin).astype(np.float32, copy=False)
    pred_shifted = np.asarray(pred_shifted_modes, dtype=np.float32)
    last_obs = past_shifted[:, -1:, :]
    pred_rel = (pred_shifted - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    features = build_moflow_eth_feature_arrays(past_shifted, future_shifted, rotate=False)
    agent_mask = np.ones((past_shifted.shape[0],), dtype=np.int64)
    social = compute_past_social_risk_features(past_shifted, agent_mask)
    return {
        "scene_index": int(scene_index),
        "obs_abs": torch.from_numpy(past_shifted.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_shifted.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(pred_shifted.astype(np.float32, copy=False)),
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
    model: torch.nn.Module,
    samples: List[Any],
    hyper_params: Mapping[str, Any],
    device: torch.device,
    k: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    past_length = int(hyper_params["past_length"])
    future_length = int(hyper_params["future_length"])
    data_scale = float(hyper_params["data_scale"])
    model.eval()

    for scene_index, sample in enumerate(samples):
        mapping = _sample_to_mapping(sample)
        past_abs = np.asarray(mapping["past_traj"], dtype=np.float32)
        future_abs = np.asarray(mapping["future_traj"], dtype=np.float32)
        agent_mask = np.asarray(mapping.get("agent_mask", np.ones((past_abs.shape[0],), dtype=np.int64)), dtype=np.int64)
        active = agent_mask.astype(bool)
        past_abs = past_abs[active]
        future_abs = future_abs[active]
        if int(past_abs.shape[0]) <= 0:
            continue

        origin = past_abs[:, :1, :]
        traj_shifted = np.concatenate([past_abs - origin, future_abs - origin], axis=1).astype(np.float32, copy=False)
        traj_scaled = torch.as_tensor(traj_shifted * data_scale, dtype=torch.float64, device=device)
        initial_pos = torch.as_tensor(past_abs[:, past_length - 1, :] / 1000.0, dtype=torch.float64, device=device)
        mask = torch.ones((past_abs.shape[0], past_abs.shape[0]), dtype=torch.float64, device=device)
        x = traj_scaled[:, :past_length, :]
        x_flat = x.contiguous().view(-1, x.shape[1] * x.shape[2])

        pred_modes: List[np.ndarray] = []
        for _sample_index in range(int(k)):
            dest_recon = model.forward(x_flat, initial_pos, device=device)
            interpolated_future = model.predict(x_flat, dest_recon, mask, initial_pos)
            predicted_future = torch.cat((interpolated_future, dest_recon), dim=1)
            predicted_future = predicted_future.reshape(-1, future_length, 2)
            pred_modes.append((predicted_future.detach().cpu().numpy() / data_scale).astype(np.float32, copy=False))

        records.append(
            _record_from_prediction(
                scene_index=scene_index,
                past_abs=past_abs,
                future_abs=future_abs,
                pred_shifted_modes=np.stack(pred_modes, axis=0),
            )
        )
        if (scene_index + 1) % 50 == 0:
            print(f"[export_pecnet_eth_predictions] processed_scenes={scene_index + 1}/{len(samples)}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    if int(args.k) <= 0:
        raise SystemExit("--k must be positive")
    _set_seed(int(args.seed))
    pecnet_root = Path(args.pecnet_root).expanduser().resolve()
    checkpoint_path = pecnet_root / "saved_models" / str(args.load_file)
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing PECNet checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hyper_params = checkpoint["hyper_params"]
    device = _resolve_device(str(args.device), int(hyper_params.get("gpu_index", 0)))
    pecnet_cls = _load_pecnet_modules(pecnet_root)["PECNet"]
    model = pecnet_cls(
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

    data_root = Path(args.data_root).expanduser().resolve()
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=str(args.subset),
            split=str(args.split),
            min_agents=int(args.min_agents),
            prefer_cache=False,
        )
    )
    limit = len(dataset) if args.max_scenes is None else min(int(args.max_scenes), len(dataset))
    samples = [dataset[index] for index in range(limit)]
    print(
        "[export_pecnet_eth_predictions] "
        f"subset={args.subset} split={args.split} scenes={len(samples)} k={args.k} device={device} "
        f"checkpoint={checkpoint_path.as_posix()}"
    )
    records = _predict_records(model=model, samples=samples, hyper_params=hyper_params, device=device, k=int(args.k))
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_pecnet_eth_predictions",
            "baseline": "PECNet",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pecnet_root": pecnet_root.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "load_file": str(args.load_file),
            "data_root": data_root.as_posix(),
            "subset": str(args.subset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "num_scenes": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "max_scenes": None if args.max_scenes is None else int(args.max_scenes),
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
