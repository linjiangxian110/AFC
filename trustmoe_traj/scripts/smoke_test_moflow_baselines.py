"""Minimal smoke test for TrustMoE-Traj MoFlow baselines.

This script verifies the following chain:
ETH cache/raw samples -> TrustMoE transform -> MoFlow slow baseline
-> teacher latent target -> MoFlow fast baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal TrustMoE-Traj MoFlow baseline smoke test.")
    parser.add_argument("--subset", type=str, default="eth", help="ETH subset name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="ETH split")
    parser.add_argument("--num-scenes", type=int, default=2, help="How many raw TrustMoE scenes to use")
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None, help="Fixed agent count for MoFlow wrapper")
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--rotate", action="store_true", help="Whether to rotate trajectories in transform")
    parser.add_argument("--rotate-time-frame", type=int, default=0, help="Rotation anchor frame")
    parser.add_argument("--num-to-gen", type=int, default=1, help="num_to_gen for IMLE forward")
    return parser


def _select_samples(dataset: ETHTrajectoryDataset, num_scenes: int) -> List[Dict]:
    limit = min(num_scenes, len(dataset))
    if limit <= 0:
        raise ValueError("Dataset is empty, cannot run smoke test")
    return [dataset[index] for index in range(limit)]


def _tensor_shape(tensor: torch.Tensor | None) -> List[int] | None:
    return None if tensor is None else list(tensor.shape)


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(__file__).resolve().parent.parent / "data" / "ETH"
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=args.split,
            prefer_cache=True,
        )
    )
    raw_samples = _select_samples(dataset, args.num_scenes)

    if args.sample_mode == "per_agent":
        agents = 1
    else:
        agents = args.agents or max(int(sample["past_traj"].shape[0]) for sample in raw_samples)

    shared_config = MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        rotate=args.rotate,
        rotate_time_frame=args.rotate_time_frame,
        device=args.device,
        num_to_gen=args.num_to_gen,
    )

    slow_predictor = MoFlowSlowPredictor(shared_config)
    fast_predictor = MoFlowFastPredictor(shared_config)

    normalization_stats = slow_predictor.infer_normalization_stats(raw_samples)
    slow_batch = slow_predictor.build_moflow_batch(raw_samples, normalization_stats=normalization_stats, as_torch=True)
    fast_batch = fast_predictor.build_moflow_batch(raw_samples, normalization_stats=normalization_stats, as_torch=True)

    slow_loss = slow_predictor.compute_loss(slow_batch, log_dict={"cur_epoch": 0})
    slow_output = slow_predictor.predict(slow_batch, return_all_states=True)

    teacher_latent = slow_output.extras["teacher_latent"]
    fast_loss = fast_predictor.compute_loss(fast_batch, teacher_latent=teacher_latent)
    fast_output = fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen)

    _check_finite("slow_loss", slow_loss["loss"])
    _check_finite("fast_loss", fast_loss["loss"])
    _check_finite("slow_pred", slow_output.slow_pred)
    _check_finite("fast_pred", fast_output.fast_pred)

    print("[smoke_test_moflow_baselines] success")
    print(f"subset={args.subset} split={args.split} sample_mode={args.sample_mode} raw_scenes={len(raw_samples)}")
    print(f"dataset_loaded_from_cache={dataset.loaded_from_cache}")
    print(f"shared_agents={agents}")
    print(f"normalization_stats={normalization_stats}")
    print(f"slow_batch_past_shape={_tensor_shape(slow_batch['past_traj'])}")
    print(f"fast_batch_past_shape={_tensor_shape(fast_batch['past_traj'])}")
    print(f"teacher_latent_shape={_tensor_shape(teacher_latent)}")
    print(f"slow_pred_shape={_tensor_shape(slow_output.slow_pred)}")
    print(f"fast_pred_shape={_tensor_shape(fast_output.fast_pred)}")
    print(f"slow_loss={float(slow_loss['loss'].detach().cpu()):.6f}")
    print(f"fast_loss={float(fast_loss['loss'].detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
