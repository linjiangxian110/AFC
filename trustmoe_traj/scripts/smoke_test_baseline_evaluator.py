"""Smoke test for TrustMoE-Traj baseline evaluator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal evaluator smoke test for MoFlow baselines.")
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--latency-runs", type=int, default=3, help="How many repeated predict calls to time")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    return parser


def _load_raw_samples(subset: str, split: str, num_scenes: int) -> List[Dict]:
    data_root = Path(__file__).resolve().parent.parent / "data" / "ETH"
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=subset,
            split=split,
            prefer_cache=True,
        )
    )
    limit = min(num_scenes, len(dataset))
    if limit <= 0:
        raise ValueError("Dataset is empty, cannot run evaluator smoke test")
    samples = [dataset[index] for index in range(limit)]
    return samples


def _measure_predict_latency_ms(fn, runs: int) -> Tuple[List[float], object]:
    latencies: List[float] = []
    last_output = None
    for _ in range(runs):
        start = time.perf_counter()
        last_output = fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
    return latencies, last_output


def main() -> None:
    args = build_parser().parse_args()
    samples = _load_raw_samples(args.subset, args.split, args.num_scenes)

    if args.sample_mode == "per_agent":
        agents = 1
    else:
        agents = args.agents or max(int(sample["past_traj"].shape[0]) for sample in samples)

    predictor_config = MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        device=args.device,
        num_to_gen=args.num_to_gen,
    )

    slow_predictor = MoFlowSlowPredictor(predictor_config)
    fast_predictor = MoFlowFastPredictor(predictor_config)

    normalization_stats = slow_predictor.infer_normalization_stats(samples)
    slow_batch = slow_predictor.build_moflow_batch(samples, normalization_stats=normalization_stats, as_torch=True)
    fast_batch = fast_predictor.build_moflow_batch(samples, normalization_stats=normalization_stats, as_torch=True)

    slow_latencies, slow_output = _measure_predict_latency_ms(
        lambda: slow_predictor.predict(slow_batch, return_all_states=False),
        args.latency_runs,
    )
    fast_latencies, fast_output = _measure_predict_latency_ms(
        lambda: fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen),
        args.latency_runs,
    )

    combined_output = {
        "fast_pred": fast_output.fast_pred,
        "slow_pred": slow_output.slow_pred,
        "final_pred": fast_output.fast_pred,
    }

    summary = evaluate_model_output(
        combined_output,
        fast_batch,
        miss_threshold=args.miss_threshold,
        latency_ms={
            "fast_pred": fast_latencies,
            "slow_pred": slow_latencies,
        },
    )

    print("[smoke_test_baseline_evaluator] success")
    print(f"subset={args.subset} split={args.split} sample_mode={args.sample_mode} raw_scenes={len(samples)}")
    print(f"normalization_stats={normalization_stats}")
    print(f"available_predictions={summary.available_predictions}")
    for key in sorted(summary.metrics):
        print(f"{key}={summary.metrics[key]}")


if __name__ == "__main__":
    main()
