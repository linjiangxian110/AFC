"""Benchmark AFC evaluation-time cost on real ETH-UCY or SDD records.

This script intentionally avoids model inference and training. It builds the
standard AFC bank from the training split, constructs deterministic prediction
sets on the test split, and times calls to ``metrics_for_prediction``. The
result is a protocol-level runtime estimate suitable for reporting AFC
evaluation overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.data.adapters.sdd import SDDAdapterConfig, SDDTrajectoryDataset
from trustmoe_traj.data.transforms import build_moflow_eth_batch, infer_moflow_eth_num_agents
from trustmoe_traj.scripts.analogical_future_coverage import (
    build_eth_analogical_future_bank,
    build_sdd_analogical_future_bank,
    split_float_list,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark AFC evaluation-time runtime.")
    parser.add_argument("--dataset", choices=["eth", "sdd"], required=True)
    parser.add_argument("--subset", default="zara2", help="ETH-UCY subset; ignored for SDD.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--sample-mode", default=None, choices=["per_agent", "per_scene"])
    parser.add_argument("--data-norm", default=None, choices=["original", "min_max"])
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--prefer-cache", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=300)
    parser.add_argument("--batch-records", type=int, default=64)
    parser.add_argument("--afc-batch-records", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", default="0.3,0.5,1.0")
    parser.add_argument("--afc-feature", default="full_past_social")
    parser.add_argument("--branches", default="cv,gt_repeat")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]


def _constant_velocity_prediction(batch: Mapping[str, Any], *, keep_k: int) -> torch.Tensor:
    past = torch.as_tensor(batch["past_traj_original_scale"], dtype=torch.float32)
    future = torch.as_tensor(batch["fut_traj_original_scale"], dtype=torch.float32)
    past_rel = past[..., 2:4]
    if int(past_rel.shape[-2]) >= 2:
        velocity = past_rel[..., -1, :] - past_rel[..., -2, :]
    else:
        velocity = torch.zeros_like(past_rel[..., -1, :])
    steps = torch.arange(1, int(future.shape[-2]) + 1, dtype=past.dtype)
    prediction = velocity[:, :, None, :] * steps[None, None, :, None]
    return prediction[:, None, ...].expand(
        int(prediction.shape[0]),
        int(keep_k),
        int(prediction.shape[1]),
        int(prediction.shape[2]),
        int(prediction.shape[3]),
    ).contiguous()


def _gt_repeat_prediction(batch: Mapping[str, Any], *, keep_k: int) -> torch.Tensor:
    future = torch.as_tensor(batch["fut_traj_original_scale"], dtype=torch.float32)
    return future[:, None, ...].expand(
        int(future.shape[0]),
        int(keep_k),
        int(future.shape[1]),
        int(future.shape[2]),
        int(future.shape[3]),
    ).contiguous()


def _select_samples(dataset: Sequence[Any], limit: Optional[int]) -> List[Any]:
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    return [dataset[index] for index in range(count)]


def _load_eval_samples(args: argparse.Namespace) -> List[Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    if args.dataset == "sdd":
        dataset = SDDTrajectoryDataset(
            SDDAdapterConfig(
                data_root=data_root,
                split=str(args.split),
                max_samples=int(args.max_records) if args.max_records is not None else None,
            )
        )
        return _select_samples(dataset, None)
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=str(args.subset),
            split=str(args.split),
            min_agents=int(args.min_agents),
            prefer_cache=bool(args.prefer_cache),
        )
    )
    return _select_samples(dataset, args.max_records)


def _build_bank(args: argparse.Namespace, eps_values: Sequence[float]):
    data_root = Path(args.data_root).expanduser().resolve()
    if args.dataset == "sdd":
        sample_mode = str(args.sample_mode or "per_scene")
        data_norm = str(args.data_norm or "original")
        return build_sdd_analogical_future_bank(
            data_root=data_root,
            train_split=str(args.train_split),
            sample_mode=sample_mode,
            data_norm=data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            normalization_stats=None,
            max_train_scenes=args.max_train_records,
            batch_scenes=int(args.afc_batch_records),
            top_m=int(args.afc_top_m),
            eps_values=eps_values,
            feature_variant=str(args.afc_feature),
        )
    sample_mode = str(args.sample_mode or "per_agent")
    data_norm = str(args.data_norm or "original")
    return build_eth_analogical_future_bank(
        data_root=data_root,
        subset=str(args.subset),
        train_split=str(args.train_split),
        sample_mode=sample_mode,
        data_norm=data_norm,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        normalization_stats=None,
        min_agents=int(args.min_agents),
        prefer_cache=bool(args.prefer_cache),
        max_train_scenes=args.max_train_records,
        batch_scenes=int(args.afc_batch_records),
        top_m=int(args.afc_top_m),
        eps_values=eps_values,
        feature_variant=str(args.afc_feature),
    )


def _make_batch(args: argparse.Namespace, samples: Sequence[Any]) -> Dict[str, Any]:
    sample_mode = str(args.sample_mode or ("per_scene" if args.dataset == "sdd" else "per_agent"))
    data_norm = str(args.data_norm or "original")
    fixed_num_agents = 1 if sample_mode == "per_agent" else infer_moflow_eth_num_agents(samples, sample_mode=sample_mode)
    return build_moflow_eth_batch(
        samples,
        data_norm=data_norm,
        sample_mode=sample_mode,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        fixed_num_agents=fixed_num_agents,
        normalization_stats=None,
        as_torch=True,
    )


def _prediction_branches(batch: Mapping[str, Any], *, keep_k: int, branches: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for branch in branches:
        if branch == "cv":
            out["cv20_pred"] = _constant_velocity_prediction(batch, keep_k=keep_k)
        elif branch == "gt_repeat":
            out["gt_repeat20_pred"] = _gt_repeat_prediction(batch, keep_k=keep_k)
        else:
            raise ValueError(f"Unsupported benchmark branch {branch!r}")
    return out


def _write_csv(path: Path, row: Mapping[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "subset",
        "split",
        "eval_records",
        "valid_agents",
        "branch_count",
        "bank_size",
        "feature_dim",
        "top_m",
        "eps",
        "bank_seconds",
        "eval_seconds",
        "total_seconds",
        "records_per_second",
        "metric_calls_per_second",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = build_parser().parse_args()
    eps_values = split_float_list(str(args.afc_eps))
    branches = _split_items(str(args.branches))
    if not branches:
        raise SystemExit("--branches must not be empty")

    total_start = time.perf_counter()
    eval_samples = _load_eval_samples(args)
    if not eval_samples:
        raise SystemExit("No evaluation samples were loaded")

    bank_start = time.perf_counter()
    afc_bank = _build_bank(args, eps_values)
    bank_seconds = time.perf_counter() - bank_start

    eval_start = time.perf_counter()
    valid_agents = 0
    metric_calls = 0
    retrieval_confidence_values: List[float] = []
    for chunk in _iter_chunks(eval_samples, int(args.batch_records)):
        batch = _make_batch(args, chunk)
        mask = torch.as_tensor(batch["agent_mask"]).bool()
        valid_agents += int(mask.sum().item())
        predictions = _prediction_branches(batch, keep_k=int(args.k), branches=branches)
        for _branch_name, prediction in predictions.items():
            metrics = afc_bank.metrics_for_prediction(prediction, batch)
            metric_calls += 1
            confidence = metrics.get("afc_retrieval_confidence")
            if confidence is not None:
                retrieval_confidence_values.append(float(confidence))
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - total_start

    row = {
        "dataset": str(args.dataset),
        "subset": str(args.subset) if args.dataset != "sdd" else "sdd",
        "split": str(args.split),
        "eval_records": int(len(eval_samples)),
        "valid_agents": int(valid_agents),
        "branch_count": int(len(branches)),
        "bank_size": int(afc_bank.bank_size),
        "feature_dim": int(afc_bank.feature_dim),
        "top_m": int(afc_bank.top_m),
        "eps": ",".join(f"{float(item):g}" for item in eps_values),
        "bank_seconds": float(bank_seconds),
        "eval_seconds": float(eval_seconds),
        "total_seconds": float(total_seconds),
        "records_per_second": float(valid_agents / eval_seconds) if eval_seconds > 0 else None,
        "metric_calls_per_second": float(metric_calls / eval_seconds) if eval_seconds > 0 else None,
    }
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.benchmark_afc_runtime",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "note": "AFC evaluation-time benchmark only; no model inference or training is included.",
        },
        "args": vars(args),
        "row": row,
        "retrieval_confidence_mean": (
            sum(retrieval_confidence_values) / len(retrieval_confidence_values)
            if retrieval_confidence_values
            else None
        ),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else output_json.with_suffix(".csv")
    _write_csv(output_csv, row)
    print(f"output_json={output_json.as_posix()}")
    print(f"output_csv={output_csv.as_posix()}")
    print(
        "AFC_RUNTIME "
        f"dataset={row['dataset']} subset={row['subset']} valid_agents={row['valid_agents']} "
        f"bank_size={row['bank_size']} bank_seconds={row['bank_seconds']:.3f} "
        f"eval_seconds={row['eval_seconds']:.3f} records_per_second={row['records_per_second']:.3f}"
    )


if __name__ == "__main__":
    main()
