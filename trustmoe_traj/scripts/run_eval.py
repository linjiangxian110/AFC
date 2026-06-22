"""Run baseline evaluation for TrustMoE-Traj MoFlow wrappers.

This script is designed as the first formal evaluation entrypoint for the
current baseline stage:
1. Load ETH samples from the TrustMoE main cache or raw files.
2. Build MoFlow-compatible batches with the shared transform layer.
3. Evaluate fast and/or slow baseline branches with the unified evaluator.
4. Optionally save the aggregated metrics to a JSON file.

It supports checkpoint-backed evaluation and an explicit random-init fallback
for smoke testing. Random-init results should not be treated as formal
baseline numbers.

The entrypoint now supports two evaluation protocols:
- ``trustmoe_internal`` keeps the main-project defaults that are convenient for
  unified system experiments.
- ``official_align`` tightens the data / normalization path so the resulting
  ADE/FDE numbers are closer to MoFlow's original ETH benchmark definition.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import (
    displacement_errors,
    evaluate_model_output,
    infer_ground_truth_from_batch,
    summarize_latency_ms,
)
from trustmoe_traj.evaluation.proxy_features import (
    compute_scene_motion_proxy_features,
    update_records_with_prediction_proxy_features,
)
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "ETH"
EVAL_PROTOCOLS: Sequence[str] = ("trustmoe_internal", "official_align")
NORMALIZATION_SOURCES: Sequence[str] = ("auto", "selected_samples", "train_split", "predictor_cfg")
ACCURACY_METRIC_NAMES: Sequence[str] = (
    "ADE_min",
    "FDE_min",
    "ADE_avg",
    "FDE_avg",
    "MissRate",
)
BENCHMARK_METRIC_NAMES: Sequence[str] = (
    "ADE_min",
    "FDE_min",
    "ADE_avg",
    "FDE_avg",
)


@dataclass(frozen=True)
class ProtocolSettings:
    protocol: str
    min_agents: int
    prefer_cache: bool
    normalization_source: str

    @property
    def comparable_metrics(self) -> Sequence[str]:
        return BENCHMARK_METRIC_NAMES

    @property
    def auxiliary_metrics(self) -> Sequence[str]:
        return ("MissRate", "Latency")


@dataclass
class BranchAccumulator:
    """Aggregate chunk-level evaluator outputs into a split-level summary."""

    prediction_field: str
    miss_threshold: float
    total_valid_agents: float = 0.0
    num_batches: int = 0
    weighted_metric_sums: Dict[str, float] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)

    def add_chunk(self, metrics: Mapping[str, float], latencies_ms: Iterable[float]) -> None:
        prefix = f"{self.prediction_field}_"
        valid_agents = float(metrics[f"{prefix}num_valid_agents"])
        if valid_agents <= 0:
            raise ValueError(f"{self.prediction_field} chunk has no valid agents")

        self.total_valid_agents += valid_agents
        self.num_batches += 1

        for metric_name in ACCURACY_METRIC_NAMES:
            key = f"{prefix}{metric_name}"
            self.weighted_metric_sums[key] = self.weighted_metric_sums.get(key, 0.0) + float(metrics[key]) * valid_agents

        self.latencies_ms.extend(float(item) for item in latencies_ms)

    def finalize(self) -> Dict[str, float]:
        if self.total_valid_agents <= 0:
            raise ValueError(f"{self.prediction_field} has no aggregated valid agents")

        prefix = f"{self.prediction_field}_"
        summary = {
            f"{prefix}num_valid_agents": float(self.total_valid_agents),
            f"{prefix}num_batches": float(self.num_batches),
            f"{prefix}miss_threshold": float(self.miss_threshold),
        }
        for metric_name in ACCURACY_METRIC_NAMES:
            key = f"{prefix}{metric_name}"
            summary[key] = float(self.weighted_metric_sums[key] / self.total_valid_agents)

        latency_summary = summarize_latency_ms(self.latencies_ms)
        summary.update({f"{prefix}{key}": value for key, value in latency_summary.items()})
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run formal baseline evaluation for MoFlow fast/slow wrappers.")
    parser.add_argument("--protocol", type=str, default="trustmoe_internal", choices=EVAL_PROTOCOLS)
    parser.add_argument("--baseline", type=str, default="both", choices=["fast", "slow", "both"])
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None, help="Required fixed agent count for per_scene mode")
    parser.add_argument("--min-agents", type=int, default=None, help="Minimum valid agents required for a raw scene")
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument(
        "--normalization-source",
        type=str,
        default="auto",
        choices=NORMALIZATION_SOURCES,
        help="auto / selected_samples / train_split / predictor_cfg",
    )
    parser.add_argument("--batch-scenes", type=int, default=8, help="How many raw TrustMoE scenes to evaluate per chunk")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on raw scenes for partial evaluation")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:0 ...")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=0)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.2,
        help="Distance threshold for prediction collision proxy in dataset coordinate units",
    )
    parser.add_argument("--latency-runs", type=int, default=1, help="Repeated predict calls per chunk for latency stats")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N chunks")
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--fast-checkpoint", type=str, default=None)
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--slow-sampling-steps", type=int, default=None, help="Override teacher sampling steps")
    parser.add_argument("--slow-solver", type=str, default=None, choices=["euler", "lin_poly"])
    parser.add_argument("--slow-lin-poly-p", type=int, default=None)
    parser.add_argument("--slow-lin-poly-long-step", type=int, default=None)
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow evaluation without checkpoints. Intended only for smoke tests.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON path for saving aggregated results")
    parser.add_argument(
        "--output-per-sample-json",
        type=str,
        default=None,
        help="Optional JSON path for saving per eval-item fast/slow ADE/FDE details.",
    )

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device!r}, but torch.cuda.is_available() is False")
    return str(resolved)


def _maybe_synchronize(device: str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _iter_chunks(items: Sequence[Mapping[str, Any]], chunk_size: int) -> Iterator[Sequence[Mapping[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _select_samples(dataset: ETHTrajectoryDataset, max_scenes: Optional[int]) -> List[Mapping[str, Any]]:
    if max_scenes is None:
        limit = len(dataset)
    else:
        limit = min(int(max_scenes), len(dataset))
    if limit <= 0:
        raise ValueError("Dataset is empty after applying max_scenes")
    return [dataset[index] for index in range(limit)]


def _infer_agents(samples: Sequence[Mapping[str, Any]], sample_mode: str, explicit_agents: Optional[int]) -> int:
    if sample_mode == "per_agent":
        return 1
    if explicit_agents is not None:
        return int(explicit_agents)
    return max(int(sample["past_traj"].shape[0]) for sample in samples)


def _count_selected_eval_items(samples: Sequence[Mapping[str, Any]], sample_mode: str) -> int:
    if sample_mode == "per_scene":
        return len(samples)

    num_items = 0
    for sample in samples:
        agent_mask = sample.get("agent_mask")
        if agent_mask is None:
            num_items += int(sample["past_traj"].shape[0])
        else:
            num_items += int(torch.as_tensor(agent_mask).sum().item())
    return num_items


def _resolve_protocol_settings(args: argparse.Namespace) -> ProtocolSettings:
    if args.protocol == "official_align":
        default_min_agents = 2
        default_prefer_cache = False
        default_norm_source = "train_split"
    else:
        default_min_agents = 1
        default_prefer_cache = True
        # Checkpoint-backed MoFlow baselines expect train-distribution min/max
        # statistics. Keep selected_samples only as an explicit diagnostic mode.
        default_norm_source = "train_split"

    min_agents = int(args.min_agents) if args.min_agents is not None else default_min_agents
    prefer_cache = bool(args.prefer_cache) if args.prefer_cache is not None else default_prefer_cache
    normalization_source = args.normalization_source
    if normalization_source == "auto":
        normalization_source = default_norm_source

    return ProtocolSettings(
        protocol=args.protocol,
        min_agents=min_agents,
        prefer_cache=prefer_cache,
        normalization_source=normalization_source,
    )


def _validate_protocol_assumptions(args: argparse.Namespace, protocol_settings: ProtocolSettings) -> None:
    if protocol_settings.protocol != "official_align":
        return

    if args.sample_mode != "per_agent":
        raise SystemExit(
            "protocol='official_align' for the current ETH MoFlow baselines requires "
            "--sample-mode per_agent. The original ETH original/*.pkl path is evaluated "
            "with agents=1 items, so per_scene will not match checkpoint dimensions."
        )

    if args.agents not in (None, 1):
        raise SystemExit(
            "protocol='official_align' ignores multi-agent batching for the current ETH baselines. "
            "Please omit --agents or set --agents 1."
        )


def _validate_checkpoint_requirements(args: argparse.Namespace) -> None:
    requested = []
    if args.baseline in ("slow", "both"):
        requested.append(("slow", args.slow_checkpoint))
    if args.baseline in ("fast", "both"):
        requested.append(("fast", args.fast_checkpoint))

    missing = [name for name, ckpt in requested if not ckpt]
    if missing and not args.allow_random_init:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing checkpoint path for baseline(s): {joined}. "
            "Provide --slow-checkpoint/--fast-checkpoint or pass --allow-random-init for smoke tests."
        )


def _measure_predict_latency_ms(fn, *, runs: int, device: str) -> tuple[List[float], Any]:
    latencies: List[float] = []
    last_output = None
    for _ in range(runs):
        _maybe_synchronize(device)
        start = time.perf_counter()
        last_output = fn()
        _maybe_synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies, last_output


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(item) for item in value]
    return value


def _scene_meta_to_dict(sample: Mapping[str, Any]) -> Dict[str, Any]:
    scene_meta = sample.get("scene_meta", {})
    if isinstance(scene_meta, Mapping):
        return dict(scene_meta)
    if hasattr(scene_meta, "to_dict"):
        return scene_meta.to_dict()
    return {"raw_scene_meta": str(scene_meta)}


def _active_agent_indices(sample: Mapping[str, Any]) -> List[int]:
    past = torch.as_tensor(sample["past_traj"])
    agent_mask = sample.get("agent_mask")
    if agent_mask is None:
        return list(range(int(past.shape[0])))

    mask = torch.as_tensor(agent_mask).reshape(-1).bool()
    if int(mask.numel()) != int(past.shape[0]):
        raise ValueError(f"agent_mask length mismatch: {int(mask.numel())} vs {int(past.shape[0])}")

    active = [int(idx) for idx, flag in enumerate(mask.tolist()) if bool(flag)]
    return active or list(range(int(past.shape[0])))


def _build_base_per_sample_records(
    *,
    samples: Sequence[Mapping[str, Any]],
    global_scene_indices: Sequence[int],
    sample_mode: str,
    eval_item_offset: int,
) -> tuple[Dict[tuple[int, int], Dict[str, Any]], int]:
    records: Dict[tuple[int, int], Dict[str, Any]] = {}
    eval_item_index = int(eval_item_offset)

    if sample_mode == "per_agent":
        batch_index = 0
        for local_scene_index, sample in enumerate(samples):
            meta = _scene_meta_to_dict(sample)
            for source_agent_index in _active_agent_indices(sample):
                records[(batch_index, 0)] = {
                    "eval_item_index": eval_item_index,
                    "selected_scene_index": int(global_scene_indices[local_scene_index]),
                    "batch_index": int(batch_index),
                    "agent_axis_index": 0,
                    "source_agent_index": int(source_agent_index),
                    "sample_mode": sample_mode,
                    "scene_meta": _coerce_jsonable(meta),
                    "dataset": meta.get("dataset"),
                    "subset": meta.get("subset"),
                    "split": meta.get("split"),
                    "sample_id": meta.get("sample_id"),
                    "seq_id": meta.get("seq_id"),
                    "frame_id": meta.get("frame_id"),
                    "source_file": meta.get("source_file"),
                }
                records[(batch_index, 0)].update(
                    compute_scene_motion_proxy_features(sample, source_agent_index=source_agent_index)
                )
                batch_index += 1
                eval_item_index += 1
        return records, eval_item_index

    if sample_mode != "per_scene":
        raise ValueError(f"Unsupported sample_mode for per-sample records: {sample_mode!r}")

    for local_scene_index, sample in enumerate(samples):
        meta = _scene_meta_to_dict(sample)
        for source_agent_index in _active_agent_indices(sample):
            records[(local_scene_index, source_agent_index)] = {
                "eval_item_index": eval_item_index,
                "selected_scene_index": int(global_scene_indices[local_scene_index]),
                "batch_index": int(local_scene_index),
                "agent_axis_index": int(source_agent_index),
                "source_agent_index": int(source_agent_index),
                "sample_mode": sample_mode,
                "scene_meta": _coerce_jsonable(meta),
                "dataset": meta.get("dataset"),
                "subset": meta.get("subset"),
                "split": meta.get("split"),
                "sample_id": meta.get("sample_id"),
                "seq_id": meta.get("seq_id"),
                "frame_id": meta.get("frame_id"),
                "source_file": meta.get("source_file"),
            }
            records[(local_scene_index, source_agent_index)].update(
                compute_scene_motion_proxy_features(sample, source_agent_index=source_agent_index)
            )
            eval_item_index += 1
    return records, eval_item_index


def _coerce_model_output_dict(output: Any) -> Mapping[str, Any]:
    if hasattr(output, "to_dict"):
        return output.to_dict()
    if isinstance(output, Mapping):
        return output
    raise TypeError(f"Unsupported model output type for per-sample records: {type(output)!r}")


def _update_per_sample_records_for_branch(
    records: Dict[tuple[int, int], Dict[str, Any]],
    *,
    branch_name: str,
    output: Any,
    batch: Mapping[str, Any],
    miss_threshold: float,
    collision_threshold: float,
) -> None:
    payload = _coerce_model_output_dict(output)
    prediction = payload.get(branch_name)
    if prediction is None:
        return

    gt_payload = infer_ground_truth_from_batch(batch)
    errors = displacement_errors(
        prediction,
        gt_payload["ground_truth"],
        agent_mask=gt_payload["agent_mask"],
    )
    ade = errors["ade_per_mode_agent"].detach().cpu()
    fde = errors["fde_per_mode_agent"].detach().cpu()
    valid = errors["valid_agents"].detach().cpu().bool()

    ade_min_values, ade_best_modes = ade.min(dim=1)
    fde_min_values, fde_best_modes = fde.min(dim=1)
    ade_avg_values = ade.mean(dim=1)
    fde_avg_values = fde.mean(dim=1)

    batch_size, num_agents = valid.shape
    for batch_index in range(int(batch_size)):
        for agent_axis_index in range(int(num_agents)):
            if not bool(valid[batch_index, agent_axis_index].item()):
                continue

            key = (batch_index, agent_axis_index)
            record = records.setdefault(
                key,
                {
                    "eval_item_index": None,
                    "batch_index": int(batch_index),
                    "agent_axis_index": int(agent_axis_index),
                    "source_agent_index": int(agent_axis_index),
                },
            )
            ade_min = float(ade_min_values[batch_index, agent_axis_index].item())
            fde_min = float(fde_min_values[batch_index, agent_axis_index].item())
            record.update(
                {
                    f"{branch_name}_ADE_min": ade_min,
                    f"{branch_name}_FDE_min": fde_min,
                    f"{branch_name}_ADE_avg": float(ade_avg_values[batch_index, agent_axis_index].item()),
                    f"{branch_name}_FDE_avg": float(fde_avg_values[batch_index, agent_axis_index].item()),
                    f"{branch_name}_Miss": bool(fde_min > float(miss_threshold)),
                    f"{branch_name}_best_ADE_mode": int(ade_best_modes[batch_index, agent_axis_index].item()),
                    f"{branch_name}_best_FDE_mode": int(fde_best_modes[batch_index, agent_axis_index].item()),
                }
            )

    update_records_with_prediction_proxy_features(
        records,
        branch_name=branch_name,
        prediction=prediction,
        batch=batch,
        collision_threshold=collision_threshold,
    )


def _ordered_per_sample_records(records: Mapping[tuple[int, int], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(record)
        for _key, record in sorted(
            records.items(),
            key=lambda item: (
                -1 if item[1].get("eval_item_index") is None else int(item[1]["eval_item_index"]),
                int(item[1].get("batch_index", item[0][0])),
                int(item[1].get("agent_axis_index", item[0][1])),
            ),
        )
    ]


def _build_predictor_config(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    cfg_path: Optional[str],
    checkpoint_path: Optional[str],
    sampling_steps: Optional[int] = None,
    solver: Optional[str] = None,
    lin_poly_p: Optional[int] = None,
    lin_poly_long_step: Optional[int] = None,
) -> MoFlowPredictorConfig:
    return MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        rotate=args.rotate,
        rotate_time_frame=args.rotate_time_frame,
        device=device,
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
        num_to_gen=args.num_to_gen,
        sampling_steps=sampling_steps,
        solver=solver,
        lin_poly_p=lin_poly_p,
        lin_poly_long_step=lin_poly_long_step,
    )


def _resolve_cfg_normalization_stats(
    predictors: Sequence[Any],
    *,
    requested_source: str,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    collected: List[tuple[str, Dict[str, float], str]] = []
    for predictor in predictors:
        if predictor is None:
            continue
        stats = predictor.get_cfg_normalization_stats()
        if not stats:
            continue
        cfg_path = getattr(getattr(predictor, "cfg", None), "cfg_path", "")
        collected.append((predictor.predictor_name, stats, str(cfg_path)))

    if not collected:
        raise ValueError(
            "normalization_source='predictor_cfg' requires training-time min/max stats in the predictor config. "
            "Pass --slow-cfg-path/--fast-cfg-path pointing to the run's *_updated.yml file."
        )

    reference_name, reference_stats, reference_cfg_path = collected[0]
    mismatch_names = [
        name
        for name, stats, _cfg_path in collected[1:]
        if any(abs(float(stats[key]) - float(reference_stats[key])) > 1e-6 for key in reference_stats)
    ]
    if mismatch_names:
        joined = ", ".join([reference_name, *mismatch_names])
        raise ValueError(
            f"Predictor configs disagree on normalization stats for normalization_source={requested_source!r}: {joined}"
        )

    return dict(reference_stats), {
        "source": requested_source,
        "source_cfg_path": reference_cfg_path,
        "source_predictor": reference_name,
        "num_predictors_with_cfg_stats": len(collected),
    }


def _resolve_normalization_stats(
    *,
    data_norm: str,
    normalization_source: str,
    predictors: Sequence[Any],
    samples: Sequence[Mapping[str, Any]],
    stats_owner: Any,
    data_root: Path,
    subset: str,
    protocol_settings: ProtocolSettings,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    if data_norm != "min_max":
        return {}, {
            "source": "disabled",
            "reason": f"data_norm={data_norm}",
        }
    if normalization_source == "selected_samples":
        return stats_owner.infer_normalization_stats(samples), {
            "source": normalization_source,
            "source_predictor": stats_owner.predictor_name,
            "num_source_scenes": len(samples),
        }
    if normalization_source == "train_split":
        train_dataset = ETHTrajectoryDataset(
            ETHAdapterConfig(
                data_root=data_root,
                subset=subset,
                split="train",
                min_agents=protocol_settings.min_agents,
                prefer_cache=protocol_settings.prefer_cache,
            )
        )
        train_samples = _select_samples(train_dataset, max_scenes=None)
        stats = stats_owner.infer_normalization_stats(train_samples)
        return stats, {
            "source": normalization_source,
            "source_predictor": stats_owner.predictor_name,
            "train_subset": subset,
            "train_num_source_scenes": len(train_samples),
            "train_loaded_from_cache": train_dataset.loaded_from_cache,
            "train_cache_compatible": train_dataset.cache_compatible,
        }
    if normalization_source == "predictor_cfg":
        return _resolve_cfg_normalization_stats(predictors, requested_source=normalization_source)
    raise ValueError(f"Unsupported normalization_source: {normalization_source!r}")


def _is_diagnostic_normalization_source(normalization_source: str) -> bool:
    return normalization_source == "selected_samples"


def _is_benchmark_comparable_run(
    *,
    protocol_settings: ProtocolSettings,
    sample_mode: str,
    agents: int,
) -> bool:
    return (
        protocol_settings.protocol == "official_align"
        and protocol_settings.normalization_source == "train_split"
        and sample_mode == "per_agent"
        and agents == 1
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_checkpoint_requirements(args)
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)

    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()

    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=args.split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, args.max_scenes)
    agents = _infer_agents(selected_samples, args.sample_mode, args.agents)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)
    missing_checkpoint_branches = [
        branch_name
        for branch_name, checkpoint_path in (
            ("slow", args.slow_checkpoint),
            ("fast", args.fast_checkpoint),
        )
        if checkpoint_path is None and args.baseline in (branch_name, "both")
    ]

    slow_predictor = None
    fast_predictor = None

    if args.baseline in ("slow", "both"):
        slow_predictor = MoFlowSlowPredictor(
            _build_predictor_config(
                args=args,
                agents=agents,
                device=device,
                cfg_path=args.slow_cfg_path,
                checkpoint_path=args.slow_checkpoint,
                sampling_steps=args.slow_sampling_steps,
                solver=args.slow_solver,
                lin_poly_p=args.slow_lin_poly_p,
                lin_poly_long_step=args.slow_lin_poly_long_step,
            )
        )

    if args.baseline in ("fast", "both"):
        fast_predictor = MoFlowFastPredictor(
            _build_predictor_config(
                args=args,
                agents=agents,
                device=device,
                cfg_path=args.fast_cfg_path,
                checkpoint_path=args.fast_checkpoint,
            )
        )

    stats_owner = slow_predictor or fast_predictor
    if stats_owner is None:
        raise RuntimeError("No predictor was instantiated")
    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=(slow_predictor, fast_predictor),
        samples=selected_samples,
        stats_owner=stats_owner,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    diagnostic_normalization = _is_diagnostic_normalization_source(protocol_settings.normalization_source)
    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )

    if diagnostic_normalization:
        print(
            "[run_eval] warning: normalization_source=selected_samples is a diagnostic mode. "
            "It recomputes min/max on the selected eval scenes and can distort benchmark-comparable metrics. "
            "Prefer train_split for checkpoint-backed MoFlow evaluation."
        )

    accumulators: Dict[str, BranchAccumulator] = {}
    if slow_predictor is not None:
        accumulators["slow_pred"] = BranchAccumulator("slow_pred", miss_threshold=args.miss_threshold)
    if fast_predictor is not None:
        accumulators["fast_pred"] = BranchAccumulator("fast_pred", miss_threshold=args.miss_threshold)

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    per_sample_records: List[Dict[str, Any]] = []
    next_eval_item_index = 0

    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        global_scene_indices = [int(scene_index) for scene_index, _sample in chunk_pairs]
        chunk = [sample for _scene_index, sample in chunk_pairs]
        chunk_per_sample_records: Dict[tuple[int, int], Dict[str, Any]] = {}
        if args.output_per_sample_json:
            chunk_per_sample_records, next_eval_item_index = _build_base_per_sample_records(
                samples=chunk,
                global_scene_indices=global_scene_indices,
                sample_mode=args.sample_mode,
                eval_item_offset=next_eval_item_index,
            )

        if slow_predictor is not None:
            slow_batch = slow_predictor.build_moflow_batch(
                chunk,
                normalization_stats=normalization_stats,
                as_torch=True,
            )
            slow_latencies, slow_output = _measure_predict_latency_ms(
                lambda: slow_predictor.predict(slow_batch, return_all_states=False),
                runs=args.latency_runs,
                device=device,
            )
            slow_summary = evaluate_model_output(
                slow_output,
                slow_batch,
                miss_threshold=args.miss_threshold,
                prediction_fields=("slow_pred",),
            )
            accumulators["slow_pred"].add_chunk(slow_summary.metrics, slow_latencies)
            if args.output_per_sample_json:
                _update_per_sample_records_for_branch(
                    chunk_per_sample_records,
                    branch_name="slow_pred",
                    output=slow_output,
                    batch=slow_batch,
                    miss_threshold=args.miss_threshold,
                    collision_threshold=args.collision_threshold,
                )

        if fast_predictor is not None:
            fast_batch = fast_predictor.build_moflow_batch(
                chunk,
                normalization_stats=normalization_stats,
                as_torch=True,
            )
            fast_latencies, fast_output = _measure_predict_latency_ms(
                lambda: fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen),
                runs=args.latency_runs,
                device=device,
            )
            fast_summary = evaluate_model_output(
                fast_output,
                fast_batch,
                miss_threshold=args.miss_threshold,
                prediction_fields=("fast_pred",),
            )
            accumulators["fast_pred"].add_chunk(fast_summary.metrics, fast_latencies)
            if args.output_per_sample_json:
                _update_per_sample_records_for_branch(
                    chunk_per_sample_records,
                    branch_name="fast_pred",
                    output=fast_output,
                    batch=fast_batch,
                    miss_threshold=args.miss_threshold,
                    collision_threshold=args.collision_threshold,
                )

        if args.output_per_sample_json:
            per_sample_records.extend(_ordered_per_sample_records(chunk_per_sample_records))

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(args.log_every, 1) == 0
        if should_log:
            print(
                f"[run_eval] processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    aggregated_metrics: Dict[str, float] = {}
    for accumulator in accumulators.values():
        aggregated_metrics.update(accumulator.finalize())

    result = {
        "meta": {
            "script": "trustmoe_traj.scripts.run_eval",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "formal_baseline": len(missing_checkpoint_branches) == 0,
            "device": device,
            "missing_checkpoint_branches": missing_checkpoint_branches,
            "protocol": protocol_settings.protocol,
            "normalization_source": protocol_settings.normalization_source,
            "diagnostic_normalization": diagnostic_normalization,
            "benchmark_comparable": benchmark_comparable,
            "metric_suite": {
                "comparable_metrics": list(protocol_settings.comparable_metrics),
                "auxiliary_metrics": list(protocol_settings.auxiliary_metrics),
            },
        },
        "args": _coerce_jsonable(vars(args)),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "predictor": {
            "subset": args.subset,
            "sample_mode": args.sample_mode,
            "agents": agents,
            "data_norm": args.data_norm,
            "rotate": bool(args.rotate),
            "rotate_time_frame": int(args.rotate_time_frame),
            "num_to_gen": int(args.num_to_gen),
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
        },
        "checkpoints": {
            "slow_checkpoint": args.slow_checkpoint,
            "fast_checkpoint": args.fast_checkpoint,
        },
        "protocol_settings": {
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
            "prefer_cache": bool(protocol_settings.prefer_cache),
            "normalization_source": protocol_settings.normalization_source,
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "available_predictions": list(accumulators.keys()),
        "metrics": aggregated_metrics,
    }

    if args.output_per_sample_json:
        result["per_sample"] = {
            "output_path": Path(args.output_per_sample_json).expanduser().as_posix(),
            "num_records": len(per_sample_records),
            "record_granularity": "eval_item_agent",
        }

    print("[run_eval] completed")
    print(
        f"subset={args.subset} split={args.split} baseline={args.baseline} "
        f"sample_mode={args.sample_mode} protocol={protocol_settings.protocol}"
    )
    print(f"data_root={data_root.as_posix()}")
    print(
        f"loaded_from_cache={dataset.loaded_from_cache} prefer_cache={protocol_settings.prefer_cache} "
        f"cache_compatible={dataset.cache_compatible}"
    )
    if dataset.cache_mismatch_fields:
        print(f"cache_mismatch_fields={','.join(dataset.cache_mismatch_fields)}")
    print(
        f"selected_scenes={len(selected_samples)} selected_eval_items={selected_eval_items} "
        f"batch_scenes={args.batch_scenes} min_agents={protocol_settings.min_agents}"
    )
    print(f"normalization_source={protocol_settings.normalization_source}")
    print(f"diagnostic_normalization={diagnostic_normalization}")
    print(f"benchmark_comparable={benchmark_comparable}")
    print(f"device={device} agents={agents} formal_baseline={len(missing_checkpoint_branches) == 0}")
    if args.slow_checkpoint:
        print(f"slow_checkpoint={Path(args.slow_checkpoint).expanduser().as_posix()}")
    if args.fast_checkpoint:
        print(f"fast_checkpoint={Path(args.fast_checkpoint).expanduser().as_posix()}")
    for key in sorted(aggregated_metrics):
        print(f"{key}={aggregated_metrics[key]}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_coerce_jsonable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"output_json={output_path.as_posix()}")

    if args.output_per_sample_json:
        per_sample_path = Path(args.output_per_sample_json).expanduser().resolve()
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_payload = {
            "meta": {
                **result["meta"],
                "subset": args.subset,
                "split": args.split,
                "baseline": args.baseline,
                "sample_mode": args.sample_mode,
                "agents": agents,
                "miss_threshold": float(args.miss_threshold),
                "collision_threshold": float(args.collision_threshold),
                "record_granularity": "eval_item_agent",
                "proxy_feature_version": "scene_motion_prediction_proxy_v1",
            },
            "dataset": result["dataset"],
            "predictor": result["predictor"],
            "protocol_settings": result["protocol_settings"],
            "checkpoints": result["checkpoints"],
            "records": per_sample_records,
        }
        per_sample_path.write_text(
            json.dumps(_coerce_jsonable(per_sample_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"output_per_sample_json={per_sample_path.as_posix()}")
        print(f"per_sample_records={len(per_sample_records)}")


if __name__ == "__main__":
    main()
