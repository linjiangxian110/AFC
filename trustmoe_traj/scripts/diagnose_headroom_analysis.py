"""Headroom analysis for multimodal prediction-set quality and AFC coverage.

This diagnostic does not train a model.  It asks whether useful set-level
improvements are already present in a larger slow-sampling pool or in a residual
candidate pool:

* slow20: the original fair K=20 teacher output;
* slow{K}_full: the raw larger teacher pool;
* slow{K}_gt_oracle20: best-of-pool selected with current GT;
* slow{K}_afc_greedy20: train-set analogical future coverage greedy selection;
* slow{K}_endpoint_fps20: endpoint farthest-point selection;
* slow{K}_random20 or slow{K}_random20_mean{T}: random best-of-pool control;
* cv_linear20 / random_spread*: weak and fake-diversity controls;
* residual_*: the same headroom checks for an optional residual candidate pool.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.analogical_future_coverage import (
    AFC_FEATURE_VARIANTS,
    AnalogicalFutureBank,
    attach_afc_metadata_to_batch,
    build_eth_analogical_future_bank,
    split_float_list,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _add_branch,
    _all_indices,
    _base_for_flat,
    _flatten_refined,
    _gather_candidates,
    _oracle_indices,
    _predictor_cfg,
    _random_global_indices,
    _set_seed,
    _structured_fps_indices,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import _checkpoint_variant, _local_temporal_energy
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    BranchAccumulator,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_benchmark_comparable_run,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _measure_predict_latency_ms,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AFC/QD headroom analysis for slow and residual candidate pools.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--slow-pool-ks", type=str, default="20,50,100")
    parser.add_argument("--oracle-select-metric", type=str, default="ade_fde", choices=["fde", "ade_fde"])
    parser.add_argument("--afc-selection-tau", type=float, default=1.0)
    parser.add_argument("--disable-random-pool-selection", action="store_true")
    parser.add_argument("--random-pool-trials", type=int, default=1)
    parser.add_argument("--random-pool-emit-trials", action="store_true")
    parser.add_argument("--disable-cv-linear", action="store_true")
    parser.add_argument("--disable-random-spread", action="store_true")
    parser.add_argument("--random-spread-source", type=str, default="slow_radial", choices=["slow_radial", "cv"])
    parser.add_argument("--random-spread-endpoint-scale", type=float, default=2.0)
    parser.add_argument("--random-spread-endpoint-scales", type=str, default="")
    parser.add_argument("--random-spread-noise-scale", type=float, default=0.05)

    parser.add_argument("--refiner-checkpoint", type=str, default=None)
    parser.add_argument("--residual-slots", type=int, default=8)

    parser.add_argument("--afc-train-split", type=str, default="train")
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.5,1.0")
    parser.add_argument("--afc-feature-variant", type=str, default="full_past_social", choices=AFC_FEATURE_VARIANTS)
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--afc-use-source-metadata", action="store_true")
    parser.add_argument("--afc-source-id-field", type=str, default="source_file")
    parser.add_argument("--afc-filter-same-source", action="store_true")
    parser.add_argument("--afc-temporal-gap-frames", type=int, default=0)
    parser.add_argument("--afc-randomize-bank-seed", type=int, default=None)

    parser.add_argument("--output-json", type=str, required=True)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item]


def _metric(metrics: Mapping[str, float], branch: str, name: str) -> Optional[float]:
    value = metrics.get(f"{branch}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _predict_slow_repeated_pool(
    slow_predictor: MoFlowSlowPredictor,
    batch: Mapping[str, Any],
    *,
    pool_k: int,
    first_prediction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a larger slow candidate pool by repeated native-K sampling.

    MoFlow's denoising head count is model-architecture dependent.  Increasing
    ``cfg.denoising_head_preds`` beyond the checkpoint's native value can index
    outside learned query/head tensors on CUDA.  For headroom analysis we only
    need a larger candidate pool, so we repeatedly sample the native K=20
    predictor and concatenate the results.
    """

    pool = int(pool_k)
    if pool <= 0:
        raise ValueError("pool_k must be positive")
    chunks: List[torch.Tensor] = []
    if first_prediction is not None:
        chunks.append(first_prediction)
    while sum(int(item.shape[1]) for item in chunks) < pool:
        output = slow_predictor.predict(batch, return_all_states=False)
        chunks.append(output.slow_pred)
    return torch.cat(chunks, dim=1)[:, :pool]


def _constant_velocity_prediction(batch: Mapping[str, Any], *, keep_k: int, device: torch.device) -> torch.Tensor:
    """Repeat a deterministic constant-velocity extrapolation to K modes."""

    past = batch["past_traj_original_scale"].to(device=device, dtype=torch.float32)
    future = batch["fut_traj_original_scale"].to(device=device, dtype=torch.float32)
    if past.ndim != 4 or int(past.shape[-1]) < 4:
        raise ValueError(f"Expected past_traj_original_scale [B,A,P,>=4], got {tuple(past.shape)}")
    if future.ndim != 4:
        raise ValueError(f"Expected fut_traj_original_scale [B,A,T,2], got {tuple(future.shape)}")
    past_rel = past[..., 2:4]
    if int(past_rel.shape[-2]) >= 2:
        velocity = past_rel[..., -1, :] - past_rel[..., -2, :]
    else:
        velocity = torch.zeros_like(past_rel[..., -1, :])
    steps = torch.arange(1, int(future.shape[-2]) + 1, device=device, dtype=past.dtype)
    prediction = velocity[:, :, None, :] * steps[None, None, :, None]
    return prediction[:, None, ...].expand(
        int(prediction.shape[0]),
        int(keep_k),
        int(prediction.shape[1]),
        int(prediction.shape[2]),
        int(prediction.shape[3]),
    ).contiguous()


def _random_spread_prediction(
    batch: Mapping[str, Any],
    *,
    keep_k: int,
    device: torch.device,
    endpoint_scale: float,
    noise_scale: float,
    source: str,
    base_prediction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Synthetic high-spread control for fake-diversity diagnostics.

    This branch intentionally creates geometric diversity without using GT or
    analogical futures. It is a negative sanity check: large spread should not
    automatically look good under AFC if the modes lack empirical support.
    """

    if source == "slow_radial":
        if base_prediction is None:
            raise ValueError("base_prediction is required for random_spread_source='slow_radial'")
        base = base_prediction.to(device=device, dtype=torch.float32)
        if base.ndim != 5:
            raise ValueError(f"Expected base_prediction [B,K,A,T,2], got {tuple(base.shape)}")
        if int(base.shape[1]) != int(keep_k):
            raise ValueError(f"Expected base modes == keep_k, got {base.shape[1]} vs {keep_k}")
        endpoint = base[..., -1, :]
        center = endpoint.mean(dim=1, keepdim=True)
        radial = endpoint - center
        norm = torch.linalg.norm(radial, dim=-1, keepdim=True)
        fallback = torch.randn_like(radial)
        fallback = fallback / torch.linalg.norm(fallback, dim=-1, keepdim=True).clamp_min(1e-6)
        global_radius = torch.linalg.norm(endpoint - center, dim=-1, keepdim=True).mean(dim=1, keepdim=True).clamp_min(0.5)
        direction = torch.where(norm > 1e-6, radial / norm.clamp_min(1e-6), fallback)
        target_endpoint = center + float(endpoint_scale) * torch.where(norm > 1e-6, radial, fallback * global_radius)
        endpoint_delta = target_endpoint - endpoint
        future_steps = int(base.shape[-2])
        ramp = torch.linspace(0.0, 1.0, future_steps, device=device, dtype=base.dtype)[None, None, None, :, None]
        prediction = base + ramp * endpoint_delta[:, :, :, None, :]
        if float(noise_scale) > 0:
            prediction = prediction + torch.randn_like(prediction) * float(noise_scale) * ramp
        return prediction.contiguous()

    if source != "cv":
        raise ValueError(f"Unsupported random_spread_source: {source!r}")

    base_cv = _constant_velocity_prediction(batch, keep_k=1, device=device)[:, 0]
    batch_size, num_agents, future_steps, dim = [int(item) for item in base_cv.shape]
    directions = torch.randn(batch_size, int(keep_k), num_agents, dim, device=device, dtype=base_cv.dtype)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-6)
    magnitudes = torch.rand(batch_size, int(keep_k), num_agents, 1, device=device, dtype=base_cv.dtype)
    endpoint_offsets = directions * (float(endpoint_scale) * magnitudes)
    ramp = torch.linspace(0.0, 1.0, future_steps, device=device, dtype=base_cv.dtype)[None, None, None, :, None]
    prediction = base_cv[:, None, ...] + ramp * endpoint_offsets[:, :, :, None, :]
    if float(noise_scale) > 0:
        prediction = prediction + torch.randn_like(prediction) * float(noise_scale) * ramp
    return prediction.contiguous()


def _tag_float(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _random_spread_branch_name(scale: float, *, num_scales: int) -> str:
    if int(num_scales) == 1:
        return "random_spread20_pred"
    return f"random_spread_s{_tag_float(float(scale))}_pred"


def _random_pool_mean_branch_name(pool_k: int, trials: int) -> str:
    if int(trials) <= 1:
        return f"slow{int(pool_k)}_random20_pred"
    return f"slow{int(pool_k)}_random20_mean{int(trials)}_pred"


def _afc_greedy_indices(
    candidates: torch.Tensor,
    batch: Mapping[str, Any],
    afc_bank: AnalogicalFutureBank,
    *,
    keep_k: int,
    tau: float,
) -> torch.Tensor:
    if candidates.ndim != 5:
        raise ValueError(f"Expected candidates [B,N,A,T,2], got {tuple(candidates.shape)}")
    batch_size, num_candidates, num_agents = [int(item) for item in candidates.shape[:3]]
    keep = int(keep_k)
    if keep <= 0:
        raise ValueError("keep_k must be positive")
    if keep > num_candidates:
        raise ValueError(f"keep_k={keep} exceeds candidate count={num_candidates}")

    _features, valid_cpu, top_indices = afc_bank._query(batch)
    result = torch.zeros((batch_size, keep, num_agents), dtype=torch.long, device=candidates.device)
    if int(top_indices.shape[0]) <= 0:
        return result
    proxies = afc_bank.futures[top_indices].to(device=candidates.device, dtype=candidates.dtype)
    candidate_by_agent = candidates.permute(0, 2, 1, 3, 4)
    valid_positions = valid_cpu.nonzero(as_tuple=False)
    tau_value = max(float(tau), 1e-6)
    for query_index, position in enumerate(valid_positions):
        batch_index = int(position[0].item())
        agent_index = int(position[1].item())
        candidate = candidate_by_agent[batch_index, agent_index]
        proxy = proxies[query_index]
        ade = torch.linalg.norm(candidate[:, None, :, :] - proxy[None, :, :, :], dim=-1).mean(dim=-1)
        support = torch.exp(-ade / tau_value)
        covered = torch.zeros((int(support.shape[1]),), dtype=support.dtype, device=support.device)
        available = torch.ones((num_candidates,), dtype=torch.bool, device=support.device)
        selected: List[int] = []
        for _ in range(keep):
            gain = torch.maximum(covered[None, :], support) - covered[None, :]
            score = gain.sum(dim=1)
            score = score.masked_fill(~available, -float("inf"))
            index = int(score.argmax().detach().cpu().item())
            selected.append(index)
            available[index] = False
            covered = torch.maximum(covered, support[index])
        result[batch_index, :, agent_index] = torch.as_tensor(selected, dtype=torch.long, device=candidates.device)
    return result


def _add_headroom_branch(
    accumulators: Mapping[str, BranchAccumulator],
    aux_accumulators: Mapping[str, AuxAccumulator],
    *,
    field_name: str,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    miss_threshold: float,
    latencies_ms: Sequence[float],
    afc_bank: AnalogicalFutureBank,
    spread_base: Optional[torch.Tensor] = None,
    base_for_delta: Optional[torch.Tensor] = None,
    selected_flat_indices: Optional[torch.Tensor] = None,
    num_base_modes: Optional[int] = None,
) -> None:
    _add_branch(
        accumulators,
        aux_accumulators,
        field_name=field_name,
        prediction=prediction,
        batch=batch,
        miss_threshold=float(miss_threshold),
        latencies_ms=latencies_ms,
        base_for_delta=base_for_delta,
        spread_base=spread_base,
        selected_flat_indices=selected_flat_indices,
        num_base_modes=num_base_modes,
    )
    valid_count = int(batch["agent_mask"].bool().sum().item())
    if valid_count > 0:
        energy_metrics = _energy_score_metrics(prediction, batch)
        for key, value in energy_metrics.items():
            aux_accumulators[field_name].add_metric(key, value, weight=valid_count)
        for key, value in afc_bank.metrics_for_prediction(prediction, batch).items():
            aux_accumulators[field_name].add_metric(key, value, weight=valid_count)


def _energy_score_metrics(prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    """Empirical Energy Score for a finite K-sample prediction set.

    ES(P, y) = E||X-y|| - 0.5 E||X-X'||, using full future trajectories as
    flattened vectors. The biased finite-sample pair term includes diagonal
    pairs, matching the plug-in ensemble estimate used for diagnostic
    comparison rather than model training.
    """

    if prediction.ndim != 5:
        raise ValueError(f"prediction must have shape [B,K,A,T,2], got {tuple(prediction.shape)}")
    gt = batch["fut_traj_original_scale"].to(device=prediction.device, dtype=prediction.dtype)
    valid = batch["agent_mask"].to(device=prediction.device).bool()
    if gt.ndim != 4:
        raise ValueError(f"ground truth must have shape [B,A,T,2], got {tuple(gt.shape)}")
    if tuple(prediction.shape[0:1] + prediction.shape[2:]) != tuple(gt.shape):
        raise ValueError(f"prediction/gt shape mismatch: pred={tuple(prediction.shape)} gt={tuple(gt.shape)}")

    pred = prediction.permute(0, 2, 1, 3, 4)  # [B,A,K,T,2]
    batch_size, num_agents, num_modes = [int(item) for item in pred.shape[:3]]
    pred_flat = pred.reshape(batch_size, num_agents, num_modes, -1)
    gt_flat = gt.reshape(batch_size, num_agents, -1)

    term_gt = torch.linalg.norm(pred_flat - gt_flat[:, :, None, :], dim=-1).mean(dim=-1)
    pair = torch.linalg.norm(pred_flat[:, :, :, None, :] - pred_flat[:, :, None, :, :], dim=-1).mean(dim=(-1, -2))
    energy = term_gt - 0.5 * pair

    if int(valid.sum().item()) <= 0:
        return {}
    return {
        "energy_score": float(energy[valid].mean().detach().cpu()),
        "energy_gt_term": float(term_gt[valid].mean().detach().cpu()),
        "energy_pair_term": float(pair[valid].mean().detach().cpu()),
    }


def _print_summary(metrics: Mapping[str, float], branches: Sequence[str]) -> None:
    slow = "slow20_pred"
    print("\n===== Headroom summary vs slow20 =====")
    header = (
        "branch | dADE_avg | dFDE_avg | dADE_min | dFDE_min | "
        "dAFCrecall@0.5 | dAFCmode@0.5 | dAFCwMode@0.5 | dUnsupported@0.5 | dChamfer | "
        "endpoint_ratio | trajectory_ratio | unique_base_mode_ratio"
    )
    print(header)
    print("-" * len(header))
    for branch in branches:
        items = [branch]
        for name in ("ADE_avg", "FDE_avg", "ADE_min", "FDE_min"):
            value = _metric(metrics, branch, name)
            base = _metric(metrics, slow, name)
            items.append(_fmt(None if value is None or base is None else value - base, signed=True))
        for name in ("afc_recall_eps05", "afc_mode_coverage_eps05", "afc_weighted_mode_recall_eps05", "afc_unsupported_ratio_eps05", "afc_chamfer"):
            value = _metric(metrics, branch, name)
            base = _metric(metrics, slow, name)
            items.append(_fmt(None if value is None or base is None else value - base, signed=name != "afc_chamfer"))
        for name in ("endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
            items.append(_fmt(_metric(metrics, branch, name)))
        print(" | ".join(items))


def main() -> None:
    args = build_parser().parse_args()
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.random_pool_trials) <= 0:
        raise SystemExit("--random-pool-trials must be positive")
    pool_ks = sorted(set(_split_ints(str(args.slow_pool_ks))))
    if int(args.keep_k) not in pool_ks:
        pool_ks.insert(0, int(args.keep_k))
    if any(item < int(args.keep_k) for item in pool_ks):
        raise SystemExit("--slow-pool-ks entries must be >= keep_k")
    if int(args.residual_slots) <= 1 and args.refiner_checkpoint:
        raise SystemExit("--residual-slots must be > 1 when --refiner-checkpoint is used")
    if int(args.afc_temporal_gap_frames) < 0:
        raise SystemExit("--afc-temporal-gap-frames must be non-negative")
    random_spread_scales = (
        split_float_list(str(args.random_spread_endpoint_scales))
        if str(args.random_spread_endpoint_scales).strip()
        else [float(args.random_spread_endpoint_scale)]
    )
    if any(float(item) <= 0 for item in random_spread_scales):
        raise SystemExit("--random-spread endpoint scales must be positive")

    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    _set_seed(args.seed)

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

    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )

    refiner = None
    refiner_variant = None
    if args.refiner_checkpoint:
        refiner_variant = _checkpoint_variant(args.refiner_checkpoint)
        refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
        refiner.eval()
        if not bool(getattr(refiner.config, "use_set_generator", False)):
            raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")
        max_slots = int(getattr(refiner.config, "max_residual_slots", 1))
        if int(args.residual_slots) > max_slots:
            raise SystemExit(f"--residual-slots {args.residual_slots} exceeds checkpoint max_residual_slots={max_slots}")

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[slow_predictor],
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)
    afc_temporal_gap = int(args.afc_temporal_gap_frames) if int(args.afc_temporal_gap_frames) > 0 else None
    afc_needs_metadata = bool(args.afc_use_source_metadata or args.afc_filter_same_source or afc_temporal_gap is not None)
    afc_bank = build_eth_analogical_future_bank(
        data_root=data_root,
        subset=args.subset,
        train_split=str(args.afc_train_split),
        sample_mode=str(args.sample_mode),
        data_norm=str(args.data_norm),
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        normalization_stats=normalization_stats,
        min_agents=int(protocol_settings.min_agents),
        prefer_cache=bool(protocol_settings.prefer_cache),
        max_train_scenes=args.afc_max_train_scenes,
        batch_scenes=int(args.afc_batch_scenes),
        top_m=int(args.afc_top_m),
        eps_values=split_float_list(str(args.afc_eps)),
        feature_variant=str(args.afc_feature_variant),
        include_source_metadata=afc_needs_metadata,
        source_id_field=str(args.afc_source_id_field),
        filter_same_source=bool(args.afc_filter_same_source),
        temporal_gap_frames=afc_temporal_gap,
        randomize_futures_seed=args.afc_randomize_bank_seed,
    )

    branches: List[str] = ["slow20_pred"]
    if not bool(args.disable_cv_linear):
        branches.append("cv_linear20_pred")
    if not bool(args.disable_random_spread):
        for scale in random_spread_scales:
            branches.append(_random_spread_branch_name(float(scale), num_scales=len(random_spread_scales)))
    for pool_k in pool_ks:
        if int(pool_k) == int(args.keep_k):
            continue
        branches.extend(
            [
                f"slow{pool_k}_full_pred",
                f"slow{pool_k}_gt_oracle20_pred",
                f"slow{pool_k}_afc_greedy20_pred",
                f"slow{pool_k}_endpoint_fps20_pred",
            ]
        )
        if not bool(args.disable_random_pool_selection):
            branches.append(_random_pool_mean_branch_name(int(pool_k), int(args.random_pool_trials)))
            if bool(args.random_pool_emit_trials) and int(args.random_pool_trials) > 1:
                for trial_index in range(int(args.random_pool_trials)):
                    branches.append(f"slow{int(pool_k)}_random20_trial{trial_index}_pred")
    if refiner is not None:
        residual_pool_size = int(args.residual_slots) * int(args.keep_k)
        branches.extend(
            [
                f"residual_full{residual_pool_size}_pred",
                "residual_gt_oracle20_pred",
                "residual_afc_greedy20_pred",
                "residual_endpoint_fps20_pred",
            ]
        )

    accumulators = {branch: BranchAccumulator(branch, args.miss_threshold) for branch in branches}
    aux_accumulators = {branch: AuxAccumulator() for branch in branches}

    print(
        "[diagnose_headroom_analysis] "
        f"subset={args.subset} split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} keep_k={args.keep_k} slow_pool_ks={pool_ks} afc_bank={afc_bank.bank_size} "
        f"afc_feature={args.afc_feature_variant} afc_filter_same_source={args.afc_filter_same_source} "
        f"afc_temporal_gap={afc_temporal_gap or 0} afc_randomize={args.afc_randomize_bank_seed or 'none'} "
        f"refiner={args.refiner_checkpoint or 'none'}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_headroom_analysis] warning: selected_samples normalization is diagnostic only")

    chunks = list(_iter_chunks(list(enumerate(selected_samples)), int(args.batch_scenes)))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        if afc_needs_metadata:
            attach_afc_metadata_to_batch(
                batch,
                samples=chunk,
                sample_mode=str(args.sample_mode),
                source_id_field=str(args.afc_source_id_field),
            )
        base_latencies, slow20_output = _measure_predict_latency_ms(
            lambda: slow_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
        slow20 = slow20_output.slow_pred
        if int(slow20.shape[1]) != int(args.keep_k):
            raise SystemExit(f"Expected slow20 modes == keep_k, got {slow20.shape[1]} vs {args.keep_k}")
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        batch_size, _base_modes, num_agents = [int(item) for item in slow20.shape[:3]]
        base_indices = torch.arange(int(args.keep_k), device=slow20.device, dtype=torch.long)[None, :, None].expand(
            batch_size,
            int(args.keep_k),
            num_agents,
        )
        _add_headroom_branch(
            accumulators,
            aux_accumulators,
            field_name="slow20_pred",
            prediction=slow20,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=base_latencies,
            afc_bank=afc_bank,
            spread_base=slow20,
            selected_flat_indices=base_indices,
            num_base_modes=int(args.keep_k),
        )
        if not bool(args.disable_cv_linear):
            cv_linear = _constant_velocity_prediction(batch, keep_k=int(args.keep_k), device=device)
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name="cv_linear20_pred",
                prediction=cv_linear,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=[0.0],
                afc_bank=afc_bank,
                spread_base=slow20,
            )
        if not bool(args.disable_random_spread):
            for scale in random_spread_scales:
                random_spread = _random_spread_prediction(
                    batch,
                    keep_k=int(args.keep_k),
                    device=device,
                    endpoint_scale=float(scale),
                    noise_scale=float(args.random_spread_noise_scale),
                    source=str(args.random_spread_source),
                    base_prediction=slow20,
                )
                _add_headroom_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=_random_spread_branch_name(float(scale), num_scales=len(random_spread_scales)),
                    prediction=random_spread,
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=[0.0],
                    afc_bank=afc_bank,
                    spread_base=slow20,
                )

        for pool_k in pool_ks:
            if int(pool_k) == int(args.keep_k):
                continue
            pool_latencies, pool_output = _measure_predict_latency_ms(
                lambda pool_k=pool_k: _predict_slow_repeated_pool(
                    slow_predictor,
                    batch,
                    pool_k=int(pool_k),
                    first_prediction=slow20,
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            pool_pred = pool_output
            if int(pool_pred.shape[1]) != int(pool_k):
                raise SystemExit(f"Expected slow pool modes == {pool_k}, got {pool_pred.shape[1]}")
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_full_pred",
                prediction=pool_pred,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            gt_indices = _oracle_indices(pool_pred, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_gt_oracle20_pred",
                prediction=_gather_candidates(pool_pred, gt_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            afc_indices = _afc_greedy_indices(
                pool_pred,
                batch,
                afc_bank,
                keep_k=int(args.keep_k),
                tau=float(args.afc_selection_tau),
            )
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_afc_greedy20_pred",
                prediction=_gather_candidates(pool_pred, afc_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            fps_indices = _structured_fps_indices(pool_pred[..., -1, :], keep_k=int(args.keep_k))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_endpoint_fps20_pred",
                prediction=_gather_candidates(pool_pred, fps_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            if not bool(args.disable_random_pool_selection):
                mean_branch = _random_pool_mean_branch_name(int(pool_k), int(args.random_pool_trials))
                for trial_index in range(int(args.random_pool_trials)):
                    random_indices = _random_global_indices(
                        batch_size,
                        int(pool_k),
                        num_agents,
                        keep_k=int(args.keep_k),
                        device=pool_pred.device,
                    )
                    random_prediction = _gather_candidates(pool_pred, random_indices)
                    _add_headroom_branch(
                        accumulators,
                        aux_accumulators,
                        field_name=mean_branch,
                        prediction=random_prediction,
                        batch=batch,
                        miss_threshold=float(args.miss_threshold),
                        latencies_ms=pool_latencies,
                        afc_bank=afc_bank,
                        spread_base=slow20,
                    )
                    if bool(args.random_pool_emit_trials) and int(args.random_pool_trials) > 1:
                        _add_headroom_branch(
                            accumulators,
                            aux_accumulators,
                            field_name=f"slow{int(pool_k)}_random20_trial{trial_index}_pred",
                            prediction=random_prediction,
                            batch=batch,
                            miss_threshold=float(args.miss_threshold),
                            latencies_ms=pool_latencies,
                            afc_bank=afc_bank,
                            spread_base=slow20,
                        )

        if refiner is not None:
            if args.sample_mode == "per_agent":
                temporal_energy = build_per_agent_scene_temporal_interaction_features(
                    chunk,
                    slow20,
                    rotate=bool(args.rotate),
                    rotate_time_frame=int(args.rotate_time_frame),
                    collision_sigma=0.5,
                    collision_radius=0.2,
                    no_neighbor_distance=10.0,
                )
            else:
                temporal_energy = _local_temporal_energy(batch, slow20)
            refiner_latencies, refiner_outputs = _measure_predict_latency_ms(
                lambda: refiner.refine(
                    slow20,
                    past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                    temporal_energy_features=temporal_energy.to(device=device),
                    num_samples=int(args.residual_slots),
                    z_mode="slots",
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            refined = refiner_outputs["refined"]
            flat = _flatten_refined(refined)
            base_flat = _base_for_flat(refined, slow20)
            residual_pool_size = int(flat.shape[1])
            full_indices = _all_indices(batch_size, residual_pool_size, num_agents, device=flat.device)
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"residual_full{residual_pool_size}_pred",
                prediction=flat,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
                base_for_delta=base_flat,
                selected_flat_indices=full_indices,
                num_base_modes=int(args.keep_k),
            )
            gt_indices = _oracle_indices(flat, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name="residual_gt_oracle20_pred",
                prediction=_gather_candidates(flat, gt_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
                base_for_delta=_gather_candidates(base_flat, gt_indices),
                selected_flat_indices=gt_indices,
                num_base_modes=int(args.keep_k),
            )
            afc_indices = _afc_greedy_indices(
                flat,
                batch,
                afc_bank,
                keep_k=int(args.keep_k),
                tau=float(args.afc_selection_tau),
            )
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name="residual_afc_greedy20_pred",
                prediction=_gather_candidates(flat, afc_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
                base_for_delta=_gather_candidates(base_flat, afc_indices),
                selected_flat_indices=afc_indices,
                num_base_modes=int(args.keep_k),
            )
            fps_indices = _structured_fps_indices(flat[..., -1, :], keep_k=int(args.keep_k))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name="residual_endpoint_fps20_pred",
                prediction=_gather_candidates(flat, fps_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
                base_for_delta=_gather_candidates(base_flat, fps_indices),
                selected_flat_indices=fps_indices,
                num_base_modes=int(args.keep_k),
            )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_headroom_analysis] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * int(args.batch_scenes), len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for branch, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[branch].finalize(branch))

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_headroom_analysis",
            "variant": "headroom_afc_qd_v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "subset": str(args.subset),
            "split": str(args.split),
            "keep_k": int(args.keep_k),
            "slow_pool_ks": [int(item) for item in pool_ks],
            "oracle_select_metric": str(args.oracle_select_metric),
            "afc_selection_tau": float(args.afc_selection_tau),
            "afc_feature_variant": str(args.afc_feature_variant),
            "afc_bank_size": int(afc_bank.bank_size),
            "afc_bank_feature_dim": int(afc_bank.feature_dim),
            "afc_use_source_metadata": bool(afc_needs_metadata),
            "afc_source_id_field": str(args.afc_source_id_field),
            "afc_filter_same_source": bool(args.afc_filter_same_source),
            "afc_temporal_gap_frames": int(args.afc_temporal_gap_frames),
            "afc_randomize_bank_seed": args.afc_randomize_bank_seed,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
            "refiner_variant": refiner_variant,
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(args.slow_checkpoint).expanduser().resolve().as_posix(),
        "refiner_checkpoint": None
        if args.refiner_checkpoint is None
        else Path(args.refiner_checkpoint).expanduser().resolve().as_posix(),
        "metrics": _coerce_jsonable(metrics),
    }
    _print_summary(metrics, branches)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
