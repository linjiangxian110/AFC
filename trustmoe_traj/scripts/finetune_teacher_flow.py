"""V23-A: fine-tune the MoFlow slow teacher flow process.

This is intentionally different from V22 teacher-flow adapters: V22 freezes the
slow teacher and trains a small external residual module, while V23 updates a
small scope of the slow teacher itself from the existing checkpoint.

The default objective keeps the original MoFlow flow-matching loss and adds two
conservative terms:

- temporal interaction-energy weighted GT loss on intermediate pred_data;
- process keep loss against a frozen copy of the original slow teacher.

The script trains from raw ETH scenes so scene-aware temporal energy can be
computed from the current per-agent predictions.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import displacement_errors
from trustmoe_traj.models import MoFlowPredictorConfig, MoFlowSlowPredictor
from trustmoe_traj.scripts.interaction_energy_features import (
    build_per_agent_scene_temporal_interaction_features,
)
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "teacher_finetune_models"
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V23-A fine-tune for the MoFlow slow teacher.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-scenes", type=int, default=4)
    parser.add_argument("--val-batch-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="teacher_finetune")

    parser.add_argument(
        "--trainable-scope",
        type=str,
        default="decoder",
        choices=["head", "decoder", "all"],
        help="Which slow-teacher parameters to update. `decoder` freezes the context encoder.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lambda-base-flow", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-energy-gt", type=float, default=0.2)
    parser.add_argument("--lambda-process-keep", type=float, default=0.05)
    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--temporal-energy-distance-scale", type=float, default=0.5)
    parser.add_argument("--temporal-energy-gt-risk-floor", type=float, default=0.05)
    parser.add_argument("--temporal-energy-gt-risk-power", type=float, default=1.0)

    parser.add_argument("--selection-metric", type=str, default="safety", choices=["fde_min", "safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-keep-weight", type=float, default=1.0)
    parser.add_argument("--selection-worse-rate-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=1)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    return _coerce_jsonable(value)


def _split_train_val(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    if not samples:
        raise ValueError("Cannot split an empty sample list")
    indices = list(range(len(samples)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if len(indices) <= 1 or float(val_fraction) <= 0.0:
        selected = [samples[index] for index in indices]
        return selected, selected
    val_count = max(1, int(round(len(indices) * float(val_fraction))))
    val_count = min(val_count, len(indices) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return [samples[index] for index in train_indices], [samples[index] for index in val_indices]


def _iter_epoch_batches(
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_scenes: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[List[Mapping[str, Any]]]:
    if int(batch_scenes) <= 0:
        raise ValueError(f"batch_scenes must be positive, got {batch_scenes}")
    indices = list(range(len(samples)))
    if shuffle:
        rng = random.Random(int(seed) + int(epoch) * 1009)
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_scenes)):
        yield [samples[index] for index in indices[start : start + int(batch_scenes)]]


def _predictor_cfg(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    checkpoint_path: str,
) -> MoFlowPredictorConfig:
    return MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        device=device,
        cfg_path=args.slow_cfg_path,
        checkpoint_path=checkpoint_path,
        num_to_gen=int(args.num_to_gen),
    )


def _set_trainable_scope(model: torch.nn.Module, scope: str) -> List[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    if scope == "head":
        prefixes = ("reg_head", "cls_head")
    elif scope == "decoder":
        prefixes = (
            "motion_query_embedding",
            "agent_order_embedding",
            "post_pe_cat_mlp",
            "time_mlp",
            "noisy_y_mlp",
            "noisy_y_attn_k",
            "noisy_y_attn_a",
            "init_emb_fusion_mlp",
            "motion_decoder",
            "reg_head",
            "cls_head",
        )
    elif scope == "all":
        prefixes = ("",)
    else:
        raise ValueError(f"Unsupported trainable scope: {scope!r}")

    names: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            names.append(name)
    if not names:
        raise ValueError(f"No trainable parameters selected for scope={scope!r}")
    return names


def _move_batch(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _prepare_moflow_batch(
    predictor: MoFlowSlowPredictor,
    samples: Sequence[Mapping[str, Any]],
    *,
    normalization_stats: Mapping[str, float],
    device: str,
) -> Dict[str, Any]:
    batch = predictor.build_moflow_batch(samples, normalization_stats=normalization_stats, as_torch=True)
    return _move_batch(batch, device)


def _unnormalize_future(normalized_future: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    return ((normalized_future + 1.0) * (max_val - min_val) / 2.0 + min_val).to(torch.float32)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum().item()) <= 0:
        return values.new_tensor(0.0)
    return values[valid].mean()


def _temporal_energy_risk(
    temporal_energy: torch.Tensor,
    *,
    distance_scale: float,
) -> torch.Tensor:
    if temporal_energy.ndim != 5 or int(temporal_energy.shape[-1]) < 5:
        raise ValueError(f"temporal energy must have shape [B,K,A,T,C>=5], got {tuple(temporal_energy.shape)}")
    min_neighbor_distance = temporal_energy[..., 0].clamp_min(0.0)
    soft_collision_energy = temporal_energy[..., 1].clamp_min(0.0)
    close_neighbor_count = temporal_energy[..., 2].clamp_min(0.0)
    approaching_score = temporal_energy[..., 3].clamp_min(0.0)
    endpoint_crowding_energy = temporal_energy[..., 4].clamp_min(0.0)
    distance_risk = torch.exp(-min_neighbor_distance / max(float(distance_scale), 1e-6))
    soft_risk = soft_collision_energy / (1.0 + soft_collision_energy)
    close_risk = close_neighbor_count / (1.0 + close_neighbor_count)
    endpoint_risk = endpoint_crowding_energy / (1.0 + endpoint_crowding_energy)
    risk = torch.stack(
        [
            distance_risk,
            soft_risk,
            close_risk,
            approaching_score.clamp(0.0, 1.0),
            endpoint_risk,
        ],
        dim=0,
    ).amax(dim=0)
    return torch.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _flow_pred_data(
    predictor: MoFlowSlowPredictor,
    x_data: Mapping[str, Any],
    *,
    y_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if bool(predictor.cfg.fm_in_scaling):
        y_t_in = y_t * predictor.engine.get_input_scaling(t).reshape(-1, 1, 1, 1)
    else:
        y_t_in = y_t
    model_out, _pred_score = predictor.engine.model(y_t_in, t, x_data=x_data)
    return predictor.engine.fm_wrapper_func(y_t, t, model_out)


def _sample_flow_state(
    predictor: MoFlowSlowPredictor,
    batch: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gt = batch["fut_traj"]
    num_modes = int(predictor.cfg.denoising_head_preds)
    gt_flat = gt[:, None].expand(-1, num_modes, -1, -1, -1).reshape(
        int(gt.shape[0]),
        num_modes,
        int(gt.shape[1]),
        int(predictor.cfg.future_frames) * 2,
    )
    t, y_t, _u_t, _target, _l_weight = predictor.engine.get_loss_input(gt_flat)
    return t, y_t


def _temporal_energy_gt_loss(
    prediction_metric: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    risk = _temporal_energy_risk(
        temporal_energy.to(device=prediction_metric.device, dtype=prediction_metric.dtype),
        distance_scale=float(args.temporal_energy_distance_scale),
    )
    weight = risk.clamp_min(float(args.temporal_energy_gt_risk_floor)).pow(
        float(args.temporal_energy_gt_risk_power)
    )
    error = torch.linalg.norm(prediction_metric - ground_truth[:, None, ...], dim=-1)
    weighted_error = error * weight
    score = weighted_error.mean(dim=-1) + float(args.fde_weight) * weighted_error[..., -1]
    best = score.min(dim=1).values
    loss = _masked_mean(best, mask)
    valid = mask[:, None, :, None].expand_as(weight).bool()
    risk_mean = float(risk[valid].detach().mean().cpu()) if bool(valid.any()) else 0.0
    weight_mean = float(weight[valid].detach().mean().cpu()) if bool(valid.any()) else 0.0
    return loss, {
        "temporal_energy_risk_mean": risk_mean,
        "temporal_energy_gt_weight_mean": weight_mean,
    }


def _process_keep_loss(
    prediction_metric: torch.Tensor,
    base_metric: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = torch.linalg.norm(prediction_metric - base_metric, dim=-1).mean(dim=-1).mean(dim=1)
    return _masked_mean(error, mask)


def _accuracy_summary(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    miss_threshold: float,
) -> Dict[str, float]:
    pred_errors = displacement_errors(prediction, ground_truth, agent_mask=mask)
    ref_errors = displacement_errors(reference, ground_truth, agent_mask=mask)
    valid = pred_errors["valid_agents"].bool()
    valid_expanded = valid[:, None, :]
    inf = torch.tensor(float("inf"), device=prediction.device, dtype=prediction.dtype)

    pred_ade = pred_errors["ade_per_mode_agent"]
    pred_fde = pred_errors["fde_per_mode_agent"]
    ref_ade = ref_errors["ade_per_mode_agent"]
    ref_fde = ref_errors["fde_per_mode_agent"]

    pred_fde_min = pred_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    ref_fde_min = ref_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    ref_best_index = ref_fde.argmin(dim=1)
    ref_best = torch.gather(ref_fde, dim=1, index=ref_best_index[:, None, :]).squeeze(1)
    pred_at_ref_best = torch.gather(pred_fde, dim=1, index=ref_best_index[:, None, :]).squeeze(1)
    ref_best_delta = pred_at_ref_best - ref_best
    pred_miss = pred_fde_min > float(miss_threshold)
    ref_miss = ref_fde_min > float(miss_threshold)
    return {
        "finetuned_ADE_min": float(pred_ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu()),
        "finetuned_FDE_min": float(pred_fde_min[valid].mean().cpu()),
        "finetuned_ADE_avg": float(pred_ade.mean(dim=1)[valid].mean().cpu()),
        "finetuned_FDE_avg": float(pred_fde.mean(dim=1)[valid].mean().cpu()),
        "finetuned_MissRate": float(pred_miss[valid].float().mean().cpu()),
        "base_ADE_min": float(ref_ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu()),
        "base_FDE_min": float(ref_fde_min[valid].mean().cpu()),
        "base_ADE_avg": float(ref_ade.mean(dim=1)[valid].mean().cpu()),
        "base_FDE_avg": float(ref_fde.mean(dim=1)[valid].mean().cpu()),
        "base_MissRate": float(ref_miss[valid].float().mean().cpu()),
        "dFDE_min": float((pred_fde_min - ref_fde_min)[valid].mean().cpu()),
        "dMissRate": float((pred_miss[valid].float() - ref_miss[valid].float()).mean().cpu()),
        "base_best_fde_delta": float(ref_best_delta[valid].mean().cpu()),
        "base_best_hurt_mean": float(F.relu(ref_best_delta[valid]).mean().cpu()),
        "base_best_worse_rate": float((ref_best_delta[valid] > 0.0).float().mean().cpu()),
    }


def _auxiliary_loss_step(
    predictor: MoFlowSlowPredictor,
    frozen_predictor: MoFlowSlowPredictor,
    samples: Sequence[Mapping[str, Any]],
    batch: Mapping[str, Any],
    *,
    normalization_stats: Mapping[str, float],
    args: argparse.Namespace,
    model_train: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predictor.engine.model.train(bool(model_train))
    frozen_predictor.engine.model.eval()

    t, y_t = _sample_flow_state(predictor, batch)
    pred_norm_flat = _flow_pred_data(predictor, batch, y_t=y_t, t=t)
    with torch.no_grad():
        base_norm_flat = _flow_pred_data(frozen_predictor, batch, y_t=y_t, t=t)

    batch_size = int(pred_norm_flat.shape[0])
    num_modes = int(pred_norm_flat.shape[1])
    num_agents = int(pred_norm_flat.shape[2])
    future_frames = int(predictor.cfg.future_frames)
    pred_norm = pred_norm_flat.reshape(batch_size, num_modes, num_agents, future_frames, 2)
    base_norm = base_norm_flat.reshape(batch_size, num_modes, num_agents, future_frames, 2)
    pred_metric = _unnormalize_future(pred_norm, normalization_stats)
    base_metric = _unnormalize_future(base_norm.detach(), normalization_stats)
    ground_truth = batch["fut_traj_original_scale"].to(device=pred_metric.device, dtype=pred_metric.dtype)
    mask = batch["agent_mask"].to(device=pred_metric.device).bool()

    temporal_energy = build_per_agent_scene_temporal_interaction_features(
        samples,
        pred_metric.detach(),
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
    ).to(device=pred_metric.device, dtype=pred_metric.dtype)
    loss_temporal_gt, temporal_metrics = _temporal_energy_gt_loss(
        pred_metric,
        ground_truth,
        temporal_energy,
        mask,
        args=args,
    )
    loss_keep = _process_keep_loss(pred_metric, base_metric, mask)
    loss = (
        float(args.lambda_temporal_energy_gt) * loss_temporal_gt
        + float(args.lambda_process_keep) * loss_keep
    )
    metrics = {
        "loss_temporal_energy_gt": float(loss_temporal_gt.detach().cpu()),
        "loss_process_keep": float(loss_keep.detach().cpu()),
        **temporal_metrics,
        **_accuracy_summary(
            pred_metric,
            base_metric,
            ground_truth,
            mask,
            miss_threshold=float(args.miss_threshold),
        ),
    }
    return loss, metrics


def _weighted_average(rows: Iterable[Mapping[str, float]], weights: Iterable[int]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    total = 0
    for row, weight in zip(rows, weights):
        total += int(weight)
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value) * int(weight)
    if total <= 0:
        raise ValueError("Cannot average metrics with zero total weight")
    return {key: value / total for key, value in sums.items()}


def _run_train_epoch(
    predictor: MoFlowSlowPredictor,
    frozen_predictor: MoFlowSlowPredictor,
    samples: Sequence[Mapping[str, Any]],
    *,
    normalization_stats: Mapping[str, float],
    optimizer: torch.optim.Optimizer,
    device: str,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    predictor.engine.model.train()
    rows: List[Dict[str, float]] = []
    weights: List[int] = []
    batches = list(
        _iter_epoch_batches(
            samples,
            batch_scenes=int(args.batch_scenes),
            seed=int(args.seed),
            epoch=int(epoch),
            shuffle=True,
        )
    )
    for batch_index, chunk in enumerate(batches, start=1):
        batch = _prepare_moflow_batch(predictor, chunk, normalization_stats=normalization_stats, device=device)
        optimizer.zero_grad(set_to_none=True)
        base_losses = predictor.compute_loss(batch, log_dict={"cur_epoch": int(epoch)})
        aux_loss, aux_metrics = _auxiliary_loss_step(
            predictor,
            frozen_predictor,
            chunk,
            batch,
            normalization_stats=normalization_stats,
            args=args,
            model_train=True,
        )
        loss = float(args.lambda_base_flow) * base_losses["loss"] + aux_loss
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in predictor.engine.model.parameters() if param.requires_grad],
                max_norm=float(args.grad_clip),
            )
        optimizer.step()
        valid = int(batch["agent_mask"].bool().sum().item())
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "loss_base_flow": float(base_losses["loss"].detach().cpu()),
                "loss_base_reg": float(base_losses["loss_reg"].detach().cpu()),
                "loss_base_cls": float(base_losses["loss_cls"].detach().cpu()),
                **aux_metrics,
            }
        )
        weights.append(valid)
        if batch_index == 1 or batch_index == len(batches) or batch_index % max(int(args.log_every), 1) == 0:
            latest = rows[-1]
            print(
                "[finetune_teacher_flow] "
                f"epoch={epoch} batch={batch_index}/{len(batches)} "
                f"loss={latest['loss']:.6f} dFDE={latest['dFDE_min']:+.6f} "
                f"risk={latest['temporal_energy_risk_mean']:.4f}"
            )
    return _weighted_average(rows, weights)


@torch.no_grad()
def _evaluate_process(
    predictor: MoFlowSlowPredictor,
    frozen_predictor: MoFlowSlowPredictor,
    samples: Sequence[Mapping[str, Any]],
    *,
    normalization_stats: Mapping[str, float],
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    predictor.engine.model.eval()
    frozen_predictor.engine.model.eval()
    rows: List[Dict[str, float]] = []
    weights: List[int] = []
    batch_scenes = int(args.val_batch_scenes) if args.val_batch_scenes is not None else int(args.batch_scenes)
    for chunk in _iter_epoch_batches(samples, batch_scenes=batch_scenes, seed=int(args.seed), epoch=0, shuffle=False):
        batch = _prepare_moflow_batch(predictor, chunk, normalization_stats=normalization_stats, device=device)
        aux_loss, aux_metrics = _auxiliary_loss_step(
            predictor,
            frozen_predictor,
            chunk,
            batch,
            normalization_stats=normalization_stats,
            args=args,
            model_train=False,
        )
        valid = int(batch["agent_mask"].bool().sum().item())
        rows.append({"loss": float(aux_loss.detach().cpu()), **aux_metrics})
        weights.append(valid)
    return _weighted_average(rows, weights)


def _selection_score(metrics: Mapping[str, float], args: argparse.Namespace) -> float:
    fde_min = float(metrics["finetuned_FDE_min"])
    if args.selection_metric == "fde_min":
        return fde_min
    miss_delta = max(0.0, float(metrics.get("dMissRate", 0.0)))
    keep_hurt = max(0.0, float(metrics.get("base_best_hurt_mean", 0.0)))
    worse_rate = max(0.0, float(metrics.get("base_best_worse_rate", 0.0)))
    return (
        fde_min
        + float(args.selection_miss_weight) * miss_delta
        + float(args.selection_keep_weight) * keep_hurt
        + float(args.selection_worse_rate_weight) * worse_rate
    )


def _save_checkpoint(
    path: Path,
    predictor: MoFlowSlowPredictor,
    *,
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    normalization_stats: Mapping[str, float],
    trainable_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": predictor.engine.state_dict(),
            "meta": {
                "variant": "v23a_teacher_flow_temporal_energy_finetune",
                "epoch": int(epoch),
                "best_selection_score": float(best_score),
                "selection_metric": args.selection_metric,
                "base_slow_checkpoint": str(Path(args.slow_checkpoint).expanduser().resolve()),
                "base_slow_cfg_path": str(Path(args.slow_cfg_path).expanduser().resolve()),
                "trainable_scope": args.trainable_scope,
                "trainable_param_names": list(trainable_names),
                "normalization_stats": dict(normalization_stats),
            },
            "args": _jsonable(vars(args)),
            "train_metrics": _jsonable(train_metrics),
            "val_metrics": _jsonable(val_metrics),
        },
        path,
    )


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split="train",
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, args.max_scenes)
    train_samples, val_samples = _split_train_val(
        selected_samples,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )
    agents = _infer_agents(selected_samples, args.sample_mode, args.agents)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)

    predictor = MoFlowSlowPredictor(
        _predictor_cfg(args=args, agents=agents, device=device, checkpoint_path=args.slow_checkpoint)
    )
    frozen_predictor = MoFlowSlowPredictor(
        _predictor_cfg(args=args, agents=agents, device=device, checkpoint_path=args.slow_checkpoint)
    )
    for param in frozen_predictor.parameters():
        param.requires_grad_(False)
    frozen_predictor.eval()

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[predictor, frozen_predictor],
        samples=selected_samples,
        stats_owner=predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    predictor._set_normalization_stats(normalization_stats)
    frozen_predictor._set_normalization_stats(normalization_stats)

    trainable_names = _set_trainable_scope(predictor.engine.model, args.trainable_scope)
    trainable_params = [param for param in predictor.engine.model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_path = output_dir / f"{args.run_name}_best.pt"
    last_path = output_dir / f"{args.run_name}_last.pt"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    best_score = float("inf")
    best_epoch = 0
    best_train_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    print(
        "[finetune_teacher_flow] "
        f"variant=v23a split=train scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"train_scenes={len(train_samples)} val_scenes={len(val_samples)} device={device} "
        f"scope={args.trainable_scope} trainable_params={sum(p.numel() for p in trainable_params)}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[finetune_teacher_flow] warning: selected_samples normalization is diagnostic only")

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_train_epoch(
            predictor,
            frozen_predictor,
            train_samples,
            normalization_stats=normalization_stats,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
        )
        val_metrics = _evaluate_process(
            predictor,
            frozen_predictor,
            val_samples,
            normalization_stats=normalization_stats,
            device=device,
            args=args,
        )
        score = _selection_score(val_metrics, args)
        history.append(
            {
                "epoch": int(epoch),
                "selection_score": float(score),
                "train_metrics": _jsonable(train_metrics),
                "val_metrics": _jsonable(val_metrics),
            }
        )
        if score < best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            _save_checkpoint(
                best_path,
                predictor,
                args=args,
                epoch=epoch,
                best_score=best_score,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                normalization_stats=normalization_stats,
                trainable_names=trainable_names,
            )
        _save_checkpoint(
            last_path,
            predictor,
            args=args,
            epoch=epoch,
            best_score=best_score,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            normalization_stats=normalization_stats,
            trainable_names=trainable_names,
        )
        print(
            "[finetune_teacher_flow] "
            f"epoch={epoch} val_FDE={val_metrics['finetuned_FDE_min']:.6f} "
            f"base_FDE={val_metrics['base_FDE_min']:.6f} dFDE={val_metrics['dFDE_min']:+.6f} "
            f"dMiss={val_metrics['dMissRate']:+.6f} score={score:.6f} best_epoch={best_epoch}"
        )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.finetune_teacher_flow",
            "variant": "v23a_teacher_flow_temporal_energy_finetune",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "selection_metric": args.selection_metric,
            "best_checkpoint": best_path.as_posix(),
            "last_checkpoint": last_path.as_posix(),
        },
        "args": _jsonable(vars(args)),
        "dataset": {
            **_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
            "num_train_scenes": len(train_samples),
            "num_val_scenes": len(val_samples),
        },
        "normalization_stats": _jsonable(normalization_stats),
        "normalization_meta": _jsonable(normalization_meta),
        "trainable": {
            "scope": args.trainable_scope,
            "num_params": int(sum(param.numel() for param in trainable_params)),
            "param_names": list(trainable_names),
        },
        "best_train_metrics": _jsonable(best_train_metrics),
        "best_val_metrics": _jsonable(best_val_metrics),
        "history": _jsonable(history),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_checkpoint={best_path.as_posix()}")
    print(f"last_checkpoint={last_path.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
