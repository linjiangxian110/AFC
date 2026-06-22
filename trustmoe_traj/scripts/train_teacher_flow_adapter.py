"""Train a frozen-teacher flow residual adapter.

V22-A inserts a lightweight residual adapter into the MoFlow slow teacher's
flow process.  The slow teacher backbone is frozen; only the adapter learns to
adjust the teacher's intermediate ``pred_data`` prediction using observed-past
social-risk/context signals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trustmoe_traj.data.transforms import DEFAULT_PAST_SOCIAL_RISK_DIM
from trustmoe_traj.models import (
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
    build_teacher_flow_adapter_for_engine,
)
from trustmoe_traj.scripts.train_student_integrated_finetune import (
    DEFAULT_CACHE_PATH,
    CacheDataset,
    _gt_min_loss,
    _jsonable,
    _load_cache,
    _move_batch,
    _normalize_future,
    _prepare_tensors,
    _resolve_device,
    _select_indices,
    _selection_score,
    _set_chamfer_loss,
    _set_seed,
    _student_best_nohurt_loss,
    _summarize_prediction,
    _unnormalize_future,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "teacher_flow_adapter_models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a teacher-side flow residual adapter.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="teacher_flow_adapter")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.10)
    parser.add_argument("--gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--use-past-social-risk", action="store_true")
    parser.add_argument("--social-risk-dim", type=int, default=DEFAULT_PAST_SOCIAL_RISK_DIM)
    parser.add_argument("--use-temporal-interaction-energy", action="store_true")
    parser.add_argument("--temporal-energy-dim", type=int, default=5)
    parser.add_argument("--temporal-energy-distance-scale", type=float, default=0.5)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-gt-min", type=float, default=1.0)
    parser.add_argument("--lambda-keep-pred-data", type=float, default=0.2)
    parser.add_argument("--lambda-teacher-best-nohurt", type=float, default=1.0)
    parser.add_argument("--teacher-best-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-flow-delta", type=float, default=0.001)
    parser.add_argument("--lambda-flow-gate", type=float, default=0.001)
    parser.add_argument("--lambda-temporal-energy-gt", type=float, default=0.0)
    parser.add_argument("--temporal-energy-gt-risk-floor", type=float, default=0.05)
    parser.add_argument("--temporal-energy-gt-risk-power", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-energy-gate-target", type=float, default=0.0)
    parser.add_argument("--temporal-energy-gate-target-min", type=float, default=0.001)
    parser.add_argument("--temporal-energy-gate-target-max", type=float, default=0.20)
    parser.add_argument("--selection-metric", type=str, default="safety", choices=["fde_min", "safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-nohurt-weight", type=float, default=1.0)
    parser.add_argument("--selection-student-best-weight", type=float, default=1.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _variant_name(args: argparse.Namespace) -> str:
    if bool(args.use_temporal_interaction_energy) and float(args.lambda_temporal_energy_gt) > 0.0:
        return "v22c_teacher_flow_temporal_energy_direction_adapter"
    if bool(args.use_temporal_interaction_energy):
        return "v22b_teacher_flow_temporal_interaction_energy_adapter"
    return (
        "v22a_teacher_flow_past_social_risk_adapter"
        if bool(args.use_past_social_risk)
        else "v22a_teacher_flow_adapter"
    )


def _prepare_teacher_tensors(
    payload: Mapping[str, Any],
    *,
    use_past_social_risk: bool,
    use_temporal_interaction_energy: bool,
) -> Dict[str, torch.Tensor]:
    tensors = _prepare_tensors(payload)
    stats = payload["normalization_stats"]
    tensors["ground_truth_normalized"] = _normalize_future(tensors["ground_truth"], stats)
    raw_tensors = payload.get("tensors", {})
    if bool(use_past_social_risk):
        if not isinstance(raw_tensors, Mapping) or "past_social_risk_features" not in raw_tensors:
            raise ValueError(
                "Teacher flow adapter with --use-past-social-risk requires cache tensor "
                "`past_social_risk_features`. Re-export the teacher/student cache first."
            )
        tensors["past_social_risk_features"] = raw_tensors["past_social_risk_features"].to(torch.float32)
    if bool(use_temporal_interaction_energy):
        if not isinstance(raw_tensors, Mapping):
            raise ValueError("Teacher flow adapter temporal energy requires cache tensors")
        energy_key = None
        for candidate in (
            "teacher_temporal_interaction_energy_features",
            "temporal_interaction_energy_features",
        ):
            if candidate in raw_tensors:
                energy_key = candidate
                break
        if energy_key is None:
            raise ValueError(
                "Teacher flow adapter with --use-temporal-interaction-energy requires cache tensor "
                "`teacher_temporal_interaction_energy_features` or `temporal_interaction_energy_features`. "
                "Re-export the teacher/student cache with temporal interaction energy first."
            )
        tensors["temporal_interaction_energy_features"] = raw_tensors[energy_key].to(torch.float32)
    return tensors


def _teacher_x_data(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    payload = {
        "past_traj_original_scale": batch["past_traj_original_scale"],
        "fut_traj_original_scale": batch["ground_truth"],
        "fut_traj": batch["ground_truth_normalized"],
        "agent_mask": batch["agent_mask"],
        "batch_size": torch.as_tensor(batch["ground_truth"].shape[0], device=batch["ground_truth"].device),
    }
    if "past_social_risk_features" in batch:
        payload["past_social_risk_features"] = batch["past_social_risk_features"]
    if "temporal_interaction_energy_features" in batch:
        payload["temporal_interaction_energy_features"] = batch["temporal_interaction_energy_features"]
    if "teacher_temporal_interaction_energy_features" in batch:
        payload["teacher_temporal_interaction_energy_features"] = batch[
            "teacher_temporal_interaction_energy_features"
        ]
    return payload


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


def _temporal_energy_gate_target_loss(
    gate: torch.Tensor,
    temporal_energy: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    risk = _temporal_energy_risk(
        temporal_energy.to(device=gate.device, dtype=gate.dtype),
        distance_scale=float(args.temporal_energy_distance_scale),
    )
    target_min = float(args.temporal_energy_gate_target_min)
    target_max = float(args.temporal_energy_gate_target_max)
    target = (target_min + (target_max - target_min) * risk).unsqueeze(-1).to(dtype=gate.dtype)
    valid = mask[:, None, :, None, None].to(device=gate.device, dtype=torch.bool).expand_as(gate)
    if not bool(valid.any()):
        zero = gate.new_tensor(0.0)
        return zero, {"temporal_energy_risk_mean": 0.0, "temporal_energy_gate_target_mean": 0.0}
    loss = F.mse_loss(gate[valid], target[valid])
    metrics = {
        "temporal_energy_risk_mean": float(risk[valid.squeeze(-1)].detach().mean().cpu()),
        "temporal_energy_gate_target_mean": float(target[valid].detach().mean().cpu()),
    }
    return loss, metrics


def _temporal_energy_gt_loss(
    refined: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Risk-weighted GT loss that gives the flow residual a correction direction."""

    risk = _temporal_energy_risk(
        temporal_energy.to(device=refined.device, dtype=refined.dtype),
        distance_scale=float(args.temporal_energy_distance_scale),
    )
    weight = risk.clamp_min(float(args.temporal_energy_gt_risk_floor)).pow(
        float(args.temporal_energy_gt_risk_power)
    )
    error = torch.linalg.norm(refined - ground_truth[:, None, ...], dim=-1)
    weighted_error = error * weight
    mode_score = weighted_error.mean(dim=-1) + float(args.fde_weight) * weighted_error[..., -1]
    best_score = mode_score.min(dim=1).values
    valid = mask.to(device=refined.device, dtype=torch.bool)
    if not bool(valid.any()):
        zero = refined.new_tensor(0.0)
        return zero, {"temporal_energy_gt_loss_weight_mean": 0.0}
    return best_score[valid].mean(), {
        "temporal_energy_gt_loss_weight_mean": float(weight[valid[:, None, :, None].expand_as(weight)].detach().mean().cpu())
    }


def _sample_flow_state(predictor: MoFlowSlowPredictor, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    gt = batch["ground_truth_normalized"]
    num_modes = int(predictor.cfg.denoising_head_preds)
    gt_flat = gt[:, None].expand(-1, num_modes, -1, -1, -1).reshape(
        int(gt.shape[0]),
        num_modes,
        int(gt.shape[1]),
        int(predictor.cfg.future_frames) * 2,
    )
    t, y_t, _u_t, _target, _l_weight = predictor.engine.get_loss_input(gt_flat)
    return t, y_t


def _set_modes(predictor: MoFlowSlowPredictor, *, adapter_train: bool) -> None:
    predictor.engine.eval()
    predictor.engine.model.eval()
    adapter = getattr(predictor.engine, "teacher_flow_adapter", None)
    if adapter is None:
        raise RuntimeError("teacher_flow_adapter is not attached")
    adapter.train(bool(adapter_train))


def _teacher_flow_loss_step(
    predictor: MoFlowSlowPredictor,
    batch: Mapping[str, torch.Tensor],
    *,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    adapter = getattr(predictor.engine, "teacher_flow_adapter", None)
    if adapter is None:
        raise RuntimeError("teacher_flow_adapter is not attached")

    x_data = _teacher_x_data(batch)
    t, y_t = _sample_flow_state(predictor, batch)
    if bool(predictor.cfg.fm_in_scaling):
        y_t_in = y_t * predictor.engine.get_input_scaling(t).reshape(-1, 1, 1, 1)
    else:
        y_t_in = y_t

    model_out, _pred_score = predictor.engine.model(y_t_in, t, x_data=x_data)
    raw_pred_data = predictor.engine.fm_wrapper_func(y_t, t, model_out)
    refined_pred_data = adapter(raw_pred_data, y_t=y_t, t=t, x_data=x_data, return_dict=False)

    batch_size = int(refined_pred_data.shape[0])
    num_modes = int(refined_pred_data.shape[1])
    num_agents = int(refined_pred_data.shape[2])
    future_frames = int(predictor.cfg.future_frames)
    refined_norm = refined_pred_data.reshape(batch_size, num_modes, num_agents, future_frames, 2)
    raw_norm = raw_pred_data.reshape(batch_size, num_modes, num_agents, future_frames, 2).detach()
    refined = _unnormalize_future(refined_norm, normalization_stats)
    raw = _unnormalize_future(raw_norm, normalization_stats)

    gt = batch["ground_truth"]
    mask = batch["agent_mask"].bool()
    pred_for_loss = refined.unsqueeze(1)
    loss_gt = _gt_min_loss(pred_for_loss, gt, mask, fde_weight=args.fde_weight)
    loss_keep = _set_chamfer_loss(pred_for_loss, raw, mask, fde_weight=args.fde_weight)
    loss_nohurt = _student_best_nohurt_loss(
        pred_for_loss,
        raw,
        gt,
        mask,
        margin=float(args.teacher_best_nohurt_margin),
    )
    if adapter.last_delta is None or adapter.last_gate is None:
        raise RuntimeError("teacher flow adapter did not expose diagnostics")
    loss_delta = adapter.last_delta.pow(2).mean()
    loss_gate = adapter.last_gate.mean()
    temporal_energy = x_data.get("teacher_temporal_interaction_energy_features")
    if temporal_energy is None:
        temporal_energy = x_data.get("temporal_interaction_energy_features")
    loss_temporal_energy_gt = refined.new_tensor(0.0)
    temporal_energy_gt_metrics: Dict[str, float] = {}
    if float(args.lambda_temporal_energy_gt) > 0.0:
        if temporal_energy is None:
            raise ValueError("--lambda-temporal-energy-gt requires temporal interaction energy features")
        loss_temporal_energy_gt, temporal_energy_gt_metrics = _temporal_energy_gt_loss(
            refined,
            gt,
            temporal_energy,
            mask,
            args=args,
        )
    loss_temporal_gate_target = refined.new_tensor(0.0)
    temporal_gate_metrics: Dict[str, float] = {}
    if float(args.lambda_temporal_energy_gate_target) > 0.0:
        if temporal_energy is None:
            raise ValueError("--lambda-temporal-energy-gate-target requires temporal interaction energy features")
        loss_temporal_gate_target, temporal_gate_metrics = _temporal_energy_gate_target_loss(
            adapter.last_gate,
            temporal_energy,
            mask,
            args=args,
        )
    loss = (
        float(args.lambda_gt_min) * loss_gt
        + float(args.lambda_keep_pred_data) * loss_keep
        + float(args.lambda_teacher_best_nohurt) * loss_nohurt
        + float(args.lambda_flow_delta) * loss_delta
        + float(args.lambda_flow_gate) * loss_gate
        + float(args.lambda_temporal_energy_gt) * loss_temporal_energy_gt
        + float(args.lambda_temporal_energy_gate_target) * loss_temporal_gate_target
    )
    metrics = _summarize_prediction(
        pred_for_loss,
        raw,
        gt,
        mask,
        miss_threshold=float(args.miss_threshold),
    )
    components = {
        "loss": float(loss.detach().cpu()),
        "loss_gt_min": float(loss_gt.detach().cpu()),
        "loss_keep_pred_data": float(loss_keep.detach().cpu()),
        "loss_teacher_best_nohurt": float(loss_nohurt.detach().cpu()),
        "loss_flow_delta": float(loss_delta.detach().cpu()),
        "loss_flow_gate": float(loss_gate.detach().cpu()),
        "loss_temporal_energy_gt": float(loss_temporal_energy_gt.detach().cpu()),
        "loss_temporal_energy_gate_target": float(loss_temporal_gate_target.detach().cpu()),
        "flow_gate_mean": float(adapter.last_gate.detach().mean().cpu()),
        "flow_delta_l2_mean": float(adapter.last_delta.detach().pow(2).mean().sqrt().cpu()),
        **temporal_energy_gt_metrics,
        **temporal_gate_metrics,
    }
    return loss, {**components, **metrics}, {"teacher_flow_pred": refined.detach(), "teacher_raw_pred": raw.detach()}


@torch.no_grad()
def _evaluate(
    predictor: MoFlowSlowPredictor,
    loader: DataLoader,
    *,
    device: str,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, float]:
    _set_modes(predictor, adapter_train=False)
    sums: Dict[str, float] = {}
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        _loss, components, _output = _teacher_flow_loss_step(
            predictor,
            batch,
            normalization_stats=normalization_stats,
            args=args,
        )
        valid = int(batch["agent_mask"].bool().sum().item())
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value) * valid
        total += valid
    if total <= 0:
        raise ValueError("Validation loader has no valid agents")
    return {key: value / total for key, value in sums.items()}


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_cache(cache_path)
    tensors = _prepare_teacher_tensors(
        payload,
        use_past_social_risk=bool(args.use_past_social_risk),
        use_temporal_interaction_energy=bool(args.use_temporal_interaction_energy),
    )
    train_indices, val_indices = _select_indices(
        int(tensors["ground_truth"].shape[0]),
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )

    predictor = MoFlowSlowPredictor(
        MoFlowPredictorConfig(
            subset=args.subset,
            sample_mode=args.sample_mode,
            agents=1,
            data_norm=args.data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    predictor._set_normalization_stats(payload["normalization_stats"])
    past_shape = list(tensors["past_traj_original_scale"].shape)
    adapter = build_teacher_flow_adapter_for_engine(
        predictor.engine,
        past_frames=int(past_shape[2]),
        past_feature_dim=int(past_shape[3]),
        social_risk_dim=int(args.social_risk_dim),
        temporal_energy_dim=int(args.temporal_energy_dim),
        hidden_dim=int(args.hidden_dim),
        max_modes=args.max_modes,
        use_past_social_risk=bool(args.use_past_social_risk),
        use_temporal_interaction_energy=bool(args.use_temporal_interaction_energy),
        residual_scale=float(args.residual_scale),
        max_delta=args.max_delta,
        gate_init_bias=float(args.gate_init_bias),
    ).to(device)
    predictor.attach_teacher_flow_adapter(adapter)
    for param in predictor.engine.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)
    trainable_names = [f"teacher_flow_adapter.{name}" for name, param in adapter.named_parameters() if param.requires_grad]

    train_loader = DataLoader(
        CacheDataset(tensors, train_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    val_loader = DataLoader(
        CacheDataset(tensors, val_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / f"{args.run_name}_best.pt"
    last_path = output_dir / f"{args.run_name}_last.pt"

    print(
        "[train_teacher_flow_adapter] "
        f"cache={cache_path.as_posix()} train_items={len(train_indices)} val_items={len(val_indices)} "
        f"device={device} trainable_params={len(trainable_names)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        _set_modes(predictor, adapter_train=True)
        train_sums: Dict[str, float] = {}
        train_total = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, _output = _teacher_flow_loss_step(
                predictor,
                batch,
                normalization_stats=payload["normalization_stats"],
                args=args,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            optimizer.step()

            valid = int(batch["agent_mask"].bool().sum().item())
            for key, value in components.items():
                train_sums[key] = train_sums.get(key, 0.0) + float(value) * valid
            train_total += valid

        train_metrics = {key: value / max(train_total, 1) for key, value in train_sums.items()}
        val_metrics = _evaluate(
            predictor,
            val_loader,
            device=device,
            normalization_stats=payload["normalization_stats"],
            args=args,
        )
        selection_score = _selection_score(val_metrics, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": selection_score}
        history.append(row)
        if float(selection_score) < best_metric:
            best_metric = float(selection_score)
            best_epoch = epoch
            torch.save(
                {
                    "model_state": adapter.state_dict(),
                    "config": asdict(adapter.config),
                    "meta": {
                        "script": "trustmoe_traj.scripts.train_teacher_flow_adapter",
                        "variant": _variant_name(args),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "base_slow_checkpoint": str(Path(args.slow_checkpoint).expanduser().resolve()),
                        "base_slow_cfg_path": str(Path(args.slow_cfg_path).expanduser().resolve()),
                        "cache_path": cache_path.as_posix(),
                        "run_name": args.run_name,
                        "seed": int(args.seed),
                        "best_epoch": int(best_epoch),
                        "selection_metric": args.selection_metric,
                        "selection_score": float(selection_score),
                    },
                    "args": _jsonable(vars(args)),
                    "normalization_stats": _jsonable(payload["normalization_stats"]),
                    "best_val_metrics": _jsonable(val_metrics),
                    "best_selection_score": float(selection_score),
                    "trainable_names": trainable_names,
                },
                best_path,
            )

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(int(args.log_every), 1) == 0:
            print(
                "[train_teacher_flow_adapter] "
                f"epoch={epoch:03d} train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_FDE_min={val_metrics['finetuned_FDE_min']:.6f} "
                f"val_dFDE={val_metrics['dFDE_min']:+.6f} "
                f"val_dMiss={val_metrics.get('dMissRate', 0.0):+.6f} "
                f"teacher_best_hurt={val_metrics.get('student_best_hurt_mean', 0.0):.6f} "
                f"gate={val_metrics.get('flow_gate_mean', 0.0):.4f} "
                f"delta_l2={val_metrics.get('flow_delta_l2_mean', 0.0):.4f} "
                f"select={selection_score:.6f}"
            )

    torch.save(
        {
            "model_state": adapter.state_dict(),
            "config": asdict(adapter.config),
            "meta": {
                "script": "trustmoe_traj.scripts.train_teacher_flow_adapter",
                "variant": _variant_name(args),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_slow_checkpoint": str(Path(args.slow_checkpoint).expanduser().resolve()),
                "base_slow_cfg_path": str(Path(args.slow_cfg_path).expanduser().resolve()),
                "cache_path": cache_path.as_posix(),
                "run_name": args.run_name,
                "seed": int(args.seed),
                "best_epoch": int(best_epoch),
                "best_checkpoint": best_path.as_posix(),
                "selection_metric": args.selection_metric,
                "best_selection_score": float(best_metric),
            },
            "args": _jsonable(vars(args)),
            "normalization_stats": _jsonable(payload["normalization_stats"]),
            "history": _jsonable(history),
            "trainable_names": trainable_names,
        },
        last_path,
    )
    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_teacher_flow_adapter",
            "variant": _variant_name(args),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cache_path": cache_path.as_posix(),
            "run_name": args.run_name,
            "seed": int(args.seed),
            "best_epoch": int(best_epoch),
            "best_checkpoint": best_path.as_posix(),
            "last_checkpoint": last_path.as_posix(),
            "selection_metric": args.selection_metric,
            "best_selection_score": float(best_metric),
        },
        "args": _jsonable(vars(args)),
        "normalization_stats": _jsonable(payload["normalization_stats"]),
        "train_items": len(train_indices),
        "val_items": len(val_indices),
        "best_val_metrics": _jsonable(history[best_epoch - 1]["val"] if best_epoch else {}),
        "history": _jsonable(history),
        "trainable_names": trainable_names,
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
