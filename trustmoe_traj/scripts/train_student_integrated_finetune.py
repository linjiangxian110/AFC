"""V18-B: fine-tune the MoFlow fast student with integrated losses.

Unlike V18-A, this script updates the fast student's own generator parameters
instead of training an external adapter.  It uses the exported teacher/student
cache as a compact supervision source:

- slow teacher set distillation;
- best-of-K ground-truth supervision;
- temporal-energy weighted supervision on risky timesteps;
- no-hurt / diversity / keep regularizers against the original fast student.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from trustmoe_traj.evaluation import displacement_errors
from trustmoe_traj.models import MoFlowFastPredictor, MoFlowPredictorConfig


DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "teacher_student_cache"
    / "official_align_eth_train_teacher_student_predictions.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "student_integrated_models"

REQUIRED_TENSOR_KEYS: Sequence[str] = (
    "student_pred",
    "teacher_pred",
    "ground_truth",
    "agent_mask",
    "past_traj_original_scale",
)


class CacheDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor], indices: Sequence[int]) -> None:
        self.tensors = dict(tensors)
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {key: tensor[source_index] for key, tensor in self.tensors.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V18-B fine-tuning for the MoFlow fast student.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="student_finetune")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--fast-cfg-path", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate", dest="rotate", action="store_true")
    rotate_group.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.set_defaults(rotate=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)

    parser.add_argument("--trainable-scope", type=str, default="decoder", choices=["head", "decoder", "all"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-set", type=float, default=1.0)
    parser.add_argument("--lambda-gt-min", type=float, default=0.3)
    parser.add_argument("--lambda-energy-gt", type=float, default=0.2)
    parser.add_argument("--energy-gt-top-k", type=int, default=2)
    parser.add_argument("--energy-risk-floor", type=float, default=0.05)
    parser.add_argument("--lambda-good-nohurt", type=float, default=1.0)
    parser.add_argument("--good-nohurt-frac", type=float, default=0.25)
    parser.add_argument("--good-nohurt-margin", type=float, default=0.0)
    parser.add_argument(
        "--lambda-student-best-nohurt",
        type=float,
        default=0.0,
        help="Weight for directly protecting the original student best-FDE mode.",
    )
    parser.add_argument(
        "--student-best-nohurt-margin",
        type=float,
        default=0.0,
        help="Tolerance before penalizing the finetuned version of the original student best-FDE mode.",
    )
    parser.add_argument("--lambda-diversity-preserve", type=float, default=0.2)
    parser.add_argument("--diversity-preserve-target-ratio", type=float, default=0.98)
    parser.add_argument("--lambda-keep-set", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="fde_min",
        choices=["fde_min", "safety"],
        help="Checkpoint selection metric. `safety` penalizes MissRate and protected-mode regressions.",
    )
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-nohurt-weight", type=float, default=1.0)
    parser.add_argument("--selection-student-best-weight", type=float, default=1.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device!r}, but CUDA is not available")
    return str(resolved)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Teacher/student cache not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "tensors" not in payload:
        raise ValueError("Invalid teacher/student cache payload")
    tensors = dict(payload["tensors"])
    missing = [key for key in REQUIRED_TENSOR_KEYS if key not in tensors]
    if missing:
        raise ValueError(f"Cache is missing required tensor(s): {', '.join(missing)}")
    stats = dict(payload.get("normalization_stats", {}))
    for key in ("past_traj_min", "past_traj_max", "fut_traj_min", "fut_traj_max"):
        if key not in stats:
            raise ValueError(f"Cache normalization_stats is missing {key!r}")
    num_items = int(tensors["ground_truth"].shape[0])
    for key, tensor in list(tensors.items()):
        if torch.is_tensor(tensor) and int(tensor.shape[0]) == num_items:
            tensors[key] = tensor.detach().cpu()
    return {**dict(payload), "tensors": tensors, "normalization_stats": stats}


def _normalize_future(metric_future: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    return (2.0 * (metric_future - min_val) / max(max_val - min_val, 1e-12) - 1.0).to(torch.float32)


def _unnormalize_future(normalized_future: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    return ((normalized_future + 1.0) * (max_val - min_val) / 2.0 + min_val).to(torch.float32)


def _prepare_tensors(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    tensors = dict(payload["tensors"])
    stats = payload["normalization_stats"]
    prepared: Dict[str, torch.Tensor] = {
        "student_pred": tensors["student_pred"].to(torch.float32),
        "teacher_pred": tensors["teacher_pred"].to(torch.float32),
        "ground_truth": tensors["ground_truth"].to(torch.float32),
        "agent_mask": tensors["agent_mask"].bool(),
        "past_traj_original_scale": tensors["past_traj_original_scale"].to(torch.float32),
    }
    prepared["teacher_pred_normalized"] = _normalize_future(prepared["teacher_pred"], stats)
    prepared["ground_truth_normalized"] = _normalize_future(prepared["ground_truth"], stats)
    if "temporal_interaction_energy_features" in tensors:
        prepared["temporal_interaction_energy_features"] = tensors["temporal_interaction_energy_features"].to(torch.float32)
    return prepared


def _select_indices(num_items: int, *, seed: int, max_items: Optional[int], val_fraction: float) -> tuple[List[int], List[int]]:
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(int(num_items), generator=generator).tolist()
    if max_items is not None:
        indices = indices[: min(int(max_items), len(indices))]
    if len(indices) <= 1 or float(val_fraction) == 0.0:
        return indices, indices
    val_count = max(1, int(round(len(indices) * float(val_fraction))))
    val_count = min(val_count, len(indices) - 1)
    return indices[val_count:], indices[:val_count]


def _move_batch(batch: Mapping[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, dtype=torch.float32) if value.dtype.is_floating_point else value.to(device=device)
        for key, value in batch.items()
    }


def _set_trainable_scope(model: torch.nn.Module, scope: str) -> List[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_prefixes: Sequence[str]
    if scope == "head":
        trainable_prefixes = ("reg_head",)
    elif scope == "decoder":
        trainable_prefixes = (
            "motion_query_embedding",
            "agent_order_embedding",
            "noisy_vec_mlp",
            "pe_mlp",
            "init_emb_fusion_mlp",
            "motion_decoder",
            "reg_head",
        )
    elif scope == "all":
        trainable_prefixes = ("",)
    else:
        raise ValueError(f"Unsupported trainable scope: {scope!r}")

    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad_(True)
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f"No trainable parameters selected for scope={scope!r}")
    return trainable_names


def _generate_student(
    predictor: MoFlowFastPredictor,
    batch: Mapping[str, torch.Tensor],
    *,
    normalization_stats: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    predictor._set_normalization_stats(normalization_stats)
    prepared = predictor._prepare_batch(
        {
            "past_traj_original_scale": batch["past_traj_original_scale"],
            "fut_traj_original_scale": batch["ground_truth"],
            "fut_traj": batch["ground_truth_normalized"],
            "agent_mask": batch["agent_mask"],
            **(
                {"past_social_risk_features": batch["past_social_risk_features"]}
                if "past_social_risk_features" in batch
                else {}
            ),
        }
    )
    generated_norm = predictor.engine.model(prepared, num_to_gen=int(predictor.cfg.num_to_gen))
    generated_norm = generated_norm.reshape(
        generated_norm.shape[0],
        generated_norm.shape[1],
        generated_norm.shape[2],
        generated_norm.shape[3],
        int(predictor.cfg.future_frames),
        2,
    )
    generated_metric = _unnormalize_future(generated_norm, normalization_stats)
    return generated_norm, generated_metric


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum().item()) <= 0:
        return values.new_tensor(0.0)
    return values[valid].mean()


def _weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = weights.to(dtype=values.dtype) * mask.to(dtype=values.dtype)
    denom = weight.sum().clamp_min(1e-6)
    return (values * weight).sum() / denom


def _score(prediction: torch.Tensor, ground_truth: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]


def _flatten_draw_modes(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 6:
        raise ValueError(f"Expected [B,M,K,A,T,2], got {tuple(prediction.shape)}")
    b, m, k, a, t, d = prediction.shape
    return prediction.reshape(b, m * k, a, t, d)


def _gt_min_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    flat_pred = _flatten_draw_modes(prediction)
    return _masked_mean(_score(flat_pred, ground_truth, fde_weight=fde_weight).min(dim=1).values, mask)


def _set_chamfer_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    flat_pred = _flatten_draw_modes(prediction).permute(0, 2, 1, 3, 4)
    target_by_agent = target.permute(0, 2, 1, 3, 4)
    pairwise = flat_pred[:, :, :, None, :, :] - target_by_agent[:, :, None, :, :, :]
    dist = torch.linalg.norm(pairwise, dim=-1)
    pair_score = dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]
    pred_to_target = pair_score.min(dim=-1).values.mean(dim=-1)
    target_to_pred = pair_score.min(dim=-2).values.mean(dim=-1)
    return _masked_mean(pred_to_target + target_to_pred, mask)


def _good_nohurt_loss(
    prediction: torch.Tensor,
    student: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    good_frac: float,
    margin: float,
) -> torch.Tensor:
    pred_first = prediction[:, 0]
    student_score = _score(student, ground_truth, fde_weight=fde_weight)
    pred_score = _score(pred_first, ground_truth, fde_weight=fde_weight)
    num_modes = int(student_score.shape[1])
    keep_k = max(1, min(num_modes, int(round(num_modes * float(good_frac)))))
    good_index = student_score.argsort(dim=1)[:, :keep_k, :]
    selected_student = torch.gather(student_score, dim=1, index=good_index)
    selected_pred = torch.gather(pred_score, dim=1, index=good_index)
    return _masked_mean(F.relu(selected_pred - selected_student - float(margin)).mean(dim=1), mask)


def _student_best_nohurt_loss(
    prediction: torch.Tensor,
    student: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Protect the original fast student's best-FDE mode from being degraded."""

    pred_first = prediction[:, 0]
    student_fde = torch.linalg.norm(student - ground_truth[:, None, ...], dim=-1)[..., -1]
    pred_fde = torch.linalg.norm(pred_first - ground_truth[:, None, ...], dim=-1)[..., -1]
    best_index = student_fde.argmin(dim=1)
    selected_student = torch.gather(student_fde, dim=1, index=best_index[:, None, :]).squeeze(1)
    selected_pred = torch.gather(pred_fde, dim=1, index=best_index[:, None, :]).squeeze(1)
    return _masked_mean(F.relu(selected_pred - selected_student - float(margin)), mask)


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return pairwise.new_zeros((pairwise.shape[0],))
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, _t, d = prediction.shape
    endpoints = prediction[..., -1, :].permute(0, 2, 1, 3).reshape(b * a, k, d)
    return _offdiag_mean(torch.cdist(endpoints, endpoints, p=2)).reshape(b, a)


def _trajectory_spread(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, t, d = prediction.shape
    traj = prediction.permute(0, 2, 1, 3, 4).reshape(b * a, k, t, d)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean(pairwise).reshape(b, a)


def _diversity_loss(prediction: torch.Tensor, student: torch.Tensor, mask: torch.Tensor, *, target_ratio: float) -> torch.Tensor:
    pred_first = prediction[:, 0]
    endpoint_penalty = F.relu(float(target_ratio) * _endpoint_spread(student) - _endpoint_spread(pred_first))
    traj_penalty = F.relu(float(target_ratio) * _trajectory_spread(student) - _trajectory_spread(pred_first))
    return _masked_mean(endpoint_penalty + traj_penalty, mask)


def _energy_risk(temporal_energy: torch.Tensor) -> torch.Tensor:
    if temporal_energy.shape[-1] <= 1:
        return temporal_energy.new_zeros(temporal_energy.shape[:-1])
    parts = [temporal_energy[..., 1]]
    if temporal_energy.shape[-1] > 2:
        parts.append(0.25 * temporal_energy[..., 2])
    if temporal_energy.shape[-1] > 3:
        parts.append(temporal_energy[..., 3])
    if temporal_energy.shape[-1] > 4:
        parts.append(temporal_energy[..., 4])
    risk = torch.stack([item.clamp_min(0.0) for item in parts], dim=0).sum(dim=0)
    scale = risk.detach().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (risk / scale).clamp(0.0, 1.0)


def _energy_gt_loss(
    prediction: torch.Tensor,
    student: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    fde_weight: float,
    top_k: int,
    risk_floor: float,
) -> torch.Tensor:
    if temporal_energy is None:
        return prediction.new_tensor(0.0)
    pred_first = prediction[:, 0]
    num_modes = int(pred_first.shape[1])
    keep_k = max(1, min(int(top_k), num_modes))
    student_score = _score(student, ground_truth, fde_weight=fde_weight)
    selected_index = student_score.argsort(dim=1)[:, :keep_k, :]
    error = torch.linalg.norm(pred_first - ground_truth[:, None, ...], dim=-1)
    risk = _energy_risk(temporal_energy).to(device=prediction.device, dtype=prediction.dtype)
    selected_error = torch.gather(
        error,
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(pred_first.shape[3])),
    )
    selected_risk = torch.gather(
        risk,
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(pred_first.shape[3])),
    )
    expanded_mask = mask[:, None, :, None].expand_as(selected_error)
    return _weighted_masked_mean(selected_error, selected_risk.clamp_min(float(risk_floor)), expanded_mask)


def _summarize_prediction(prediction: torch.Tensor, student: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor, *, miss_threshold: float) -> Dict[str, float]:
    pred_first = prediction[:, 0]
    pred_errors = displacement_errors(pred_first, ground_truth, agent_mask=mask)
    student_errors = displacement_errors(student, ground_truth, agent_mask=mask)
    valid = pred_errors["valid_agents"].bool()
    valid_expanded = valid[:, None, :]
    inf = torch.tensor(float("inf"), device=prediction.device, dtype=prediction.dtype)
    pred_ade = pred_errors["ade_per_mode_agent"]
    pred_fde = pred_errors["fde_per_mode_agent"]
    student_fde = student_errors["fde_per_mode_agent"]
    pred_fde_min = pred_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    student_fde_min = student_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    student_best_index = student_fde.argmin(dim=1)
    student_best_fde = torch.gather(student_fde, dim=1, index=student_best_index[:, None, :]).squeeze(1)
    pred_at_student_best_fde = torch.gather(pred_fde, dim=1, index=student_best_index[:, None, :]).squeeze(1)
    student_best_delta = pred_at_student_best_fde - student_best_fde
    student_miss = student_fde_min > float(miss_threshold)
    pred_miss = pred_fde_min > float(miss_threshold)
    return {
        "finetuned_ADE_min": float(pred_ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu()),
        "finetuned_FDE_min": float(pred_fde_min[valid].mean().cpu()),
        "finetuned_ADE_avg": float(pred_ade.mean(dim=1)[valid].mean().cpu()),
        "finetuned_FDE_avg": float(pred_fde.mean(dim=1)[valid].mean().cpu()),
        "finetuned_MissRate": float((pred_fde_min > float(miss_threshold))[valid].float().mean().cpu()),
        "student_MissRate": float(student_miss[valid].float().mean().cpu()),
        "student_FDE_min": float(student_fde_min[valid].mean().cpu()),
        "dFDE_min": float((pred_fde_min - student_fde_min)[valid].mean().cpu()),
        "dMissRate": float((pred_miss[valid].float() - student_miss[valid].float()).mean().cpu()),
        "student_best_fde_delta": float(student_best_delta[valid].mean().cpu()),
        "student_best_worse_rate": float((student_best_delta[valid] > 0.0).float().mean().cpu()),
        "student_best_hurt_mean": float(F.relu(student_best_delta[valid]).mean().cpu()),
        "endpoint_ratio": float((_endpoint_spread(pred_first) / _endpoint_spread(student).abs().clamp_min(1e-8))[valid].mean().cpu()),
        "trajectory_ratio": float((_trajectory_spread(pred_first) / _trajectory_spread(student).abs().clamp_min(1e-8))[valid].mean().cpu()),
    }


def _loss_step(
    predictor: MoFlowFastPredictor,
    batch: Mapping[str, torch.Tensor],
    *,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    _generated_norm, generated = _generate_student(predictor, batch, normalization_stats=normalization_stats)
    student = batch["student_pred"]
    teacher = batch["teacher_pred"]
    gt = batch["ground_truth"]
    mask = batch["agent_mask"].bool()
    loss_teacher = _set_chamfer_loss(generated, teacher, mask, fde_weight=args.fde_weight)
    loss_gt = _gt_min_loss(generated, gt, mask, fde_weight=args.fde_weight)
    loss_energy = _energy_gt_loss(
        generated,
        student,
        gt,
        batch.get("temporal_interaction_energy_features"),
        mask,
        fde_weight=args.fde_weight,
        top_k=args.energy_gt_top_k,
        risk_floor=args.energy_risk_floor,
    )
    loss_nohurt = _good_nohurt_loss(
        generated,
        student,
        gt,
        mask,
        fde_weight=args.fde_weight,
        good_frac=args.good_nohurt_frac,
        margin=args.good_nohurt_margin,
    )
    loss_student_best_nohurt = _student_best_nohurt_loss(
        generated,
        student,
        gt,
        mask,
        margin=args.student_best_nohurt_margin,
    )
    loss_div = _diversity_loss(
        generated,
        student,
        mask,
        target_ratio=args.diversity_preserve_target_ratio,
    )
    loss_keep = _set_chamfer_loss(generated, student, mask, fde_weight=args.fde_weight)
    loss = (
        float(args.lambda_teacher_set) * loss_teacher
        + float(args.lambda_gt_min) * loss_gt
        + float(args.lambda_energy_gt) * loss_energy
        + float(args.lambda_good_nohurt) * loss_nohurt
        + float(args.lambda_student_best_nohurt) * loss_student_best_nohurt
        + float(args.lambda_diversity_preserve) * loss_div
        + float(args.lambda_keep_set) * loss_keep
    )
    components = {
        "loss": float(loss.detach().cpu()),
        "loss_teacher_set": float(loss_teacher.detach().cpu()),
        "loss_gt_min": float(loss_gt.detach().cpu()),
        "loss_energy_gt": float(loss_energy.detach().cpu()),
        "loss_good_nohurt": float(loss_nohurt.detach().cpu()),
        "loss_student_best_nohurt": float(loss_student_best_nohurt.detach().cpu()),
        "loss_diversity": float(loss_div.detach().cpu()),
        "loss_keep_set": float(loss_keep.detach().cpu()),
    }
    return loss, components, {"finetuned_pred": generated[:, 0].detach()}


@torch.no_grad()
def _evaluate(
    predictor: MoFlowFastPredictor,
    loader: DataLoader,
    *,
    device: str,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, float]:
    predictor.engine.model.eval()
    sums: Dict[str, float] = {}
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        _loss, components, output = _loss_step(predictor, batch, normalization_stats=normalization_stats, args=args)
        valid = int(batch["agent_mask"].bool().sum().item())
        metrics = _summarize_prediction(
            output["finetuned_pred"].unsqueeze(1),
            batch["student_pred"],
            batch["ground_truth"],
            batch["agent_mask"].bool(),
            miss_threshold=args.miss_threshold,
        )
        for key, value in {**components, **metrics}.items():
            sums[key] = sums.get(key, 0.0) + float(value) * valid
        total += valid
    if total <= 0:
        raise ValueError("Validation loader has no valid agents")
    return {key: value / total for key, value in sums.items()}


def _selection_score(metrics: Mapping[str, float], args: argparse.Namespace) -> float:
    fde_min = float(metrics["finetuned_FDE_min"])
    if args.selection_metric == "fde_min":
        return fde_min
    miss_delta = max(0.0, float(metrics.get("dMissRate", 0.0)))
    student_best_hurt = max(0.0, float(metrics.get("student_best_hurt_mean", 0.0)))
    student_best_worse = max(0.0, float(metrics.get("student_best_worse_rate", 0.0)))
    return (
        fde_min
        + float(args.selection_miss_weight) * miss_delta
        + float(args.selection_nohurt_weight) * student_best_hurt
        + float(args.selection_student_best_weight) * student_best_worse
    )


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_cache(cache_path)
    tensors = _prepare_tensors(payload)
    train_indices, val_indices = _select_indices(
        int(tensors["ground_truth"].shape[0]),
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )
    predictor = MoFlowFastPredictor(
        MoFlowPredictorConfig(
            subset=args.subset,
            sample_mode=args.sample_mode,
            agents=1,
            data_norm=args.data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            device=device,
            cfg_path=args.fast_cfg_path,
            checkpoint_path=args.fast_checkpoint,
            num_to_gen=int(args.num_to_gen),
        )
    )
    predictor._set_normalization_stats(payload["normalization_stats"])
    trainable_names = _set_trainable_scope(predictor.engine.model, args.trainable_scope)
    optimizer = torch.optim.AdamW(
        [param for param in predictor.engine.model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

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
    best_metric = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / f"{args.run_name}_best.pt"
    last_path = output_dir / f"{args.run_name}_last.pt"

    print(
        "[train_student_integrated_finetune] "
        f"cache={cache_path.as_posix()} train_items={len(train_indices)} val_items={len(val_indices)} "
        f"device={device} scope={args.trainable_scope} trainable_params={len(trainable_names)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        predictor.engine.model.train()
        train_sums: Dict[str, float] = {}
        train_total = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, _output = _loss_step(
                predictor,
                batch,
                normalization_stats=payload["normalization_stats"],
                args=args,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in predictor.engine.model.parameters() if param.requires_grad],
                    float(args.grad_clip),
                )
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
        current = float(selection_score)
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            torch.save(
                {
                    "model": predictor.engine.state_dict(),
                    "meta": {
                        "script": "trustmoe_traj.scripts.train_student_integrated_finetune",
                        "variant": "v18b_fast_student_decoder_finetune",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "base_fast_checkpoint": str(Path(args.fast_checkpoint).expanduser().resolve()),
                        "base_fast_cfg_path": str(Path(args.fast_cfg_path).expanduser().resolve()),
                        "cache_path": cache_path.as_posix(),
                        "run_name": args.run_name,
                        "seed": int(args.seed),
                        "best_epoch": int(best_epoch),
                        "selection_metric": args.selection_metric,
                        "selection_score": float(current),
                    },
                    "args": _jsonable(vars(args)),
                    "normalization_stats": _jsonable(payload["normalization_stats"]),
                    "best_val_metrics": _jsonable(val_metrics),
                    "best_selection_score": float(current),
                    "trainable_names": trainable_names,
                },
                best_path,
            )

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(int(args.log_every), 1) == 0:
            print(
                "[train_student_integrated_finetune] "
                f"epoch={epoch:03d} train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_FDE_min={val_metrics['finetuned_FDE_min']:.6f} "
                f"val_dFDE={val_metrics['dFDE_min']:+.6f} "
                f"val_dMiss={val_metrics.get('dMissRate', 0.0):+.6f} "
                f"student_best_hurt={val_metrics.get('student_best_hurt_mean', 0.0):.6f} "
                f"select={current:.6f} "
                f"endpoint_ratio={val_metrics.get('endpoint_ratio', 0.0):.4f}"
            )

    torch.save(
        {
            "model": predictor.engine.state_dict(),
            "meta": {
                "script": "trustmoe_traj.scripts.train_student_integrated_finetune",
                "variant": "v18b_fast_student_decoder_finetune",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_fast_checkpoint": str(Path(args.fast_checkpoint).expanduser().resolve()),
                "base_fast_cfg_path": str(Path(args.fast_cfg_path).expanduser().resolve()),
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
            "script": "trustmoe_traj.scripts.train_student_integrated_finetune",
            "variant": "v18b_fast_student_decoder_finetune",
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
