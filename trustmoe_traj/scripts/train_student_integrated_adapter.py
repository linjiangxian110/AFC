"""Train the V18-A student-integrated adapter from teacher/student caches."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - numpy is present in normal experiment envs.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from trustmoe_traj.evaluation import displacement_errors
from trustmoe_traj.models import build_student_integrated_adapter_from_cache_shapes


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


class TeacherStudentCacheDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor], indices: Sequence[int]) -> None:
        self.tensors = dict(tensors)
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {key: tensor[source_index] for key, tensor in self.tensors.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V18-A student-integrated adapter from exported caches.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="student_integrated_v18a")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mode-context-num-layers", type=int, default=1)
    parser.add_argument("--mode-context-num-heads", type=int, default=4)
    parser.add_argument("--mode-context-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-energy-dim", type=int, default=5)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--no-temporal-energy", action="store_true")

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-gt-min", type=float, default=1.0)
    parser.add_argument("--lambda-teacher", type=float, default=0.2)
    parser.add_argument("--lambda-energy-gt", type=float, default=0.2)
    parser.add_argument("--energy-gt-top-k", type=int, default=2)
    parser.add_argument("--energy-risk-floor", type=float, default=0.05)
    parser.add_argument("--lambda-keep", type=float, default=0.05)
    parser.add_argument("--lambda-good-nohurt", type=float, default=1.0)
    parser.add_argument("--good-nohurt-frac", type=float, default=0.25)
    parser.add_argument("--good-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-diversity-preserve", type=float, default=0.2)
    parser.add_argument("--diversity-preserve-target-ratio", type=float, default=0.98)
    parser.add_argument("--lambda-delta", type=float, default=0.001)
    parser.add_argument("--lambda-gate", type=float, default=0.001)
    parser.add_argument("--lambda-temporal-smoothness", type=float, default=0.005)
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
    if "normalization_stats" not in payload:
        raise ValueError("Cache is missing normalization_stats; V18-A trains in normalized MoFlow coordinates")

    num_items = int(tensors["ground_truth"].shape[0])
    for key, tensor in list(tensors.items()):
        if torch.is_tensor(tensor) and int(tensor.shape[0]) == num_items:
            tensors[key] = tensor.detach().cpu()
    stats = dict(payload["normalization_stats"])
    for key in ("fut_traj_min", "fut_traj_max"):
        if key not in stats:
            raise ValueError(f"normalization_stats is missing {key!r}")
    return {**dict(payload), "tensors": tensors, "normalization_stats": stats}


def _normalize_future(metric_future: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    return (2.0 * (metric_future - min_val) / max(max_val - min_val, 1e-12) - 1.0).to(torch.float32)


def _unnormalize_future(normalized_future: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    return ((normalized_future + 1.0) * (max_val - min_val) / 2.0 + min_val).to(torch.float32)


def _select_indices(num_items: int, *, seed: int, max_items: Optional[int], val_fraction: float) -> tuple[List[int], List[int]]:
    if num_items <= 0:
        raise ValueError("Cache contains no rows")
    if max_items is not None and int(max_items) <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(num_items, generator=generator).tolist()
    if max_items is not None:
        indices = indices[: min(int(max_items), len(indices))]
    if len(indices) <= 1 or float(val_fraction) == 0.0:
        return indices, indices
    val_count = max(1, int(round(len(indices) * float(val_fraction))))
    val_count = min(val_count, len(indices) - 1)
    return indices[val_count:], indices[:val_count]


def _move_batch(batch: Mapping[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.dtype.is_floating_point:
            moved[key] = value.to(device=device, dtype=torch.float32)
        else:
            moved[key] = value.to(device=device)
    return moved


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum().item()) <= 0:
        return values.new_tensor(0.0)
    return values[valid].mean()


def _weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = weights.to(dtype=values.dtype) * mask.to(dtype=values.dtype)
    denom = weight.sum().clamp_min(1e-6)
    return (values * weight).sum() / denom


def _mode_scores(prediction: torch.Tensor, ground_truth: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]


def _gt_min_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    score = _mode_scores(prediction, ground_truth, fde_weight=fde_weight)
    return _masked_mean(score.min(dim=1).values, agent_mask)


def _teacher_distill_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    pred_by_agent = prediction.permute(0, 2, 1, 3, 4)
    teacher_by_agent = teacher.permute(0, 2, 1, 3, 4)
    pairwise = pred_by_agent[:, :, :, None, :, :] - teacher_by_agent[:, :, None, :, :, :]
    dist = torch.linalg.norm(pairwise, dim=-1)
    score = dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]
    nearest = score.min(dim=-1).values.mean(dim=-1)
    return _masked_mean(nearest, agent_mask)


def _keep_loss(prediction: torch.Tensor, student: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - student, dim=-1).mean(dim=-1)
    return _masked_mean(dist.mean(dim=1), agent_mask)


def _good_nohurt_loss(
    prediction: torch.Tensor,
    student: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
    good_frac: float,
    margin: float,
) -> torch.Tensor:
    student_score = _mode_scores(student, ground_truth, fde_weight=fde_weight)
    pred_score = _mode_scores(prediction, ground_truth, fde_weight=fde_weight)
    num_modes = int(student_score.shape[1])
    keep_k = max(1, min(num_modes, int(round(num_modes * float(good_frac)))))
    good_index = student_score.argsort(dim=1)[:, :keep_k, :]
    selected_student = torch.gather(student_score, dim=1, index=good_index)
    selected_pred = torch.gather(pred_score, dim=1, index=good_index)
    penalty = F.relu(selected_pred - selected_student - float(margin)).mean(dim=1)
    return _masked_mean(penalty, agent_mask)


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return pairwise.new_zeros((pairwise.shape[0],))
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    batch_size, num_modes, num_agents, _pred_len, coord_dim = prediction.shape
    endpoints = prediction[..., -1, :].permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_modes, coord_dim)
    pairwise = torch.cdist(endpoints, endpoints, p=2)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _trajectory_spread(prediction: torch.Tensor) -> torch.Tensor:
    batch_size, num_modes, num_agents, pred_len, coord_dim = prediction.shape
    traj = prediction.permute(0, 2, 1, 3, 4).reshape(batch_size * num_agents, num_modes, pred_len, coord_dim)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _diversity_loss(
    prediction: torch.Tensor,
    student: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    target_ratio: float,
) -> torch.Tensor:
    endpoint_penalty = F.relu(float(target_ratio) * _endpoint_spread(student) - _endpoint_spread(prediction))
    traj_penalty = F.relu(float(target_ratio) * _trajectory_spread(student) - _trajectory_spread(prediction))
    return _masked_mean(endpoint_penalty + traj_penalty, agent_mask)


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
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
    top_k: int,
    risk_floor: float,
) -> torch.Tensor:
    if temporal_energy is None:
        return prediction.new_tensor(0.0)
    num_modes = int(prediction.shape[1])
    keep_k = max(1, min(int(top_k), num_modes))
    student_score = _mode_scores(student, ground_truth, fde_weight=fde_weight)
    selected_index = student_score.argsort(dim=1)[:, :keep_k, :]

    error = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    risk = _energy_risk(temporal_energy).to(device=prediction.device, dtype=prediction.dtype)
    selected_error = torch.gather(
        error,
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(prediction.shape[3])),
    )
    selected_risk = torch.gather(
        risk,
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(prediction.shape[3])),
    )
    weights = selected_risk.clamp_min(float(risk_floor))
    expanded_mask = agent_mask[:, None, :, None].expand_as(selected_error)
    return _weighted_masked_mean(selected_error, weights, expanded_mask)


def _loss_step(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    student_norm = batch["student_pred_normalized"]
    output = model(
        student_norm,
        past_traj_original_scale=batch["past_traj_original_scale"],
        temporal_interaction_energy_features=batch.get("temporal_interaction_energy_features"),
        return_dict=True,
    )
    adapted_norm = output["adapted_future_normalized"]
    adapted = _unnormalize_future(adapted_norm, normalization_stats)
    student = batch["student_pred"]
    teacher = batch["teacher_pred"]
    ground_truth = batch["ground_truth"]
    mask = batch["agent_mask"].bool()

    loss_gt = _gt_min_loss(adapted, ground_truth, mask, fde_weight=args.fde_weight)
    loss_teacher = _teacher_distill_loss(adapted, teacher, mask, fde_weight=args.fde_weight)
    loss_keep = _keep_loss(adapted, student, mask)
    loss_nohurt = _good_nohurt_loss(
        adapted,
        student,
        ground_truth,
        mask,
        fde_weight=args.fde_weight,
        good_frac=args.good_nohurt_frac,
        margin=args.good_nohurt_margin,
    )
    loss_div = _diversity_loss(
        adapted,
        student,
        mask,
        target_ratio=args.diversity_preserve_target_ratio,
    )
    loss_delta = output["delta_normalized"].pow(2).mean()
    loss_gate = output["gate"].mean()
    if int(output["delta_normalized"].shape[-2]) > 1:
        temporal_step = output["delta_normalized"][..., 1:, :] - output["delta_normalized"][..., :-1, :]
        loss_smooth = temporal_step.pow(2).mean()
    else:
        loss_smooth = adapted.new_tensor(0.0)
    loss_energy = _energy_gt_loss(
        adapted,
        student,
        ground_truth,
        batch.get("temporal_interaction_energy_features"),
        mask,
        fde_weight=args.fde_weight,
        top_k=args.energy_gt_top_k,
        risk_floor=args.energy_risk_floor,
    )

    loss = (
        float(args.lambda_gt_min) * loss_gt
        + float(args.lambda_teacher) * loss_teacher
        + float(args.lambda_energy_gt) * loss_energy
        + float(args.lambda_keep) * loss_keep
        + float(args.lambda_good_nohurt) * loss_nohurt
        + float(args.lambda_diversity_preserve) * loss_div
        + float(args.lambda_delta) * loss_delta
        + float(args.lambda_gate) * loss_gate
        + float(args.lambda_temporal_smoothness) * loss_smooth
    )
    components = {
        "loss": float(loss.detach().cpu()),
        "loss_gt_min": float(loss_gt.detach().cpu()),
        "loss_teacher": float(loss_teacher.detach().cpu()),
        "loss_energy_gt": float(loss_energy.detach().cpu()),
        "loss_keep": float(loss_keep.detach().cpu()),
        "loss_good_nohurt": float(loss_nohurt.detach().cpu()),
        "loss_diversity": float(loss_div.detach().cpu()),
        "loss_delta": float(loss_delta.detach().cpu()),
        "loss_gate": float(loss_gate.detach().cpu()),
        "loss_temporal_smoothness": float(loss_smooth.detach().cpu()),
        "gate_mean": float(output["gate"].detach().mean().cpu()),
        "delta_l2_mean": float(output["delta_normalized"].detach().pow(2).mean().sqrt().cpu()),
    }
    return loss, components, {"student_integrated_pred": adapted.detach(), "gate": output["gate"].detach()}


def _summarize_predictions(
    *,
    prediction: torch.Tensor,
    student: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    miss_threshold: float,
) -> Dict[str, float]:
    pred_errors = displacement_errors(prediction, ground_truth, agent_mask=agent_mask)
    student_errors = displacement_errors(student, ground_truth, agent_mask=agent_mask)
    valid = pred_errors["valid_agents"].bool()
    valid_expanded = valid[:, None, :]
    inf = torch.tensor(float("inf"), device=prediction.device, dtype=prediction.dtype)

    ade = pred_errors["ade_per_mode_agent"]
    fde = pred_errors["fde_per_mode_agent"]
    student_ade = student_errors["ade_per_mode_agent"]
    student_fde = student_errors["fde_per_mode_agent"]
    pred_fde_min = fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    student_fde_min = student_fde.masked_fill(~valid_expanded, inf).min(dim=1).values

    endpoint_ratio = _endpoint_spread(prediction) / _endpoint_spread(student).abs().clamp_min(1e-8)
    traj_ratio = _trajectory_spread(prediction) / _trajectory_spread(student).abs().clamp_min(1e-8)
    return {
        "student_integrated_ADE_min": float(ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu()),
        "student_integrated_FDE_min": float(pred_fde_min[valid].mean().cpu()),
        "student_integrated_ADE_avg": float(ade.mean(dim=1)[valid].mean().cpu()),
        "student_integrated_FDE_avg": float(fde.mean(dim=1)[valid].mean().cpu()),
        "student_integrated_MissRate": float((pred_fde_min > float(miss_threshold))[valid].float().mean().cpu()),
        "student_FDE_min": float(student_fde_min[valid].mean().cpu()),
        "dFDE_min": float((pred_fde_min - student_fde_min)[valid].mean().cpu()),
        "endpoint_ratio": float(endpoint_ratio[valid].mean().cpu()),
        "trajectory_ratio": float(traj_ratio[valid].mean().cpu()),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        _loss, components, output = _loss_step(model, batch, normalization_stats=normalization_stats, args=args)
        valid = int(batch["agent_mask"].bool().sum().item())
        metrics = _summarize_predictions(
            prediction=output["student_integrated_pred"],
            student=batch["student_pred"],
            ground_truth=batch["ground_truth"],
            agent_mask=batch["agent_mask"].bool(),
            miss_threshold=args.miss_threshold,
        )
        for key, value in {**components, **metrics}.items():
            sums[key] = sums.get(key, 0.0) + float(value) * valid
        total += valid
    if total <= 0:
        raise ValueError("Validation loader has no valid agents")
    return {key: value / total for key, value in sums.items()}


def _prepare_tensors(payload: Mapping[str, Any], *, use_temporal_energy: bool) -> Dict[str, torch.Tensor]:
    tensors = dict(payload["tensors"])
    stats = payload["normalization_stats"]
    prepared: Dict[str, torch.Tensor] = {
        "student_pred": tensors["student_pred"].to(torch.float32),
        "teacher_pred": tensors["teacher_pred"].to(torch.float32),
        "ground_truth": tensors["ground_truth"].to(torch.float32),
        "agent_mask": tensors["agent_mask"].bool(),
        "past_traj_original_scale": tensors["past_traj_original_scale"].to(torch.float32),
    }
    prepared["student_pred_normalized"] = _normalize_future(prepared["student_pred"], stats)
    if use_temporal_energy:
        if "temporal_interaction_energy_features" not in tensors:
            raise ValueError(
                "V18-A temporal-energy training requires cache tensor `temporal_interaction_energy_features`. "
                "Re-export with --include-temporal-interaction-energy-features or pass --no-temporal-energy."
            )
        prepared["temporal_interaction_energy_features"] = tensors["temporal_interaction_energy_features"].to(torch.float32)
    return prepared


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_cache(cache_path)
    use_temporal_energy = not bool(args.no_temporal_energy)
    tensors = _prepare_tensors(payload, use_temporal_energy=use_temporal_energy)
    tensor_shapes = {key: list(value.shape) for key, value in tensors.items() if torch.is_tensor(value)}
    train_indices, val_indices = _select_indices(
        int(tensors["ground_truth"].shape[0]),
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )

    model = build_student_integrated_adapter_from_cache_shapes(
        tensor_shapes,
        hidden_dim=int(args.hidden_dim),
        num_mode_context_layers=int(args.mode_context_num_layers),
        num_mode_context_heads=int(args.mode_context_num_heads),
        mode_context_dropout=float(args.mode_context_dropout),
        use_temporal_energy=use_temporal_energy,
        temporal_energy_dim=int(args.temporal_energy_dim),
        residual_scale=float(args.residual_scale),
        max_delta=args.max_delta,
        gate_init_bias=float(args.gate_init_bias),
    ).to(device)

    train_loader = DataLoader(
        TeacherStudentCacheDataset(tensors, train_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    val_loader = DataLoader(
        TeacherStudentCacheDataset(tensors, val_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / f"{args.run_name}_best.pt"
    last_path = output_dir / f"{args.run_name}_last.pt"

    print(
        "[train_student_integrated_adapter] "
        f"cache={cache_path.as_posix()} train_items={len(train_indices)} val_items={len(val_indices)} device={device}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_sums: Dict[str, float] = {}
        train_total = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, _output = _loss_step(
                model,
                batch,
                normalization_stats=payload["normalization_stats"],
                args=args,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            valid = int(batch["agent_mask"].bool().sum().item())
            for key, value in components.items():
                train_sums[key] = train_sums.get(key, 0.0) + float(value) * valid
            train_total += valid

        train_metrics = {key: value / max(train_total, 1) for key, value in train_sums.items()}
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            normalization_stats=payload["normalization_stats"],
            args=args,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        current = float(val_metrics["student_integrated_FDE_min"])
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(model.config),
                    "normalization_stats": _jsonable(payload["normalization_stats"]),
                    "meta": {
                        "script": "trustmoe_traj.scripts.train_student_integrated_adapter",
                        "variant": "v18a_student_integrated_adapter",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "cache_path": cache_path.as_posix(),
                        "run_name": args.run_name,
                        "seed": int(args.seed),
                        "best_epoch": int(best_epoch),
                    },
                    "args": _jsonable(vars(args)),
                    "best_val_metrics": _jsonable(val_metrics),
                    "tensor_shapes": _jsonable(tensor_shapes),
                },
                best_path,
            )

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(int(args.log_every), 1) == 0:
            print(
                "[train_student_integrated_adapter] "
                f"epoch={epoch:03d} train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_FDE_min={val_metrics['student_integrated_FDE_min']:.6f} "
                f"val_dFDE={val_metrics['dFDE_min']:+.6f} "
                f"gate={val_metrics.get('gate_mean', 0.0):.4f} "
                f"delta_l2={val_metrics.get('delta_l2_mean', 0.0):.4f}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
            "normalization_stats": _jsonable(payload["normalization_stats"]),
            "meta": {
                "script": "trustmoe_traj.scripts.train_student_integrated_adapter",
                "variant": "v18a_student_integrated_adapter",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cache_path": cache_path.as_posix(),
                "run_name": args.run_name,
                "seed": int(args.seed),
                "best_epoch": int(best_epoch),
                "best_checkpoint": best_path.as_posix(),
            },
            "args": _jsonable(vars(args)),
            "history": _jsonable(history),
            "tensor_shapes": _jsonable(tensor_shapes),
        },
        last_path,
    )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_student_integrated_adapter",
            "variant": "v18a_student_integrated_adapter",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cache_path": cache_path.as_posix(),
            "run_name": args.run_name,
            "seed": int(args.seed),
            "best_epoch": int(best_epoch),
            "best_checkpoint": best_path.as_posix(),
            "last_checkpoint": last_path.as_posix(),
        },
        "args": _jsonable(vars(args)),
        "normalization_stats": _jsonable(payload["normalization_stats"]),
        "tensor_shapes": _jsonable(tensor_shapes),
        "train_items": len(train_indices),
        "val_items": len(val_indices),
        "best_val_metrics": _jsonable(history[best_epoch - 1]["val"] if best_epoch else {}),
        "history": _jsonable(history),
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
