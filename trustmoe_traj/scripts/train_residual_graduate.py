"""Train the first Residual Graduate head from exported prediction caches."""

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

try:  # pragma: no cover - numpy is present in normal project environments.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from trustmoe_traj.evaluation import displacement_errors
from trustmoe_traj.models import (
    ResidualGraduateModel,
    build_residual_graduate_from_cache_shapes,
)


DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "teacher_student_cache"
    / "official_align_eth_train_teacher_student_predictions.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "graduate_models"

REQUIRED_TENSOR_KEYS: Sequence[str] = (
    "student_pred",
    "teacher_pred",
    "ground_truth",
    "agent_mask",
    "past_traj_original_scale",
    "student_best_FDE_pred",
    "teacher_best_FDE_pred",
    "student_FDE_min_per_agent",
    "teacher_FDE_min_per_agent",
    "teacher_advantage_FDE_min",
)


class TeacherStudentCacheDataset(Dataset):
    """Tensor dataset backed by the exported teacher/student cache."""

    def __init__(self, tensors: Mapping[str, torch.Tensor], indices: Sequence[int]) -> None:
        self.tensors = dict(tensors)
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {key: tensor[source_index] for key, tensor in self.tensors.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Residual Graduate head from teacher/student cache tensors.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="residual_graduate_eth_v1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None, help="Optional subset size for smoke runs")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-residual-blocks", type=int, default=4)
    parser.add_argument(
        "--residual-block-type",
        type=str,
        default="mlp",
        choices=["mlp", "film"],
        help=(
            "Internal graduate block type. `mlp` is the original pre-norm "
            "ResidualMLPBlock; `film` conditions each residual block on "
            "student endpoint/displacement and optional social context."
        ),
    )
    parser.add_argument("--block-expansion", type=float, default=2.0)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Deprecated alias for --num-residual-blocks.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=None)
    social_group = parser.add_mutually_exclusive_group()
    social_group.add_argument(
        "--use-social-context",
        dest="use_social_context",
        action="store_true",
        help="Enable V10-A social context encoder over past_traj_original_scale.",
    )
    social_group.add_argument(
        "--no-social-context",
        dest="use_social_context",
        action="store_false",
        help="Disable the V10-A social context encoder.",
    )
    parser.set_defaults(use_social_context=False)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--context-hidden-dim", type=int, default=256)
    parser.add_argument("--context-num-layers", type=int, default=2)
    parser.add_argument("--context-num-heads", type=int, default=2)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    energy_group = parser.add_mutually_exclusive_group()
    energy_group.add_argument(
        "--use-interaction-energy",
        dest="use_interaction_energy",
        action="store_true",
        help="Enable V14 interaction-energy features in the residual graduate head.",
    )
    energy_group.add_argument(
        "--no-interaction-energy",
        dest="use_interaction_energy",
        action="store_false",
        help="Disable V14 interaction-energy features.",
    )
    parser.set_defaults(use_interaction_energy=False)
    parser.add_argument("--interaction-energy-dim", type=int, default=5)
    parser.add_argument("--collision-sigma", type=float, default=0.5)
    parser.add_argument("--collision-radius", type=float, default=0.2)
    parser.add_argument("--no-neighbor-distance", type=float, default=10.0)
    parser.add_argument(
        "--interaction-energy-temporal-stride",
        type=int,
        default=1,
        help="Use every Nth future step when computing interaction energy. The endpoint is always included.",
    )
    energy_head_group = parser.add_mutually_exclusive_group()
    energy_head_group.add_argument(
        "--use-energy-conditioned-heads",
        dest="use_energy_conditioned_heads",
        action="store_true",
        help="Enable V15-light direct energy-conditioned delta/gate heads.",
    )
    energy_head_group.add_argument(
        "--no-energy-conditioned-heads",
        dest="use_energy_conditioned_heads",
        action="store_false",
        help="Disable direct energy-conditioned delta/gate heads.",
    )
    parser.set_defaults(use_energy_conditioned_heads=False)
    parser.add_argument("--energy-condition-dim", type=int, default=32)
    time_gate_group = parser.add_mutually_exclusive_group()
    time_gate_group.add_argument(
        "--use-time-aware-gate",
        dest="use_time_aware_gate",
        action="store_true",
        help="Predict a [B,K,A,T,1] gate instead of a trajectory-level [B,K,A,1,1] gate.",
    )
    time_gate_group.add_argument(
        "--no-time-aware-gate",
        dest="use_time_aware_gate",
        action="store_false",
        help="Use the original trajectory-level gate.",
    )
    parser.set_defaults(use_time_aware_gate=False)
    mode_context_group = parser.add_mutually_exclusive_group()
    mode_context_group.add_argument(
        "--use-mode-set-context",
        dest="use_mode_set_context",
        action="store_true",
        help="Enable V16 candidate-mode self-attention before delta/gate heads.",
    )
    mode_context_group.add_argument(
        "--no-mode-set-context",
        dest="use_mode_set_context",
        action="store_false",
        help="Disable candidate-mode self-attention.",
    )
    parser.set_defaults(use_mode_set_context=False)
    parser.add_argument("--mode-context-num-layers", type=int, default=1)
    parser.add_argument("--mode-context-num-heads", type=int, default=4)
    parser.add_argument("--mode-context-dropout", type=float, default=0.0)
    best_refiner_group = parser.add_mutually_exclusive_group()
    best_refiner_group.add_argument(
        "--use-best-mode-refiner",
        dest="use_best_mode_refiner",
        action="store_true",
        help="Enable V17-A best-mode selector plus endpoint refiner.",
    )
    best_refiner_group.add_argument(
        "--no-best-mode-refiner",
        dest="use_best_mode_refiner",
        action="store_false",
        help="Disable V17-A best-mode selector/refiner.",
    )
    parser.set_defaults(use_best_mode_refiner=False)
    parser.add_argument("--best-refine-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-best-refine",
        type=float,
        default=None,
        help="Optional tanh clamp for the V17-A endpoint refinement vector.",
    )
    temporal_refiner_group = parser.add_mutually_exclusive_group()
    temporal_refiner_group.add_argument(
        "--use-temporal-energy-refiner",
        dest="use_temporal_energy_refiner",
        action="store_true",
        help="Enable V17-B per-timestep interaction-energy guided refiner.",
    )
    temporal_refiner_group.add_argument(
        "--no-temporal-energy-refiner",
        dest="use_temporal_energy_refiner",
        action="store_false",
        help="Disable V17-B per-timestep interaction-energy guided refiner.",
    )
    parser.set_defaults(use_temporal_energy_refiner=False)
    parser.add_argument("--temporal-interaction-energy-dim", type=int, default=5)
    parser.add_argument("--temporal-refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-refine-scale", type=float, default=1.0)
    parser.add_argument("--max-temporal-refine", type=float, default=None)
    parser.add_argument(
        "--temporal-gate-init-bias",
        type=float,
        default=0.0,
        help="Initial bias for the V17-B temporal repair gate before sigmoid.",
    )

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument(
        "--lambda-gt-min",
        type=float,
        default=1.0,
        help="Weight for the original best-of-K GT loss.",
    )
    parser.add_argument(
        "--lambda-rank-gt",
        type=float,
        default=0.0,
        help="Weight for rank-balanced mode-level GT supervision.",
    )
    parser.add_argument(
        "--rank-gt-good-frac",
        type=float,
        default=0.25,
        help="Fraction of student-ranked modes treated as good modes.",
    )
    parser.add_argument(
        "--rank-gt-mid-frac",
        type=float,
        default=0.50,
        help="Fraction of student-ranked modes treated as middle modes. Remaining modes are bad modes.",
    )
    parser.add_argument("--rank-gt-good-weight", type=float, default=1.0)
    parser.add_argument("--rank-gt-mid-weight", type=float, default=1.0)
    parser.add_argument("--rank-gt-bad-weight", type=float, default=1.0)
    parser.add_argument(
        "--lambda-good-nohurt",
        type=float,
        default=0.0,
        help="Weight for no-hurt loss on student-ranked good modes.",
    )
    parser.add_argument(
        "--good-nohurt-frac",
        type=float,
        default=0.25,
        help="Fraction of student-ranked modes protected by good-mode no-hurt loss.",
    )
    parser.add_argument(
        "--good-nohurt-margin",
        type=float,
        default=0.0,
        help="Tolerance before penalizing a good mode that becomes worse than the original student mode.",
    )
    parser.add_argument(
        "--lambda-best-selector",
        type=float,
        default=0.0,
        help="Weight for V17-A BCE supervision of the best-mode selector.",
    )
    parser.add_argument(
        "--best-selector-top-k",
        type=int,
        default=1,
        help="Number of student-ranked modes treated as selector positives.",
    )
    parser.add_argument(
        "--best-selector-positive-weight",
        type=float,
        default=1.0,
        help="Positive-class weight for best-mode selector BCE.",
    )
    parser.add_argument(
        "--lambda-best-refine",
        type=float,
        default=0.0,
        help="Weight for extra GT supervision on student-ranked best modes after V17-A refinement.",
    )
    parser.add_argument(
        "--best-refine-top-k",
        type=int,
        default=1,
        help="Number of student-ranked modes supervised by the best-mode refiner loss.",
    )
    parser.add_argument(
        "--best-refine-ade-weight",
        type=float,
        default=0.25,
        help="ADE weight inside the V17-A best-mode refiner loss.",
    )
    parser.add_argument(
        "--best-refine-fde-weight",
        type=float,
        default=1.0,
        help="FDE weight inside the V17-A best-mode refiner loss.",
    )
    parser.add_argument(
        "--lambda-temporal-gate",
        type=float,
        default=0.0,
        help="Weight for sparsifying V17-B temporal repair gates.",
    )
    parser.add_argument(
        "--lambda-temporal-smoothness",
        type=float,
        default=0.0,
        help="Weight for smoothing adjacent V17-B temporal corrections.",
    )
    parser.add_argument(
        "--lambda-temporal-energy-gate",
        type=float,
        default=0.0,
        help="Weight for discouraging temporal repairs on low-energy timesteps.",
    )
    parser.add_argument(
        "--lambda-temporal-energy-gt",
        type=float,
        default=0.0,
        help="Weight for V17-B2 energy-weighted temporal GT supervision.",
    )
    parser.add_argument(
        "--temporal-energy-gt-top-k",
        type=int,
        default=2,
        help="Number of student-ranked modes supervised by temporal energy GT loss.",
    )
    parser.add_argument(
        "--temporal-energy-gt-risk-floor",
        type=float,
        default=0.05,
        help="Minimum risk weight used by temporal energy GT loss.",
    )
    parser.add_argument(
        "--lambda-diversity-preserve",
        type=float,
        default=0.0,
        help="Weight for preserving student mode spread.",
    )
    parser.add_argument(
        "--diversity-preserve-target-ratio",
        type=float,
        default=0.98,
        help="Minimum graduate/student spread ratio before diversity preservation is penalized.",
    )
    parser.add_argument(
        "--diversity-preserve-margin",
        type=float,
        default=0.0,
        help="Absolute tolerance subtracted from the spread preservation target.",
    )
    parser.add_argument(
        "--diversity-preserve-kind",
        type=str,
        default="endpoint",
        choices=["endpoint", "trajectory", "both"],
        help="Which mode spread to preserve.",
    )
    parser.add_argument(
        "--teacher-distill-mode",
        type=str,
        default="nearest",
        choices=["nearest", "best"],
        help="nearest matches each student mode to its nearest teacher mode; best uses the oracle teacher-best mode.",
    )
    parser.add_argument("--lambda-teacher", type=float, default=0.5)
    parser.add_argument("--lambda-keep", type=float, default=0.2)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    parser.add_argument("--lambda-gate", type=float, default=0.001)
    parser.add_argument("--teacher-margin", type=float, default=0.0)
    parser.add_argument("--max-teacher-weight", type=float, default=5.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device!r}, but torch.cuda.is_available() is False")
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
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid cache payload type: {type(payload)!r}")
    if "tensors" not in payload:
        raise ValueError("Invalid cache payload: missing `tensors`")

    tensors = dict(payload["tensors"])
    missing = [key for key in REQUIRED_TENSOR_KEYS if key not in tensors]
    if missing:
        raise ValueError(f"Cache is missing required tensor(s): {', '.join(missing)}")

    num_items = int(tensors["ground_truth"].shape[0])
    for key in REQUIRED_TENSOR_KEYS:
        if int(tensors[key].shape[0]) != num_items:
            raise ValueError(f"Tensor {key!r} has mismatched first dimension: {tuple(tensors[key].shape)}")
        tensors[key] = tensors[key].detach().cpu()

    return {
        **dict(payload),
        "tensors": tensors,
    }


def _select_indices(num_items: int, *, seed: int, max_items: Optional[int], val_fraction: float) -> tuple[List[int], List[int]]:
    if num_items <= 0:
        raise ValueError("Cache contains no items")
    if max_items is not None and int(max_items) <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    generator = torch.Generator().manual_seed(int(seed))
    shuffled = torch.randperm(num_items, generator=generator).tolist()
    if max_items is not None:
        shuffled = shuffled[: min(int(max_items), len(shuffled))]

    if len(shuffled) <= 1 or float(val_fraction) == 0.0:
        return shuffled, shuffled

    val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _move_batch(batch: Mapping[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, dtype=torch.float32) if value.dtype.is_floating_point else value.to(device=device)
        for key, value in batch.items()
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum().item()) <= 0:
        raise ValueError("Cannot average over an empty valid-agent mask")
    return values[valid].mean()


def _weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_weights = weights.to(dtype=values.dtype) * mask.to(dtype=values.dtype)
    denom = valid_weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return values.new_tensor(0.0)
    return (values * valid_weights).sum() / denom.clamp_min(1e-8)


def _weighted_mode_agent_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape [B, K, A], got {tuple(values.shape)}")
    if weights.shape != mask.shape:
        raise ValueError(f"weights/mask shape mismatch: weights={tuple(weights.shape)}, mask={tuple(mask.shape)}")
    if values.shape[0] != weights.shape[0] or values.shape[2] != weights.shape[1]:
        raise ValueError(
            f"values/weights shape mismatch: values={tuple(values.shape)}, weights={tuple(weights.shape)}"
        )

    valid_weights = weights[:, None, :].to(dtype=values.dtype) * mask[:, None, :].to(dtype=values.dtype)
    valid_weights = valid_weights.expand_as(values)
    denom = valid_weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return values.new_tensor(0.0)
    return (values * valid_weights).sum() / denom.clamp_min(1e-8)


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    if pairwise.ndim != 3:
        raise ValueError(f"pairwise must have shape [N, K, K], got {tuple(pairwise.shape)}")
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return torch.zeros((pairwise.shape[0],), dtype=pairwise.dtype, device=pairwise.device)
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    pred = _ensure_prediction5(prediction, name="prediction")
    batch_size, num_modes, num_agents, _pred_len, coord_dim = pred.shape
    endpoints = pred[..., -1, :].permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_modes, coord_dim)
    pairwise = torch.cdist(endpoints, endpoints, p=2)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _trajectory_spread(prediction: torch.Tensor) -> torch.Tensor:
    pred = _ensure_prediction5(prediction, name="prediction")
    batch_size, num_modes, num_agents, pred_len, coord_dim = pred.shape
    traj = pred.permute(0, 2, 1, 3, 4).reshape(batch_size * num_agents, num_modes, pred_len, coord_dim)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _diversity_preserve_loss(
    graduate_pred: torch.Tensor,
    student_pred: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    kind: str,
    target_ratio: float,
    margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if kind not in {"endpoint", "trajectory", "both"}:
        raise ValueError(f"Unsupported diversity_preserve_kind: {kind!r}")
    if float(target_ratio) < 0.0:
        raise ValueError(f"diversity_preserve_target_ratio must be non-negative, got {target_ratio}")
    if float(margin) < 0.0:
        raise ValueError(f"diversity_preserve_margin must be non-negative, got {margin}")

    kind_names = ("endpoint", "trajectory") if kind == "both" else (kind,)
    zero = _ensure_prediction5(graduate_pred, name="graduate_pred").new_tensor(0.0)
    total = zero
    components: Dict[str, float] = {}

    for name in kind_names:
        spread_fn = _endpoint_spread if name == "endpoint" else _trajectory_spread
        graduate_spread = spread_fn(graduate_pred)
        with torch.no_grad():
            student_spread = spread_fn(student_pred)
            target_spread = float(target_ratio) * student_spread
        penalty = (target_spread - graduate_spread - float(margin)).clamp_min(0.0)
        loss = _masked_mean(penalty, agent_mask)
        total = total + loss

        valid = agent_mask.bool()
        if bool(valid.any().detach().cpu()):
            student_valid = student_spread[valid]
            graduate_valid = graduate_spread[valid]
            ratio = graduate_valid / student_valid.abs().clamp_min(1e-8)
            shortfall = penalty[valid]
            shrink_rate = (ratio < float(target_ratio)).to(dtype=graduate_spread.dtype).mean()
            components.update(
                {
                    f"loss_diversity_preserve_{name}": float(loss.detach().cpu()),
                    f"diversity_preserve_{name}_student_spread": float(student_valid.mean().detach().cpu()),
                    f"diversity_preserve_{name}_graduate_spread": float(graduate_valid.mean().detach().cpu()),
                    f"diversity_preserve_{name}_ratio": float(ratio.mean().detach().cpu()),
                    f"diversity_preserve_{name}_shortfall": float(shortfall.mean().detach().cpu()),
                    f"diversity_preserve_{name}_shrink_rate": float(shrink_rate.detach().cpu()),
                }
            )
        else:
            components.update(
                {
                    f"loss_diversity_preserve_{name}": 0.0,
                    f"diversity_preserve_{name}_student_spread": 0.0,
                    f"diversity_preserve_{name}_graduate_spread": 0.0,
                    f"diversity_preserve_{name}_ratio": 0.0,
                    f"diversity_preserve_{name}_shortfall": 0.0,
                    f"diversity_preserve_{name}_shrink_rate": 0.0,
                }
            )

    total = total / float(len(kind_names))
    components["loss_diversity_preserve"] = float(total.detach().cpu())
    components["diversity_preserve_target_ratio"] = float(target_ratio)
    components["diversity_preserve_margin"] = float(margin)
    return total, components


def _rank_group_counts(num_modes: int, *, good_frac: float, mid_frac: float) -> tuple[int, int, int]:
    if int(num_modes) <= 0:
        raise ValueError(f"num_modes must be positive, got {num_modes}")
    if not 0.0 <= float(good_frac) <= 1.0:
        raise ValueError(f"rank_gt_good_frac must be in [0, 1], got {good_frac}")
    if not 0.0 <= float(mid_frac) <= 1.0:
        raise ValueError(f"rank_gt_mid_frac must be in [0, 1], got {mid_frac}")
    if float(good_frac) + float(mid_frac) > 1.0 + 1e-8:
        raise ValueError(
            "rank_gt_good_frac + rank_gt_mid_frac must be <= 1, "
            f"got {float(good_frac) + float(mid_frac):.6f}"
        )

    good_count = int(round(int(num_modes) * float(good_frac)))
    mid_count = int(round(int(num_modes) * float(mid_frac)))
    if float(good_frac) > 0.0:
        good_count = max(1, good_count)
    if float(mid_frac) > 0.0:
        mid_count = max(1, mid_count)
    if good_count + mid_count > int(num_modes):
        overflow = good_count + mid_count - int(num_modes)
        mid_count = max(0, mid_count - overflow)
    bad_count = int(num_modes) - good_count - mid_count
    return good_count, mid_count, bad_count


def _rank_balanced_gt_loss(
    graduate_pred: torch.Tensor,
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
    good_frac: float,
    mid_frac: float,
    good_weight: float,
    mid_weight: float,
    bad_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    for name, weight in (
        ("rank_gt_good_weight", good_weight),
        ("rank_gt_mid_weight", mid_weight),
        ("rank_gt_bad_weight", bad_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {weight}")

    graduate_score = _mode_agent_distances(graduate_pred, ground_truth, fde_weight=float(fde_weight))
    with torch.no_grad():
        student_score = _mode_agent_distances(student_pred, ground_truth, fde_weight=float(fde_weight))
        rank_index = student_score.argsort(dim=1)

    sorted_graduate_score = graduate_score.gather(dim=1, index=rank_index)
    num_modes = int(sorted_graduate_score.shape[1])
    good_count, mid_count, bad_count = _rank_group_counts(
        num_modes,
        good_frac=float(good_frac),
        mid_frac=float(mid_frac),
    )

    zero = sorted_graduate_score.new_tensor(0.0)
    total = zero
    active_weight = 0.0
    components: Dict[str, float] = {
        "rank_gt_good_count": float(good_count),
        "rank_gt_mid_count": float(mid_count),
        "rank_gt_bad_count": float(bad_count),
    }

    group_specs = (
        ("good", 0, good_count, float(good_weight)),
        ("mid", good_count, good_count + mid_count, float(mid_weight)),
        ("bad", good_count + mid_count, num_modes, float(bad_weight)),
    )
    for name, start, end, weight in group_specs:
        if end <= start or weight == 0.0:
            group_loss = zero
        else:
            group_values = sorted_graduate_score[:, start:end, :].mean(dim=1)
            group_loss = _masked_mean(group_values, agent_mask)
            total = total + weight * group_loss
            active_weight += abs(weight)
        components[f"loss_rank_gt_{name}"] = float(group_loss.detach().cpu())

    if active_weight <= 0.0:
        total = zero
    else:
        total = total / float(active_weight)
    return total, components


def _good_mode_nohurt_loss(
    graduate_pred: torch.Tensor,
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
    good_frac: float,
    margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if not 0.0 <= float(good_frac) <= 1.0:
        raise ValueError(f"good_nohurt_frac must be in [0, 1], got {good_frac}")

    graduate_score = _mode_agent_distances(graduate_pred, ground_truth, fde_weight=float(fde_weight))
    with torch.no_grad():
        student_score = _mode_agent_distances(student_pred, ground_truth, fde_weight=float(fde_weight))
        rank_index = student_score.argsort(dim=1)

    num_modes = int(graduate_score.shape[1])
    good_count, _mid_count, _bad_count = _rank_group_counts(
        num_modes,
        good_frac=float(good_frac),
        mid_frac=0.0,
    )
    zero = graduate_score.new_tensor(0.0)
    components: Dict[str, float] = {"good_nohurt_count": float(good_count)}
    if good_count <= 0:
        components.update(
            {
                "loss_good_nohurt": 0.0,
                "good_nohurt_worse_rate": 0.0,
                "good_nohurt_student_score": 0.0,
                "good_nohurt_graduate_score": 0.0,
            }
        )
        return zero, components

    good_index = rank_index[:, :good_count, :]
    good_graduate_score = graduate_score.gather(dim=1, index=good_index)
    good_student_score = student_score.gather(dim=1, index=good_index).detach()
    score_delta = good_graduate_score - good_student_score
    penalty = (score_delta - float(margin)).clamp_min(0.0)
    loss = _weighted_mode_agent_mean(
        penalty,
        torch.ones_like(agent_mask, dtype=graduate_score.dtype),
        agent_mask,
    )

    valid = agent_mask[:, None, :].expand_as(score_delta)
    if bool(valid.any().detach().cpu()):
        worse_rate = (score_delta[valid] > float(margin)).to(dtype=graduate_score.dtype).mean()
        student_mean = good_student_score[valid].mean()
        graduate_mean = good_graduate_score[valid].mean()
    else:
        worse_rate = zero
        student_mean = zero
        graduate_mean = zero

    components.update(
        {
            "loss_good_nohurt": float(loss.detach().cpu()),
            "good_nohurt_worse_rate": float(worse_rate.detach().cpu()),
            "good_nohurt_student_score": float(student_mean.detach().cpu()),
            "good_nohurt_graduate_score": float(graduate_mean.detach().cpu()),
        }
    )
    return loss, components


def _normalized_teacher_weights(
    teacher_advantage: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    margin: float,
    max_weight: float,
) -> torch.Tensor:
    positive = (teacher_advantage - float(margin)).clamp_min(0.0)
    valid_positive = positive[agent_mask.bool()]
    if int((valid_positive > 0).sum().item()) <= 0:
        return torch.zeros_like(positive)
    mean_positive = valid_positive[valid_positive > 0].mean().clamp_min(1e-8)
    return (positive / mean_positive).clamp(max=float(max_weight))


def _ensure_prediction5(prediction: torch.Tensor, *, name: str) -> torch.Tensor:
    if prediction.ndim == 4:
        return prediction.unsqueeze(1)
    if prediction.ndim == 5:
        return prediction
    raise ValueError(f"{name} must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(prediction.shape)}")


def _mode_agent_distances(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    pred = _ensure_prediction5(prediction, name="prediction")
    if target.ndim == 4:
        target = target.unsqueeze(1).expand_as(pred)
    elif target.ndim == 5:
        target = target
    else:
        raise ValueError(f"target must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(target.shape)}")
    if target.shape != pred.shape:
        raise ValueError(f"prediction/target shape mismatch: prediction={tuple(pred.shape)}, target={tuple(target.shape)}")

    distances = torch.linalg.norm(pred - target, dim=-1)
    ade = distances.mean(dim=-1)
    fde = distances[..., -1]
    return ade + float(fde_weight) * fde


def _student_rank_index(
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    fde_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_score = _mode_agent_distances(student_pred, ground_truth, fde_weight=float(fde_weight))
    return student_score, student_score.argsort(dim=1)


def _clamped_top_k(top_k: int, num_modes: int, *, name: str) -> int:
    if int(top_k) <= 0:
        raise ValueError(f"{name} must be positive, got {top_k}")
    return min(int(top_k), int(num_modes))


def _best_selector_loss(
    selector_logits: torch.Tensor,
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    fde_weight: float,
    top_k: int,
    positive_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if selector_logits.ndim != 3:
        raise ValueError(f"selector_logits must have shape [B, K, A], got {tuple(selector_logits.shape)}")
    if float(positive_weight) <= 0.0:
        raise ValueError(f"best_selector_positive_weight must be positive, got {positive_weight}")

    with torch.no_grad():
        _student_score, rank_index = _student_rank_index(
            student_pred,
            ground_truth,
            fde_weight=float(fde_weight),
        )
        num_modes = int(selector_logits.shape[1])
        keep_k = _clamped_top_k(top_k, num_modes, name="best_selector_top_k")
        positive_index = rank_index[:, :keep_k, :]
        targets = torch.zeros_like(selector_logits)
        targets.scatter_(dim=1, index=positive_index, value=1.0)

    valid = agent_mask[:, None, :].expand_as(selector_logits).bool()
    if not bool(valid.any().detach().cpu()):
        zero = selector_logits.new_tensor(0.0)
        return zero, {
            "loss_best_selector": 0.0,
            "best_selector_top1_acc": 0.0,
            "best_selector_positive_prob": 0.0,
            "best_selector_negative_prob": 0.0,
        }

    pos_weight = selector_logits.new_tensor(float(positive_weight))
    per_mode = F.binary_cross_entropy_with_logits(
        selector_logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    loss = per_mode[valid].mean()
    selector_prob = torch.sigmoid(selector_logits.detach())
    top1_pred = selector_logits.detach().argmax(dim=1)
    top1_target = rank_index[:, 0, :]
    top1_acc = (top1_pred[agent_mask.bool()] == top1_target[agent_mask.bool()]).to(dtype=selector_logits.dtype).mean()

    positive_mask = targets.bool() & valid
    negative_mask = (~targets.bool()) & valid
    positive_prob = selector_prob[positive_mask].mean() if bool(positive_mask.any().detach().cpu()) else loss.new_tensor(0.0)
    negative_prob = selector_prob[negative_mask].mean() if bool(negative_mask.any().detach().cpu()) else loss.new_tensor(0.0)
    return loss, {
        "loss_best_selector": float(loss.detach().cpu()),
        "best_selector_top_k": float(keep_k),
        "best_selector_top1_acc": float(top1_acc.detach().cpu()),
        "best_selector_positive_prob": float(positive_prob.detach().cpu()),
        "best_selector_negative_prob": float(negative_prob.detach().cpu()),
    }


def _best_refine_loss(
    graduate_pred: torch.Tensor,
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    rank_fde_weight: float,
    top_k: int,
    ade_weight: float,
    fde_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if float(ade_weight) < 0.0:
        raise ValueError(f"best_refine_ade_weight must be non-negative, got {ade_weight}")
    if float(fde_weight) < 0.0:
        raise ValueError(f"best_refine_fde_weight must be non-negative, got {fde_weight}")

    grad_errors = displacement_errors(graduate_pred, ground_truth, agent_mask=agent_mask)
    graduate_score = (
        float(ade_weight) * grad_errors["ade_per_mode_agent"]
        + float(fde_weight) * grad_errors["fde_per_mode_agent"]
    )
    with torch.no_grad():
        _rank_score, rank_index = _student_rank_index(
            student_pred,
            ground_truth,
            fde_weight=float(rank_fde_weight),
        )
        student_errors = displacement_errors(student_pred, ground_truth, agent_mask=agent_mask)
        student_refine_score = (
            float(ade_weight) * student_errors["ade_per_mode_agent"]
            + float(fde_weight) * student_errors["fde_per_mode_agent"]
        )
        num_modes = int(graduate_score.shape[1])
        keep_k = _clamped_top_k(top_k, num_modes, name="best_refine_top_k")
        selected_index = rank_index[:, :keep_k, :]
        selected_student_score = student_refine_score.gather(dim=1, index=selected_index)

    selected_graduate_score = graduate_score.gather(dim=1, index=selected_index)
    selected_loss = selected_graduate_score.mean(dim=1)
    loss = _masked_mean(selected_loss, agent_mask)

    valid = agent_mask[:, None, :].expand_as(selected_graduate_score).bool()
    if bool(valid.any().detach().cpu()):
        graduate_mean = selected_graduate_score[valid].mean()
        student_mean = selected_student_score[valid].mean()
        improve_rate = (selected_graduate_score[valid] < selected_student_score[valid]).to(dtype=graduate_score.dtype).mean()
    else:
        graduate_mean = loss.new_tensor(0.0)
        student_mean = loss.new_tensor(0.0)
        improve_rate = loss.new_tensor(0.0)

    return loss, {
        "loss_best_refine": float(loss.detach().cpu()),
        "best_refine_top_k": float(keep_k),
        "best_refine_student_score": float(student_mean.detach().cpu()),
        "best_refine_graduate_score": float(graduate_mean.detach().cpu()),
        "best_refine_improve_rate": float(improve_rate.detach().cpu()),
    }


def _temporal_low_energy_gate_loss(
    temporal_gate: torch.Tensor,
    temporal_energy_features: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if temporal_gate.ndim != 5 or int(temporal_gate.shape[-1]) != 1:
        raise ValueError(f"temporal_gate must have shape [B, K, A, T, 1], got {tuple(temporal_gate.shape)}")
    if temporal_energy_features.ndim != 5 or int(temporal_energy_features.shape[-1]) < 5:
        raise ValueError(
            "temporal_energy_features must have shape [B, K, A, T, C>=5], "
            f"got {tuple(temporal_energy_features.shape)}"
        )
    if tuple(temporal_energy_features.shape[:4]) != tuple(temporal_gate.shape[:4]):
        raise ValueError(
            "temporal gate/energy shape mismatch: "
            f"gate={tuple(temporal_gate.shape)}, energy={tuple(temporal_energy_features.shape)}"
        )

    risk = _temporal_energy_risk(temporal_energy_features)
    low_energy = 1.0 - risk
    valid = agent_mask[:, None, :, None, None].expand_as(temporal_gate).bool()
    if not bool(valid.any().detach().cpu()):
        zero = temporal_gate.new_tensor(0.0)
        return zero, {
            "loss_temporal_energy_gate": 0.0,
            "temporal_energy_risk_mean": 0.0,
            "temporal_low_energy_gate_mean": 0.0,
        }

    penalty = temporal_gate * low_energy
    loss = penalty[valid].mean()
    return loss, {
        "loss_temporal_energy_gate": float(loss.detach().cpu()),
        "temporal_energy_risk_mean": float(risk[valid].mean().detach().cpu()),
        "temporal_low_energy_gate_mean": float(penalty[valid].mean().detach().cpu()),
    }


def _temporal_energy_risk(temporal_energy_features: torch.Tensor) -> torch.Tensor:
    if temporal_energy_features.ndim != 5 or int(temporal_energy_features.shape[-1]) < 5:
        raise ValueError(
            "temporal_energy_features must have shape [B, K, A, T, C>=5], "
            f"got {tuple(temporal_energy_features.shape)}"
        )
    energy = temporal_energy_features.detach()
    min_distance = energy[..., 0:1].clamp_min(1e-3)
    crowding = energy[..., 1:2] + energy[..., 2:3] + energy[..., 3:4] + energy[..., 4:5]
    return (crowding / (crowding + min_distance)).clamp(0.0, 1.0)


def _temporal_energy_gt_loss(
    temporal_refine_delta: torch.Tensor,
    temporal_energy_features: torch.Tensor,
    student_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    rank_fde_weight: float,
    top_k: int,
    risk_floor: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    student = _ensure_prediction5(student_pred, name="student_pred")
    if tuple(temporal_refine_delta.shape) != tuple(student.shape):
        raise ValueError(
            "temporal_refine_delta/student shape mismatch: "
            f"temporal_refine_delta={tuple(temporal_refine_delta.shape)}, student={tuple(student.shape)}"
        )
    if ground_truth.ndim != 4:
        raise ValueError(f"ground_truth must have shape [B, A, T, 2], got {tuple(ground_truth.shape)}")
    expected_gt_shape = (int(student.shape[0]), int(student.shape[2]), int(student.shape[3]), int(student.shape[4]))
    if tuple(ground_truth.shape) != expected_gt_shape:
        raise ValueError(f"student/ground_truth shape mismatch: student={tuple(student.shape)}, gt={tuple(ground_truth.shape)}")
    if float(risk_floor) < 0.0:
        raise ValueError(f"temporal_energy_gt_risk_floor must be non-negative, got {risk_floor}")

    with torch.no_grad():
        _student_score, rank_index = _student_rank_index(
            student,
            ground_truth,
            fde_weight=float(rank_fde_weight),
        )
        num_modes = int(student.shape[1])
        keep_k = _clamped_top_k(top_k, num_modes, name="temporal_energy_gt_top_k")
        selected_index = rank_index[:, :keep_k, :]
        risk = _temporal_energy_risk(temporal_energy_features).squeeze(-1)
        risk = risk.clamp_min(float(risk_floor))
        risk = risk / risk.detach().amax(dim=3, keepdim=True).clamp_min(1e-6)

    temporal_pred = student + temporal_refine_delta
    error = torch.linalg.norm(temporal_pred - ground_truth[:, None, :, :, :], dim=-1)
    selected_error = error.gather(
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(student.shape[3])),
    )
    selected_risk = risk.gather(
        dim=1,
        index=selected_index[:, :, :, None].expand(-1, -1, -1, int(student.shape[3])),
    )
    valid = agent_mask[:, None, :, None].expand_as(selected_error).to(dtype=selected_error.dtype)
    weights = selected_risk.to(dtype=selected_error.dtype) * valid
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        zero = temporal_refine_delta.new_tensor(0.0)
        return zero, {
            "loss_temporal_energy_gt": 0.0,
            "temporal_energy_gt_top_k": float(keep_k),
            "temporal_energy_gt_risk_mean": 0.0,
        }
    loss = (selected_error * weights).sum() / denom.clamp_min(1e-8)
    return loss, {
        "loss_temporal_energy_gt": float(loss.detach().cpu()),
        "temporal_energy_gt_top_k": float(keep_k),
        "temporal_energy_gt_risk_mean": float(selected_risk[valid.bool()].mean().detach().cpu()),
    }


def _nearest_teacher_mode_targets(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    student = _ensure_prediction5(student_pred, name="student_pred")
    teacher = _ensure_prediction5(teacher_pred, name="teacher_pred")
    if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
        raise ValueError(f"student/teacher shape mismatch: student={tuple(student.shape)}, teacher={tuple(teacher.shape)}")

    batch_size, num_student_modes, num_agents, pred_len, coord_dim = student.shape
    student_by_agent = student.permute(0, 2, 1, 3, 4)  # [B, A, Ks, T, 2]
    teacher_by_agent = teacher.permute(0, 2, 1, 3, 4)  # [B, A, Kt, T, 2]

    pairwise = student_by_agent[:, :, :, None, :, :] - teacher_by_agent[:, :, None, :, :, :]
    pairwise_distance = torch.linalg.norm(pairwise, dim=-1)  # [B, A, Ks, Kt, T]
    pairwise_score = pairwise_distance.mean(dim=-1) + float(fde_weight) * pairwise_distance[..., -1]
    nearest_teacher_index = pairwise_score.argmin(dim=-1)  # [B, A, Ks]

    gather_index = nearest_teacher_index[..., None, None].expand(
        batch_size,
        num_agents,
        num_student_modes,
        pred_len,
        coord_dim,
    )
    matched_by_agent = teacher_by_agent.gather(dim=2, index=gather_index)
    return matched_by_agent.permute(0, 2, 1, 3, 4).contiguous()


def _loss_components(
    model: ResidualGraduateModel,
    batch: Mapping[str, torch.Tensor],
    *,
    fde_weight: float,
    lambda_gt_min: float,
    lambda_rank_gt: float,
    rank_gt_good_frac: float,
    rank_gt_mid_frac: float,
    rank_gt_good_weight: float,
    rank_gt_mid_weight: float,
    rank_gt_bad_weight: float,
    lambda_good_nohurt: float,
    good_nohurt_frac: float,
    good_nohurt_margin: float,
    lambda_best_selector: float,
    best_selector_top_k: int,
    best_selector_positive_weight: float,
    lambda_best_refine: float,
    best_refine_top_k: int,
    best_refine_ade_weight: float,
    best_refine_fde_weight: float,
    lambda_temporal_gate: float,
    lambda_temporal_smoothness: float,
    lambda_temporal_energy_gate: float,
    lambda_temporal_energy_gt: float,
    temporal_energy_gt_top_k: int,
    temporal_energy_gt_risk_floor: float,
    lambda_diversity_preserve: float,
    diversity_preserve_target_ratio: float,
    diversity_preserve_margin: float,
    diversity_preserve_kind: str,
    teacher_distill_mode: str,
    lambda_teacher: float,
    lambda_keep: float,
    lambda_residual: float,
    lambda_gate: float,
    teacher_margin: float,
    max_teacher_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    output = model(
        batch["student_pred"],
        batch["past_traj_original_scale"],
        agent_mask=batch.get("agent_mask"),
        interaction_energy_features=batch.get("interaction_energy_features"),
        temporal_interaction_energy_features=batch.get("temporal_interaction_energy_features"),
    )
    graduate_pred = output["graduate_pred"]
    agent_mask = batch["agent_mask"].bool()

    grad_errors = displacement_errors(graduate_pred, batch["ground_truth"], agent_mask=agent_mask)
    grad_ade_min = grad_errors["ade_per_mode_agent"].min(dim=1).values
    grad_fde_min = grad_errors["fde_per_mode_agent"].min(dim=1).values
    loss_gt_ade = _masked_mean(grad_ade_min, agent_mask)
    loss_gt_fde = _masked_mean(grad_fde_min, agent_mask)
    loss_gt = loss_gt_ade + float(fde_weight) * loss_gt_fde
    loss_rank_gt, rank_gt_components = _rank_balanced_gt_loss(
        graduate_pred,
        batch["student_pred"],
        batch["ground_truth"],
        agent_mask,
        fde_weight=float(fde_weight),
        good_frac=float(rank_gt_good_frac),
        mid_frac=float(rank_gt_mid_frac),
        good_weight=float(rank_gt_good_weight),
        mid_weight=float(rank_gt_mid_weight),
        bad_weight=float(rank_gt_bad_weight),
    )
    loss_good_nohurt, good_nohurt_components = _good_mode_nohurt_loss(
        graduate_pred,
        batch["student_pred"],
        batch["ground_truth"],
        agent_mask,
        fde_weight=float(fde_weight),
        good_frac=float(good_nohurt_frac),
        margin=float(good_nohurt_margin),
    )
    if float(lambda_best_selector) > 0.0:
        if "best_mode_selector_logits" not in output:
            raise ValueError("lambda_best_selector > 0 requires --use-best-mode-refiner")
        loss_best_selector, best_selector_components = _best_selector_loss(
            output["best_mode_selector_logits"],
            batch["student_pred"],
            batch["ground_truth"],
            agent_mask,
            fde_weight=float(fde_weight),
            top_k=int(best_selector_top_k),
            positive_weight=float(best_selector_positive_weight),
        )
    else:
        loss_best_selector = graduate_pred.new_tensor(0.0)
        best_selector_components = {
            "loss_best_selector": 0.0,
            "best_selector_top_k": float(best_selector_top_k),
        }
    if float(lambda_best_refine) > 0.0:
        loss_best_refine, best_refine_components = _best_refine_loss(
            graduate_pred,
            batch["student_pred"],
            batch["ground_truth"],
            agent_mask,
            rank_fde_weight=float(fde_weight),
            top_k=int(best_refine_top_k),
            ade_weight=float(best_refine_ade_weight),
            fde_weight=float(best_refine_fde_weight),
        )
    else:
        loss_best_refine = graduate_pred.new_tensor(0.0)
        best_refine_components = {
            "loss_best_refine": 0.0,
            "best_refine_top_k": float(best_refine_top_k),
        }
    if "temporal_repair_gate" in output:
        temporal_gate = output["temporal_repair_gate"]
        loss_temporal_gate = temporal_gate.mean()
    else:
        temporal_gate = None
        loss_temporal_gate = graduate_pred.new_tensor(0.0)
    if "temporal_refine_delta" in output and int(output["temporal_refine_delta"].shape[3]) > 1:
        temporal_step_delta = output["temporal_refine_delta"][:, :, :, 1:, :] - output["temporal_refine_delta"][:, :, :, :-1, :]
        loss_temporal_smoothness = temporal_step_delta.pow(2).mean()
    else:
        loss_temporal_smoothness = graduate_pred.new_tensor(0.0)
    if (
        float(lambda_temporal_energy_gate) > 0.0
        and temporal_gate is not None
        and "temporal_interaction_energy_features" in output
    ):
        loss_temporal_energy_gate, temporal_energy_components = _temporal_low_energy_gate_loss(
            temporal_gate,
            output["temporal_interaction_energy_features"],
            agent_mask,
        )
    else:
        loss_temporal_energy_gate = graduate_pred.new_tensor(0.0)
        temporal_energy_components = {
            "loss_temporal_energy_gate": 0.0,
        }
    if (
        float(lambda_temporal_energy_gt) > 0.0
        and "temporal_refine_delta" in output
        and "temporal_interaction_energy_features" in output
    ):
        loss_temporal_energy_gt, temporal_gt_components = _temporal_energy_gt_loss(
            output["temporal_refine_delta"],
            output["temporal_interaction_energy_features"],
            batch["student_pred"],
            batch["ground_truth"],
            agent_mask,
            rank_fde_weight=float(fde_weight),
            top_k=int(temporal_energy_gt_top_k),
            risk_floor=float(temporal_energy_gt_risk_floor),
        )
    else:
        loss_temporal_energy_gt = graduate_pred.new_tensor(0.0)
        temporal_gt_components = {
            "loss_temporal_energy_gt": 0.0,
            "temporal_energy_gt_top_k": float(temporal_energy_gt_top_k),
        }
    if float(lambda_diversity_preserve) > 0.0:
        loss_diversity_preserve, diversity_preserve_components = _diversity_preserve_loss(
            graduate_pred,
            batch["student_pred"],
            agent_mask,
            kind=str(diversity_preserve_kind),
            target_ratio=float(diversity_preserve_target_ratio),
            margin=float(diversity_preserve_margin),
        )
    else:
        loss_diversity_preserve = graduate_pred.new_tensor(0.0)
        diversity_preserve_components = {
            "loss_diversity_preserve": 0.0,
            "diversity_preserve_target_ratio": float(diversity_preserve_target_ratio),
            "diversity_preserve_margin": float(diversity_preserve_margin),
        }

    teacher_weights = _normalized_teacher_weights(
        batch["teacher_advantage_FDE_min"],
        agent_mask,
        margin=float(teacher_margin),
        max_weight=float(max_teacher_weight),
    )
    if teacher_distill_mode == "nearest":
        teacher_target = _nearest_teacher_mode_targets(
            batch["student_pred"],
            batch["teacher_pred"],
            fde_weight=float(fde_weight),
        )
    elif teacher_distill_mode == "best":
        teacher_target = batch["teacher_best_FDE_pred"]
    else:
        raise ValueError(f"Unsupported teacher_distill_mode: {teacher_distill_mode!r}")
    teacher_distance = _mode_agent_distances(graduate_pred, teacher_target, fde_weight=float(fde_weight))
    loss_teacher = _weighted_mode_agent_mean(teacher_distance, teacher_weights, agent_mask)

    keep_distance = _mode_agent_distances(graduate_pred, batch["student_pred"], fde_weight=float(fde_weight))
    keep_weights = (batch["teacher_advantage_FDE_min"] <= float(teacher_margin)).to(dtype=graduate_pred.dtype)
    loss_keep = _weighted_mode_agent_mean(keep_distance, keep_weights, agent_mask)

    loss_residual = output["delta_pred"].pow(2).mean()
    if "best_mode_refine_delta" in output:
        loss_residual = loss_residual + output["best_mode_refine_delta"].pow(2).mean()
    if "temporal_refine_delta" in output:
        loss_residual = loss_residual + output["temporal_refine_delta"].pow(2).mean()
    loss_gate = output["gate"].mean()
    total = (
        float(lambda_gt_min) * loss_gt
        + float(lambda_rank_gt) * loss_rank_gt
        + float(lambda_good_nohurt) * loss_good_nohurt
        + float(lambda_best_selector) * loss_best_selector
        + float(lambda_best_refine) * loss_best_refine
        + float(lambda_temporal_gate) * loss_temporal_gate
        + float(lambda_temporal_smoothness) * loss_temporal_smoothness
        + float(lambda_temporal_energy_gate) * loss_temporal_energy_gate
        + float(lambda_temporal_energy_gt) * loss_temporal_energy_gt
        + float(lambda_diversity_preserve) * loss_diversity_preserve
        + float(lambda_teacher) * loss_teacher
        + float(lambda_keep) * loss_keep
        + float(lambda_residual) * loss_residual
        + float(lambda_gate) * loss_gate
    )

    components = {
        "loss_total": float(total.detach().cpu()),
        "loss_gt": float(loss_gt.detach().cpu()),
        "loss_gt_ADE_min": float(loss_gt_ade.detach().cpu()),
        "loss_gt_FDE_min": float(loss_gt_fde.detach().cpu()),
        "loss_rank_gt": float(loss_rank_gt.detach().cpu()),
        "loss_good_nohurt": float(loss_good_nohurt.detach().cpu()),
        "loss_best_selector": float(loss_best_selector.detach().cpu()),
        "loss_best_refine": float(loss_best_refine.detach().cpu()),
        "loss_temporal_gate": float(loss_temporal_gate.detach().cpu()),
        "loss_temporal_smoothness": float(loss_temporal_smoothness.detach().cpu()),
        "loss_temporal_energy_gate": float(loss_temporal_energy_gate.detach().cpu()),
        "loss_temporal_energy_gt": float(loss_temporal_energy_gt.detach().cpu()),
        "loss_diversity_preserve": float(loss_diversity_preserve.detach().cpu()),
        "loss_teacher": float(loss_teacher.detach().cpu()),
        "loss_keep": float(loss_keep.detach().cpu()),
        "loss_residual": float(loss_residual.detach().cpu()),
        "loss_gate": float(loss_gate.detach().cpu()),
        "teacher_weight_mean": float(teacher_weights[agent_mask].mean().detach().cpu()),
        "gate_mean": float(output["gate"].detach().mean().cpu()),
        "delta_l2_mean": float(output["delta_pred"].detach().pow(2).mean().sqrt().cpu()),
    }
    if "interaction_energy_features" in output:
        components["interaction_energy_feature_mean"] = float(
            output["interaction_energy_features"].detach().mean().cpu()
        )
    if "energy_delta_pred" in output:
        components["energy_delta_l2_mean"] = float(output["energy_delta_pred"].detach().pow(2).mean().sqrt().cpu())
    if "best_mode_selector_prob" in output:
        components["best_selector_prob_mean"] = float(output["best_mode_selector_prob"].detach().mean().cpu())
    if "best_mode_refine_delta" in output:
        components["best_refine_delta_l2_mean"] = float(
            output["best_mode_refine_delta"].detach().pow(2).mean().sqrt().cpu()
        )
    if "temporal_repair_gate" in output:
        gate_detached = output["temporal_repair_gate"].detach()
        components["temporal_gate_mean"] = float(gate_detached.mean().cpu())
        pred_len = int(gate_detached.shape[3])
        first = max(pred_len // 3, 1)
        second = max((2 * pred_len) // 3, first + 1)
        components["temporal_gate_early_mean"] = float(gate_detached[:, :, :, :first, :].mean().cpu())
        components["temporal_gate_mid_mean"] = float(gate_detached[:, :, :, first:second, :].mean().cpu())
        components["temporal_gate_late_mean"] = float(gate_detached[:, :, :, second:, :].mean().cpu())
    if "temporal_refine_delta" in output:
        components["temporal_refine_delta_l2_mean"] = float(
            output["temporal_refine_delta"].detach().pow(2).mean().sqrt().cpu()
        )
    components.update(rank_gt_components)
    components.update(good_nohurt_components)
    components.update(best_selector_components)
    components.update(best_refine_components)
    components.update(temporal_energy_components)
    components.update(temporal_gt_components)
    components.update(diversity_preserve_components)
    return total, components


def _accumulate_mean(running: Dict[str, float], values: Mapping[str, float], *, weight: int) -> None:
    running["_weight"] = running.get("_weight", 0.0) + float(weight)
    for key, value in values.items():
        running[key] = running.get(key, 0.0) + float(value) * float(weight)


def _finalize_mean(running: Mapping[str, float]) -> Dict[str, float]:
    weight = float(running.get("_weight", 0.0))
    if weight <= 0:
        return {}
    return {key: float(value / weight) for key, value in running.items() if key != "_weight"}


def train_one_epoch(
    model: ResidualGraduateModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = _loss_components(
            model,
            batch,
            fde_weight=args.fde_weight,
            lambda_gt_min=args.lambda_gt_min,
            lambda_rank_gt=args.lambda_rank_gt,
            rank_gt_good_frac=args.rank_gt_good_frac,
            rank_gt_mid_frac=args.rank_gt_mid_frac,
            rank_gt_good_weight=args.rank_gt_good_weight,
            rank_gt_mid_weight=args.rank_gt_mid_weight,
            rank_gt_bad_weight=args.rank_gt_bad_weight,
            lambda_good_nohurt=args.lambda_good_nohurt,
            good_nohurt_frac=args.good_nohurt_frac,
            good_nohurt_margin=args.good_nohurt_margin,
            lambda_best_selector=args.lambda_best_selector,
            best_selector_top_k=args.best_selector_top_k,
            best_selector_positive_weight=args.best_selector_positive_weight,
            lambda_best_refine=args.lambda_best_refine,
            best_refine_top_k=args.best_refine_top_k,
            best_refine_ade_weight=args.best_refine_ade_weight,
            best_refine_fde_weight=args.best_refine_fde_weight,
            lambda_temporal_gate=args.lambda_temporal_gate,
            lambda_temporal_smoothness=args.lambda_temporal_smoothness,
            lambda_temporal_energy_gate=args.lambda_temporal_energy_gate,
            lambda_temporal_energy_gt=args.lambda_temporal_energy_gt,
            temporal_energy_gt_top_k=args.temporal_energy_gt_top_k,
            temporal_energy_gt_risk_floor=args.temporal_energy_gt_risk_floor,
            lambda_diversity_preserve=args.lambda_diversity_preserve,
            diversity_preserve_target_ratio=args.diversity_preserve_target_ratio,
            diversity_preserve_margin=args.diversity_preserve_margin,
            diversity_preserve_kind=args.diversity_preserve_kind,
            teacher_distill_mode=args.teacher_distill_mode,
            lambda_teacher=args.lambda_teacher,
            lambda_keep=args.lambda_keep,
            lambda_residual=args.lambda_residual,
            lambda_gate=args.lambda_gate,
            teacher_margin=args.teacher_margin,
            max_teacher_weight=args.max_teacher_weight,
        )
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        _accumulate_mean(running, components, weight=int(batch["ground_truth"].shape[0]))
    return _finalize_mean(running)


def _prediction_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    miss_threshold: float,
) -> Dict[str, float]:
    errors = displacement_errors(prediction, ground_truth, agent_mask=agent_mask)
    ade = errors["ade_per_mode_agent"]
    fde = errors["fde_per_mode_agent"]
    valid = errors["valid_agents"].bool()
    ade_min = ade.min(dim=1).values
    fde_min = fde.min(dim=1).values
    ade_avg = ade.mean(dim=1)
    fde_avg = fde.mean(dim=1)
    return {
        "ADE_min": float(_masked_mean(ade_min, valid).detach().cpu()),
        "FDE_min": float(_masked_mean(fde_min, valid).detach().cpu()),
        "ADE_avg": float(_masked_mean(ade_avg, valid).detach().cpu()),
        "FDE_avg": float(_masked_mean(fde_avg, valid).detach().cpu()),
        "MissRate": float(_masked_mean((fde_min > float(miss_threshold)).float(), valid).detach().cpu()),
        "num_valid_agents": float(valid.sum().detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: ResidualGraduateModel,
    loader: DataLoader,
    *,
    device: str,
    miss_threshold: float,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    valid_total = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(
            batch["student_pred"],
            batch["past_traj_original_scale"],
            agent_mask=batch.get("agent_mask"),
            interaction_energy_features=batch.get("interaction_energy_features"),
            temporal_interaction_energy_features=batch.get("temporal_interaction_energy_features"),
        )
        predictions = {
            "student": batch["student_pred"],
            "teacher": batch["teacher_pred"],
            "graduate": output["graduate_pred"],
        }
        batch_valid = float(batch["agent_mask"].bool().sum().detach().cpu())
        valid_total += batch_valid
        for prefix, prediction in predictions.items():
            metrics = _prediction_metrics(
                prediction,
                batch["ground_truth"],
                batch["agent_mask"].bool(),
                miss_threshold=miss_threshold,
            )
            for metric_name, metric_value in metrics.items():
                if metric_name == "num_valid_agents":
                    continue
                key = f"{prefix}_{metric_name}"
                sums[key] = sums.get(key, 0.0) + float(metric_value) * batch_valid
        sums["graduate_gate_mean"] = sums.get("graduate_gate_mean", 0.0) + float(output["gate"].mean().cpu()) * batch_valid
        sums["graduate_delta_l2_mean"] = (
            sums.get("graduate_delta_l2_mean", 0.0)
            + float(output["delta_pred"].pow(2).mean().sqrt().cpu()) * batch_valid
        )
        if "interaction_energy_features" in output:
            sums["interaction_energy_feature_mean"] = (
                sums.get("interaction_energy_feature_mean", 0.0)
                + float(output["interaction_energy_features"].mean().cpu()) * batch_valid
            )
        if "energy_delta_pred" in output:
            sums["energy_delta_l2_mean"] = (
                sums.get("energy_delta_l2_mean", 0.0)
                + float(output["energy_delta_pred"].pow(2).mean().sqrt().cpu()) * batch_valid
            )
        if "best_mode_selector_prob" in output:
            sums["best_selector_prob_mean"] = (
                sums.get("best_selector_prob_mean", 0.0)
                + float(output["best_mode_selector_prob"].mean().cpu()) * batch_valid
            )
        if "best_mode_refine_delta" in output:
            sums["best_refine_delta_l2_mean"] = (
                sums.get("best_refine_delta_l2_mean", 0.0)
                + float(output["best_mode_refine_delta"].pow(2).mean().sqrt().cpu()) * batch_valid
            )
        if "temporal_repair_gate" in output:
            sums["temporal_gate_mean"] = (
                sums.get("temporal_gate_mean", 0.0)
                + float(output["temporal_repair_gate"].mean().cpu()) * batch_valid
            )
        if "temporal_refine_delta" in output:
            sums["temporal_refine_delta_l2_mean"] = (
                sums.get("temporal_refine_delta_l2_mean", 0.0)
                + float(output["temporal_refine_delta"].pow(2).mean().sqrt().cpu()) * batch_valid
            )

    if valid_total <= 0:
        raise ValueError("Evaluation loader had no valid agents")
    finalized = {key: float(value / valid_total) for key, value in sums.items()}
    finalized["num_valid_agents"] = float(valid_total)
    return finalized


def _save_checkpoint(
    path: Path,
    *,
    model: ResidualGraduateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    cache_meta: Mapping[str, Any],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "cache_meta": dict(cache_meta),
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_cache(cache_path)
    tensors = payload["tensors"]
    num_items = int(tensors["ground_truth"].shape[0])
    train_indices, val_indices = _select_indices(
        num_items,
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )

    train_dataset = TeacherStudentCacheDataset(tensors, train_indices)
    val_dataset = TeacherStudentCacheDataset(tensors, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )

    model = build_residual_graduate_from_cache_shapes(
        payload.get("tensor_shapes") or {key: list(value.shape) for key, value in tensors.items()},
        hidden_dim=int(args.hidden_dim),
        num_residual_blocks=int(args.num_hidden_layers or args.num_residual_blocks),
        residual_block_type=str(args.residual_block_type),
        block_expansion=float(args.block_expansion),
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
        max_delta=args.max_delta,
        use_social_context=bool(args.use_social_context),
        context_dim=int(args.context_dim),
        context_hidden_dim=int(args.context_hidden_dim),
        context_num_layers=int(args.context_num_layers),
        context_num_heads=int(args.context_num_heads),
        context_dropout=float(args.context_dropout),
        use_interaction_energy=bool(args.use_interaction_energy),
        interaction_energy_dim=int(args.interaction_energy_dim),
        collision_sigma=float(args.collision_sigma),
        collision_radius=float(args.collision_radius),
        no_neighbor_distance=float(args.no_neighbor_distance),
        interaction_energy_temporal_stride=int(args.interaction_energy_temporal_stride),
        use_energy_conditioned_heads=bool(args.use_energy_conditioned_heads),
        energy_condition_dim=int(args.energy_condition_dim),
        use_time_aware_gate=bool(args.use_time_aware_gate),
        use_mode_set_context=bool(args.use_mode_set_context),
        mode_context_num_layers=int(args.mode_context_num_layers),
        mode_context_num_heads=int(args.mode_context_num_heads),
        mode_context_dropout=float(args.mode_context_dropout),
        use_best_mode_refiner=bool(args.use_best_mode_refiner),
        best_refine_scale=float(args.best_refine_scale),
        max_best_refine=args.max_best_refine,
        use_temporal_energy_refiner=bool(args.use_temporal_energy_refiner),
        temporal_interaction_energy_dim=int(args.temporal_interaction_energy_dim),
        temporal_refiner_hidden_dim=int(args.temporal_refiner_hidden_dim),
        temporal_refine_scale=float(args.temporal_refine_scale),
        max_temporal_refine=args.max_temporal_refine,
        temporal_gate_init_bias=float(args.temporal_gate_init_bias),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print("[train_residual_graduate] started")
    print(f"cache_path={cache_path.as_posix()}")
    print(f"device={device} train_items={len(train_dataset)} val_items={len(val_dataset)}")
    print(f"model_config={model.config.to_dict()}")

    best_epoch = 0
    best_val_fde = float("inf")
    best_val_metrics: Dict[str, float] = {}
    best_train_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []
    best_checkpoint = output_dir / f"{args.run_name}_best.pt"
    last_checkpoint = output_dir / f"{args.run_name}_last.pt"
    summary_path = output_dir / f"{args.run_name}_summary.json"

    for epoch in range(1, int(args.epochs) + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, device=device, args=args)
        val_metrics = evaluate(model, val_loader, device=device, miss_threshold=float(args.miss_threshold))
        train_metrics = evaluate(model, train_loader, device=device, miss_threshold=float(args.miss_threshold))

        current_val_fde = float(val_metrics["graduate_FDE_min"])
        improved = current_val_fde < best_val_fde
        if improved:
            best_epoch = int(epoch)
            best_val_fde = current_val_fde
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            _save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                cache_meta=payload.get("meta", {}),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        _save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            cache_meta=payload.get("meta", {}),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        row = {
            "epoch": int(epoch),
            "improved": bool(improved),
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(row)
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(int(args.log_every), 1) == 0:
            print(
                f"[train_residual_graduate] epoch={epoch}/{args.epochs} "
                f"loss={train_losses.get('loss_total', 0.0):.6f} "
                f"val_student_FDE={val_metrics['student_FDE_min']:.6f} "
                f"val_teacher_FDE={val_metrics['teacher_FDE_min']:.6f} "
                f"val_graduate_FDE={val_metrics['graduate_FDE_min']:.6f} "
                f"gate={val_metrics['graduate_gate_mean']:.4f} "
                f"improved={improved}"
            )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_residual_graduate",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_name": args.run_name,
            "device": device,
            "seed": int(args.seed),
            "cache_path": cache_path.as_posix(),
            "best_epoch": int(best_epoch),
            "best_checkpoint": best_checkpoint.as_posix(),
            "last_checkpoint": last_checkpoint.as_posix(),
        },
        "args": _jsonable(vars(args)),
        "cache_meta": _jsonable(payload.get("meta", {})),
        "cache_dataset": _jsonable(payload.get("dataset", {})),
        "model_config": model.config.to_dict(),
        "data_split": {
            "num_cache_items": int(num_items),
            "num_train_items": len(train_dataset),
            "num_val_items": len(val_dataset),
            "val_fraction": float(args.val_fraction),
            "max_items": args.max_items,
        },
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "history": history,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("[train_residual_graduate] completed")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_student_FDE_min={best_val_metrics.get('student_FDE_min')}")
    print(f"best_val_teacher_FDE_min={best_val_metrics.get('teacher_FDE_min')}")
    print(f"best_val_graduate_FDE_min={best_val_metrics.get('graduate_FDE_min')}")
    print(f"best_checkpoint={best_checkpoint.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
