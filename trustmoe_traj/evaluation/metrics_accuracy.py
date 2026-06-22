"""Accuracy metrics for TrustMoE-Traj evaluators."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch


def _to_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _ensure_prediction_shape(prediction: Any) -> torch.Tensor:
    tensor = _to_tensor(prediction, dtype=torch.float32)
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"Prediction must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(tensor.shape)}"
    )


def _ensure_ground_truth_shape(ground_truth: Any) -> torch.Tensor:
    tensor = _to_tensor(ground_truth, dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"Ground truth must have shape [B, A, T, 2], got {tuple(tensor.shape)}")
    return tensor


def _resolve_agent_mask(
    agent_mask: Optional[Any],
    *,
    batch_size: int,
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    if agent_mask is None:
        return torch.ones((batch_size, num_agents), dtype=torch.bool, device=device)
    mask = _to_tensor(agent_mask).to(device=device)
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, num_agents):
        raise ValueError(
            f"agent_mask must have shape [{batch_size}, {num_agents}], got {tuple(mask.shape)}"
        )
    return mask.bool()


def displacement_errors(
    prediction: Any,
    ground_truth: Any,
    *,
    agent_mask: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-agent displacement errors for multi-modal predictions.

    Returns tensors shaped:
    - ade_per_mode_agent: [B, K, A]
    - fde_per_mode_agent: [B, K, A]
    """

    pred = _ensure_prediction_shape(prediction)
    gt = _ensure_ground_truth_shape(ground_truth).to(device=pred.device)

    if pred.shape[0] != gt.shape[0] or pred.shape[2:] != gt.shape[1:]:
        raise ValueError(
            "Prediction / ground truth shape mismatch: "
            f"prediction={tuple(pred.shape)}, ground_truth={tuple(gt.shape)}"
        )

    batch_size, _, num_agents, _, _ = pred.shape
    valid_agents = _resolve_agent_mask(
        agent_mask,
        batch_size=batch_size,
        num_agents=num_agents,
        device=pred.device,
    )

    distances = torch.linalg.norm(pred - gt[:, None, ...], dim=-1)  # [B, K, A, T]
    ade_per_mode_agent = distances.mean(dim=-1)
    fde_per_mode_agent = distances[..., -1]

    return {
        "ade_per_mode_agent": ade_per_mode_agent,
        "fde_per_mode_agent": fde_per_mode_agent,
        "valid_agents": valid_agents,
    }


def summarize_accuracy_metrics(
    prediction: Any,
    ground_truth: Any,
    *,
    agent_mask: Optional[Any] = None,
    miss_threshold: float = 2.0,
) -> Dict[str, float]:
    """Summarize ADE / FDE / Miss Rate on valid agents."""

    errors = displacement_errors(prediction, ground_truth, agent_mask=agent_mask)
    ade = errors["ade_per_mode_agent"]
    fde = errors["fde_per_mode_agent"]
    valid = errors["valid_agents"]

    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        raise ValueError("No valid agents available for accuracy evaluation")

    valid_expanded = valid[:, None, :]
    inf = torch.tensor(float("inf"), device=ade.device, dtype=ade.dtype)

    ade_min = ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean()
    fde_min = fde.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean()

    ade_avg = ade.mean(dim=1)[valid].mean()
    fde_avg = fde.mean(dim=1)[valid].mean()

    miss = (fde.masked_fill(~valid_expanded, inf).min(dim=1).values > miss_threshold)[valid].float().mean()

    return {
        "num_valid_agents": float(valid_count),
        "ADE_min": float(ade_min.detach().cpu()),
        "FDE_min": float(fde_min.detach().cpu()),
        "ADE_avg": float(ade_avg.detach().cpu()),
        "FDE_avg": float(fde_avg.detach().cpu()),
        "MissRate": float(miss.detach().cpu()),
        "miss_threshold": float(miss_threshold),
    }


def infer_ground_truth_from_batch(batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Infer evaluator ground truth tensors from either standard or MoFlow batch dicts."""

    if "fut_traj_original_scale" in batch:
        gt = _ensure_ground_truth_shape(batch["fut_traj_original_scale"])
    elif "future_traj" in batch:
        gt = _ensure_ground_truth_shape(batch["future_traj"])
    else:
        raise KeyError("Batch does not contain `fut_traj_original_scale` or `future_traj` ground truth")

    if "agent_mask" in batch:
        agent_mask = _to_tensor(batch["agent_mask"]).bool()
    else:
        agent_mask = torch.ones((gt.shape[0], gt.shape[1]), dtype=torch.bool, device=gt.device)

    return {
        "ground_truth": gt,
        "agent_mask": agent_mask,
    }


__all__ = [
    "displacement_errors",
    "summarize_accuracy_metrics",
    "infer_ground_truth_from_batch",
]
