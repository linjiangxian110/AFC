"""Interaction-energy features for residual trajectory refinement.

The functions here are intentionally lightweight and deterministic. They turn
scene-level predicted futures into per-mode, per-agent risk descriptors that a
Residual Graduate head can consume without adding a second heavy predictor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn


INTERACTION_ENERGY_FEATURE_NAMES: Sequence[str] = (
    "min_neighbor_distance",
    "soft_collision_energy",
    "close_neighbor_count",
    "approaching_score",
    "endpoint_crowding_energy",
)

TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES: Sequence[str] = (
    "min_neighbor_distance",
    "soft_collision_energy_mean",
    "close_neighbor_count_mean",
    "approaching_score_max",
    "endpoint_crowding_energy",
    "soft_collision_energy_max",
    "late_soft_collision_energy_mean",
    "soft_collision_energy_trend",
    "late_min_neighbor_distance",
    "min_distance_risk_trend",
    "path_deviation_vs_mean",
    "endpoint_deviation_vs_mean",
    "path_deviation_vs_base",
    "endpoint_deviation_vs_base",
    "smoothness_delta_vs_mean",
    "smoothness_delta_vs_base",
)

TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM = len(TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES)


@dataclass
class InteractionEnergyConfig:
    """Configuration for handcrafted social-interaction energy features."""

    collision_sigma: float = 0.5
    collision_radius: float = 0.2
    no_neighbor_distance: float = 10.0
    temporal_stride: int = 1
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_future_abs_shape(future_abs: torch.Tensor) -> torch.Tensor:
    if future_abs.ndim != 5 or int(future_abs.shape[-1]) != 2:
        raise ValueError(
            "future_abs must have shape [B, K, A, T, 2], "
            f"got {tuple(future_abs.shape)}"
        )
    return future_abs


def _ensure_past_abs_shape(
    past_abs: torch.Tensor,
    *,
    batch_size: int,
    num_agents: int,
) -> torch.Tensor:
    if past_abs.ndim != 4 or int(past_abs.shape[-1]) != 2:
        raise ValueError(
            "past_abs must have shape [B, A, P, 2], "
            f"got {tuple(past_abs.shape)}"
        )
    if int(past_abs.shape[0]) != int(batch_size):
        raise ValueError(f"past_abs batch mismatch: {int(past_abs.shape[0])} vs {batch_size}")
    if int(past_abs.shape[1]) != int(num_agents):
        raise ValueError(f"past_abs agent mismatch: {int(past_abs.shape[1])} vs {num_agents}")
    return past_abs


def _resolve_agent_mask(
    agent_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    if agent_mask is None:
        return torch.ones((batch_size, num_agents), dtype=torch.bool, device=device)
    mask = agent_mask.to(device=device).bool()
    if tuple(mask.shape) != (batch_size, num_agents):
        raise ValueError(
            "agent_mask must have shape [B, A], "
            f"got {tuple(mask.shape)} for B={batch_size}, A={num_agents}"
        )
    return mask


def compute_interaction_energy_features(
    future_abs: torch.Tensor,
    past_abs: torch.Tensor,
    *,
    agent_mask: Optional[torch.Tensor] = None,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
    temporal_stride: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-mode interaction features.

    Args:
        future_abs: Predicted future in a shared scene coordinate frame,
            shaped [B, K, A, T, 2].
        past_abs: Observed history in the same coordinate frame, shaped
            [B, A, P, 2].
        agent_mask: Optional valid-agent mask [B, A].

    Returns:
        Tensor [B, K, A, 5] matching INTERACTION_ENERGY_FEATURE_NAMES.
    """

    future = _ensure_future_abs_shape(future_abs).to(dtype=torch.float32)
    batch_size, num_modes, num_agents, num_steps, _coord_dim = future.shape
    past = _ensure_past_abs_shape(
        past_abs.to(device=future.device, dtype=torch.float32),
        batch_size=int(batch_size),
        num_agents=int(num_agents),
    )
    valid_agents = _resolve_agent_mask(
        agent_mask,
        batch_size=int(batch_size),
        num_agents=int(num_agents),
        device=future.device,
    )

    if float(collision_sigma) <= 0.0:
        raise ValueError(f"collision_sigma must be positive, got {collision_sigma}")
    if float(collision_radius) < 0.0:
        raise ValueError(f"collision_radius must be non-negative, got {collision_radius}")
    if float(no_neighbor_distance) <= 0.0:
        raise ValueError(f"no_neighbor_distance must be positive, got {no_neighbor_distance}")
    if int(temporal_stride) <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")

    if int(temporal_stride) > 1:
        future_pair = future[:, :, :, :: int(temporal_stride), :]
        if int(future_pair.shape[3]) <= 0:
            future_pair = future[:, :, :, -1:, :]
        elif (int(num_steps) - 1) % int(temporal_stride) != 0:
            future_pair = torch.cat([future_pair, future[:, :, :, -1:, :]], dim=3)
    else:
        future_pair = future
    num_pair_steps = int(future_pair.shape[3])

    dtype = future.dtype
    eye = torch.eye(num_agents, dtype=torch.bool, device=future.device)
    pair_valid = (
        valid_agents[:, None, :, None, None]
        & valid_agents[:, None, None, :, None]
        & (~eye)[None, None, :, :, None]
    )
    pair_valid_time = pair_valid.expand(batch_size, num_modes, num_agents, num_agents, num_pair_steps)
    has_neighbor = pair_valid.any(dim=4).any(dim=3)

    pair_delta = future_pair[:, :, :, None, :, :] - future_pair[:, :, None, :, :, :]
    distances = torch.linalg.norm(pair_delta, dim=-1)
    large_distance = future.new_tensor(float(no_neighbor_distance))
    masked_distances = torch.where(pair_valid_time, distances, large_distance)
    min_neighbor_distance = masked_distances.amin(dim=4).amin(dim=3)
    min_neighbor_distance = torch.where(
        has_neighbor,
        min_neighbor_distance,
        large_distance.expand_as(min_neighbor_distance),
    )

    sigma = max(float(collision_sigma), float(eps))
    collision_energy = torch.exp(-(distances / sigma).pow(2)) * pair_valid_time.to(dtype=dtype)
    soft_collision_energy = collision_energy.sum(dim=4).sum(dim=3) / float(max(num_pair_steps, 1))

    close_neighbor_count = (
        ((distances < float(collision_radius)) & pair_valid_time).to(dtype=dtype).sum(dim=3).mean(dim=-1)
    )

    past_last = past[:, :, -1, :]
    past_distances = torch.linalg.norm(past_last[:, :, None, :] - past_last[:, None, :, :], dim=-1)
    min_future_pair_distance = torch.where(pair_valid_time, distances, large_distance).amin(dim=-1)
    pair_approach = (past_distances[:, None, :, :] - min_future_pair_distance).clamp_min(0.0)
    pair_approach = pair_approach / past_distances[:, None, :, :].clamp_min(float(eps))
    pair_approach = torch.where(pair_valid.squeeze(-1), pair_approach, torch.zeros_like(pair_approach))
    approaching_score = pair_approach.amax(dim=3)

    endpoints = future[..., -1, :]
    endpoint_distances = torch.linalg.norm(
        endpoints[:, :, :, None, :] - endpoints[:, :, None, :, :],
        dim=-1,
    )
    endpoint_valid = pair_valid.squeeze(-1)
    endpoint_crowding_energy = (
        torch.exp(-(endpoint_distances / sigma).pow(2)) * endpoint_valid.to(dtype=dtype)
    ).sum(dim=3)

    features = torch.stack(
        [
            min_neighbor_distance,
            soft_collision_energy,
            close_neighbor_count,
            approaching_score,
            endpoint_crowding_energy,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=float(no_neighbor_distance), neginf=0.0)


def compute_temporal_interaction_energy_features(
    future_abs: torch.Tensor,
    past_abs: torch.Tensor,
    *,
    agent_mask: Optional[torch.Tensor] = None,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-timestep interaction features.

    Returns [B, K, A, T, 5] using the same feature order as
    INTERACTION_ENERGY_FEATURE_NAMES. The endpoint crowding channel is computed
    once from final positions and broadcast over time, so a temporal head can
    still condition local repairs on final-mode crowding.
    """

    future = _ensure_future_abs_shape(future_abs).to(dtype=torch.float32)
    batch_size, num_modes, num_agents, num_steps, _coord_dim = future.shape
    past = _ensure_past_abs_shape(
        past_abs.to(device=future.device, dtype=torch.float32),
        batch_size=int(batch_size),
        num_agents=int(num_agents),
    )
    valid_agents = _resolve_agent_mask(
        agent_mask,
        batch_size=int(batch_size),
        num_agents=int(num_agents),
        device=future.device,
    )

    if float(collision_sigma) <= 0.0:
        raise ValueError(f"collision_sigma must be positive, got {collision_sigma}")
    if float(collision_radius) < 0.0:
        raise ValueError(f"collision_radius must be non-negative, got {collision_radius}")
    if float(no_neighbor_distance) <= 0.0:
        raise ValueError(f"no_neighbor_distance must be positive, got {no_neighbor_distance}")

    dtype = future.dtype
    eye = torch.eye(num_agents, dtype=torch.bool, device=future.device)
    pair_valid = (
        valid_agents[:, None, :, None, None]
        & valid_agents[:, None, None, :, None]
        & (~eye)[None, None, :, :, None]
    )
    pair_valid_time = pair_valid.expand(batch_size, num_modes, num_agents, num_agents, num_steps)
    has_neighbor = pair_valid.any(dim=4).any(dim=3).unsqueeze(-1)

    pair_delta = future[:, :, :, None, :, :] - future[:, :, None, :, :, :]
    distances = torch.linalg.norm(pair_delta, dim=-1)
    large_distance = future.new_tensor(float(no_neighbor_distance))
    masked_distances = torch.where(pair_valid_time, distances, large_distance)
    min_neighbor_distance = masked_distances.amin(dim=3)
    min_neighbor_distance = torch.where(
        has_neighbor,
        min_neighbor_distance,
        large_distance.expand_as(min_neighbor_distance),
    )

    sigma = max(float(collision_sigma), float(eps))
    collision_energy = torch.exp(-(distances / sigma).pow(2)) * pair_valid_time.to(dtype=dtype)
    soft_collision_energy = collision_energy.sum(dim=3)
    close_neighbor_count = ((distances < float(collision_radius)) & pair_valid_time).to(dtype=dtype).sum(dim=3)

    past_last = past[:, :, -1, :]
    past_distances = torch.linalg.norm(past_last[:, :, None, :] - past_last[:, None, :, :], dim=-1)
    pair_approach = (past_distances[:, None, :, :, None] - distances).clamp_min(0.0)
    pair_approach = pair_approach / past_distances[:, None, :, :, None].clamp_min(float(eps))
    pair_approach = torch.where(pair_valid_time, pair_approach, torch.zeros_like(pair_approach))
    approaching_score = pair_approach.amax(dim=3)

    endpoints = future[..., -1, :]
    endpoint_distances = torch.linalg.norm(
        endpoints[:, :, :, None, :] - endpoints[:, :, None, :, :],
        dim=-1,
    )
    endpoint_valid = pair_valid.squeeze(-1)
    endpoint_crowding_energy = (
        torch.exp(-(endpoint_distances / sigma).pow(2)) * endpoint_valid.to(dtype=dtype)
    ).sum(dim=3)
    endpoint_crowding_energy = endpoint_crowding_energy[..., None].expand(
        batch_size,
        num_modes,
        num_agents,
        num_steps,
    )

    features = torch.stack(
        [
            min_neighbor_distance,
            soft_collision_energy,
            close_neighbor_count,
            approaching_score,
            endpoint_crowding_energy,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=float(no_neighbor_distance), neginf=0.0)


def _trajectory_smoothness(traj: torch.Tensor) -> torch.Tensor:
    if int(traj.shape[-2]) < 3:
        return torch.zeros(traj.shape[:-2], dtype=traj.dtype, device=traj.device)
    accel = traj[..., 2:, :] - 2.0 * traj[..., 1:-1, :] + traj[..., :-2, :]
    return torch.linalg.norm(accel, dim=-1).mean(dim=-1)


def compute_trajectory_aware_interaction_summary_features(
    temporal_energy_features: torch.Tensor,
    candidate_future: torch.Tensor,
    base_future: torch.Tensor,
) -> torch.Tensor:
    """Summarize candidate interaction energy with trajectory-shape features.

    Args:
        temporal_energy_features: Candidate temporal energy [B, S, K, A, T, C].
            The first five channels must follow INTERACTION_ENERGY_FEATURE_NAMES.
        candidate_future: Candidate futures [B, S, K, A, T, 2].
        base_future: Base teacher futures [B, K, A, T, 2] or [B, 1, K, A, T, 2].

    Returns:
        Tensor [B, S, K, A, 16] following
        TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES.
    """

    if temporal_energy_features.ndim != 6:
        raise ValueError(
            "temporal_energy_features must have shape [B,S,K,A,T,C], "
            f"got {tuple(temporal_energy_features.shape)}"
        )
    if candidate_future.ndim != 6 or int(candidate_future.shape[-1]) != 2:
        raise ValueError(
            "candidate_future must have shape [B,S,K,A,T,2], "
            f"got {tuple(candidate_future.shape)}"
        )
    if tuple(temporal_energy_features.shape[:5]) != tuple(candidate_future.shape[:5]):
        raise ValueError(
            "temporal energy/candidate shape mismatch: "
            f"energy={tuple(temporal_energy_features.shape)} candidate={tuple(candidate_future.shape)}"
        )
    if int(temporal_energy_features.shape[-1]) < 5:
        raise ValueError(
            "trajectory-aware summary requires at least 5 temporal energy channels, "
            f"got {int(temporal_energy_features.shape[-1])}"
        )

    if base_future.ndim == 5:
        base = base_future[:, None, ...]
    elif base_future.ndim == 6:
        base = base_future
    else:
        raise ValueError(f"base_future must have shape [B,K,A,T,2] or [B,1,K,A,T,2], got {tuple(base_future.shape)}")
    if int(base.shape[1]) == 1:
        base = base.expand(
            int(candidate_future.shape[0]),
            int(candidate_future.shape[1]),
            int(candidate_future.shape[2]),
            int(candidate_future.shape[3]),
            int(candidate_future.shape[4]),
            2,
        )
    if tuple(base.shape) != tuple(candidate_future.shape):
        raise ValueError(f"base/candidate shape mismatch: base={tuple(base.shape)} candidate={tuple(candidate_future.shape)}")

    energy = torch.nan_to_num(temporal_energy_features.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    candidates = candidate_future.to(device=energy.device, dtype=energy.dtype)
    base = base.to(device=energy.device, dtype=energy.dtype)
    mean_candidate = candidates[:, :1, ...].expand_as(candidates)
    window = min(3, int(energy.shape[-2]))

    min_neighbor_distance = energy[..., 0].amin(dim=-1)
    soft_collision_mean = energy[..., 1].mean(dim=-1)
    close_neighbor_mean = energy[..., 2].mean(dim=-1)
    approaching_max = energy[..., 3].amax(dim=-1)
    endpoint_crowding = energy[..., -1, 4]

    soft_collision_max = energy[..., 1].amax(dim=-1)
    early_collision_mean = energy[..., :window, 1].mean(dim=-1)
    late_collision_mean = energy[..., -window:, 1].mean(dim=-1)
    collision_trend = late_collision_mean - early_collision_mean

    early_min_distance = energy[..., :window, 0].amin(dim=-1)
    late_min_distance = energy[..., -window:, 0].amin(dim=-1)
    min_distance_risk_trend = early_min_distance - late_min_distance

    path_deviation_vs_mean = torch.linalg.norm(candidates - mean_candidate, dim=-1).mean(dim=-1)
    endpoint_deviation_vs_mean = torch.linalg.norm(candidates[..., -1, :] - mean_candidate[..., -1, :], dim=-1)
    path_deviation_vs_base = torch.linalg.norm(candidates - base, dim=-1).mean(dim=-1)
    endpoint_deviation_vs_base = torch.linalg.norm(candidates[..., -1, :] - base[..., -1, :], dim=-1)

    candidate_smoothness = _trajectory_smoothness(candidates)
    mean_smoothness = _trajectory_smoothness(mean_candidate)
    base_smoothness = _trajectory_smoothness(base)
    smoothness_delta_vs_mean = candidate_smoothness - mean_smoothness
    smoothness_delta_vs_base = candidate_smoothness - base_smoothness

    features = torch.stack(
        [
            min_neighbor_distance,
            soft_collision_mean,
            close_neighbor_mean,
            approaching_max,
            endpoint_crowding,
            soft_collision_max,
            late_collision_mean,
            collision_trend,
            late_min_distance,
            min_distance_risk_trend,
            path_deviation_vs_mean,
            endpoint_deviation_vs_mean,
            path_deviation_vs_base,
            endpoint_deviation_vs_base,
            smoothness_delta_vs_mean,
            smoothness_delta_vs_base,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class InteractionEnergyFeatureBuilder(nn.Module):
    """nn.Module wrapper for interaction-energy feature computation."""

    feature_names = tuple(INTERACTION_ENERGY_FEATURE_NAMES)

    def __init__(self, config: InteractionEnergyConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            self.config = InteractionEnergyConfig()
        elif isinstance(config, InteractionEnergyConfig):
            self.config = config
        else:
            self.config = InteractionEnergyConfig(**dict(config))

    @property
    def output_dim(self) -> int:
        return len(INTERACTION_ENERGY_FEATURE_NAMES)

    def forward(
        self,
        future_abs: torch.Tensor,
        past_abs: torch.Tensor,
        *,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        return compute_interaction_energy_features(
            future_abs,
            past_abs,
            agent_mask=agent_mask,
            collision_sigma=float(cfg.collision_sigma),
            collision_radius=float(cfg.collision_radius),
            no_neighbor_distance=float(cfg.no_neighbor_distance),
            temporal_stride=int(cfg.temporal_stride),
            eps=float(cfg.eps),
        )


class TemporalInteractionEnergyFeatureBuilder(InteractionEnergyFeatureBuilder):
    """nn.Module wrapper for per-timestep interaction-energy features."""

    def forward(
        self,
        future_abs: torch.Tensor,
        past_abs: torch.Tensor,
        *,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        return compute_temporal_interaction_energy_features(
            future_abs,
            past_abs,
            agent_mask=agent_mask,
            collision_sigma=float(cfg.collision_sigma),
            collision_radius=float(cfg.collision_radius),
            no_neighbor_distance=float(cfg.no_neighbor_distance),
            eps=float(cfg.eps),
        )


__all__ = [
    "INTERACTION_ENERGY_FEATURE_NAMES",
    "TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM",
    "TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES",
    "InteractionEnergyConfig",
    "InteractionEnergyFeatureBuilder",
    "TemporalInteractionEnergyFeatureBuilder",
    "compute_interaction_energy_features",
    "compute_temporal_interaction_energy_features",
    "compute_trajectory_aware_interaction_summary_features",
]
