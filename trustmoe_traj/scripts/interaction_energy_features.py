"""Helpers for scene-aware interaction energy in per-agent MoFlow runs."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.models.interaction_energy import (
    compute_interaction_energy_features,
    compute_temporal_interaction_energy_features,
    compute_trajectory_aware_interaction_summary_features,
)


def _to_float_tensor(value: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _active_agent_indices(sample: Mapping[str, Any]) -> List[int]:
    past = torch.as_tensor(sample["past_traj"])
    agent_mask = sample.get("agent_mask")
    if agent_mask is None:
        return list(range(int(past.shape[0])))

    mask = torch.as_tensor(agent_mask).reshape(-1).bool()
    if int(mask.numel()) != int(past.shape[0]):
        raise ValueError(f"agent_mask length mismatch: {int(mask.numel())} vs {int(past.shape[0])}")
    active = [int(index) for index, flag in enumerate(mask.tolist()) if bool(flag)]
    return active or list(range(int(past.shape[0])))


def _rotation_matrices_for_agents(
    past_abs: torch.Tensor,
    *,
    agent_indices: Sequence[int],
    rotate_time_frame: int,
) -> torch.Tensor:
    if not (0 <= int(rotate_time_frame) < int(past_abs.shape[1])):
        raise ValueError(
            f"rotate_time_frame out of range: {rotate_time_frame}, "
            f"expected [0, {int(past_abs.shape[1]) - 1}]"
        )
    if not agent_indices:
        raise ValueError("agent_indices must not be empty")

    index = torch.as_tensor([int(item) for item in agent_indices], dtype=torch.long, device=past_abs.device)
    selected = past_abs.index_select(dim=0, index=index)
    initial_pos = selected[:, -1, :]
    past_rel = selected - initial_pos[:, None, :]
    direction = past_rel[:, int(rotate_time_frame), :]
    theta = torch.atan2(direction[:, 1], direction[:, 0] + 1e-5)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta, cos_theta], dim=-1),
        ],
        dim=1,
    )


def build_per_agent_scene_interaction_features(
    samples: Sequence[Mapping[str, Any]],
    prediction: torch.Tensor,
    *,
    rotate: bool,
    rotate_time_frame: int,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
    temporal_stride: int = 1,
) -> torch.Tensor:
    """Build [B, K, 1, C] scene-aware energy features for per-agent rows.

    The row order must match build_moflow_eth_batch(..., sample_mode='per_agent'):
    raw scenes are traversed in order, and active agents are expanded in their
    original scene order.
    """

    pred = prediction.detach()
    if pred.ndim == 4:
        pred = pred.unsqueeze(2)
    if pred.ndim != 5 or int(pred.shape[2]) != 1 or int(pred.shape[-1]) != 2:
        raise ValueError(
            "prediction must have shape [B, K, T, 2] or [B, K, 1, T, 2] for per-agent energy, "
            f"got {tuple(prediction.shape)}"
        )

    feature_chunks: List[torch.Tensor] = []
    row_index = 0
    for sample in samples:
        active_indices = _active_agent_indices(sample)
        num_active = len(active_indices)
        if num_active <= 0:
            continue
        if row_index + num_active > int(pred.shape[0]):
            raise ValueError("prediction has fewer per-agent rows than the provided samples require")

        scene_rel = pred[row_index : row_index + num_active, :, 0, :, :]
        row_index += num_active

        raw_past = _to_float_tensor(sample["past_traj"], device=scene_rel.device)
        if raw_past.ndim != 3 or int(raw_past.shape[-1]) != 2:
            raise ValueError(f"sample past_traj must have shape [A, P, 2], got {tuple(raw_past.shape)}")

        if rotate:
            rotations = _rotation_matrices_for_agents(
                raw_past,
                agent_indices=active_indices,
                rotate_time_frame=int(rotate_time_frame),
            )
            scene_rel = torch.einsum("nktc,ncd->nktd", scene_rel, rotations)

        active_index = torch.as_tensor(active_indices, dtype=torch.long, device=scene_rel.device)
        last_observed = raw_past.index_select(dim=0, index=active_index)[:, -1, :]
        scene_abs = scene_rel + last_observed[:, None, None, :]
        scene_future = scene_abs.permute(1, 0, 2, 3).unsqueeze(0)
        scene_past = raw_past.index_select(dim=0, index=active_index).unsqueeze(0)
        scene_mask = torch.ones((1, len(active_indices)), dtype=torch.bool, device=scene_future.device)
        scene_energy = compute_interaction_energy_features(
            scene_future,
            scene_past,
            agent_mask=scene_mask,
            collision_sigma=float(collision_sigma),
            collision_radius=float(collision_radius),
            no_neighbor_distance=float(no_neighbor_distance),
            temporal_stride=int(temporal_stride),
        )
        feature_chunks.append(scene_energy.squeeze(0).permute(1, 0, 2).unsqueeze(2))

    if row_index != int(pred.shape[0]):
        raise ValueError(
            "prediction has more per-agent rows than the provided samples require: "
            f"used={row_index}, total={int(pred.shape[0])}"
        )
    if not feature_chunks:
        raise ValueError("No interaction energy rows were produced")
    return torch.cat(feature_chunks, dim=0).to(device=prediction.device, dtype=torch.float32)


def build_per_agent_scene_candidate_interaction_features(
    samples: Sequence[Mapping[str, Any]],
    prediction: torch.Tensor,
    *,
    rotate: bool,
    rotate_time_frame: int,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
    temporal_stride: int = 1,
) -> torch.Tensor:
    """Build [B, S, K, 1, C] scene-aware features for sampled candidates.

    This is the batched counterpart of build_per_agent_scene_interaction_features.
    It treats the sample and mode axes as one expanded mode axis while computing
    pairwise scene energy, then reshapes back to [S, K].
    """

    pred = prediction.detach()
    if pred.ndim == 5:
        pred = pred.unsqueeze(3)
    if pred.ndim != 6 or int(pred.shape[3]) != 1 or int(pred.shape[-1]) != 2:
        raise ValueError(
            "prediction must have shape [B, S, K, T, 2] or [B, S, K, 1, T, 2] "
            f"for per-agent candidate energy, got {tuple(prediction.shape)}"
        )

    feature_chunks: List[torch.Tensor] = []
    row_index = 0
    for sample in samples:
        active_indices = _active_agent_indices(sample)
        num_active = len(active_indices)
        if num_active <= 0:
            continue
        if row_index + num_active > int(pred.shape[0]):
            raise ValueError("prediction has fewer per-agent rows than the provided samples require")

        scene_rel = pred[row_index : row_index + num_active, :, :, 0, :, :]
        row_index += num_active

        raw_past = _to_float_tensor(sample["past_traj"], device=scene_rel.device)
        if raw_past.ndim != 3 or int(raw_past.shape[-1]) != 2:
            raise ValueError(f"sample past_traj must have shape [A, P, 2], got {tuple(raw_past.shape)}")

        if rotate:
            rotations = _rotation_matrices_for_agents(
                raw_past,
                agent_indices=active_indices,
                rotate_time_frame=int(rotate_time_frame),
            )
            scene_rel = torch.einsum("nsktc,ncd->nsktd", scene_rel, rotations)

        active_index = torch.as_tensor(active_indices, dtype=torch.long, device=scene_rel.device)
        last_observed = raw_past.index_select(dim=0, index=active_index)[:, -1, :]
        scene_abs = scene_rel + last_observed[:, None, None, None, :]
        num_samples = int(scene_abs.shape[1])
        num_modes = int(scene_abs.shape[2])
        scene_future = (
            scene_abs.permute(1, 2, 0, 3, 4)
            .reshape(num_samples * num_modes, num_active, int(scene_abs.shape[-2]), 2)
            .unsqueeze(0)
        )
        scene_past = raw_past.index_select(dim=0, index=active_index).unsqueeze(0)
        scene_mask = torch.ones((1, len(active_indices)), dtype=torch.bool, device=scene_future.device)
        scene_energy = compute_interaction_energy_features(
            scene_future,
            scene_past,
            agent_mask=scene_mask,
            collision_sigma=float(collision_sigma),
            collision_radius=float(collision_radius),
            no_neighbor_distance=float(no_neighbor_distance),
            temporal_stride=int(temporal_stride),
        )
        scene_energy = scene_energy.squeeze(0).reshape(num_samples, num_modes, num_active, -1)
        feature_chunks.append(scene_energy.permute(2, 0, 1, 3).unsqueeze(3))

    if row_index != int(pred.shape[0]):
        raise ValueError(
            "prediction has more per-agent rows than the provided samples require: "
            f"used={row_index}, total={int(pred.shape[0])}"
        )
    if not feature_chunks:
        raise ValueError("No candidate interaction energy rows were produced")
    return torch.cat(feature_chunks, dim=0).to(device=prediction.device, dtype=torch.float32)


def build_per_agent_scene_candidate_trajectory_aware_interaction_features(
    samples: Sequence[Mapping[str, Any]],
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    *,
    rotate: bool,
    rotate_time_frame: int,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
) -> torch.Tensor:
    """Build [B, S, K, 1, C] trajectory-aware scene features for candidates."""

    pred = prediction.detach()
    if pred.ndim == 5:
        pred = pred.unsqueeze(3)
    if pred.ndim != 6 or int(pred.shape[3]) != 1 or int(pred.shape[-1]) != 2:
        raise ValueError(
            "prediction must have shape [B, S, K, T, 2] or [B, S, K, 1, T, 2] "
            f"for trajectory-aware candidate energy, got {tuple(prediction.shape)}"
        )
    base = base_prediction.detach()
    if base.ndim == 4:
        base = base.unsqueeze(2)
    if base.ndim != 5 or int(base.shape[2]) != 1 or int(base.shape[-1]) != 2:
        raise ValueError(
            "base_prediction must have shape [B, K, T, 2] or [B, K, 1, T, 2], "
            f"got {tuple(base_prediction.shape)}"
        )
    if int(base.shape[0]) != int(pred.shape[0]) or int(base.shape[1]) != int(pred.shape[2]):
        raise ValueError(f"base/prediction shape mismatch: base={tuple(base.shape)} prediction={tuple(pred.shape)}")

    feature_chunks: List[torch.Tensor] = []
    row_index = 0
    for sample in samples:
        active_indices = _active_agent_indices(sample)
        num_active = len(active_indices)
        if num_active <= 0:
            continue
        if row_index + num_active > int(pred.shape[0]):
            raise ValueError("prediction has fewer per-agent rows than the provided samples require")

        scene_rel = pred[row_index : row_index + num_active, :, :, 0, :, :]
        base_rel = base[row_index : row_index + num_active, :, 0, :, :]
        row_index += num_active

        raw_past = _to_float_tensor(sample["past_traj"], device=scene_rel.device)
        if raw_past.ndim != 3 or int(raw_past.shape[-1]) != 2:
            raise ValueError(f"sample past_traj must have shape [A, P, 2], got {tuple(raw_past.shape)}")

        if rotate:
            rotations = _rotation_matrices_for_agents(
                raw_past,
                agent_indices=active_indices,
                rotate_time_frame=int(rotate_time_frame),
            )
            scene_rel = torch.einsum("nsktc,ncd->nsktd", scene_rel, rotations)
            base_rel = torch.einsum("nktc,ncd->nktd", base_rel, rotations)

        active_index = torch.as_tensor(active_indices, dtype=torch.long, device=scene_rel.device)
        last_observed = raw_past.index_select(dim=0, index=active_index)[:, -1, :]
        scene_abs = scene_rel + last_observed[:, None, None, None, :]
        base_abs = base_rel + last_observed[:, None, None, :]
        num_samples = int(scene_abs.shape[1])
        num_modes = int(scene_abs.shape[2])
        num_steps = int(scene_abs.shape[-2])
        scene_future = (
            scene_abs.permute(1, 2, 0, 3, 4)
            .reshape(num_samples * num_modes, num_active, num_steps, 2)
            .unsqueeze(0)
        )
        scene_past = raw_past.index_select(dim=0, index=active_index).unsqueeze(0)
        scene_mask = torch.ones((1, len(active_indices)), dtype=torch.bool, device=scene_future.device)
        temporal_energy = compute_temporal_interaction_energy_features(
            scene_future,
            scene_past,
            agent_mask=scene_mask,
            collision_sigma=float(collision_sigma),
            collision_radius=float(collision_radius),
            no_neighbor_distance=float(no_neighbor_distance),
        )
        temporal_energy = temporal_energy.squeeze(0).reshape(num_samples, num_modes, num_active, num_steps, -1)
        candidate_future = scene_abs.permute(1, 2, 0, 3, 4).unsqueeze(0)
        base_future = base_abs.permute(1, 0, 2, 3).unsqueeze(0)
        scene_summary = compute_trajectory_aware_interaction_summary_features(
            temporal_energy.unsqueeze(0),
            candidate_future,
            base_future,
        )
        feature_chunks.append(scene_summary.squeeze(0).permute(2, 0, 1, 3).unsqueeze(3))

    if row_index != int(pred.shape[0]):
        raise ValueError(
            "prediction has more per-agent rows than the provided samples require: "
            f"used={row_index}, total={int(pred.shape[0])}"
        )
    if not feature_chunks:
        raise ValueError("No trajectory-aware candidate interaction energy rows were produced")
    return torch.cat(feature_chunks, dim=0).to(device=prediction.device, dtype=torch.float32)


def build_per_agent_scene_temporal_interaction_features(
    samples: Sequence[Mapping[str, Any]],
    prediction: torch.Tensor,
    *,
    rotate: bool,
    rotate_time_frame: int,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
) -> torch.Tensor:
    """Build [B, K, 1, T, C] temporal scene-aware energy features."""

    pred = prediction.detach()
    if pred.ndim == 4:
        pred = pred.unsqueeze(2)
    if pred.ndim != 5 or int(pred.shape[2]) != 1 or int(pred.shape[-1]) != 2:
        raise ValueError(
            "prediction must have shape [B, K, T, 2] or [B, K, 1, T, 2] for per-agent energy, "
            f"got {tuple(prediction.shape)}"
        )

    feature_chunks: List[torch.Tensor] = []
    row_index = 0
    for sample in samples:
        active_indices = _active_agent_indices(sample)
        num_active = len(active_indices)
        if num_active <= 0:
            continue
        if row_index + num_active > int(pred.shape[0]):
            raise ValueError("prediction has fewer per-agent rows than the provided samples require")

        scene_rel = pred[row_index : row_index + num_active, :, 0, :, :]
        row_index += num_active

        raw_past = _to_float_tensor(sample["past_traj"], device=scene_rel.device)
        if raw_past.ndim != 3 or int(raw_past.shape[-1]) != 2:
            raise ValueError(f"sample past_traj must have shape [A, P, 2], got {tuple(raw_past.shape)}")

        if rotate:
            rotations = _rotation_matrices_for_agents(
                raw_past,
                agent_indices=active_indices,
                rotate_time_frame=int(rotate_time_frame),
            )
            scene_rel = torch.einsum("nktc,ncd->nktd", scene_rel, rotations)

        active_index = torch.as_tensor(active_indices, dtype=torch.long, device=scene_rel.device)
        last_observed = raw_past.index_select(dim=0, index=active_index)[:, -1, :]
        scene_abs = scene_rel + last_observed[:, None, None, :]
        scene_future = scene_abs.permute(1, 0, 2, 3).unsqueeze(0)
        scene_past = raw_past.index_select(dim=0, index=active_index).unsqueeze(0)
        scene_mask = torch.ones((1, len(active_indices)), dtype=torch.bool, device=scene_future.device)
        scene_energy = compute_temporal_interaction_energy_features(
            scene_future,
            scene_past,
            agent_mask=scene_mask,
            collision_sigma=float(collision_sigma),
            collision_radius=float(collision_radius),
            no_neighbor_distance=float(no_neighbor_distance),
        )
        feature_chunks.append(scene_energy.squeeze(0).permute(1, 0, 2, 3).unsqueeze(2))

    if row_index != int(pred.shape[0]):
        raise ValueError(
            "prediction has more per-agent rows than the provided samples require: "
            f"used={row_index}, total={int(pred.shape[0])}"
        )
    if not feature_chunks:
        raise ValueError("No temporal interaction energy rows were produced")
    return torch.cat(feature_chunks, dim=0).to(device=prediction.device, dtype=torch.float32)


__all__ = [
    "build_per_agent_scene_candidate_interaction_features",
    "build_per_agent_scene_candidate_trajectory_aware_interaction_features",
    "build_per_agent_scene_interaction_features",
    "build_per_agent_scene_temporal_interaction_features",
]
