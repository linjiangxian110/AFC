"""Observable-feature scorer for V58 residual slot quality.

The scorer is intentionally inference-time only: labels can use ground truth
during training, but features are built from candidate residuals, slot0
reference corrections, slow base trajectories, past trajectories, and temporal
interaction energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn


EPS = 1e-8


@dataclass
class V58SlotQualityScorerConfig:
    feature_dim: int
    hidden_dim: int = 192
    layers: int = 3
    dropout: float = 0.05
    output_dim: int = 1
    rank_head_index: int = 0
    accept_head_index: int = 1


def _trajectory_velocity(traj: torch.Tensor) -> torch.Tensor:
    velocity = torch.cat([torch.zeros_like(traj[..., :1, :]), traj[..., 1:, :] - traj[..., :-1, :]], dim=-2)
    return torch.cat([traj, velocity], dim=-1)


def _path_length(traj: torch.Tensor) -> torch.Tensor:
    if int(traj.shape[-2]) <= 1:
        return torch.linalg.norm(traj[..., -1, :], dim=-1)
    steps = torch.linalg.norm(traj[..., 1:, :] - traj[..., :-1, :], dim=-1)
    return steps.sum(dim=-1)


def _base_direction(base: torch.Tensor) -> torch.Tensor:
    direction = base[..., -1, :] - base[..., 0, :]
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)


def _past_motion_summary(past: torch.Tensor) -> torch.Tensor:
    xy = past[..., :2]
    if int(xy.shape[-2]) <= 1:
        return xy.new_zeros(*xy.shape[:2], 8)
    disp = xy[..., -1, :] - xy[..., 0, :]
    disp_norm = torch.linalg.norm(disp, dim=-1)
    steps = xy[..., 1:, :] - xy[..., :-1, :]
    step_norm = torch.linalg.norm(steps, dim=-1)
    path = step_norm.sum(dim=-1)
    mean_step = step_norm.mean(dim=-1)
    max_step = step_norm.amax(dim=-1)
    straightness = disp_norm / path.clamp_min(EPS)
    if int(steps.shape[-2]) > 1:
        v1 = steps[..., :-1, :]
        v2 = steps[..., 1:, :]
        denom = torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)
        cos = (v1 * v2).sum(dim=-1) / denom.clamp_min(EPS)
        turn = (1.0 - cos.clamp(-1.0, 1.0)).mean(dim=-1)
    else:
        turn = torch.zeros_like(disp_norm)
    return torch.stack([disp[..., 0], disp[..., 1], disp_norm, path, mean_step, max_step, straightness, turn], dim=-1)


def _ensure_energy(energy: Optional[torch.Tensor], candidates: torch.Tensor, energy_dim: int = 5) -> torch.Tensor:
    batch_size, _num_slots, num_modes, num_agents = candidates.shape[:4]
    future_frames = int(candidates.shape[-2])
    if energy is None:
        return candidates.new_zeros(batch_size, num_modes, num_agents, future_frames, int(energy_dim))
    value = energy.to(device=candidates.device, dtype=candidates.dtype)
    if value.ndim == 4:
        value = value[:, :, :, None, :].expand(batch_size, num_modes, num_agents, future_frames, int(value.shape[-1]))
    if value.ndim != 5:
        raise ValueError(f"temporal_energy_features must be [B,K,A,T,C] or [B,K,A,C], got {tuple(value.shape)}")
    if int(value.shape[1]) == 1 and num_modes > 1:
        value = value.expand(batch_size, num_modes, num_agents, int(value.shape[3]), int(value.shape[4]))
    if tuple(value.shape[:3]) != (batch_size, num_modes, num_agents):
        raise ValueError(f"energy/candidate shape mismatch: energy={tuple(value.shape)} candidates={tuple(candidates.shape)}")
    if int(value.shape[3]) != future_frames:
        if int(value.shape[3]) == 1:
            value = value.expand(batch_size, num_modes, num_agents, future_frames, int(value.shape[-1]))
        else:
            raise ValueError(f"energy future length mismatch: energy={tuple(value.shape)} candidates={tuple(candidates.shape)}")
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _flatten_names(prefix: str, length: int) -> List[str]:
    return [f"{prefix}_{index:02d}" for index in range(int(length))]


def v58_slot_quality_feature_names(
    *,
    future_frames: int = 12,
    coord_dim: int = 2,
    past_frames: int = 8,
    past_feature_dim: int = 6,
    temporal_energy_dim: int = 5,
    include_index_features: bool = False,
) -> List[str]:
    flat = int(future_frames) * int(coord_dim)
    velocity_flat = int(future_frames) * int(coord_dim) * 2
    names: List[str] = []
    names += _flatten_names("residual_xy", flat)
    names += _flatten_names("residual_xy_velocity", velocity_flat)
    names += _flatten_names("residual_minus_slot0_xy", flat)
    names += _flatten_names("base_xy_velocity", velocity_flat)
    names += _flatten_names("slot0_residual_xy", flat)
    names += _flatten_names("past_flat", int(past_frames) * int(past_feature_dim))
    names += _flatten_names("energy_flat", int(future_frames) * int(temporal_energy_dim))
    names += [
        "endpoint_dx",
        "endpoint_dy",
        "endpoint_norm",
        "trajectory_norm_mean",
        "trajectory_norm_max",
        "mean_dx",
        "mean_dy",
        "smoothness",
        "acceleration",
        "forward_endpoint",
        "lateral_endpoint",
        "abs_lateral_endpoint",
        "early_norm",
        "late_norm",
        "late_minus_early_norm",
        "slot0_endpoint_norm",
        "delta_endpoint_vs_slot0",
        "base_endpoint_norm",
        "base_path_norm",
        "candidate_path_norm",
        "candidate_to_base_path_ratio",
        "base_endpoint_spread",
        "base_path_spread",
    ]
    names += [
        "past_disp_x",
        "past_disp_y",
        "past_disp_norm",
        "past_path",
        "past_mean_step",
        "past_max_step",
        "past_straightness",
        "past_turn",
    ]
    names += [
        "energy_mean_min_neighbor_distance",
        "energy_mean_soft_collision",
        "energy_mean_close_neighbor_count",
        "energy_mean_approaching",
        "energy_mean_endpoint_crowding",
        "energy_max_soft_collision",
        "energy_max_close_neighbor_count",
        "energy_final_endpoint_crowding",
    ]
    if include_index_features:
        names += ["slot_index_norm", "slot_is_zero", "base_mode_index_norm"]
    return names


def build_v58_slot_quality_features(
    candidates: torch.Tensor,
    *,
    base_trajectory: torch.Tensor,
    past_traj_original_scale: torch.Tensor,
    temporal_energy_features: Optional[torch.Tensor] = None,
    candidate_slot_ids: Optional[torch.Tensor] = None,
    max_slot_id: Optional[int] = None,
    include_index_features: bool = False,
) -> torch.Tensor:
    if candidates.ndim != 6:
        raise ValueError(f"candidates must be [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    if base_trajectory.ndim != 5:
        raise ValueError(f"base_trajectory must be [B,K,A,T,2], got {tuple(base_trajectory.shape)}")
    if tuple(base_trajectory.shape[:3]) != (int(candidates.shape[0]), int(candidates.shape[2]), int(candidates.shape[3])):
        raise ValueError(f"base/candidate shape mismatch: base={tuple(base_trajectory.shape)} candidates={tuple(candidates.shape)}")
    if past_traj_original_scale.ndim != 4:
        raise ValueError(f"past_traj_original_scale must be [B,A,P,C], got {tuple(past_traj_original_scale.shape)}")
    batch_size, num_slots, num_modes, num_agents, future_frames, coord_dim = candidates.shape
    if int(coord_dim) != 2:
        raise ValueError(f"Expected coord_dim=2, got {coord_dim}")
    base = base_trajectory.to(device=candidates.device, dtype=candidates.dtype)
    past = past_traj_original_scale.to(device=candidates.device, dtype=candidates.dtype)
    energy = _ensure_energy(temporal_energy_features, candidates)
    if candidate_slot_ids is None:
        candidate_slot_ids = torch.arange(num_slots, device=candidates.device, dtype=torch.long)
    else:
        candidate_slot_ids = candidate_slot_ids.to(device=candidates.device, dtype=torch.long)
    if int(candidate_slot_ids.numel()) != int(num_slots):
        raise ValueError(f"candidate_slot_ids length {candidate_slot_ids.numel()} does not match S={num_slots}")
    slot0_positions = (candidate_slot_ids == 0).nonzero(as_tuple=False).reshape(-1)
    if int(slot0_positions.numel()) <= 0:
        raise ValueError("candidate_slot_ids must include slot0 so quality is scored relative to the safety fallback")
    slot0_pos = int(slot0_positions[0].item())

    base_exp = base[:, None, ...].expand_as(candidates)
    residual = candidates - base_exp
    slot0_residual = residual[:, slot0_pos : slot0_pos + 1, ...].expand_as(residual)
    residual_vs_slot0 = residual - slot0_residual

    residual_norm = torch.linalg.norm(residual, dim=-1)
    endpoint = residual[..., -1, :]
    endpoint_norm = torch.linalg.norm(endpoint, dim=-1)
    trajectory_norm_mean = residual_norm.mean(dim=-1)
    trajectory_norm_max = residual_norm.amax(dim=-1)
    mean_delta = residual.mean(dim=-2)
    diff = residual[..., 1:, :] - residual[..., :-1, :]
    smoothness = torch.linalg.norm(diff, dim=-1).mean(dim=-1)
    if future_frames >= 3:
        accel = diff[..., 1:, :] - diff[..., :-1, :]
        acceleration = torch.linalg.norm(accel, dim=-1).mean(dim=-1)
    else:
        acceleration = torch.zeros_like(smoothness)
    direction = _base_direction(base)
    perp = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    forward = (endpoint * direction[:, None, ...]).sum(dim=-1)
    lateral = (endpoint * perp[:, None, ...]).sum(dim=-1)
    half = max(int(future_frames) // 2, 1)
    early_norm = residual_norm[..., :half].mean(dim=-1)
    late_norm = residual_norm[..., half:].mean(dim=-1)

    slot0_endpoint_norm = torch.linalg.norm(slot0_residual[..., -1, :], dim=-1)
    delta_endpoint_vs_slot0 = endpoint_norm - slot0_endpoint_norm
    base_endpoint = torch.linalg.norm(base[..., -1, :] - base[..., 0, :], dim=-1)
    base_path = _path_length(base)
    candidate_path = _path_length(candidates)
    base_endpoint_mean = base_endpoint.mean(dim=1, keepdim=True)
    base_path_mean = base_path.mean(dim=1, keepdim=True)
    base_endpoint_spread = (base_endpoint - base_endpoint_mean).abs()[:, None, ...].expand(
        batch_size, num_slots, num_modes, num_agents
    )
    base_path_spread = (base_path - base_path_mean).abs()[:, None, ...].expand_as(base_endpoint_spread)

    past_flat = past.reshape(batch_size, num_agents, -1)[:, None, None, :, :].expand(
        batch_size, num_slots, num_modes, num_agents, -1
    )
    past_summary = _past_motion_summary(past)[:, None, None, :, :].expand(batch_size, num_slots, num_modes, num_agents, -1)
    energy_flat = energy.reshape(batch_size, num_modes, num_agents, -1)[:, None, ...].expand(
        batch_size, num_slots, num_modes, num_agents, -1
    )
    if int(energy.shape[-1]) >= 5:
        e0 = energy[..., 0].clamp_min(0.0)
        e1 = energy[..., 1].clamp_min(0.0)
        e2 = energy[..., 2].clamp_min(0.0)
        e3 = energy[..., 3].clamp(0.0, 1.0)
        e4 = energy[..., 4].clamp_min(0.0)
        energy_summary_base = torch.stack(
            [
                e0.mean(dim=-1),
                e1.mean(dim=-1),
                e2.mean(dim=-1),
                e3.mean(dim=-1),
                e4.mean(dim=-1),
                e1.amax(dim=-1),
                e2.amax(dim=-1),
                e4[..., -1],
            ],
            dim=-1,
        )
    else:
        mean = energy.mean(dim=-2)
        pad = mean.new_zeros(*mean.shape[:-1], max(8 - int(mean.shape[-1]), 0))
        energy_summary_base = torch.cat([mean, pad], dim=-1)[..., :8]
    energy_summary = energy_summary_base[:, None, ...].expand(batch_size, num_slots, num_modes, num_agents, 8)

    scalar = torch.stack(
        [
            endpoint[..., 0],
            endpoint[..., 1],
            endpoint_norm,
            trajectory_norm_mean,
            trajectory_norm_max,
            mean_delta[..., 0],
            mean_delta[..., 1],
            smoothness,
            acceleration,
            forward,
            lateral,
            lateral.abs(),
            early_norm,
            late_norm,
            late_norm - early_norm,
            slot0_endpoint_norm,
            delta_endpoint_vs_slot0,
            base_endpoint[:, None, ...].expand(batch_size, num_slots, num_modes, num_agents),
            base_path[:, None, ...].expand(batch_size, num_slots, num_modes, num_agents),
            candidate_path,
            candidate_path / base_path[:, None, ...].clamp_min(EPS),
            base_endpoint_spread,
            base_path_spread,
        ],
        dim=-1,
    )

    pieces = [
        residual.reshape(batch_size, num_slots, num_modes, num_agents, -1),
        _trajectory_velocity(residual).reshape(batch_size, num_slots, num_modes, num_agents, -1),
        residual_vs_slot0.reshape(batch_size, num_slots, num_modes, num_agents, -1),
        _trajectory_velocity(base)[:, None, ...].expand(batch_size, num_slots, num_modes, num_agents, future_frames, 4).reshape(
            batch_size,
            num_slots,
            num_modes,
            num_agents,
            -1,
        ),
        slot0_residual.reshape(batch_size, num_slots, num_modes, num_agents, -1),
        past_flat,
        energy_flat,
        scalar,
        past_summary,
        energy_summary,
    ]
    if include_index_features:
        max_slot = max(int(max_slot_id if max_slot_id is not None else candidate_slot_ids.max().item()), 1)
        slot_norm = (candidate_slot_ids.to(dtype=candidates.dtype) / float(max_slot))[None, :, None, None]
        slot_norm = slot_norm.expand(batch_size, num_slots, num_modes, num_agents)
        slot_zero = (candidate_slot_ids == 0).to(dtype=candidates.dtype)[None, :, None, None].expand_as(slot_norm)
        mode_index = torch.arange(num_modes, device=candidates.device, dtype=candidates.dtype)
        mode_norm = (mode_index / float(max(num_modes - 1, 1)))[None, None, :, None].expand_as(slot_norm)
        pieces.append(torch.stack([slot_norm, slot_zero, mode_norm], dim=-1))
    features = torch.cat(pieces, dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class V58SlotQualityScorer(nn.Module):
    """Binary scorer estimating whether a nonzero slot is safe and useful vs slot0."""

    def __init__(self, config: V58SlotQualityScorerConfig) -> None:
        super().__init__()
        self.config = config
        layers: List[nn.Module] = [nn.LayerNorm(int(config.feature_dim))]
        in_dim = int(config.feature_dim)
        hidden_dim = int(config.hidden_dim)
        for _index in range(max(int(config.layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if float(config.dropout) > 0.0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(config.output_dim)))
        self.network = nn.Sequential(*layers)
        self.register_buffer("feature_mean", torch.zeros(int(config.feature_dim), dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(int(config.feature_dim), dtype=torch.float32))

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if int(mean.numel()) != int(self.config.feature_dim) or int(std.numel()) != int(self.config.feature_dim):
            raise ValueError("feature normalization size does not match model feature_dim")
        self.feature_mean.copy_(mean.detach().to(device=self.feature_mean.device, dtype=self.feature_mean.dtype).reshape(-1))
        self.feature_std.copy_(std.detach().to(device=self.feature_std.device, dtype=self.feature_std.dtype).reshape(-1).clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = (features.to(dtype=self.feature_mean.dtype) - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        output = self.network(x)
        if int(self.config.output_dim) == 1:
            return output.squeeze(-1)
        return output

    def rank_logits(self, features: torch.Tensor) -> torch.Tensor:
        output = self.forward(features)
        if int(self.config.output_dim) == 1:
            return output
        return output[..., int(self.config.rank_head_index)]

    def accept_logits(self, features: torch.Tensor) -> torch.Tensor:
        output = self.forward(features)
        if int(self.config.output_dim) == 1:
            return output
        return output[..., int(self.config.accept_head_index)]

    @torch.no_grad()
    def score_candidates(
        self,
        candidates: torch.Tensor,
        *,
        base_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_slot_ids: Optional[torch.Tensor] = None,
        max_slot_id: Optional[int] = None,
        include_index_features: bool = False,
    ) -> torch.Tensor:
        features = build_v58_slot_quality_features(
            candidates,
            base_trajectory=base_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            candidate_slot_ids=candidate_slot_ids,
            max_slot_id=max_slot_id,
            include_index_features=include_index_features,
        )
        return self.rank_logits(features)

    @torch.no_grad()
    def score_candidate_outputs(
        self,
        candidates: torch.Tensor,
        *,
        base_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_slot_ids: Optional[torch.Tensor] = None,
        max_slot_id: Optional[int] = None,
        include_index_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = build_v58_slot_quality_features(
            candidates,
            base_trajectory=base_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            candidate_slot_ids=candidate_slot_ids,
            max_slot_id=max_slot_id,
            include_index_features=include_index_features,
        )
        output = self.forward(features)
        if int(self.config.output_dim) == 1:
            return {"rank_logits": output, "accept_logits": output}
        return {
            "rank_logits": output[..., int(self.config.rank_head_index)],
            "accept_logits": output[..., int(self.config.accept_head_index)],
            "raw_outputs": output,
        }


def load_v58_slot_quality_scorer(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[V58SlotQualityScorer, Dict[str, Any]]:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    config = V58SlotQualityScorerConfig(**dict(checkpoint["config"]))
    model = V58SlotQualityScorer(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain model weights: {path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict(checkpoint)


__all__ = [
    "V58SlotQualityScorer",
    "V58SlotQualityScorerConfig",
    "build_v58_slot_quality_features",
    "load_v58_slot_quality_scorer",
    "v58_slot_quality_feature_names",
]
