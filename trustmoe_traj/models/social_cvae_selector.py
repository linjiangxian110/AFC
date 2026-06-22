"""Candidate selector for SocialCVAE residual expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SocialCVAEGroupSelectorConfig:
    future_frames: int = 12
    coord_dim: int = 2
    past_frames: int = 8
    past_feature_dim: int = 6
    temporal_energy_dim: int = 5
    candidate_energy_summary_dim: int = 0
    hidden_dim: int = 128
    max_modes: int = 20
    use_mode_embedding: bool = True
    use_energy_risk_map: bool = False
    energy_risk_distance_scale: float = 0.5
    use_temporal_energy_encoder: bool = False
    energy_temporal_hidden_dim: int = 64
    use_mean_candidate_comparison: bool = False
    use_candidate_energy_context: bool = False
    use_candidate_energy_summary_context: bool = False
    use_energy_gated_fusion: bool = False
    use_candidate_safety_penalty: bool = False
    candidate_safety_penalty_strength: float = 1.0
    use_residual_accept_gate: bool = False
    residual_accept_gate_strength: float = 1.0
    use_base_best_guard: bool = False
    base_best_guard_strength: float = 1.0
    use_observable_feature_context: bool = False
    observable_feature_dim: int = 25


def _traj_with_velocity(traj: torch.Tensor) -> torch.Tensor:
    velocity = torch.cat([torch.zeros_like(traj[..., :1, :]), traj[..., 1:, :] - traj[..., :-1, :]], dim=-2)
    return torch.cat([traj, velocity], dim=-1).reshape(*traj.shape[:-2], traj.shape[-2] * traj.shape[-1] * 2)


def _ensure_candidates(tensor: torch.Tensor, cfg: SocialCVAEGroupSelectorConfig) -> torch.Tensor:
    if tensor.ndim != 6:
        raise ValueError(f"candidate_trajectory must have shape [B,S,K,A,T,2], got {tuple(tensor.shape)}")
    if int(tensor.shape[-2]) != int(cfg.future_frames) or int(tensor.shape[-1]) != int(cfg.coord_dim):
        raise ValueError(
            "candidate_trajectory future shape mismatch: "
            f"got trailing {tuple(tensor.shape[-2:])}, expected ({cfg.future_frames}, {cfg.coord_dim})"
        )
    return tensor


def _ensure_base(tensor: torch.Tensor, candidates: torch.Tensor, cfg: SocialCVAEGroupSelectorConfig) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"base_trajectory must have shape [B,K,A,T,2], got {tuple(tensor.shape)}")
    if tuple(tensor.shape[:3]) != (int(candidates.shape[0]), int(candidates.shape[2]), int(candidates.shape[3])):
        raise ValueError(f"base/candidate shape mismatch: base={tuple(tensor.shape)} candidates={tuple(candidates.shape)}")
    if int(tensor.shape[-2]) != int(cfg.future_frames) or int(tensor.shape[-1]) != int(cfg.coord_dim):
        raise ValueError(
            "base_trajectory future shape mismatch: "
            f"got trailing {tuple(tensor.shape[-2:])}, expected ({cfg.future_frames}, {cfg.coord_dim})"
        )
    return tensor.to(device=candidates.device, dtype=candidates.dtype)


def _ensure_past(
    tensor: torch.Tensor,
    candidates: torch.Tensor,
    cfg: SocialCVAEGroupSelectorConfig,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"past_traj_original_scale must have shape [B,A,P,C], got {tuple(tensor.shape)}")
    if tuple(tensor.shape[:2]) != (int(candidates.shape[0]), int(candidates.shape[3])):
        raise ValueError(f"past/candidate shape mismatch: past={tuple(tensor.shape)} candidates={tuple(candidates.shape)}")
    if int(tensor.shape[-2]) != int(cfg.past_frames) or int(tensor.shape[-1]) != int(cfg.past_feature_dim):
        raise ValueError(
            "past_traj_original_scale shape mismatch: "
            f"got trailing {tuple(tensor.shape[-2:])}, expected ({cfg.past_frames}, {cfg.past_feature_dim})"
        )
    return tensor.to(device=candidates.device, dtype=candidates.dtype)


def _ensure_energy(
    tensor: Optional[torch.Tensor],
    candidates: torch.Tensor,
    cfg: SocialCVAEGroupSelectorConfig,
) -> torch.Tensor:
    batch_size, _num_samples, num_modes, num_agents = candidates.shape[:4]
    if tensor is None:
        return candidates.new_zeros(
            batch_size,
            num_modes,
            num_agents,
            int(cfg.future_frames),
            int(cfg.temporal_energy_dim),
        )
    energy = tensor.to(device=candidates.device, dtype=candidates.dtype)
    if energy.ndim == 4:
        energy = energy[:, :, :, None, :].expand(
            batch_size,
            num_modes,
            num_agents,
            int(cfg.future_frames),
            int(energy.shape[-1]),
        )
    if energy.ndim != 5:
        raise ValueError(f"temporal_energy_features must have shape [B,K,A,T,C], got {tuple(energy.shape)}")
    if int(energy.shape[1]) == 1 and int(num_modes) > 1:
        energy = energy.expand(batch_size, num_modes, num_agents, int(energy.shape[3]), int(energy.shape[4]))
    if tuple(energy.shape[:3]) != (batch_size, num_modes, num_agents):
        raise ValueError(f"energy/candidate shape mismatch: energy={tuple(energy.shape)} candidates={tuple(candidates.shape)}")
    if int(energy.shape[3]) != int(cfg.future_frames) or int(energy.shape[-1]) != int(cfg.temporal_energy_dim):
        raise ValueError(
            "temporal_energy_features shape mismatch: "
            f"got trailing {tuple(energy.shape[3:])}, expected ({cfg.future_frames}, {cfg.temporal_energy_dim})"
        )
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_candidate_energy(
    tensor: Optional[torch.Tensor],
    candidates: torch.Tensor,
    cfg: SocialCVAEGroupSelectorConfig,
) -> Optional[torch.Tensor]:
    if tensor is None:
        if bool(cfg.use_candidate_energy_context):
            raise ValueError("use_candidate_energy_context requires candidate_temporal_energy_features")
        return None
    energy = tensor.to(device=candidates.device, dtype=candidates.dtype)
    if energy.ndim != 6:
        raise ValueError(
            "candidate_temporal_energy_features must have shape [B,S,K,A,T,C], "
            f"got {tuple(energy.shape)}"
        )
    if tuple(energy.shape[:5]) != tuple(candidates.shape[:5]):
        raise ValueError(
            "candidate energy/candidate shape mismatch: "
            f"energy={tuple(energy.shape)} candidates={tuple(candidates.shape)}"
        )
    if int(energy.shape[-1]) != int(cfg.temporal_energy_dim):
        raise ValueError(
            "candidate temporal energy dim mismatch: "
            f"got {int(energy.shape[-1])}, expected {cfg.temporal_energy_dim}"
        )
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_candidate_energy_summary(
    tensor: Optional[torch.Tensor],
    candidates: torch.Tensor,
    cfg: SocialCVAEGroupSelectorConfig,
) -> Optional[torch.Tensor]:
    if tensor is None:
        if bool(cfg.use_candidate_energy_summary_context):
            raise ValueError("use_candidate_energy_summary_context requires candidate_energy_summary_features")
        return None
    summary = tensor.to(device=candidates.device, dtype=candidates.dtype)
    if summary.ndim != 5:
        raise ValueError(
            "candidate_energy_summary_features must have shape [B,S,K,A,C], "
            f"got {tuple(summary.shape)}"
        )
    if tuple(summary.shape[:4]) != tuple(candidates.shape[:4]):
        raise ValueError(
            "candidate energy summary/candidate shape mismatch: "
            f"summary={tuple(summary.shape)} candidates={tuple(candidates.shape)}"
        )
    expected_dim = int(cfg.candidate_energy_summary_dim) if int(cfg.candidate_energy_summary_dim) > 0 else int(
        cfg.temporal_energy_dim
    )
    if int(summary.shape[-1]) != expected_dim:
        raise ValueError(
            "candidate energy summary dim mismatch: "
            f"got {int(summary.shape[-1])}, expected {expected_dim}"
        )
    return torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def _energy_with_risk_map(energy: torch.Tensor, cfg: SocialCVAEGroupSelectorConfig) -> torch.Tensor:
    if not bool(cfg.use_energy_risk_map):
        return energy
    if int(energy.shape[-1]) < 5:
        raise ValueError(
            "use_energy_risk_map requires temporal energy with at least 5 channels, "
            f"got {int(energy.shape[-1])}"
        )
    raw = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
    distance_scale = max(float(cfg.energy_risk_distance_scale), 1e-6)
    min_neighbor_distance = raw[..., 0:1].clamp_min(0.0)
    soft_collision_energy = raw[..., 1:2].clamp_min(0.0)
    close_neighbor_count = raw[..., 2:3].clamp_min(0.0)
    approaching_score = raw[..., 3:4].clamp(0.0, 1.0)
    endpoint_crowding_energy = raw[..., 4:5].clamp_min(0.0)
    risk = torch.cat(
        [
            torch.exp(-min_neighbor_distance / distance_scale),
            soft_collision_energy / (1.0 + soft_collision_energy),
            close_neighbor_count / (1.0 + close_neighbor_count),
            approaching_score,
            endpoint_crowding_energy / (1.0 + endpoint_crowding_energy),
        ],
        dim=-1,
    )
    return torch.cat([raw, risk], dim=-1)


def _path_length(traj: torch.Tensor) -> torch.Tensor:
    first = torch.linalg.norm(traj[..., :1, :], dim=-1)
    if int(traj.shape[-2]) <= 1:
        return first.squeeze(-1)
    steps = torch.linalg.norm(traj[..., 1:, :] - traj[..., :-1, :], dim=-1)
    return torch.cat([first, steps], dim=-1).sum(dim=-1)


def _base_direction(base: torch.Tensor) -> torch.Tensor:
    direction = base[..., -1, :] - base[..., 0, :]
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, direction / norm.clamp_min(1.0e-6), fallback)


def _past_motion_summary(past: torch.Tensor) -> torch.Tensor:
    xy = past[..., :2]
    if int(xy.shape[-2]) <= 1:
        return xy.new_zeros(*xy.shape[:2], 6)
    displacement = torch.linalg.norm(xy[..., -1, :] - xy[..., 0, :], dim=-1)
    steps = xy[..., 1:, :] - xy[..., :-1, :]
    step_norm = torch.linalg.norm(steps, dim=-1)
    path = step_norm.sum(dim=-1)
    mean_step = step_norm.mean(dim=-1)
    max_step = step_norm.amax(dim=-1)
    straightness = displacement / path.clamp_min(1.0e-8)
    if int(steps.shape[-2]) > 1:
        v1 = steps[..., :-1, :]
        v2 = steps[..., 1:, :]
        denom = torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)
        cos = (v1 * v2).sum(dim=-1) / denom.clamp_min(1.0e-8)
        turn = (1.0 - cos.clamp(-1.0, 1.0)).mean(dim=-1)
    else:
        turn = torch.zeros_like(displacement)
    return torch.stack([displacement, path, mean_step, max_step, straightness, turn], dim=-1)


def _energy_summary_observable(energy: torch.Tensor) -> torch.Tensor:
    if int(energy.shape[-1]) < 5:
        mean = energy.mean(dim=-2)
        if int(mean.shape[-1]) >= 6:
            return mean[..., :6]
        pad = mean.new_zeros(*mean.shape[:-1], 6 - int(mean.shape[-1]))
        return torch.cat([mean, pad], dim=-1)
    min_neighbor_distance = energy[..., 0].clamp_min(0.0)
    soft_collision_energy = energy[..., 1].clamp_min(0.0)
    close_neighbor_count = energy[..., 2].clamp_min(0.0)
    approaching_score = energy[..., 3].clamp(0.0, 1.0)
    endpoint_crowding_energy = energy[..., 4].clamp_min(0.0)
    risk = torch.stack(
        [
            torch.exp(-min_neighbor_distance / 0.5),
            soft_collision_energy / (1.0 + soft_collision_energy),
            close_neighbor_count / (1.0 + close_neighbor_count),
            approaching_score,
            endpoint_crowding_energy / (1.0 + endpoint_crowding_energy),
        ],
        dim=0,
    ).amax(dim=0)
    return torch.stack(
        [
            risk.mean(dim=-1),
            risk.amax(dim=-1),
            min_neighbor_distance.mean(dim=-1),
            close_neighbor_count.mean(dim=-1),
            approaching_score.mean(dim=-1),
            endpoint_crowding_energy.mean(dim=-1),
        ],
        dim=-1,
    )


def _observable_features(
    candidates: torch.Tensor,
    base: torch.Tensor,
    past: torch.Tensor,
    energy: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_samples, num_modes, num_agents = candidates.shape[:4]
    residual = candidates - base[:, None, ...]
    direction = _base_direction(base)
    perp = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    endpoint = residual[..., -1, :]
    forward = (endpoint * direction[:, None, ...]).sum(dim=-1)
    lateral = (endpoint * perp[:, None, ...]).sum(dim=-1)
    residual_endpoint = torch.linalg.norm(endpoint, dim=-1)
    residual_traj = torch.linalg.norm(residual, dim=-1).mean(dim=-1)
    base_path = _path_length(base)
    base_endpoint = torch.linalg.norm(base[..., -1, :], dim=-1)
    candidate_path = _path_length(candidates)
    past_summary = _past_motion_summary(past)[:, None, None, :, :].expand(batch_size, num_samples, num_modes, num_agents, 6)
    energy_summary = _energy_summary_observable(energy)[:, None, ...].expand(batch_size, num_samples, num_modes, num_agents, 6)
    sample = torch.arange(num_samples, device=candidates.device, dtype=candidates.dtype)
    sample_scale = max(num_samples - 1, 1)
    sample_norm = (sample / float(sample_scale))[None, :, None, None].expand(batch_size, num_samples, num_modes, num_agents)
    sample_is_reference = (sample == 0).to(dtype=candidates.dtype)[None, :, None, None].expand_as(sample_norm)
    base_path_exp = base_path[:, None, :, :].expand(batch_size, num_samples, num_modes, num_agents)
    base_endpoint_exp = base_endpoint[:, None, :, :].expand_as(base_path_exp)
    ratio_base = base_path_exp.clamp_min(1.0e-6)
    scalar_features = torch.stack(
        [
            base_endpoint_exp,
            base_path_exp,
            candidate_path,
            candidate_path / ratio_base,
            residual_endpoint,
            residual_traj,
            residual_endpoint / ratio_base,
            residual_traj / ratio_base,
            forward,
            lateral,
            lateral.abs(),
            sample_norm,
            sample_is_reference,
        ],
        dim=-1,
    )
    features = torch.cat([scalar_features, past_summary, energy_summary], dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class SocialCVAEGroupSelector(nn.Module):
    """Score sampled residual candidates within each original teacher mode."""

    def __init__(self, config: Optional[SocialCVAEGroupSelectorConfig] = None) -> None:
        super().__init__()
        self.config = config or SocialCVAEGroupSelectorConfig()
        cfg = self.config
        if bool(cfg.use_candidate_energy_context) and bool(cfg.use_candidate_energy_summary_context):
            raise ValueError("candidate energy context and summary context are mutually exclusive")
        flat_dim = int(cfg.future_frames * cfg.coord_dim)
        traj_feature_dim = int(flat_dim * 2)
        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        if bool(cfg.use_energy_risk_map) and int(cfg.temporal_energy_dim) < 5:
            raise ValueError("use_energy_risk_map requires temporal_energy_dim >= 5")
        energy_feature_dim = int(cfg.temporal_energy_dim) + (5 if bool(cfg.use_energy_risk_map) else 0)
        self.candidate_energy_summary_dim = (
            int(cfg.candidate_energy_summary_dim) if int(cfg.candidate_energy_summary_dim) > 0 else int(cfg.temporal_energy_dim)
        )
        candidate_energy_summary_feature_dim = int(self.candidate_energy_summary_dim) + (
            5 if bool(cfg.use_energy_risk_map) else 0
        )
        energy_flat_dim = int(cfg.future_frames * energy_feature_dim)
        hidden_dim = int(cfg.hidden_dim)

        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(traj_feature_dim),
            nn.Linear(traj_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(traj_feature_dim),
            nn.Linear(traj_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.past_encoder = nn.Sequential(
            nn.LayerNorm(past_flat_dim),
            nn.Linear(past_flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_token_encoder: Optional[nn.Module]
        self.energy_temporal_conv: Optional[nn.Module]
        if bool(cfg.use_temporal_energy_encoder):
            temporal_hidden_dim = int(cfg.energy_temporal_hidden_dim)
            if temporal_hidden_dim <= 0:
                raise ValueError(f"energy_temporal_hidden_dim must be positive, got {temporal_hidden_dim}")
            self.energy_token_encoder = nn.Sequential(
                nn.LayerNorm(energy_feature_dim),
                nn.Linear(energy_feature_dim, temporal_hidden_dim),
                nn.SiLU(),
                nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
                nn.SiLU(),
            )
            self.energy_temporal_conv = nn.Sequential(
                nn.Conv1d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.energy_encoder = nn.Sequential(
                nn.LayerNorm(temporal_hidden_dim * 3),
                nn.Linear(temporal_hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.energy_token_encoder = None
            self.energy_temporal_conv = None
            self.energy_encoder = nn.Sequential(
                nn.LayerNorm(energy_flat_dim),
                nn.Linear(energy_flat_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        if bool(cfg.use_mean_candidate_comparison):
            self.mean_delta_encoder: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(flat_dim),
                nn.Linear(flat_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.mean_delta_encoder = None
        if bool(cfg.use_candidate_energy_summary_context):
            self.candidate_energy_summary_encoder: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(candidate_energy_summary_feature_dim * 3),
                nn.Linear(candidate_energy_summary_feature_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.candidate_energy_summary_encoder = None
        if bool(cfg.use_observable_feature_context):
            self.observable_feature_encoder: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(int(cfg.observable_feature_dim)),
                nn.Linear(int(cfg.observable_feature_dim), hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.observable_feature_encoder = None
        if bool(cfg.use_energy_gated_fusion):
            self.energy_fusion_gate: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim * 5),
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.energy_fusion_gate = None
        if bool(cfg.use_candidate_safety_penalty):
            if float(cfg.candidate_safety_penalty_strength) <= 0.0:
                raise ValueError(
                    "candidate_safety_penalty_strength must be positive when candidate safety penalty is enabled"
                )
            self.candidate_safety_penalty_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim * 5),
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.candidate_safety_penalty_head = None
        if bool(cfg.use_residual_accept_gate):
            if float(cfg.residual_accept_gate_strength) <= 0.0:
                raise ValueError("residual_accept_gate_strength must be positive when residual accept gate is enabled")
            self.residual_accept_gate_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim * 15),
                nn.Linear(hidden_dim * 15, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.residual_accept_gate_head = None
        if bool(cfg.use_base_best_guard):
            if float(cfg.base_best_guard_strength) <= 0.0:
                raise ValueError("base_best_guard_strength must be positive when base-best guard is enabled")
            self.base_best_guard_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim * 5),
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.base_best_guard_head = None
        if bool(cfg.use_mode_embedding):
            self.mode_embedding: Optional[nn.Embedding] = nn.Embedding(int(cfg.max_modes), hidden_dim)
        else:
            self.mode_embedding = None

        mode_dim = hidden_dim if self.mode_embedding is not None else 0
        mean_delta_dim = hidden_dim if self.mean_delta_encoder is not None else 0
        observable_dim = hidden_dim if self.observable_feature_encoder is not None else 0
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5 + mode_dim + mean_delta_dim + observable_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def _encode_energy(self, energy: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        energy_features = _energy_with_risk_map(energy, cfg)
        if self.energy_token_encoder is None or self.energy_temporal_conv is None:
            return self.energy_encoder(energy_features.reshape(*energy_features.shape[:-2], -1))

        leading_shape = energy_features.shape[:-2]
        num_steps = int(energy_features.shape[-2])
        tokens = self.energy_token_encoder(energy_features)
        flat_tokens = tokens.reshape(-1, num_steps, tokens.shape[-1]).transpose(1, 2)
        encoded = self.energy_temporal_conv(flat_tokens).transpose(1, 2)
        encoded = encoded.reshape(*leading_shape, num_steps, -1)
        pooled = torch.cat(
            [
                encoded.mean(dim=-2),
                encoded.amax(dim=-2),
                encoded[..., -1, :],
            ],
            dim=-1,
        )
        return self.energy_encoder(pooled)

    @staticmethod
    def _temporal_energy_summary(energy: torch.Tensor) -> torch.Tensor:
        if energy.ndim != 5:
            raise ValueError(f"temporal energy must have shape [B,K,A,T,C], got {tuple(energy.shape)}")
        if int(energy.shape[-1]) < 5:
            return energy.mean(dim=-2)
        min_neighbor_distance = energy[..., 0:1].amin(dim=-2)
        soft_collision_energy = energy[..., 1:2].mean(dim=-2)
        close_neighbor_count = energy[..., 2:3].mean(dim=-2)
        approaching_score = energy[..., 3:4].amax(dim=-2)
        endpoint_crowding_energy = energy[..., -1:, 4]
        pieces = [
            min_neighbor_distance,
            soft_collision_energy,
            close_neighbor_count,
            approaching_score,
            endpoint_crowding_energy,
        ]
        if int(energy.shape[-1]) > 5:
            pieces.append(energy[..., 5:].mean(dim=-2))
        return torch.cat(pieces, dim=-1)

    def _match_candidate_summary_dim(self, summary: torch.Tensor) -> torch.Tensor:
        expected_dim = int(self.candidate_energy_summary_dim)
        current_dim = int(summary.shape[-1])
        if current_dim == expected_dim:
            return summary
        if current_dim > expected_dim:
            return summary[..., :expected_dim]
        pad_width = expected_dim - current_dim
        pad_shape = (*summary.shape[:-1], pad_width)
        pad = summary.new_zeros(pad_shape)
        return torch.cat([summary, pad], dim=-1)

    def _encode_candidate_energy_summary(
        self,
        candidate_summary: torch.Tensor,
        base_temporal_energy: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate_energy_summary_encoder is None:
            raise ValueError("candidate energy summary encoder is disabled")
        candidate_summary = self._match_candidate_summary_dim(candidate_summary)
        candidate_features = _energy_with_risk_map(candidate_summary, self.config)
        mean_features = candidate_features[:, :1, ...].expand_as(candidate_features)
        base_summary = self._temporal_energy_summary(base_temporal_energy)
        base_summary = self._match_candidate_summary_dim(base_summary)
        base_features = _energy_with_risk_map(base_summary, self.config)
        base_features = base_features[:, None, ...].expand_as(candidate_features)
        summary_features = torch.cat(
            [
                candidate_features,
                candidate_features - mean_features,
                candidate_features - base_features,
            ],
            dim=-1,
        )
        return self.candidate_energy_summary_encoder(summary_features)

    def forward(
        self,
        candidate_trajectory: torch.Tensor,
        *,
        base_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_energy_summary_features: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        cfg = self.config
        candidates = _ensure_candidates(candidate_trajectory, cfg)
        base = _ensure_base(base_trajectory, candidates, cfg)
        past = _ensure_past(past_traj_original_scale, candidates, cfg)
        energy = _ensure_energy(temporal_energy_features, candidates, cfg)
        candidate_energy = _ensure_candidate_energy(candidate_temporal_energy_features, candidates, cfg)
        candidate_energy_summary = _ensure_candidate_energy_summary(candidate_energy_summary_features, candidates, cfg)

        batch_size, num_samples, num_modes, num_agents = candidates.shape[:4]
        candidate_context = self.candidate_encoder(_traj_with_velocity(candidates))
        base_context = self.base_encoder(_traj_with_velocity(base))
        base_context = base_context[:, None, ...].expand(batch_size, num_samples, num_modes, num_agents, -1)
        residual_context = self.residual_encoder((candidates - base[:, None, ...]).reshape(*candidates.shape[:-2], -1))
        past_context = self.past_encoder(past.reshape(batch_size, num_agents, -1))
        past_context = past_context[:, None, None, :, :].expand(batch_size, num_samples, num_modes, num_agents, -1)
        if bool(cfg.use_candidate_energy_context):
            if candidate_energy is None:
                raise ValueError("use_candidate_energy_context requires candidate_temporal_energy_features")
            energy_context = self._encode_energy(candidate_energy)
        elif bool(cfg.use_candidate_energy_summary_context):
            if candidate_energy_summary is None:
                raise ValueError("use_candidate_energy_summary_context requires candidate_energy_summary_features")
            energy_context = self._encode_candidate_energy_summary(candidate_energy_summary, energy)
        else:
            energy_context = self._encode_energy(energy)
            energy_context = energy_context[:, None, ...].expand(batch_size, num_samples, num_modes, num_agents, -1)
        safety_input = torch.cat(
            [candidate_context, base_context, residual_context, past_context, energy_context],
            dim=-1,
        )
        if self.energy_fusion_gate is not None:
            gate_input = safety_input
            energy_context = energy_context * self.energy_fusion_gate(gate_input)
        pieces = [candidate_context, base_context, residual_context, past_context, energy_context]
        if self.observable_feature_encoder is not None:
            observable = _observable_features(candidates, base, past, energy)
            pieces.append(self.observable_feature_encoder(observable))
        if self.mean_delta_encoder is not None:
            mean_candidate = candidates[:, :1, ...].expand_as(candidates)
            mean_delta_context = self.mean_delta_encoder((candidates - mean_candidate).reshape(*candidates.shape[:-2], -1))
            pieces.append(mean_delta_context)
        if self.mode_embedding is not None:
            mode_index = torch.arange(num_modes, device=candidates.device).clamp_max(int(cfg.max_modes) - 1)
            mode_context = self.mode_embedding(mode_index)[None, None, :, None, :].expand(
                batch_size,
                num_samples,
                num_modes,
                num_agents,
                -1,
            )
            pieces.append(mode_context)
        raw_logits = self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1)
        safety_logits: Optional[torch.Tensor] = None
        accept_logits: Optional[torch.Tensor] = None
        base_best_guard_logits: Optional[torch.Tensor] = None
        logits = raw_logits
        sample_index = torch.arange(num_samples, device=candidates.device)
        non_mean = (sample_index > 0)[None, :, None, None].to(dtype=raw_logits.dtype)
        if self.candidate_safety_penalty_head is not None:
            safety_logits = self.candidate_safety_penalty_head(safety_input).squeeze(-1)
            penalty = F.softplus(safety_logits) * float(cfg.candidate_safety_penalty_strength)
            logits = raw_logits - penalty * non_mean
        if self.base_best_guard_head is not None:
            base_best_guard_logits = self.base_best_guard_head(safety_input[:, 0, ...]).squeeze(-1)
            base_best_penalty = F.softplus(base_best_guard_logits)[:, None, :, :] * float(cfg.base_best_guard_strength)
            logits = logits - base_best_penalty * non_mean
        if self.residual_accept_gate_head is not None:
            mean_input = safety_input[:, :1, ...]
            if int(num_samples) > 1:
                non_mean_input = safety_input[:, 1:, ...]
            else:
                non_mean_input = mean_input
            gate_input = torch.cat(
                [
                    mean_input.squeeze(1),
                    non_mean_input.mean(dim=1),
                    non_mean_input.amax(dim=1),
                ],
                dim=-1,
            )
            accept_logits = self.residual_accept_gate_head(gate_input).squeeze(-1)
            logits = logits + accept_logits[:, None, :, :] * non_mean * float(cfg.residual_accept_gate_strength)
        if bool(return_aux):
            aux: Dict[str, torch.Tensor] = {"logits": logits, "raw_logits": raw_logits}
            if safety_logits is not None:
                aux["safety_logits"] = safety_logits
            if accept_logits is not None:
                aux["accept_logits"] = accept_logits
            if base_best_guard_logits is not None:
                aux["base_best_guard_logits"] = base_best_guard_logits
            return aux
        return logits

    @torch.no_grad()
    def select(
        self,
        candidate_trajectory: torch.Tensor,
        *,
        base_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_temporal_energy_features: Optional[torch.Tensor] = None,
        candidate_energy_summary_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.forward(
            candidate_trajectory,
            base_trajectory=base_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            candidate_temporal_energy_features=candidate_temporal_energy_features,
            candidate_energy_summary_features=candidate_energy_summary_features,
        )
        selected_index = logits.argmax(dim=1)
        index = selected_index[:, None, :, :, None, None].expand(
            candidate_trajectory.shape[0],
            1,
            candidate_trajectory.shape[2],
            candidate_trajectory.shape[3],
            candidate_trajectory.shape[4],
            candidate_trajectory.shape[5],
        )
        selected = torch.gather(candidate_trajectory, dim=1, index=index).squeeze(1)
        return {"selected": selected, "logits": logits, "selected_index": selected_index}


def load_social_cvae_group_selector(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> SocialCVAEGroupSelector:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if isinstance(checkpoint, Mapping) and "config" in checkpoint:
        config = SocialCVAEGroupSelectorConfig(**dict(checkpoint["config"]))
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    else:
        config = SocialCVAEGroupSelectorConfig()
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain selector weights: {path}")
    model = SocialCVAEGroupSelector(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = [
    "SocialCVAEGroupSelector",
    "SocialCVAEGroupSelectorConfig",
    "load_social_cvae_group_selector",
]
