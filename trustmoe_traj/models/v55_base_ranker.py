"""High-potential base ranker for V55-D adaptive residual budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class V55BaseRankerConfig:
    future_frames: int = 12
    coord_dim: int = 2
    past_frames: int = 8
    past_feature_dim: int = 6
    temporal_energy_dim: int = 5
    residual_slots: int = 4
    hidden_dim: int = 128
    max_modes: int = 20
    use_mode_embedding: bool = True
    use_energy_risk_map: bool = True
    energy_risk_distance_scale: float = 0.5
    dropout: float = 0.0


def _traj_with_velocity(traj: torch.Tensor) -> torch.Tensor:
    velocity = torch.cat([torch.zeros_like(traj[..., :1, :]), traj[..., 1:, :] - traj[..., :-1, :]], dim=-2)
    return torch.cat([traj, velocity], dim=-1).reshape(*traj.shape[:-2], traj.shape[-2] * traj.shape[-1] * 2)


def _energy_with_risk_map(energy: torch.Tensor, cfg: V55BaseRankerConfig) -> torch.Tensor:
    raw = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(cfg.use_energy_risk_map):
        return raw
    if int(raw.shape[-1]) < 5:
        raise ValueError(f"use_energy_risk_map requires at least 5 energy channels, got {int(raw.shape[-1])}")
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


def _ensure_base(base: torch.Tensor, cfg: V55BaseRankerConfig) -> torch.Tensor:
    if base.ndim != 5:
        raise ValueError(f"base_trajectory must have shape [B,K,A,T,2], got {tuple(base.shape)}")
    if int(base.shape[-2]) != int(cfg.future_frames) or int(base.shape[-1]) != int(cfg.coord_dim):
        raise ValueError(
            "base future shape mismatch: "
            f"got trailing {tuple(base.shape[-2:])}, expected ({cfg.future_frames}, {cfg.coord_dim})"
        )
    return base


def _ensure_refined(refined: torch.Tensor, base: torch.Tensor, cfg: V55BaseRankerConfig) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"refined_trajectory must have shape [B,S,K,A,T,2], got {tuple(refined.shape)}")
    expected = (
        int(base.shape[0]),
        int(cfg.residual_slots),
        int(base.shape[1]),
        int(base.shape[2]),
        int(cfg.future_frames),
        int(cfg.coord_dim),
    )
    if tuple(refined.shape) != expected:
        raise ValueError(f"refined/base shape mismatch: got {tuple(refined.shape)}, expected {expected}")
    return refined.to(device=base.device, dtype=base.dtype)


def _ensure_past(past: torch.Tensor, base: torch.Tensor, cfg: V55BaseRankerConfig) -> torch.Tensor:
    if past.ndim != 4:
        raise ValueError(f"past_traj_original_scale must have shape [B,A,P,C], got {tuple(past.shape)}")
    if tuple(past.shape[:2]) != (int(base.shape[0]), int(base.shape[2])):
        raise ValueError(f"past/base shape mismatch: past={tuple(past.shape)} base={tuple(base.shape)}")
    if int(past.shape[-2]) != int(cfg.past_frames) or int(past.shape[-1]) != int(cfg.past_feature_dim):
        raise ValueError(
            "past shape mismatch: "
            f"got trailing {tuple(past.shape[-2:])}, expected ({cfg.past_frames}, {cfg.past_feature_dim})"
        )
    return past.to(device=base.device, dtype=base.dtype)


def _ensure_energy(
    energy: Optional[torch.Tensor],
    base: torch.Tensor,
    cfg: V55BaseRankerConfig,
) -> torch.Tensor:
    batch_size, num_modes, num_agents = int(base.shape[0]), int(base.shape[1]), int(base.shape[2])
    if energy is None:
        return base.new_zeros(batch_size, num_modes, num_agents, int(cfg.future_frames), int(cfg.temporal_energy_dim))
    out = energy.to(device=base.device, dtype=base.dtype)
    if out.ndim == 4:
        out = out[:, :, :, None, :].expand(batch_size, num_modes, num_agents, int(cfg.future_frames), int(out.shape[-1]))
    if out.ndim != 5:
        raise ValueError(f"temporal_energy_features must have shape [B,K,A,T,C], got {tuple(out.shape)}")
    if int(out.shape[1]) == 1 and num_modes > 1:
        out = out.expand(batch_size, num_modes, num_agents, int(out.shape[3]), int(out.shape[4]))
    if tuple(out.shape[:4]) != (batch_size, num_modes, num_agents, int(cfg.future_frames)):
        raise ValueError(f"energy/base shape mismatch: energy={tuple(out.shape)} base={tuple(base.shape)}")
    if int(out.shape[-1]) != int(cfg.temporal_energy_dim):
        raise ValueError(f"energy dim mismatch: got {int(out.shape[-1])}, expected {cfg.temporal_energy_dim}")
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _residual_features(refined: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    delta = refined - base[:, None, ...]
    batch_size, num_slots, num_modes, num_agents, num_steps, coord_dim = delta.shape
    delta_by_base = delta.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_modes, num_agents, -1)
    endpoint = delta[..., -1, :].permute(0, 2, 3, 1, 4).reshape(batch_size, num_modes, num_agents, -1)
    endpoint_norm = torch.linalg.norm(delta[..., -1, :], dim=-1)
    traj_norm = torch.linalg.norm(delta, dim=-1).mean(dim=-1)
    stats = torch.stack(
        [
            endpoint_norm.mean(dim=1),
            endpoint_norm.amin(dim=1),
            endpoint_norm.amax(dim=1),
            endpoint_norm.std(dim=1, unbiased=False),
            traj_norm.mean(dim=1),
            traj_norm.std(dim=1, unbiased=False),
        ],
        dim=-1,
    )
    return torch.cat([delta_by_base, endpoint, stats], dim=-1)


class V55BaseRanker(nn.Module):
    """Score base modes by potential for V38 residual-budget allocation."""

    def __init__(self, config: Optional[V55BaseRankerConfig] = None) -> None:
        super().__init__()
        self.config = config or V55BaseRankerConfig()
        cfg = self.config
        if int(cfg.residual_slots) <= 1:
            raise ValueError("residual_slots must be > 1")
        if not (0.0 <= float(cfg.dropout) < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        flat_dim = int(cfg.future_frames * cfg.coord_dim)
        traj_feature_dim = int(flat_dim * 2)
        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        energy_feature_dim = int(cfg.temporal_energy_dim) + (5 if bool(cfg.use_energy_risk_map) else 0)
        energy_flat_dim = int(cfg.future_frames * energy_feature_dim)
        residual_feature_dim = int(cfg.residual_slots * cfg.future_frames * cfg.coord_dim)
        residual_feature_dim += int(cfg.residual_slots * cfg.coord_dim) + 6
        hidden_dim = int(cfg.hidden_dim)

        def dropout_layer() -> nn.Module:
            return nn.Dropout(float(cfg.dropout))

        self.base_encoder = nn.Sequential(
            nn.LayerNorm(traj_feature_dim),
            nn.Linear(traj_feature_dim, hidden_dim),
            nn.SiLU(),
            dropout_layer(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.past_encoder = nn.Sequential(
            nn.LayerNorm(past_flat_dim),
            nn.Linear(past_flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_encoder = nn.Sequential(
            nn.LayerNorm(energy_flat_dim),
            nn.Linear(energy_flat_dim, hidden_dim),
            nn.SiLU(),
            dropout_layer(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(residual_feature_dim),
            nn.Linear(residual_feature_dim, hidden_dim),
            nn.SiLU(),
            dropout_layer(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if bool(cfg.use_mode_embedding):
            self.mode_embedding: Optional[nn.Embedding] = nn.Embedding(int(cfg.max_modes), hidden_dim)
        else:
            self.mode_embedding = None
        mode_dim = hidden_dim if self.mode_embedding is not None else 0
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + mode_dim, hidden_dim),
            nn.SiLU(),
            dropout_layer(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(
        self,
        base_trajectory: torch.Tensor,
        *,
        refined_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        base = _ensure_base(base_trajectory, cfg)
        refined = _ensure_refined(refined_trajectory, base, cfg)
        past = _ensure_past(past_traj_original_scale, base, cfg)
        energy = _ensure_energy(temporal_energy_features, base, cfg)

        batch_size, num_modes, num_agents = int(base.shape[0]), int(base.shape[1]), int(base.shape[2])
        base_context = self.base_encoder(_traj_with_velocity(base))
        past_context = self.past_encoder(past.reshape(batch_size, num_agents, -1))
        past_context = past_context[:, None, :, :].expand(batch_size, num_modes, num_agents, -1)
        energy_context = self.energy_encoder(_energy_with_risk_map(energy, cfg).reshape(batch_size, num_modes, num_agents, -1))
        residual_context = self.residual_encoder(_residual_features(refined, base))
        pieces = [base_context, past_context, energy_context, residual_context]
        if self.mode_embedding is not None:
            mode_index = torch.arange(num_modes, device=base.device).clamp_max(int(cfg.max_modes) - 1)
            mode_context = self.mode_embedding(mode_index)[None, :, None, :].expand(batch_size, num_modes, num_agents, -1)
            pieces.append(mode_context)
        return self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1)

    @torch.no_grad()
    def rank(
        self,
        base_trajectory: torch.Tensor,
        *,
        refined_trajectory: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.forward(
            base_trajectory,
            refined_trajectory=refined_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
        )
        order = torch.argsort(logits, dim=1, descending=True)
        return {"logits": logits, "base_order": order}


def load_v55_base_ranker(path: str | Path, *, map_location: str | torch.device = "cpu") -> V55BaseRanker:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if isinstance(checkpoint, Mapping) and "config" in checkpoint:
        config = V55BaseRankerConfig(**dict(checkpoint["config"]))
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    else:
        config = V55BaseRankerConfig()
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain base ranker weights: {path}")
    model = V55BaseRanker(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = [
    "V55BaseRanker",
    "V55BaseRankerConfig",
    "load_v55_base_ranker",
]
