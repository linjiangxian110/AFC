"""SocialCVAE-style residual refiner for teacher trajectories.

The refiner treats an existing teacher trajectory set as a coarse prediction
and learns a conditional latent distribution over trajectory-time residuals.
Unlike V22/V23 flow-step repairs, the residual target is defined directly in
future trajectory time: ``ground_truth - coarse_teacher``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class SocialCVAETeacherRefinerConfig:
    """Configuration for a teacher-output SocialCVAE residual refiner."""

    future_frames: int = 12
    coord_dim: int = 2
    past_frames: int = 8
    past_feature_dim: int = 6
    temporal_energy_dim: int = 5
    hidden_dim: int = 128
    latent_dim: int = 16
    max_modes: int = 20
    use_mode_embedding: bool = True
    residual_scale: float = 1.0
    max_delta: Optional[float] = 1.0
    min_logvar: float = -6.0
    max_logvar: float = 2.0
    use_energy_risk_map: bool = False
    energy_risk_distance_scale: float = 0.5
    use_temporal_energy_encoder: bool = False
    energy_temporal_hidden_dim: int = 64
    decoder_hidden_dim: int = 0
    decoder_layers: int = 2
    use_energy_conditioned_generator: bool = False
    use_set_generator: bool = False
    max_residual_slots: int = 1
    set_slot_scale: float = 1.0
    use_dynamic_slot_offsets: bool = False
    dynamic_slot_hidden_dim: int = 0
    dynamic_slot_offset_scale: float = 1.0
    dynamic_slot0_zero: bool = True


def _ensure_traj5(tensor: torch.Tensor, cfg: SocialCVAETeacherRefinerConfig, name: str) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape [B,K,A,T,2], got {tuple(tensor.shape)}")
    if int(tensor.shape[-2]) != int(cfg.future_frames) or int(tensor.shape[-1]) != int(cfg.coord_dim):
        raise ValueError(
            f"{name} future shape mismatch: got {tuple(tensor.shape)}, "
            f"expected trailing ({cfg.future_frames}, {cfg.coord_dim})"
        )
    return tensor


def _ensure_past(
    past_traj_original_scale: torch.Tensor,
    *,
    batch_size: int,
    num_agents: int,
    cfg: SocialCVAETeacherRefinerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if past_traj_original_scale.ndim != 4:
        raise ValueError(
            "past_traj_original_scale must have shape [B,A,P,C], "
            f"got {tuple(past_traj_original_scale.shape)}"
        )
    past = past_traj_original_scale.to(device=device, dtype=dtype)
    if tuple(past.shape[:2]) != (batch_size, num_agents):
        raise ValueError(f"past shape mismatch: {tuple(past.shape)} vs B={batch_size} A={num_agents}")
    if int(past.shape[-2]) != int(cfg.past_frames) or int(past.shape[-1]) != int(cfg.past_feature_dim):
        raise ValueError(
            "past_traj_original_scale does not match refiner config: "
            f"got trailing {tuple(past.shape[-2:])}, expected ({cfg.past_frames}, {cfg.past_feature_dim})"
        )
    return past


def _ensure_temporal_energy(
    temporal_energy_features: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_modes: int,
    num_agents: int,
    cfg: SocialCVAETeacherRefinerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if temporal_energy_features is None:
        return torch.zeros(
            batch_size,
            num_modes,
            num_agents,
            int(cfg.future_frames),
            int(cfg.temporal_energy_dim),
            device=device,
            dtype=dtype,
        )
    energy = temporal_energy_features.to(device=device, dtype=dtype)
    if energy.ndim == 4:
        if int(energy.shape[0]) != batch_size or int(energy.shape[2]) != num_agents:
            raise ValueError(f"temporal energy shape mismatch: {tuple(energy.shape)}")
        if int(energy.shape[1]) == 1 and num_modes > 1:
            energy = energy.expand(batch_size, num_modes, num_agents, int(energy.shape[-1]))
        elif int(energy.shape[1]) != num_modes:
            raise ValueError(f"temporal energy mode mismatch: got {int(energy.shape[1])}, expected {num_modes}")
        energy = energy[:, :, :, None, :].expand(
            batch_size,
            num_modes,
            num_agents,
            int(cfg.future_frames),
            int(energy.shape[-1]),
        )
    elif energy.ndim == 5:
        if int(energy.shape[0]) != batch_size or int(energy.shape[2]) != num_agents:
            raise ValueError(f"temporal energy shape mismatch: {tuple(energy.shape)}")
        if int(energy.shape[1]) == 1 and num_modes > 1:
            energy = energy.expand(batch_size, num_modes, num_agents, int(energy.shape[3]), int(energy.shape[4]))
        elif int(energy.shape[1]) != num_modes:
            raise ValueError(f"temporal energy mode mismatch: got {int(energy.shape[1])}, expected {num_modes}")
        if int(energy.shape[3]) != int(cfg.future_frames):
            raise ValueError(f"temporal energy time mismatch: got {int(energy.shape[3])}, expected {cfg.future_frames}")
    else:
        raise ValueError(
            "temporal_energy_features must have shape [B,K,A,C] or [B,K,A,T,C], "
            f"got {tuple(energy.shape)}"
        )
    if int(energy.shape[-1]) != int(cfg.temporal_energy_dim):
        raise ValueError(
            f"temporal energy dim mismatch: got {int(energy.shape[-1])}, expected {cfg.temporal_energy_dim}"
        )
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def _traj_with_velocity(traj: torch.Tensor) -> torch.Tensor:
    velocity = torch.cat([torch.zeros_like(traj[..., :1, :]), traj[..., 1:, :] - traj[..., :-1, :]], dim=-2)
    return torch.cat([traj, velocity], dim=-1).reshape(*traj.shape[:-2], traj.shape[-2] * traj.shape[-1] * 2)


def _energy_with_risk_map(energy: torch.Tensor, cfg: SocialCVAETeacherRefinerConfig) -> torch.Tensor:
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


class SocialCVAETeacherRefiner(nn.Module):
    """CVAE residual distribution conditioned on teacher coarse trajectories."""

    def __init__(self, config: Optional[SocialCVAETeacherRefinerConfig] = None) -> None:
        super().__init__()
        self.config = config or SocialCVAETeacherRefinerConfig()
        cfg = self.config
        flat_dim = int(cfg.future_frames * cfg.coord_dim)
        traj_feature_dim = int(flat_dim * 2)
        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        if int(cfg.decoder_layers) <= 0:
            raise ValueError(f"decoder_layers must be positive, got {cfg.decoder_layers}")
        if bool(cfg.use_energy_risk_map) and int(cfg.temporal_energy_dim) < 5:
            raise ValueError("use_energy_risk_map requires temporal_energy_dim >= 5")
        if int(cfg.max_residual_slots) <= 0:
            raise ValueError(f"max_residual_slots must be positive, got {cfg.max_residual_slots}")
        if bool(cfg.use_set_generator) and int(cfg.max_residual_slots) <= 1:
            raise ValueError("use_set_generator requires max_residual_slots > 1")
        if bool(cfg.use_dynamic_slot_offsets) and not bool(cfg.use_set_generator):
            raise ValueError("use_dynamic_slot_offsets requires use_set_generator=True")
        if float(cfg.dynamic_slot_offset_scale) < 0.0:
            raise ValueError("dynamic_slot_offset_scale must be non-negative")
        energy_feature_dim = int(cfg.temporal_energy_dim) + (5 if bool(cfg.use_energy_risk_map) else 0)
        energy_flat_dim = int(cfg.future_frames * energy_feature_dim)
        hidden_dim = int(cfg.hidden_dim)
        latent_dim = int(cfg.latent_dim)
        decoder_hidden_dim = int(cfg.decoder_hidden_dim) if int(cfg.decoder_hidden_dim) > 0 else hidden_dim

        self.coarse_encoder = nn.Sequential(
            nn.LayerNorm(traj_feature_dim),
            nn.Linear(traj_feature_dim, hidden_dim),
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
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if bool(cfg.use_mode_embedding):
            self.mode_embedding: Optional[nn.Embedding] = nn.Embedding(int(cfg.max_modes), hidden_dim)
        else:
            self.mode_embedding = None
        if bool(cfg.use_set_generator):
            self.set_slot_embedding: Optional[nn.Embedding] = nn.Embedding(int(cfg.max_residual_slots), latent_dim)
            nn.init.normal_(self.set_slot_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.set_slot_embedding.weight[0].zero_()
            self.set_slot_condition: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if bool(cfg.use_dynamic_slot_offsets):
                dynamic_hidden_dim = (
                    int(cfg.dynamic_slot_hidden_dim) if int(cfg.dynamic_slot_hidden_dim) > 0 else hidden_dim
                )
                dynamic_output = nn.Linear(dynamic_hidden_dim, latent_dim)
                nn.init.zeros_(dynamic_output.weight)
                nn.init.zeros_(dynamic_output.bias)
                self.dynamic_slot_offset: Optional[nn.Module] = nn.Sequential(
                    nn.LayerNorm(hidden_dim + latent_dim),
                    nn.Linear(hidden_dim + latent_dim, dynamic_hidden_dim),
                    nn.SiLU(),
                    dynamic_output,
                )
            else:
                self.dynamic_slot_offset = None
        else:
            self.set_slot_embedding = None
            self.set_slot_condition = None
            self.dynamic_slot_offset = None

        mode_dim = hidden_dim if self.mode_embedding is not None else 0
        condition_dim = hidden_dim * 3 + mode_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prior_head = nn.Linear(hidden_dim, latent_dim * 2)
        if bool(cfg.use_energy_conditioned_generator):
            self.energy_prior_modulator: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )
            self.energy_decoder_context: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.energy_decoder_film: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, decoder_hidden_dim * 2),
            )
        else:
            self.energy_prior_modulator = None
            self.energy_decoder_context = None
            self.energy_decoder_film = None
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        decoder_layers = []
        decoder_input_dim = hidden_dim + latent_dim
        for layer_index in range(int(cfg.decoder_layers)):
            in_dim = decoder_input_dim if layer_index == 0 else decoder_hidden_dim
            decoder_layers.extend([nn.Linear(in_dim, decoder_hidden_dim), nn.SiLU()])
        decoder_layers.append(nn.Linear(decoder_hidden_dim, flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_hidden_dim = int(decoder_hidden_dim)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def _split_gaussian(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        cfg = self.config
        return mu, logvar.clamp(float(cfg.min_logvar), float(cfg.max_logvar))

    def _encode_energy(self, energy: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        energy_features = _energy_with_risk_map(energy, cfg)
        if self.energy_token_encoder is None or self.energy_temporal_conv is None:
            return self.energy_encoder(energy_features.reshape(*energy_features.shape[:-2], -1))

        batch_size, num_modes, num_agents, num_steps, _num_features = energy_features.shape
        tokens = self.energy_token_encoder(energy_features)
        flat_tokens = tokens.reshape(batch_size * num_modes * num_agents, num_steps, -1).transpose(1, 2)
        encoded = self.energy_temporal_conv(flat_tokens).transpose(1, 2)
        encoded = encoded.reshape(batch_size, num_modes, num_agents, num_steps, -1)
        pooled = torch.cat(
            [
                encoded.mean(dim=-2),
                encoded.amax(dim=-2),
                encoded[..., -1, :],
            ],
            dim=-1,
        )
        return self.energy_encoder(pooled)

    def _condition(
        self,
        coarse_trajectory: torch.Tensor,
        *,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        coarse = _ensure_traj5(coarse_trajectory, cfg, "coarse_trajectory")
        batch_size, num_modes, num_agents, _num_steps, _coord_dim = coarse.shape
        past = _ensure_past(
            past_traj_original_scale,
            batch_size=int(batch_size),
            num_agents=int(num_agents),
            cfg=cfg,
            device=coarse.device,
            dtype=coarse.dtype,
        )
        energy = _ensure_temporal_energy(
            temporal_energy_features,
            batch_size=int(batch_size),
            num_modes=int(num_modes),
            num_agents=int(num_agents),
            cfg=cfg,
            device=coarse.device,
            dtype=coarse.dtype,
        )

        coarse_context = self.coarse_encoder(_traj_with_velocity(coarse))
        past_context = self.past_encoder(past.reshape(batch_size, num_agents, -1))
        past_context = past_context[:, None, :, :].expand(batch_size, num_modes, num_agents, -1)
        energy_context = self._encode_energy(energy)
        pieces = [coarse_context, past_context, energy_context]
        if self.mode_embedding is not None:
            mode_index = torch.arange(num_modes, device=coarse.device).clamp_max(int(cfg.max_modes) - 1)
            mode_context = self.mode_embedding(mode_index)[None, :, None, :].expand(
                batch_size,
                num_modes,
                num_agents,
                -1,
            )
            pieces.append(mode_context)
        condition = self.condition_mlp(torch.cat(pieces, dim=-1))
        return condition, coarse, energy, energy_context

    def _sample_latent(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        num_samples: int,
        z_mode: str,
        condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if int(num_samples) <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if z_mode == "slots":
            if self.set_slot_embedding is None:
                raise ValueError("z_mode='slots' requires use_set_generator=True")
            if int(num_samples) > int(self.config.max_residual_slots):
                raise ValueError(
                    f"num_samples={num_samples} exceeds max_residual_slots={self.config.max_residual_slots}"
                )
            slot = self.set_slot_embedding.weight[: int(num_samples)].to(device=mu.device, dtype=mu.dtype)
            slot = slot * float(self.config.set_slot_scale)
            dynamic_offset = None
            if self.dynamic_slot_offset is not None:
                if condition is None:
                    raise ValueError("condition is required when use_dynamic_slot_offsets=True")
                slot_expanded = slot[None, :, None, None, :].expand(
                    mu.shape[0],
                    int(num_samples),
                    mu.shape[1],
                    mu.shape[2],
                    mu.shape[3],
                )
                condition_expanded = condition[:, None, ...].expand(
                    mu.shape[0],
                    int(num_samples),
                    mu.shape[1],
                    mu.shape[2],
                    condition.shape[-1],
                )
                dynamic_input = torch.cat([condition_expanded, slot_expanded], dim=-1)
                dynamic_offset = self.dynamic_slot_offset(dynamic_input) * float(
                    self.config.dynamic_slot_offset_scale
                )
                if bool(self.config.dynamic_slot0_zero) and int(num_samples) > 0:
                    slot_mask = torch.ones(
                        int(num_samples),
                        device=mu.device,
                        dtype=mu.dtype,
                    )
                    slot_mask[0] = 0.0
                    dynamic_offset = dynamic_offset * slot_mask[None, :, None, None, None]
            z = mu[:, None, ...] + slot[None, :, None, None, :]
            if dynamic_offset is not None:
                z = z + dynamic_offset
            return z, dynamic_offset
        if z_mode == "mean":
            return mu[:, None, ...].expand(-1, int(num_samples), -1, -1, -1), None
        if z_mode != "sample":
            raise ValueError(f"Unsupported z_mode: {z_mode!r}")
        eps = torch.randn(
            mu.shape[0],
            int(num_samples),
            mu.shape[1],
            mu.shape[2],
            mu.shape[3],
            device=mu.device,
            dtype=mu.dtype,
        )
        return mu[:, None, ...] + eps * torch.exp(0.5 * logvar[:, None, ...]), None

    def _decode(
        self,
        condition: torch.Tensor,
        z: torch.Tensor,
        energy_context: Optional[torch.Tensor] = None,
        slot_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        decoder_condition = condition
        if self.energy_decoder_context is not None:
            if energy_context is None:
                raise ValueError("energy_context is required for energy-conditioned decoder")
            decoder_condition = decoder_condition + self.energy_decoder_context(energy_context)
        condition_expanded = decoder_condition[:, None, ...].expand(
            condition.shape[0],
            z.shape[1],
            condition.shape[1],
            condition.shape[2],
            condition.shape[3],
        )
        if slot_context is not None:
            condition_expanded = condition_expanded + slot_context[None, :, None, None, :]
        decoder_input = torch.cat([condition_expanded, z], dim=-1)
        if self.energy_decoder_film is None:
            raw = self.decoder(decoder_input)
        else:
            if energy_context is None:
                raise ValueError("energy_context is required for energy-conditioned decoder")
            film = self.energy_decoder_film(energy_context)
            gamma, beta = film.chunk(2, dim=-1)
            gamma = gamma[:, None, ...]
            beta = beta[:, None, ...]
            hidden = decoder_input
            modules = list(self.decoder)
            module_index = 0
            while module_index < len(modules) - 1:
                linear = modules[module_index]
                activation = modules[module_index + 1]
                hidden = activation(linear(hidden))
                if int(hidden.shape[-1]) == int(self.decoder_hidden_dim):
                    hidden = hidden * (1.0 + torch.tanh(gamma)) + beta
                module_index += 2
            raw = modules[-1](hidden)
        if cfg.max_delta is not None:
            raw = torch.tanh(raw) * float(cfg.max_delta)
        delta = raw.reshape(
            condition.shape[0],
            z.shape[1],
            condition.shape[1],
            condition.shape[2],
            int(cfg.future_frames),
            int(cfg.coord_dim),
        )
        return delta * float(cfg.residual_scale)

    def forward(
        self,
        coarse_trajectory: torch.Tensor,
        *,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        ground_truth: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        z_source: str = "prior",
        z_mode: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Refine a coarse teacher trajectory set.

        Args:
            coarse_trajectory: [B,K,A,T,2] teacher coarse predictions.
            past_traj_original_scale: [B,A,P,C] MoFlow past features.
            temporal_energy_features: [B,K,A,T,C] future interaction energy.
            ground_truth: [B,A,T,2], required when ``z_source='posterior'``.
            num_samples: residual samples per teacher mode.
            z_source: ``'prior'`` for inference or ``'posterior'`` for training.
            z_mode: ``'mean'``, ``'sample'``, or ``'slots'``.
        """

        condition, coarse, energy, energy_context = self._condition(
            coarse_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
        )
        prior_stats = self.prior_head(condition)
        if self.energy_prior_modulator is not None:
            energy_delta_mu, energy_delta_logvar = self.energy_prior_modulator(energy_context).chunk(2, dim=-1)
            prior_mu_raw, prior_logvar_raw = prior_stats.chunk(2, dim=-1)
            prior_stats = torch.cat(
                [
                    prior_mu_raw + energy_delta_mu,
                    prior_logvar_raw + torch.tanh(energy_delta_logvar),
                ],
                dim=-1,
            )
        prior_mu, prior_logvar = self._split_gaussian(prior_stats)

        posterior_mu = posterior_logvar = None
        if z_source == "posterior":
            if ground_truth is None:
                raise ValueError("ground_truth is required when z_source='posterior'")
            if ground_truth.ndim != 4:
                raise ValueError(f"ground_truth must have shape [B,A,T,2], got {tuple(ground_truth.shape)}")
            target_residual = ground_truth[:, None, ...].to(device=coarse.device, dtype=coarse.dtype) - coarse
            residual_context = self.residual_encoder(target_residual.reshape(*target_residual.shape[:-2], -1))
            posterior_mu, posterior_logvar = self._split_gaussian(
                self.posterior_head(torch.cat([condition, residual_context], dim=-1))
            )
            z_mu, z_logvar = posterior_mu, posterior_logvar
        elif z_source == "prior":
            z_mu, z_logvar = prior_mu, prior_logvar
        else:
            raise ValueError(f"Unsupported z_source: {z_source!r}")

        z, dynamic_slot_offset = self._sample_latent(
            z_mu,
            z_logvar,
            num_samples=int(num_samples),
            z_mode=z_mode,
            condition=condition,
        )
        slot_context = None
        if z_mode == "slots":
            if self.set_slot_embedding is None or self.set_slot_condition is None:
                raise ValueError("z_mode='slots' requires set generator modules")
            slot = self.set_slot_embedding.weight[: int(num_samples)].to(device=coarse.device, dtype=coarse.dtype)
            slot_context = self.set_slot_condition(slot * float(self.config.set_slot_scale))
        delta = self._decode(condition, z, energy_context=energy_context, slot_context=slot_context)
        refined = coarse[:, None, ...] + delta
        output: Dict[str, torch.Tensor] = {
            "refined": refined,
            "delta": delta,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "temporal_energy_features": energy,
        }
        if posterior_mu is not None and posterior_logvar is not None:
            output["posterior_mu"] = posterior_mu
            output["posterior_logvar"] = posterior_logvar
        if dynamic_slot_offset is not None:
            output["dynamic_slot_offset"] = dynamic_slot_offset
        return output

    @torch.no_grad()
    def refine(
        self,
        coarse_trajectory: torch.Tensor,
        *,
        past_traj_original_scale: torch.Tensor,
        temporal_energy_features: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        z_mode: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        return self.forward(
            coarse_trajectory,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            num_samples=int(num_samples),
            z_source="prior",
            z_mode=z_mode,
        )


def load_social_cvae_teacher_refiner(path: str | Path, *, map_location: str | torch.device = "cpu") -> SocialCVAETeacherRefiner:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if isinstance(checkpoint, Mapping) and "config" in checkpoint:
        config = SocialCVAETeacherRefinerConfig(**dict(checkpoint["config"]))
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    else:
        config = SocialCVAETeacherRefinerConfig()
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain model weights: {path}")
    model = SocialCVAETeacherRefiner(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = [
    "SocialCVAETeacherRefiner",
    "SocialCVAETeacherRefinerConfig",
    "load_social_cvae_teacher_refiner",
]
