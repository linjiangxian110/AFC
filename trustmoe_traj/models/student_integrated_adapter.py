"""Student-integrated adapters for the MoFlow fast student.

V18-A moves the repair capacity from an external graduate head into the fast
student branch.  The adapter is intentionally lightweight: it consumes the
student generator output in normalized MoFlow future coordinates, uses past
motion and optional scene-aware temporal interaction energy, and returns an
adapted student trajectory in the same normalized coordinate frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StudentIntegratedAdapterConfig:
    """Configuration for the V18-A decoder-output adapter."""

    pred_len: int = 12
    coord_dim: int = 2
    past_frames: int = 8
    past_feature_dim: int = 6
    temporal_energy_dim: int = 5
    hidden_dim: int = 128
    num_mode_context_layers: int = 1
    num_mode_context_heads: int = 4
    mode_context_dropout: float = 0.0
    use_temporal_energy: bool = True
    residual_scale: float = 1.0
    max_delta: Optional[float] = 0.25
    gate_init_bias: float = 0.0


def _ensure_future5(future: torch.Tensor, cfg: StudentIntegratedAdapterConfig) -> Tuple[torch.Tensor, bool]:
    """Return [B, K, A, T, D] and whether a leading M dimension existed."""

    if future.ndim == 4:
        future = future.unsqueeze(1)
    if future.ndim == 5:
        if int(future.shape[-2]) != int(cfg.pred_len) or int(future.shape[-1]) != int(cfg.coord_dim):
            raise ValueError(
                "future shape does not match adapter config: "
                f"future={tuple(future.shape)} pred_len={cfg.pred_len} coord_dim={cfg.coord_dim}"
            )
        return future, False
    if future.ndim == 6:
        if int(future.shape[-2]) != int(cfg.pred_len) or int(future.shape[-1]) != int(cfg.coord_dim):
            raise ValueError(
                "future shape does not match adapter config: "
                f"future={tuple(future.shape)} pred_len={cfg.pred_len} coord_dim={cfg.coord_dim}"
            )
        b, m, k, a, t, d = future.shape
        return future.reshape(b * m, k, a, t, d), True
    raise ValueError(
        "future must have shape [B,A,T,2], [B,K,A,T,2], or [B,M,K,A,T,2], "
        f"got {tuple(future.shape)}"
    )


def _restore_future_shape(future5: torch.Tensor, *, had_m_dim: bool, batch_size: int, num_draws: int) -> torch.Tensor:
    if had_m_dim:
        return future5.reshape(batch_size, num_draws, *future5.shape[1:])
    return future5


def _ensure_past(past_traj_original_scale: torch.Tensor, cfg: StudentIntegratedAdapterConfig) -> torch.Tensor:
    if past_traj_original_scale.ndim != 4:
        raise ValueError(
            "past_traj_original_scale must have shape [B,A,P,C], "
            f"got {tuple(past_traj_original_scale.shape)}"
        )
    if int(past_traj_original_scale.shape[-2]) != int(cfg.past_frames):
        raise ValueError(
            "past_traj_original_scale past length does not match adapter config: "
            f"{int(past_traj_original_scale.shape[-2])} vs {cfg.past_frames}"
        )
    if int(past_traj_original_scale.shape[-1]) != int(cfg.past_feature_dim):
        raise ValueError(
            "past_traj_original_scale feature dim does not match adapter config: "
            f"{int(past_traj_original_scale.shape[-1])} vs {cfg.past_feature_dim}"
        )
    return past_traj_original_scale


def _ensure_temporal_energy(
    temporal_energy: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_modes: int,
    num_agents: int,
    had_m_dim: bool,
    num_draws: int,
    cfg: StudentIntegratedAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(cfg.use_temporal_energy):
        return torch.zeros(
            (batch_size * num_draws if had_m_dim else batch_size, num_modes, num_agents, cfg.pred_len, 0),
            device=device,
            dtype=dtype,
        )
    if temporal_energy is None:
        return torch.zeros(
            (
                batch_size * num_draws if had_m_dim else batch_size,
                num_modes,
                num_agents,
                cfg.pred_len,
                cfg.temporal_energy_dim,
            ),
            device=device,
            dtype=dtype,
        )

    energy = temporal_energy.to(device=device, dtype=dtype)
    if energy.ndim == 5:
        if had_m_dim:
            energy = energy[:, None, ...].expand(
                batch_size, num_draws, num_modes, num_agents, cfg.pred_len, cfg.temporal_energy_dim
            )
            energy = energy.reshape(batch_size * num_draws, num_modes, num_agents, cfg.pred_len, cfg.temporal_energy_dim)
    elif energy.ndim == 6:
        if not had_m_dim:
            if int(energy.shape[1]) != 1:
                raise ValueError(
                    "temporal energy with M dimension can only be used with a matching M future input "
                    f"or M=1, got {tuple(energy.shape)}"
                )
            energy = energy[:, 0]
        else:
            energy = energy.reshape(
                batch_size * num_draws, num_modes, num_agents, cfg.pred_len, cfg.temporal_energy_dim
            )
    else:
        raise ValueError(
            "temporal_interaction_energy_features must have shape [B,K,A,T,C] or [B,M,K,A,T,C], "
            f"got {tuple(energy.shape)}"
        )

    expected = (
        batch_size * num_draws if had_m_dim else batch_size,
        num_modes,
        num_agents,
        cfg.pred_len,
        cfg.temporal_energy_dim,
    )
    if tuple(energy.shape) != expected:
        raise ValueError(
            "temporal_interaction_energy_features shape mismatch: "
            f"got {tuple(energy.shape)}, expected {expected}"
        )
    return energy


class StudentIntegratedEnergyAdapter(nn.Module):
    """V18-A interaction-aware adapter attached to the fast student output."""

    def __init__(self, config: Optional[StudentIntegratedAdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or StudentIntegratedAdapterConfig()
        cfg = self.config

        future_flat_dim = int(cfg.pred_len * cfg.coord_dim)
        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        summary_dim = 4
        pooled_energy_dim = int(cfg.temporal_energy_dim) if cfg.use_temporal_energy else 0

        self.past_encoder = nn.Sequential(
            nn.LayerNorm(past_flat_dim),
            nn.Linear(past_flat_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.mode_encoder = nn.Sequential(
            nn.LayerNorm(future_flat_dim + summary_dim + cfg.hidden_dim + pooled_energy_dim),
            nn.Linear(future_flat_dim + summary_dim + cfg.hidden_dim + pooled_energy_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        if int(cfg.num_mode_context_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(cfg.hidden_dim),
                nhead=int(cfg.num_mode_context_heads),
                dim_feedforward=int(cfg.hidden_dim) * 4,
                dropout=float(cfg.mode_context_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mode_context = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg.num_mode_context_layers))
        else:
            self.mode_context = nn.Identity()

        if cfg.use_temporal_energy:
            self.temporal_energy_encoder: nn.Module = nn.Sequential(
                nn.LayerNorm(int(cfg.temporal_energy_dim)),
                nn.Linear(int(cfg.temporal_energy_dim), int(cfg.hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            )
        else:
            self.temporal_energy_encoder = nn.Identity()

        self.time_embedding = nn.Embedding(int(cfg.pred_len), int(cfg.hidden_dim))
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(int(cfg.hidden_dim)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(int(cfg.hidden_dim), int(cfg.coord_dim))
        self.gate_head = nn.Linear(int(cfg.hidden_dim), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(self.config.gate_init_bias))

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(
        self,
        future_normalized: torch.Tensor,
        *,
        past_traj_original_scale: Optional[torch.Tensor] = None,
        temporal_interaction_energy_features: Optional[torch.Tensor] = None,
        x_data: Optional[Mapping[str, Any]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        cfg = self.config
        if x_data is not None:
            if past_traj_original_scale is None and "past_traj_original_scale" in x_data:
                past_traj_original_scale = x_data["past_traj_original_scale"]
            if temporal_interaction_energy_features is None and "temporal_interaction_energy_features" in x_data:
                temporal_interaction_energy_features = x_data["temporal_interaction_energy_features"]
        if past_traj_original_scale is None:
            raise ValueError("past_traj_original_scale is required for StudentIntegratedEnergyAdapter")

        original_shape = tuple(future_normalized.shape)
        flattened_input = future_normalized
        output_is_flat = False
        if future_normalized.ndim == 5 and int(future_normalized.shape[-1]) == int(cfg.pred_len * cfg.coord_dim):
            output_is_flat = True
            b, m, k, a, _flat = future_normalized.shape
            future_normalized = future_normalized.reshape(b, m, k, a, int(cfg.pred_len), int(cfg.coord_dim))
        elif future_normalized.ndim == 4 and int(future_normalized.shape[-1]) == int(cfg.pred_len * cfg.coord_dim):
            output_is_flat = True
            b, k, a, _flat = future_normalized.shape
            future_normalized = future_normalized.reshape(b, k, a, int(cfg.pred_len), int(cfg.coord_dim))

        future5, had_m_dim = _ensure_future5(future_normalized.to(dtype=torch.float32), cfg)
        if had_m_dim:
            batch_size = int(future_normalized.shape[0])
            num_draws = int(future_normalized.shape[1])
        else:
            batch_size = int(future5.shape[0])
            num_draws = 1
        num_modes = int(future5.shape[1])
        num_agents = int(future5.shape[2])

        past = _ensure_past(past_traj_original_scale.to(device=future5.device, dtype=future5.dtype), cfg)
        if had_m_dim:
            past = past[:, None, ...].expand(batch_size, num_draws, num_agents, cfg.past_frames, cfg.past_feature_dim)
            past = past.reshape(batch_size * num_draws, num_agents, cfg.past_frames, cfg.past_feature_dim)
        if tuple(past.shape[:2]) != tuple(future5.shape[:1] + future5.shape[2:3]):
            raise ValueError(
                "past/future batch-agent shape mismatch: "
                f"past={tuple(past.shape)} future={tuple(future5.shape)}"
            )

        energy = _ensure_temporal_energy(
            temporal_interaction_energy_features,
            batch_size=batch_size,
            num_modes=num_modes,
            num_agents=num_agents,
            had_m_dim=had_m_dim,
            num_draws=num_draws,
            cfg=cfg,
            device=future5.device,
            dtype=future5.dtype,
        )

        b_eff = int(future5.shape[0])
        future_flat = future5.reshape(b_eff, num_modes, num_agents, -1)
        displacement = future5[..., -1, :] - future5[..., 0, :]
        endpoint = future5[..., -1, :]
        summary = torch.cat([displacement, endpoint], dim=-1)

        past_flat = past.reshape(b_eff, num_agents, -1)
        past_context = self.past_encoder(past_flat)
        past_context = past_context[:, None, :, :].expand(b_eff, num_modes, num_agents, -1)

        feature_parts = [future_flat, summary, past_context]
        if bool(cfg.use_temporal_energy):
            pooled_energy = energy.mean(dim=-2)
            feature_parts.append(pooled_energy)
        mode_features = torch.cat(feature_parts, dim=-1)
        encoded = self.mode_encoder(mode_features)

        encoded_for_context = encoded.permute(0, 2, 1, 3).reshape(b_eff * num_agents, num_modes, -1)
        encoded_context = self.mode_context(encoded_for_context)
        encoded = encoded_context.reshape(b_eff, num_agents, num_modes, -1).permute(0, 2, 1, 3)

        temporal_hidden = encoded[:, :, :, None, :].expand(b_eff, num_modes, num_agents, cfg.pred_len, -1)
        if bool(cfg.use_temporal_energy):
            temporal_hidden = temporal_hidden + self.temporal_energy_encoder(
                energy.reshape(b_eff * num_modes * num_agents * cfg.pred_len, -1)
            ).reshape(b_eff, num_modes, num_agents, cfg.pred_len, -1)

        time_index = torch.arange(int(cfg.pred_len), device=future5.device)
        temporal_hidden = temporal_hidden + self.time_embedding(time_index).reshape(1, 1, 1, cfg.pred_len, -1)
        temporal_hidden = self.temporal_mlp(temporal_hidden)

        delta = self.delta_head(temporal_hidden)
        if cfg.max_delta is not None:
            max_delta = float(cfg.max_delta)
            delta = max_delta * torch.tanh(delta / max(max_delta, 1e-6))
        delta = float(cfg.residual_scale) * delta
        gate = torch.sigmoid(self.gate_head(temporal_hidden))
        adapted5 = future5 + gate * delta

        adapted = _restore_future_shape(adapted5, had_m_dim=had_m_dim, batch_size=batch_size, num_draws=num_draws)
        delta_out = _restore_future_shape(delta, had_m_dim=had_m_dim, batch_size=batch_size, num_draws=num_draws)
        gate_out = _restore_future_shape(gate, had_m_dim=had_m_dim, batch_size=batch_size, num_draws=num_draws)

        if output_is_flat:
            adapted = adapted.reshape(original_shape)
            delta_out = delta_out.reshape(original_shape)
            gate_out = gate_out.reshape(*original_shape[:-1], cfg.pred_len)
            gate_out = gate_out.reshape(*original_shape[:-1], cfg.pred_len, 1)
            if not return_dict:
                return adapted.to(dtype=flattened_input.dtype)

        if not return_dict:
            return adapted.to(dtype=flattened_input.dtype)
        return {
            "adapted_future_normalized": adapted,
            "delta_normalized": delta_out,
            "gate": gate_out,
            "raw_future_normalized": future_normalized,
        }


def build_student_integrated_adapter_from_cache_shapes(
    tensor_shapes: Mapping[str, Any],
    *,
    hidden_dim: int = 128,
    num_mode_context_layers: int = 1,
    num_mode_context_heads: int = 4,
    mode_context_dropout: float = 0.0,
    use_temporal_energy: bool = True,
    temporal_energy_dim: int = 5,
    residual_scale: float = 1.0,
    max_delta: Optional[float] = 0.25,
    gate_init_bias: float = 0.0,
) -> StudentIntegratedEnergyAdapter:
    student_shape = list(tensor_shapes["student_pred"])
    past_shape = list(tensor_shapes["past_traj_original_scale"])
    if len(student_shape) != 5:
        raise ValueError(f"Expected student_pred shape [N,K,A,T,2], got {student_shape}")
    if len(past_shape) != 4:
        raise ValueError(f"Expected past_traj_original_scale shape [N,A,P,C], got {past_shape}")
    cfg = StudentIntegratedAdapterConfig(
        pred_len=int(student_shape[3]),
        coord_dim=int(student_shape[4]),
        past_frames=int(past_shape[2]),
        past_feature_dim=int(past_shape[3]),
        temporal_energy_dim=int(temporal_energy_dim),
        hidden_dim=int(hidden_dim),
        num_mode_context_layers=int(num_mode_context_layers),
        num_mode_context_heads=int(num_mode_context_heads),
        mode_context_dropout=float(mode_context_dropout),
        use_temporal_energy=bool(use_temporal_energy),
        residual_scale=float(residual_scale),
        max_delta=max_delta,
        gate_init_bias=float(gate_init_bias),
    )
    return StudentIntegratedEnergyAdapter(cfg)


def load_student_integrated_adapter(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StudentIntegratedEnergyAdapter, Dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid student-integrated checkpoint payload: {type(payload)!r}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Student-integrated checkpoint is missing `config`")
    cfg = StudentIntegratedAdapterConfig(**dict(config_payload))
    model = StudentIntegratedEnergyAdapter(cfg)
    state_dict = payload.get("model_state") or payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Student-integrated checkpoint is missing `model_state`")
    model.load_state_dict(state_dict)
    return model, dict(payload)


__all__ = [
    "StudentIntegratedAdapterConfig",
    "StudentIntegratedEnergyAdapter",
    "build_student_integrated_adapter_from_cache_shapes",
    "load_student_integrated_adapter",
]
