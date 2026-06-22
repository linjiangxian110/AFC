"""Hidden-state adapters for the MoFlow fast student.

V19-A inserts a lightweight, zero-initialized residual branch at the
``readout_token`` level, after the fast student's motion decoder and before the
trajectory regression head.  This keeps the original student backbone intact
while allowing interaction/risk-aware corrections to affect the hidden
trajectory representation instead of patching final coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StudentHiddenAdapterConfig:
    """Configuration for student hidden-token adapters."""

    adapter_site: str = "readout"
    token_dim: int = 256
    past_frames: int = 8
    past_feature_dim: int = 6
    hidden_dim: int = 128
    noise_dim: int = 256
    social_risk_dim: int = 10
    max_modes: int = 20
    use_noise: bool = True
    use_past_social_risk: bool = False
    use_mode_embedding: bool = True
    num_mode_context_layers: int = 1
    num_mode_context_heads: int = 4
    mode_context_dropout: float = 0.0
    residual_scale: float = 1.0
    max_token_delta: Optional[float] = 0.5
    gate_init_bias: float = -2.0


def _ensure_token5(token: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return [B, M, K, A, D] and whether the input had an M dimension."""

    if token.ndim == 4:
        return token.unsqueeze(1), False
    if token.ndim == 5:
        return token, True
    raise ValueError(f"hidden token must have shape [B,K,A,D] or [B,M,K,A,D], got {tuple(token.shape)}")


def _restore_token_shape(token5: torch.Tensor, *, had_m_dim: bool) -> torch.Tensor:
    if had_m_dim:
        return token5
    return token5[:, 0]


def _ensure_past(
    past_traj_original_scale: torch.Tensor,
    *,
    batch_size: int,
    num_draws: int,
    num_agents: int,
    cfg: StudentHiddenAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if past_traj_original_scale.ndim != 4:
        raise ValueError(
            "past_traj_original_scale must have shape [B,A,P,C], "
            f"got {tuple(past_traj_original_scale.shape)}"
        )
    past = past_traj_original_scale.to(device=device, dtype=dtype)
    if int(past.shape[-2]) != int(cfg.past_frames) or int(past.shape[-1]) != int(cfg.past_feature_dim):
        raise ValueError(
            "past_traj_original_scale shape does not match hidden adapter config: "
            f"got {tuple(past.shape)}, expected trailing ({cfg.past_frames}, {cfg.past_feature_dim})"
        )

    if int(past.shape[0]) == int(batch_size):
        past = past[:, None].expand(batch_size, num_draws, num_agents, cfg.past_frames, cfg.past_feature_dim)
        return past.reshape(batch_size * num_draws, num_agents, cfg.past_frames, cfg.past_feature_dim)
    if int(past.shape[0]) == int(batch_size * num_draws):
        if int(past.shape[1]) != int(num_agents):
            raise ValueError(f"past agent count mismatch: {tuple(past.shape)} vs num_agents={num_agents}")
        return past
    raise ValueError(
        "past batch size must match B or B*M from readout_token: "
        f"past={tuple(past.shape)} B={batch_size} M={num_draws}"
    )


def _ensure_noise(
    noise_latent: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_draws: int,
    cfg: StudentHiddenAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(cfg.use_noise):
        return torch.zeros(batch_size * num_draws, int(cfg.noise_dim), device=device, dtype=dtype)
    if noise_latent is None:
        return torch.zeros(batch_size * num_draws, int(cfg.noise_dim), device=device, dtype=dtype)

    noise = noise_latent.to(device=device, dtype=dtype)
    if noise.ndim == 3:
        if tuple(noise.shape[:2]) != (batch_size, num_draws):
            raise ValueError(
                "noise_latent with shape [B,M,D] must match readout_token B/M: "
                f"noise={tuple(noise.shape)} B={batch_size} M={num_draws}"
            )
        noise = noise.reshape(batch_size * num_draws, int(noise.shape[-1]))
    elif noise.ndim == 2:
        if int(noise.shape[0]) == int(batch_size) and int(num_draws) == 1:
            pass
        elif int(noise.shape[0]) == int(batch_size * num_draws):
            pass
        else:
            raise ValueError(
                "noise_latent with shape [B,D] or [B*M,D] must match readout_token: "
                f"noise={tuple(noise.shape)} B={batch_size} M={num_draws}"
            )
    else:
        raise ValueError(f"noise_latent must have shape [B,M,D], [B,D], or [B*M,D], got {tuple(noise.shape)}")

    if int(noise.shape[-1]) != int(cfg.noise_dim):
        raise ValueError(f"noise_dim mismatch: got {int(noise.shape[-1])}, expected {cfg.noise_dim}")
    return noise


def _ensure_social_risk(
    social_risk_features: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_draws: int,
    num_agents: int,
    cfg: StudentHiddenAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(cfg.use_past_social_risk):
        return torch.zeros(batch_size * num_draws, num_agents, 0, device=device, dtype=dtype)
    if social_risk_features is None:
        raise ValueError("past_social_risk_features is required when use_past_social_risk=True")
    risk = social_risk_features.to(device=device, dtype=dtype)
    if risk.ndim != 3:
        raise ValueError(f"past_social_risk_features must have shape [B,A,C] or [B*M,A,C], got {tuple(risk.shape)}")
    if int(risk.shape[-1]) != int(cfg.social_risk_dim):
        raise ValueError(f"social_risk_dim mismatch: got {int(risk.shape[-1])}, expected {cfg.social_risk_dim}")
    if int(risk.shape[0]) == int(batch_size):
        if int(risk.shape[1]) != int(num_agents):
            raise ValueError(f"social risk agent count mismatch: {tuple(risk.shape)} vs num_agents={num_agents}")
        risk = risk[:, None].expand(batch_size, num_draws, num_agents, cfg.social_risk_dim)
        return risk.reshape(batch_size * num_draws, num_agents, cfg.social_risk_dim)
    if int(risk.shape[0]) == int(batch_size * num_draws):
        if int(risk.shape[1]) != int(num_agents):
            raise ValueError(f"social risk agent count mismatch: {tuple(risk.shape)} vs num_agents={num_agents}")
        return risk
    raise ValueError(
        "past_social_risk_features batch size must match B or B*M from readout_token: "
        f"risk={tuple(risk.shape)} B={batch_size} M={num_draws}"
    )


class StudentReadoutHiddenAdapter(nn.Module):
    """Zero-initialized residual adapter on student hidden tokens."""

    def __init__(self, config: Optional[StudentHiddenAdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or StudentHiddenAdapterConfig()
        cfg = self.config

        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        cond_dim = int(cfg.hidden_dim)
        mode_dim = cond_dim if bool(cfg.use_mode_embedding) else 0
        noise_dim = cond_dim if bool(cfg.use_noise) else 0
        social_risk_dim = cond_dim if bool(cfg.use_past_social_risk) else 0

        self.token_norm = nn.LayerNorm(int(cfg.token_dim))
        self.past_encoder = nn.Sequential(
            nn.LayerNorm(past_flat_dim),
            nn.Linear(past_flat_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if bool(cfg.use_noise):
            self.noise_encoder: nn.Module = nn.Sequential(
                nn.LayerNorm(int(cfg.noise_dim)),
                nn.Linear(int(cfg.noise_dim), cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.noise_encoder = nn.Identity()

        if bool(cfg.use_past_social_risk):
            self.social_risk_encoder: nn.Module = nn.Sequential(
                nn.LayerNorm(int(cfg.social_risk_dim)),
                nn.Linear(int(cfg.social_risk_dim), cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.social_gate_head: Optional[nn.Linear] = nn.Linear(cond_dim, 1)
        else:
            self.social_risk_encoder = nn.Identity()
            self.social_gate_head = None

        if bool(cfg.use_mode_embedding):
            self.mode_embedding = nn.Embedding(int(cfg.max_modes), cond_dim)
        else:
            self.mode_embedding = None

        adapter_input_dim = int(cfg.token_dim) + cond_dim + noise_dim + social_risk_dim + mode_dim
        self.adapter_mlp = nn.Sequential(
            nn.Linear(adapter_input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        if int(cfg.num_mode_context_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cond_dim,
                nhead=int(cfg.num_mode_context_heads),
                dim_feedforward=cond_dim * 4,
                dropout=float(cfg.mode_context_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mode_context = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg.num_mode_context_layers))
        else:
            self.mode_context = nn.Identity()

        self.delta_head = nn.Linear(cond_dim, int(cfg.token_dim))
        self.gate_head = nn.Linear(cond_dim, 1)
        self.last_gate: Optional[torch.Tensor] = None
        self.last_delta: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(self.config.gate_init_bias))
        if self.social_gate_head is not None:
            nn.init.zeros_(self.social_gate_head.weight)
            nn.init.zeros_(self.social_gate_head.bias)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(
        self,
        readout_token: torch.Tensor,
        *,
        past_traj_original_scale: Optional[torch.Tensor] = None,
        past_social_risk_features: Optional[torch.Tensor] = None,
        noise_latent: Optional[torch.Tensor] = None,
        x_data: Optional[Mapping[str, Any]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        cfg = self.config
        if x_data is not None and past_traj_original_scale is None and "past_traj_original_scale" in x_data:
            past_traj_original_scale = x_data["past_traj_original_scale"]
        if x_data is not None and past_social_risk_features is None and "past_social_risk_features" in x_data:
            past_social_risk_features = x_data["past_social_risk_features"]
        if past_traj_original_scale is None:
            raise ValueError("past_traj_original_scale is required for StudentReadoutHiddenAdapter")

        original_dtype = readout_token.dtype
        token5, had_m_dim = _ensure_token5(readout_token.to(dtype=torch.float32))
        batch_size, num_draws, num_modes, num_agents, token_dim = [int(item) for item in token5.shape]
        if token_dim != int(cfg.token_dim):
            raise ValueError(f"readout token dim mismatch: got {token_dim}, expected {cfg.token_dim}")
        if bool(cfg.use_mode_embedding) and num_modes > int(cfg.max_modes):
            raise ValueError(f"num_modes={num_modes} exceeds hidden adapter max_modes={cfg.max_modes}")

        past = _ensure_past(
            past_traj_original_scale,
            batch_size=batch_size,
            num_draws=num_draws,
            num_agents=num_agents,
            cfg=cfg,
            device=token5.device,
            dtype=token5.dtype,
        )
        noise = _ensure_noise(
            noise_latent,
            batch_size=batch_size,
            num_draws=num_draws,
            cfg=cfg,
            device=token5.device,
            dtype=token5.dtype,
        )
        social_risk = _ensure_social_risk(
            past_social_risk_features,
            batch_size=batch_size,
            num_draws=num_draws,
            num_agents=num_agents,
            cfg=cfg,
            device=token5.device,
            dtype=token5.dtype,
        )

        b_eff = batch_size * num_draws
        token = token5.reshape(b_eff, num_modes, num_agents, token_dim)
        token_features = self.token_norm(token)

        past_context = self.past_encoder(past.reshape(b_eff, num_agents, -1))
        past_context = past_context[:, None, :, :].expand(b_eff, num_modes, num_agents, -1)

        feature_parts = [token_features, past_context]
        if bool(cfg.use_noise):
            noise_context = self.noise_encoder(noise)
            noise_context = noise_context[:, None, None, :].expand(b_eff, num_modes, num_agents, -1)
            feature_parts.append(noise_context)
        social_context = None
        if bool(cfg.use_past_social_risk):
            social_context = self.social_risk_encoder(social_risk)
            feature_parts.append(social_context[:, None, :, :].expand(b_eff, num_modes, num_agents, -1))
        if bool(cfg.use_mode_embedding) and self.mode_embedding is not None:
            mode_ids = torch.arange(num_modes, device=token5.device)
            mode_context = self.mode_embedding(mode_ids).reshape(1, num_modes, 1, -1)
            mode_context = mode_context.expand(b_eff, num_modes, num_agents, -1)
            feature_parts.append(mode_context)

        hidden = self.adapter_mlp(torch.cat(feature_parts, dim=-1))
        hidden_for_context = hidden.permute(0, 2, 1, 3).reshape(b_eff * num_agents, num_modes, -1)
        hidden = self.mode_context(hidden_for_context)
        hidden = hidden.reshape(b_eff, num_agents, num_modes, -1).permute(0, 2, 1, 3)

        delta = self.delta_head(hidden)
        if cfg.max_token_delta is not None:
            max_delta = float(cfg.max_token_delta)
            delta = max_delta * torch.tanh(delta / max(max_delta, 1e-6))
        delta = float(cfg.residual_scale) * delta
        gate_logits = self.gate_head(hidden)
        if social_context is not None and self.social_gate_head is not None:
            gate_logits = gate_logits + self.social_gate_head(
                social_context[:, None, :, :].expand(b_eff, num_modes, num_agents, -1)
            )
        gate = torch.sigmoid(gate_logits)
        refined = token + gate * delta

        refined5 = refined.reshape(batch_size, num_draws, num_modes, num_agents, token_dim)
        delta5 = delta.reshape(batch_size, num_draws, num_modes, num_agents, token_dim)
        gate5 = gate.reshape(batch_size, num_draws, num_modes, num_agents, 1)
        self.last_delta = _restore_token_shape(delta5, had_m_dim=had_m_dim)
        self.last_gate = _restore_token_shape(gate5, had_m_dim=had_m_dim)

        refined_out = _restore_token_shape(refined5, had_m_dim=had_m_dim).to(dtype=original_dtype)
        if not return_dict:
            return refined_out
        return {
            "refined_readout_token": refined_out,
            "delta_hidden": self.last_delta,
            "gate": self.last_gate,
            "raw_readout_token": readout_token,
        }


def build_student_hidden_adapter_for_model(
    model: nn.Module,
    *,
    past_frames: int,
    past_feature_dim: int,
    adapter_site: str = "readout",
    hidden_dim: int = 128,
    max_modes: Optional[int] = None,
    use_noise: bool = True,
    use_past_social_risk: bool = False,
    social_risk_dim: int = 10,
    num_mode_context_layers: int = 1,
    num_mode_context_heads: int = 4,
    mode_context_dropout: float = 0.0,
    residual_scale: float = 1.0,
    max_token_delta: Optional[float] = 0.5,
    gate_init_bias: float = -2.0,
) -> StudentReadoutHiddenAdapter:
    if adapter_site not in {"readout", "query"}:
        raise ValueError(f"Unsupported adapter_site={adapter_site!r}; expected 'readout' or 'query'")
    token_dim = int(getattr(model, "dim"))
    inferred_modes = int(max_modes or getattr(getattr(model, "model_cfg", {}), "NUM_PROPOSED_QUERY", 20))
    cfg = StudentHiddenAdapterConfig(
        adapter_site=str(adapter_site),
        token_dim=token_dim,
        past_frames=int(past_frames),
        past_feature_dim=int(past_feature_dim),
        hidden_dim=int(hidden_dim),
        noise_dim=token_dim,
        social_risk_dim=int(social_risk_dim),
        max_modes=inferred_modes,
        use_noise=bool(use_noise),
        use_past_social_risk=bool(use_past_social_risk),
        use_mode_embedding=True,
        num_mode_context_layers=int(num_mode_context_layers),
        num_mode_context_heads=int(num_mode_context_heads),
        mode_context_dropout=float(mode_context_dropout),
        residual_scale=float(residual_scale),
        max_token_delta=max_token_delta,
        gate_init_bias=float(gate_init_bias),
    )
    return StudentReadoutHiddenAdapter(cfg)


def load_student_hidden_adapter(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StudentReadoutHiddenAdapter, Dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid student hidden adapter checkpoint payload: {type(payload)!r}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Student hidden adapter checkpoint is missing `config`")
    adapter = StudentReadoutHiddenAdapter(StudentHiddenAdapterConfig(**dict(config_payload)))
    state_dict = payload.get("model_state") or payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Student hidden adapter checkpoint is missing `model_state`")
    adapter.load_state_dict(state_dict)
    return adapter, dict(payload)


__all__ = [
    "StudentHiddenAdapterConfig",
    "StudentReadoutHiddenAdapter",
    "build_student_hidden_adapter_for_model",
    "load_student_hidden_adapter",
]
