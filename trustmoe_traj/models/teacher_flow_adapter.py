"""Teacher-side flow residual adapters.

This module implements a lightweight adapter for the MoFlow slow teacher.  The
adapter is inserted inside the teacher flow process, after the frozen teacher
predicts ``pred_data`` at a sampling/training time and before that prediction is
converted back into a velocity update.  The first version is deliberately
conservative: zero-initialized delta, low initial gate, and optional
observed-past social-risk conditioning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class TeacherFlowAdapterConfig:
    """Configuration for a frozen-teacher flow residual adapter."""

    future_frames: int = 12
    coord_dim: int = 2
    past_frames: int = 8
    past_feature_dim: int = 6
    social_risk_dim: int = 10
    temporal_energy_dim: int = 5
    hidden_dim: int = 128
    max_modes: int = 20
    use_past_social_risk: bool = True
    use_temporal_interaction_energy: bool = False
    use_mode_embedding: bool = True
    residual_scale: float = 1.0
    max_delta: Optional[float] = 0.10
    gate_init_bias: float = -2.0


def _ensure_pred5(tensor: torch.Tensor, cfg: TeacherFlowAdapterConfig, name: str) -> torch.Tensor:
    expected_flat = int(cfg.future_frames * cfg.coord_dim)
    if tensor.ndim == 4:
        if int(tensor.shape[-1]) != expected_flat:
            raise ValueError(f"{name} trailing dim mismatch: got {tuple(tensor.shape)}, expected {expected_flat}")
        return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], cfg.future_frames, cfg.coord_dim)
    if tensor.ndim == 5:
        if int(tensor.shape[-2]) != int(cfg.future_frames) or int(tensor.shape[-1]) != int(cfg.coord_dim):
            raise ValueError(
                f"{name} future shape mismatch: got {tuple(tensor.shape)}, "
                f"expected trailing ({cfg.future_frames}, {cfg.coord_dim})"
            )
        return tensor
    raise ValueError(f"{name} must have shape [B,K,A,F*D] or [B,K,A,F,D], got {tuple(tensor.shape)}")


def _ensure_past(
    past_traj_original_scale: torch.Tensor,
    *,
    batch_size: int,
    num_agents: int,
    cfg: TeacherFlowAdapterConfig,
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
            "past_traj_original_scale shape does not match teacher flow adapter config: "
            f"got {tuple(past.shape)}, expected trailing ({cfg.past_frames}, {cfg.past_feature_dim})"
        )
    return past


def _ensure_social_risk(
    social_risk_features: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_agents: int,
    cfg: TeacherFlowAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(cfg.use_past_social_risk):
        return torch.zeros(batch_size, num_agents, 0, device=device, dtype=dtype)
    if social_risk_features is None:
        raise ValueError("past_social_risk_features is required when use_past_social_risk=True")
    risk = social_risk_features.to(device=device, dtype=dtype)
    if risk.ndim != 3:
        raise ValueError(f"past_social_risk_features must have shape [B,A,C], got {tuple(risk.shape)}")
    if tuple(risk.shape[:2]) != (batch_size, num_agents):
        raise ValueError(f"social risk shape mismatch: {tuple(risk.shape)} vs B={batch_size} A={num_agents}")
    if int(risk.shape[-1]) != int(cfg.social_risk_dim):
        raise ValueError(f"social_risk_dim mismatch: got {int(risk.shape[-1])}, expected {cfg.social_risk_dim}")
    return risk


def _ensure_temporal_energy(
    temporal_energy_features: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_modes: int,
    num_agents: int,
    cfg: TeacherFlowAdapterConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not bool(cfg.use_temporal_interaction_energy):
        return torch.zeros(batch_size, num_modes, num_agents, cfg.future_frames, 0, device=device, dtype=dtype)
    if temporal_energy_features is None:
        raise ValueError(
            "temporal_interaction_energy_features is required when use_temporal_interaction_energy=True"
        )
    energy = temporal_energy_features.to(device=device, dtype=dtype)
    if energy.ndim == 4:
        if tuple(energy.shape[:3]) != (batch_size, num_modes, num_agents):
            if int(energy.shape[0]) != batch_size or int(energy.shape[1]) not in (1, num_modes) or int(energy.shape[2]) != num_agents:
                raise ValueError(
                    "temporal_interaction_energy_features shape mismatch: "
                    f"{tuple(energy.shape)} vs B={batch_size} K={num_modes} A={num_agents}"
                )
            if int(energy.shape[1]) == 1:
                energy = energy.expand(batch_size, num_modes, num_agents, int(energy.shape[-1]))
        energy = energy[:, :, :, None, :].expand(
            batch_size,
            num_modes,
            num_agents,
            int(cfg.future_frames),
            int(energy.shape[-1]),
        )
    elif energy.ndim == 5:
        if int(energy.shape[0]) != batch_size or int(energy.shape[2]) != num_agents:
            raise ValueError(
                "temporal_interaction_energy_features shape mismatch: "
                f"{tuple(energy.shape)} vs B={batch_size} A={num_agents}"
            )
        if int(energy.shape[1]) == 1 and num_modes > 1:
            energy = energy.expand(batch_size, num_modes, num_agents, int(energy.shape[3]), int(energy.shape[4]))
        elif int(energy.shape[1]) != num_modes:
            raise ValueError(f"temporal energy mode mismatch: got K={int(energy.shape[1])}, expected {num_modes}")
        if int(energy.shape[3]) != int(cfg.future_frames):
            raise ValueError(
                f"temporal energy time mismatch: got T={int(energy.shape[3])}, expected {cfg.future_frames}"
            )
    else:
        raise ValueError(
            "temporal_interaction_energy_features must have shape [B,K,A,C] or [B,K,A,T,C], "
            f"got {tuple(energy.shape)}"
        )
    if int(energy.shape[-1]) != int(cfg.temporal_energy_dim):
        raise ValueError(
            f"temporal_energy_dim mismatch: got {int(energy.shape[-1])}, expected {cfg.temporal_energy_dim}"
        )
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


class TeacherFlowAdapter(nn.Module):
    """Energy/risk-conditioned residual on teacher ``pred_data``."""

    def __init__(self, config: Optional[TeacherFlowAdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or TeacherFlowAdapterConfig()
        cfg = self.config

        flat_dim = int(cfg.future_frames * cfg.coord_dim)
        past_flat_dim = int(cfg.past_frames * cfg.past_feature_dim)
        cond_dim = int(cfg.hidden_dim)
        social_dim = cond_dim if bool(cfg.use_past_social_risk) else 0
        temporal_dim = cond_dim if bool(cfg.use_temporal_interaction_energy) else 0
        mode_dim = cond_dim if bool(cfg.use_mode_embedding) else 0

        self.pred_encoder = nn.Sequential(
            nn.LayerNorm(flat_dim * 2),
            nn.Linear(flat_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.past_encoder = nn.Sequential(
            nn.LayerNorm(past_flat_dim),
            nn.Linear(past_flat_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if bool(cfg.use_past_social_risk):
            self.social_risk_encoder: nn.Module = nn.Sequential(
                nn.LayerNorm(int(cfg.social_risk_dim)),
                nn.Linear(int(cfg.social_risk_dim), cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.social_risk_encoder = nn.Identity()

        if bool(cfg.use_temporal_interaction_energy):
            self.temporal_energy_encoder: nn.Module = nn.Sequential(
                nn.LayerNorm(int(cfg.temporal_energy_dim)),
                nn.Linear(int(cfg.temporal_energy_dim), cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.temporal_gate_head: Optional[nn.Linear] = nn.Linear(cond_dim, 1)
        else:
            self.temporal_energy_encoder = nn.Identity()
            self.temporal_gate_head = None

        if bool(cfg.use_mode_embedding):
            self.mode_embedding = nn.Embedding(int(cfg.max_modes), cond_dim)
        else:
            self.mode_embedding = None

        input_dim = cond_dim * 3 + social_dim + temporal_dim + mode_dim
        self.adapter_mlp = nn.Sequential(
            nn.Linear(input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(cond_dim, flat_dim)
        self.gate_head = nn.Linear(cond_dim, int(cfg.future_frames))
        self.last_delta: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(self.config.gate_init_bias))
        if self.temporal_gate_head is not None:
            nn.init.zeros_(self.temporal_gate_head.weight)
            nn.init.zeros_(self.temporal_gate_head.bias)

    @property
    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(
        self,
        pred_data: torch.Tensor,
        *,
        y_t: torch.Tensor,
        t: torch.Tensor,
        x_data: Mapping[str, Any],
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        cfg = self.config
        original_shape = pred_data.shape
        original_dtype = pred_data.dtype
        pred5 = _ensure_pred5(pred_data.to(dtype=torch.float32), cfg, "pred_data")
        y5 = _ensure_pred5(y_t.to(device=pred5.device, dtype=torch.float32), cfg, "y_t")
        batch_size, num_modes, num_agents, _future_frames, _coord_dim = [int(item) for item in pred5.shape]
        if bool(cfg.use_mode_embedding) and num_modes > int(cfg.max_modes):
            raise ValueError(f"num_modes={num_modes} exceeds max_modes={cfg.max_modes}")

        past = _ensure_past(
            x_data["past_traj_original_scale"],
            batch_size=batch_size,
            num_agents=num_agents,
            cfg=cfg,
            device=pred5.device,
            dtype=pred5.dtype,
        )
        social_risk = _ensure_social_risk(
            x_data.get("past_social_risk_features"),
            batch_size=batch_size,
            num_agents=num_agents,
            cfg=cfg,
            device=pred5.device,
            dtype=pred5.dtype,
        )
        temporal_energy_raw = None
        if "teacher_temporal_interaction_energy_features" in x_data:
            temporal_energy_raw = x_data["teacher_temporal_interaction_energy_features"]
        elif "temporal_interaction_energy_features" in x_data:
            temporal_energy_raw = x_data["temporal_interaction_energy_features"]
        temporal_energy = _ensure_temporal_energy(
            temporal_energy_raw,
            batch_size=batch_size,
            num_modes=num_modes,
            num_agents=num_agents,
            cfg=cfg,
            device=pred5.device,
            dtype=pred5.dtype,
        )

        pred_flat = pred5.reshape(batch_size, num_modes, num_agents, -1)
        y_flat = y5.reshape(batch_size, num_modes, num_agents, -1)
        pred_context = self.pred_encoder(torch.cat([pred_flat, y_flat], dim=-1))

        past_context = self.past_encoder(past.reshape(batch_size, num_agents, -1))
        past_context = past_context[:, None, :, :].expand(batch_size, num_modes, num_agents, -1)

        if t.ndim != 1 or int(t.shape[0]) != batch_size:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)} for B={batch_size}")
        time_context = self.time_encoder(t.to(device=pred5.device, dtype=pred5.dtype).reshape(batch_size, 1))
        time_context = time_context[:, None, None, :].expand(batch_size, num_modes, num_agents, -1)

        feature_parts = [pred_context, past_context, time_context]
        if bool(cfg.use_past_social_risk):
            social_context = self.social_risk_encoder(social_risk)
            feature_parts.append(social_context[:, None, :, :].expand(batch_size, num_modes, num_agents, -1))
        temporal_context = None
        if bool(cfg.use_temporal_interaction_energy):
            temporal_context = self.temporal_energy_encoder(temporal_energy)
            feature_parts.append(temporal_context.mean(dim=3))
        if bool(cfg.use_mode_embedding) and self.mode_embedding is not None:
            mode_ids = torch.arange(num_modes, device=pred5.device)
            mode_context = self.mode_embedding(mode_ids).reshape(1, num_modes, 1, -1)
            feature_parts.append(mode_context.expand(batch_size, num_modes, num_agents, -1))

        hidden = self.adapter_mlp(torch.cat(feature_parts, dim=-1))
        delta = self.delta_head(hidden).reshape(batch_size, num_modes, num_agents, cfg.future_frames, cfg.coord_dim)
        if cfg.max_delta is not None:
            max_delta = float(cfg.max_delta)
            delta = max_delta * torch.tanh(delta / max(max_delta, 1e-6))
        delta = float(cfg.residual_scale) * delta
        gate_logits = self.gate_head(hidden)
        if temporal_context is not None and self.temporal_gate_head is not None:
            gate_logits = gate_logits + self.temporal_gate_head(temporal_context).squeeze(-1)
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        refined = pred5 + gate * delta

        self.last_delta = delta
        self.last_gate = gate
        if len(original_shape) == 4:
            refined_out = refined.reshape(original_shape).to(dtype=original_dtype)
        else:
            refined_out = refined.to(dtype=original_dtype)
        if not return_dict:
            return refined_out
        return {
            "refined_pred_data": refined_out,
            "delta_pred_data": self.last_delta,
            "gate": self.last_gate,
            "raw_pred_data": pred_data,
        }


def build_teacher_flow_adapter_for_engine(
    engine: Any,
    *,
    past_frames: int,
    past_feature_dim: int,
    social_risk_dim: int = 10,
    temporal_energy_dim: int = 5,
    hidden_dim: int = 128,
    max_modes: Optional[int] = None,
    use_past_social_risk: bool = True,
    use_temporal_interaction_energy: bool = False,
    residual_scale: float = 1.0,
    max_delta: Optional[float] = 0.10,
    gate_init_bias: float = -2.0,
) -> TeacherFlowAdapter:
    cfg = getattr(engine, "cfg")
    future_frames = int(cfg.future_frames)
    model_cfg = getattr(cfg, "MODEL", {})
    model_out_dim = (
        model_cfg.get("MODEL_OUT_DIM", future_frames * 2)
        if isinstance(model_cfg, Mapping)
        else getattr(model_cfg, "MODEL_OUT_DIM", future_frames * 2)
    )
    model_out_dim = int(model_out_dim)
    if model_out_dim % future_frames != 0:
        raise ValueError(f"MODEL_OUT_DIM={model_out_dim} is not divisible by future_frames={future_frames}")
    coord_dim = int(model_out_dim // future_frames)
    inferred_modes = int(max_modes or getattr(cfg, "denoising_head_preds", 20))
    return TeacherFlowAdapter(
        TeacherFlowAdapterConfig(
            future_frames=future_frames,
            coord_dim=coord_dim,
            past_frames=int(past_frames),
            past_feature_dim=int(past_feature_dim),
            social_risk_dim=int(social_risk_dim),
            temporal_energy_dim=int(temporal_energy_dim),
            hidden_dim=int(hidden_dim),
            max_modes=inferred_modes,
            use_past_social_risk=bool(use_past_social_risk),
            use_temporal_interaction_energy=bool(use_temporal_interaction_energy),
            use_mode_embedding=True,
            residual_scale=float(residual_scale),
            max_delta=max_delta,
            gate_init_bias=float(gate_init_bias),
        )
    )


def load_teacher_flow_adapter(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TeacherFlowAdapter, Dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid teacher flow adapter checkpoint payload: {type(payload)!r}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Teacher flow adapter checkpoint is missing `config`")
    adapter = TeacherFlowAdapter(TeacherFlowAdapterConfig(**dict(config_payload)))
    state_dict = payload.get("model_state") or payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Teacher flow adapter checkpoint is missing `model_state`")
    adapter.load_state_dict(state_dict)
    return adapter, dict(payload)


__all__ = [
    "TeacherFlowAdapter",
    "TeacherFlowAdapterConfig",
    "build_teacher_flow_adapter_for_engine",
    "load_teacher_flow_adapter",
]
