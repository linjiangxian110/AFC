"""Residual Graduate model for offline teacher-student fusion.

The graduate model does not modify the MoFlow student backbone. It takes the
cached student trajectories and observed history features, predicts a gated
residual for every student candidate, and returns:

    graduate_pred = student_pred + gate * delta_pred

The internal learner is a pre-norm residual MLP stack. The residual and gate
heads are zero-initialized so the whole model starts as an identity wrapper
around the fast student.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from .interaction_energy import (
    InteractionEnergyConfig,
    InteractionEnergyFeatureBuilder,
    TemporalInteractionEnergyFeatureBuilder,
)


@dataclass
class ResidualGraduateConfig:
    """Configuration for the output-end residual graduate head."""

    pred_len: int = 12
    coord_dim: int = 2
    past_len: int = 8
    past_feature_dim: int = 6
    hidden_dim: int = 256
    num_residual_blocks: int = 4
    residual_block_type: str = "mlp"
    block_expansion: float = 2.0
    dropout: float = 0.0
    residual_scale: float = 1.0
    max_delta: Optional[float] = None
    use_social_context: bool = False
    context_dim: int = 128
    context_hidden_dim: int = 256
    context_num_layers: int = 2
    context_num_heads: int = 2
    context_dropout: float = 0.0
    use_interaction_energy: bool = False
    interaction_energy_dim: int = 5
    collision_sigma: float = 0.5
    collision_radius: float = 0.2
    no_neighbor_distance: float = 10.0
    interaction_energy_temporal_stride: int = 1
    use_energy_conditioned_heads: bool = False
    energy_condition_dim: int = 32
    use_time_aware_gate: bool = False
    use_mode_set_context: bool = False
    mode_context_num_layers: int = 1
    mode_context_num_heads: int = 4
    mode_context_dropout: float = 0.0
    use_best_mode_refiner: bool = False
    best_refine_scale: float = 1.0
    max_best_refine: Optional[float] = None
    use_temporal_energy_refiner: bool = False
    temporal_interaction_energy_dim: int = 5
    temporal_refiner_hidden_dim: int = 64
    temporal_refine_scale: float = 1.0
    max_temporal_refine: Optional[float] = None
    temporal_gate_init_bias: float = 0.0
    # Backward-compatible alias for older checkpoints / CLI arguments.
    num_hidden_layers: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_student_shape(student_pred: torch.Tensor) -> torch.Tensor:
    if student_pred.ndim == 4:
        return student_pred.unsqueeze(2)
    if student_pred.ndim == 5:
        return student_pred
    raise ValueError(
        "student_pred must have shape [B, K, T, 2] or [B, K, A, T, 2], "
        f"got {tuple(student_pred.shape)}"
    )


def _ensure_past_shape(past_traj_original_scale: torch.Tensor, *, batch_size: int, num_agents: int) -> torch.Tensor:
    if past_traj_original_scale.ndim != 4:
        raise ValueError(
            "past_traj_original_scale must have shape [B, A, P, F], "
            f"got {tuple(past_traj_original_scale.shape)}"
        )
    if int(past_traj_original_scale.shape[0]) != int(batch_size):
        raise ValueError(
            "past_traj_original_scale batch mismatch: "
            f"{int(past_traj_original_scale.shape[0])} vs {int(batch_size)}"
        )
    if int(past_traj_original_scale.shape[1]) != int(num_agents):
        raise ValueError(
            "past_traj_original_scale agent mismatch: "
            f"{int(past_traj_original_scale.shape[1])} vs {int(num_agents)}"
        )
    return past_traj_original_scale


class ResidualMLPBlock(nn.Module):
    """Pre-norm feed-forward residual block.

    Shape is preserved: [N, hidden_dim] -> [N, hidden_dim].
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(expansion) <= 0:
            raise ValueError("expansion must be positive")

        inner_dim = max(int(round(hidden_dim * float(expansion))), hidden_dim)
        layers = [
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(inner_dim, hidden_dim))
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class FiLMResidualMLPBlock(nn.Module):
    """Conditioned residual MLP block.

    The residual path is modulated by per-mode conditioning features. The FiLM
    projection is zero-initialized, so this block starts exactly as the plain
    ResidualMLPBlock and only learns conditioning when it helps.
    """

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        *,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if float(expansion) <= 0:
            raise ValueError("expansion must be positive")

        inner_dim = max(int(round(hidden_dim * float(expansion))), hidden_dim)
        layers = [
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(inner_dim, hidden_dim))
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(*layers)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.film = nn.Linear(condition_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(f"condition must have shape [N, C], got {tuple(condition.shape)}")
        if int(condition.shape[0]) != int(x.shape[0]):
            raise ValueError(
                "condition batch mismatch: "
                f"{int(condition.shape[0])} vs {int(x.shape[0])}"
            )

        residual = self.ffn(self.norm(x))
        gamma, beta = self.film(self.condition_norm(condition)).chunk(2, dim=-1)
        return x + residual * (1.0 + gamma) + beta


class SocialContextEncoder(nn.Module):
    """Lightweight agent-interaction encoder for observed ETH history."""

    def __init__(
        self,
        *,
        past_len: int,
        past_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(past_len) <= 0:
            raise ValueError("past_len must be positive")
        if int(past_feature_dim) <= 0:
            raise ValueError("past_feature_dim must be positive")
        if int(context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("context_hidden_dim must be positive")
        if int(num_layers) < 0:
            raise ValueError("context_num_layers must be non-negative")
        if int(num_heads) <= 0:
            raise ValueError("context_num_heads must be positive")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError(
                "context_hidden_dim must be divisible by context_num_heads, "
                f"got hidden_dim={int(hidden_dim)}, num_heads={int(num_heads)}"
            )

        self.past_flat_dim = int(past_len * past_feature_dim)
        self.input_proj = nn.Linear(self.past_flat_dim, int(hidden_dim), bias=False)
        if int(num_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=int(hidden_dim),
                nhead=int(num_heads),
                dim_feedforward=int(hidden_dim) * 2,
                dropout=float(dropout),
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        else:
            self.transformer = None
        self.output_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(context_dim)),
        )

    def forward(self, past_traj_original_scale: torch.Tensor) -> torch.Tensor:
        if past_traj_original_scale.ndim != 4:
            raise ValueError(
                "past_traj_original_scale must have shape [B, A, P, F], "
                f"got {tuple(past_traj_original_scale.shape)}"
            )
        batch_size, num_agents, _past_len, _past_feature_dim = past_traj_original_scale.shape
        past_flat = past_traj_original_scale.reshape(batch_size, num_agents, -1)
        if int(past_flat.shape[-1]) != self.past_flat_dim:
            raise ValueError(
                "past_traj_original_scale flattened feature mismatch: "
                f"got {int(past_flat.shape[-1])}, expected {self.past_flat_dim}"
            )

        encoded = self.input_proj(past_flat)
        if self.transformer is not None:
            encoded = encoded + self.transformer(encoded)
        return self.output_proj(encoded)


class ModeSetContextEncoder(nn.Module):
    """Self-attention over the candidate-mode set for each agent.

    The original residual graduate head encodes each candidate trajectory
    independently. This module lets the K candidates communicate before the
    residual/gate heads decide which modes to move.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_layers) <= 0:
            raise ValueError("mode_context_num_layers must be positive")
        if int(num_heads) <= 0:
            raise ValueError("mode_context_num_heads must be positive")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError(
                "hidden_dim must be divisible by mode_context_num_heads, "
                f"got hidden_dim={int(hidden_dim)}, num_heads={int(num_heads)}"
            )

        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, encoded: torch.Tensor, *, batch_size: int, num_modes: int, num_agents: int) -> torch.Tensor:
        if encoded.ndim != 2:
            raise ValueError(f"encoded must have shape [B*K*A, H], got {tuple(encoded.shape)}")
        expected_rows = int(batch_size) * int(num_modes) * int(num_agents)
        if int(encoded.shape[0]) != expected_rows:
            raise ValueError(
                "encoded row mismatch for mode set context: "
                f"got {int(encoded.shape[0])}, expected {expected_rows}"
            )

        hidden_dim = int(encoded.shape[-1])
        mode_tokens = encoded.reshape(batch_size, num_modes, num_agents, hidden_dim)
        mode_tokens = mode_tokens.permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_modes, hidden_dim)
        mode_tokens = self.transformer(mode_tokens)
        mode_tokens = self.output_norm(mode_tokens)
        return (
            mode_tokens.reshape(batch_size, num_agents, num_modes, hidden_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_modes * num_agents, hidden_dim)
        )


class ResidualGraduateModel(nn.Module):
    """A gated residual head with residual MLP blocks."""

    def __init__(self, config: ResidualGraduateConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            self.config = ResidualGraduateConfig()
        elif isinstance(config, ResidualGraduateConfig):
            self.config = config
        else:
            payload = dict(config)
            if "num_hidden_layers" in payload and "num_residual_blocks" not in payload:
                payload["num_residual_blocks"] = payload["num_hidden_layers"]
            self.config = ResidualGraduateConfig(**payload)

        cfg = self.config
        if cfg.num_hidden_layers is not None:
            cfg.num_residual_blocks = int(cfg.num_hidden_layers)
        cfg.residual_block_type = str(cfg.residual_block_type).lower()
        if cfg.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if cfg.coord_dim <= 0:
            raise ValueError("coord_dim must be positive")
        if cfg.past_len <= 0:
            raise ValueError("past_len must be positive")
        if cfg.past_feature_dim <= 0:
            raise ValueError("past_feature_dim must be positive")
        if cfg.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if cfg.num_residual_blocks <= 0:
            raise ValueError("num_residual_blocks must be positive")
        if cfg.residual_block_type not in {"mlp", "film"}:
            raise ValueError(
                "residual_block_type must be one of {'mlp', 'film'}, "
                f"got {cfg.residual_block_type!r}"
            )
        if float(cfg.block_expansion) <= 0:
            raise ValueError("block_expansion must be positive")
        if int(cfg.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(cfg.context_hidden_dim) <= 0:
            raise ValueError("context_hidden_dim must be positive")
        if int(cfg.context_num_layers) < 0:
            raise ValueError("context_num_layers must be non-negative")
        if int(cfg.context_num_heads) <= 0:
            raise ValueError("context_num_heads must be positive")
        if int(cfg.interaction_energy_dim) <= 0:
            raise ValueError("interaction_energy_dim must be positive")
        if float(cfg.collision_sigma) <= 0.0:
            raise ValueError("collision_sigma must be positive")
        if float(cfg.collision_radius) < 0.0:
            raise ValueError("collision_radius must be non-negative")
        if float(cfg.no_neighbor_distance) <= 0.0:
            raise ValueError("no_neighbor_distance must be positive")
        if int(cfg.interaction_energy_temporal_stride) <= 0:
            raise ValueError("interaction_energy_temporal_stride must be positive")
        if bool(cfg.use_energy_conditioned_heads) and not bool(cfg.use_interaction_energy):
            raise ValueError("use_energy_conditioned_heads requires use_interaction_energy=True")
        if int(cfg.energy_condition_dim) <= 0:
            raise ValueError("energy_condition_dim must be positive")
        if int(cfg.mode_context_num_layers) <= 0:
            raise ValueError("mode_context_num_layers must be positive")
        if int(cfg.mode_context_num_heads) <= 0:
            raise ValueError("mode_context_num_heads must be positive")
        if int(cfg.hidden_dim) % int(cfg.mode_context_num_heads) != 0:
            raise ValueError(
                "hidden_dim must be divisible by mode_context_num_heads, "
                f"got hidden_dim={int(cfg.hidden_dim)}, num_heads={int(cfg.mode_context_num_heads)}"
            )
        if float(cfg.best_refine_scale) < 0.0:
            raise ValueError("best_refine_scale must be non-negative")
        if cfg.max_best_refine is not None and float(cfg.max_best_refine) <= 0.0:
            raise ValueError("max_best_refine must be positive when provided")
        if int(cfg.temporal_interaction_energy_dim) <= 0:
            raise ValueError("temporal_interaction_energy_dim must be positive")
        if int(cfg.temporal_refiner_hidden_dim) <= 0:
            raise ValueError("temporal_refiner_hidden_dim must be positive")
        if float(cfg.temporal_refine_scale) < 0.0:
            raise ValueError("temporal_refine_scale must be non-negative")
        if cfg.max_temporal_refine is not None and float(cfg.max_temporal_refine) <= 0.0:
            raise ValueError("max_temporal_refine must be positive when provided")

        self.student_flat_dim = int(cfg.pred_len * cfg.coord_dim)
        self.past_flat_dim = int(cfg.past_len * cfg.past_feature_dim)
        self.social_context_encoder: Optional[SocialContextEncoder]
        if bool(cfg.use_social_context):
            self.social_context_encoder = SocialContextEncoder(
                past_len=int(cfg.past_len),
                past_feature_dim=int(cfg.past_feature_dim),
                context_dim=int(cfg.context_dim),
                hidden_dim=int(cfg.context_hidden_dim),
                num_layers=int(cfg.context_num_layers),
                num_heads=int(cfg.context_num_heads),
                dropout=float(cfg.context_dropout),
            )
            self.context_dim = int(cfg.context_dim)
        else:
            self.social_context_encoder = None
            self.context_dim = 0

        self.interaction_energy_builder: Optional[InteractionEnergyFeatureBuilder]
        if bool(cfg.use_interaction_energy):
            self.interaction_energy_builder = InteractionEnergyFeatureBuilder(
                InteractionEnergyConfig(
                    collision_sigma=float(cfg.collision_sigma),
                    collision_radius=float(cfg.collision_radius),
                    no_neighbor_distance=float(cfg.no_neighbor_distance),
                    temporal_stride=int(cfg.interaction_energy_temporal_stride),
                )
            )
            if int(cfg.interaction_energy_dim) != int(self.interaction_energy_builder.output_dim):
                raise ValueError(
                    "interaction_energy_dim must match InteractionEnergyFeatureBuilder.output_dim, "
                    f"got {int(cfg.interaction_energy_dim)} vs {int(self.interaction_energy_builder.output_dim)}"
                )
            self.interaction_energy_dim = int(cfg.interaction_energy_dim)
        else:
            self.interaction_energy_builder = None
            self.interaction_energy_dim = 0
        self.temporal_interaction_energy_builder: Optional[TemporalInteractionEnergyFeatureBuilder]
        if bool(cfg.use_temporal_energy_refiner):
            self.temporal_interaction_energy_builder = TemporalInteractionEnergyFeatureBuilder(
                InteractionEnergyConfig(
                    collision_sigma=float(cfg.collision_sigma),
                    collision_radius=float(cfg.collision_radius),
                    no_neighbor_distance=float(cfg.no_neighbor_distance),
                    temporal_stride=1,
                )
            )
            if int(cfg.temporal_interaction_energy_dim) != int(self.temporal_interaction_energy_builder.output_dim):
                raise ValueError(
                    "temporal_interaction_energy_dim must match TemporalInteractionEnergyFeatureBuilder.output_dim, "
                    f"got {int(cfg.temporal_interaction_energy_dim)} vs "
                    f"{int(self.temporal_interaction_energy_builder.output_dim)}"
                )
            self.temporal_interaction_energy_dim = int(cfg.temporal_interaction_energy_dim)
        else:
            self.temporal_interaction_energy_builder = None
            self.temporal_interaction_energy_dim = 0

        input_dim = self.student_flat_dim + self.past_flat_dim + 4 + self.context_dim + self.interaction_energy_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, int(cfg.hidden_dim)),
            nn.LayerNorm(int(cfg.hidden_dim)),
            nn.SiLU(),
        )
        self.condition_dim = 4 + self.context_dim + self.interaction_energy_dim
        if cfg.residual_block_type == "film":
            self.residual_blocks = nn.ModuleList(
                [
                    FiLMResidualMLPBlock(
                        int(cfg.hidden_dim),
                        condition_dim=int(self.condition_dim),
                        expansion=float(cfg.block_expansion),
                        dropout=float(cfg.dropout),
                    )
                    for _ in range(int(cfg.num_residual_blocks))
                ]
            )
        else:
            self.residual_blocks = nn.ModuleList(
                [
                    ResidualMLPBlock(
                        int(cfg.hidden_dim),
                        expansion=float(cfg.block_expansion),
                        dropout=float(cfg.dropout),
                    )
                    for _ in range(int(cfg.num_residual_blocks))
                ]
            )
        self.final_norm = nn.LayerNorm(int(cfg.hidden_dim))
        self.mode_set_context: Optional[ModeSetContextEncoder]
        if bool(cfg.use_mode_set_context):
            self.mode_set_context = ModeSetContextEncoder(
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.mode_context_num_layers),
                num_heads=int(cfg.mode_context_num_heads),
                dropout=float(cfg.mode_context_dropout),
            )
        else:
            self.mode_set_context = None

        self.delta_head = nn.Linear(int(cfg.hidden_dim), self.student_flat_dim)
        self.gate_head = nn.Linear(int(cfg.hidden_dim), 1)
        self.energy_condition: Optional[nn.Module]
        self.energy_delta_head: Optional[nn.Linear]
        self.energy_gate_head: Optional[nn.Linear]
        self.time_gate_head: Optional[nn.Linear]
        self.energy_time_gate_head: Optional[nn.Linear]
        self.best_mode_selector_head: Optional[nn.Linear]
        self.best_mode_endpoint_head: Optional[nn.Linear]
        self.temporal_energy_condition: Optional[nn.Module]
        self.temporal_context_proj: Optional[nn.Linear]
        self.temporal_delta_head: Optional[nn.Linear]
        self.temporal_gate_head: Optional[nn.Linear]
        if bool(cfg.use_energy_conditioned_heads):
            self.energy_condition = nn.Sequential(
                nn.LayerNorm(int(self.interaction_energy_dim)),
                nn.Linear(int(self.interaction_energy_dim), int(cfg.energy_condition_dim)),
                nn.SiLU(),
            )
            self.energy_delta_head = nn.Linear(int(cfg.energy_condition_dim), self.student_flat_dim)
            self.energy_gate_head = nn.Linear(int(cfg.energy_condition_dim), 1)
            if bool(cfg.use_time_aware_gate):
                self.energy_time_gate_head = nn.Linear(int(cfg.energy_condition_dim), int(cfg.pred_len))
            else:
                self.energy_time_gate_head = None
        else:
            self.energy_condition = None
            self.energy_delta_head = None
            self.energy_gate_head = None
            self.energy_time_gate_head = None
        if bool(cfg.use_time_aware_gate):
            self.time_gate_head = nn.Linear(int(cfg.hidden_dim), int(cfg.pred_len))
        else:
            self.time_gate_head = None
        if bool(cfg.use_best_mode_refiner):
            self.best_mode_selector_head = nn.Linear(int(cfg.hidden_dim), 1)
            self.best_mode_endpoint_head = nn.Linear(int(cfg.hidden_dim), int(cfg.coord_dim))
        else:
            self.best_mode_selector_head = None
            self.best_mode_endpoint_head = None
        if bool(cfg.use_temporal_energy_refiner):
            temporal_hidden_dim = int(cfg.temporal_refiner_hidden_dim)
            self.temporal_energy_condition = nn.Sequential(
                nn.LayerNorm(int(self.temporal_interaction_energy_dim)),
                nn.Linear(int(self.temporal_interaction_energy_dim), temporal_hidden_dim),
                nn.SiLU(),
            )
            self.temporal_context_proj = nn.Linear(int(cfg.hidden_dim), temporal_hidden_dim)
            self.temporal_delta_head = nn.Linear(temporal_hidden_dim, int(cfg.coord_dim))
            self.temporal_gate_head = nn.Linear(temporal_hidden_dim, 1)
        else:
            self.temporal_energy_condition = None
            self.temporal_context_proj = None
            self.temporal_delta_head = None
            self.temporal_gate_head = None
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start exactly from the fast student: delta=0, gate=0.5."""

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        for head in (
            self.energy_delta_head,
            self.energy_gate_head,
            self.energy_time_gate_head,
            self.time_gate_head,
            self.best_mode_selector_head,
            self.best_mode_endpoint_head,
            self.temporal_delta_head,
            self.temporal_gate_head,
        ):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.temporal_gate_head is not None:
            nn.init.constant_(self.temporal_gate_head.bias, float(self.config.temporal_gate_init_bias))

    def _build_features(
        self,
        student_pred: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        *,
        agent_mask: Optional[torch.Tensor] = None,
        interaction_energy_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cfg = self.config
        student = _ensure_student_shape(student_pred).to(dtype=torch.float32)
        batch_size, num_modes, num_agents, pred_len, coord_dim = student.shape
        if int(pred_len) != int(cfg.pred_len) or int(coord_dim) != int(cfg.coord_dim):
            raise ValueError(
                "student_pred shape does not match config: "
                f"got pred_len={int(pred_len)}, coord_dim={int(coord_dim)}, "
                f"expected pred_len={int(cfg.pred_len)}, coord_dim={int(cfg.coord_dim)}"
            )

        past = _ensure_past_shape(
            past_traj_original_scale.to(device=student.device, dtype=torch.float32),
            batch_size=int(batch_size),
            num_agents=int(num_agents),
        )
        if int(past.shape[2]) != int(cfg.past_len) or int(past.shape[3]) != int(cfg.past_feature_dim):
            raise ValueError(
                "past_traj_original_scale shape does not match config: "
                f"got past_len={int(past.shape[2])}, past_feature_dim={int(past.shape[3])}, "
                f"expected past_len={int(cfg.past_len)}, past_feature_dim={int(cfg.past_feature_dim)}"
            )

        student_flat = student.reshape(batch_size, num_modes, num_agents, -1)
        past_flat = past.reshape(batch_size, 1, num_agents, -1).expand(batch_size, num_modes, num_agents, -1)

        displacement = student[..., -1, :] - student[..., 0, :]
        endpoint = student[..., -1, :]
        summary = torch.cat([displacement, endpoint], dim=-1)
        feature_parts = [student_flat, past_flat, summary]
        condition_parts = [summary]
        context_feat: Optional[torch.Tensor] = None
        if self.social_context_encoder is not None:
            context_feat = self.social_context_encoder(past)
            context_flat = context_feat.reshape(batch_size, 1, num_agents, -1).expand(
                batch_size,
                num_modes,
                num_agents,
                -1,
            )
            feature_parts.append(context_flat)
            condition_parts.append(context_flat)

        energy_feat: Optional[torch.Tensor] = None
        if self.interaction_energy_builder is not None:
            if interaction_energy_features is not None:
                energy_feat = interaction_energy_features.to(device=student.device, dtype=torch.float32)
                expected_shape = (
                    int(batch_size),
                    int(num_modes),
                    int(num_agents),
                    int(self.interaction_energy_dim),
                )
                if tuple(energy_feat.shape) != expected_shape:
                    raise ValueError(
                        "interaction_energy_features must have shape [B, K, A, C], "
                        f"got {tuple(energy_feat.shape)}, expected {expected_shape}"
                    )
            else:
                past_abs = past[..., :2]
                student_abs = student + past_abs[:, None, :, -1:, :]
                energy_feat = self.interaction_energy_builder(
                    student_abs,
                    past_abs,
                    agent_mask=agent_mask,
                )
            feature_parts.append(energy_feat)
            condition_parts.append(energy_feat)

        features = torch.cat(feature_parts, dim=-1)
        condition = torch.cat(condition_parts, dim=-1)
        return student, features, condition, context_feat, energy_feat

    def forward(
        self,
        student_pred: torch.Tensor,
        past_traj_original_scale: torch.Tensor,
        *,
        agent_mask: Optional[torch.Tensor] = None,
        interaction_energy_features: Optional[torch.Tensor] = None,
        temporal_interaction_energy_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        student, features, block_condition, context_feat, energy_feat = self._build_features(
            student_pred,
            past_traj_original_scale,
            agent_mask=agent_mask,
            interaction_energy_features=interaction_energy_features,
        )
        batch_size, num_modes, num_agents, pred_len, coord_dim = student.shape

        encoded = self.input_proj(features.reshape(batch_size * num_modes * num_agents, -1))
        condition_flat = block_condition.reshape(batch_size * num_modes * num_agents, -1)
        for block in self.residual_blocks:
            if isinstance(block, FiLMResidualMLPBlock):
                encoded = block(encoded, condition_flat)
            else:
                encoded = block(encoded)
        encoded = self.final_norm(encoded)
        if self.mode_set_context is not None:
            encoded = self.mode_set_context(
                encoded,
                batch_size=int(batch_size),
                num_modes=int(num_modes),
                num_agents=int(num_agents),
            )

        energy_condition: Optional[torch.Tensor] = None
        if self.energy_condition is not None and energy_feat is not None:
            energy_condition = self.energy_condition(energy_feat.reshape(batch_size * num_modes * num_agents, -1))

        delta_raw = self.delta_head(encoded).reshape(batch_size, num_modes, num_agents, pred_len, coord_dim)
        energy_delta: Optional[torch.Tensor] = None
        if energy_condition is not None and self.energy_delta_head is not None:
            energy_delta = self.energy_delta_head(energy_condition).reshape(
                batch_size,
                num_modes,
                num_agents,
                pred_len,
                coord_dim,
            )
            delta_raw = delta_raw + energy_delta
        if self.config.max_delta is not None:
            max_delta = float(self.config.max_delta)
            delta_raw = max_delta * torch.tanh(delta_raw / max(max_delta, 1e-6))
        delta = float(self.config.residual_scale) * delta_raw

        gate_logits = self.gate_head(encoded)
        if energy_condition is not None and self.energy_gate_head is not None:
            gate_logits = gate_logits + self.energy_gate_head(energy_condition)
        if self.time_gate_head is not None:
            time_gate_logits = self.time_gate_head(encoded)
            if energy_condition is not None and self.energy_time_gate_head is not None:
                time_gate_logits = time_gate_logits + self.energy_time_gate_head(energy_condition)
            gate_logits = gate_logits + time_gate_logits
            gate = torch.sigmoid(gate_logits).reshape(batch_size, num_modes, num_agents, pred_len, 1)
        else:
            gate = torch.sigmoid(gate_logits).reshape(batch_size, num_modes, num_agents, 1, 1)
        graduate_pred = student + gate * delta
        temporal_energy_feat: Optional[torch.Tensor] = None
        temporal_repair_gate: Optional[torch.Tensor] = None
        temporal_refine_delta: Optional[torch.Tensor] = None
        if (
            self.temporal_energy_condition is not None
            and self.temporal_context_proj is not None
            and self.temporal_delta_head is not None
            and self.temporal_gate_head is not None
        ):
            if temporal_interaction_energy_features is not None:
                temporal_energy_feat = temporal_interaction_energy_features.to(device=student.device, dtype=torch.float32)
                expected_temporal_shape = (
                    int(batch_size),
                    int(num_modes),
                    int(num_agents),
                    int(pred_len),
                    int(self.temporal_interaction_energy_dim),
                )
                if tuple(temporal_energy_feat.shape) != expected_temporal_shape:
                    raise ValueError(
                        "temporal_interaction_energy_features must have shape [B, K, A, T, C], "
                        f"got {tuple(temporal_energy_feat.shape)}, expected {expected_temporal_shape}"
                    )
            elif self.temporal_interaction_energy_builder is not None:
                past = _ensure_past_shape(
                    past_traj_original_scale.to(device=student.device, dtype=torch.float32),
                    batch_size=int(batch_size),
                    num_agents=int(num_agents),
                )
                past_abs = past[..., :2]
                student_abs = student + past_abs[:, None, :, -1:, :]
                temporal_energy_feat = self.temporal_interaction_energy_builder(
                    student_abs,
                    past_abs,
                    agent_mask=agent_mask,
                )
            if temporal_energy_feat is not None:
                temporal_hidden = self.temporal_energy_condition(
                    temporal_energy_feat.reshape(batch_size * num_modes * num_agents * pred_len, -1)
                ).reshape(batch_size, num_modes, num_agents, pred_len, -1)
                temporal_context = self.temporal_context_proj(encoded).reshape(batch_size, num_modes, num_agents, 1, -1)
                temporal_hidden = torch.nn.functional.silu(temporal_hidden + temporal_context)
                temporal_delta = self.temporal_delta_head(
                    temporal_hidden.reshape(batch_size * num_modes * num_agents * pred_len, -1)
                ).reshape(batch_size, num_modes, num_agents, pred_len, coord_dim)
                if self.config.max_temporal_refine is not None:
                    max_temporal = float(self.config.max_temporal_refine)
                    temporal_delta = max_temporal * torch.tanh(temporal_delta / max(max_temporal, 1e-6))
                temporal_delta = float(self.config.temporal_refine_scale) * temporal_delta
                temporal_gate_logits = self.temporal_gate_head(
                    temporal_hidden.reshape(batch_size * num_modes * num_agents * pred_len, -1)
                )
                temporal_repair_gate = torch.sigmoid(temporal_gate_logits).reshape(
                    batch_size,
                    num_modes,
                    num_agents,
                    pred_len,
                    1,
                )
                temporal_refine_delta = temporal_repair_gate * temporal_delta
                graduate_pred = graduate_pred + temporal_refine_delta
        best_selector_logits: Optional[torch.Tensor] = None
        best_selector_prob: Optional[torch.Tensor] = None
        best_endpoint_refine: Optional[torch.Tensor] = None
        best_refine_delta: Optional[torch.Tensor] = None
        if self.best_mode_selector_head is not None and self.best_mode_endpoint_head is not None:
            best_selector_logits = self.best_mode_selector_head(encoded).reshape(batch_size, num_modes, num_agents)
            best_selector_prob = torch.sigmoid(best_selector_logits)
            best_endpoint_refine = self.best_mode_endpoint_head(encoded).reshape(
                batch_size,
                num_modes,
                num_agents,
                1,
                coord_dim,
            )
            if self.config.max_best_refine is not None:
                max_refine = float(self.config.max_best_refine)
                best_endpoint_refine = max_refine * torch.tanh(best_endpoint_refine / max(max_refine, 1e-6))
            best_endpoint_refine = float(self.config.best_refine_scale) * best_endpoint_refine
            time_ramp = torch.linspace(
                1.0 / float(pred_len),
                1.0,
                steps=int(pred_len),
                dtype=student.dtype,
                device=student.device,
            ).reshape(1, 1, 1, pred_len, 1)
            best_refine_delta = best_selector_prob[..., None, None] * best_endpoint_refine * time_ramp
            graduate_pred = graduate_pred + best_refine_delta
        result = {
            "graduate_pred": graduate_pred,
            "delta_pred": delta,
            "gate": gate,
        }
        if context_feat is not None:
            result["context_feat"] = context_feat
        if energy_feat is not None:
            result["interaction_energy_features"] = energy_feat
        if energy_delta is not None:
            result["energy_delta_pred"] = float(self.config.residual_scale) * energy_delta
        if best_selector_logits is not None and best_selector_prob is not None and best_refine_delta is not None:
            result["best_mode_selector_logits"] = best_selector_logits
            result["best_mode_selector_prob"] = best_selector_prob
            result["best_mode_endpoint_refine"] = best_endpoint_refine.squeeze(3)
            result["best_mode_refine_delta"] = best_refine_delta
        if temporal_energy_feat is not None:
            result["temporal_interaction_energy_features"] = temporal_energy_feat
        if temporal_repair_gate is not None and temporal_refine_delta is not None:
            result["temporal_repair_gate"] = temporal_repair_gate
            result["temporal_refine_delta"] = temporal_refine_delta
        return result


def build_residual_graduate_from_cache_shapes(
    tensor_shapes: Mapping[str, Any],
    *,
    hidden_dim: int = 256,
    num_residual_blocks: int = 4,
    residual_block_type: str = "mlp",
    block_expansion: float = 2.0,
    dropout: float = 0.0,
    residual_scale: float = 1.0,
    max_delta: Optional[float] = None,
    use_social_context: bool = False,
    context_dim: int = 128,
    context_hidden_dim: int = 256,
    context_num_layers: int = 2,
    context_num_heads: int = 2,
    context_dropout: float = 0.0,
    use_interaction_energy: bool = False,
    interaction_energy_dim: int = 5,
    collision_sigma: float = 0.5,
    collision_radius: float = 0.2,
    no_neighbor_distance: float = 10.0,
    interaction_energy_temporal_stride: int = 1,
    use_energy_conditioned_heads: bool = False,
    energy_condition_dim: int = 32,
    use_time_aware_gate: bool = False,
    use_mode_set_context: bool = False,
    mode_context_num_layers: int = 1,
    mode_context_num_heads: int = 4,
    mode_context_dropout: float = 0.0,
    use_best_mode_refiner: bool = False,
    best_refine_scale: float = 1.0,
    max_best_refine: Optional[float] = None,
    use_temporal_energy_refiner: bool = False,
    temporal_interaction_energy_dim: int = 5,
    temporal_refiner_hidden_dim: int = 64,
    temporal_refine_scale: float = 1.0,
    max_temporal_refine: Optional[float] = None,
    temporal_gate_init_bias: float = 0.0,
    num_hidden_layers: Optional[int] = None,
) -> ResidualGraduateModel:
    """Create a model using shape metadata from the teacher/student cache."""

    if num_hidden_layers is not None:
        num_residual_blocks = int(num_hidden_layers)

    student_shape = list(tensor_shapes["student_pred"])
    past_shape = list(tensor_shapes["past_traj_original_scale"])
    if len(student_shape) != 5:
        raise ValueError(f"Expected student_pred shape [N, K, A, T, 2], got {student_shape}")
    if len(past_shape) != 4:
        raise ValueError(f"Expected past_traj_original_scale shape [N, A, P, F], got {past_shape}")

    config = ResidualGraduateConfig(
        pred_len=int(student_shape[3]),
        coord_dim=int(student_shape[4]),
        past_len=int(past_shape[2]),
        past_feature_dim=int(past_shape[3]),
        hidden_dim=int(hidden_dim),
        num_residual_blocks=int(num_residual_blocks),
        residual_block_type=str(residual_block_type),
        block_expansion=float(block_expansion),
        dropout=float(dropout),
        residual_scale=float(residual_scale),
        max_delta=max_delta,
        use_social_context=bool(use_social_context),
        context_dim=int(context_dim),
        context_hidden_dim=int(context_hidden_dim),
        context_num_layers=int(context_num_layers),
        context_num_heads=int(context_num_heads),
        context_dropout=float(context_dropout),
        use_interaction_energy=bool(use_interaction_energy),
        interaction_energy_dim=int(interaction_energy_dim),
        collision_sigma=float(collision_sigma),
        collision_radius=float(collision_radius),
        no_neighbor_distance=float(no_neighbor_distance),
        interaction_energy_temporal_stride=int(interaction_energy_temporal_stride),
        use_energy_conditioned_heads=bool(use_energy_conditioned_heads),
        energy_condition_dim=int(energy_condition_dim),
        use_time_aware_gate=bool(use_time_aware_gate),
        use_mode_set_context=bool(use_mode_set_context),
        mode_context_num_layers=int(mode_context_num_layers),
        mode_context_num_heads=int(mode_context_num_heads),
        mode_context_dropout=float(mode_context_dropout),
        use_best_mode_refiner=bool(use_best_mode_refiner),
        best_refine_scale=float(best_refine_scale),
        max_best_refine=max_best_refine,
        use_temporal_energy_refiner=bool(use_temporal_energy_refiner),
        temporal_interaction_energy_dim=int(temporal_interaction_energy_dim),
        temporal_refiner_hidden_dim=int(temporal_refiner_hidden_dim),
        temporal_refine_scale=float(temporal_refine_scale),
        max_temporal_refine=max_temporal_refine,
        temporal_gate_init_bias=float(temporal_gate_init_bias),
    )
    return ResidualGraduateModel(config)


__all__ = [
    "ResidualGraduateConfig",
    "ResidualMLPBlock",
    "FiLMResidualMLPBlock",
    "ModeSetContextEncoder",
    "SocialContextEncoder",
    "ResidualGraduateModel",
    "build_residual_graduate_from_cache_shapes",
]
