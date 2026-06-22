"""TrustMoE-Traj 标准样本到 MoFlow ETH 输入格式的适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .adapters import ETHAdapterConfig, ETHTrajectoryDataset

try:  # pragma: no cover - torch 可能尚未安装
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[override]
        """最小 Dataset 占位。"""

        pass


DEFAULT_MOFLOW_DATA_NORM = "original"
DEFAULT_MOFLOW_PAST_DIM = 6
PAST_SOCIAL_RISK_FEATURE_NAMES = (
    "nearest_distance",
    "nearest_distance_inv",
    "mean_distance_inv",
    "close_count_0p5",
    "close_count_1p0",
    "soft_density_sigma1",
    "max_approaching_speed",
    "mean_approaching_speed",
    "min_tca_inv",
    "min_constvel_distance_inv",
)
DEFAULT_PAST_SOCIAL_RISK_DIM = len(PAST_SOCIAL_RISK_FEATURE_NAMES)
SUPPORTED_MOFLOW_DATA_NORMS = ("original", "min_max")
DEFAULT_MOFLOW_SAMPLE_MODE = "per_agent"
SUPPORTED_MOFLOW_SAMPLE_MODES = ("per_agent", "per_scene")


@dataclass(frozen=True)
class MoFlowETHTransformConfig:
    """MoFlow ETH 输入适配配置。"""

    data_norm: str = DEFAULT_MOFLOW_DATA_NORM
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE
    rotate: bool = False
    rotate_time_frame: int = 0
    fixed_num_agents: Optional[int] = None
    as_torch: bool = True
    past_traj_min: Optional[float] = None
    past_traj_max: Optional[float] = None
    fut_traj_min: Optional[float] = None
    fut_traj_max: Optional[float] = None

    def normalization_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if self.past_traj_min is not None:
            stats["past_traj_min"] = float(self.past_traj_min)
        if self.past_traj_max is not None:
            stats["past_traj_max"] = float(self.past_traj_max)
        if self.fut_traj_min is not None:
            stats["fut_traj_min"] = float(self.fut_traj_min)
        if self.fut_traj_max is not None:
            stats["fut_traj_max"] = float(self.fut_traj_max)
        return stats


def _validate_data_norm(data_norm: str) -> None:
    if data_norm not in SUPPORTED_MOFLOW_DATA_NORMS:
        raise ValueError(
            f"Unsupported MoFlow ETH data_norm: {data_norm!r}. "
            f"Expected one of {SUPPORTED_MOFLOW_DATA_NORMS}"
        )


def _validate_sample_mode(sample_mode: str) -> None:
    if sample_mode not in SUPPORTED_MOFLOW_SAMPLE_MODES:
        raise ValueError(
            f"Unsupported MoFlow ETH sample_mode: {sample_mode!r}. "
            f"Expected one of {SUPPORTED_MOFLOW_SAMPLE_MODES}"
        )


def _to_numpy(array_like: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if _TORCH_AVAILABLE and torch is not None and isinstance(array_like, torch.Tensor):
        array = array_like.detach().cpu().numpy()
    else:
        array = np.asarray(array_like)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _sample_to_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    raise TypeError(f"Unsupported sample type for MoFlow transform: {type(sample)!r}")


def _resolve_active_agent_indices(mapping: Mapping[str, Any]) -> List[int]:
    past = _to_numpy(mapping["past_traj"])
    agent_mask_raw = mapping.get("agent_mask")
    if agent_mask_raw is None:
        return list(range(int(past.shape[0])))

    agent_mask = _to_numpy(agent_mask_raw, dtype=np.int64).reshape(-1)
    if agent_mask.shape[0] != past.shape[0]:
        raise ValueError(
            f"agent_mask length mismatch: expected {past.shape[0]}, got {agent_mask.shape[0]}"
        )
    active = [idx for idx, flag in enumerate(agent_mask.tolist()) if int(flag) != 0]
    return active or list(range(int(past.shape[0])))


def _normalize_min_max(array: np.ndarray, min_val: float, max_val: float, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    denom = float(max_val) - float(min_val)
    if abs(denom) < 1e-12:
        raise ValueError("Cannot apply min-max normalization when max equals min")
    return ((b - a) * (array - float(min_val)) / denom + a).astype(np.float32, copy=False)


def _pad_or_truncate_agent_axis(array: np.ndarray, target_agents: int, pad_value: float = 0.0) -> np.ndarray:
    if target_agents <= 0:
        raise ValueError(f"target_agents must be positive, got {target_agents}")

    num_agents = int(array.shape[0])
    if num_agents == target_agents:
        return array
    if num_agents > target_agents:
        return array[:target_agents]

    pad_shape = (target_agents - num_agents, *array.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


def compute_past_social_risk_features(
    past_traj: Any,
    agent_mask: Optional[Any] = None,
    *,
    no_neighbor_distance: float = 10.0,
    close_radius_small: float = 0.5,
    close_radius_large: float = 1.0,
    density_sigma: float = 1.0,
    horizon_steps: float = 12.0,
) -> np.ndarray:
    """Compute lightweight observed-past social-risk features per agent.

    The features use only observed past positions and constant-velocity
    extrapolation from the last observed displacement.  They are intentionally
    low dimensional so they can condition the student hidden adapter without
    introducing a full future energy map.
    """

    past = _to_numpy(past_traj, dtype=np.float32)
    if past.ndim != 3 or past.shape[-1] != 2:
        raise ValueError(f"past_traj must have shape [A, P, 2], got {past.shape}")
    num_agents = int(past.shape[0])
    if agent_mask is None:
        active_mask = np.ones((num_agents,), dtype=bool)
    else:
        active_mask = _to_numpy(agent_mask, dtype=np.int64).reshape(-1).astype(bool)
        if int(active_mask.shape[0]) != num_agents:
            raise ValueError(f"agent_mask length mismatch: {int(active_mask.shape[0])} vs {num_agents}")

    current_pos = past[:, -1, :]
    if int(past.shape[1]) >= 2:
        current_vel = past[:, -1, :] - past[:, -2, :]
    else:
        current_vel = np.zeros_like(current_pos)

    features = np.zeros((num_agents, DEFAULT_PAST_SOCIAL_RISK_DIM), dtype=np.float32)
    eps = 1e-6
    no_neighbor_distance = float(no_neighbor_distance)
    for agent_idx in range(num_agents):
        if not bool(active_mask[agent_idx]):
            continue
        neighbor_mask = active_mask.copy()
        neighbor_mask[agent_idx] = False
        neighbor_indices = np.nonzero(neighbor_mask)[0]
        if int(neighbor_indices.shape[0]) <= 0:
            continue

        rel_pos = current_pos[neighbor_indices] - current_pos[agent_idx : agent_idx + 1]
        rel_vel = current_vel[neighbor_indices] - current_vel[agent_idx : agent_idx + 1]
        dist = np.linalg.norm(rel_pos, axis=-1).astype(np.float32)
        dist_safe = np.maximum(dist, eps)
        nearest = float(np.min(dist))
        inv_dist = 1.0 / (1.0 + dist)

        closing_speed = -np.sum(rel_pos * rel_vel, axis=-1) / dist_safe
        approaching = np.maximum(closing_speed, 0.0)
        rel_speed_sq = np.sum(rel_vel * rel_vel, axis=-1)
        tca = -np.sum(rel_pos * rel_vel, axis=-1) / np.maximum(rel_speed_sq, eps)
        tca = np.clip(tca, 0.0, float(horizon_steps))
        closest_pos = rel_pos + tca[:, None] * rel_vel
        closest_dist = np.linalg.norm(closest_pos, axis=-1).astype(np.float32)

        valid_tca = (closing_speed > 0.0) & (rel_speed_sq > eps)
        if np.any(valid_tca):
            min_tca = float(np.min(tca[valid_tca]))
            min_constvel = float(np.min(closest_dist[valid_tca]))
        else:
            min_tca = float(horizon_steps)
            min_constvel = no_neighbor_distance

        features[agent_idx] = np.asarray(
            [
                min(nearest, no_neighbor_distance) / max(no_neighbor_distance, eps),
                float(np.max(1.0 / (1.0 + dist))),
                float(np.mean(inv_dist)),
                float(np.sum(dist < float(close_radius_small))) / max(float(num_agents - 1), 1.0),
                float(np.sum(dist < float(close_radius_large))) / max(float(num_agents - 1), 1.0),
                float(np.sum(np.exp(-(dist ** 2) / max(2.0 * float(density_sigma) ** 2, eps)))),
                float(np.max(approaching)),
                float(np.mean(approaching)),
                1.0 / (1.0 + min_tca),
                1.0 / (1.0 + min_constvel),
            ],
            dtype=np.float32,
        )
    return features


def rotate_moflow_eth_trajectories(
    past_rel: Any,
    future_rel: Any,
    past_abs: Any,
    *,
    rotate_time_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现 MoFlow ETH dataloader 中的旋转逻辑。"""

    past_rel_np = _to_numpy(past_rel, dtype=np.float32)
    future_rel_np = _to_numpy(future_rel, dtype=np.float32)
    past_abs_np = _to_numpy(past_abs, dtype=np.float32)

    if past_rel_np.ndim != 3 or future_rel_np.ndim != 3 or past_abs_np.ndim != 3:
        raise ValueError("rotate_moflow_eth_trajectories expects [A, T, 2] arrays")

    if not (0 <= rotate_time_frame < past_rel_np.shape[1]):
        raise ValueError(
            f"rotate_time_frame out of range: {rotate_time_frame}, expected [0, {past_rel_np.shape[1] - 1}]"
        )

    past_diff = past_rel_np[:, rotate_time_frame, :]
    past_theta = np.arctan2(past_diff[:, 1], past_diff[:, 0] + 1e-5)

    rotate_matrix = np.zeros((past_theta.shape[0], 2, 2), dtype=np.float32)
    rotate_matrix[:, 0, 0] = np.cos(past_theta)
    rotate_matrix[:, 0, 1] = np.sin(past_theta)
    rotate_matrix[:, 1, 0] = -np.sin(past_theta)
    rotate_matrix[:, 1, 1] = np.cos(past_theta)

    past_after = np.matmul(rotate_matrix, np.swapaxes(past_rel_np, 1, 2)).swapaxes(1, 2)
    future_after = np.matmul(rotate_matrix, np.swapaxes(future_rel_np, 1, 2)).swapaxes(1, 2)
    past_abs_after = np.matmul(rotate_matrix, np.swapaxes(past_abs_np, 1, 2)).swapaxes(1, 2)
    return past_after.astype(np.float32), future_after.astype(np.float32), past_abs_after.astype(np.float32)


def build_moflow_eth_feature_arrays(
    past_traj: Any,
    future_traj: Any,
    *,
    rotate: bool = False,
    rotate_time_frame: int = 0,
) -> Dict[str, np.ndarray]:
    """把标准轨迹样本转换为 MoFlow ETH 需要的中间特征。"""

    past_abs = _to_numpy(past_traj, dtype=np.float32)
    future_abs = _to_numpy(future_traj, dtype=np.float32)

    if past_abs.ndim != 3 or past_abs.shape[-1] != 2:
        raise ValueError(f"past_traj must have shape [A, P, 2], got {past_abs.shape}")
    if future_abs.ndim != 3 or future_abs.shape[-1] != 2:
        raise ValueError(f"future_traj must have shape [A, F, 2], got {future_abs.shape}")
    if past_abs.shape[0] != future_abs.shape[0]:
        raise ValueError(
            f"past_traj / future_traj agent count mismatch: {past_abs.shape[0]} vs {future_abs.shape[0]}"
        )

    initial_pos = past_abs[:, -1:, :]
    past_rel = (past_abs - initial_pos).astype(np.float32, copy=False)
    future_rel = (future_abs - initial_pos).astype(np.float32, copy=False)

    if rotate:
        past_rel, future_rel, past_abs = rotate_moflow_eth_trajectories(
            past_rel,
            future_rel,
            past_abs,
            rotate_time_frame=rotate_time_frame,
        )

    past_vel = np.concatenate(
        [past_rel[:, 1:, :] - past_rel[:, :-1, :], np.zeros_like(past_rel[:, -1:, :])],
        axis=1,
    ).astype(np.float32, copy=False)
    future_vel = np.concatenate(
        [future_rel[:, 1:, :] - future_rel[:, :-1, :], np.zeros_like(future_rel[:, -1:, :])],
        axis=1,
    ).astype(np.float32, copy=False)

    past_feature = np.concatenate([past_abs, past_rel, past_vel], axis=-1).astype(np.float32, copy=False)
    return {
        "past_traj_original_scale": past_feature,
        "fut_traj_original_scale": future_rel,
        "fut_traj_vel": future_vel,
    }


def infer_moflow_eth_fixed_num_agents(samples: Sequence[Any]) -> int:
    """根据样本列表推断适配到 MoFlow 时的固定 agent 数。"""

    max_agents = 0
    for sample in samples:
        mapping = _sample_to_mapping(sample)
        past = _to_numpy(mapping["past_traj"])
        max_agents = max(max_agents, int(past.shape[0]))
    if max_agents <= 0:
        raise ValueError("Cannot infer fixed_num_agents from an empty sample collection")
    return max_agents


def infer_moflow_eth_num_agents(
    samples: Sequence[Any],
    *,
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE,
) -> int:
    """推断适配到 MoFlow 后的 agent 维度。"""

    _validate_sample_mode(sample_mode)
    if sample_mode == "per_agent":
        return 1
    return infer_moflow_eth_fixed_num_agents(samples)


def compute_moflow_eth_norm_stats(
    samples: Sequence[Any],
    *,
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE,
    rotate: bool = False,
    rotate_time_frame: int = 0,
    fixed_num_agents: Optional[int] = None,
) -> Dict[str, float]:
    """统计 MoFlow ETH 所需的 min-max 归一化参数。"""

    if not samples:
        raise ValueError("compute_moflow_eth_norm_stats received an empty sample list")

    _validate_sample_mode(sample_mode)

    past_min = np.inf
    past_max = -np.inf
    fut_min = np.inf
    fut_max = -np.inf
    agent_count_max = 0

    for sample in samples:
        mapping = _sample_to_mapping(sample)
        past_all = _to_numpy(mapping["past_traj"], dtype=np.float32)
        future_all = _to_numpy(mapping["future_traj"], dtype=np.float32)

        if sample_mode == "per_agent":
            candidate_indices = _resolve_active_agent_indices(mapping)
        else:
            candidate_indices = [-1]

        for agent_index in candidate_indices:
            if sample_mode == "per_agent":
                past = past_all[agent_index : agent_index + 1]
                future = future_all[agent_index : agent_index + 1]
            else:
                past = past_all
                future = future_all
                if fixed_num_agents is not None:
                    past = past[:fixed_num_agents]
                    future = future[:fixed_num_agents]

            features = build_moflow_eth_feature_arrays(
                past,
                future,
                rotate=rotate,
                rotate_time_frame=rotate_time_frame,
            )
            past_feature = features["past_traj_original_scale"]
            future_feature = features["fut_traj_original_scale"]

            past_min = min(past_min, float(past_feature.min()))
            past_max = max(past_max, float(past_feature.max()))
            fut_min = min(fut_min, float(future_feature.min()))
            fut_max = max(fut_max, float(future_feature.max()))
            agent_count_max = max(agent_count_max, int(past.shape[0]))

    return {
        "past_traj_min": float(past_min),
        "past_traj_max": float(past_max),
        "fut_traj_min": float(fut_min),
        "fut_traj_max": float(fut_max),
        "max_agents": float(agent_count_max),
    }


def build_moflow_eth_sample(
    sample: Any,
    *,
    index: int = 0,
    data_norm: str = DEFAULT_MOFLOW_DATA_NORM,
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE,
    agent_index: Optional[int] = None,
    rotate: bool = False,
    rotate_time_frame: int = 0,
    fixed_num_agents: Optional[int] = None,
    normalization_stats: Optional[Mapping[str, float]] = None,
    as_torch: bool = True,
) -> Dict[str, Any]:
    """将单个 TrustMoE 标准样本转换成 MoFlow ETH 单样本格式。"""

    _validate_data_norm(data_norm)
    _validate_sample_mode(sample_mode)

    mapping = _sample_to_mapping(sample)
    past_all = _to_numpy(mapping["past_traj"], dtype=np.float32)
    future_all = _to_numpy(mapping["future_traj"], dtype=np.float32)
    raw_agent_mask_all = _to_numpy(
        mapping.get("agent_mask", np.ones((past_all.shape[0],), dtype=np.int64)),
        dtype=np.int64,
    )
    social_risk_all = compute_past_social_risk_features(past_all, raw_agent_mask_all)

    if sample_mode == "per_agent":
        active_indices = _resolve_active_agent_indices(mapping)
        if agent_index is None:
            raise ValueError("agent_index is required when sample_mode='per_agent'")
        if agent_index not in active_indices:
            raise ValueError(f"agent_index {agent_index} is not an active agent in the provided sample")
        past = past_all[agent_index : agent_index + 1]
        future = future_all[agent_index : agent_index + 1]
        social_risk = social_risk_all[agent_index : agent_index + 1]
        raw_agent_mask = np.ones((1,), dtype=np.int64)
        target_agents = 1
        effective_agents = 1
    else:
        past = past_all
        future = future_all
        social_risk = social_risk_all
        raw_agent_mask = raw_agent_mask_all
        target_agents = fixed_num_agents or int(past.shape[0])
        effective_agents = min(int(past.shape[0]), int(target_agents))

    features = build_moflow_eth_feature_arrays(
        past,
        future,
        rotate=rotate,
        rotate_time_frame=rotate_time_frame,
    )

    past_feature = _pad_or_truncate_agent_axis(features["past_traj_original_scale"], target_agents)
    future_feature = _pad_or_truncate_agent_axis(features["fut_traj_original_scale"], target_agents)
    future_vel = _pad_or_truncate_agent_axis(features["fut_traj_vel"], target_agents)
    past_social_risk_features = _pad_or_truncate_agent_axis(social_risk, target_agents)
    agent_mask = _pad_or_truncate_agent_axis(raw_agent_mask.reshape(-1), target_agents, pad_value=0).astype(np.int64)
    if effective_agents < target_agents:
        agent_mask[effective_agents:] = 0

    if data_norm == "min_max":
        if normalization_stats is None:
            raise ValueError("normalization_stats is required when data_norm='min_max'")
        past_norm = _normalize_min_max(
            past_feature,
            float(normalization_stats["past_traj_min"]),
            float(normalization_stats["past_traj_max"]),
        )
        future_norm = _normalize_min_max(
            future_feature,
            float(normalization_stats["fut_traj_min"]),
            float(normalization_stats["fut_traj_max"]),
        )
    else:
        past_norm = past_feature.copy()
        future_norm = future_feature.copy()

    scene_meta = mapping.get("scene_meta", {})
    if isinstance(scene_meta, Mapping):
        scene_meta_payload: Dict[str, Any] = dict(scene_meta)
    elif hasattr(scene_meta, "to_dict"):
        scene_meta_payload = scene_meta.to_dict()
    else:
        scene_meta_payload = {"raw_scene_meta": scene_meta}

    payload: Dict[str, Any] = {
        "index": np.asarray([index], dtype=np.int32),
        "past_traj": past_norm.astype(np.float32, copy=False),
        "fut_traj": future_norm.astype(np.float32, copy=False),
        "past_traj_original_scale": past_feature.astype(np.float32, copy=False),
        "past_social_risk_features": past_social_risk_features.astype(np.float32, copy=False),
        "fut_traj_original_scale": future_feature.astype(np.float32, copy=False),
        "fut_traj_vel": future_vel.astype(np.float32, copy=False),
        "agent_mask": agent_mask,
        "num_agents": np.asarray(effective_agents, dtype=np.int64),
        "sample_mode": sample_mode,
        "target_agent_index": np.asarray(-1 if agent_index is None else agent_index, dtype=np.int64),
        "scene_meta": scene_meta_payload,
    }

    if as_torch and _TORCH_AVAILABLE and torch is not None:
        payload = {
            "index": torch.tensor([index], dtype=torch.int32),
            "past_traj": torch.from_numpy(payload["past_traj"]),
            "fut_traj": torch.from_numpy(payload["fut_traj"]),
            "past_traj_original_scale": torch.from_numpy(payload["past_traj_original_scale"]),
            "past_social_risk_features": torch.from_numpy(payload["past_social_risk_features"]),
            "fut_traj_original_scale": torch.from_numpy(payload["fut_traj_original_scale"]),
            "fut_traj_vel": torch.from_numpy(payload["fut_traj_vel"]),
            "agent_mask": torch.from_numpy(payload["agent_mask"]),
            "num_agents": torch.tensor(effective_agents, dtype=torch.int64),
            "sample_mode": sample_mode,
            "target_agent_index": torch.tensor(-1 if agent_index is None else agent_index, dtype=torch.int64),
            "scene_meta": scene_meta_payload,
        }
    return payload


def seq_collate_moflow_eth(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """把单样本适配结果堆叠成 MoFlow ETH trainer 需要的 batch dict。"""

    if not batch:
        raise ValueError("seq_collate_moflow_eth received an empty batch")

    if _TORCH_AVAILABLE and torch is not None and isinstance(batch[0]["past_traj"], torch.Tensor):
        indexes = torch.stack([item["index"] for item in batch], dim=0)
        past_traj = torch.stack([item["past_traj"] for item in batch], dim=0)
        fut_traj = torch.stack([item["fut_traj"] for item in batch], dim=0)
        past_traj_original_scale = torch.stack([item["past_traj_original_scale"] for item in batch], dim=0)
        past_social_risk_features = torch.stack([item["past_social_risk_features"] for item in batch], dim=0)
        fut_traj_original_scale = torch.stack([item["fut_traj_original_scale"] for item in batch], dim=0)
        fut_traj_vel = torch.stack([item["fut_traj_vel"] for item in batch], dim=0)
        agent_mask = torch.stack([item["agent_mask"] for item in batch], dim=0)
        num_agents = torch.stack([item["num_agents"] for item in batch], dim=0)
        target_agent_indices = torch.stack([
            item["target_agent_index"] if isinstance(item["target_agent_index"], torch.Tensor)
            else torch.tensor(item["target_agent_index"], dtype=torch.int64)
            for item in batch
        ], dim=0)
        batch_size = torch.tensor(past_traj.shape[0], dtype=torch.int64)
    else:
        indexes = np.stack([_to_numpy(item["index"], dtype=np.int32) for item in batch], axis=0)
        past_traj = np.stack([_to_numpy(item["past_traj"], dtype=np.float32) for item in batch], axis=0)
        fut_traj = np.stack([_to_numpy(item["fut_traj"], dtype=np.float32) for item in batch], axis=0)
        past_traj_original_scale = np.stack(
            [_to_numpy(item["past_traj_original_scale"], dtype=np.float32) for item in batch],
            axis=0,
        )
        past_social_risk_features = np.stack(
            [_to_numpy(item["past_social_risk_features"], dtype=np.float32) for item in batch],
            axis=0,
        )
        fut_traj_original_scale = np.stack(
            [_to_numpy(item["fut_traj_original_scale"], dtype=np.float32) for item in batch],
            axis=0,
        )
        fut_traj_vel = np.stack([_to_numpy(item["fut_traj_vel"], dtype=np.float32) for item in batch], axis=0)
        agent_mask = np.stack([_to_numpy(item["agent_mask"], dtype=np.int64) for item in batch], axis=0)
        num_agents = np.stack([_to_numpy(item["num_agents"], dtype=np.int64) for item in batch], axis=0)
        target_agent_indices = np.stack(
            [_to_numpy(item["target_agent_index"], dtype=np.int64) for item in batch],
            axis=0,
        )
        batch_size = np.asarray(past_traj.shape[0], dtype=np.int64)

    return {
        "indexes": indexes,
        "batch_size": batch_size,
        "past_traj": past_traj,
        "fut_traj": fut_traj,
        "past_traj_original_scale": past_traj_original_scale,
        "past_social_risk_features": past_social_risk_features,
        "fut_traj_original_scale": fut_traj_original_scale,
        "fut_traj_vel": fut_traj_vel,
        "agent_mask": agent_mask,
        "num_agents": num_agents,
        "target_agent_indices": target_agent_indices,
    }


def build_moflow_eth_batch(
    samples: Sequence[Any],
    *,
    data_norm: str = DEFAULT_MOFLOW_DATA_NORM,
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE,
    rotate: bool = False,
    rotate_time_frame: int = 0,
    fixed_num_agents: Optional[int] = None,
    normalization_stats: Optional[Mapping[str, float]] = None,
    as_torch: bool = True,
) -> Dict[str, Any]:
    """直接把标准样本列表转换成 MoFlow ETH batch dict。"""

    if not samples:
        raise ValueError("build_moflow_eth_batch received an empty sample list")

    _validate_sample_mode(sample_mode)

    target_agents = fixed_num_agents or infer_moflow_eth_num_agents(samples, sample_mode=sample_mode)
    adapted: List[Dict[str, Any]] = []
    running_index = 0
    for sample in samples:
        mapping = _sample_to_mapping(sample)
        if sample_mode == "per_agent":
            for agent_index in _resolve_active_agent_indices(mapping):
                adapted.append(
                    build_moflow_eth_sample(
                        sample,
                        index=running_index,
                        data_norm=data_norm,
                        sample_mode=sample_mode,
                        agent_index=agent_index,
                        rotate=rotate,
                        rotate_time_frame=rotate_time_frame,
                        fixed_num_agents=target_agents,
                        normalization_stats=normalization_stats,
                        as_torch=as_torch,
                    )
                )
                running_index += 1
        else:
            adapted.append(
                build_moflow_eth_sample(
                    sample,
                    index=running_index,
                    data_norm=data_norm,
                    sample_mode=sample_mode,
                    rotate=rotate,
                    rotate_time_frame=rotate_time_frame,
                    fixed_num_agents=target_agents,
                    normalization_stats=normalization_stats,
                    as_torch=as_torch,
                )
            )
            running_index += 1

    return seq_collate_moflow_eth(adapted)


class MoFlowETHDataset(Dataset):
    """从 TrustMoE 主缓存/标准样本直接读取的 MoFlow ETH 兼容数据集。"""

    def __init__(
        self,
        base_dataset: Optional[ETHTrajectoryDataset] = None,
        *,
        eth_config: Optional[ETHAdapterConfig] = None,
        transform_config: Optional[MoFlowETHTransformConfig] = None,
        data_norm: str = DEFAULT_MOFLOW_DATA_NORM,
        sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE,
        rotate: bool = False,
        rotate_time_frame: int = 0,
        fixed_num_agents: Optional[int] = None,
        normalization_stats: Optional[Mapping[str, float]] = None,
        as_torch: bool = True,
    ) -> None:
        if base_dataset is None:
            if eth_config is None:
                eth_config = ETHAdapterConfig(prefer_cache=True)
            base_dataset = ETHTrajectoryDataset(eth_config)

        if transform_config is not None:
            data_norm = transform_config.data_norm
            sample_mode = transform_config.sample_mode
            rotate = transform_config.rotate
            rotate_time_frame = transform_config.rotate_time_frame
            fixed_num_agents = transform_config.fixed_num_agents
            as_torch = transform_config.as_torch
            normalization_stats = normalization_stats or transform_config.normalization_stats()

        _validate_data_norm(data_norm)
        _validate_sample_mode(sample_mode)

        self.base_dataset = base_dataset
        self.samples = list(base_dataset.samples)
        self.data_norm = data_norm
        self.sample_mode = sample_mode
        self.rotate = rotate
        self.rotate_time_frame = rotate_time_frame
        self.as_torch = as_torch
        self.fixed_num_agents = fixed_num_agents or infer_moflow_eth_num_agents(self.samples, sample_mode=sample_mode)

        if sample_mode == "per_agent":
            self.index_map = [
                (scene_index, agent_index)
                for scene_index, sample in enumerate(self.samples)
                for agent_index in _resolve_active_agent_indices(_sample_to_mapping(sample))
            ]
        else:
            self.index_map = [(scene_index, None) for scene_index in range(len(self.samples))]

        if data_norm == "min_max":
            computed_stats = normalization_stats or compute_moflow_eth_norm_stats(
                self.samples,
                sample_mode=sample_mode,
                rotate=rotate,
                rotate_time_frame=rotate_time_frame,
                fixed_num_agents=self.fixed_num_agents,
            )
            self.normalization_stats = {
                "past_traj_min": float(computed_stats["past_traj_min"]),
                "past_traj_max": float(computed_stats["past_traj_max"]),
                "fut_traj_min": float(computed_stats["fut_traj_min"]),
                "fut_traj_max": float(computed_stats["fut_traj_max"]),
            }
        else:
            self.normalization_stats = {}

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene_index, agent_index = self.index_map[index]
        return build_moflow_eth_sample(
            self.samples[scene_index],
            index=index,
            data_norm=self.data_norm,
            sample_mode=self.sample_mode,
            agent_index=agent_index,
            rotate=self.rotate,
            rotate_time_frame=self.rotate_time_frame,
            fixed_num_agents=self.fixed_num_agents,
            normalization_stats=self.normalization_stats,
            as_torch=self.as_torch,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "num_source_scenes": len(self.samples),
            "num_moflow_items": len(self.index_map),
            "fixed_num_agents": self.fixed_num_agents,
            "data_norm": self.data_norm,
            "sample_mode": self.sample_mode,
            "rotate": self.rotate,
            "rotate_time_frame": self.rotate_time_frame,
            "loaded_from_cache": getattr(self.base_dataset, "loaded_from_cache", False),
            "normalization_stats": dict(self.normalization_stats),
        }


__all__ = [
    "DEFAULT_MOFLOW_DATA_NORM",
    "DEFAULT_MOFLOW_PAST_DIM",
    "DEFAULT_MOFLOW_SAMPLE_MODE",
    "SUPPORTED_MOFLOW_DATA_NORMS",
    "SUPPORTED_MOFLOW_SAMPLE_MODES",
    "MoFlowETHTransformConfig",
    "PAST_SOCIAL_RISK_FEATURE_NAMES",
    "DEFAULT_PAST_SOCIAL_RISK_DIM",
    "rotate_moflow_eth_trajectories",
    "compute_past_social_risk_features",
    "build_moflow_eth_feature_arrays",
    "infer_moflow_eth_fixed_num_agents",
    "infer_moflow_eth_num_agents",
    "compute_moflow_eth_norm_stats",
    "build_moflow_eth_sample",
    "seq_collate_moflow_eth",
    "build_moflow_eth_batch",
    "MoFlowETHDataset",
]
