"""Sample-level proxy features for routing diagnostics.

The features in this module are intentionally heuristic. They are meant to
answer the first router question: can observable scene / motion / uncertainty
signals predict when the fast branch is likely to fail relative to slow?
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


EPS = 1e-9


def _to_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def _active_agent_indices(sample: Mapping[str, Any]) -> List[int]:
    past = _to_tensor(sample["past_traj"], dtype=torch.float32)
    agent_mask = sample.get("agent_mask")
    if agent_mask is None:
        return list(range(int(past.shape[0])))

    mask = _to_tensor(agent_mask).reshape(-1).bool()
    if int(mask.numel()) != int(past.shape[0]):
        raise ValueError(f"agent_mask length mismatch: {int(mask.numel())} vs {int(past.shape[0])}")

    active = [int(index) for index, flag in enumerate(mask.tolist()) if bool(flag)]
    return active or list(range(int(past.shape[0])))


def _offdiag_distances(points: torch.Tensor) -> Optional[torch.Tensor]:
    if int(points.shape[0]) <= 1:
        return None
    distances = torch.cdist(points, points)
    offdiag = ~torch.eye(int(points.shape[0]), dtype=torch.bool)
    return distances[offdiag]


def _heading_change_stats(velocity: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if int(velocity.shape[0]) <= 1:
        return None, None

    speed = torch.linalg.norm(velocity, dim=-1)
    valid = (speed[:-1] > EPS) & (speed[1:] > EPS)
    if int(valid.sum().item()) <= 0:
        return None, None

    heading = torch.atan2(velocity[:, 1], velocity[:, 0])
    delta = heading[1:] - heading[:-1]
    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
    selected = wrapped[valid]
    return _to_optional_float(selected.mean().item()), _to_optional_float(selected.max().item())


def compute_scene_motion_proxy_features(
    sample: Mapping[str, Any],
    *,
    source_agent_index: int,
) -> Dict[str, Any]:
    """Compute observable scene / history-motion proxy features for one agent.

    Only the observed history is used here, so these fields are safe as router
    inputs at inference time. Prediction-derived proxies are computed separately.
    """

    past = _to_tensor(sample["past_traj"], dtype=torch.float32)
    if past.ndim != 3 or int(past.shape[-1]) != 2:
        raise ValueError(f"past_traj must have shape [A, T, 2], got {tuple(past.shape)}")

    source_agent_index = int(source_agent_index)
    if source_agent_index < 0 or source_agent_index >= int(past.shape[0]):
        raise ValueError(f"source_agent_index out of range: {source_agent_index}")

    active_indices = _active_agent_indices(sample)
    active_past = past[active_indices]
    active_last = active_past[:, -1, :]
    num_agents = int(active_last.shape[0])

    bbox_min = active_last.min(dim=0).values
    bbox_max = active_last.max(dim=0).values
    bbox_size = bbox_max - bbox_min
    bbox_area = _to_optional_float((bbox_size[0] * bbox_size[1]).item())
    bbox_diag = _to_optional_float(torch.linalg.norm(bbox_size).item())
    scene_density = None
    if bbox_area is not None and bbox_area > EPS:
        scene_density = float(num_agents / bbox_area)

    offdiag = _offdiag_distances(active_last)
    scene_neighbor_min = _to_optional_float(offdiag.min().item()) if offdiag is not None else None
    scene_neighbor_mean = _to_optional_float(offdiag.mean().item()) if offdiag is not None else None

    target_last = past[source_agent_index, -1, :]
    neighbor_positions = [
        active_last[position]
        for position, original_index in enumerate(active_indices)
        if int(original_index) != source_agent_index
    ]
    if neighbor_positions:
        neighbor_stack = torch.stack(neighbor_positions, dim=0)
        target_neighbor_distances = torch.linalg.norm(neighbor_stack - target_last[None, :], dim=-1)
        target_neighbor_min = _to_optional_float(target_neighbor_distances.min().item())
        target_neighbor_mean = _to_optional_float(target_neighbor_distances.mean().item())
    else:
        target_neighbor_min = None
        target_neighbor_mean = None

    target_history = past[source_agent_index]
    velocity = target_history[1:] - target_history[:-1]
    speed = torch.linalg.norm(velocity, dim=-1)
    acceleration = velocity[1:] - velocity[:-1] if int(velocity.shape[0]) > 1 else torch.empty((0, 2))
    accel_norm = torch.linalg.norm(acceleration, dim=-1) if int(acceleration.shape[0]) else torch.empty((0,))

    path_length = _to_optional_float(speed.sum().item())
    displacement = _to_optional_float(torch.linalg.norm(target_history[-1] - target_history[0]).item())
    straightness = None
    if path_length is not None and displacement is not None and path_length > EPS:
        straightness = float(displacement / path_length)

    heading_change_mean, heading_change_max = _heading_change_stats(velocity)

    return {
        "proxy_scene_num_agents": num_agents,
        "proxy_scene_bbox_area_last": bbox_area,
        "proxy_scene_bbox_diag_last": bbox_diag,
        "proxy_scene_density_last": scene_density,
        "proxy_scene_neighbor_min_dist_last": scene_neighbor_min,
        "proxy_scene_neighbor_mean_dist_last": scene_neighbor_mean,
        "proxy_target_neighbor_min_dist_last": target_neighbor_min,
        "proxy_target_neighbor_mean_dist_last": target_neighbor_mean,
        "proxy_history_speed_mean": _to_optional_float(speed.mean().item()) if int(speed.numel()) else None,
        "proxy_history_speed_max": _to_optional_float(speed.max().item()) if int(speed.numel()) else None,
        "proxy_history_speed_std": _to_optional_float(speed.std(unbiased=False).item()) if int(speed.numel()) else None,
        "proxy_history_accel_mean": _to_optional_float(accel_norm.mean().item()) if int(accel_norm.numel()) else None,
        "proxy_history_accel_max": _to_optional_float(accel_norm.max().item()) if int(accel_norm.numel()) else None,
        "proxy_history_heading_change_mean": heading_change_mean,
        "proxy_history_heading_change_max": heading_change_max,
        "proxy_history_path_length": path_length,
        "proxy_history_displacement": displacement,
        "proxy_history_straightness": straightness,
    }


def _ensure_prediction_shape(prediction: Any) -> torch.Tensor:
    tensor = _to_tensor(prediction, dtype=torch.float32)
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"Prediction must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(tensor.shape)}"
    )


def _resolve_agent_mask(batch: Mapping[str, Any], *, batch_size: int, num_agents: int) -> torch.Tensor:
    if "agent_mask" not in batch:
        return torch.ones((batch_size, num_agents), dtype=torch.bool)
    mask = _to_tensor(batch["agent_mask"]).bool()
    if tuple(mask.shape) != (batch_size, num_agents):
        raise ValueError(f"agent_mask must have shape [{batch_size}, {num_agents}], got {tuple(mask.shape)}")
    return mask


def _prediction_to_absolute(prediction: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    pred = _ensure_prediction_shape(prediction)
    if "past_traj_original_scale" in batch:
        past = _to_tensor(batch["past_traj_original_scale"], dtype=torch.float32)
    elif "past_traj" in batch:
        past = _to_tensor(batch["past_traj"], dtype=torch.float32)
    else:
        raise KeyError("Batch does not contain `past_traj_original_scale` or `past_traj`")

    if past.ndim != 4 or int(past.shape[-1]) < 2:
        raise ValueError(f"Past trajectory must have shape [B, A, T, >=2], got {tuple(past.shape)}")
    if pred.shape[0] != past.shape[0] or pred.shape[2] != past.shape[1]:
        raise ValueError(f"Prediction / past shape mismatch: prediction={tuple(pred.shape)}, past={tuple(past.shape)}")

    last_observed_abs = past[:, :, -1, :2]
    return pred + last_observed_abs[:, None, :, None, :]


def _endpoint_pairwise_stats(endpoint: torch.Tensor) -> Tuple[float, float]:
    if int(endpoint.shape[0]) <= 1:
        return 0.0, 0.0
    distances = torch.cdist(endpoint, endpoint)
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    selected = distances[mask]
    if int(selected.numel()) <= 0:
        return 0.0, 0.0
    return float(selected.mean().item()), float(selected.max().item())


def _valid_record_keys(
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
    valid_agents: torch.Tensor,
) -> List[Tuple[int, int]]:
    keys: List[Tuple[int, int]] = []
    batch_size, num_agents = valid_agents.shape
    for batch_index in range(int(batch_size)):
        for agent_axis_index in range(int(num_agents)):
            if not bool(valid_agents[batch_index, agent_axis_index].item()):
                continue
            key = (batch_index, agent_axis_index)
            if key in records:
                keys.append(key)
    return keys


def _collision_min_distances(
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
    pred_abs: torch.Tensor,
    valid_agents: torch.Tensor,
) -> Dict[Tuple[int, int], Optional[float]]:
    valid_keys = _valid_record_keys(records, valid_agents)
    groups: Dict[Any, List[Tuple[int, int]]] = {}
    for key in valid_keys:
        record = records[key]
        group_id = record.get("selected_scene_index")
        if group_id is None:
            group_id = ("batch", key[0])
        groups.setdefault(group_id, []).append(key)

    min_distances: Dict[Tuple[int, int], Optional[float]] = {key: None for key in valid_keys}
    for keys in groups.values():
        if len(keys) <= 1:
            continue
        for left_index, left_key in enumerate(keys):
            left_batch, left_agent = left_key
            left_traj = pred_abs[left_batch, :, left_agent, :, :]
            for right_key in keys[left_index + 1 :]:
                right_batch, right_agent = right_key
                right_traj = pred_abs[right_batch, :, right_agent, :, :]
                distances = torch.linalg.norm(left_traj[:, None, :, :] - right_traj[None, :, :, :], dim=-1)
                pair_min = float(distances.min().item())
                current_left = min_distances[left_key]
                current_right = min_distances[right_key]
                min_distances[left_key] = pair_min if current_left is None else min(current_left, pair_min)
                min_distances[right_key] = pair_min if current_right is None else min(current_right, pair_min)
    return min_distances


def update_records_with_prediction_proxy_features(
    records: MutableMapping[Tuple[int, int], MutableMapping[str, Any]],
    *,
    branch_name: str,
    prediction: Any,
    batch: Mapping[str, Any],
    collision_threshold: float = 0.2,
) -> None:
    """Append prediction-derived uncertainty and collision proxies to records."""

    pred_abs = _prediction_to_absolute(prediction, batch)
    batch_size, num_modes, num_agents, _num_frames, _xy = pred_abs.shape
    valid_agents = _resolve_agent_mask(batch, batch_size=batch_size, num_agents=num_agents)

    mode_std = pred_abs.std(dim=1, unbiased=False)
    trajectory_spread = torch.linalg.norm(mode_std, dim=-1)
    endpoint = pred_abs[:, :, :, -1, :]
    endpoint_variance = endpoint.var(dim=1, unbiased=False).sum(dim=-1)
    collision_mins = _collision_min_distances(records, pred_abs, valid_agents)

    for batch_index in range(int(batch_size)):
        for agent_axis_index in range(int(num_agents)):
            if not bool(valid_agents[batch_index, agent_axis_index].item()):
                continue
            key = (batch_index, agent_axis_index)
            if key not in records:
                continue

            record = records[key]
            current_endpoint = endpoint[batch_index, :, agent_axis_index, :]
            endpoint_pairwise_mean, endpoint_pairwise_max = _endpoint_pairwise_stats(current_endpoint)
            collision_min = collision_mins.get(key)
            record.update(
                {
                    f"{branch_name}_num_modes": int(num_modes),
                    f"{branch_name}_trajectory_spread_mean": float(
                        trajectory_spread[batch_index, agent_axis_index].mean().item()
                    ),
                    f"{branch_name}_trajectory_spread_max": float(
                        trajectory_spread[batch_index, agent_axis_index].max().item()
                    ),
                    f"{branch_name}_endpoint_variance": float(
                        endpoint_variance[batch_index, agent_axis_index].item()
                    ),
                    f"{branch_name}_endpoint_pairwise_dist_mean": endpoint_pairwise_mean,
                    f"{branch_name}_endpoint_pairwise_dist_max": endpoint_pairwise_max,
                    f"{branch_name}_collision_min_dist": collision_min,
                    f"{branch_name}_collision_risk": (
                        None if collision_min is None else bool(collision_min < float(collision_threshold))
                    ),
                    f"{branch_name}_collision_threshold": float(collision_threshold),
                }
            )


__all__ = [
    "compute_scene_motion_proxy_features",
    "update_records_with_prediction_proxy_features",
]
