"""TrustMoE-Traj 数据层 batch 拼接工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .schema import SceneMeta, TrajectoryBatch

try:  # pragma: no cover - torch 可能尚未安装
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 保持数据层无强制 torch 依赖
    torch = None
    _TORCH_AVAILABLE = False


def _sample_to_mapping(sample: Any) -> MutableMapping[str, Any]:
    """统一样本访问方式，支持 dict / dataclass batch。"""

    if isinstance(sample, Mapping):
        return dict(sample)
    if isinstance(sample, TrajectoryBatch):
        return sample.to_dict()
    raise TypeError(f"Unsupported sample type for collate: {type(sample)!r}")


def _to_numpy(array_like: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """把输入转成 numpy 数组。"""

    if _TORCH_AVAILABLE and torch is not None and isinstance(array_like, torch.Tensor):
        result = array_like.detach().cpu().numpy()
    else:
        result = np.asarray(array_like)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    """统一 scene_meta 表示。"""

    if isinstance(meta, SceneMeta):
        return meta.to_dict()
    if isinstance(meta, Mapping):
        return dict(meta)
    return {"raw_scene_meta": meta}


def collate_trajectory_samples(
    samples: Sequence[Any],
    *,
    as_torch: bool = False,
) -> TrajectoryBatch:
    """把 variable-agent 样本拼成统一 batch。"""

    if not samples:
        raise ValueError("collate_trajectory_samples received an empty sample list")

    normalized = [_sample_to_mapping(sample) for sample in samples]
    past_list: List[np.ndarray] = []
    future_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    scene_meta_list: List[Dict[str, Any]] = []
    num_agents_list: List[int] = []
    agent_ids_list: List[Optional[np.ndarray]] = []
    frame_ids_list: List[Optional[np.ndarray]] = []

    obs_len: Optional[int] = None
    pred_len: Optional[int] = None

    for index, sample in enumerate(normalized):
        past = _to_numpy(sample["past_traj"], dtype=np.float32)
        future = _to_numpy(sample["future_traj"], dtype=np.float32)

        if past.ndim != 3 or past.shape[-1] != 2:
            raise ValueError(
                f"Sample {index} past_traj must have shape [A, T_obs, 2], got {past.shape}"
            )
        if future.ndim != 3 or future.shape[-1] != 2:
            raise ValueError(
                f"Sample {index} future_traj must have shape [A, T_pred, 2], got {future.shape}"
            )
        if past.shape[0] != future.shape[0]:
            raise ValueError(
                f"Sample {index} past/future agent count mismatch: {past.shape[0]} vs {future.shape[0]}"
            )

        if obs_len is None:
            obs_len = int(past.shape[1])
        elif obs_len != int(past.shape[1]):
            raise ValueError(f"Inconsistent obs_len in batch: {obs_len} vs {past.shape[1]}")

        if pred_len is None:
            pred_len = int(future.shape[1])
        elif pred_len != int(future.shape[1]):
            raise ValueError(f"Inconsistent pred_len in batch: {pred_len} vs {future.shape[1]}")

        agent_mask = sample.get("agent_mask")
        if agent_mask is None:
            agent_mask_np = np.ones((past.shape[0],), dtype=np.int64)
        else:
            agent_mask_np = _to_numpy(agent_mask, dtype=np.int64).reshape(-1)
            if agent_mask_np.shape[0] != past.shape[0]:
                raise ValueError(
                    f"Sample {index} agent_mask length mismatch: {agent_mask_np.shape[0]} vs {past.shape[0]}"
                )

        extras = sample.get("extras") or {}
        agent_ids = extras.get("agent_ids")
        frame_ids = extras.get("frame_ids")

        past_list.append(past)
        future_list.append(future)
        mask_list.append(agent_mask_np)
        scene_meta_list.append(_meta_to_dict(sample.get("scene_meta", {})))
        num_agents_list.append(int(past.shape[0]))
        agent_ids_list.append(None if agent_ids is None else _to_numpy(agent_ids))
        frame_ids_list.append(None if frame_ids is None else _to_numpy(frame_ids))

    batch_size = len(normalized)
    max_agents = max(num_agents_list)
    assert obs_len is not None and pred_len is not None

    batch_past = np.zeros((batch_size, max_agents, obs_len, 2), dtype=np.float32)
    batch_future = np.zeros((batch_size, max_agents, pred_len, 2), dtype=np.float32)
    batch_agent_mask = np.zeros((batch_size, max_agents), dtype=np.int64)

    for batch_idx, (past, future, agent_mask_np) in enumerate(zip(past_list, future_list, mask_list)):
        num_agents = past.shape[0]
        batch_past[batch_idx, :num_agents] = past
        batch_future[batch_idx, :num_agents] = future
        batch_agent_mask[batch_idx, :num_agents] = agent_mask_np

    if as_torch and _TORCH_AVAILABLE and torch is not None:
        past_payload = torch.from_numpy(batch_past)
        future_payload = torch.from_numpy(batch_future)
        mask_payload = torch.from_numpy(batch_agent_mask)
        num_agents_payload = torch.from_numpy(np.asarray(num_agents_list, dtype=np.int64))
    else:
        past_payload = batch_past
        future_payload = batch_future
        mask_payload = batch_agent_mask
        num_agents_payload = np.asarray(num_agents_list, dtype=np.int64)

    return TrajectoryBatch(
        past_traj=past_payload,
        future_traj=future_payload,
        agent_mask=mask_payload,
        scene_meta=scene_meta_list,
        extras={
            "num_agents": num_agents_payload,
            "agent_ids": agent_ids_list,
            "frame_ids": frame_ids_list,
        },
    )


def trajectory_collate_fn(samples: Sequence[Any]) -> TrajectoryBatch:
    """PyTorch DataLoader 可直接使用的默认 collate。"""

    return collate_trajectory_samples(samples, as_torch=_TORCH_AVAILABLE)


__all__ = [
    "collate_trajectory_samples",
    "trajectory_collate_fn",
]
