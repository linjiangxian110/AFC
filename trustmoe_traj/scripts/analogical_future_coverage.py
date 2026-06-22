"""Analogical Future Coverage MVP metrics.

This module implements a non-learning retrieval diagnostic. It builds a bank of
training-split futures keyed by observed-past/social-state features, then checks
whether a prediction set covers the retrieved analogical futures.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.data.adapters.sdd import SDDAdapterConfig, SDDTrajectoryDataset
from trustmoe_traj.data.transforms import build_moflow_eth_batch, infer_moflow_eth_num_agents


AFC_FEATURE_VARIANTS: Sequence[str] = (
    "past_shape",
    "past_velocity",
    "past_velocity_accel",
    "past_velocity_social",
    "full_past_social",
)


def split_float_list(raw: str) -> List[float]:
    return [float(item) for item in str(raw).replace(",", " ").split() if item]


def eps_label(eps: float) -> str:
    return f"eps{int(round(float(eps) * 10.0)):02d}"


def _iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]


def _to_cpu_float(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_cpu_bool(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").bool()
    return torch.as_tensor(value).bool()


def _to_cpu_long(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.long)


def _stable_text_id(value: Any) -> int:
    raw = str(value if value is not None else "").encode("utf-8", errors="replace")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def _sample_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if hasattr(sample, "to_dict"):
        out = sample.to_dict()
        if isinstance(out, Mapping):
            return out
    raise TypeError(f"Unsupported sample type for AFC metadata: {type(sample)!r}")


def _scene_meta(sample: Any) -> Dict[str, Any]:
    mapping = _sample_mapping(sample)
    meta = mapping.get("scene_meta", {})
    if isinstance(meta, Mapping):
        return dict(meta)
    if hasattr(meta, "to_dict"):
        return dict(meta.to_dict())
    return {"raw_scene_meta": str(meta)}


def _active_agent_indices(sample: Any) -> List[int]:
    mapping = _sample_mapping(sample)
    past = torch.as_tensor(mapping["past_traj"])
    mask = mapping.get("agent_mask")
    if mask is None:
        return list(range(int(past.shape[0])))
    mask_tensor = torch.as_tensor(mask).reshape(-1).bool()
    if int(mask_tensor.numel()) != int(past.shape[0]):
        raise ValueError(f"agent_mask length mismatch: {int(mask_tensor.numel())} vs {int(past.shape[0])}")
    active = [int(index) for index, flag in enumerate(mask_tensor.tolist()) if bool(flag)]
    return active or list(range(int(past.shape[0])))


def _metadata_values_for_sample(sample: Any, *, source_id_field: str) -> tuple[int, int]:
    meta = _scene_meta(sample)
    source_value = (
        meta.get(str(source_id_field))
        or meta.get("source_file")
        or meta.get("seq_id")
        or meta.get("sample_id")
        or "unknown_source"
    )
    frame_value = meta.get("frame_id")
    if frame_value is None:
        extras = _sample_mapping(sample).get("extras", {})
        frame_ids = extras.get("frame_ids") if isinstance(extras, Mapping) else None
        if frame_ids is not None:
            frame_tensor = torch.as_tensor(frame_ids).reshape(-1)
            if int(frame_tensor.numel()) > 0:
                frame_value = int(frame_tensor[0].item())
    return _stable_text_id(source_value), int(frame_value or 0)


def build_afc_metadata_tensors(
    samples: Sequence[Any],
    *,
    sample_mode: str,
    fixed_num_agents: Optional[int] = None,
    source_id_field: str = "source_file",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-agent source/frame tensors aligned with build_moflow_eth_batch."""

    if sample_mode == "per_agent":
        source_rows: List[List[int]] = []
        frame_rows: List[List[int]] = []
        for sample in samples:
            source_id, frame_id = _metadata_values_for_sample(sample, source_id_field=source_id_field)
            for _agent_index in _active_agent_indices(sample):
                source_rows.append([source_id])
                frame_rows.append([frame_id])
        return torch.as_tensor(source_rows, dtype=torch.long), torch.as_tensor(frame_rows, dtype=torch.long)

    if sample_mode != "per_scene":
        raise ValueError(f"Unsupported sample_mode for AFC metadata: {sample_mode!r}")
    target_agents = int(fixed_num_agents or infer_moflow_eth_num_agents(samples, sample_mode=sample_mode))
    source_rows = []
    frame_rows = []
    for sample in samples:
        source_id, frame_id = _metadata_values_for_sample(sample, source_id_field=source_id_field)
        source_rows.append([source_id] * target_agents)
        frame_rows.append([frame_id] * target_agents)
    return torch.as_tensor(source_rows, dtype=torch.long), torch.as_tensor(frame_rows, dtype=torch.long)


def attach_afc_metadata_to_batch(
    batch: Dict[str, Any],
    *,
    samples: Sequence[Any],
    sample_mode: str,
    source_id_field: str = "source_file",
) -> Dict[str, Any]:
    fixed_num_agents = int(torch.as_tensor(batch["agent_mask"]).shape[1])
    source_ids, frame_ids = build_afc_metadata_tensors(
        samples,
        sample_mode=sample_mode,
        fixed_num_agents=fixed_num_agents,
        source_id_field=source_id_field,
    )
    if tuple(source_ids.shape) != tuple(torch.as_tensor(batch["agent_mask"]).shape):
        raise ValueError(
            f"AFC metadata shape {tuple(source_ids.shape)} does not match agent_mask {tuple(torch.as_tensor(batch['agent_mask']).shape)}"
        )
    batch["afc_source_id"] = source_ids
    batch["afc_frame_id"] = frame_ids
    return batch


def _agent_features_from_batch(batch: Mapping[str, Any], *, feature_variant: str = "full_past_social") -> torch.Tensor:
    feature_variant = str(feature_variant)
    if feature_variant not in AFC_FEATURE_VARIANTS:
        raise ValueError(f"Unsupported AFC feature_variant={feature_variant!r}; expected one of {AFC_FEATURE_VARIANTS}")
    past = _to_cpu_float(batch["past_traj_original_scale"])
    if past.ndim != 4 or int(past.shape[-1]) < 6:
        raise ValueError(f"past_traj_original_scale must have shape [B,A,P,>=6], got {tuple(past.shape)}")

    rel = past[..., 2:4]
    vel = past[..., 4:6]
    batch_size, num_agents, past_len, _dim = rel.shape

    if past_len >= 2:
        motion_vel = vel[..., :-1, :]
        final_vel = vel[..., -2, :]
    else:
        motion_vel = vel
        final_vel = vel[..., -1, :]
    mean_vel = motion_vel.mean(dim=-2)
    speed = torch.linalg.norm(motion_vel, dim=-1)
    speed_mean = speed.mean(dim=-1, keepdim=True)
    speed_max = speed.max(dim=-1, keepdim=True).values
    displacement = rel[..., -1, :] - rel[..., 0, :]
    if past_len >= 3:
        accel = vel[..., -2, :] - vel[..., -3, :]
    else:
        accel = torch.zeros_like(final_vel)

    social = batch.get("past_social_risk_features")
    if social is None:
        social_features = torch.zeros((batch_size, num_agents, 0), dtype=torch.float32)
    else:
        social_features = _to_cpu_float(social)
        if social_features.ndim != 3:
            raise ValueError(f"past_social_risk_features must have shape [B,A,F], got {tuple(social_features.shape)}")

    past_shape = rel.reshape(batch_size, num_agents, -1)
    velocity_parts = [displacement, final_vel, mean_vel, speed_mean, speed_max]
    if feature_variant == "past_shape":
        feature_parts = [past_shape]
    elif feature_variant == "past_velocity":
        feature_parts = [past_shape, *velocity_parts]
    elif feature_variant == "past_velocity_accel":
        feature_parts = [past_shape, *velocity_parts, accel]
    elif feature_variant == "past_velocity_social":
        feature_parts = [past_shape, *velocity_parts, social_features]
    else:
        feature_parts = [past_shape, displacement, final_vel, mean_vel, accel, speed_mean, speed_max, social_features]
    return torch.cat(feature_parts, dim=-1)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(item) for item in values) / len(values))


def _finite_mean(values: torch.Tensor) -> Optional[float]:
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return None
    return float(finite.mean().item())


def _distance_weights(distances: torch.Tensor, *, scale: float) -> torch.Tensor:
    if distances.ndim != 1:
        raise ValueError(f"distances must have shape [N], got {tuple(distances.shape)}")
    count = int(distances.shape[0])
    if count <= 0:
        return torch.empty((0,), dtype=torch.float32)
    finite = torch.isfinite(distances)
    if not bool(finite.any().item()):
        return torch.full((count,), 1.0 / float(count), dtype=torch.float32)
    safe = distances.detach().to(device="cpu", dtype=torch.float32).clone()
    max_finite = safe[finite].max()
    safe[~finite] = max_finite + max(float(scale), 1e-6) * 50.0
    return torch.softmax(-safe / max(float(scale), 1e-6), dim=0)


def _weighted_entropy_effective_count(weights: torch.Tensor) -> float:
    if int(weights.numel()) <= 0:
        return 0.0
    safe = weights.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum()
    return float(torch.exp(entropy).item())


def _cluster_coverage_one(pairwise: torch.Tensor, covered: torch.Tensor, eps: float) -> tuple[float, float]:
    num_items = int(pairwise.shape[0])
    if num_items <= 0:
        return 0.0, 0.0
    adjacency = pairwise <= float(eps)
    seen = [False] * num_items
    cluster_count = 0
    covered_count = 0
    covered_cpu = covered.detach().cpu().bool()
    for start in range(num_items):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: List[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            neighbors = adjacency[current].nonzero(as_tuple=False).reshape(-1).tolist()
            for neighbor in neighbors:
                neighbor_index = int(neighbor)
                if not seen[neighbor_index]:
                    seen[neighbor_index] = True
                    stack.append(neighbor_index)
        cluster_count += 1
        if any(bool(covered_cpu[index].item()) for index in members):
            covered_count += 1
    return float(cluster_count), float(covered_count / max(cluster_count, 1))


def _weighted_modes_one(
    futures: torch.Tensor,
    distances: torch.Tensor,
    pairwise: torch.Tensor,
    eps: float,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster retrieved futures into weighted analogical future modes."""

    centers_tensor, mode_weights_tensor, _intra_distance, _entropy = _weighted_mode_summary_one(
        futures,
        distances,
        pairwise,
        eps,
        scale=scale,
    )
    return centers_tensor, mode_weights_tensor


def _weighted_mode_summary_one(
    futures: torch.Tensor,
    distances: torch.Tensor,
    pairwise: torch.Tensor,
    eps: float,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Cluster retrieved futures and summarize mode compactness/entropy."""

    num_items = int(futures.shape[0])
    if num_items <= 0:
        return (
            torch.empty((0, int(futures.shape[-2]), int(futures.shape[-1])), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            0.0,
            0.0,
        )
    sample_weights = _distance_weights(distances, scale=scale)
    adjacency = pairwise <= float(eps)
    seen = [False] * num_items
    centers: List[torch.Tensor] = []
    mode_weights: List[torch.Tensor] = []
    intra_distances: List[torch.Tensor] = []
    for start in range(num_items):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: List[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            neighbors = adjacency[current].nonzero(as_tuple=False).reshape(-1).tolist()
            for neighbor in neighbors:
                neighbor_index = int(neighbor)
                if not seen[neighbor_index]:
                    seen[neighbor_index] = True
                    stack.append(neighbor_index)
        member_index = torch.as_tensor(members, dtype=torch.long)
        weights = sample_weights[member_index]
        weight_sum = weights.sum().clamp_min(1e-12)
        normalized = weights / weight_sum
        center = (futures[member_index] * normalized[:, None, None]).sum(dim=0)
        member_distance = torch.linalg.norm(futures[member_index] - center[None, :, :], dim=-1).mean(dim=-1)
        intra_distance = (normalized * member_distance).sum()
        centers.append(center)
        mode_weights.append(weight_sum)
        intra_distances.append(intra_distance)
    centers_tensor = torch.stack(centers, dim=0)
    mode_weights_tensor = torch.stack(mode_weights, dim=0)
    mode_weights_tensor = mode_weights_tensor / mode_weights_tensor.sum().clamp_min(1e-12)
    intra_tensor = torch.stack(intra_distances, dim=0)
    weighted_intra = float((mode_weights_tensor * intra_tensor).sum().item())
    safe = mode_weights_tensor.clamp_min(1e-12)
    entropy = float((-(safe * safe.log()).sum()).item())
    return centers_tensor, mode_weights_tensor, weighted_intra, entropy


@dataclass
class AnalogicalFutureBank:
    features: torch.Tensor
    futures: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    top_m: int = 20
    eps_values: Sequence[float] = (0.5, 1.0)
    source_ids: Optional[torch.Tensor] = None
    frame_ids: Optional[torch.Tensor] = None
    filter_same_source: bool = True
    temporal_gap_frames: Optional[int] = None
    feature_variant: str = "full_past_social"
    randomized_bank_seed: Optional[int] = None

    @classmethod
    def from_tensors(
        cls,
        features: torch.Tensor,
        futures: torch.Tensor,
        *,
        top_m: int = 20,
        eps_values: Sequence[float] = (0.5, 1.0),
        source_ids: Optional[torch.Tensor] = None,
        frame_ids: Optional[torch.Tensor] = None,
        filter_same_source: bool = True,
        temporal_gap_frames: Optional[int] = None,
        feature_variant: str = "full_past_social",
        randomize_futures_seed: Optional[int] = None,
    ) -> "AnalogicalFutureBank":
        feature_variant = str(feature_variant)
        if feature_variant not in AFC_FEATURE_VARIANTS:
            raise ValueError(f"Unsupported AFC feature_variant={feature_variant!r}; expected one of {AFC_FEATURE_VARIANTS}")
        if features.ndim != 2:
            raise ValueError(f"features must have shape [N,D], got {tuple(features.shape)}")
        if futures.ndim != 3:
            raise ValueError(f"futures must have shape [N,T,2], got {tuple(futures.shape)}")
        if int(features.shape[0]) != int(futures.shape[0]):
            raise ValueError(f"features/futures count mismatch: {features.shape[0]} vs {futures.shape[0]}")
        if int(features.shape[0]) <= 0:
            raise ValueError("AFC bank cannot be empty")
        features = features.detach().to(device="cpu", dtype=torch.float32)
        futures = futures.detach().to(device="cpu", dtype=torch.float32)
        if randomize_futures_seed is not None and int(futures.shape[0]) > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(randomize_futures_seed))
            futures = futures[torch.randperm(int(futures.shape[0]), generator=generator)]
        source_ids_cpu = None
        if source_ids is not None:
            source_ids_cpu = source_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            if int(source_ids_cpu.shape[0]) != int(features.shape[0]):
                raise ValueError(
                    f"source_ids count mismatch: {int(source_ids_cpu.shape[0])} vs features={int(features.shape[0])}"
                )
        frame_ids_cpu = None
        if frame_ids is not None:
            frame_ids_cpu = frame_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            if int(frame_ids_cpu.shape[0]) != int(features.shape[0]):
                raise ValueError(
                    f"frame_ids count mismatch: {int(frame_ids_cpu.shape[0])} vs features={int(features.shape[0])}"
                )
        feature_mean = features.mean(dim=0, keepdim=True)
        feature_std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(
            features=(features - feature_mean) / feature_std,
            futures=futures,
            feature_mean=feature_mean,
            feature_std=feature_std,
            top_m=int(top_m),
            eps_values=tuple(float(item) for item in eps_values),
            source_ids=source_ids_cpu,
            frame_ids=frame_ids_cpu,
            filter_same_source=bool(filter_same_source),
            temporal_gap_frames=None if temporal_gap_frames is None else int(temporal_gap_frames),
            feature_variant=feature_variant,
            randomized_bank_seed=None if randomize_futures_seed is None else int(randomize_futures_seed),
        )

    @property
    def bank_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def retrieval_scale(self) -> float:
        return max(math.sqrt(float(self.feature_dim)), 1.0)

    def _query_with_distances(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = _agent_features_from_batch(batch, feature_variant=self.feature_variant)
        valid = _to_cpu_bool(batch["agent_mask"])
        if tuple(valid.shape) != tuple(features.shape[:2]):
            raise ValueError(f"agent_mask shape {tuple(valid.shape)} does not match features {tuple(features.shape[:2])}")
        flat_features = features[valid]
        if int(flat_features.shape[0]) <= 0:
            return (
                flat_features,
                valid,
                torch.empty((0, 0), dtype=torch.long),
                torch.empty((0, 0), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        standardized = (flat_features - self.feature_mean) / self.feature_std
        distances = torch.cdist(standardized, self.features, p=2)
        source = batch.get("afc_source_id")
        query_source = None
        if self.source_ids is not None and source is not None:
            query_source = _to_cpu_long(source)[valid].reshape(-1, 1)
            same_source = query_source == self.source_ids.reshape(1, -1)
            if bool(self.filter_same_source):
                distances = distances.masked_fill(same_source, float("inf"))
        if self.frame_ids is not None and batch.get("afc_frame_id") is not None and self.temporal_gap_frames is not None:
            query_frame = _to_cpu_long(batch["afc_frame_id"])[valid].reshape(-1, 1)
            close_in_time = torch.abs(query_frame - self.frame_ids.reshape(1, -1)) <= int(self.temporal_gap_frames)
            if query_source is not None and self.source_ids is not None:
                close_in_time = close_in_time & (query_source == self.source_ids.reshape(1, -1))
            distances = distances.masked_fill(close_in_time, float("inf"))
        finite_counts = torch.isfinite(distances).sum(dim=1).to(dtype=torch.long)
        keep = min(int(self.top_m), int(self.features.shape[0]))
        top_values, top_indices = torch.topk(distances, k=keep, dim=1, largest=False)
        return flat_features, valid, top_indices, top_values, finite_counts

    def _query(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_features, valid, top_indices, _top_distances, _finite_counts = self._query_with_distances(batch)
        return flat_features, valid, top_indices

    def metrics_for_prediction(self, prediction: torch.Tensor, batch: Mapping[str, Any]) -> Dict[str, float]:
        if prediction.ndim != 5:
            raise ValueError(f"prediction must have shape [B,K,A,T,2], got {tuple(prediction.shape)}")
        _features, valid, top_indices, top_distances, finite_counts = self._query_with_distances(batch)
        raw_query_count = int(top_indices.shape[0])
        if raw_query_count <= 0:
            return {}

        pred = prediction.detach().to(device="cpu", dtype=torch.float32).permute(0, 2, 1, 3, 4)
        if tuple(pred.shape[:2]) != tuple(valid.shape):
            raise ValueError(f"prediction batch/agent shape {tuple(pred.shape[:2])} does not match mask {tuple(valid.shape)}")
        pred_valid = pred[valid]
        valid_query = finite_counts > 0
        invalid_query_count = int((~valid_query).sum().item())
        if invalid_query_count > 0:
            pred_valid = pred_valid[valid_query]
            top_indices = top_indices[valid_query]
            top_distances = top_distances[valid_query]
        query_count = int(top_indices.shape[0])
        if query_count <= 0:
            return {
                "afc_bank_size": float(self.bank_size),
                "afc_raw_query_count": float(raw_query_count),
                "afc_valid_query_count": 0.0,
                "afc_retrieval_invalid_query_count": float(invalid_query_count),
                "afc_retrieval_candidate_count": 0.0,
                "afc_retrieval_min_candidate_count": 0.0,
                "afc_retrieval_finite_fraction": 0.0,
                "afc_filter_same_source": 1.0 if bool(self.filter_same_source) else 0.0,
                "afc_temporal_gap_frames": float(self.temporal_gap_frames or 0),
                "afc_randomized_bank": 1.0 if self.randomized_bank_seed is not None else 0.0,
            }
        proxies = self.futures[top_indices]

        if int(pred_valid.shape[-2]) != int(proxies.shape[-2]):
            raise ValueError(f"prediction/proxy horizon mismatch: {pred_valid.shape[-2]} vs {proxies.shape[-2]}")

        gt_valid: Optional[torch.Tensor] = None
        gt_payload = batch.get("fut_traj_original_scale", batch.get("future_traj"))
        if gt_payload is not None:
            gt = _to_cpu_float(gt_payload)
            if tuple(gt.shape[:2]) == tuple(valid.shape) and int(gt.shape[-2]) == int(pred_valid.shape[-2]):
                gt_valid = gt[valid]
                if invalid_query_count > 0:
                    gt_valid = gt_valid[valid_query]

        ade_pairwise = torch.linalg.norm(
            pred_valid[:, :, None, :, :] - proxies[:, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)
        proxy_to_pred = ade_pairwise.min(dim=1).values
        pred_to_proxy = ade_pairwise.min(dim=2).values

        proxy_pairwise = torch.linalg.norm(
            proxies[:, :, None, :, :] - proxies[:, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)

        scale = self.retrieval_scale
        top1_distance_mean = _finite_mean(top_distances[:, 0])
        top_m_distance_mean = _finite_mean(top_distances)
        retrieval_weights = torch.stack(
            [_distance_weights(top_distances[index], scale=scale) for index in range(query_count)],
            dim=0,
        )
        retrieval_confidence = torch.exp(-top_distances[:, 0].clamp_min(0.0) / scale)
        retrieval_confidence = torch.where(torch.isfinite(retrieval_confidence), retrieval_confidence, torch.zeros_like(retrieval_confidence))
        retrieval_effective_m = [_weighted_entropy_effective_count(retrieval_weights[index]) for index in range(query_count)]

        result: Dict[str, float] = {
            "afc_bank_size": float(self.bank_size),
            "afc_raw_query_count": float(raw_query_count),
            "afc_top_m": float(int(proxies.shape[1])),
            "afc_valid_query_count": float(query_count),
            "afc_retrieval_invalid_query_count": float(invalid_query_count),
            "afc_retrieval_candidate_count": float(finite_counts.to(dtype=torch.float32).mean().item()),
            "afc_retrieval_min_candidate_count": float(finite_counts.min().item()),
            "afc_retrieval_finite_fraction": float(torch.isfinite(top_distances).to(dtype=torch.float32).mean().item()),
            "afc_proxy_to_pred_ade": float(proxy_to_pred.mean().item()),
            "afc_pred_to_proxy_ade": float(pred_to_proxy.mean().item()),
            "afc_chamfer": float(0.5 * (proxy_to_pred.mean().item() + pred_to_proxy.mean().item())),
            "afc_retrieval_scale": float(scale),
            "afc_retrieval_confidence": float(retrieval_confidence.mean().item()),
            "afc_retrieval_effective_m": float(_mean(retrieval_effective_m) or 0.0),
            "afc_filter_same_source": 1.0 if bool(self.filter_same_source) else 0.0,
            "afc_temporal_gap_frames": float(self.temporal_gap_frames or 0),
            "afc_randomized_bank": 1.0 if self.randomized_bank_seed is not None else 0.0,
        }
        if top1_distance_mean is not None:
            result["afc_retrieval_top1_distance"] = float(top1_distance_mean)
        if top_m_distance_mean is not None:
            result["afc_retrieval_top_m_distance"] = float(top_m_distance_mean)

        for eps in self.eps_values:
            label = eps_label(float(eps))
            proxy_covered = proxy_to_pred <= float(eps)
            pred_matched = pred_to_proxy <= float(eps)
            result[f"afc_recall_{label}"] = float(proxy_covered.to(dtype=torch.float32).mean().item())
            result[f"afc_precision_{label}"] = float(pred_matched.to(dtype=torch.float32).mean().item())
            counts: List[float] = []
            coverages: List[float] = []
            mode_recalls: List[float] = []
            weighted_mode_recalls: List[float] = []
            mode_precisions: List[float] = []
            mode_chamfers: List[float] = []
            unsupported_ratios: List[float] = []
            intra_distances: List[float] = []
            mode_entropies: List[float] = []
            for query_index in range(query_count):
                count, coverage = _cluster_coverage_one(
                    proxy_pairwise[query_index],
                    proxy_covered[query_index],
                    float(eps),
                )
                counts.append(count)
                coverages.append(coverage)
                centers, mode_weights, intra_distance, mode_entropy = _weighted_mode_summary_one(
                    proxies[query_index],
                    top_distances[query_index],
                    proxy_pairwise[query_index],
                    float(eps),
                    scale=scale,
                )
                if int(centers.shape[0]) <= 0:
                    continue
                intra_distances.append(float(intra_distance))
                mode_entropies.append(float(mode_entropy))
                mode_pairwise = torch.linalg.norm(
                    pred_valid[query_index, :, None, :, :] - centers[None, :, :, :],
                    dim=-1,
                ).mean(dim=-1)
                mode_to_pred = mode_pairwise.min(dim=0).values
                pred_to_mode = mode_pairwise.min(dim=1).values
                mode_covered = mode_to_pred <= float(eps)
                pred_matched_mode = pred_to_mode <= float(eps)
                mode_recalls.append(float(mode_covered.to(dtype=torch.float32).mean().item()))
                weighted_mode_recalls.append(float((mode_weights * mode_covered.to(dtype=torch.float32)).sum().item()))
                mode_precisions.append(float(pred_matched_mode.to(dtype=torch.float32).mean().item()))
                mode_chamfers.append(float(0.5 * ((mode_weights * mode_to_pred).sum().item() + pred_to_mode.mean().item())))
                if gt_valid is not None:
                    gt_ade = torch.linalg.norm(
                        pred_valid[query_index] - gt_valid[query_index][None, :, :],
                        dim=-1,
                    ).mean(dim=-1)
                    unsupported = (~pred_matched_mode) & (gt_ade > float(eps))
                    unsupported_ratios.append(float(unsupported.to(dtype=torch.float32).mean().item()))
            mode_count = _mean(counts)
            mode_coverage = _mean(coverages)
            if mode_count is not None:
                result[f"afc_mode_count_{label}"] = float(mode_count)
            if mode_coverage is not None:
                result[f"afc_mode_coverage_{label}"] = float(mode_coverage)
            mode_recall = _mean(mode_recalls)
            weighted_mode_recall = _mean(weighted_mode_recalls)
            mode_precision = _mean(mode_precisions)
            mode_chamfer = _mean(mode_chamfers)
            unsupported_ratio = _mean(unsupported_ratios)
            mode_intra_distance = _mean(intra_distances)
            mode_entropy = _mean(mode_entropies)
            if mode_recall is not None:
                result[f"afc_mode_recall_{label}"] = float(mode_recall)
            if weighted_mode_recall is not None:
                result[f"afc_weighted_mode_recall_{label}"] = float(weighted_mode_recall)
            if mode_precision is not None:
                result[f"afc_mode_precision_{label}"] = float(mode_precision)
            if mode_chamfer is not None:
                result[f"afc_mode_chamfer_{label}"] = float(mode_chamfer)
            if unsupported_ratio is not None:
                result[f"afc_unsupported_ratio_{label}"] = float(unsupported_ratio)
            if mode_intra_distance is not None:
                result[f"afc_mode_intra_distance_{label}"] = float(mode_intra_distance)
            if mode_entropy is not None:
                result[f"afc_mode_entropy_{label}"] = float(mode_entropy)
        return result

    def support_for_prediction(self, prediction: torch.Tensor, batch: Mapping[str, Any], *, tau: float = 1.0) -> torch.Tensor:
        """Return per-candidate support from retrieved analogical futures.

        The score is ``exp(-min_A ADE(Y, A) / tau)``. Higher means the candidate
        is closer to at least one retrieved plausible future. The output keeps
        the candidate dimensions of ``prediction``:

        * ``[B,K,A,T,2] -> [B,K,A]``
        * ``[B,S,K,A,T,2] -> [B,S,K,A]``
        """
        if prediction.ndim not in {5, 6}:
            raise ValueError(f"prediction must have shape [B,K,A,T,2] or [B,S,K,A,T,2], got {tuple(prediction.shape)}")
        _features, valid, top_indices = self._query(batch)
        tau_value = max(float(tau), 1e-6)
        if prediction.ndim == 5:
            batch_size, num_modes, num_agents = [int(item) for item in prediction.shape[:3]]
            support = torch.zeros((batch_size, num_modes, num_agents), dtype=torch.float32)
        else:
            batch_size, num_slots, num_modes, num_agents = [int(item) for item in prediction.shape[:4]]
            support = torch.zeros((batch_size, num_slots, num_modes, num_agents), dtype=torch.float32)
        query_count = int(top_indices.shape[0])
        if query_count <= 0:
            return support

        proxies = self.futures[top_indices]
        pred = prediction.detach().to(device="cpu", dtype=torch.float32)
        if prediction.ndim == 5:
            pred_by_agent = pred.permute(0, 2, 1, 3, 4)
            if tuple(pred_by_agent.shape[:2]) != tuple(valid.shape):
                raise ValueError(f"prediction batch/agent shape {tuple(pred_by_agent.shape[:2])} does not match mask {tuple(valid.shape)}")
            pred_valid = pred_by_agent[valid]
            if int(pred_valid.shape[-2]) != int(proxies.shape[-2]):
                raise ValueError(f"prediction/proxy horizon mismatch: {pred_valid.shape[-2]} vs {proxies.shape[-2]}")
            ade_pairwise = torch.linalg.norm(
                pred_valid[:, :, None, :, :] - proxies[:, None, :, :, :],
                dim=-1,
            ).mean(dim=-1)
            min_ade = ade_pairwise.min(dim=-1).values
            valid_support = torch.exp(-min_ade / tau_value)
            support_by_agent = torch.zeros((batch_size, num_agents, num_modes), dtype=torch.float32)
            support_by_agent[valid] = valid_support
            return support_by_agent.permute(0, 2, 1)

        pred_by_agent = pred.permute(0, 3, 1, 2, 4, 5)
        if tuple(pred_by_agent.shape[:2]) != tuple(valid.shape):
            raise ValueError(f"prediction batch/agent shape {tuple(pred_by_agent.shape[:2])} does not match mask {tuple(valid.shape)}")
        pred_valid = pred_by_agent[valid]
        if int(pred_valid.shape[-2]) != int(proxies.shape[-2]):
            raise ValueError(f"prediction/proxy horizon mismatch: {pred_valid.shape[-2]} vs {proxies.shape[-2]}")
        ade_pairwise = torch.linalg.norm(
            pred_valid[:, :, :, None, :, :] - proxies[:, None, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)
        min_ade = ade_pairwise.min(dim=-1).values
        valid_support = torch.exp(-min_ade / tau_value)
        support_by_agent = torch.zeros((batch_size, num_agents, num_slots, num_modes), dtype=torch.float32)
        support_by_agent[valid] = valid_support
        return support_by_agent.permute(0, 2, 3, 1)


def build_eth_analogical_future_bank(
    *,
    data_root: str | Path,
    subset: str,
    train_split: str = "train",
    sample_mode: str = "per_agent",
    data_norm: str = "min_max",
    rotate: bool = True,
    rotate_time_frame: int = 6,
    normalization_stats: Optional[Mapping[str, float]] = None,
    min_agents: int = 1,
    prefer_cache: bool = False,
    max_train_scenes: Optional[int] = None,
    batch_scenes: int = 64,
    top_m: int = 20,
    eps_values: Sequence[float] = (0.5, 1.0),
    feature_variant: str = "full_past_social",
    include_source_metadata: bool = False,
    source_id_field: str = "source_file",
    filter_same_source: bool = False,
    temporal_gap_frames: Optional[int] = None,
    randomize_futures_seed: Optional[int] = None,
) -> AnalogicalFutureBank:
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=subset,
            split=train_split,
            min_agents=int(min_agents),
            prefer_cache=bool(prefer_cache),
        )
    )
    limit = len(dataset) if max_train_scenes is None else min(int(max_train_scenes), len(dataset))
    if limit <= 0:
        raise ValueError(f"AFC train split is empty for subset={subset!r}")
    samples = [dataset[index] for index in range(limit)]
    fixed_num_agents = 1 if sample_mode == "per_agent" else infer_moflow_eth_num_agents(samples, sample_mode=sample_mode)

    feature_chunks: List[torch.Tensor] = []
    future_chunks: List[torch.Tensor] = []
    source_chunks: List[torch.Tensor] = []
    frame_chunks: List[torch.Tensor] = []
    needs_metadata = bool(include_source_metadata or filter_same_source or temporal_gap_frames is not None)
    for chunk in _iter_chunks(samples, int(batch_scenes)):
        batch = build_moflow_eth_batch(
            chunk,
            data_norm=data_norm,
            sample_mode=sample_mode,
            rotate=bool(rotate),
            rotate_time_frame=int(rotate_time_frame),
            fixed_num_agents=fixed_num_agents,
            normalization_stats=normalization_stats,
            as_torch=True,
        )
        if needs_metadata:
            attach_afc_metadata_to_batch(
                batch,
                samples=chunk,
                sample_mode=sample_mode,
                source_id_field=source_id_field,
            )
        features = _agent_features_from_batch(batch, feature_variant=str(feature_variant))
        futures = _to_cpu_float(batch["fut_traj_original_scale"])
        valid = _to_cpu_bool(batch["agent_mask"])
        feature_chunks.append(features[valid])
        future_chunks.append(futures[valid])
        if needs_metadata:
            source_chunks.append(_to_cpu_long(batch["afc_source_id"])[valid])
            frame_chunks.append(_to_cpu_long(batch["afc_frame_id"])[valid])

    features_all = torch.cat(feature_chunks, dim=0)
    futures_all = torch.cat(future_chunks, dim=0)
    return AnalogicalFutureBank.from_tensors(
        features_all,
        futures_all,
        top_m=int(top_m),
        eps_values=eps_values,
        source_ids=torch.cat(source_chunks, dim=0) if source_chunks else None,
        frame_ids=torch.cat(frame_chunks, dim=0) if frame_chunks else None,
        filter_same_source=bool(filter_same_source),
        temporal_gap_frames=temporal_gap_frames,
        feature_variant=str(feature_variant),
        randomize_futures_seed=randomize_futures_seed,
    )


def build_sdd_analogical_future_bank(
    *,
    data_root: str | Path,
    train_split: str = "train",
    sample_mode: str = "per_scene",
    data_norm: str = "original",
    rotate: bool = False,
    rotate_time_frame: int = 0,
    normalization_stats: Optional[Mapping[str, float]] = None,
    max_train_scenes: Optional[int] = None,
    batch_scenes: int = 256,
    top_m: int = 20,
    eps_values: Sequence[float] = (0.5, 1.0),
    feature_variant: str = "full_past_social",
    include_source_metadata: bool = False,
    source_id_field: str = "source_file",
    filter_same_source: bool = False,
    temporal_gap_frames: Optional[int] = None,
    randomize_futures_seed: Optional[int] = None,
) -> AnalogicalFutureBank:
    """Build an AFC retrieval bank from MoFlow-style SDD pickle data.

    SDD samples are single-agent scenes in the current MoFlow data format.  The
    returned bank uses the same feature extractor and metric implementation as
    ETH-UCY AFC, so SDD smoke tests validate protocol portability rather than a
    separate metric definition.
    """

    dataset = SDDTrajectoryDataset(
        SDDAdapterConfig(
            data_root=data_root,
            split=train_split,
            max_samples=max_train_scenes,
        )
    )
    limit = len(dataset)
    if limit <= 0:
        raise ValueError(f"AFC SDD train split is empty for split={train_split!r}")
    samples = [dataset[index] for index in range(limit)]
    fixed_num_agents = 1 if sample_mode == "per_agent" else infer_moflow_eth_num_agents(samples, sample_mode=sample_mode)

    feature_chunks: List[torch.Tensor] = []
    future_chunks: List[torch.Tensor] = []
    source_chunks: List[torch.Tensor] = []
    frame_chunks: List[torch.Tensor] = []
    needs_metadata = bool(include_source_metadata or filter_same_source or temporal_gap_frames is not None)
    for chunk in _iter_chunks(samples, int(batch_scenes)):
        batch = build_moflow_eth_batch(
            chunk,
            data_norm=data_norm,
            sample_mode=sample_mode,
            rotate=bool(rotate),
            rotate_time_frame=int(rotate_time_frame),
            fixed_num_agents=fixed_num_agents,
            normalization_stats=normalization_stats,
            as_torch=True,
        )
        if needs_metadata:
            attach_afc_metadata_to_batch(
                batch,
                samples=chunk,
                sample_mode=sample_mode,
                source_id_field=source_id_field,
            )
        features = _agent_features_from_batch(batch, feature_variant=str(feature_variant))
        futures = _to_cpu_float(batch["fut_traj_original_scale"])
        valid = _to_cpu_bool(batch["agent_mask"])
        feature_chunks.append(features[valid])
        future_chunks.append(futures[valid])
        if needs_metadata:
            source_chunks.append(_to_cpu_long(batch["afc_source_id"])[valid])
            frame_chunks.append(_to_cpu_long(batch["afc_frame_id"])[valid])

    features_all = torch.cat(feature_chunks, dim=0)
    futures_all = torch.cat(future_chunks, dim=0)
    return AnalogicalFutureBank.from_tensors(
        features_all,
        futures_all,
        top_m=int(top_m),
        eps_values=eps_values,
        source_ids=torch.cat(source_chunks, dim=0) if source_chunks else None,
        frame_ids=torch.cat(frame_chunks, dim=0) if frame_chunks else None,
        filter_same_source=bool(filter_same_source),
        temporal_gap_frames=temporal_gap_frames,
        feature_variant=str(feature_variant),
        randomize_futures_seed=randomize_futures_seed,
    )
