"""Diagnose whether good residual slots have separable correction distributions.

This script studies the full residual-slot space rather than a compressed
front-slot selector.  It labels each nonzero slot by whether its correction is
min-safe relative to slot0 for the same base mode, summarizes the residual
distribution, and trains lightweight probes to test whether good and harmful
slots are separable from observable residual/base features.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    _base_for_flat,
    _flatten_refined,
    _predictor_cfg,
    _set_seed,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import _checkpoint_variant, _local_temporal_energy
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


LOW_FEATURE_NAMES: Sequence[str] = (
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_norm",
    "trajectory_norm_mean",
    "trajectory_norm_max",
    "mean_dx",
    "mean_dy",
    "smoothness",
    "acceleration",
    "forward_endpoint",
    "lateral_endpoint",
    "abs_lateral_endpoint",
    "early_norm",
    "late_norm",
    "late_minus_early_norm",
    "base_endpoint_norm",
    "base_path_norm",
    "base_rank_norm",
)
EPS = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V58 slot residual distribution separability.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--improve-margin", type=float, default=0.0)
    parser.add_argument("--hurt-margin", type=float, default=0.0)
    parser.add_argument("--strong-improve-margin", type=float, default=0.05)
    parser.add_argument("--include-slot0-in-probe", action="store_true")
    parser.add_argument("--max-probe-samples-per-class", type=int, default=120000)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=8192)
    parser.add_argument("--output-json", type=str, required=True)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0.0 else ""
    return f"{prefix}{numeric:.6f}"


def _candidate_ade_fde(prediction: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _base_direction(base: torch.Tensor) -> torch.Tensor:
    direction = base[..., -1, :] - base[..., 0, :]
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)


def _rank_from_fde(fde: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(fde, dim=1)
    rank = torch.empty_like(order)
    values = torch.arange(fde.shape[1], device=fde.device, dtype=torch.long)[None, :, None].expand_as(order)
    rank.scatter_(1, order, values)
    return rank


def _repeat_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    return values[:, None, ...].expand(values.shape[0], int(num_slots), *values.shape[1:]).reshape(
        values.shape[0],
        int(num_slots) * values.shape[1],
        values.shape[2],
    )


def _make_slot_ids(batch_size: int, num_slots: int, num_base_modes: int, num_agents: int, device: str) -> torch.Tensor:
    slot = torch.arange(num_slots, device=device, dtype=torch.long).repeat_interleave(num_base_modes)
    return slot[None, :, None].expand(batch_size, num_slots * num_base_modes, num_agents)


def _low_features(
    residual: torch.Tensor,
    base: torch.Tensor,
    *,
    base_rank: torch.Tensor,
    num_base_modes: int,
) -> torch.Tensor:
    endpoint = residual[..., -1, :]
    residual_norm = torch.linalg.norm(residual, dim=-1)
    endpoint_norm = torch.linalg.norm(endpoint, dim=-1)
    trajectory_norm_mean = residual_norm.mean(dim=-1)
    trajectory_norm_max = residual_norm.max(dim=-1).values
    mean_delta = residual.mean(dim=-2)
    diff = residual[..., 1:, :] - residual[..., :-1, :]
    smoothness = torch.linalg.norm(diff, dim=-1).mean(dim=-1)
    if residual.shape[-2] >= 3:
        accel = diff[..., 1:, :] - diff[..., :-1, :]
        acceleration = torch.linalg.norm(accel, dim=-1).mean(dim=-1)
    else:
        acceleration = torch.zeros_like(smoothness)
    direction = _base_direction(base)
    perp = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    forward = (endpoint * direction).sum(dim=-1)
    lateral = (endpoint * perp).sum(dim=-1)
    half = max(int(residual.shape[-2]) // 2, 1)
    early_norm = residual_norm[..., :half].mean(dim=-1)
    late_norm = residual_norm[..., half:].mean(dim=-1)
    base_step = base[..., 1:, :] - base[..., :-1, :]
    base_path_norm = torch.linalg.norm(base_step, dim=-1).sum(dim=-1)
    base_endpoint_norm = torch.linalg.norm(base[..., -1, :] - base[..., 0, :], dim=-1)
    base_rank_norm = base_rank.to(dtype=residual.dtype) / max(float(num_base_modes - 1), 1.0)
    return torch.stack(
        [
            endpoint[..., 0],
            endpoint[..., 1],
            endpoint_norm,
            trajectory_norm_mean,
            trajectory_norm_max,
            mean_delta[..., 0],
            mean_delta[..., 1],
            smoothness,
            acceleration,
            forward,
            lateral,
            lateral.abs(),
            early_norm,
            late_norm,
            late_norm - early_norm,
            base_endpoint_norm,
            base_path_norm,
            base_rank_norm,
        ],
        dim=-1,
    )


class GroupStats:
    def __init__(self, feature_dim: int, residual_dim: int) -> None:
        self.count = 0
        self.sums: Dict[str, float] = {}
        self.low_sum = torch.zeros(feature_dim, dtype=torch.float64)
        self.low_sq_sum = torch.zeros(feature_dim, dtype=torch.float64)
        self.residual_sum = torch.zeros(residual_dim, dtype=torch.float64)
        self.residual_sq_sum = torch.zeros(residual_dim, dtype=torch.float64)

    def add(
        self,
        *,
        low_features: torch.Tensor,
        residual_vectors: torch.Tensor,
        dade: torch.Tensor,
        dfde: torch.Tensor,
        endpoint_norm: torch.Tensor,
        trajectory_norm: torch.Tensor,
        base_rank: torch.Tensor,
    ) -> None:
        if int(low_features.shape[0]) <= 0:
            return
        x = low_features.detach().to(dtype=torch.float64, device="cpu")
        r = residual_vectors.detach().to(dtype=torch.float64, device="cpu")
        self.count += int(x.shape[0])
        self.low_sum += x.sum(dim=0)
        self.low_sq_sum += (x * x).sum(dim=0)
        self.residual_sum += r.sum(dim=0)
        self.residual_sq_sum += (r * r).sum(dim=0)
        values = {
            "dADE_vs_slot0": dade,
            "dFDE_vs_slot0": dfde,
            "endpoint_norm": endpoint_norm,
            "trajectory_norm": trajectory_norm,
            "base_rank": base_rank.to(dtype=torch.float32),
        }
        for key, tensor in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(tensor.detach().sum().cpu())

    def finalize(self) -> Dict[str, Any]:
        count = max(int(self.count), 1)
        low_mean = self.low_sum / count
        low_var = (self.low_sq_sum / count - low_mean * low_mean).clamp_min(0.0)
        residual_mean = self.residual_sum / count
        residual_var = (self.residual_sq_sum / count - residual_mean * residual_mean).clamp_min(0.0)
        result: Dict[str, Any] = {
            "count": int(self.count),
            "low_feature_names": list(LOW_FEATURE_NAMES),
            "low_mean": [float(item) for item in low_mean.tolist()],
            "low_std": [float(item) for item in torch.sqrt(low_var).tolist()],
            "residual_mean_l2": float(torch.linalg.norm(residual_mean).item()),
            "residual_std_mean": float(torch.sqrt(residual_var).mean().item()),
            "residual_endpoint_mean": [float(item) for item in residual_mean.reshape(-1, 2)[-1].tolist()],
        }
        for key, value in self.sums.items():
            result[f"mean_{key}"] = float(value / count)
        return result


class SplitAccumulator:
    def __init__(self, feature_dim: int, residual_dim: int, max_per_class: int, seed: int) -> None:
        self.feature_dim = int(feature_dim)
        self.residual_dim = int(residual_dim)
        self.max_per_class = int(max_per_class)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.total_candidates = 0
        self.total_valid_candidates = 0
        self.groups = {
            "good_min_safe": GroupStats(feature_dim, residual_dim),
            "bad_min_hurt": GroupStats(feature_dim, residual_dim),
            "neutral": GroupStats(feature_dim, residual_dim),
            "strong_good": GroupStats(feature_dim, residual_dim),
            "all_nonzero": GroupStats(feature_dim, residual_dim),
        }
        self.probe_low_pos: List[torch.Tensor] = []
        self.probe_low_neg: List[torch.Tensor] = []
        self.probe_res_pos: List[torch.Tensor] = []
        self.probe_res_neg: List[torch.Tensor] = []

    def _append_probe(self, low: torch.Tensor, residual: torch.Tensor, *, positive: bool) -> None:
        low_store = self.probe_low_pos if positive else self.probe_low_neg
        res_store = self.probe_res_pos if positive else self.probe_res_neg
        current = sum(int(item.shape[0]) for item in low_store)
        remaining = max(int(self.max_per_class) - current, 0)
        if remaining <= 0 or int(low.shape[0]) <= 0:
            return
        take = min(remaining, int(low.shape[0]))
        perm = torch.randperm(int(low.shape[0]), generator=self.generator)[:take]
        low_store.append(low.detach().cpu()[perm])
        res_store.append(residual.detach().cpu()[perm])

    def add(
        self,
        *,
        low_features: torch.Tensor,
        residual_vectors: torch.Tensor,
        dade: torch.Tensor,
        dfde: torch.Tensor,
        good: torch.Tensor,
        bad: torch.Tensor,
        strong_good: torch.Tensor,
        probe_mask: torch.Tensor,
        endpoint_norm: torch.Tensor,
        trajectory_norm: torch.Tensor,
        base_rank: torch.Tensor,
    ) -> None:
        flat_valid = probe_mask.reshape(-1).bool()
        self.total_candidates += int(probe_mask.numel())
        self.total_valid_candidates += int(flat_valid.sum().item())
        low = low_features.reshape(-1, low_features.shape[-1])[flat_valid]
        residual = residual_vectors.reshape(-1, residual_vectors.shape[-1])[flat_valid]
        dade_v = dade.reshape(-1)[flat_valid]
        dfde_v = dfde.reshape(-1)[flat_valid]
        endpoint_v = endpoint_norm.reshape(-1)[flat_valid]
        trajectory_v = trajectory_norm.reshape(-1)[flat_valid]
        rank_v = base_rank.reshape(-1)[flat_valid]
        good_v = good.reshape(-1)[flat_valid]
        bad_v = bad.reshape(-1)[flat_valid]
        strong_v = strong_good.reshape(-1)[flat_valid]
        neutral_v = ~(good_v | bad_v)
        self.groups["all_nonzero"].add(
            low_features=low,
            residual_vectors=residual,
            dade=dade_v,
            dfde=dfde_v,
            endpoint_norm=endpoint_v,
            trajectory_norm=trajectory_v,
            base_rank=rank_v,
        )
        for name, mask in (
            ("good_min_safe", good_v),
            ("bad_min_hurt", bad_v),
            ("neutral", neutral_v),
            ("strong_good", strong_v),
        ):
            self.groups[name].add(
                low_features=low[mask],
                residual_vectors=residual[mask],
                dade=dade_v[mask],
                dfde=dfde_v[mask],
                endpoint_norm=endpoint_v[mask],
                trajectory_norm=trajectory_v[mask],
                base_rank=rank_v[mask],
            )
        self._append_probe(low[good_v], residual[good_v], positive=True)
        self._append_probe(low[bad_v], residual[bad_v], positive=False)

    def probe_tensors(self, kind: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if kind == "low":
            pos, neg = self.probe_low_pos, self.probe_low_neg
        elif kind == "residual":
            pos, neg = self.probe_res_pos, self.probe_res_neg
        else:
            raise ValueError(f"Unknown probe kind: {kind}")
        if not pos or not neg:
            return None, None
        x_pos = torch.cat(pos, dim=0).to(dtype=torch.float32)
        x_neg = torch.cat(neg, dim=0).to(dtype=torch.float32)
        y_pos = torch.ones(int(x_pos.shape[0]), dtype=torch.float32)
        y_neg = torch.zeros(int(x_neg.shape[0]), dtype=torch.float32)
        return torch.cat([x_pos, x_neg], dim=0), torch.cat([y_pos, y_neg], dim=0)

    def finalize(self) -> Dict[str, Any]:
        groups = {name: group.finalize() for name, group in self.groups.items()}
        good = groups["good_min_safe"]
        bad = groups["bad_min_hurt"]
        all_count = max(int(groups["all_nonzero"]["count"]), 1)
        good_mean = torch.tensor(good["low_mean"], dtype=torch.float64)
        bad_mean = torch.tensor(bad["low_mean"], dtype=torch.float64)
        good_std = torch.tensor(good["low_std"], dtype=torch.float64)
        bad_std = torch.tensor(bad["low_std"], dtype=torch.float64)
        pooled_std = torch.sqrt((good_std * good_std + bad_std * bad_std) * 0.5).clamp_min(EPS)
        low_z_gap = (good_mean - bad_mean) / pooled_std
        groups["label_ratios"] = {
            "good_min_safe_ratio": float(int(good["count"]) / all_count),
            "bad_min_hurt_ratio": float(int(bad["count"]) / all_count),
            "neutral_ratio": float(int(groups["neutral"]["count"]) / all_count),
            "strong_good_ratio": float(int(groups["strong_good"]["count"]) / all_count),
        }
        groups["separation"] = {
            "low_centroid_l2": float(torch.linalg.norm(good_mean - bad_mean).item()),
            "low_mean_abs_z_gap": float(low_z_gap.abs().mean().item()),
            "low_max_abs_z_gap": float(low_z_gap.abs().max().item()),
            "low_top_z_features": [
                {
                    "name": LOW_FEATURE_NAMES[int(index)],
                    "z_gap": float(low_z_gap[int(index)].item()),
                    "good_mean": float(good_mean[int(index)].item()),
                    "bad_mean": float(bad_mean[int(index)].item()),
                }
                for index in torch.argsort(low_z_gap.abs(), descending=True)[:8].tolist()
            ],
        }
        return groups


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    labels = labels.to(dtype=torch.float32)
    n_pos = int((labels > 0.5).sum().item())
    n_neg = int((labels <= 0.5).sum().item())
    if n_pos <= 0 or n_neg <= 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, int(scores.numel()) + 1, dtype=torch.float32)
    rank_sum_pos = ranks[labels > 0.5].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc.item())


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    labels = labels.to(dtype=torch.float32)
    n_pos = int((labels > 0.5).sum().item())
    if n_pos <= 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    cumsum = torch.cumsum(sorted_labels, dim=0)
    denom = torch.arange(1, int(sorted_labels.numel()) + 1, dtype=torch.float32)
    precision = cumsum / denom
    ap = (precision * sorted_labels).sum() / max(n_pos, 1)
    return float(ap.item())


def _standardize(train_x: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, mean, std


def _probe_metrics(model: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Dict[str, Any]:
    if x is None or y is None or int(x.shape[0]) <= 0:
        return {"available": False}
    x_std = (x.to(dtype=torch.float32) - mean) / std
    with torch.no_grad():
        logits = model(x_std).squeeze(-1).cpu()
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    labels = y.cpu() > 0.5
    return {
        "available": True,
        "samples": int(labels.numel()),
        "positive_rate": float(labels.to(dtype=torch.float32).mean().item()),
        "auc": _auc(probs, labels.to(dtype=torch.float32)),
        "ap": _average_precision(probs, labels.to(dtype=torch.float32)),
        "accuracy_at_0p5": float((pred == labels).to(dtype=torch.float32).mean().item()),
        "mean_prob_positive": float(probs[labels].mean().item()) if bool(labels.any().item()) else None,
        "mean_prob_negative": float(probs[~labels].mean().item()) if bool((~labels).any().item()) else None,
    }


def _train_probe(
    train_x: Optional[torch.Tensor],
    train_y: Optional[torch.Tensor],
    eval_sets: Mapping[str, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    if train_x is None or train_y is None:
        return {"available": False, "reason": "missing positive or negative train samples"}
    if int(train_x.shape[0]) < 10:
        return {"available": False, "reason": "too few train samples", "samples": int(train_x.shape[0])}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x_train, mean, std = _standardize(train_x.to(dtype=torch.float32), train_x.to(dtype=torch.float32))
    y_train = train_y.to(dtype=torch.float32)
    model = torch.nn.Linear(int(x_train.shape[1]), 1)
    torch.manual_seed(int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos = float((y_train > 0.5).sum().item())
    neg = float((y_train <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    for _epoch in range(int(epochs)):
        order = torch.randperm(int(x_train.shape[0]), generator=generator)
        for start in range(0, int(order.numel()), max(int(batch_size), 1)):
            index = order[start : start + max(int(batch_size), 1)]
            logits = model(x_train[index]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train[index], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    metrics = {"available": True, "train_samples": int(x_train.shape[0]), "feature_dim": int(x_train.shape[1])}
    for split, (x_eval, y_eval) in eval_sets.items():
        if x_eval is None or y_eval is None:
            metrics[split] = {"available": False}
        else:
            metrics[split] = _probe_metrics(model, x_eval, y_eval, mean, std)
    return metrics


def _process_split(
    args: argparse.Namespace,
    *,
    split: str,
    protocol_settings: Any,
    device: str,
    slow_predictor: MoFlowSlowPredictor,
    refiner: Any,
    normalization_stats: Mapping[str, Any],
    data_root: Path,
    seed_offset: int,
) -> tuple[Dict[str, Any], SplitAccumulator]:
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, args.max_scenes)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    accumulator: Optional[SplitAccumulator] = None
    print(
        "[diagnose_v58_slot_residual_distribution] "
        f"split={split} scenes={len(selected_samples)} eval_items={selected_eval_items}"
    )
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        slow_output = slow_predictor.predict(batch, return_all_states=False)
        if args.sample_mode == "per_agent":
            temporal_energy = build_per_agent_scene_temporal_interaction_features(
                chunk,
                slow_output.slow_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=0.5,
                collision_radius=0.2,
                no_neighbor_distance=10.0,
            )
        else:
            temporal_energy = _local_temporal_energy(batch, slow_output.slow_pred)
        with torch.no_grad():
            refiner_outputs = refiner.refine(
                slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                num_samples=int(args.residual_slots),
                z_mode="slots",
            )
        refined = refiner_outputs["refined"]
        flat = _flatten_refined(refined)
        base_flat = _base_for_flat(refined, slow_output.slow_pred)
        residual_flat = flat - base_flat
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        agent_mask = batch["agent_mask"].to(device=device).bool()
        cand_ade, cand_fde = _candidate_ade_fde(flat, ground_truth)
        slot0_ade, slot0_fde = _candidate_ade_fde(refined[:, 0], ground_truth)
        slot0_ade_flat = _repeat_slots(slot0_ade, int(args.residual_slots))
        slot0_fde_flat = _repeat_slots(slot0_fde, int(args.residual_slots))
        dade = cand_ade - slot0_ade_flat
        dfde = cand_fde - slot0_fde_flat
        base_ade, base_fde = _candidate_ade_fde(slow_output.slow_pred, ground_truth)
        base_rank_flat = _repeat_slots(_rank_from_fde(base_fde), int(args.residual_slots))
        bsz, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])
        slot_ids = _make_slot_ids(bsz, int(args.residual_slots), num_base_modes, num_agents, device)
        nonzero = slot_ids != 0
        if bool(args.include_slot0_in_probe):
            nonzero = torch.ones_like(nonzero, dtype=torch.bool)
        valid = agent_mask[:, None, :].expand_as(dade).bool() & nonzero
        safe = (dade <= float(args.hurt_margin) + EPS) & (dfde <= float(args.hurt_margin) + EPS)
        improves = (dade < -float(args.improve_margin) - EPS) | (dfde < -float(args.improve_margin) - EPS)
        strong_improves = (dade < -float(args.strong_improve_margin) - EPS) | (
            dfde < -float(args.strong_improve_margin) - EPS
        )
        good = safe & improves
        strong_good = safe & strong_improves
        bad = (dade > float(args.hurt_margin) + EPS) | (dfde > float(args.hurt_margin) + EPS)
        low = _low_features(
            residual_flat,
            base_flat,
            base_rank=base_rank_flat,
            num_base_modes=num_base_modes,
        )
        residual_vec = residual_flat.reshape(*residual_flat.shape[:3], -1)
        endpoint_norm = torch.linalg.norm(residual_flat[..., -1, :], dim=-1)
        trajectory_norm = torch.linalg.norm(residual_flat, dim=-1).mean(dim=-1)
        if accumulator is None:
            accumulator = SplitAccumulator(
                feature_dim=int(low.shape[-1]),
                residual_dim=int(residual_vec.shape[-1]),
                max_per_class=int(args.max_probe_samples_per_class),
                seed=int(args.seed) + int(seed_offset),
            )
        accumulator.add(
            low_features=low,
            residual_vectors=residual_vec,
            dade=dade,
            dfde=dfde,
            good=good,
            bad=bad,
            strong_good=strong_good,
            probe_mask=valid,
            endpoint_norm=endpoint_norm,
            trajectory_norm=trajectory_norm,
            base_rank=base_rank_flat,
        )
        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0:
            print(
                "[diagnose_v58_slot_residual_distribution] "
                f"split={split} chunks={chunk_index}/{len(chunks)}"
            )
    if accumulator is None:
        accumulator = SplitAccumulator(
            feature_dim=len(LOW_FEATURE_NAMES),
            residual_dim=24 * 2,
            max_per_class=int(args.max_probe_samples_per_class),
            seed=int(args.seed) + int(seed_offset),
        )
    split_payload = {
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "groups": accumulator.finalize(),
    }
    return split_payload, accumulator


def _print_summary(payload: Mapping[str, Any]) -> None:
    print("\n[diagnose_v58_slot_residual_distribution] summary")
    for split, split_payload in payload.get("splits", {}).items():
        groups = split_payload.get("groups", {})
        ratios = groups.get("label_ratios", {})
        sep = groups.get("separation", {})
        good = groups.get("good_min_safe", {})
        bad = groups.get("bad_min_hurt", {})
        print(f"\n-- {split} --")
        print(
            "labels: "
            f"good={_fmt(ratios.get('good_min_safe_ratio'))} "
            f"bad={_fmt(ratios.get('bad_min_hurt_ratio'))} "
            f"neutral={_fmt(ratios.get('neutral_ratio'))} "
            f"strong_good={_fmt(ratios.get('strong_good_ratio'))}"
        )
        print(
            "quality: "
            f"good_dADE={_fmt(good.get('mean_dADE_vs_slot0'), signed=True)} "
            f"good_dFDE={_fmt(good.get('mean_dFDE_vs_slot0'), signed=True)} "
            f"bad_dADE={_fmt(bad.get('mean_dADE_vs_slot0'), signed=True)} "
            f"bad_dFDE={_fmt(bad.get('mean_dFDE_vs_slot0'), signed=True)}"
        )
        print(
            "separation: "
            f"low_l2={_fmt(sep.get('low_centroid_l2'))} "
            f"mean_abs_z={_fmt(sep.get('low_mean_abs_z_gap'))} "
            f"max_abs_z={_fmt(sep.get('low_max_abs_z_gap'))}"
        )
        if isinstance(sep.get("low_top_z_features"), list):
            items = sep["low_top_z_features"][:5]
            print("top_low_features:", [(item["name"], round(float(item["z_gap"]), 3)) for item in items])
    print("\n-- probes --")
    for kind, result in payload.get("probes", {}).items():
        print(f"{kind}:")
        for split in ("train", "val", "test"):
            row = result.get(split, {})
            print(
                f"  {split}: auc={_fmt(row.get('auc'))} ap={_fmt(row.get('ap'))} "
                f"acc={_fmt(row.get('accuracy_at_0p5'))} pos={_fmt(row.get('positive_rate'))}"
            )


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    splits = _split_items(args.splits)
    if "train" not in splits:
        raise SystemExit("--splits must include train so the probe has a training split")
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    _set_seed(args.seed)
    random.seed(int(args.seed))

    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=1 if args.agents is None else int(args.agents),
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    refiner_variant = _checkpoint_variant(args.refiner_checkpoint)
    refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    refiner.eval()
    if not bool(getattr(refiner.config, "use_set_generator", False)):
        raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")
    if int(args.residual_slots) > int(getattr(refiner.config, "max_residual_slots", 1)):
        raise SystemExit("--residual-slots exceeds checkpoint max_residual_slots")

    stats_dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split="train" if "train" in splits else splits[0],
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(stats_dataset, args.max_scenes)
    agents = _infer_agents(selected_samples, args.sample_mode, args.agents)
    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[slow_predictor],
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)

    print(
        "[diagnose_v58_slot_residual_distribution] "
        f"splits={splits} device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v58_slot_residual_distribution] warning: selected_samples normalization is diagnostic only")

    split_payloads: Dict[str, Any] = {}
    accumulators: Dict[str, SplitAccumulator] = {}
    for split_index, split in enumerate(splits):
        split_payload, accumulator = _process_split(
            args,
            split=split,
            protocol_settings=protocol_settings,
            device=device,
            slow_predictor=slow_predictor,
            refiner=refiner,
            normalization_stats=normalization_stats,
            data_root=data_root,
            seed_offset=split_index * 1000,
        )
        split_payloads[split] = split_payload
        accumulators[split] = accumulator

    probes: Dict[str, Any] = {}
    for kind in ("low", "residual"):
        train_x, train_y = accumulators["train"].probe_tensors(kind)
        eval_sets = {split: accumulators[split].probe_tensors(kind) for split in splits}
        probes[kind] = _train_probe(
            train_x,
            train_y,
            eval_sets,
            epochs=int(args.probe_epochs),
            lr=float(args.probe_lr),
            weight_decay=float(args.probe_weight_decay),
            batch_size=int(args.probe_batch_size),
            seed=int(args.seed) + (17 if kind == "low" else 31),
        )

    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v58_slot_residual_distribution",
            "variant": "v58_slot_residual_distribution_probe",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "splits": splits,
            "residual_slots": int(args.residual_slots),
            "improve_margin": float(args.improve_margin),
            "hurt_margin": float(args.hurt_margin),
            "strong_improve_margin": float(args.strong_improve_margin),
            "include_slot0_in_probe": bool(args.include_slot0_in_probe),
        },
        "args": _coerce_jsonable(vars(args)),
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(args.slow_checkpoint).expanduser().resolve().as_posix(),
        "refiner_checkpoint": Path(args.refiner_checkpoint).expanduser().resolve().as_posix(),
        "splits": _coerce_jsonable(split_payloads),
        "probes": _coerce_jsonable(probes),
    }
    _print_summary(payload)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
