"""Diagnose elite residual structure in a V57/V58 residual candidate pool.

This is an offline diagnosis script.  It does not train a selector.  It runs a
set-generator refiner, scores the full residual pool with GT, and summarizes
where useful residuals live: slot id, base rank, gain, residual norm, local
forward/lateral direction, and optional residual-shape clusters.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

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
    _is_benchmark_comparable_run,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


GAIN_BINS: Sequence[float] = (-1.0, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0)
NORM_BINS: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V58-B0 elite residual space structure.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
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
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--global-elite-ks", type=str, default="1,5,20")
    parser.add_argument("--per-base-topks", type=str, default="1,2,3")
    parser.add_argument("--gain-thresholds", type=str, default="0.0,0.02,0.05,0.10,0.15")
    parser.add_argument("--cluster-gain-threshold", type=float, default=0.05)
    parser.add_argument("--kmeans-clusters", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--max-cluster-residuals", type=int, default=50000)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in _split_items(raw)]


def _split_floats(raw: str) -> List[float]:
    return [float(item) for item in _split_items(raw)]


def _tag_float(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p").rstrip("0").rstrip("p")


def _candidate_score(prediction: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _base_score(base: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _rank_from_score(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score, dim=1)
    rank = torch.empty_like(order)
    values = torch.arange(score.shape[1], device=score.device, dtype=torch.long)[None, :, None].expand_as(order)
    rank.scatter_(1, order, values)
    return rank


def _repeat_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    return values[:, None, ...].expand(values.shape[0], int(num_slots), *values.shape[1:]).reshape(
        values.shape[0],
        int(num_slots) * values.shape[1],
        values.shape[2],
    )


def _base_direction(base: torch.Tensor) -> torch.Tensor:
    direction = base[..., -1, :] - base[..., 0, :]
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)


def _gather_flat(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        return torch.gather(values, dim=1, index=indices.to(device=values.device, dtype=torch.long))
    if values.ndim == 5:
        gather_index = indices.to(device=values.device, dtype=torch.long)[:, :, :, None, None].expand(
            indices.shape[0],
            indices.shape[1],
            indices.shape[2],
            values.shape[-2],
            values.shape[-1],
        )
        return torch.gather(values, dim=1, index=gather_index)
    if values.ndim == 4:
        gather_index = indices.to(device=values.device, dtype=torch.long)[:, :, :, None].expand(
            indices.shape[0],
            indices.shape[1],
            indices.shape[2],
            values.shape[-1],
        )
        return torch.gather(values, dim=1, index=gather_index)
    raise ValueError(f"Unsupported gather tensor shape: {tuple(values.shape)}")


def _hist(values: torch.Tensor, bins: Sequence[float]) -> List[int]:
    if int(values.numel()) <= 0:
        return [0 for _ in range(len(bins) + 1)]
    edges = torch.tensor(list(bins), device=values.device, dtype=values.dtype)
    bucket = torch.bucketize(values, edges)
    return [int(item) for item in torch.bincount(bucket, minlength=len(bins) + 1).detach().cpu().tolist()]


class ResidualGroupStats:
    def __init__(self, *, num_slots: int, num_base_modes: int) -> None:
        self.num_slots = int(num_slots)
        self.num_base_modes = int(num_base_modes)
        self.count = 0
        self.valid_agent_count = 0
        self.sums: Dict[str, float] = {}
        self.gain_hist = [0 for _ in range(len(GAIN_BINS) + 1)]
        self.endpoint_norm_hist = [0 for _ in range(len(NORM_BINS) + 1)]
        self.slot_counts = [0 for _ in range(self.num_slots)]
        self.base_rank_counts = [0 for _ in range(self.num_base_modes)]

    def _add_sum(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def add(
        self,
        *,
        gain: torch.Tensor,
        candidate_score: torch.Tensor,
        base_score: torch.Tensor,
        slot_id: torch.Tensor,
        base_rank: torch.Tensor,
        residual: torch.Tensor,
        base_direction: torch.Tensor,
        valid: torch.Tensor,
        valid_agent_count: int,
    ) -> None:
        flat_valid = valid.reshape(-1).bool()
        self.valid_agent_count += int(valid_agent_count)
        if int(flat_valid.sum().item()) <= 0:
            return
        gain_v = gain.reshape(-1)[flat_valid].detach()
        cand_v = candidate_score.reshape(-1)[flat_valid].detach()
        base_v = base_score.reshape(-1)[flat_valid].detach()
        slot_v = slot_id.reshape(-1)[flat_valid].to(dtype=torch.long).detach()
        rank_v = base_rank.reshape(-1)[flat_valid].to(dtype=torch.long).detach()
        residual_v = residual.reshape(-1, residual.shape[-2], residual.shape[-1])[flat_valid].detach()
        direction_v = base_direction.reshape(-1, base_direction.shape[-1])[flat_valid].detach()
        endpoint = residual_v[:, -1, :]
        endpoint_norm = torch.linalg.norm(endpoint, dim=-1)
        traj_norm = torch.linalg.norm(residual_v, dim=-1).mean(dim=-1)
        perp = torch.stack([-direction_v[:, 1], direction_v[:, 0]], dim=-1)
        forward = (endpoint * direction_v).sum(dim=-1)
        lateral = (endpoint * perp).sum(dim=-1)

        count = int(gain_v.numel())
        self.count += count
        self._add_sum("gain", float(gain_v.sum().cpu()))
        self._add_sum("positive_gain", float((gain_v > 0.0).to(dtype=torch.float32).sum().cpu()))
        self._add_sum("candidate_score", float(cand_v.sum().cpu()))
        self._add_sum("base_score", float(base_v.sum().cpu()))
        self._add_sum("endpoint_norm", float(endpoint_norm.sum().cpu()))
        self._add_sum("trajectory_norm", float(traj_norm.sum().cpu()))
        self._add_sum("endpoint_x", float(endpoint[:, 0].sum().cpu()))
        self._add_sum("endpoint_y", float(endpoint[:, 1].sum().cpu()))
        self._add_sum("forward", float(forward.sum().cpu()))
        self._add_sum("lateral", float(lateral.sum().cpu()))
        self._add_sum("abs_lateral", float(lateral.abs().sum().cpu()))
        self._add_sum("base_rank", float(rank_v.to(dtype=torch.float32).sum().cpu()))
        self._add_sum("slot_id", float(slot_v.to(dtype=torch.float32).sum().cpu()))
        gain_hist = _hist(gain_v, GAIN_BINS)
        norm_hist = _hist(endpoint_norm, NORM_BINS)
        self.gain_hist = [a + b for a, b in zip(self.gain_hist, gain_hist)]
        self.endpoint_norm_hist = [a + b for a, b in zip(self.endpoint_norm_hist, norm_hist)]
        slot_counts = torch.bincount(slot_v.clamp(0, self.num_slots - 1), minlength=self.num_slots)
        rank_counts = torch.bincount(rank_v.clamp(0, self.num_base_modes - 1), minlength=self.num_base_modes)
        self.slot_counts = [a + int(b) for a, b in zip(self.slot_counts, slot_counts.cpu().tolist())]
        self.base_rank_counts = [a + int(b) for a, b in zip(self.base_rank_counts, rank_counts.cpu().tolist())]

    def finalize(self) -> Dict[str, Any]:
        count = max(int(self.count), 1)
        slot_total = max(sum(self.slot_counts), 1)
        rank_total = max(sum(self.base_rank_counts), 1)
        result: Dict[str, Any] = {
            "count": int(self.count),
            "valid_agent_count": int(self.valid_agent_count),
            "selected_per_valid_agent": float(self.count / max(self.valid_agent_count, 1)),
            "gain_bins": list(GAIN_BINS),
            "gain_hist": list(self.gain_hist),
            "endpoint_norm_bins": list(NORM_BINS),
            "endpoint_norm_hist": list(self.endpoint_norm_hist),
            "slot_counts": list(self.slot_counts),
            "slot_ratios": [float(item / slot_total) for item in self.slot_counts],
            "base_rank_counts": list(self.base_rank_counts),
            "base_rank_ratios": [float(item / rank_total) for item in self.base_rank_counts],
        }
        for key, value in self.sums.items():
            result[f"mean_{key}"] = float(value / count)
        return result


def _add_group_from_indices(
    group: ResidualGroupStats,
    *,
    indices: torch.Tensor,
    gain_flat: torch.Tensor,
    score_flat: torch.Tensor,
    base_score_flat: torch.Tensor,
    slot_flat: torch.Tensor,
    base_rank_flat: torch.Tensor,
    residual_flat: torch.Tensor,
    base_direction_flat: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    valid = agent_mask[:, None, :].expand(indices.shape[0], indices.shape[1], indices.shape[2]).bool()
    group.add(
        gain=_gather_flat(gain_flat, indices),
        candidate_score=_gather_flat(score_flat, indices),
        base_score=_gather_flat(base_score_flat, indices),
        slot_id=_gather_flat(slot_flat, indices),
        base_rank=_gather_flat(base_rank_flat, indices),
        residual=_gather_flat(residual_flat, indices),
        base_direction=_gather_flat(base_direction_flat, indices),
        valid=valid,
        valid_agent_count=int(agent_mask.bool().sum().item()),
    )


def _add_group_from_mask(
    group: ResidualGroupStats,
    *,
    mask: torch.Tensor,
    gain_flat: torch.Tensor,
    score_flat: torch.Tensor,
    base_score_flat: torch.Tensor,
    slot_flat: torch.Tensor,
    base_rank_flat: torch.Tensor,
    residual_flat: torch.Tensor,
    base_direction_flat: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    valid = mask & agent_mask[:, None, :].expand_as(mask).bool()
    group.add(
        gain=gain_flat,
        candidate_score=score_flat,
        base_score=base_score_flat,
        slot_id=slot_flat,
        base_rank=base_rank_flat,
        residual=residual_flat,
        base_direction=base_direction_flat,
        valid=valid,
        valid_agent_count=int(agent_mask.bool().sum().item()),
    )


def _append_cluster_residuals(
    chunks: List[torch.Tensor],
    *,
    residual_flat: torch.Tensor,
    gain_flat: torch.Tensor,
    agent_mask: torch.Tensor,
    threshold: float,
    limit: int,
) -> None:
    if int(limit) <= 0:
        return
    current = sum(int(chunk.shape[0]) for chunk in chunks)
    if current >= int(limit):
        return
    valid = (gain_flat > float(threshold)) & agent_mask[:, None, :].expand_as(gain_flat).bool()
    if int(valid.sum().item()) <= 0:
        return
    selected = residual_flat.reshape(-1, residual_flat.shape[-2], residual_flat.shape[-1])[valid.reshape(-1)]
    remaining = int(limit) - current
    chunks.append(selected[:remaining].detach().cpu())


def _kmeans(data: torch.Tensor, *, num_clusters: int, iters: int, seed: int, chunk_size: int = 32768) -> torch.Tensor:
    if int(data.shape[0]) < int(num_clusters):
        raise ValueError(f"Need at least {num_clusters} residuals, got {int(data.shape[0])}")
    generator = torch.Generator(device=data.device)
    generator.manual_seed(int(seed))
    centers = data[torch.randperm(int(data.shape[0]), generator=generator, device=data.device)[: int(num_clusters)]].clone()
    for _ in range(int(iters)):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(int(num_clusters), device=data.device, dtype=data.dtype)
        for chunk in data.split(max(int(chunk_size), 1), dim=0):
            labels = torch.cdist(chunk, centers, p=2).argmin(dim=1)
            sums.index_add_(0, labels, chunk)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=data.dtype))
        empty = counts <= 0
        if bool(empty.any().item()):
            replacement = torch.randperm(int(data.shape[0]), generator=generator, device=data.device)[: int(empty.sum().item())]
            sums[empty] = data[replacement]
            counts[empty] = 1.0
        centers = sums / counts[:, None].clamp_min(1.0)
    return centers


def _cluster_summary(chunks: Sequence[torch.Tensor], *, clusters: int, iters: int, seed: int) -> Dict[str, Any]:
    if not chunks:
        return {"available": False, "reason": "no residuals above cluster threshold"}
    residuals = torch.cat(list(chunks), dim=0).to(dtype=torch.float32)
    if int(residuals.shape[0]) < int(clusters):
        return {"available": False, "reason": "too few residuals", "num_residuals": int(residuals.shape[0])}
    data = residuals.reshape(residuals.shape[0], -1)
    centers = _kmeans(data, num_clusters=int(clusters), iters=int(iters), seed=int(seed))
    labels = torch.cdist(data, centers, p=2).argmin(dim=1)
    counts = torch.bincount(labels, minlength=int(clusters)).cpu().tolist()
    prototypes = centers.reshape(int(clusters), residuals.shape[1], residuals.shape[2])
    endpoint_norm = torch.linalg.norm(prototypes[:, -1, :], dim=-1)
    trajectory_norm = torch.linalg.norm(prototypes, dim=-1).mean(dim=-1)
    return {
        "available": True,
        "num_residuals": int(residuals.shape[0]),
        "cluster_counts": [int(item) for item in counts],
        "cluster_ratios": [float(item / max(sum(counts), 1)) for item in counts],
        "prototype_endpoint": prototypes[:, -1, :].cpu().tolist(),
        "prototype_endpoint_norm": [float(item) for item in endpoint_norm.cpu().tolist()],
        "prototype_trajectory_norm": [float(item) for item in trajectory_norm.cpu().tolist()],
    }


def _print_summary(group_stats: Mapping[str, Mapping[str, Any]], cluster_stats: Mapping[str, Any]) -> None:
    print("\n[diagnose_v58_elite_residual_space] key groups")
    for name, stats in group_stats.items():
        print(f"\n-- {name} --")
        for key in (
            "count",
            "selected_per_valid_agent",
            "mean_gain",
            "mean_positive_gain",
            "mean_endpoint_norm",
            "mean_trajectory_norm",
            "mean_forward",
            "mean_lateral",
            "mean_abs_lateral",
            "mean_base_rank",
            "mean_slot_id",
        ):
            if key in stats:
                value = stats[key]
                print(f"{key}: {value if isinstance(value, int) else float(value):.6f}" if not isinstance(value, int) else f"{key}: {value}")
        if "slot_ratios" in stats:
            print("slot_ratios:", [round(float(item), 4) for item in stats["slot_ratios"]])
        if "base_rank_ratios" in stats:
            print("base_rank_ratios_top8:", [round(float(item), 4) for item in stats["base_rank_ratios"][:8]])
    print("\n-- residual clusters --")
    print(json.dumps(cluster_stats, ensure_ascii=False, indent=2)[:4000])


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    global_ks = _split_ints(args.global_elite_ks)
    per_base_topks = _split_ints(args.per_base_topks)
    gain_thresholds = _split_floats(args.gain_thresholds)
    if not global_ks:
        raise SystemExit("--global-elite-ks must not be empty")
    if not per_base_topks:
        raise SystemExit("--per-base-topks must not be empty")
    if int(args.kmeans_clusters) <= 0:
        raise SystemExit("--kmeans-clusters must be positive")
    if int(args.kmeans_iters) <= 0:
        raise SystemExit("--kmeans-iters must be positive")

    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=args.split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, args.max_scenes)
    agents = _infer_agents(selected_samples, args.sample_mode, args.agents)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)

    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
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

    groups: Dict[str, ResidualGroupStats] = {}
    num_slots = int(args.residual_slots)
    num_base_modes = 20
    for name in ["all_candidates", *[f"global_top{k}" for k in global_ks], *[f"per_base_top{k}" for k in per_base_topks]]:
        groups[name] = ResidualGroupStats(num_slots=num_slots, num_base_modes=num_base_modes)
    for threshold in gain_thresholds:
        groups[f"gain_gt_{_tag_float(threshold)}"] = ResidualGroupStats(num_slots=num_slots, num_base_modes=num_base_modes)

    print(
        "[diagnose_v58_elite_residual_space] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots} metric={args.oracle_select_metric}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v58_elite_residual_space] warning: selected_samples normalization is diagnostic only")

    cluster_chunks: List[torch.Tensor] = []
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
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
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        agent_mask = batch["agent_mask"].to(device=device).bool()
        score_flat = _candidate_score(flat, ground_truth, metric=str(args.oracle_select_metric))
        base_score = _base_score(slow_output.slow_pred, ground_truth, metric=str(args.oracle_select_metric))
        base_score_flat = _repeat_slots(base_score, int(args.residual_slots))
        gain_flat = base_score_flat - score_flat
        base_rank = _rank_from_score(base_score)
        base_rank_flat = _repeat_slots(base_rank, int(args.residual_slots))
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])
        slot_flat = torch.arange(int(args.residual_slots), device=device, dtype=torch.long).repeat_interleave(num_base_modes)
        slot_flat = slot_flat[None, :, None].expand(batch_size, num_candidates, num_agents)
        residual_flat = flat - base_flat
        direction = _base_direction(slow_output.slow_pred)
        direction_flat = direction[:, None, ...].expand(
            batch_size,
            int(args.residual_slots),
            num_base_modes,
            num_agents,
            2,
        ).reshape(batch_size, num_candidates, num_agents, 2)

        all_indices = torch.arange(num_candidates, device=device, dtype=torch.long)[None, :, None].expand(
            batch_size,
            num_candidates,
            num_agents,
        )
        _add_group_from_indices(
            groups["all_candidates"],
            indices=all_indices,
            gain_flat=gain_flat,
            score_flat=score_flat,
            base_score_flat=base_score_flat,
            slot_flat=slot_flat,
            base_rank_flat=base_rank_flat,
            residual_flat=residual_flat,
            base_direction_flat=direction_flat,
            agent_mask=agent_mask,
        )

        for keep_k in global_ks:
            keep = max(1, min(int(keep_k), num_candidates))
            indices = torch.topk(score_flat, k=keep, dim=1, largest=False).indices
            _add_group_from_indices(
                groups[f"global_top{keep_k}"],
                indices=indices,
                gain_flat=gain_flat,
                score_flat=score_flat,
                base_score_flat=base_score_flat,
                slot_flat=slot_flat,
                base_rank_flat=base_rank_flat,
                residual_flat=residual_flat,
                base_direction_flat=direction_flat,
                agent_mask=agent_mask,
            )

        slot_score = score_flat.reshape(batch_size, int(args.residual_slots), num_base_modes, num_agents)
        for top_k in per_base_topks:
            keep = max(1, min(int(top_k), int(args.residual_slots)))
            slot_indices = torch.topk(slot_score, k=keep, dim=1, largest=False).indices.permute(0, 2, 1, 3)
            mode_ids = torch.arange(num_base_modes, device=device, dtype=torch.long)[None, :, None, None].expand(
                batch_size,
                num_base_modes,
                keep,
                num_agents,
            )
            indices = (slot_indices * num_base_modes + mode_ids).reshape(batch_size, num_base_modes * keep, num_agents)
            _add_group_from_indices(
                groups[f"per_base_top{top_k}"],
                indices=indices,
                gain_flat=gain_flat,
                score_flat=score_flat,
                base_score_flat=base_score_flat,
                slot_flat=slot_flat,
                base_rank_flat=base_rank_flat,
                residual_flat=residual_flat,
                base_direction_flat=direction_flat,
                agent_mask=agent_mask,
            )

        for threshold in gain_thresholds:
            _add_group_from_mask(
                groups[f"gain_gt_{_tag_float(threshold)}"],
                mask=gain_flat > float(threshold),
                gain_flat=gain_flat,
                score_flat=score_flat,
                base_score_flat=base_score_flat,
                slot_flat=slot_flat,
                base_rank_flat=base_rank_flat,
                residual_flat=residual_flat,
                base_direction_flat=direction_flat,
                agent_mask=agent_mask,
            )
        _append_cluster_residuals(
            cluster_chunks,
            residual_flat=residual_flat,
            gain_flat=gain_flat,
            agent_mask=agent_mask,
            threshold=float(args.cluster_gain_threshold),
            limit=int(args.max_cluster_residuals),
        )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_v58_elite_residual_space] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    group_stats = {name: group.finalize() for name, group in groups.items()}
    cluster_stats = _cluster_summary(
        cluster_chunks,
        clusters=int(args.kmeans_clusters),
        iters=int(args.kmeans_iters),
        seed=int(args.seed),
    )
    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v58_elite_residual_space",
            "variant": "v58b0_elite_residual_space_diagnosis",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "oracle_select_metric": args.oracle_select_metric,
            "global_elite_ks": list(global_ks),
            "per_base_topks": list(per_base_topks),
            "gain_thresholds": list(gain_thresholds),
            "cluster_gain_threshold": float(args.cluster_gain_threshold),
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(args.slow_checkpoint).expanduser().resolve().as_posix(),
        "refiner_checkpoint": Path(args.refiner_checkpoint).expanduser().resolve().as_posix(),
        "groups": _coerce_jsonable(group_stats),
        "residual_clusters": _coerce_jsonable(cluster_stats),
    }
    _print_summary(group_stats, cluster_stats)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
