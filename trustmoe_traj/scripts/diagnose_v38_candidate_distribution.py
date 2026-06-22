"""V54-D diagnosis for V38 residual set-generator candidate distribution.

The script evaluates a trained V38-A set generator without retraining.  It
compares the full slots4 candidate pool against several fair-K=20 reductions:

* slot0: keep the first residual slot for every base mode;
* random20_global: randomly sample 20 candidates from the 80-candidate pool;
* random20_per_base: choose one random residual slot for each base mode;
* structured20_endpoint_fps: farthest-point sampling in endpoint space;
* structured20_residual_endpoint_fps: farthest-point sampling in residual
  endpoint space;
* oracle20: GT-based top-20 candidates from the 80-candidate pool.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowPredictorConfig, MoFlowSlowPredictor, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.eval_social_cvae_refiner import _checkpoint_variant, _local_temporal_energy
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    BranchAccumulator,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_benchmark_comparable_run,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _measure_predict_latency_ms,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


class AuxAccumulator:
    """Weighted split-level averages for diagnostic auxiliary metrics."""

    def __init__(self) -> None:
        self.total_valid_agents = 0.0
        self.sums: Dict[str, float] = {}
        self.denominators: Dict[str, float] = {}

    def add(self, values: Mapping[str, float], *, weight: int) -> None:
        if int(weight) <= 0:
            return
        self.total_valid_agents += float(weight)
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * float(weight)

    def add_metric(self, key: str, value: Optional[float], *, weight: int) -> None:
        if value is None or int(weight) <= 0:
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(value) * float(weight)
        self.denominators[key] = self.denominators.get(key, 0.0) + float(weight)

    def finalize(self, prefix: str) -> Dict[str, float]:
        if not self.sums:
            return {}
        result: Dict[str, float] = {}
        for key, value in self.sums.items():
            denominator = self.denominators.get(key, self.total_valid_agents)
            if denominator <= 0:
                continue
            result[f"{prefix}_{key}"] = float(value / denominator)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V38-A slots4 candidate distribution and K=20 reductions.")
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
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _predictor_cfg(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    cfg_path: str,
    checkpoint_path: str,
) -> MoFlowPredictorConfig:
    return MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        device=device,
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
        num_to_gen=int(args.num_to_gen),
    )


def _flatten_refined(refined: torch.Tensor) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    b, s, k, a, t, d = refined.shape
    return refined.reshape(b, s * k, a, t, d)


def _base_for_flat(refined: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    b, s, k, a, t, d = refined.shape
    if tuple(base.shape) != (b, k, a, t, d):
        raise ValueError(f"Base shape {tuple(base.shape)} does not match refined shape {tuple(refined.shape)}")
    return base[:, None, ...].expand(b, s, k, a, t, d).reshape(b, s * k, a, t, d)


def _gather_candidates(flat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if flat.ndim != 5:
        raise ValueError(f"Expected flat [B,N,A,T,2], got {tuple(flat.shape)}")
    if indices.ndim != 3:
        raise ValueError(f"Expected indices [B,K,A], got {tuple(indices.shape)}")
    gather_index = indices.to(device=flat.device, dtype=torch.long)[:, :, :, None, None].expand(
        int(indices.shape[0]),
        int(indices.shape[1]),
        int(indices.shape[2]),
        int(flat.shape[3]),
        int(flat.shape[4]),
    )
    return torch.gather(flat, dim=1, index=gather_index)


def _candidate_score(flat: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(flat - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported oracle metric: {metric!r}")


def _oracle_indices(flat: torch.Tensor, ground_truth: torch.Tensor, *, keep_k: int, metric: str) -> torch.Tensor:
    score = _candidate_score(flat, ground_truth, metric=metric)
    keep = min(int(keep_k), int(score.shape[1]))
    return torch.topk(score, k=keep, dim=1, largest=False).indices


def _all_indices(batch_size: int, num_candidates: int, num_agents: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(num_candidates, device=device, dtype=torch.long)[None, :, None].expand(
        batch_size,
        num_candidates,
        num_agents,
    )


def _slot0_indices(batch_size: int, num_modes: int, num_agents: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(num_modes, device=device, dtype=torch.long)[None, :, None].expand(batch_size, num_modes, num_agents)


def _random_global_indices(
    batch_size: int,
    num_candidates: int,
    num_agents: int,
    *,
    keep_k: int,
    device: torch.device,
) -> torch.Tensor:
    keep = int(keep_k)
    if keep > int(num_candidates):
        raise ValueError(f"keep_k={keep} exceeds num_candidates={num_candidates}")
    result = torch.empty(batch_size, keep, num_agents, dtype=torch.long, device=device)
    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            result[batch_index, :, agent_index] = torch.randperm(num_candidates, device=device)[:keep]
    return result


def _random_per_base_indices(refined: torch.Tensor, *, keep_k: int) -> torch.Tensor:
    b, s, k, a, _t, _d = refined.shape
    keep = int(keep_k)
    if keep != int(k):
        raise ValueError(f"random_per_base expects keep_k == base modes ({k}), got {keep}")
    device = refined.device
    slots = torch.randint(low=0, high=int(s), size=(b, keep, a), device=device)
    modes = torch.arange(k, device=device, dtype=torch.long)[None, :, None].expand(b, keep, a)
    return slots.to(torch.long) * int(k) + modes


def _fps_one(points: torch.Tensor, keep_k: int) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"Expected points [N,F], got {tuple(points.shape)}")
    num_points = int(points.shape[0])
    keep = min(int(keep_k), num_points)
    if keep <= 0:
        raise ValueError("keep_k must be positive")
    centroid = points.mean(dim=0, keepdim=True)
    selected = [int(torch.argmin(torch.sum((points - centroid) ** 2, dim=-1)).item())]
    min_dist = torch.sum((points - points[selected[0]][None, :]) ** 2, dim=-1)
    for _ in range(1, keep):
        next_index = int(torch.argmax(min_dist).item())
        selected.append(next_index)
        next_dist = torch.sum((points - points[next_index][None, :]) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, next_dist)
    return torch.tensor(selected, dtype=torch.long, device=points.device)


def _structured_fps_indices(features: torch.Tensor, *, keep_k: int) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"Expected features [B,N,A,F], got {tuple(features.shape)}")
    b, _n, a, _f = features.shape
    keep = int(keep_k)
    result = torch.empty(b, keep, a, dtype=torch.long, device=features.device)
    for batch_index in range(b):
        for agent_index in range(a):
            result[batch_index, :, agent_index] = _fps_one(features[batch_index, :, agent_index, :], keep)
    return result


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return pairwise.new_zeros((pairwise.shape[0],))
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_pairwise(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, _t, d = prediction.shape
    endpoints = prediction[..., -1, :].permute(0, 2, 1, 3).reshape(b * a, k, d)
    return torch.cdist(endpoints, endpoints, p=2).reshape(b, a, k, k)


def _trajectory_pairwise(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, t, d = prediction.shape
    traj = prediction.permute(0, 2, 1, 3, 4).reshape(b * a, k, t, d)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return pairwise.reshape(b, a, k, k)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    pairwise = _endpoint_pairwise(prediction)
    b, a, k, _ = pairwise.shape
    return _offdiag_mean(pairwise.reshape(b * a, k, k)).reshape(b, a)


def _trajectory_spread(prediction: torch.Tensor) -> torch.Tensor:
    pairwise = _trajectory_pairwise(prediction)
    b, a, k, _ = pairwise.shape
    return _offdiag_mean(pairwise.reshape(b * a, k, k)).reshape(b, a)


def _cluster_count_entropy_one(pairwise: torch.Tensor, eps: float) -> tuple[float, float]:
    num_modes = int(pairwise.shape[0])
    if num_modes <= 0:
        return 0.0, 0.0
    adjacency = pairwise <= float(eps)
    seen = [False] * num_modes
    sizes: List[int] = []
    for start in range(num_modes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = adjacency[current].nonzero(as_tuple=False).reshape(-1).tolist()
            for neighbor in neighbors:
                neighbor_index = int(neighbor)
                if not seen[neighbor_index]:
                    seen[neighbor_index] = True
                    stack.append(neighbor_index)
        sizes.append(size)
    probs = torch.tensor([size / max(num_modes, 1) for size in sizes], dtype=torch.float64)
    entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    if num_modes > 1:
        entropy /= float(torch.log(torch.tensor(float(num_modes), dtype=torch.float64)).item())
    else:
        entropy = 0.0
    return float(len(sizes)), float(entropy)


def _cluster_count_entropy_values(
    pairwise: torch.Tensor,
    *,
    mask: torch.Tensor,
    eps: float,
) -> tuple[List[float], List[float]]:
    pairwise_cpu = pairwise.detach().cpu()
    valid = mask.detach().cpu().bool()
    counts: List[float] = []
    entropies: List[float] = []
    for batch_index in range(int(valid.shape[0])):
        for agent_index in range(int(valid.shape[1])):
            if not bool(valid[batch_index, agent_index].item()):
                continue
            count, entropy = _cluster_count_entropy_one(pairwise_cpu[batch_index, agent_index], float(eps))
            counts.append(count)
            entropies.append(entropy)
    return counts, entropies


def _mean_float(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _mean_pair_ratio(values: Sequence[float], base_values: Sequence[float]) -> Optional[float]:
    ratios = [
        float(value) / max(float(base), 1e-8)
        for value, base in zip(values, base_values)
        if float(base) > 1e-8
    ]
    return _mean_float(ratios)


def _merge_cluster_aux(
    result: Dict[str, float],
    *,
    prefix: str,
    pairwise: torch.Tensor,
    base_pairwise: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    label: str,
) -> None:
    counts, entropies = _cluster_count_entropy_values(pairwise, mask=mask, eps=float(eps))
    base_counts, base_entropies = _cluster_count_entropy_values(base_pairwise, mask=mask, eps=float(eps))
    count_mean = _mean_float(counts)
    entropy_mean = _mean_float(entropies)
    count_ratio = _mean_pair_ratio(counts, base_counts)
    entropy_ratio = _mean_pair_ratio(entropies, base_entropies)
    if count_mean is not None:
        result[f"{prefix}_cluster_count_{label}"] = float(count_mean)
    if count_ratio is not None:
        result[f"{prefix}_cluster_count_ratio_{label}"] = float(count_ratio)
    if entropy_mean is not None:
        result[f"{prefix}_cluster_entropy_{label}"] = float(entropy_mean)
    if entropy_ratio is not None:
        result[f"{prefix}_cluster_entropy_ratio_{label}"] = float(entropy_ratio)


def _unique_base_mode_ratio(selected_flat_indices: Optional[torch.Tensor], *, num_modes: int, mask: torch.Tensor) -> Optional[float]:
    if selected_flat_indices is None:
        return None
    valid = mask.detach().cpu().bool()
    modes = (selected_flat_indices.detach().cpu() % int(num_modes)).to(torch.long)
    values: List[float] = []
    for batch_index in range(int(valid.shape[0])):
        for agent_index in range(int(valid.shape[1])):
            if not bool(valid[batch_index, agent_index].item()):
                continue
            selected = modes[batch_index, :, agent_index].tolist()
            values.append(float(len(set(int(item) for item in selected)) / max(int(num_modes), 1)))
    if not values:
        return None
    return float(sum(values) / len(values))


def _branch_aux(
    prediction: torch.Tensor,
    *,
    base_for_delta: Optional[torch.Tensor],
    spread_base: torch.Tensor,
    mask: torch.Tensor,
    selected_flat_indices: Optional[torch.Tensor],
    num_base_modes: int,
) -> Dict[str, float]:
    valid = mask.to(device=prediction.device, dtype=torch.bool)
    if int(valid.sum().item()) <= 0:
        return {}
    result: Dict[str, float] = {}
    if base_for_delta is not None:
        delta_l2 = torch.linalg.norm(prediction - base_for_delta, dim=-1).mean(dim=-1).mean(dim=1)
        result["delta_l2_mean"] = float(delta_l2[valid].mean().detach().cpu())
    endpoint_pairwise = _endpoint_pairwise(prediction)
    trajectory_pairwise = _trajectory_pairwise(prediction)
    base_endpoint_pairwise = _endpoint_pairwise(spread_base)
    base_trajectory_pairwise = _trajectory_pairwise(spread_base)
    b, a, k, _ = endpoint_pairwise.shape
    base_b, base_a, base_k, _ = base_endpoint_pairwise.shape
    if (base_b, base_a) != (b, a):
        raise ValueError(
            f"spread_base batch/agent shape {(base_b, base_a)} does not match prediction {(b, a)}"
        )
    endpoint_spread = _offdiag_mean(endpoint_pairwise.reshape(b * a, k, k)).reshape(b, a)
    base_endpoint_spread = _offdiag_mean(base_endpoint_pairwise.reshape(base_b * base_a, base_k, base_k)).reshape(
        base_b,
        base_a,
    )
    b, a, k, _ = trajectory_pairwise.shape
    base_b, base_a, base_k, _ = base_trajectory_pairwise.shape
    if (base_b, base_a) != (b, a):
        raise ValueError(
            f"spread_base batch/agent shape {(base_b, base_a)} does not match prediction {(b, a)}"
        )
    trajectory_spread = _offdiag_mean(trajectory_pairwise.reshape(b * a, k, k)).reshape(b, a)
    base_trajectory_spread = _offdiag_mean(base_trajectory_pairwise.reshape(base_b * base_a, base_k, base_k)).reshape(
        base_b,
        base_a,
    )
    endpoint_ratio = endpoint_spread / base_endpoint_spread.abs().clamp_min(1e-8)
    trajectory_ratio = trajectory_spread / base_trajectory_spread.abs().clamp_min(1e-8)
    result["endpoint_ratio"] = float(endpoint_ratio[valid].mean().detach().cpu())
    result["trajectory_ratio"] = float(trajectory_ratio[valid].mean().detach().cpu())
    for eps, label in ((0.5, "eps05"), (1.0, "eps10")):
        _merge_cluster_aux(
            result,
            prefix="endpoint",
            pairwise=endpoint_pairwise,
            base_pairwise=base_endpoint_pairwise,
            mask=valid,
            eps=eps,
            label=label,
        )
        _merge_cluster_aux(
            result,
            prefix="trajectory",
            pairwise=trajectory_pairwise,
            base_pairwise=base_trajectory_pairwise,
            mask=valid,
            eps=eps,
            label=label,
        )
    unique_ratio = _unique_base_mode_ratio(selected_flat_indices, num_modes=int(num_base_modes), mask=valid)
    if unique_ratio is not None:
        result["unique_base_mode_ratio"] = float(unique_ratio)
    return result


def _add_branch(
    accumulators: Mapping[str, BranchAccumulator],
    aux_accumulators: Mapping[str, AuxAccumulator],
    *,
    field_name: str,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    miss_threshold: float,
    latencies_ms: Iterable[float],
    base_for_delta: Optional[torch.Tensor] = None,
    spread_base: Optional[torch.Tensor] = None,
    selected_flat_indices: Optional[torch.Tensor] = None,
    num_base_modes: Optional[int] = None,
) -> None:
    summary = evaluate_model_output(
        {field_name: prediction},
        batch,
        miss_threshold=float(miss_threshold),
        prediction_fields=(field_name,),
    )
    accumulators[field_name].add_chunk(summary.metrics, latencies_ms)
    if spread_base is None:
        return
    valid_count = int(batch["agent_mask"].bool().sum().item())
    aux_accumulators[field_name].add(
        _branch_aux(
            prediction,
            base_for_delta=base_for_delta,
            spread_base=spread_base,
            mask=batch["agent_mask"].bool(),
            selected_flat_indices=selected_flat_indices,
            num_base_modes=int(num_base_modes or spread_base.shape[1]),
        ),
        weight=valid_count,
    )


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "None"
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.6f}"


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def _std(values: Sequence[float]) -> Optional[float]:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = float(sum(values) / len(values))
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def _add_random_aggregates(
    metrics: Dict[str, float],
    *,
    group_name: str,
    trial_branches: Sequence[str],
) -> None:
    for metric_name in METRICS:
        values = [
            float(metrics[f"{branch}_{metric_name}"])
            for branch in trial_branches
            if f"{branch}_{metric_name}" in metrics
        ]
        mean = _mean(values)
        std = _std(values)
        if mean is not None:
            metrics[f"{group_name}_mean_{metric_name}"] = float(mean)
        if std is not None:
            metrics[f"{group_name}_std_{metric_name}"] = float(std)
    for aux_name in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
        values = [
            float(metrics[f"{branch}_{aux_name}"])
            for branch in trial_branches
            if f"{branch}_{aux_name}" in metrics
        ]
        mean = _mean(values)
        std = _std(values)
        if mean is not None:
            metrics[f"{group_name}_mean_{aux_name}"] = float(mean)
        if std is not None:
            metrics[f"{group_name}_std_{aux_name}"] = float(std)


def _print_summary(
    metrics: Mapping[str, float],
    *,
    deterministic_branches: Sequence[str],
    random_groups: Mapping[str, Sequence[str]],
) -> None:
    print("\n[diagnose_v38_candidate_distribution] branch - slow deltas")
    for field_name in deterministic_branches:
        print(f"\n-- {field_name} --")
        for metric_name in METRICS:
            branch = _metric(metrics, field_name, metric_name)
            slow = _metric(metrics, "slow_pred", metric_name)
            delta = None if branch is None or slow is None else branch - slow
            print(f"d{metric_name}: {_fmt(delta, signed=True)}  branch={_fmt(branch)}  slow={_fmt(slow)}")
        for aux_name in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(float(metrics[key]))}")
    for group_name, branches in random_groups.items():
        print(f"\n-- {group_name} ({len(branches)} trials) --")
        for metric_name in METRICS:
            mean_key = f"{group_name}_mean_{metric_name}"
            std_key = f"{group_name}_std_{metric_name}"
            slow = _metric(metrics, "slow_pred", metric_name)
            mean_value = metrics.get(mean_key)
            delta = None if mean_value is None or slow is None else float(mean_value) - slow
            print(
                f"mean d{metric_name}: {_fmt(delta, signed=True)}  "
                f"mean={_fmt(mean_value)}  std={_fmt(metrics.get(std_key))}  slow={_fmt(slow)}"
            )
        for aux_name in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
            mean_key = f"{group_name}_mean_{aux_name}"
            std_key = f"{group_name}_std_{aux_name}"
            if mean_key in metrics:
                print(f"mean {aux_name}: {_fmt(float(metrics[mean_key]))}  std={_fmt(metrics.get(std_key))}")


def main() -> None:
    args = build_parser().parse_args()
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1 for V38-A")
    if int(args.random_trials) < 0:
        raise SystemExit("--random-trials must be non-negative")
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
    max_slots = int(getattr(refiner.config, "max_residual_slots", 1))
    if int(args.residual_slots) > max_slots:
        raise SystemExit(f"--residual-slots {args.residual_slots} exceeds checkpoint max_residual_slots={max_slots}")

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

    random_global_branches = [f"v38_random20_global_t{index:02d}_pred" for index in range(int(args.random_trials))]
    random_per_base_branches = [f"v38_random20_per_base_t{index:02d}_pred" for index in range(int(args.random_trials))]
    deterministic_branches = [
        "v38_slot0_20_pred",
        "v38_structured20_endpoint_fps_pred",
        "v38_structured20_residual_endpoint_fps_pred",
        "v38_oracle20_from80_pred",
        "v38_full80_pred",
    ]
    branches = ["slow_pred", *deterministic_branches, *random_global_branches, *random_per_base_branches]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}
    random_groups = {
        "v38_random20_global": random_global_branches,
        "v38_random20_per_base": random_per_base_branches,
    }

    print(
        "[diagnose_v38_candidate_distribution] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots} keep_k={args.keep_k} "
        f"random_trials={args.random_trials} oracle_metric={args.oracle_select_metric}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v38_candidate_distribution] warning: selected_samples normalization is diagnostic only")

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        slow_latencies, slow_output = _measure_predict_latency_ms(
            lambda: slow_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
        slow_summary = evaluate_model_output(
            slow_output,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("slow_pred",),
        )
        accumulators["slow_pred"].add_chunk(slow_summary.metrics, slow_latencies)

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

        refiner_latencies, refiner_outputs = _measure_predict_latency_ms(
            lambda: refiner.refine(
                slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                num_samples=int(args.residual_slots),
                z_mode="slots",
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        refined = refiner_outputs["refined"]
        flat = _flatten_refined(refined)
        base_flat = _base_for_flat(refined, slow_output.slow_pred)
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])

        slot0_indices = _slot0_indices(batch_size, num_base_modes, num_agents, device=flat.device)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v38_slot0_20_pred",
            prediction=refined[:, 0],
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=slot0_indices,
            num_base_modes=num_base_modes,
        )

        endpoint_indices = _structured_fps_indices(flat[..., -1, :], keep_k=int(args.keep_k))
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v38_structured20_endpoint_fps_pred",
            prediction=_gather_candidates(flat, endpoint_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=_gather_candidates(base_flat, endpoint_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=endpoint_indices,
            num_base_modes=num_base_modes,
        )

        residual_endpoint_indices = _structured_fps_indices((flat - base_flat)[..., -1, :], keep_k=int(args.keep_k))
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v38_structured20_residual_endpoint_fps_pred",
            prediction=_gather_candidates(flat, residual_endpoint_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=_gather_candidates(base_flat, residual_endpoint_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=residual_endpoint_indices,
            num_base_modes=num_base_modes,
        )

        oracle_indices = _oracle_indices(flat, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v38_oracle20_from80_pred",
            prediction=_gather_candidates(flat, oracle_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=_gather_candidates(base_flat, oracle_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=oracle_indices,
            num_base_modes=num_base_modes,
        )

        full_indices = _all_indices(batch_size, num_candidates, num_agents, device=flat.device)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v38_full80_pred",
            prediction=flat,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=base_flat,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=full_indices,
            num_base_modes=num_base_modes,
        )

        for trial_index, field_name in enumerate(random_global_branches):
            random.seed(int(args.seed) + trial_index)
            random_indices = _random_global_indices(
                batch_size,
                num_candidates,
                num_agents,
                keep_k=int(args.keep_k),
                device=flat.device,
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=_gather_candidates(flat, random_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=_gather_candidates(base_flat, random_indices),
                spread_base=slow_output.slow_pred,
                selected_flat_indices=random_indices,
                num_base_modes=num_base_modes,
            )

        for trial_index, field_name in enumerate(random_per_base_branches):
            random.seed(int(args.seed) + 10_000 + trial_index)
            random_indices = _random_per_base_indices(refined, keep_k=int(args.keep_k))
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=_gather_candidates(flat, random_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=_gather_candidates(base_flat, random_indices),
                spread_base=slow_output.slow_pred,
                selected_flat_indices=random_indices,
                num_base_modes=num_base_modes,
            )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_v38_candidate_distribution] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[field_name].finalize(field_name))
    for group_name, trial_branches in random_groups.items():
        _add_random_aggregates(metrics, group_name=group_name, trial_branches=trial_branches)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v38_candidate_distribution",
            "variant": "v54d_v38_candidate_distribution_diagnosis",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "random_trials": int(args.random_trials),
            "oracle_select_metric": args.oracle_select_metric,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "deterministic_branches": list(deterministic_branches),
        "random_groups": {key: list(value) for key, value in random_groups.items()},
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
        "metrics": _coerce_jsonable(metrics),
    }
    _print_summary(metrics, deterministic_branches=deterministic_branches, random_groups=random_groups)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
