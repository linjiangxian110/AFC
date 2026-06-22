"""Diagnose why a Residual Graduate checkpoint changes best-of-K metrics.

This script reruns the fast student and Residual Graduate predictions, then
compares their full K-mode output distributions. It is meant to answer two
questions before changing the training loss:

1. Did graduate reduce mode diversity / endpoint spread?
2. Did graduate harm the original student-best mode?
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import displacement_errors, infer_ground_truth_from_batch
from trustmoe_traj.models import MoFlowFastPredictor
from trustmoe_traj.scripts.eval_residual_graduate import _load_graduate_model
from trustmoe_traj.scripts.interaction_energy_features import (
    build_per_agent_scene_interaction_features,
    build_per_agent_scene_temporal_interaction_features,
)
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _build_predictor_config,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Residual Graduate diversity and best-mode effects on ETH splits."
    )
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate", dest="rotate", action="store_true")
    rotate_group.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.set_defaults(rotate=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)

    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--top-k-records", type=int, default=20)
    parser.add_argument("--save-records", action="store_true")

    parser.add_argument("--graduate-checkpoint", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _ensure_prediction5(prediction: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(prediction, dtype=torch.float32)
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(f"{name} must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(tensor.shape)}")


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    if pairwise.ndim != 3:
        raise ValueError(f"pairwise must have shape [N, K, K], got {tuple(pairwise.shape)}")
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return torch.zeros((pairwise.shape[0],), dtype=pairwise.dtype, device=pairwise.device)
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal pairwise distance between mode endpoints.

    Returns [B, A].
    """

    pred = _ensure_prediction5(prediction, name="prediction")
    batch_size, num_modes, num_agents, _pred_len, coord_dim = pred.shape
    endpoints = pred[..., -1, :].permute(0, 2, 1, 3).reshape(batch_size * num_agents, num_modes, coord_dim)
    pairwise = torch.cdist(endpoints, endpoints, p=2)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _trajectory_diversity(prediction: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal pairwise distance between full trajectories.

    Pair distance is averaged over time, so the unit remains dataset distance.
    Returns [B, A].
    """

    pred = _ensure_prediction5(prediction, name="prediction")
    batch_size, num_modes, num_agents, pred_len, coord_dim = pred.shape
    traj = pred.permute(0, 2, 1, 3, 4).reshape(batch_size * num_agents, num_modes, pred_len, coord_dim)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean(pairwise).reshape(batch_size, num_agents)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return numerator / denominator.abs().clamp_min(float(eps))


def _to_float(value: Any) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def _active_agent_count(sample: Mapping[str, Any]) -> int:
    agent_mask = sample.get("agent_mask")
    if agent_mask is None:
        return int(torch.as_tensor(sample["past_traj"]).shape[0])
    return int(torch.as_tensor(agent_mask).reshape(-1).bool().sum().item())


def _expanded_selected_sample_indices(
    chunk_pairs: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    sample_mode: str,
) -> List[int]:
    if sample_mode != "per_agent":
        return [int(sample_index) for sample_index, _sample in chunk_pairs]

    expanded: List[int] = []
    for sample_index, sample in chunk_pairs:
        expanded.extend([int(sample_index)] * _active_agent_count(sample))
    return expanded


def _quantiles(values: Sequence[float], quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor([float(item) for item in values], dtype=torch.float32)
    return {f"p{int(q * 100):02d}": float(torch.quantile(tensor, float(q)).item()) for q in quantiles}


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(item) for item in values) / len(values))


def _rate(flags: Iterable[bool]) -> Optional[float]:
    items = [bool(item) for item in flags]
    if not items:
        return None
    return float(sum(1 for item in items if item) / len(items))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = torch.tensor([float(item) for item in xs], dtype=torch.float32)
    y = torch.tensor([float(item) for item in ys], dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float((x * y).sum().item() / denom.item())


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        return [float(record[key]) for record in records if record.get(key) is not None]

    def flags(key: str) -> List[bool]:
        return [bool(record[key]) for record in records if record.get(key) is not None]

    fde_min_delta = values("fde_min_delta")
    ade_min_delta = values("ade_min_delta")
    fde_avg_delta = values("fde_avg_delta")
    ade_avg_delta = values("ade_avg_delta")
    endpoint_delta = values("endpoint_spread_delta")
    endpoint_ratio = values("endpoint_spread_ratio")
    traj_delta = values("trajectory_diversity_delta")
    traj_ratio = values("trajectory_diversity_ratio")
    student_best_fde_delta = values("student_best_mode_fde_delta")
    student_best_ade_delta = values("student_best_mode_ade_delta")

    return {
        "num_valid_agents": int(len(records)),
        "fde_min_delta_mean": _mean(fde_min_delta),
        "fde_min_delta_quantiles": _quantiles(fde_min_delta),
        "fde_min_worse_rate": _rate(item > 0 for item in fde_min_delta),
        "fde_min_improve_rate": _rate(item < 0 for item in fde_min_delta),
        "ade_min_delta_mean": _mean(ade_min_delta),
        "ade_min_delta_quantiles": _quantiles(ade_min_delta),
        "fde_avg_delta_mean": _mean(fde_avg_delta),
        "ade_avg_delta_mean": _mean(ade_avg_delta),
        "endpoint_spread_student_mean": _mean(values("endpoint_spread_student")),
        "endpoint_spread_graduate_mean": _mean(values("endpoint_spread_graduate")),
        "endpoint_spread_delta_mean": _mean(endpoint_delta),
        "endpoint_spread_ratio_mean": _mean(endpoint_ratio),
        "endpoint_spread_delta_quantiles": _quantiles(endpoint_delta),
        "endpoint_spread_decrease_rate": _rate(item < 0 for item in endpoint_delta),
        "trajectory_diversity_student_mean": _mean(values("trajectory_diversity_student")),
        "trajectory_diversity_graduate_mean": _mean(values("trajectory_diversity_graduate")),
        "trajectory_diversity_delta_mean": _mean(traj_delta),
        "trajectory_diversity_ratio_mean": _mean(traj_ratio),
        "trajectory_diversity_delta_quantiles": _quantiles(traj_delta),
        "trajectory_diversity_decrease_rate": _rate(item < 0 for item in traj_delta),
        "student_best_mode_fde_delta_mean": _mean(student_best_fde_delta),
        "student_best_mode_fde_delta_quantiles": _quantiles(student_best_fde_delta),
        "student_best_mode_worse_rate": _rate(item > 0 for item in student_best_fde_delta),
        "student_best_mode_improve_rate": _rate(item < 0 for item in student_best_fde_delta),
        "student_best_mode_ade_delta_mean": _mean(student_best_ade_delta),
        "student_best_mode_ade_delta_quantiles": _quantiles(student_best_ade_delta),
        "best_mode_switch_rate": _rate(flags("best_fde_mode_switched")),
        "student_best_hurt_and_fde_min_worse_rate": _rate(
            bool(record["student_best_mode_fde_delta"] > 0 and record["fde_min_delta"] > 0)
            for record in records
        ),
        "student_best_hurt_but_recovered_by_other_mode_rate": _rate(
            bool(record["student_best_mode_fde_delta"] > 0 and record["fde_min_delta"] <= 0)
            for record in records
        ),
        "gate_mean": _mean(values("gate_mean")),
        "delta_l2_mean": _mean(values("delta_l2_mean")),
        "best_selector_prob_mean": _mean(values("best_selector_prob_mean")),
        "best_selector_student_best_prob_mean": _mean(values("best_selector_student_best_prob")),
        "best_refine_delta_l2_mean": _mean(values("best_refine_delta_l2_mean")),
        "temporal_gate_mean": _mean(values("temporal_gate_mean")),
        "temporal_gate_early_mean": _mean(values("temporal_gate_early_mean")),
        "temporal_gate_mid_mean": _mean(values("temporal_gate_mid_mean")),
        "temporal_gate_late_mean": _mean(values("temporal_gate_late_mean")),
        "temporal_refine_delta_l2_mean": _mean(values("temporal_refine_delta_l2_mean")),
        "pearson_endpoint_spread_delta_vs_fde_min_delta": _pearson(endpoint_delta, fde_min_delta),
        "pearson_trajectory_diversity_delta_vs_fde_min_delta": _pearson(traj_delta, fde_min_delta),
        "pearson_student_best_fde_delta_vs_fde_min_delta": _pearson(student_best_fde_delta, fde_min_delta),
    }


def _top_records(records: Sequence[Mapping[str, Any]], *, top_k: int) -> Dict[str, Any]:
    if top_k <= 0:
        return {}

    def sort_records(key: str, *, reverse: bool) -> List[Mapping[str, Any]]:
        return sorted(records, key=lambda item: float(item.get(key, 0.0)), reverse=reverse)[:top_k]

    return {
        "worst_fde_min_regressions": sort_records("fde_min_delta", reverse=True),
        "best_fde_min_improvements": sort_records("fde_min_delta", reverse=False),
        "largest_endpoint_spread_decreases": sort_records("endpoint_spread_delta", reverse=False),
        "largest_student_best_mode_hurts": sort_records("student_best_mode_fde_delta", reverse=True),
    }


def _append_chunk_records(
    records: List[Dict[str, Any]],
    *,
    selected_indices: Sequence[int],
    student_pred: torch.Tensor,
    graduate_pred: torch.Tensor,
    gate: torch.Tensor,
    delta_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    miss_threshold: float,
    best_mode_selector_prob: Optional[torch.Tensor] = None,
    best_mode_refine_delta: Optional[torch.Tensor] = None,
    temporal_repair_gate: Optional[torch.Tensor] = None,
    temporal_refine_delta: Optional[torch.Tensor] = None,
) -> None:
    student = _ensure_prediction5(student_pred, name="student_pred")
    graduate = _ensure_prediction5(graduate_pred, name="graduate_pred").to(device=student.device)
    gt = ground_truth.to(device=student.device, dtype=torch.float32)
    valid = agent_mask.to(device=student.device).bool()

    student_errors = displacement_errors(student, gt, agent_mask=valid)
    graduate_errors = displacement_errors(graduate, gt, agent_mask=valid)
    student_ade = student_errors["ade_per_mode_agent"]
    student_fde = student_errors["fde_per_mode_agent"]
    graduate_ade = graduate_errors["ade_per_mode_agent"]
    graduate_fde = graduate_errors["fde_per_mode_agent"]

    student_ade_min, student_best_ade_mode = student_ade.min(dim=1)
    student_fde_min, student_best_fde_mode = student_fde.min(dim=1)
    graduate_ade_min, graduate_best_ade_mode = graduate_ade.min(dim=1)
    graduate_fde_min, graduate_best_fde_mode = graduate_fde.min(dim=1)

    student_ade_at_student_best_fde = student_ade.gather(1, student_best_fde_mode[:, None, :]).squeeze(1)
    graduate_ade_at_student_best_fde = graduate_ade.gather(1, student_best_fde_mode[:, None, :]).squeeze(1)
    graduate_fde_at_student_best_fde = graduate_fde.gather(1, student_best_fde_mode[:, None, :]).squeeze(1)

    student_ade_avg = student_ade.mean(dim=1)
    student_fde_avg = student_fde.mean(dim=1)
    graduate_ade_avg = graduate_ade.mean(dim=1)
    graduate_fde_avg = graduate_fde.mean(dim=1)

    student_endpoint = _endpoint_spread(student)
    graduate_endpoint = _endpoint_spread(graduate)
    student_traj_div = _trajectory_diversity(student)
    graduate_traj_div = _trajectory_diversity(graduate)

    gate_agent_mean = gate.detach().to(device=student.device, dtype=torch.float32).mean(dim=(1, 3, 4))
    delta_l2 = torch.linalg.norm(delta_pred.detach().to(device=student.device, dtype=torch.float32), dim=-1).mean(
        dim=(1, 3)
    )
    selector_agent_mean = None
    selector_at_student_best = None
    if best_mode_selector_prob is not None:
        selector_prob = best_mode_selector_prob.detach().to(device=student.device, dtype=torch.float32)
        if tuple(selector_prob.shape) != (student.shape[0], student.shape[1], student.shape[2]):
            raise ValueError(
                "best_mode_selector_prob must have shape [B, K, A], "
                f"got {tuple(selector_prob.shape)} for student shape {tuple(student.shape)}"
            )
        selector_agent_mean = selector_prob.mean(dim=1)
        selector_at_student_best = selector_prob.gather(dim=1, index=student_best_fde_mode[:, None, :]).squeeze(1)
    best_refine_l2 = None
    if best_mode_refine_delta is not None:
        refine_delta = best_mode_refine_delta.detach().to(device=student.device, dtype=torch.float32)
        if tuple(refine_delta.shape) != tuple(student.shape):
            raise ValueError(
                "best_mode_refine_delta must have shape [B, K, A, T, C], "
                f"got {tuple(refine_delta.shape)} for student shape {tuple(student.shape)}"
            )
        best_refine_l2 = torch.linalg.norm(refine_delta, dim=-1).mean(dim=(1, 3))
    temporal_gate_agent_mean = None
    temporal_gate_early = None
    temporal_gate_mid = None
    temporal_gate_late = None
    if temporal_repair_gate is not None:
        temporal_gate = temporal_repair_gate.detach().to(device=student.device, dtype=torch.float32)
        if tuple(temporal_gate.shape[:4]) != tuple(student.shape[:4]):
            raise ValueError(
                "temporal_repair_gate must have shape [B, K, A, T, 1], "
                f"got {tuple(temporal_gate.shape)} for student shape {tuple(student.shape)}"
            )
        temporal_gate_agent_mean = temporal_gate.mean(dim=(1, 3, 4))
        pred_len = int(temporal_gate.shape[3])
        first = max(pred_len // 3, 1)
        second = max((2 * pred_len) // 3, first + 1)
        temporal_gate_early = temporal_gate[:, :, :, :first, :].mean(dim=(1, 3, 4))
        temporal_gate_mid = temporal_gate[:, :, :, first:second, :].mean(dim=(1, 3, 4))
        temporal_gate_late = temporal_gate[:, :, :, second:, :].mean(dim=(1, 3, 4))
    temporal_refine_l2 = None
    if temporal_refine_delta is not None:
        temporal_delta = temporal_refine_delta.detach().to(device=student.device, dtype=torch.float32)
        if tuple(temporal_delta.shape) != tuple(student.shape):
            raise ValueError(
                "temporal_refine_delta must have shape [B, K, A, T, C], "
                f"got {tuple(temporal_delta.shape)} for student shape {tuple(student.shape)}"
            )
        temporal_refine_l2 = torch.linalg.norm(temporal_delta, dim=-1).mean(dim=(1, 3))

    batch_size, num_agents = valid.shape
    for batch_index in range(int(batch_size)):
        selected_index = int(selected_indices[batch_index])
        for agent_index in range(int(num_agents)):
            if not bool(valid[batch_index, agent_index].item()):
                continue

            s_fde_min = student_fde_min[batch_index, agent_index]
            g_fde_min = graduate_fde_min[batch_index, agent_index]
            s_ade_min = student_ade_min[batch_index, agent_index]
            g_ade_min = graduate_ade_min[batch_index, agent_index]
            g_fde_at_sbest = graduate_fde_at_student_best_fde[batch_index, agent_index]
            s_ade_at_sbest = student_ade_at_student_best_fde[batch_index, agent_index]
            g_ade_at_sbest = graduate_ade_at_student_best_fde[batch_index, agent_index]

            endpoint_s = student_endpoint[batch_index, agent_index]
            endpoint_g = graduate_endpoint[batch_index, agent_index]
            traj_s = student_traj_div[batch_index, agent_index]
            traj_g = graduate_traj_div[batch_index, agent_index]

            fde_min_delta = g_fde_min - s_fde_min
            ade_min_delta = g_ade_min - s_ade_min
            student_best_fde_delta = g_fde_at_sbest - s_fde_min
            student_best_ade_delta = g_ade_at_sbest - s_ade_at_sbest
            endpoint_delta = endpoint_g - endpoint_s
            traj_delta = traj_g - traj_s

            record = {
                "eval_item_index": int(len(records)),
                "selected_sample_index": selected_index,
                "batch_index": int(batch_index),
                "agent_axis_index": int(agent_index),
                "student_ADE_min": _to_float(s_ade_min),
                "graduate_ADE_min": _to_float(g_ade_min),
                "ade_min_delta": _to_float(ade_min_delta),
                "student_FDE_min": _to_float(s_fde_min),
                "graduate_FDE_min": _to_float(g_fde_min),
                "fde_min_delta": _to_float(fde_min_delta),
                "student_ADE_avg": _to_float(student_ade_avg[batch_index, agent_index]),
                "graduate_ADE_avg": _to_float(graduate_ade_avg[batch_index, agent_index]),
                "ade_avg_delta": _to_float(graduate_ade_avg[batch_index, agent_index] - student_ade_avg[batch_index, agent_index]),
                "student_FDE_avg": _to_float(student_fde_avg[batch_index, agent_index]),
                "graduate_FDE_avg": _to_float(graduate_fde_avg[batch_index, agent_index]),
                "fde_avg_delta": _to_float(graduate_fde_avg[batch_index, agent_index] - student_fde_avg[batch_index, agent_index]),
                "student_best_FDE_mode": int(student_best_fde_mode[batch_index, agent_index].item()),
                "graduate_best_FDE_mode": int(graduate_best_fde_mode[batch_index, agent_index].item()),
                "best_fde_mode_switched": bool(
                    int(student_best_fde_mode[batch_index, agent_index].item())
                    != int(graduate_best_fde_mode[batch_index, agent_index].item())
                ),
                "student_best_ADE_mode": int(student_best_ade_mode[batch_index, agent_index].item()),
                "graduate_best_ADE_mode": int(graduate_best_ade_mode[batch_index, agent_index].item()),
                "student_best_mode_FDE_after_graduate": _to_float(g_fde_at_sbest),
                "student_best_mode_fde_delta": _to_float(student_best_fde_delta),
                "student_best_mode_ADE_after_graduate": _to_float(g_ade_at_sbest),
                "student_best_mode_ade_delta": _to_float(student_best_ade_delta),
                "student_miss": bool(float(s_fde_min.detach().cpu()) > float(miss_threshold)),
                "graduate_miss": bool(float(g_fde_min.detach().cpu()) > float(miss_threshold)),
                "endpoint_spread_student": _to_float(endpoint_s),
                "endpoint_spread_graduate": _to_float(endpoint_g),
                "endpoint_spread_delta": _to_float(endpoint_delta),
                "endpoint_spread_ratio": _to_float(_safe_ratio(endpoint_g, endpoint_s)),
                "trajectory_diversity_student": _to_float(traj_s),
                "trajectory_diversity_graduate": _to_float(traj_g),
                "trajectory_diversity_delta": _to_float(traj_delta),
                "trajectory_diversity_ratio": _to_float(_safe_ratio(traj_g, traj_s)),
                "gate_mean": _to_float(gate_agent_mean[batch_index, agent_index]),
                "delta_l2_mean": _to_float(delta_l2[batch_index, agent_index]),
            }
            if selector_agent_mean is not None and selector_at_student_best is not None:
                record["best_selector_prob_mean"] = _to_float(selector_agent_mean[batch_index, agent_index])
                record["best_selector_student_best_prob"] = _to_float(
                    selector_at_student_best[batch_index, agent_index]
                )
            if best_refine_l2 is not None:
                record["best_refine_delta_l2_mean"] = _to_float(best_refine_l2[batch_index, agent_index])
            if temporal_gate_agent_mean is not None:
                record["temporal_gate_mean"] = _to_float(temporal_gate_agent_mean[batch_index, agent_index])
            if temporal_gate_early is not None and temporal_gate_mid is not None and temporal_gate_late is not None:
                record["temporal_gate_early_mean"] = _to_float(temporal_gate_early[batch_index, agent_index])
                record["temporal_gate_mid_mean"] = _to_float(temporal_gate_mid[batch_index, agent_index])
                record["temporal_gate_late_mean"] = _to_float(temporal_gate_late[batch_index, agent_index])
            if temporal_refine_l2 is not None:
                record["temporal_refine_delta_l2_mean"] = _to_float(temporal_refine_l2[batch_index, agent_index])
            records.append(record)


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("[diagnose_residual_graduate] summary")
    keys = [
        "num_valid_agents",
        "fde_min_delta_mean",
        "fde_min_worse_rate",
        "fde_min_improve_rate",
        "fde_avg_delta_mean",
        "endpoint_spread_student_mean",
        "endpoint_spread_graduate_mean",
        "endpoint_spread_delta_mean",
        "endpoint_spread_ratio_mean",
        "endpoint_spread_decrease_rate",
        "trajectory_diversity_student_mean",
        "trajectory_diversity_graduate_mean",
        "trajectory_diversity_delta_mean",
        "trajectory_diversity_ratio_mean",
        "trajectory_diversity_decrease_rate",
        "student_best_mode_fde_delta_mean",
        "student_best_mode_worse_rate",
        "student_best_mode_improve_rate",
        "best_mode_switch_rate",
        "student_best_hurt_and_fde_min_worse_rate",
        "student_best_hurt_but_recovered_by_other_mode_rate",
        "gate_mean",
        "delta_l2_mean",
        "best_selector_prob_mean",
        "best_selector_student_best_prob_mean",
        "best_refine_delta_l2_mean",
        "temporal_gate_mean",
        "temporal_gate_early_mean",
        "temporal_gate_mid_mean",
        "temporal_gate_late_mean",
        "temporal_refine_delta_l2_mean",
        "pearson_endpoint_spread_delta_vs_fde_min_delta",
        "pearson_trajectory_diversity_delta_vs_fde_min_delta",
        "pearson_student_best_fde_delta_vs_fde_min_delta",
    ]
    for key in keys:
        print(f"{key}={summary.get(key)}")


def main() -> None:
    args = build_parser().parse_args()
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)

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

    fast_predictor = MoFlowFastPredictor(
        _build_predictor_config(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.fast_cfg_path,
            checkpoint_path=args.fast_checkpoint,
        )
    )
    graduate_model, graduate_payload = _load_graduate_model(args.graduate_checkpoint, device=device)

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=(fast_predictor,),
        samples=selected_samples,
        stats_owner=fast_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    diagnostic_normalization = _is_diagnostic_normalization_source(protocol_settings.normalization_source)
    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )

    records: List[Dict[str, Any]] = []
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))

    with torch.no_grad():
        for chunk_index, chunk_pairs in enumerate(chunks, start=1):
            selected_indices = _expanded_selected_sample_indices(
                chunk_pairs,
                sample_mode=args.sample_mode,
            )
            chunk = [sample for _sample_index, sample in chunk_pairs]

            fast_batch = fast_predictor.build_moflow_batch(
                chunk,
                normalization_stats=normalization_stats,
                as_torch=True,
            )
            fast_output = fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen)
            interaction_energy_features = None
            if bool(getattr(graduate_model.config, "use_interaction_energy", False)):
                interaction_energy_features = build_per_agent_scene_interaction_features(
                    chunk,
                    fast_output.fast_pred,
                    rotate=bool(args.rotate),
                    rotate_time_frame=int(args.rotate_time_frame),
                    collision_sigma=float(getattr(graduate_model.config, "collision_sigma", 0.5)),
                    collision_radius=float(getattr(graduate_model.config, "collision_radius", 0.2)),
                    no_neighbor_distance=float(getattr(graduate_model.config, "no_neighbor_distance", 10.0)),
                    temporal_stride=int(getattr(graduate_model.config, "interaction_energy_temporal_stride", 1)),
                )
            temporal_interaction_energy_features = None
            if bool(getattr(graduate_model.config, "use_temporal_energy_refiner", False)):
                temporal_interaction_energy_features = build_per_agent_scene_temporal_interaction_features(
                    chunk,
                    fast_output.fast_pred,
                    rotate=bool(args.rotate),
                    rotate_time_frame=int(args.rotate_time_frame),
                    collision_sigma=float(getattr(graduate_model.config, "collision_sigma", 0.5)),
                    collision_radius=float(getattr(graduate_model.config, "collision_radius", 0.2)),
                    no_neighbor_distance=float(getattr(graduate_model.config, "no_neighbor_distance", 10.0)),
                )
            graduate_output = graduate_model(
                fast_output.fast_pred,
                fast_batch["past_traj_original_scale"],
                agent_mask=fast_batch.get("agent_mask"),
                interaction_energy_features=interaction_energy_features,
                temporal_interaction_energy_features=temporal_interaction_energy_features,
            )
            gt_payload = infer_ground_truth_from_batch(fast_batch)

            batch_items = int(gt_payload["ground_truth"].shape[0])
            if len(selected_indices) != batch_items:
                raise RuntimeError(
                    "Internal eval-item mapping mismatch: "
                    f"expanded_selected_indices={len(selected_indices)} vs batch_items={batch_items}. "
                    "This usually means sample expansion logic no longer matches build_moflow_eth_batch."
                )

            _append_chunk_records(
                records,
                selected_indices=selected_indices,
                student_pred=fast_output.fast_pred,
                graduate_pred=graduate_output["graduate_pred"],
                gate=graduate_output["gate"],
                delta_pred=graduate_output["delta_pred"],
                ground_truth=gt_payload["ground_truth"],
                agent_mask=gt_payload["agent_mask"],
                miss_threshold=float(args.miss_threshold),
                best_mode_selector_prob=graduate_output.get("best_mode_selector_prob"),
                best_mode_refine_delta=graduate_output.get("best_mode_refine_delta"),
                temporal_repair_gate=graduate_output.get("temporal_repair_gate"),
                temporal_refine_delta=graduate_output.get("temporal_refine_delta"),
            )

            should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(args.log_every, 1) == 0
            if should_log:
                print(
                    f"[diagnose_residual_graduate] processed_chunks={chunk_index}/{len(chunks)} "
                    f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)} "
                    f"records={len(records)}"
                )

    summary = _summarize_records(records)
    top_records = _top_records(records, top_k=int(args.top_k_records))

    result = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_residual_graduate",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "protocol": protocol_settings.protocol,
            "normalization_source": protocol_settings.normalization_source,
            "diagnostic_normalization": diagnostic_normalization,
            "benchmark_comparable": benchmark_comparable,
        },
        "args": _coerce_jsonable(vars(args)),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "predictor": {
            "subset": args.subset,
            "split": args.split,
            "sample_mode": args.sample_mode,
            "agents": agents,
            "data_norm": args.data_norm,
            "rotate": bool(args.rotate),
            "rotate_time_frame": int(args.rotate_time_frame),
            "num_to_gen": int(args.num_to_gen),
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
        },
        "checkpoints": {
            "graduate_checkpoint": Path(args.graduate_checkpoint).expanduser().as_posix(),
            "graduate_epoch": graduate_payload.get("epoch"),
            "graduate_model_config": _coerce_jsonable(graduate_payload.get("model_config", {})),
            "fast_checkpoint": args.fast_checkpoint,
            "fast_cfg_path": args.fast_cfg_path,
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "summary": summary,
        "top_records": top_records,
    }
    if args.save_records:
        result["records"] = records

    print("[diagnose_residual_graduate] completed")
    print(
        f"subset={args.subset} split={args.split} protocol={protocol_settings.protocol} "
        f"selected_scenes={len(selected_samples)} selected_eval_items={selected_eval_items}"
    )
    print(f"graduate_checkpoint={Path(args.graduate_checkpoint).expanduser().as_posix()}")
    _print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_coerce_jsonable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
