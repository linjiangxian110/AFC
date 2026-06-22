"""V54-E: sparse good-candidate structure analysis for V38-A.

This script does not train a model.  It loads a trained V38-A set generator,
builds its 80-candidate residual pool, and analyzes where the oracle-good
candidates live:

* top-k candidate base / slot histograms;
* whether V38's best candidate uses the slow teacher's best base mode;
* per-base residual-slot quality gaps;
* whether random / FPS K=20 reductions contain oracle top-k candidates;
* residual magnitude and smoothness statistics for good and selected groups.

Implementation note: V38 flattens ``[slot, base]`` as ``slot * K + base``.
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
    _candidate_score,
    _flatten_refined,
    _predictor_cfg,
    _random_global_indices,
    _random_per_base_indices,
    _set_seed,
    _structured_fps_indices,
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
    _measure_predict_latency_ms,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


SCORE_METRICS: Sequence[str] = ("fde", "ade")
STAT_NAMES: Sequence[str] = (
    "residual_norm",
    "endpoint_residual_norm",
    "skeleton_residual_norm",
    "step_norm",
    "smoothness",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sparse good residual candidates in V38-A slots4 pools.")
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
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--top-ks", type=str, default="1,3,5")
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item]
    if not values:
        raise ValueError("--top-ks must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"--top-ks values must be positive: {values}")
    return sorted(dict.fromkeys(values))


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None, "count": 0}
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
        "count": int(tensor.numel()),
    }


def _hist_payload(counts: Sequence[int]) -> Dict[str, Any]:
    total = int(sum(int(value) for value in counts))
    if total <= 0:
        freq = [0.0 for _ in counts]
    else:
        freq = [float(value) / float(total) for value in counts]
    return {"counts": [int(value) for value in counts], "freq": freq, "total": total}


def _zero_hist(size: int) -> List[int]:
    return [0 for _ in range(int(size))]


def _score_tensors(flat: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
    dist = torch.linalg.norm(flat - ground_truth[:, None, ...], dim=-1)
    return {"fde": dist[..., -1], "ade": dist.mean(dim=-1)}


def _base_score_tensors(base: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    return {"fde": dist[..., -1], "ade": dist.mean(dim=-1)}


def _contains_any(selected: torch.Tensor, top_indices: torch.Tensor, valid: torch.Tensor) -> tuple[int, int]:
    hit = (selected[:, :, None, :] == top_indices[:, None, :, :]).any(dim=1).any(dim=1)
    valid_cpu = valid.to(device=hit.device, dtype=torch.bool)
    return int(hit[valid_cpu].sum().item()), int(valid_cpu.sum().item())


def _flatten_valid(values: torch.Tensor, valid: torch.Tensor) -> List[float]:
    if values.ndim != 3:
        raise ValueError(f"Expected values [B,M,A], got {tuple(values.shape)}")
    mask = valid.to(device=values.device, dtype=torch.bool)[:, None, :].expand_as(values)
    return [float(item) for item in values[mask].detach().cpu().flatten().tolist()]


def _gather_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected values [B,N,A], got {tuple(values.shape)}")
    gather_index = indices.to(device=values.device, dtype=torch.long)
    return torch.gather(values, dim=1, index=gather_index)


def _gather_traj(flat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.to(device=flat.device, dtype=torch.long)[:, :, :, None, None].expand(
        int(indices.shape[0]),
        int(indices.shape[1]),
        int(indices.shape[2]),
        int(flat.shape[3]),
        int(flat.shape[4]),
    )
    return torch.gather(flat, dim=1, index=gather_index)


def _residual_stat_values(prediction: torch.Tensor, base_prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
    residual = prediction - base_prediction
    residual_norm = torch.linalg.norm(residual, dim=-1).mean(dim=-1)
    endpoint_residual_norm = torch.linalg.norm(residual[..., -1, :], dim=-1)
    if int(residual.shape[-2]) > 2:
        skeleton = residual[..., 1:-1, :]
    else:
        skeleton = residual
    skeleton_residual_norm = torch.linalg.norm(skeleton, dim=-1).mean(dim=-1)
    step_norm = torch.linalg.norm(prediction[..., 1:, :] - prediction[..., :-1, :], dim=-1).mean(dim=-1)
    if int(prediction.shape[-2]) > 2:
        accel = prediction[..., 2:, :] - 2.0 * prediction[..., 1:-1, :] + prediction[..., :-2, :]
        smoothness = torch.linalg.norm(accel, dim=-1).mean(dim=-1)
    else:
        smoothness = prediction.new_zeros(prediction.shape[:-2])
    return {
        "residual_norm": residual_norm,
        "endpoint_residual_norm": endpoint_residual_norm,
        "skeleton_residual_norm": skeleton_residual_norm,
        "step_norm": step_norm,
        "smoothness": smoothness,
    }


class StructureAccumulator:
    def __init__(self, *, num_modes: int, num_slots: int, top_ks: Sequence[int]) -> None:
        self.num_modes = int(num_modes)
        self.num_slots = int(num_slots)
        self.top_ks = list(top_ks)
        self.valid_agents = 0
        self.top_hist: Dict[str, Dict[int, Dict[str, List[int]]]] = {
            metric: {
                top_k: {"base": _zero_hist(num_modes), "slot": _zero_hist(num_slots)}
                for top_k in self.top_ks
            }
            for metric in SCORE_METRICS
        }
        self.best_base_match = {metric: 0 for metric in SCORE_METRICS}
        self.contains_slow_best = {
            metric: {top_k: 0 for top_k in self.top_ks}
            for metric in SCORE_METRICS
        }
        self.top1_fde_ade_same = 0
        self.slot_quality: Dict[str, Dict[str, Any]] = {
            metric: {
                "best_slot_counts": _zero_hist(num_slots),
                "expected_random_gap": [],
                "all_slot_gap": [],
                "slot_range": [],
                "random_miss_best_slot_rate": [],
            }
            for metric in SCORE_METRICS
        }
        self.selection_containment: Dict[str, Dict[str, Dict[int, Dict[str, int]]]] = {
            metric: {} for metric in SCORE_METRICS
        }
        self.group_stats: Dict[str, Dict[str, List[float]]] = {}

    def _ensure_method(self, method: str) -> None:
        for metric in SCORE_METRICS:
            if method not in self.selection_containment[metric]:
                self.selection_containment[metric][method] = {
                    top_k: {"hits": 0, "total": 0}
                    for top_k in self.top_ks
                }

    def _add_group_values(self, group: str, values: Mapping[str, torch.Tensor], valid: torch.Tensor) -> None:
        if group not in self.group_stats:
            self.group_stats[group] = {name: [] for name in STAT_NAMES}
        for name in STAT_NAMES:
            self.group_stats[group][name].extend(_flatten_valid(values[name], valid))

    def update(
        self,
        *,
        flat: torch.Tensor,
        base_flat: torch.Tensor,
        base: torch.Tensor,
        ground_truth: torch.Tensor,
        valid: torch.Tensor,
        random_trials: int,
        keep_k: int,
    ) -> None:
        valid = valid.to(device=flat.device, dtype=torch.bool)
        valid_count = int(valid.sum().item())
        self.valid_agents += valid_count
        if valid_count <= 0:
            return

        scores = _score_tensors(flat, ground_truth)
        base_scores = _base_score_tensors(base, ground_truth)
        max_top_k = max(self.top_ks)
        top_indices = {
            metric: torch.topk(score, k=max_top_k, dim=1, largest=False).indices
            for metric, score in scores.items()
        }
        best_base_slow = {
            metric: torch.argmin(score, dim=1)
            for metric, score in base_scores.items()
        }

        same = top_indices["fde"][:, 0, :] == top_indices["ade"][:, 0, :]
        self.top1_fde_ade_same += int(same[valid].sum().item())

        for metric in SCORE_METRICS:
            best_candidate = top_indices[metric][:, 0, :]
            best_base_v38 = best_candidate % self.num_modes
            self.best_base_match[metric] += int((best_base_v38 == best_base_slow[metric])[valid].sum().item())
            for top_k in self.top_ks:
                selected = top_indices[metric][:, :top_k, :]
                bases = selected % self.num_modes
                slots = selected // self.num_modes
                expanded_valid = valid[:, None, :].expand_as(selected)
                base_values = bases[expanded_valid].detach().cpu().flatten()
                slot_values = slots[expanded_valid].detach().cpu().flatten()
                for value in base_values.tolist():
                    self.top_hist[metric][top_k]["base"][int(value)] += 1
                for value in slot_values.tolist():
                    self.top_hist[metric][top_k]["slot"][int(value)] += 1
                contains = (bases == best_base_slow[metric][:, None, :]).any(dim=1)
                self.contains_slow_best[metric][top_k] += int(contains[valid].sum().item())

        for metric in SCORE_METRICS:
            by_slot = scores[metric].reshape(flat.shape[0], self.num_slots, self.num_modes, flat.shape[2])
            min_values = by_slot.min(dim=1).values
            best_slots = by_slot.argmin(dim=1)
            valid_base = valid[:, None, :].expand(flat.shape[0], self.num_modes, flat.shape[2])
            best_slot_values = best_slots[valid_base].detach().cpu().flatten()
            for value in best_slot_values.tolist():
                self.slot_quality[metric]["best_slot_counts"][int(value)] += 1
            expected_random_gap = by_slot.mean(dim=1) - min_values
            slot_range = by_slot.max(dim=1).values - min_values
            all_slot_gap = by_slot - min_values[:, None, :, :]
            tie_count = (by_slot == min_values[:, None, :, :]).sum(dim=1).to(dtype=torch.float32)
            miss_best_rate = 1.0 - tie_count / float(self.num_slots)
            self.slot_quality[metric]["expected_random_gap"].extend(
                [float(item) for item in expected_random_gap[valid_base].detach().cpu().flatten().tolist()]
            )
            self.slot_quality[metric]["slot_range"].extend(
                [float(item) for item in slot_range[valid_base].detach().cpu().flatten().tolist()]
            )
            self.slot_quality[metric]["random_miss_best_slot_rate"].extend(
                [float(item) for item in miss_best_rate[valid_base].detach().cpu().flatten().tolist()]
            )
            valid_all_slot = valid[:, None, None, :].expand_as(all_slot_gap)
            self.slot_quality[metric]["all_slot_gap"].extend(
                [float(item) for item in all_slot_gap[valid_all_slot].detach().cpu().flatten().tolist()]
            )

        endpoint_indices = _structured_fps_indices(flat[..., -1, :], keep_k=int(keep_k))
        residual_endpoint_indices = _structured_fps_indices((flat - base_flat)[..., -1, :], keep_k=int(keep_k))
        selection_methods: Dict[str, List[torch.Tensor]] = {
            "endpoint_fps20": [endpoint_indices],
            "residual_endpoint_fps20": [residual_endpoint_indices],
        }
        random_global: List[torch.Tensor] = []
        random_per_base: List[torch.Tensor] = []
        for _trial in range(int(random_trials)):
            random_global.append(
                _random_global_indices(
                    int(flat.shape[0]),
                    int(flat.shape[1]),
                    int(flat.shape[2]),
                    keep_k=int(keep_k),
                    device=flat.device,
                )
            )
            random_per_base.append(_random_per_base_indices(flat.reshape(flat.shape[0], self.num_slots, self.num_modes, flat.shape[2], flat.shape[3], flat.shape[4]), keep_k=int(keep_k)))
        selection_methods["random20_global"] = random_global
        selection_methods["random20_per_base"] = random_per_base

        for method, trials in selection_methods.items():
            self._ensure_method(method)
            for selected_indices in trials:
                for metric in SCORE_METRICS:
                    for top_k in self.top_ks:
                        hits, total = _contains_any(selected_indices, top_indices[metric][:, :top_k, :], valid)
                        self.selection_containment[metric][method][top_k]["hits"] += hits
                        self.selection_containment[metric][method][top_k]["total"] += total

        residual_values_all = _residual_stat_values(flat, base_flat)
        for top_k in self.top_ks:
            idx = top_indices["fde"][:, :top_k, :]
            values = {name: _gather_values(value, idx) for name, value in residual_values_all.items()}
            self._add_group_values(f"top{top_k}_fde", values, valid)
        idx = top_indices["ade"][:, :1, :]
        values = {name: _gather_values(value, idx) for name, value in residual_values_all.items()}
        self._add_group_values("top1_ade", values, valid)
        values = {name: _gather_values(value, endpoint_indices) for name, value in residual_values_all.items()}
        self._add_group_values("endpoint_fps20", values, valid)
        values = {name: _gather_values(value, residual_endpoint_indices) for name, value in residual_values_all.items()}
        self._add_group_values("residual_endpoint_fps20", values, valid)
        if random_global:
            rg = random_global[0]
            values = {name: _gather_values(value, rg) for name, value in residual_values_all.items()}
            self._add_group_values("random20_global_trial0", values, valid)
        if random_per_base:
            rb = random_per_base[0]
            values = {name: _gather_values(value, rb) for name, value in residual_values_all.items()}
            self._add_group_values("random20_per_base_trial0", values, valid)

    def finalize(self) -> Dict[str, Any]:
        top_hist_payload: Dict[str, Any] = {}
        for metric in SCORE_METRICS:
            top_hist_payload[metric] = {}
            for top_k in self.top_ks:
                top_hist_payload[metric][f"top{top_k}"] = {
                    "base": _hist_payload(self.top_hist[metric][top_k]["base"]),
                    "slot": _hist_payload(self.top_hist[metric][top_k]["slot"]),
                }

        base_relation: Dict[str, Any] = {}
        total = max(int(self.valid_agents), 1)
        for metric in SCORE_METRICS:
            base_relation[metric] = {
                "best_base_v38_equals_slow_best_base_rate": float(self.best_base_match[metric] / total),
                "topk_contains_slow_best_base_rate": {
                    f"top{top_k}": float(self.contains_slow_best[metric][top_k] / total)
                    for top_k in self.top_ks
                },
            }
        base_relation["top1_fde_equals_top1_ade_candidate_rate"] = float(self.top1_fde_ade_same / total)

        slot_quality: Dict[str, Any] = {}
        for metric in SCORE_METRICS:
            slot_quality[metric] = {
                "best_slot": _hist_payload(self.slot_quality[metric]["best_slot_counts"]),
                "expected_random_gap": _quantiles(self.slot_quality[metric]["expected_random_gap"]),
                "all_slot_gap": _quantiles(self.slot_quality[metric]["all_slot_gap"]),
                "slot_range": _quantiles(self.slot_quality[metric]["slot_range"]),
                "random_miss_best_slot_rate": _quantiles(self.slot_quality[metric]["random_miss_best_slot_rate"]),
            }

        containment: Dict[str, Any] = {}
        for metric in SCORE_METRICS:
            containment[metric] = {}
            for method, rows in self.selection_containment[metric].items():
                containment[metric][method] = {}
                for top_k, counts in rows.items():
                    total_count = max(int(counts["total"]), 1)
                    containment[metric][method][f"top{top_k}"] = {
                        "contains_any_rate": float(counts["hits"] / total_count),
                        "hits": int(counts["hits"]),
                        "total": int(counts["total"]),
                    }

        residual_stats = {
            group: {name: _quantiles(values) for name, values in stats.items()}
            for group, stats in self.group_stats.items()
        }
        return {
            "valid_agents": int(self.valid_agents),
            "candidate_index_layout": "flat_index = slot_id * num_base_modes + base_id",
            "top_histograms": top_hist_payload,
            "base_relation": base_relation,
            "slot_quality": slot_quality,
            "selection_containment": containment,
            "residual_stats": residual_stats,
        }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def _render_summary(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("===== V54-E GOOD CANDIDATE STRUCTURE =====")
    lines.append(f"valid_agents: {summary.get('valid_agents')}")
    lines.append(f"candidate_index_layout: {summary.get('candidate_index_layout')}")
    lines.append("")

    lines.append("===== TOP-K SLOT HISTOGRAMS =====")
    for metric in SCORE_METRICS:
        lines.append(f"-- {metric.upper()} --")
        for key, row in summary["top_histograms"][metric].items():
            slot_freq = row["slot"]["freq"]
            lines.append(f"{key} slot_freq: " + ", ".join(f"s{idx}={_fmt(value, 4)}" for idx, value in enumerate(slot_freq)))
        lines.append("")

    lines.append("===== BASE RELATION =====")
    for metric in SCORE_METRICS:
        row = summary["base_relation"][metric]
        lines.append(
            f"{metric}: best_base_match={_fmt(row['best_base_v38_equals_slow_best_base_rate'])}"
        )
        contains = row["topk_contains_slow_best_base_rate"]
        lines.append(
            f"{metric}: " + ", ".join(f"{key}_contains_slow_best={_fmt(value)}" for key, value in contains.items())
        )
    lines.append(
        "top1_fde_equals_top1_ade_candidate_rate: "
        f"{_fmt(summary['base_relation']['top1_fde_equals_top1_ade_candidate_rate'])}"
    )
    lines.append("")

    lines.append("===== SLOT QUALITY GAP =====")
    for metric in SCORE_METRICS:
        row = summary["slot_quality"][metric]
        slot_freq = row["best_slot"]["freq"]
        lines.append(f"-- {metric.upper()} --")
        lines.append("best_slot_freq: " + ", ".join(f"s{idx}={_fmt(value, 4)}" for idx, value in enumerate(slot_freq)))
        for name in ("expected_random_gap", "slot_range", "random_miss_best_slot_rate"):
            q = row[name]
            lines.append(
                f"{name}: mean={_fmt(q['mean'])} median={_fmt(q['median'])} "
                f"p90={_fmt(q['p90'])} p95={_fmt(q['p95'])}"
            )
        lines.append("")

    lines.append("===== SELECTION CONTAINMENT FDE =====")
    for method, rows in summary["selection_containment"]["fde"].items():
        pieces = []
        for key, row in rows.items():
            pieces.append(f"{key}={_fmt(row['contains_any_rate'])}")
        lines.append(f"{method}: " + ", ".join(pieces))
    lines.append("")

    lines.append("===== RESIDUAL STATS =====")
    for group, stats in summary["residual_stats"].items():
        lines.append(f"-- {group} --")
        for name in ("residual_norm", "endpoint_residual_norm", "smoothness"):
            row = stats[name]
            lines.append(
                f"{name}: mean={_fmt(row['mean'])} median={_fmt(row['median'])} "
                f"p90={_fmt(row['p90'])} p95={_fmt(row['p95'])}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.random_trials) < 0:
        raise SystemExit("--random-trials must be non-negative")
    top_ks = _split_ints(args.top_ks)
    if max(top_ks) > int(args.keep_k):
        raise SystemExit("--top-ks must not exceed --keep-k")

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

    accumulator: Optional[StructureAccumulator] = None
    print(
        "[diagnose_v38_good_candidate_structure] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} variant={refiner_variant} slots={args.residual_slots} "
        f"keep_k={args.keep_k} top_ks={top_ks} random_trials={args.random_trials}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v38_good_candidate_structure] warning: selected_samples normalization is diagnostic only")

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        _slow_latencies, slow_output = _measure_predict_latency_ms(
            lambda: slow_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
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

        _refiner_latencies, refiner_outputs = _measure_predict_latency_ms(
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
        valid = batch["agent_mask"].to(device=device).bool()
        if accumulator is None:
            accumulator = StructureAccumulator(
                num_modes=int(slow_output.slow_pred.shape[1]),
                num_slots=int(args.residual_slots),
                top_ks=top_ks,
            )
        accumulator.update(
            flat=flat,
            base_flat=base_flat,
            base=slow_output.slow_pred,
            ground_truth=ground_truth,
            valid=valid,
            random_trials=int(args.random_trials),
            keep_k=int(args.keep_k),
        )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_v38_good_candidate_structure] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    if accumulator is None:
        raise RuntimeError("No samples were processed")
    summary = accumulator.finalize()
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v38_good_candidate_structure",
            "variant": "v54e_sparse_good_residual_candidate_structure_analysis",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "top_ks": top_ks,
            "random_trials": int(args.random_trials),
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
        "summary": _coerce_jsonable(summary),
    }
    rendered = _render_summary(summary)
    print(rendered)
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_json.as_posix()}")
    if args.output_txt:
        output_txt = Path(args.output_txt).expanduser().resolve()
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(rendered, encoding="utf-8")
        print(f"output_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
