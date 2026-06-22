"""Diagnose when V58 front slots contain useful compressed residuals.

This diagnostic is intentionally per-base/per-agent.  For every valid
``(scene, base mode, agent)`` item, it asks whether the forced front slots
(``slot1/slot2`` by default) contain at least one correction that improves the
slot0/base trajectory.  It then compares observable features between improved,
neutral, and hurt cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_group_selector, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    _predictor_cfg,
    _set_seed,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import _checkpoint_variant, _local_temporal_energy
from trustmoe_traj.scripts.eval_v58c_fair20_residual_slots import _candidate_score_slots
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPS = 1.0e-8
OBSERVABLE_DISTRIBUTION_FEATURES = (
    "base_endpoint_norm,base_path_length,"
    "past_displacement,past_path_length,past_mean_step,past_max_step,past_straightness,past_turn_intensity,"
    "energy_risk_mean,energy_risk_max,min_neighbor_distance_mean,close_neighbor_count_mean,"
    "approaching_score_mean,endpoint_crowding_energy_mean,"
    "front_slot_endpoint_gap,front_slot_trajectory_gap,"
    "slot1_residual_endpoint_norm,slot1_residual_trajectory_norm,slot1_residual_forward,slot1_residual_abs_lateral,"
    "slot2_residual_endpoint_norm,slot2_residual_trajectory_norm,slot2_residual_forward,slot2_residual_abs_lateral,"
    "selector_local_slot,selector_selected_prob,selector_logit_margin,selector_entropy,"
    "selected_residual_endpoint_norm,selected_residual_trajectory_norm,selected_residual_forward,selected_residual_abs_lateral"
)
DEFAULT_BINNING_FEATURES = (
    "base_path_length,base_endpoint_norm,past_path_length,past_max_step,past_turn_intensity,"
    "energy_risk_mean,min_neighbor_distance_mean,front_slot_endpoint_gap,"
    "slot1_residual_endpoint_norm,slot2_residual_endpoint_norm,"
    "selector_logit_margin,selector_entropy"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V58 front-slot compression success/failure cases.")
    parser.add_argument("--summarize-only", action="store_true", help="Aggregate saved diagnostic JSON files.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, default=None)
    parser.add_argument("--diag-file-prefix", type=str, default="v58_front2_case_diag")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)

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

    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--refiner-checkpoint", type=str, default=None)
    parser.add_argument("--selector-checkpoint", type=str, default=None)
    parser.add_argument(
        "--selector-confidence-fallback-to-slot0",
        action="store_true",
        help="Use slot0 unless the selector prefers slot1/2 with sufficient confidence.",
    )
    parser.add_argument("--selector-fallback-prob-margin", type=float, default=0.05)
    parser.add_argument("--selector-fallback-min-selected-prob", type=float, default=0.35)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--front-slot-start", type=int, default=1)
    parser.add_argument("--front-slots", type=int, default=2)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--success-margin", type=float, default=0.0)
    parser.add_argument("--strong-gain-thresholds", type=str, default="0.02,0.05,0.10")
    parser.add_argument(
        "--focus-base-topks",
        type=str,
        default="1,3,5",
        help="Base-rank top-k groups to diagnose separately; ranks are by the selected oracle metric.",
    )
    parser.add_argument(
        "--separability-features",
        type=str,
        default=(
            "base_fde,base_ade,base_path_length,base_endpoint_norm,"
            "past_displacement,past_path_length,past_mean_step,past_max_step,past_straightness,past_turn_intensity,"
            "energy_risk_mean,energy_risk_max,min_neighbor_distance_mean,close_neighbor_count_mean,"
            "approaching_score_mean,endpoint_crowding_energy_mean,"
            "best_front_fde_gain,best_front_ade_gain,best_front_residual_endpoint_norm,"
            "best_front_residual_forward,best_front_residual_abs_lateral,front_slot_endpoint_gap,"
            "selected_front_fde_gain,selected_score_minus_front_oracle,selected_residual_endpoint_norm,"
            "selected_residual_forward,selected_residual_abs_lateral,"
            "selector_selected_prob,selector_logit_margin,selector_entropy"
        ),
        help="Comma-separated scalar features used for improve-vs-hurt separability diagnostics.",
    )
    parser.add_argument("--top-separability-features", type=int, default=12)
    parser.add_argument(
        "--binning-features",
        type=str,
        default=DEFAULT_BINNING_FEATURES,
        help="Observable scalar features for quantile-bin distribution diagnostics.",
    )
    parser.add_argument("--distribution-bins", type=int, default=8)
    parser.add_argument(
        "--window-stat-every-chunks",
        type=int,
        default=0,
        help="If positive, flush rolling mean/median diagnostics every N eval chunks.",
    )
    parser.add_argument(
        "--window-stat-features",
        type=str,
        default=DEFAULT_BINNING_FEATURES,
        help="Features tracked by rolling mean/median diagnostics.",
    )
    parser.add_argument(
        "--embedding-diagnostic",
        action="store_true",
        help="Train a small diagnostic MLP on observable features to test high-dimensional separability.",
    )
    parser.add_argument(
        "--embedding-features",
        type=str,
        default=OBSERVABLE_DISTRIBUTION_FEATURES,
        help="Observable features used by the diagnostic MLP. Features missing from a comparison are skipped.",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--embedding-epochs", type=int, default=160)
    parser.add_argument("--embedding-max-per-class", type=int, default=4000)
    parser.add_argument("--embedding-min-per-class", type=int, default=40)
    parser.add_argument("--embedding-train-fraction", type=float, default=0.7)
    parser.add_argument("--embedding-lr", type=float, default=1.0e-3)
    parser.add_argument("--embedding-weight-decay", type=float, default=1.0e-4)

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


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _require_eval_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("slow_cfg_path", "slow_checkpoint", "refiner_checkpoint", "output_json")
        if not getattr(args, name)
    ]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(f"Missing required diagnostic arguments: {joined}")
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.front_slots) <= 0:
        raise SystemExit("--front-slots must be positive")
    if int(args.front_slot_start) < 0:
        raise SystemExit("--front-slot-start must be non-negative")
    if int(args.front_slot_start) + int(args.front_slots) > int(args.residual_slots):
        raise SystemExit("--front-slot-start + --front-slots must not exceed --residual-slots")
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if float(args.selector_fallback_prob_margin) < 0.0:
        raise SystemExit("--selector-fallback-prob-margin must be non-negative")
    if not (0.0 <= float(args.selector_fallback_min_selected_prob) <= 1.0):
        raise SystemExit("--selector-fallback-min-selected-prob must be in [0, 1]")
    if bool(args.selector_confidence_fallback_to_slot0) and int(args.front_slot_start) != 0:
        raise SystemExit("--selector-confidence-fallback-to-slot0 requires --front-slot-start 0")


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.to(device=values.device, dtype=torch.bool)
    return values[keep].detach().to(dtype=torch.float32)


def _selector_indices_with_optional_slot0_fallback(
    logits: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    selected = logits.argmax(dim=1).to(dtype=torch.long)
    if not bool(getattr(args, "selector_confidence_fallback_to_slot0", False)):
        return selected
    probs = torch.softmax(logits, dim=1)
    selected_prob = torch.gather(probs, dim=1, index=selected[:, None, :, :]).squeeze(1)
    slot0_prob = probs[:, 0]
    accept = (
        (selected != 0)
        & (selected_prob >= slot0_prob + float(args.selector_fallback_prob_margin))
        & (selected_prob >= float(args.selector_fallback_min_selected_prob))
    )
    return torch.where(accept, selected, torch.zeros_like(selected)).to(dtype=torch.long)


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    if int(values.numel()) <= 0:
        return {}
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=values.device, dtype=values.dtype)
    out = torch.quantile(values, qs)
    names = ("p10", "p25", "p50", "p75", "p90")
    return {name: float(item.detach().cpu()) for name, item in zip(names, out)}


class GroupStats:
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.count = 0
        self.sums: Dict[str, float] = {}
        self.sq_sums: Dict[str, float] = {}
        self.quantile_values: Dict[str, List[torch.Tensor]] = {}

    def add(self, features: Mapping[str, torch.Tensor], mask: torch.Tensor) -> None:
        keep = mask.bool()
        count = int(keep.sum().item())
        if count <= 0:
            return
        self.count += count
        for key, value in features.items():
            selected = _masked_values(value, keep)
            if int(selected.numel()) <= 0:
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(selected.sum().cpu())
            self.sq_sums[key] = self.sq_sums.get(key, 0.0) + float((selected * selected).sum().cpu())
            if key in {
                "base_fde",
                "base_ade",
                "best_front_fde_gain",
                "best_front_ade_gain",
                "best_front_metric_gain",
                "base_rank_fde",
                "past_displacement",
                "past_path_length",
                "past_mean_step",
                "past_turn_intensity",
                "best_front_residual_endpoint_norm",
                "selected_front_fde_gain",
                "selected_score_minus_front_oracle",
                "selected_residual_endpoint_norm",
                "front_slot_endpoint_gap",
                "energy_risk_mean",
                "selector_logit_margin",
                "selector_selected_prob",
                "selector_entropy",
            }:
                self.quantile_values.setdefault(key, []).append(selected.detach().cpu())

    def finalize(self, total_valid: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "count": int(self.count),
            "ratio": float(self.count / max(int(total_valid), 1)),
        }
        count = max(int(self.count), 1)
        for key, value in sorted(self.sums.items()):
            mean = float(value / count)
            sq_mean = float(self.sq_sums.get(key, 0.0) / count)
            std = max(sq_mean - mean * mean, 0.0) ** 0.5
            result[f"mean_{key}"] = mean
            result[f"std_{key}"] = std
        quantiles: Dict[str, Dict[str, float]] = {}
        for key, chunks in sorted(self.quantile_values.items()):
            if not chunks:
                continue
            values = torch.cat(chunks, dim=0)
            quantiles[key] = _quantiles(values)
        if quantiles:
            result["quantiles"] = quantiles
        return result


def _binary_auc(pos: torch.Tensor, neg: torch.Tensor) -> Optional[float]:
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    if n_pos <= 0 or n_neg <= 0:
        return None
    values = torch.cat([pos.detach().cpu().to(dtype=torch.float64), neg.detach().cpu().to(dtype=torch.float64)])
    labels = torch.cat(
        [
            torch.ones(n_pos, dtype=torch.bool),
            torch.zeros(n_neg, dtype=torch.bool),
        ]
    )
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_labels = labels[order]
    ranks = torch.empty(values.shape[0], dtype=torch.float64)
    start = 0
    total = int(values.shape[0])
    while start < total:
        end = start + 1
        while end < total and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        avg_rank = float(start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end
    rank_sum_pos = ranks[sorted_labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _ks_distance(pos: torch.Tensor, neg: torch.Tensor) -> Optional[float]:
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    if n_pos <= 0 or n_neg <= 0:
        return None
    pos_sorted = torch.sort(pos.detach().cpu().to(dtype=torch.float64)).values
    neg_sorted = torch.sort(neg.detach().cpu().to(dtype=torch.float64)).values
    values = torch.unique(torch.cat([pos_sorted, neg_sorted]))
    pos_cdf = torch.searchsorted(pos_sorted, values, right=True).to(dtype=torch.float64) / float(n_pos)
    neg_cdf = torch.searchsorted(neg_sorted, values, right=True).to(dtype=torch.float64) / float(n_neg)
    return float((pos_cdf - neg_cdf).abs().max())


def _separation_label(best_auc: Optional[float], ks: Optional[float]) -> str:
    auc = 0.5 if best_auc is None else float(best_auc)
    ks_v = 0.0 if ks is None else float(ks)
    if auc >= 0.75 or ks_v >= 0.50:
        return "strong"
    if auc >= 0.65 or ks_v >= 0.35:
        return "moderate"
    if auc >= 0.58 or ks_v >= 0.20:
        return "weak"
    return "none"


class SeparabilityStats:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = [str(item) for item in feature_names if str(item)]
        self._values: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}

    def add(
        self,
        comparison: str,
        features: Mapping[str, torch.Tensor],
        *,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> None:
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        if int(pos_mask.sum().item()) <= 0 or int(neg_mask.sum().item()) <= 0:
            return
        comparison_values = self._values.setdefault(str(comparison), {})
        for key in self.feature_names:
            value = features.get(key)
            if value is None:
                continue
            pos = _masked_values(value, pos_mask)
            neg = _masked_values(value, neg_mask)
            if int(pos.numel()) <= 0 or int(neg.numel()) <= 0:
                continue
            store = comparison_values.setdefault(key, {"pos": [], "neg": []})
            store["pos"].append(pos.detach().cpu())
            store["neg"].append(neg.detach().cpu())

    def finalize(self, *, top_n: int) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for comparison, feature_map in sorted(self._values.items()):
            features: Dict[str, Any] = {}
            ranked: List[Dict[str, Any]] = []
            for key, parts in sorted(feature_map.items()):
                if not parts.get("pos") or not parts.get("neg"):
                    continue
                pos = torch.cat(parts["pos"], dim=0).to(dtype=torch.float64)
                neg = torch.cat(parts["neg"], dim=0).to(dtype=torch.float64)
                n_pos = int(pos.numel())
                n_neg = int(neg.numel())
                if n_pos <= 0 or n_neg <= 0:
                    continue
                mean_pos = float(pos.mean())
                mean_neg = float(neg.mean())
                std_pos = float(pos.std(unbiased=False)) if n_pos > 1 else 0.0
                std_neg = float(neg.std(unbiased=False)) if n_neg > 1 else 0.0
                pooled = (((std_pos * std_pos) + (std_neg * std_neg)) / 2.0) ** 0.5
                smd = 0.0 if pooled <= EPS else float((mean_pos - mean_neg) / pooled)
                auc = _binary_auc(pos, neg)
                best_auc = None if auc is None else max(float(auc), 1.0 - float(auc))
                ks = _ks_distance(pos, neg)
                direction = None
                if auc is not None:
                    direction = "higher_in_positive" if float(auc) >= 0.5 else "higher_in_negative"
                item = {
                    "feature": key,
                    "positive_count": n_pos,
                    "negative_count": n_neg,
                    "positive_mean": mean_pos,
                    "negative_mean": mean_neg,
                    "positive_std": std_pos,
                    "negative_std": std_neg,
                    "smd": smd,
                    "auc_positive_higher": auc,
                    "best_auc": best_auc,
                    "ks": ks,
                    "direction": direction,
                    "separation": _separation_label(best_auc, ks),
                    "positive_quantiles": _quantiles(pos.to(dtype=torch.float32)),
                    "negative_quantiles": _quantiles(neg.to(dtype=torch.float32)),
                }
                features[key] = item
                ranked.append(item)
            ranked.sort(
                key=lambda row: (
                    float(row.get("best_auc") or 0.5),
                    float(row.get("ks") or 0.0),
                    abs(float(row.get("smd") or 0.0)),
                ),
                reverse=True,
            )
            output[comparison] = {
                "features": features,
                "ranked_features": [item["feature"] for item in ranked],
                "top_features": ranked[: max(int(top_n), 0)],
            }
        return output


class BinnedDistributionStats:
    def __init__(self, feature_names: Sequence[str], *, bins: int) -> None:
        self.feature_names = [str(item) for item in feature_names if str(item)]
        self.bins = max(int(bins), 1)
        self._values: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}

    def add(
        self,
        comparison: str,
        features: Mapping[str, torch.Tensor],
        *,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> None:
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        if int(pos_mask.sum().item()) <= 0 or int(neg_mask.sum().item()) <= 0:
            return
        comparison_values = self._values.setdefault(str(comparison), {})
        for key in self.feature_names:
            value = features.get(key)
            if value is None:
                continue
            pos = _masked_values(value, pos_mask)
            neg = _masked_values(value, neg_mask)
            if int(pos.numel()) <= 0 or int(neg.numel()) <= 0:
                continue
            store = comparison_values.setdefault(key, {"pos": [], "neg": []})
            store["pos"].append(pos.detach().cpu())
            store["neg"].append(neg.detach().cpu())

    def finalize(self, *, top_n: int) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for comparison, feature_map in sorted(self._values.items()):
            features: Dict[str, Any] = {}
            ranked: List[Dict[str, Any]] = []
            for key, parts in sorted(feature_map.items()):
                if not parts.get("pos") or not parts.get("neg"):
                    continue
                pos = torch.cat(parts["pos"], dim=0).to(dtype=torch.float64)
                neg = torch.cat(parts["neg"], dim=0).to(dtype=torch.float64)
                pos = pos[torch.isfinite(pos)]
                neg = neg[torch.isfinite(neg)]
                if int(pos.numel()) <= 0 or int(neg.numel()) <= 0:
                    continue
                values = torch.cat([pos, neg], dim=0)
                labels = torch.cat(
                    [
                        torch.ones(int(pos.numel()), dtype=torch.bool),
                        torch.zeros(int(neg.numel()), dtype=torch.bool),
                    ],
                    dim=0,
                )
                if int(torch.unique(values).numel()) <= 1:
                    edges = torch.stack([values.min(), values.max()])
                else:
                    qs = torch.linspace(0.0, 1.0, steps=self.bins + 1, dtype=values.dtype)
                    edges = torch.unique(torch.quantile(values, qs))
                    if int(edges.numel()) < 2:
                        edges = torch.stack([values.min(), values.max()])
                bin_rows: List[Dict[str, Any]] = []
                rates: List[float] = []
                for bin_index in range(max(int(edges.numel()) - 1, 1)):
                    if int(edges.numel()) <= 1:
                        low = high = values[0]
                        bin_mask = torch.ones_like(labels, dtype=torch.bool)
                    else:
                        low = edges[bin_index]
                        high = edges[bin_index + 1]
                        if bin_index == int(edges.numel()) - 2:
                            bin_mask = (values >= low) & (values <= high)
                        else:
                            bin_mask = (values >= low) & (values < high)
                    count = int(bin_mask.sum().item())
                    pos_count = int((labels & bin_mask).sum().item())
                    neg_count = int(((~labels) & bin_mask).sum().item())
                    rate = None if count <= 0 else float(pos_count / count)
                    if rate is not None and count >= 10:
                        rates.append(rate)
                    bin_rows.append(
                        {
                            "bin": int(bin_index),
                            "low": float(low.detach().cpu()),
                            "high": float(high.detach().cpu()),
                            "count": count,
                            "positive_count": pos_count,
                            "negative_count": neg_count,
                            "positive_rate": rate,
                        }
                    )
                rate_range = None if not rates else float(max(rates) - min(rates))
                trend = None if len(rates) < 2 else float(rates[-1] - rates[0])
                item = {
                    "feature": key,
                    "positive_count": int(pos.numel()),
                    "negative_count": int(neg.numel()),
                    "positive_mean": float(pos.mean()),
                    "negative_mean": float(neg.mean()),
                    "positive_quantiles": _quantiles(pos.to(dtype=torch.float32)),
                    "negative_quantiles": _quantiles(neg.to(dtype=torch.float32)),
                    "bins": bin_rows,
                    "usable_bins": len(rates),
                    "positive_rate_range": rate_range,
                    "positive_rate_trend": trend,
                }
                features[key] = item
                ranked.append(item)
            ranked.sort(
                key=lambda row: (
                    float(row.get("positive_rate_range") or 0.0),
                    abs(float(row.get("positive_rate_trend") or 0.0)),
                ),
                reverse=True,
            )
            output[comparison] = {
                "features": features,
                "ranked_features": [item["feature"] for item in ranked],
                "top_features": ranked[: max(int(top_n), 0)],
            }
        return output


class RollingWindowStats:
    def __init__(self, feature_names: Sequence[str], *, every_chunks: int) -> None:
        self.feature_names = [str(item) for item in feature_names if str(item)]
        self.every_chunks = int(every_chunks)
        self.enabled = self.every_chunks > 0
        self._start_chunk = 1
        self._current: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}
        self._windows: List[Dict[str, Any]] = []

    def add(
        self,
        comparison: str,
        features: Mapping[str, torch.Tensor],
        *,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        if int(pos_mask.sum().item()) <= 0 and int(neg_mask.sum().item()) <= 0:
            return
        comparison_values = self._current.setdefault(str(comparison), {})
        for key in self.feature_names:
            value = features.get(key)
            if value is None:
                continue
            store = comparison_values.setdefault(key, {"positive": [], "negative": []})
            if int(pos_mask.sum().item()) > 0:
                pos = _masked_values(value, pos_mask)
                if int(pos.numel()) > 0:
                    store["positive"].append(pos.detach().cpu())
            if int(neg_mask.sum().item()) > 0:
                neg = _masked_values(value, neg_mask)
                if int(neg.numel()) > 0:
                    store["negative"].append(neg.detach().cpu())

    def maybe_flush(self, chunk_index: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and int(chunk_index) % self.every_chunks != 0:
            return
        if not self._current:
            self._start_chunk = int(chunk_index) + 1
            return
        window: Dict[str, Any] = {
            "start_chunk": int(self._start_chunk),
            "end_chunk": int(chunk_index),
            "comparisons": {},
        }
        for comparison, feature_map in sorted(self._current.items()):
            comparison_out: Dict[str, Any] = {}
            for key, labels in sorted(feature_map.items()):
                feature_out: Dict[str, Any] = {}
                for label, chunks in labels.items():
                    if not chunks:
                        continue
                    values = torch.cat(chunks, dim=0).to(dtype=torch.float32)
                    values = values[torch.isfinite(values)]
                    if int(values.numel()) <= 0:
                        continue
                    feature_out[label] = {
                        "count": int(values.numel()),
                        "mean": float(values.mean()),
                        "std": float(values.std(unbiased=False)) if int(values.numel()) > 1 else 0.0,
                        "median": float(values.median()),
                    }
                if feature_out:
                    comparison_out[key] = feature_out
            if comparison_out:
                window["comparisons"][comparison] = comparison_out
        if window["comparisons"]:
            self._windows.append(window)
        self._current = {}
        self._start_chunk = int(chunk_index) + 1

    def finalize(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for window in self._windows:
            for comparison, feature_map in window.get("comparisons", {}).items():
                comparison_out = summary.setdefault(comparison, {})
                for key, labels in feature_map.items():
                    feature_out = comparison_out.setdefault(key, {})
                    for label, stats in labels.items():
                        label_out = feature_out.setdefault(
                            label,
                            {"means": [], "medians": [], "counts": []},
                        )
                        label_out["means"].append(float(stats.get("mean")))
                        label_out["medians"].append(float(stats.get("median")))
                        label_out["counts"].append(float(stats.get("count")))
        compact: Dict[str, Any] = {}
        for comparison, feature_map in sorted(summary.items()):
            compact_features: Dict[str, Any] = {}
            for key, labels in sorted(feature_map.items()):
                compact_labels: Dict[str, Any] = {}
                for label, stats in sorted(labels.items()):
                    means = stats.get("means", [])
                    medians = stats.get("medians", [])
                    counts = stats.get("counts", [])
                    if not means or not medians or not counts:
                        continue
                    compact_labels[label] = {
                        "windows": len(means),
                        "mean_min": min(means),
                        "mean_max": max(means),
                        "mean_range": max(means) - min(means),
                        "median_min": min(medians),
                        "median_max": max(medians),
                        "median_range": max(medians) - min(medians),
                        "count_min": min(counts),
                        "count_max": max(counts),
                    }
                if compact_labels:
                    compact_features[key] = compact_labels
            if compact_features:
                compact[comparison] = compact_features
        return {
            "every_chunks": int(self.every_chunks),
            "windows": self._windows,
            "summary": compact,
        }


class _DiagnosticMLP(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(int(embedding_dim), 2)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class EmbeddingDiagnosticStats:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = [str(item) for item in feature_names if str(item)]
        self._data: Dict[str, Dict[str, Any]] = {}

    def _comparison_features(self, comparison: str, features: Mapping[str, torch.Tensor]) -> List[str]:
        store = self._data.setdefault(str(comparison), {"feature_names": None, "pos": [], "neg": []})
        if store["feature_names"] is None:
            store["feature_names"] = [name for name in self.feature_names if name in features]
        return list(store["feature_names"])

    @staticmethod
    def _matrix(features: Mapping[str, torch.Tensor], feature_names: Sequence[str], mask: torch.Tensor) -> torch.Tensor:
        cols = []
        for key in feature_names:
            value = features.get(key)
            if value is None:
                return torch.empty(0, len(feature_names), dtype=torch.float32)
            cols.append(_masked_values(value, mask.bool()).detach().cpu())
        if not cols:
            return torch.empty(0, 0, dtype=torch.float32)
        matrix = torch.stack(cols, dim=1).to(dtype=torch.float32)
        finite = torch.isfinite(matrix).all(dim=1)
        return matrix[finite]

    def add(
        self,
        comparison: str,
        features: Mapping[str, torch.Tensor],
        *,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> None:
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        if int(pos_mask.sum().item()) <= 0 or int(neg_mask.sum().item()) <= 0:
            return
        feature_names = self._comparison_features(str(comparison), features)
        if len(feature_names) < 2:
            return
        pos = self._matrix(features, feature_names, pos_mask)
        neg = self._matrix(features, feature_names, neg_mask)
        if int(pos.shape[0]) <= 0 or int(neg.shape[0]) <= 0:
            return
        store = self._data[str(comparison)]
        store["pos"].append(pos)
        store["neg"].append(neg)

    @staticmethod
    def _perm(n: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randperm(int(n), generator=generator)

    @staticmethod
    def _stable_seed(base_seed: int, name: str) -> int:
        digest = hashlib.sha1(f"{int(base_seed)}:{name}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _train_one(
        self,
        comparison: str,
        store: Mapping[str, Any],
        *,
        seed: int,
        embedding_dim: int,
        epochs: int,
        max_per_class: int,
        min_per_class: int,
        train_fraction: float,
        lr: float,
        weight_decay: float,
    ) -> Dict[str, Any]:
        feature_names = list(store.get("feature_names") or [])
        if len(feature_names) < 2 or not store.get("pos") or not store.get("neg"):
            return {"status": "insufficient", "feature_names": feature_names}
        pos = torch.cat(store["pos"], dim=0).to(dtype=torch.float32)
        neg = torch.cat(store["neg"], dim=0).to(dtype=torch.float32)
        if int(pos.shape[0]) < int(min_per_class) or int(neg.shape[0]) < int(min_per_class):
            return {
                "status": "insufficient",
                "feature_names": feature_names,
                "positive_count": int(pos.shape[0]),
                "negative_count": int(neg.shape[0]),
            }
        local_seed = self._stable_seed(seed, comparison)
        cap = min(int(max_per_class), int(pos.shape[0]), int(neg.shape[0]))
        pos = pos[self._perm(int(pos.shape[0]), local_seed + 1)[:cap]]
        neg = neg[self._perm(int(neg.shape[0]), local_seed + 2)[:cap]]
        n_train = int(round(float(train_fraction) * cap))
        n_train = min(max(n_train, 1), cap - 1)
        pos_order = self._perm(cap, local_seed + 3)
        neg_order = self._perm(cap, local_seed + 4)
        pos_train = pos[pos_order[:n_train]]
        neg_train = neg[neg_order[:n_train]]
        pos_test = pos[pos_order[n_train:]]
        neg_test = neg[neg_order[n_train:]]
        if int(pos_test.shape[0]) <= 0 or int(neg_test.shape[0]) <= 0:
            return {"status": "insufficient_test", "feature_names": feature_names}
        train_x = torch.cat([pos_train, neg_train], dim=0)
        train_y = torch.cat(
            [
                torch.ones(int(pos_train.shape[0]), dtype=torch.float32),
                torch.zeros(int(neg_train.shape[0]), dtype=torch.float32),
            ],
            dim=0,
        )
        test_x = torch.cat([pos_test, neg_test], dim=0)
        test_y = torch.cat(
            [
                torch.ones(int(pos_test.shape[0]), dtype=torch.float32),
                torch.zeros(int(neg_test.shape[0]), dtype=torch.float32),
            ],
            dim=0,
        )
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std
        order = self._perm(int(train_x.shape[0]), local_seed + 5)
        train_x = train_x[order]
        train_y = train_y[order]
        torch.manual_seed(local_seed)
        model = _DiagnosticMLP(int(train_x.shape[1]), int(embedding_dim))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        final_loss = None
        for _epoch in range(max(int(epochs), 0)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        with torch.no_grad():
            train_logits = model(train_x)
            test_logits = model(test_x)
            test_pred = test_logits >= 0.0
            positive = test_y.bool()
            negative = ~positive
            tp_rate = float((test_pred[positive] == positive[positive]).to(dtype=torch.float32).mean())
            tn_rate = float((test_pred[negative] == positive[negative]).to(dtype=torch.float32).mean())
            accuracy = float((test_pred == positive).to(dtype=torch.float32).mean())
            balanced_accuracy = float((tp_rate + tn_rate) / 2.0)
            auc = _binary_auc(test_logits[positive].detach().cpu(), test_logits[negative].detach().cpu())
            embedding = model.encoder(test_x)
            pos_embedding = embedding[positive]
            neg_embedding = embedding[negative]
            pos_center = pos_embedding.mean(dim=0)
            neg_center = neg_embedding.mean(dim=0)
            centroid_distance = float(torch.linalg.norm(pos_center - neg_center).detach().cpu())
            pos_within = torch.linalg.norm(pos_embedding - pos_center, dim=1).pow(2).mean()
            neg_within = torch.linalg.norm(neg_embedding - neg_center, dim=1).pow(2).mean()
            within_rms = float(torch.sqrt((pos_within + neg_within) / 2.0).detach().cpu())
            fisher_ratio = float(centroid_distance / max(within_rms, EPS))
        return {
            "status": "ok",
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "embedding_dim": int(embedding_dim),
            "positive_count": int(pos.shape[0]),
            "negative_count": int(neg.shape[0]),
            "train_count": int(train_x.shape[0]),
            "test_count": int(test_x.shape[0]),
            "train_logit_mean": float(train_logits.mean().detach().cpu()),
            "test_logit_mean": float(test_logits.mean().detach().cpu()),
            "final_train_loss": final_loss,
            "test_accuracy": accuracy,
            "test_balanced_accuracy": balanced_accuracy,
            "test_auc_positive_higher": auc,
            "test_best_auc": None if auc is None else max(float(auc), 1.0 - float(auc)),
            "embedding_centroid_distance": centroid_distance,
            "embedding_within_rms": within_rms,
            "embedding_fisher_ratio": fisher_ratio,
        }

    def finalize(
        self,
        *,
        seed: int,
        embedding_dim: int,
        epochs: int,
        max_per_class: int,
        min_per_class: int,
        train_fraction: float,
        lr: float,
        weight_decay: float,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for comparison, store in sorted(self._data.items()):
            output[comparison] = self._train_one(
                comparison,
                store,
                seed=int(seed),
                embedding_dim=int(embedding_dim),
                epochs=int(epochs),
                max_per_class=int(max_per_class),
                min_per_class=int(min_per_class),
                train_fraction=float(train_fraction),
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
        return output


def _base_errors(base: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _slot_errors(candidates: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _rank_from_score(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score, dim=1)
    rank = torch.empty_like(order)
    values = torch.arange(score.shape[1], device=score.device, dtype=torch.long)[None, :, None].expand_as(order)
    rank.scatter_(1, order, values)
    return rank


def _base_direction(base: torch.Tensor) -> torch.Tensor:
    direction = base[..., -1, :] - base[..., 0, :]
    norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)


def _path_length(traj: torch.Tensor) -> torch.Tensor:
    first = torch.linalg.norm(traj[..., :1, :], dim=-1)
    if int(traj.shape[-2]) <= 1:
        return first.squeeze(-1)
    steps = torch.linalg.norm(traj[..., 1:, :] - traj[..., :-1, :], dim=-1)
    return torch.cat([first, steps], dim=-1).sum(dim=-1)


def _expand_agent_feature(values: torch.Tensor, num_base_modes: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected [B,A] feature, got {tuple(values.shape)}")
    return values[:, None, :].expand(values.shape[0], int(num_base_modes), values.shape[1])


def _past_motion_features(past_traj: torch.Tensor, num_base_modes: int) -> Dict[str, torch.Tensor]:
    if past_traj.ndim != 4:
        return {}
    xy = past_traj[..., :2]
    if int(xy.shape[-2]) <= 1:
        zero = torch.zeros(xy.shape[0], xy.shape[1], device=xy.device, dtype=xy.dtype)
        return {
            "past_displacement": _expand_agent_feature(zero, num_base_modes),
            "past_path_length": _expand_agent_feature(zero, num_base_modes),
            "past_mean_step": _expand_agent_feature(zero, num_base_modes),
            "past_max_step": _expand_agent_feature(zero, num_base_modes),
            "past_straightness": _expand_agent_feature(zero, num_base_modes),
            "past_turn_intensity": _expand_agent_feature(zero, num_base_modes),
        }
    displacement = torch.linalg.norm(xy[..., -1, :] - xy[..., 0, :], dim=-1)
    steps = xy[..., 1:, :] - xy[..., :-1, :]
    step_norm = torch.linalg.norm(steps, dim=-1)
    path = step_norm.sum(dim=-1)
    mean_step = step_norm.mean(dim=-1)
    max_step = step_norm.amax(dim=-1)
    straightness = displacement / path.clamp_min(EPS)
    if int(steps.shape[-2]) > 1:
        v1 = steps[..., :-1, :]
        v2 = steps[..., 1:, :]
        denom = torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)
        cos = (v1 * v2).sum(dim=-1) / denom.clamp_min(EPS)
        turn_intensity = (1.0 - cos.clamp(-1.0, 1.0)).mean(dim=-1)
    else:
        turn_intensity = torch.zeros_like(displacement)
    return {
        "past_displacement": _expand_agent_feature(displacement, num_base_modes),
        "past_path_length": _expand_agent_feature(path, num_base_modes),
        "past_mean_step": _expand_agent_feature(mean_step, num_base_modes),
        "past_max_step": _expand_agent_feature(max_step, num_base_modes),
        "past_straightness": _expand_agent_feature(straightness, num_base_modes),
        "past_turn_intensity": _expand_agent_feature(turn_intensity, num_base_modes),
    }


def _gather_slot_values(values: torch.Tensor, slot_index: torch.Tensor) -> torch.Tensor:
    if values.ndim == 4:
        index = slot_index[:, None, :, :].to(dtype=torch.long, device=values.device)
        return torch.gather(values, dim=1, index=index).squeeze(1)
    if values.ndim == 6:
        index = slot_index[:, None, :, :, None, None].to(dtype=torch.long, device=values.device).expand(
            values.shape[0],
            1,
            values.shape[2],
            values.shape[3],
            values.shape[4],
            values.shape[5],
        )
        return torch.gather(values, dim=1, index=index).squeeze(1)
    raise ValueError(f"Unsupported gather shape: {tuple(values.shape)}")


def _front_pair_gap(front_candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(front_candidates.shape[1]) < 2:
        zero = front_candidates.new_zeros(front_candidates.shape[0], front_candidates.shape[2], front_candidates.shape[3])
        return zero, zero
    slot1 = front_candidates[:, 0]
    slot2 = front_candidates[:, 1]
    endpoint_gap = torch.linalg.norm(slot1[..., -1, :] - slot2[..., -1, :], dim=-1)
    traj_gap = torch.linalg.norm(slot1 - slot2, dim=-1).mean(dim=-1)
    return endpoint_gap, traj_gap


def _residual_shape_features(
    residual: torch.Tensor,
    base: torch.Tensor,
    *,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    endpoint = residual[..., -1, :]
    endpoint_norm = torch.linalg.norm(endpoint, dim=-1)
    traj_norm = torch.linalg.norm(residual, dim=-1).mean(dim=-1)
    direction = _base_direction(base)
    perp = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    forward = (endpoint * direction).sum(dim=-1)
    lateral = (endpoint * perp).sum(dim=-1)
    return {
        f"{prefix}_residual_endpoint_norm": endpoint_norm,
        f"{prefix}_residual_trajectory_norm": traj_norm,
        f"{prefix}_residual_forward": forward,
        f"{prefix}_residual_lateral": lateral,
        f"{prefix}_residual_abs_lateral": lateral.abs(),
    }


def _energy_features(temporal_energy: torch.Tensor, num_base_modes: int) -> Dict[str, torch.Tensor]:
    if temporal_energy.ndim != 5 or int(temporal_energy.shape[1]) != int(num_base_modes):
        return {}
    energy = temporal_energy
    if int(energy.shape[-1]) < 5:
        return {}
    min_neighbor_distance = energy[..., 0].clamp_min(0.0)
    soft_collision_energy = energy[..., 1].clamp_min(0.0)
    close_neighbor_count = energy[..., 2].clamp_min(0.0)
    approaching_score = energy[..., 3].clamp_min(0.0)
    endpoint_crowding_energy = energy[..., 4].clamp_min(0.0)
    risk = torch.stack(
        [
            torch.exp(-min_neighbor_distance / 0.5),
            soft_collision_energy / (1.0 + soft_collision_energy),
            close_neighbor_count / (1.0 + close_neighbor_count),
            approaching_score.clamp(0.0, 1.0),
            endpoint_crowding_energy / (1.0 + endpoint_crowding_energy),
        ],
        dim=0,
    ).amax(dim=0)
    return {
        "energy_risk_mean": risk.mean(dim=-1),
        "energy_risk_max": risk.amax(dim=-1),
        "min_neighbor_distance_mean": min_neighbor_distance.mean(dim=-1),
        "close_neighbor_count_mean": close_neighbor_count.mean(dim=-1),
        "approaching_score_mean": approaching_score.mean(dim=-1),
        "endpoint_crowding_energy_mean": endpoint_crowding_energy.mean(dim=-1),
    }


def _build_features(
    *,
    base: torch.Tensor,
    front_candidates: torch.Tensor,
    past_traj_original_scale: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: torch.Tensor,
    front_best_slot_local: torch.Tensor,
    front_score: torch.Tensor,
    base_metric_score: torch.Tensor,
    front_ade: torch.Tensor,
    front_fde: torch.Tensor,
    base_ade: torch.Tensor,
    base_fde: torch.Tensor,
    front_slot_start: int,
) -> Dict[str, torch.Tensor]:
    num_base_modes = int(base.shape[1])
    best_front = _gather_slot_values(front_candidates, front_best_slot_local)
    best_front_ade = torch.gather(front_ade, dim=1, index=front_best_slot_local[:, None, :, :]).squeeze(1)
    best_front_fde = torch.gather(front_fde, dim=1, index=front_best_slot_local[:, None, :, :]).squeeze(1)
    best_front_score = torch.gather(front_score, dim=1, index=front_best_slot_local[:, None, :, :]).squeeze(1)
    slot1_fde = front_fde[:, 0]
    slot2_fde = front_fde[:, min(1, int(front_fde.shape[1]) - 1)]
    slot1_ade = front_ade[:, 0]
    slot2_ade = front_ade[:, min(1, int(front_ade.shape[1]) - 1)]

    residual = best_front - base
    endpoint_gap, traj_gap = _front_pair_gap(front_candidates)

    best_other_score = front_score.masked_fill(
        torch.nn.functional.one_hot(front_best_slot_local, num_classes=int(front_score.shape[1]))
        .permute(0, 3, 1, 2)
        .bool(),
        float("inf"),
    ).amin(dim=1)
    best_other_score = torch.where(torch.isfinite(best_other_score), best_other_score, best_front_score)

    features: Dict[str, torch.Tensor] = {
        "base_ade": base_ade,
        "base_fde": base_fde,
        "base_metric_score": base_metric_score,
        "base_rank_fde": _rank_from_score(base_fde).to(dtype=torch.float32),
        "base_rank_metric": _rank_from_score(base_metric_score).to(dtype=torch.float32),
        "base_endpoint_norm": torch.linalg.norm(base[..., -1, :], dim=-1),
        "base_path_length": _path_length(base),
        "best_front_ade": best_front_ade,
        "best_front_fde": best_front_fde,
        "best_front_metric_score": best_front_score,
        "best_front_ade_gain": base_ade - best_front_ade,
        "best_front_fde_gain": base_fde - best_front_fde,
        "best_front_metric_gain": base_metric_score - best_front_score,
        "best_front_slot_actual": front_best_slot_local.to(dtype=torch.float32) + float(front_slot_start),
        "slot1_ade_gain": base_ade - slot1_ade,
        "slot2_ade_gain": base_ade - slot2_ade,
        "slot1_fde_gain": base_fde - slot1_fde,
        "slot2_fde_gain": base_fde - slot2_fde,
        "front_best_margin_vs_other": best_other_score - best_front_score,
        "front_slot_endpoint_gap": endpoint_gap,
        "front_slot_trajectory_gap": traj_gap,
    }
    features.update(_residual_shape_features(residual, base, prefix="best_front"))
    features.update(_residual_shape_features(front_candidates[:, 0] - base, base, prefix="slot1"))
    features.update(
        _residual_shape_features(
            front_candidates[:, min(1, int(front_candidates.shape[1]) - 1)] - base,
            base,
            prefix="slot2",
        )
    )
    features.update(_past_motion_features(past_traj_original_scale.to(device=base.device), num_base_modes))
    features.update(_energy_features(temporal_energy.to(device=base.device), num_base_modes))
    return features


def _add_counts(counts: Dict[str, int], name: str, mask: torch.Tensor) -> None:
    counts[name] = counts.get(name, 0) + int(mask.sum().detach().cpu().item())


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    _require_eval_args(args)
    _set_seed(int(args.seed))
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

    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=str(args.slow_cfg_path),
            checkpoint_path=str(args.slow_checkpoint),
        )
    )
    refiner = load_social_cvae_teacher_refiner(str(args.refiner_checkpoint), map_location=device).to(device)
    refiner.eval()
    selector = None
    selector_variant = None
    if args.selector_checkpoint:
        selector_variant = _checkpoint_variant(str(args.selector_checkpoint))
        selector = load_social_cvae_group_selector(str(args.selector_checkpoint), map_location=device).to(device)
        selector.eval()
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

    refiner_variant = _checkpoint_variant(str(args.refiner_checkpoint))
    print(
        "[diagnose_v58_front2_compression_cases] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} front={args.front_slot_start}:{int(args.front_slot_start) + int(args.front_slots)} "
        f"metric={args.oracle_select_metric} selector_variant={selector_variant}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v58_front2_compression_cases] warning: selected_samples normalization is diagnostic only")

    groups: Dict[str, GroupStats] = {
        "all_valid": GroupStats("all_valid"),
        "front_improve": GroupStats("front_improve"),
        "front_neutral": GroupStats("front_neutral"),
        "front_hurt": GroupStats("front_hurt"),
        "front_slot1_best": GroupStats("front_slot1_best"),
        "front_slot2_best": GroupStats("front_slot2_best"),
    }
    for threshold in _split_floats(str(args.strong_gain_thresholds)):
        groups[f"front_gain_ge_{_tag_float(threshold)}"] = GroupStats(f"front_gain_ge_{_tag_float(threshold)}")
    focus_topks = sorted({int(item) for item in _split_ints(str(args.focus_base_topks)) if int(item) > 0})
    for top_k in focus_topks:
        prefix = f"base_top{top_k}"
        groups[f"{prefix}_front_improve"] = GroupStats(f"{prefix}_front_improve")
        groups[f"{prefix}_front_neutral"] = GroupStats(f"{prefix}_front_neutral")
        groups[f"{prefix}_front_hurt"] = GroupStats(f"{prefix}_front_hurt")
        groups[f"{prefix}_front_slot1_best"] = GroupStats(f"{prefix}_front_slot1_best")
        groups[f"{prefix}_front_slot2_best"] = GroupStats(f"{prefix}_front_slot2_best")
    if selector is not None:
        groups.update(
            {
                "selector_hits_front_oracle": GroupStats("selector_hits_front_oracle"),
                "selector_misses_front_oracle": GroupStats("selector_misses_front_oracle"),
                "selector_improve": GroupStats("selector_improve"),
                "selector_hurt": GroupStats("selector_hurt"),
            }
        )
        for top_k in focus_topks:
            prefix = f"base_top{top_k}"
            groups[f"{prefix}_selector_hits_front_oracle"] = GroupStats(f"{prefix}_selector_hits_front_oracle")
            groups[f"{prefix}_selector_misses_front_oracle"] = GroupStats(f"{prefix}_selector_misses_front_oracle")
            groups[f"{prefix}_selector_improve"] = GroupStats(f"{prefix}_selector_improve")
            groups[f"{prefix}_selector_hurt"] = GroupStats(f"{prefix}_selector_hurt")
    separability = SeparabilityStats(_split_items(str(args.separability_features)))
    binned_distribution = BinnedDistributionStats(
        _split_items(str(args.binning_features)),
        bins=int(args.distribution_bins),
    )
    rolling_windows = RollingWindowStats(
        _split_items(str(args.window_stat_features)),
        every_chunks=int(args.window_stat_every_chunks),
    )
    embedding_diagnostic = (
        EmbeddingDiagnosticStats(_split_items(str(args.embedding_features)))
        if bool(args.embedding_diagnostic)
        else None
    )

    def add_distribution_comparison(
        name: str,
        feature_map: Mapping[str, torch.Tensor],
        *,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> None:
        separability.add(
            name,
            feature_map,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
        )
        binned_distribution.add(
            name,
            feature_map,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
        )
        rolling_windows.add(
            name,
            feature_map,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
        )
        if embedding_diagnostic is not None:
            embedding_diagnostic.add(
                name,
                feature_map,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
            )

    counts: Dict[str, int] = {}
    total_valid = 0
    total_scenes = 0
    total_latency_ms = 0.0
    chunks = list(_iter_chunks(list(enumerate(selected_samples)), args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        total_scenes += len(chunk)
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        slow_latencies, slow_output = _measure_predict_latency_ms(
            lambda: slow_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
        total_latency_ms += sum(float(item) for item in slow_latencies)
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
        total_latency_ms += sum(float(item) for item in refiner_latencies)

        refined = refiner_outputs["refined"]
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        agent_mask = batch["agent_mask"].to(device=device).bool()
        base = slow_output.slow_pred
        batch_size, num_slots, num_base_modes, num_agents = (
            int(refined.shape[0]),
            int(refined.shape[1]),
            int(refined.shape[2]),
            int(refined.shape[3]),
        )
        if int(num_slots) < int(args.front_slot_start) + int(args.front_slots):
            raise SystemExit("refined output does not contain requested front slots")
        if int(num_base_modes) != int(args.keep_k):
            raise SystemExit(f"Expected keep_k={args.keep_k}, got base_modes={num_base_modes}")

        front_start = int(args.front_slot_start)
        front_stop = front_start + int(args.front_slots)
        front_candidates = refined[:, front_start:front_stop]
        valid_base = agent_mask[:, None, :].expand(batch_size, num_base_modes, num_agents)
        valid_count = int(valid_base.sum().detach().cpu().item())
        total_valid += valid_count

        base_ade, base_fde = _base_errors(base, ground_truth)
        front_ade, front_fde = _slot_errors(front_candidates, ground_truth)
        front_score = _candidate_score_slots(front_candidates, ground_truth, metric=str(args.oracle_select_metric))
        if str(args.oracle_select_metric) == "fde":
            base_metric_score = base_fde
        else:
            base_metric_score = base_ade + base_fde
        front_best_slot_local = front_score.argmin(dim=1)
        best_front_score = torch.gather(front_score, dim=1, index=front_best_slot_local[:, None, :, :]).squeeze(1)
        metric_gain = base_metric_score - best_front_score
        margin = float(args.success_margin)
        base_metric_rank = _rank_from_score(base_metric_score)

        features = _build_features(
            base=base,
            front_candidates=front_candidates,
            past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
            ground_truth=ground_truth,
            temporal_energy=temporal_energy,
            front_best_slot_local=front_best_slot_local,
            front_score=front_score,
            base_metric_score=base_metric_score,
            front_ade=front_ade,
            front_fde=front_fde,
            base_ade=base_ade,
            base_fde=base_fde,
            front_slot_start=front_start,
        )

        improve = valid_base & (metric_gain > margin)
        neutral = valid_base & (metric_gain.abs() <= margin)
        hurt = valid_base & (metric_gain < -margin)
        slot1_best = valid_base & (front_best_slot_local == 0)
        slot2_best = valid_base & (front_best_slot_local == min(1, int(args.front_slots) - 1))

        groups["all_valid"].add(features, valid_base)
        groups["front_improve"].add(features, improve)
        groups["front_neutral"].add(features, neutral)
        groups["front_hurt"].add(features, hurt)
        groups["front_slot1_best"].add(features, slot1_best)
        groups["front_slot2_best"].add(features, slot2_best)
        add_distribution_comparison(
            "front_improve_vs_hurt",
            features,
            positive_mask=improve,
            negative_mask=hurt,
        )
        _add_counts(counts, "front_improve", improve)
        _add_counts(counts, "front_neutral", neutral)
        _add_counts(counts, "front_hurt", hurt)
        _add_counts(counts, "front_slot1_best", slot1_best)
        _add_counts(counts, "front_slot2_best", slot2_best)
        for top_k in focus_topks:
            prefix = f"base_top{top_k}"
            top_mask = valid_base & (base_metric_rank < int(top_k))
            top_improve = top_mask & (metric_gain > margin)
            top_neutral = top_mask & (metric_gain.abs() <= margin)
            top_hurt = top_mask & (metric_gain < -margin)
            top_slot1_best = top_mask & (front_best_slot_local == 0)
            top_slot2_best = top_mask & (front_best_slot_local == min(1, int(args.front_slots) - 1))
            groups[f"{prefix}_front_improve"].add(features, top_improve)
            groups[f"{prefix}_front_neutral"].add(features, top_neutral)
            groups[f"{prefix}_front_hurt"].add(features, top_hurt)
            groups[f"{prefix}_front_slot1_best"].add(features, top_slot1_best)
            groups[f"{prefix}_front_slot2_best"].add(features, top_slot2_best)
            add_distribution_comparison(
                f"{prefix}_front_improve_vs_hurt",
                features,
                positive_mask=top_improve,
                negative_mask=top_hurt,
            )
            _add_counts(counts, f"{prefix}_items", top_mask)
            _add_counts(counts, f"{prefix}_front_improve", top_improve)
            _add_counts(counts, f"{prefix}_front_neutral", top_neutral)
            _add_counts(counts, f"{prefix}_front_hurt", top_hurt)
            _add_counts(counts, f"{prefix}_front_slot1_best", top_slot1_best)
            _add_counts(counts, f"{prefix}_front_slot2_best", top_slot2_best)
        for threshold in _split_floats(str(args.strong_gain_thresholds)):
            mask = valid_base & (metric_gain >= float(threshold))
            name = f"front_gain_ge_{_tag_float(threshold)}"
            groups[name].add(features, mask)
            _add_counts(counts, name, mask)

        if selector is not None:
            selector_latencies, selector_outputs = _measure_predict_latency_ms(
                lambda: selector.select(
                    front_candidates,
                    base_trajectory=base,
                    past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                    temporal_energy_features=temporal_energy.to(device=device),
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            total_latency_ms += sum(float(item) for item in selector_latencies)
            logits = selector_outputs["logits"].to(device=device)
            selected_slot = _selector_indices_with_optional_slot0_fallback(logits, args=args)
            selected_score = torch.gather(front_score, dim=1, index=selected_slot[:, None, :, :]).squeeze(1)
            selected_gain = base_metric_score - selected_score
            selected_features = dict(features)
            selected_front = _gather_slot_values(front_candidates, selected_slot)
            selected_residual = selected_front - base
            selected_ade = torch.gather(front_ade, dim=1, index=selected_slot[:, None, :, :]).squeeze(1)
            selected_fde = torch.gather(front_fde, dim=1, index=selected_slot[:, None, :, :]).squeeze(1)
            selected_logit = torch.gather(logits, dim=1, index=selected_slot[:, None, :, :]).squeeze(1)
            other_logit = logits.masked_fill(
                torch.nn.functional.one_hot(selected_slot, num_classes=int(logits.shape[1])).permute(0, 3, 1, 2).bool(),
                float("-inf"),
            ).amax(dim=1)
            probs = torch.softmax(logits, dim=1)
            selected_prob = torch.gather(probs, dim=1, index=selected_slot[:, None, :, :]).squeeze(1)
            entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=1)
            selected_features["selector_local_slot"] = selected_slot.to(dtype=torch.float32)
            selected_features["selector_actual_slot"] = selected_slot.to(dtype=torch.float32) + float(front_start)
            selected_features["selector_metric_gain"] = selected_gain
            selected_features["selected_front_ade_gain"] = base_ade - selected_ade
            selected_features["selected_front_fde_gain"] = base_fde - selected_fde
            selected_features["selected_score_minus_front_oracle"] = selected_score - best_front_score
            selected_features["selector_selected_logit"] = selected_logit
            selected_features["selector_logit_margin"] = selected_logit - other_logit
            selected_features["selector_selected_prob"] = selected_prob
            selected_features["selector_entropy"] = entropy
            selected_features.update(_residual_shape_features(selected_residual, base, prefix="selected"))
            hit = valid_base & (selected_slot == front_best_slot_local)
            miss = valid_base & (selected_slot != front_best_slot_local)
            selector_improve = valid_base & (selected_gain > margin)
            selector_hurt = valid_base & (selected_gain < -margin)
            groups["selector_hits_front_oracle"].add(selected_features, hit)
            groups["selector_misses_front_oracle"].add(selected_features, miss)
            groups["selector_improve"].add(selected_features, selector_improve)
            groups["selector_hurt"].add(selected_features, selector_hurt)
            add_distribution_comparison(
                "selector_improve_vs_hurt",
                selected_features,
                positive_mask=selector_improve,
                negative_mask=selector_hurt,
            )
            add_distribution_comparison(
                "selector_hit_vs_miss",
                selected_features,
                positive_mask=hit,
                negative_mask=miss,
            )
            _add_counts(counts, "selector_hits_front_oracle", hit)
            _add_counts(counts, "selector_misses_front_oracle", miss)
            _add_counts(counts, "selector_improve", selector_improve)
            _add_counts(counts, "selector_hurt", selector_hurt)
            for top_k in focus_topks:
                prefix = f"base_top{top_k}"
                top_mask = valid_base & (base_metric_rank < int(top_k))
                top_hit = top_mask & (selected_slot == front_best_slot_local)
                top_miss = top_mask & (selected_slot != front_best_slot_local)
                top_selector_improve = top_mask & (selected_gain > margin)
                top_selector_hurt = top_mask & (selected_gain < -margin)
                groups[f"{prefix}_selector_hits_front_oracle"].add(selected_features, top_hit)
                groups[f"{prefix}_selector_misses_front_oracle"].add(selected_features, top_miss)
                groups[f"{prefix}_selector_improve"].add(selected_features, top_selector_improve)
                groups[f"{prefix}_selector_hurt"].add(selected_features, top_selector_hurt)
                add_distribution_comparison(
                    f"{prefix}_selector_improve_vs_hurt",
                    selected_features,
                    positive_mask=top_selector_improve,
                    negative_mask=top_selector_hurt,
                )
                add_distribution_comparison(
                    f"{prefix}_selector_hit_vs_miss",
                    selected_features,
                    positive_mask=top_hit,
                    negative_mask=top_miss,
                )
                _add_counts(counts, f"{prefix}_selector_hits_front_oracle", top_hit)
                _add_counts(counts, f"{prefix}_selector_misses_front_oracle", top_miss)
                _add_counts(counts, f"{prefix}_selector_improve", top_selector_improve)
                _add_counts(counts, f"{prefix}_selector_hurt", top_selector_hurt)

        if int(args.log_every) > 0 and (chunk_index % int(args.log_every) == 0 or chunk_index == len(chunks)):
            print(
                "[diagnose_v58_front2_compression_cases] "
                f"chunk {chunk_index}/{len(chunks)} valid={total_valid} "
                f"improve={counts.get('front_improve', 0)} hurt={counts.get('front_hurt', 0)}"
            )
        rolling_windows.maybe_flush(chunk_index)

    rolling_windows.maybe_flush(len(chunks), force=True)
    group_summary = {name: group.finalize(total_valid) for name, group in sorted(groups.items())}
    ratios = {name: float(count / max(total_valid, 1)) for name, count in sorted(counts.items())}
    topk_ratios: Dict[str, Dict[str, float]] = {}
    for top_k in focus_topks:
        prefix = f"base_top{top_k}"
        denom = max(int(counts.get(f"{prefix}_items", 0)), 1)
        keys = [
            "front_improve",
            "front_neutral",
            "front_hurt",
            "front_slot1_best",
            "front_slot2_best",
            "selector_hits_front_oracle",
            "selector_misses_front_oracle",
            "selector_improve",
            "selector_hurt",
        ]
        topk_ratios[prefix] = {
            key: float(counts.get(f"{prefix}_{key}", 0) / denom)
            for key in keys
            if f"{prefix}_{key}" in counts
        }
        topk_ratios[prefix]["items"] = float(counts.get(f"{prefix}_items", 0))
    embedding_summary = (
        {}
        if embedding_diagnostic is None
        else embedding_diagnostic.finalize(
            seed=int(args.seed),
            embedding_dim=int(args.embedding_dim),
            epochs=int(args.embedding_epochs),
            max_per_class=int(args.embedding_max_per_class),
            min_per_class=int(args.embedding_min_per_class),
            train_fraction=float(args.embedding_train_fraction),
            lr=float(args.embedding_lr),
            weight_decay=float(args.embedding_weight_decay),
        )
    )
    result: Dict[str, Any] = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v58_front2_compression_cases",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": Path(args.project_root).expanduser().resolve().as_posix(),
            "protocol": str(args.protocol),
            "subset": str(args.subset),
            "split": str(args.split),
            "seed": int(args.seed),
            "sample_mode": str(args.sample_mode),
            "data_root": data_root.as_posix(),
            "refiner_checkpoint": Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix(),
            "refiner_variant": refiner_variant,
            "selector_checkpoint": None
            if not args.selector_checkpoint
            else Path(str(args.selector_checkpoint)).expanduser().resolve().as_posix(),
            "selector_variant": selector_variant,
            "selector_confidence_fallback_to_slot0": bool(args.selector_confidence_fallback_to_slot0),
            "selector_fallback_prob_margin": float(args.selector_fallback_prob_margin),
            "selector_fallback_min_selected_prob": float(args.selector_fallback_min_selected_prob),
            "residual_slots": int(args.residual_slots),
            "front_slot_start": int(args.front_slot_start),
            "front_slots": int(args.front_slots),
            "keep_k": int(args.keep_k),
            "oracle_select_metric": str(args.oracle_select_metric),
            "success_margin": float(args.success_margin),
            "focus_base_topks": list(focus_topks),
            "separability_features": _split_items(str(args.separability_features)),
            "top_separability_features": int(args.top_separability_features),
            "binning_features": _split_items(str(args.binning_features)),
            "distribution_bins": int(args.distribution_bins),
            "window_stat_every_chunks": int(args.window_stat_every_chunks),
            "window_stat_features": _split_items(str(args.window_stat_features)),
            "embedding_diagnostic": bool(args.embedding_diagnostic),
            "embedding_features": _split_items(str(args.embedding_features)),
            "embedding_dim": int(args.embedding_dim),
            "embedding_epochs": int(args.embedding_epochs),
            "embedding_max_per_class": int(args.embedding_max_per_class),
            "embedding_min_per_class": int(args.embedding_min_per_class),
            "normalization_meta": _jsonable(normalization_meta),
        },
        "counts": {
            "selected_scenes": int(len(selected_samples)),
            "selected_eval_items": int(selected_eval_items),
            "valid_base_agent_items": int(total_valid),
            **{name: int(count) for name, count in sorted(counts.items())},
        },
        "ratios": ratios,
        "topk_ratios": topk_ratios,
        "groups": group_summary,
        "separability": separability.finalize(top_n=int(args.top_separability_features)),
        "binned_distributions": binned_distribution.finalize(top_n=int(args.top_separability_features)),
        "window_stats": rolling_windows.finalize() if rolling_windows.enabled else {},
        "embedding_diagnostics": embedding_summary,
        "latency_total_ms": float(total_latency_ms),
    }
    output_json = Path(str(args.output_json)).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    if args.output_txt:
        output_txt = Path(str(args.output_txt)).expanduser().resolve()
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(_render_single(result), encoding="utf-8")
    print(_render_single(result))
    print(f"case_diag_json={output_json.as_posix()}")
    if args.output_txt:
        print(f"case_diag_txt={Path(str(args.output_txt)).expanduser().resolve().as_posix()}")
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _group_line(name: str, group: Mapping[str, Any]) -> str:
    return (
        f"{name}: count={int(group.get('count', 0))} ratio={_fmt(group.get('ratio'))} "
        f"gain_fde={_fmt(group.get('mean_best_front_fde_gain'))} "
        f"gain_ade={_fmt(group.get('mean_best_front_ade_gain'))} "
        f"base_fde={_fmt(group.get('mean_base_fde'))} "
        f"base_rank={_fmt(group.get('mean_base_rank_fde'))} "
        f"res_endpoint={_fmt(group.get('mean_best_front_residual_endpoint_norm'))} "
        f"risk={_fmt(group.get('mean_energy_risk_mean'))}"
    )


def _rich_group_line(name: str, group: Mapping[str, Any]) -> str:
    parts = [
        f"{name}: count={int(group.get('mean_count', group.get('count', 0)) or 0)}",
        f"ratio={_fmt(group.get('ratio'))}",
        (
            "base("
            f"fde={_fmt(group.get('mean_base_fde'))}, "
            f"ade={_fmt(group.get('mean_base_ade'))}, "
            f"rank={_fmt(group.get('mean_base_rank_fde'))}, "
            f"path={_fmt(group.get('mean_base_path_length'))})"
        ),
        (
            "past("
            f"disp={_fmt(group.get('mean_past_displacement'))}, "
            f"path={_fmt(group.get('mean_past_path_length'))}, "
            f"step={_fmt(group.get('mean_past_mean_step'))}, "
            f"turn={_fmt(group.get('mean_past_turn_intensity'))})"
        ),
        (
            "social("
            f"risk={_fmt(group.get('mean_energy_risk_mean'))}, "
            f"mindist={_fmt(group.get('mean_min_neighbor_distance_mean'))}, "
            f"close={_fmt(group.get('mean_close_neighbor_count_mean'))}, "
            f"approach={_fmt(group.get('mean_approaching_score_mean'))})"
        ),
        (
            "oracle_corr("
            f"gain_fde={_fmt(group.get('mean_best_front_fde_gain'))}, "
            f"gain_ade={_fmt(group.get('mean_best_front_ade_gain'))}, "
            f"res_ep={_fmt(group.get('mean_best_front_residual_endpoint_norm'))}, "
            f"forward={_fmt(group.get('mean_best_front_residual_forward'))}, "
            f"abs_lat={_fmt(group.get('mean_best_front_residual_abs_lateral'))}, "
            f"slot_gap={_fmt(group.get('mean_front_slot_endpoint_gap'))})"
        ),
    ]
    if group.get("mean_selected_front_fde_gain") is not None or group.get("mean_selector_metric_gain") is not None:
        parts.append(
            "selected("
            f"gain_fde={_fmt(group.get('mean_selected_front_fde_gain'))}, "
            f"metric_gain={_fmt(group.get('mean_selector_metric_gain'))}, "
            f"score_gap={_fmt(group.get('mean_selected_score_minus_front_oracle'))}, "
            f"res_ep={_fmt(group.get('mean_selected_residual_endpoint_norm'))}, "
            f"prob={_fmt(group.get('mean_selector_selected_prob'))}, "
            f"logit_margin={_fmt(group.get('mean_selector_logit_margin'))}, "
            f"entropy={_fmt(group.get('mean_selector_entropy'))})"
        )
    return " ".join(parts)


def _topbase_group_names(groups: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    suffixes = [
        "front_improve",
        "front_hurt",
        "selector_improve",
        "selector_hurt",
        "selector_hits_front_oracle",
        "selector_misses_front_oracle",
    ]
    for prefix in ("base_top1", "base_top3", "base_top5"):
        for suffix in suffixes:
            name = f"{prefix}_{suffix}"
            if name in groups:
                names.append(name)
    for name in sorted(groups):
        if name.startswith("base_top") and name not in names:
            if any(name.endswith(suffix) for suffix in suffixes):
                names.append(name)
    return names


def _separability_line(item: Mapping[str, Any]) -> str:
    return (
        f"{item.get('feature')}: "
        f"best_auc={_fmt(item.get('best_auc'))} "
        f"auc_pos_high={_fmt(item.get('auc_positive_higher'))} "
        f"ks={_fmt(item.get('ks'))} "
        f"smd={_fmt(item.get('smd'))} "
        f"pos_mean={_fmt(item.get('positive_mean'))} "
        f"neg_mean={_fmt(item.get('negative_mean'))} "
        f"dir={item.get('direction') or 'NA'} "
        f"sep={item.get('separation') or 'NA'}"
    )


def _binned_line(item: Mapping[str, Any]) -> str:
    return (
        f"{item.get('feature')}: "
        f"rate_range={_fmt(item.get('positive_rate_range'))} "
        f"trend={_fmt(item.get('positive_rate_trend'))} "
        f"pos_mean={_fmt(item.get('positive_mean'))} "
        f"neg_mean={_fmt(item.get('negative_mean'))} "
        f"bins={_fmt(item.get('usable_bins'), digits=0)}"
    )


def _embedding_line(name: str, item: Mapping[str, Any]) -> str:
    if item.get("status") != "ok":
        return f"{name}: status={item.get('status') or 'NA'}"
    return (
        f"{name}: "
        f"auc={_fmt(item.get('test_auc_positive_higher'))} "
        f"best_auc={_fmt(item.get('test_best_auc'))} "
        f"bal_acc={_fmt(item.get('test_balanced_accuracy'))} "
        f"fisher={_fmt(item.get('embedding_fisher_ratio'))} "
        f"features={_fmt(item.get('feature_count'), digits=0)} "
        f"dim={_fmt(item.get('embedding_dim'), digits=0)}"
    )


def _window_line(feature: str, labels: Mapping[str, Any]) -> str:
    pos = labels.get("positive", {})
    neg = labels.get("negative", {})
    return (
        f"{feature}: "
        f"pos_med_range={_fmt(pos.get('median_range'))} "
        f"neg_med_range={_fmt(neg.get('median_range'))} "
        f"pos_mean_range={_fmt(pos.get('mean_range'))} "
        f"neg_mean_range={_fmt(neg.get('mean_range'))}"
    )


def _interesting_separability_names(separability: Mapping[str, Any]) -> List[str]:
    preferred = [
        "base_top1_selector_improve_vs_hurt",
        "base_top3_selector_improve_vs_hurt",
        "base_top5_selector_improve_vs_hurt",
        "base_top1_selector_hit_vs_miss",
        "base_top3_selector_hit_vs_miss",
        "base_top5_selector_hit_vs_miss",
        "selector_improve_vs_hurt",
        "front_improve_vs_hurt",
    ]
    names = [name for name in preferred if name in separability]
    for name in sorted(separability):
        if name not in names and (
            name.endswith("selector_improve_vs_hurt")
            or name.endswith("selector_hit_vs_miss")
            or name.endswith("front_improve_vs_hurt")
        ):
            names.append(name)
    return names


def _interesting_named_sections(section: Mapping[str, Any]) -> List[str]:
    preferred = [
        "base_top1_selector_improve_vs_hurt",
        "base_top3_selector_improve_vs_hurt",
        "base_top5_selector_improve_vs_hurt",
        "selector_improve_vs_hurt",
        "front_improve_vs_hurt",
    ]
    names = [name for name in preferred if name in section]
    for name in sorted(section):
        if name not in names and (name.endswith("selector_improve_vs_hurt") or name.endswith("front_improve_vs_hurt")):
            names.append(name)
    return names


def _render_single(result: Mapping[str, Any]) -> str:
    meta = result.get("meta", {})
    counts = result.get("counts", {})
    ratios = result.get("ratios", {})
    topk_ratios = result.get("topk_ratios", {})
    groups = result.get("groups", {})
    separability = result.get("separability", {})
    binned = result.get("binned_distributions", {})
    embeddings = result.get("embedding_diagnostics", {})
    window_stats = result.get("window_stats", {})
    lines = [
        "",
        "[front2 compression case diagnosis]",
        f"split={meta.get('split')} seed={meta.get('seed')} metric={meta.get('oracle_select_metric')} "
        f"front_start={meta.get('front_slot_start')} front_slots={meta.get('front_slots')} "
        f"valid={counts.get('valid_base_agent_items')}",
        "case ratios:",
        f"  front_improve: {_fmt(ratios.get('front_improve'))}",
        f"  front_neutral: {_fmt(ratios.get('front_neutral'))}",
        f"  front_hurt:    {_fmt(ratios.get('front_hurt'))}",
        f"  slot1_best:    {_fmt(ratios.get('front_slot1_best'))}",
        f"  slot2_best:    {_fmt(ratios.get('front_slot2_best'))}",
    ]
    for key in sorted(ratios):
        if key.startswith("front_gain_ge_"):
            lines.append(f"  {key}: {_fmt(ratios.get(key))}")
    if "selector_hits_front_oracle" in ratios:
        lines.extend(
            [
                f"  selector_hits_front_oracle: {_fmt(ratios.get('selector_hits_front_oracle'))}",
                f"  selector_improve:           {_fmt(ratios.get('selector_improve'))}",
                f"  selector_hurt:              {_fmt(ratios.get('selector_hurt'))}",
            ]
        )
    if topk_ratios:
        lines.append("top base conditional ratios:")
        for name in sorted(topk_ratios):
            row = topk_ratios[name]
            lines.append(
                f"  {name}: items={int(row.get('items', 0))} "
                f"improve={_fmt(row.get('front_improve'))} "
                f"hurt={_fmt(row.get('front_hurt'))} "
                f"slot1_best={_fmt(row.get('front_slot1_best'))} "
                f"slot2_best={_fmt(row.get('front_slot2_best'))} "
                f"selector_hit={_fmt(row.get('selector_hits_front_oracle'))}"
            )
    lines.append("feature means:")
    for name in [
        "front_improve",
        "front_neutral",
        "front_hurt",
        "front_slot1_best",
        "front_slot2_best",
        "selector_hits_front_oracle",
        "selector_misses_front_oracle",
    ]:
        if name in groups:
            lines.append(f"  {_group_line(name, groups[name])}")
    topbase_names = _topbase_group_names(groups)
    if topbase_names:
        lines.append("top-base feature means:")
        for name in topbase_names:
            lines.append(f"  {_rich_group_line(name, groups[name])}")
    if separability:
        lines.append("feature separability:")
        for name in _interesting_separability_names(separability):
            top_features = list(separability.get(name, {}).get("top_features", []))
            if not top_features:
                continue
            lines.append(f"  [{name}]")
            for item in top_features[:5]:
                lines.append(f"    {_separability_line(item)}")
    if binned:
        lines.append("feature bin distributions:")
        for name in _interesting_named_sections(binned):
            top_features = list(binned.get(name, {}).get("top_features", []))
            if not top_features:
                continue
            lines.append(f"  [{name}]")
            for item in top_features[:5]:
                lines.append(f"    {_binned_line(item)}")
    if embeddings:
        lines.append("observable MLP separability:")
        for name in _interesting_named_sections(embeddings):
            lines.append(f"  {_embedding_line(name, embeddings.get(name, {}))}")
    window_summary = window_stats.get("summary", {}) if isinstance(window_stats, Mapping) else {}
    if window_summary:
        lines.append("rolling window drift:")
        for name in _interesting_named_sections(window_summary):
            feature_rows = list(window_summary.get(name, {}).items())
            feature_rows.sort(
                key=lambda item: max(
                    float(item[1].get("positive", {}).get("median_range") or 0.0),
                    float(item[1].get("negative", {}).get("median_range") or 0.0),
                ),
                reverse=True,
            )
            if not feature_rows:
                continue
            lines.append(f"  [{name}]")
            for feature, labels in feature_rows[:5]:
                lines.append(f"    {_window_line(feature, labels)}")
    return "\n".join(lines)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    items = [float(v) for v in values if v is not None]
    if not items:
        return None
    return float(sum(items) / len(items))


SEPARABILITY_NUMERIC_FIELDS: Sequence[str] = (
    "positive_count",
    "negative_count",
    "positive_mean",
    "negative_mean",
    "positive_std",
    "negative_std",
    "smd",
    "auc_positive_higher",
    "best_auc",
    "ks",
)

BINNED_NUMERIC_FIELDS: Sequence[str] = (
    "positive_count",
    "negative_count",
    "positive_mean",
    "negative_mean",
    "usable_bins",
    "positive_rate_range",
    "positive_rate_trend",
)

EMBEDDING_NUMERIC_FIELDS: Sequence[str] = (
    "feature_count",
    "embedding_dim",
    "positive_count",
    "negative_count",
    "train_count",
    "test_count",
    "final_train_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_auc_positive_higher",
    "test_best_auc",
    "embedding_centroid_distance",
    "embedding_within_rms",
    "embedding_fisher_ratio",
)

WINDOW_NUMERIC_FIELDS: Sequence[str] = (
    "windows",
    "mean_min",
    "mean_max",
    "mean_range",
    "median_min",
    "median_max",
    "median_range",
    "count_min",
    "count_max",
)


def _aggregate_separability(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    comparison_names = sorted({name for row in rows for name in row.get("separability", {})})
    for comparison in comparison_names:
        feature_names = sorted(
            {
                feature
                for row in rows
                for feature in row.get("separability", {}).get(comparison, {}).get("features", {})
            }
        )
        features: Dict[str, Any] = {}
        ranked: List[Dict[str, Any]] = []
        for feature in feature_names:
            items = [
                row.get("separability", {}).get(comparison, {}).get("features", {}).get(feature, {})
                for row in rows
            ]
            aggregate_item: Dict[str, Any] = {"feature": feature}
            for field in SEPARABILITY_NUMERIC_FIELDS:
                aggregate_item[field] = _mean(item.get(field) for item in items)
            directions = [str(item.get("direction")) for item in items if item.get("direction")]
            if directions:
                aggregate_item["direction"] = max(set(directions), key=directions.count)
            aggregate_item["separation"] = _separation_label(
                aggregate_item.get("best_auc"),
                aggregate_item.get("ks"),
            )
            features[feature] = aggregate_item
            ranked.append(aggregate_item)
        ranked.sort(
            key=lambda row: (
                float(row.get("best_auc") or 0.5),
                float(row.get("ks") or 0.0),
                abs(float(row.get("smd") or 0.0)),
            ),
            reverse=True,
        )
        output[comparison] = {
            "features": features,
            "ranked_features": [item["feature"] for item in ranked],
            "top_features": ranked[:12],
        }
    return output


def _aggregate_binned_distributions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    comparison_names = sorted({name for row in rows for name in row.get("binned_distributions", {})})
    for comparison in comparison_names:
        feature_names = sorted(
            {
                feature
                for row in rows
                for feature in row.get("binned_distributions", {}).get(comparison, {}).get("features", {})
            }
        )
        features: Dict[str, Any] = {}
        ranked: List[Dict[str, Any]] = []
        for feature in feature_names:
            items = [
                row.get("binned_distributions", {}).get(comparison, {}).get("features", {}).get(feature, {})
                for row in rows
            ]
            aggregate_item: Dict[str, Any] = {"feature": feature}
            for field in BINNED_NUMERIC_FIELDS:
                aggregate_item[field] = _mean(item.get(field) for item in items)
            features[feature] = aggregate_item
            ranked.append(aggregate_item)
        ranked.sort(
            key=lambda row: (
                float(row.get("positive_rate_range") or 0.0),
                abs(float(row.get("positive_rate_trend") or 0.0)),
            ),
            reverse=True,
        )
        output[comparison] = {
            "features": features,
            "ranked_features": [item["feature"] for item in ranked],
            "top_features": ranked[:12],
        }
    return output


def _aggregate_embedding_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    comparison_names = sorted({name for row in rows for name in row.get("embedding_diagnostics", {})})
    for comparison in comparison_names:
        items = [row.get("embedding_diagnostics", {}).get(comparison, {}) for row in rows]
        aggregate_item: Dict[str, Any] = {"status": "ok"}
        statuses = [str(item.get("status")) for item in items if item.get("status")]
        if statuses:
            aggregate_item["status"] = max(set(statuses), key=statuses.count)
            aggregate_item["ok_runs"] = int(sum(1 for status in statuses if status == "ok"))
        for field in EMBEDDING_NUMERIC_FIELDS:
            aggregate_item[field] = _mean(item.get(field) for item in items)
        feature_names = next((item.get("feature_names") for item in items if item.get("feature_names")), None)
        if feature_names:
            aggregate_item["feature_names"] = list(feature_names)
        output[comparison] = aggregate_item
    return output


def _aggregate_window_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {"summary": {}}
    comparison_names = sorted(
        {
            comparison
            for row in rows
            for comparison in row.get("window_stats", {}).get("summary", {})
        }
    )
    for comparison in comparison_names:
        comparison_out: Dict[str, Any] = {}
        feature_names = sorted(
            {
                feature
                for row in rows
                for feature in row.get("window_stats", {}).get("summary", {}).get(comparison, {})
            }
        )
        for feature in feature_names:
            label_out: Dict[str, Any] = {}
            labels = sorted(
                {
                    label
                    for row in rows
                    for label in row.get("window_stats", {})
                    .get("summary", {})
                    .get(comparison, {})
                    .get(feature, {})
                }
            )
            for label in labels:
                items = [
                    row.get("window_stats", {})
                    .get("summary", {})
                    .get(comparison, {})
                    .get(feature, {})
                    .get(label, {})
                    for row in rows
                ]
                label_out[label] = {field: _mean(item.get(field) for item in items) for field in WINDOW_NUMERIC_FIELDS}
            if label_out:
                comparison_out[feature] = label_out
        if comparison_out:
            output["summary"][comparison] = comparison_out
    return output


def _load_case_file(project_root: Path, run_prefix: str, diag_file_prefix: str, seed: int, split: str) -> Mapping[str, Any]:
    path = (
        project_root
        / "trustmoe_traj"
        / "analysis"
        / "experiment_runs"
        / f"{run_prefix}_seed{seed}"
        / f"{diag_file_prefix}_{split}.json"
    )
    if not path.exists():
        alt = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "experiment_runs"
            / run_prefix
            / f"{diag_file_prefix}_seed{seed}_{split}.json"
        )
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_saved(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.run_prefix:
        raise SystemExit("--summarize-only requires --run-prefix")
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(str(args.seeds))
    splits = _split_items(str(args.splits))
    rows: List[Mapping[str, Any]] = []
    for seed in seeds:
        for split in splits:
            rows.append(_load_case_file(project_root, str(args.run_prefix), str(args.diag_file_prefix), seed, split))

    aggregate: Dict[str, Any] = {"meta": {"requested": len(seeds), "seeds": seeds, "splits": splits}, "splits": {}}
    for split in splits:
        split_rows = [row for row in rows if row.get("meta", {}).get("split") == split]
        split_out: Dict[str, Any] = {"available": len(split_rows), "groups": {}, "ratios": {}}
        split_out["topk_ratios"] = {}
        ratio_keys = sorted({key for row in split_rows for key in row.get("ratios", {})})
        for key in ratio_keys:
            split_out["ratios"][key] = _mean(row.get("ratios", {}).get(key) for row in split_rows)
        topk_names = sorted({key for row in split_rows for key in row.get("topk_ratios", {})})
        for topk_name in topk_names:
            metric_keys = sorted(
                {
                    key
                    for row in split_rows
                    for key in row.get("topk_ratios", {}).get(topk_name, {})
                }
            )
            split_out["topk_ratios"][topk_name] = {
                key: _mean(row.get("topk_ratios", {}).get(topk_name, {}).get(key) for row in split_rows)
                for key in metric_keys
            }
        group_names = sorted({key for row in split_rows for key in row.get("groups", {})})
        for group_name in group_names:
            group_rows = [row.get("groups", {}).get(group_name, {}) for row in split_rows]
            metric_keys = sorted({key for group in group_rows for key in group if key.startswith("mean_") or key == "ratio"})
            split_out["groups"][group_name] = {
                key: _mean(group.get(key) for group in group_rows) for key in metric_keys
            }
            split_out["groups"][group_name]["mean_count"] = _mean(group.get("count") for group in group_rows)
        split_out["separability"] = _aggregate_separability(split_rows)
        split_out["binned_distributions"] = _aggregate_binned_distributions(split_rows)
        split_out["embedding_diagnostics"] = _aggregate_embedding_diagnostics(split_rows)
        split_out["window_stats"] = _aggregate_window_stats(split_rows)
        aggregate["splits"][split] = split_out

    output_json = None
    output_txt = None
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
    else:
        output_json = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "experiment_runs"
            / str(args.run_prefix)
            / f"{args.diag_file_prefix}_summary.json"
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(aggregate), indent=2, sort_keys=True), encoding="utf-8")
    if args.output_txt:
        output_txt = Path(args.output_txt).expanduser().resolve()
    else:
        output_txt = output_json.with_suffix(".txt")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_aggregate(aggregate)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")
    return aggregate


def _render_aggregate(aggregate: Mapping[str, Any]) -> str:
    lines = ["", "===== FRONT2 COMPRESSION CASE SUMMARY ====="]
    for split, data in aggregate.get("splits", {}).items():
        lines.append(f"-- {split} -- available={data.get('available')}")
        ratios = data.get("ratios", {})
        for key in [
            "front_improve",
            "front_neutral",
            "front_hurt",
            "front_slot1_best",
            "front_slot2_best",
            "selector_hits_front_oracle",
            "selector_improve",
            "selector_hurt",
        ]:
            if key in ratios:
                lines.append(f"{key}: {_fmt(ratios.get(key))}")
        for key in sorted(ratios):
            if key.startswith("front_gain_ge_"):
                lines.append(f"{key}: {_fmt(ratios.get(key))}")
        groups = data.get("groups", {})
        topk_ratios = data.get("topk_ratios", {})
        if topk_ratios:
            lines.append("top base conditional ratios:")
            for name in sorted(topk_ratios):
                row = topk_ratios[name]
                lines.append(
                    f"{name}: items={_fmt(row.get('items'))} "
                    f"improve={_fmt(row.get('front_improve'))} "
                    f"hurt={_fmt(row.get('front_hurt'))} "
                    f"slot1_best={_fmt(row.get('front_slot1_best'))} "
                    f"slot2_best={_fmt(row.get('front_slot2_best'))} "
                    f"selector_hit={_fmt(row.get('selector_hits_front_oracle'))} "
                    f"selector_improve={_fmt(row.get('selector_improve'))} "
                    f"selector_hurt={_fmt(row.get('selector_hurt'))}"
                )
        lines.append("feature means:")
        for name in [
            "front_improve",
            "front_neutral",
            "front_hurt",
            "selector_hits_front_oracle",
            "selector_misses_front_oracle",
        ]:
            if name not in groups:
                continue
            group = groups[name]
            lines.append(
                f"  {name}: ratio={_fmt(group.get('ratio'))} "
                f"gain_fde={_fmt(group.get('mean_best_front_fde_gain'))} "
                f"gain_ade={_fmt(group.get('mean_best_front_ade_gain'))} "
                f"base_fde={_fmt(group.get('mean_base_fde'))} "
                f"base_rank={_fmt(group.get('mean_base_rank_fde'))} "
                f"res_endpoint={_fmt(group.get('mean_best_front_residual_endpoint_norm'))} "
                f"risk={_fmt(group.get('mean_energy_risk_mean'))}"
            )
        topbase_names = _topbase_group_names(groups)
        if topbase_names:
            lines.append("top-base feature means:")
            for name in topbase_names:
                lines.append(f"  {_rich_group_line(name, groups[name])}")
        separability = data.get("separability", {})
        if separability:
            lines.append("feature separability:")
            for name in _interesting_separability_names(separability):
                top_features = list(separability.get(name, {}).get("top_features", []))
                if not top_features:
                    continue
                lines.append(f"  [{name}]")
                for item in top_features[:8]:
                    lines.append(f"    {_separability_line(item)}")
        binned = data.get("binned_distributions", {})
        if binned:
            lines.append("feature bin distributions:")
            for name in _interesting_named_sections(binned):
                top_features = list(binned.get(name, {}).get("top_features", []))
                if not top_features:
                    continue
                lines.append(f"  [{name}]")
                for item in top_features[:6]:
                    lines.append(f"    {_binned_line(item)}")
        embeddings = data.get("embedding_diagnostics", {})
        if embeddings:
            lines.append("observable MLP separability:")
            for name in _interesting_named_sections(embeddings):
                lines.append(f"  {_embedding_line(name, embeddings.get(name, {}))}")
        window_summary = data.get("window_stats", {}).get("summary", {})
        if window_summary:
            lines.append("rolling window drift:")
            for name in _interesting_named_sections(window_summary):
                feature_rows = list(window_summary.get(name, {}).items())
                feature_rows.sort(
                    key=lambda item: max(
                        float(item[1].get("positive", {}).get("median_range") or 0.0),
                        float(item[1].get("negative", {}).get("median_range") or 0.0),
                    ),
                    reverse=True,
                )
                if not feature_rows:
                    continue
                lines.append(f"  [{name}]")
                for feature, labels in feature_rows[:6]:
                    lines.append(f"    {_window_line(feature, labels)}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_saved(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
