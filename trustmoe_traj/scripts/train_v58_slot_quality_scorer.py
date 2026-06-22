"""Train a V58-K observable residual-slot quality scorer.

This trains a binary scorer over full residual-slot candidates.  Labels use
ground truth only during training; features are inference-visible and are built
from residual corrections, slot0 reference corrections, slow base trajectories,
past trajectories, and temporal energy context.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import (
    MoFlowSlowPredictor,
    V58SlotQualityScorer,
    V58SlotQualityScorerConfig,
    build_v58_slot_quality_features,
    load_social_cvae_teacher_refiner,
    v58_slot_quality_feature_names,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import _predictor_cfg, _set_seed
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trustmoe_traj" / "analysis" / "v58_slot_quality_scorer_models"
EPS = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V58-K residual slot quality scorer.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
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
    parser.add_argument("--train-slots", type=str, default="1,2,3,4,5,6,7")
    parser.add_argument(
        "--training-mode",
        type=str,
        default="binary_good",
        choices=["binary_good", "two_stage_replacement"],
        help="binary_good trains the original good/bad scorer; two_stage_replacement trains rank+accept heads.",
    )
    parser.add_argument("--rank-label-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--lambda-rank-ce", type=float, default=1.0)
    parser.add_argument("--lambda-accept-bce", type=float, default=1.0)
    parser.add_argument("--improve-margin", type=float, default=0.0)
    parser.add_argument("--hurt-margin", type=float, default=0.0)
    parser.add_argument(
        "--accept-improve-mode",
        type=str,
        default="any",
        choices=["any", "both", "pareto"],
        help=(
            "How to mark a slot0 replacement as positive: any keeps the old ADE-or-FDE improvement label; "
            "both requires ADE and FDE to improve; pareto allows both to improve or one to stay flat while "
            "the other improves strongly."
        ),
    )
    parser.add_argument(
        "--accept-flat-tolerance",
        type=float,
        default=0.0,
        help="Pareto accept mode: maximum allowed non-improving ADE/FDE delta for the metric that stays flat.",
    )
    parser.add_argument(
        "--accept-strong-improve-margin",
        type=float,
        default=None,
        help="Pareto accept mode: required improvement for the metric that changes strongly. Defaults to improve-margin.",
    )
    parser.add_argument(
        "--accept-require-improve-slow",
        action="store_true",
        help="Positive accept labels must also improve over the slow/base trajectory score.",
    )
    parser.add_argument("--accept-slow-improve-margin", type=float, default=0.0)
    parser.add_argument("--include-index-features", action="store_true")

    parser.add_argument("--max-samples-per-class", type=int, default=250000)
    parser.add_argument("--max-val-samples-per-class", type=int, default=80000)
    parser.add_argument("--max-groups", type=int, default=80000)
    parser.add_argument("--max-val-groups", type=int, default=30000)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="v58k_slot_quality")
    parser.add_argument("--output-checkpoint", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item]
    if not values:
        raise SystemExit("Expected at least one integer")
    return values


def _candidate_ade_fde_slots(candidates: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _score_from_ade_fde(ade: torch.Tensor, fde: torch.Tensor, *, metric: str) -> torch.Tensor:
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return ade + fde
    raise ValueError(f"Unsupported rank label metric: {metric!r}")


def _improvement_vs_slot0(
    dade: torch.Tensor,
    dfde: torch.Tensor,
    *,
    improve_margin: float,
    improve_mode: str,
    flat_tolerance: float = 0.0,
    strong_improve_margin: Optional[float] = None,
) -> torch.Tensor:
    margin = float(improve_margin)
    mode = str(improve_mode)
    if mode == "any":
        return (dade < -margin - EPS) | (dfde < -margin - EPS)
    if mode == "both":
        return (dade < -margin - EPS) & (dfde < -margin - EPS)
    if mode == "pareto":
        strong_margin = margin if strong_improve_margin is None else float(strong_improve_margin)
        flat = float(flat_tolerance)
        both_improve = (dade < -margin - EPS) & (dfde < -margin - EPS)
        ade_flat_fde_strong = (dade <= flat + EPS) & (dfde < -strong_margin - EPS)
        fde_flat_ade_strong = (dfde <= flat + EPS) & (dade < -strong_margin - EPS)
        return both_improve | ade_flat_fde_strong | fde_flat_ade_strong
    raise ValueError(f"Unsupported accept improve mode: {mode!r}")


def _labels_vs_slot0(
    candidates: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    improve_margin: float,
    hurt_margin: float,
    improve_mode: str = "any",
    flat_tolerance: float = 0.0,
    strong_improve_margin: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ade, fde = _candidate_ade_fde_slots(candidates, ground_truth)
    slot0_ade = ade[:, 0:1]
    slot0_fde = fde[:, 0:1]
    dade = ade - slot0_ade
    dfde = fde - slot0_fde
    safe = (dade <= float(hurt_margin) + EPS) & (dfde <= float(hurt_margin) + EPS)
    improves = _improvement_vs_slot0(
        dade,
        dfde,
        improve_margin=float(improve_margin),
        improve_mode=str(improve_mode),
        flat_tolerance=float(flat_tolerance),
        strong_improve_margin=strong_improve_margin,
    )
    good = safe & improves
    bad = (dade > float(hurt_margin) + EPS) | (dfde > float(hurt_margin) + EPS)
    return good, bad, dade, dfde


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


def _classification_metrics(model: V58SlotQualityScorer, x: torch.Tensor, y: torch.Tensor, device: str) -> Dict[str, Any]:
    if int(x.shape[0]) <= 0:
        return {"available": False}
    model.eval()
    probs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), 65536):
            batch = x[start : start + 65536].to(device=device)
            probs.append(torch.sigmoid(model(batch)).detach().cpu())
    score = torch.cat(probs, dim=0)
    labels = y.cpu().to(dtype=torch.float32)
    pred = score >= 0.5
    positive = labels > 0.5
    return {
        "available": True,
        "samples": int(labels.numel()),
        "positive_rate": float(positive.to(dtype=torch.float32).mean().item()),
        "auc": _auc(score, labels),
        "ap": _average_precision(score, labels),
        "accuracy_at_0p5": float((pred == positive).to(dtype=torch.float32).mean().item()),
        "mean_prob_positive": float(score[positive].mean().item()) if bool(positive.any().item()) else None,
        "mean_prob_negative": float(score[~positive].mean().item()) if bool((~positive).any().item()) else None,
    }


class BalancedFeatureBuffer:
    def __init__(self, max_per_class: int, seed: int) -> None:
        self.max_per_class = int(max_per_class)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.pos: List[torch.Tensor] = []
        self.neg: List[torch.Tensor] = []
        self.total_valid = 0
        self.total_good = 0
        self.total_bad = 0
        self.total_neutral = 0
        self.feature_dim: Optional[int] = None

    def _append(self, store: List[torch.Tensor], x: torch.Tensor) -> None:
        current = sum(int(item.shape[0]) for item in store)
        remaining = max(self.max_per_class - current, 0)
        if remaining <= 0 or int(x.shape[0]) <= 0:
            return
        take = min(remaining, int(x.shape[0]))
        perm = torch.randperm(int(x.shape[0]), generator=self.generator)[:take]
        store.append(x.detach().cpu()[perm])

    def add(self, features: torch.Tensor, good: torch.Tensor, bad: torch.Tensor, mask: torch.Tensor) -> None:
        flat_mask = mask.reshape(-1).bool()
        if int(flat_mask.sum().item()) <= 0:
            return
        x = features.reshape(-1, int(features.shape[-1]))[flat_mask].to(dtype=torch.float32)
        good_v = good.reshape(-1)[flat_mask].bool()
        bad_v = bad.reshape(-1)[flat_mask].bool()
        self.feature_dim = int(x.shape[-1])
        self.total_valid += int(flat_mask.sum().item())
        self.total_good += int(good_v.sum().item())
        self.total_bad += int(bad_v.sum().item())
        self.total_neutral += int((~(good_v | bad_v)).sum().item())
        self._append(self.pos, x[good_v])
        self._append(self.neg, x[bad_v])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.pos or not self.neg:
            raise RuntimeError("Both positive and negative samples are required")
        x_pos = torch.cat(self.pos, dim=0)
        x_neg = torch.cat(self.neg, dim=0)
        y_pos = torch.ones(int(x_pos.shape[0]), dtype=torch.float32)
        y_neg = torch.zeros(int(x_neg.shape[0]), dtype=torch.float32)
        return torch.cat([x_pos, x_neg], dim=0), torch.cat([y_pos, y_neg], dim=0)

    def summary(self) -> Dict[str, Any]:
        denominator = max(int(self.total_valid), 1)
        pos_kept = sum(int(item.shape[0]) for item in self.pos)
        neg_kept = sum(int(item.shape[0]) for item in self.neg)
        return {
            "total_valid": int(self.total_valid),
            "total_good": int(self.total_good),
            "total_bad": int(self.total_bad),
            "total_neutral": int(self.total_neutral),
            "good_ratio": float(self.total_good / denominator),
            "bad_ratio": float(self.total_bad / denominator),
            "neutral_ratio": float(self.total_neutral / denominator),
            "kept_positive": int(pos_kept),
            "kept_negative": int(neg_kept),
            "feature_dim": self.feature_dim,
        }


class TwoStageFeatureBuffer:
    """Stores per-base groups for rank-then-accept training."""

    def __init__(self, max_groups: int, seed: int) -> None:
        self.max_groups = int(max_groups)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.features: List[torch.Tensor] = []
        self.rank_targets: List[torch.Tensor] = []
        self.accept_labels: List[torch.Tensor] = []
        self.total_valid_groups = 0
        self.total_accept_positive = 0
        self.total_accept_negative = 0
        self.kept_groups = 0
        self.num_slots: Optional[int] = None
        self.feature_dim: Optional[int] = None

    def add(self, features: torch.Tensor, score: torch.Tensor, accept: torch.Tensor, valid: torch.Tensor) -> None:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,S,K,A,F], got {tuple(features.shape)}")
        if score.ndim != 4 or accept.ndim != 4:
            raise ValueError("score/accept must be [B,S,K,A]")
        group_features = features.permute(0, 2, 3, 1, 4).reshape(-1, int(features.shape[1]), int(features.shape[-1]))
        group_score = score.permute(0, 2, 3, 1).reshape(-1, int(score.shape[1]))
        group_accept = accept.permute(0, 2, 3, 1).reshape(-1, int(accept.shape[1]))
        group_valid = valid.reshape(-1).bool()
        if int(group_valid.sum().item()) <= 0:
            return
        group_features = group_features[group_valid].to(dtype=torch.float32)
        group_score = group_score[group_valid]
        group_accept = group_accept[group_valid].to(dtype=torch.float32)
        rank_target = group_score.argmin(dim=1).to(dtype=torch.long)

        self.total_valid_groups += int(group_features.shape[0])
        self.total_accept_positive += int((group_accept > 0.5).sum().item())
        self.total_accept_negative += int((group_accept <= 0.5).sum().item())
        self.num_slots = int(group_features.shape[1])
        self.feature_dim = int(group_features.shape[-1])

        remaining = max(self.max_groups - self.kept_groups, 0)
        if remaining <= 0:
            return
        take = min(remaining, int(group_features.shape[0]))
        perm = torch.randperm(int(group_features.shape[0]), generator=self.generator)[:take]
        self.features.append(group_features[perm])
        self.rank_targets.append(rank_target[perm].cpu())
        self.accept_labels.append(group_accept[perm].cpu())
        self.kept_groups += int(take)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.features:
            raise RuntimeError("No two-stage groups were collected")
        return torch.cat(self.features, dim=0), torch.cat(self.rank_targets, dim=0), torch.cat(self.accept_labels, dim=0)

    def summary(self) -> Dict[str, Any]:
        total_accept = max(int(self.total_accept_positive + self.total_accept_negative), 1)
        return {
            "total_valid_groups": int(self.total_valid_groups),
            "kept_groups": int(self.kept_groups),
            "num_slots": self.num_slots,
            "feature_dim": self.feature_dim,
            "accept_positive": int(self.total_accept_positive),
            "accept_negative": int(self.total_accept_negative),
            "accept_positive_ratio": float(self.total_accept_positive / total_accept),
        }


def _slot_mask(slot_ids: torch.Tensor, selected_slots: Sequence[int]) -> torch.Tensor:
    selected = torch.tensor([int(item) for item in selected_slots], device=slot_ids.device, dtype=torch.long)
    return (slot_ids[:, None] == selected[None, :]).any(dim=1)


def _collect_split(
    args: argparse.Namespace,
    *,
    split: str,
    protocol_settings: Any,
    device: str,
    slow_predictor: MoFlowSlowPredictor,
    refiner: Any,
    normalization_stats: Mapping[str, Any],
    data_root: Path,
    max_scenes: Optional[int],
    max_per_class: int,
    train_slots: Sequence[int],
    seed_offset: int,
) -> tuple[BalancedFeatureBuffer, List[str]]:
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, max_scenes)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    collector = BalancedFeatureBuffer(max_per_class=max_per_class, seed=int(args.seed) + int(seed_offset))
    feature_names: List[str] = []
    print(
        "[train_v58_slot_quality_scorer] "
        f"collect split={split} scenes={len(selected_samples)} eval_items={selected_eval_items}"
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
            slot_ids = torch.arange(int(refined.shape[1]), device=refined.device, dtype=torch.long)
            features = build_v58_slot_quality_features(
                refined,
                base_trajectory=slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                candidate_slot_ids=slot_ids,
                max_slot_id=int(args.residual_slots) - 1,
                include_index_features=bool(args.include_index_features),
            )
            if not feature_names:
                feature_names = v58_slot_quality_feature_names(
                    future_frames=int(refined.shape[-2]),
                    coord_dim=int(refined.shape[-1]),
                    past_frames=int(batch["past_traj_original_scale"].shape[-2]),
                    past_feature_dim=int(batch["past_traj_original_scale"].shape[-1]),
                    temporal_energy_dim=int(temporal_energy.shape[-1]),
                    include_index_features=bool(args.include_index_features),
                )
                if len(feature_names) != int(features.shape[-1]):
                    feature_names = [f"feature_{index:04d}" for index in range(int(features.shape[-1]))]
            ground_truth = batch["fut_traj_original_scale"].to(device=device)
            good, bad, _dade, _dfde = _labels_vs_slot0(
                refined,
                ground_truth,
                improve_margin=float(args.improve_margin),
                hurt_margin=float(args.hurt_margin),
                improve_mode=str(args.accept_improve_mode),
                flat_tolerance=float(args.accept_flat_tolerance),
                strong_improve_margin=args.accept_strong_improve_margin,
            )
            agent_mask = batch["agent_mask"].to(device=device).bool()
            selected_slot_mask = _slot_mask(slot_ids, train_slots)[None, :, None, None]
            valid = agent_mask[:, None, None, :].expand_as(good) & selected_slot_mask
            collector.add(features, good, bad, valid)
        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0:
            summary = collector.summary()
            print(
                "[train_v58_slot_quality_scorer] "
                f"split={split} chunks={chunk_index}/{len(chunks)} "
                f"kept_pos={summary['kept_positive']} kept_neg={summary['kept_negative']} "
                f"good={summary['good_ratio']:.4f} bad={summary['bad_ratio']:.4f}"
            )
    return collector, feature_names


def _collect_two_stage_split(
    args: argparse.Namespace,
    *,
    split: str,
    protocol_settings: Any,
    device: str,
    slow_predictor: MoFlowSlowPredictor,
    refiner: Any,
    normalization_stats: Mapping[str, Any],
    data_root: Path,
    max_scenes: Optional[int],
    max_groups: int,
    train_slots: Sequence[int],
    seed_offset: int,
) -> tuple[TwoStageFeatureBuffer, List[str]]:
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    selected_samples = _select_samples(dataset, max_scenes)
    selected_eval_items = _count_selected_eval_items(selected_samples, args.sample_mode)
    chunks = list(_iter_chunks(list(enumerate(selected_samples)), args.batch_scenes))
    collector = TwoStageFeatureBuffer(max_groups=max_groups, seed=int(args.seed) + int(seed_offset))
    feature_names: List[str] = []
    print(
        "[train_v58_slot_quality_scorer] "
        f"collect_two_stage split={split} scenes={len(selected_samples)} eval_items={selected_eval_items}"
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
            slot_ids = torch.arange(int(refined.shape[1]), device=refined.device, dtype=torch.long)
            train_slot_tensor = torch.tensor(list(train_slots), device=refined.device, dtype=torch.long)
            feature_slot_tensor = torch.cat(
                [torch.zeros(1, device=refined.device, dtype=torch.long), train_slot_tensor],
                dim=0,
            )
            feature_candidates = refined.index_select(dim=1, index=feature_slot_tensor)
            features_all = build_v58_slot_quality_features(
                feature_candidates,
                base_trajectory=slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                candidate_slot_ids=feature_slot_tensor,
                max_slot_id=int(args.residual_slots) - 1,
                include_index_features=bool(args.include_index_features),
            )
            features = features_all[:, 1:]
            if not feature_names:
                feature_names = v58_slot_quality_feature_names(
                    future_frames=int(refined.shape[-2]),
                    coord_dim=int(refined.shape[-1]),
                    past_frames=int(batch["past_traj_original_scale"].shape[-2]),
                    past_feature_dim=int(batch["past_traj_original_scale"].shape[-1]),
                    temporal_energy_dim=int(temporal_energy.shape[-1]),
                    include_index_features=bool(args.include_index_features),
                )
                if len(feature_names) != int(features.shape[-1]):
                    feature_names = [f"feature_{index:04d}" for index in range(int(features.shape[-1]))]
            ground_truth = batch["fut_traj_original_scale"].to(device=device)
            full_ade, full_fde = _candidate_ade_fde_slots(refined, ground_truth)
            slot_ade = full_ade.index_select(dim=1, index=train_slot_tensor)
            slot_fde = full_fde.index_select(dim=1, index=train_slot_tensor)
            score = _score_from_ade_fde(slot_ade, slot_fde, metric=str(args.rank_label_metric))
            slot0_ade = full_ade[:, 0:1]
            slot0_fde = full_fde[:, 0:1]
            dade = slot_ade - slot0_ade
            dfde = slot_fde - slot0_fde
            safe = (dade <= float(args.hurt_margin) + EPS) & (dfde <= float(args.hurt_margin) + EPS)
            improves_slot0 = _improvement_vs_slot0(
                dade,
                dfde,
                improve_margin=float(args.improve_margin),
                improve_mode=str(args.accept_improve_mode),
                flat_tolerance=float(args.accept_flat_tolerance),
                strong_improve_margin=args.accept_strong_improve_margin,
            )
            accept = safe & improves_slot0
            if bool(args.accept_require_improve_slow):
                base_ade, base_fde = _candidate_ade_fde_slots(slow_output.slow_pred[:, None, ...], ground_truth)
                base_score = _score_from_ade_fde(
                    base_ade.squeeze(1),
                    base_fde.squeeze(1),
                    metric=str(args.rank_label_metric),
                )
                accept = accept & (score < (base_score[:, None, :, :] - float(args.accept_slow_improve_margin) - EPS))
            valid = batch["agent_mask"].to(device=device).bool()[:, None, :].expand(
                int(refined.shape[0]),
                int(refined.shape[2]),
                int(refined.shape[3]),
            )
            collector.add(features, score, accept, valid)
        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0:
            summary = collector.summary()
            print(
                "[train_v58_slot_quality_scorer] "
                f"split={split} chunks={chunk_index}/{len(chunks)} kept_groups={summary['kept_groups']} "
                f"accept_pos={summary['accept_positive_ratio']:.4f}"
            )
    return collector, feature_names


def _train_model(
    args: argparse.Namespace,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: str,
) -> tuple[V58SlotQualityScorer, Dict[str, Any]]:
    order = torch.randperm(int(train_x.shape[0]))
    train_x = train_x[order]
    train_y = train_y[order]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    model = V58SlotQualityScorer(
        V58SlotQualityScorerConfig(
            feature_dim=int(train_x.shape[1]),
            hidden_dim=int(args.hidden_dim),
            layers=int(args.layers),
            dropout=float(args.dropout),
        )
    ).to(device)
    model.set_normalization(mean.to(device=device), std.to(device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_count = max(float((train_y > 0.5).sum().item()), 1.0)
    neg_count = max(float((train_y <= 0.5).sum().item()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], device=device, dtype=torch.float32)
    history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 3001)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_order = torch.randperm(int(train_x.shape[0]), generator=generator)
        loss_sum = 0.0
        loss_weight = 0
        for start in range(0, int(epoch_order.numel()), max(int(args.batch_size), 1)):
            index = epoch_order[start : start + max(int(args.batch_size), 1)]
            xb = train_x[index].to(device=device)
            yb = train_y[index].to(device=device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(index.numel())
            loss_weight += int(index.numel())
        train_metrics = _classification_metrics(model, train_x[: min(int(train_x.shape[0]), 200000)], train_y[: min(int(train_y.shape[0]), 200000)], device)
        val_metrics = _classification_metrics(model, val_x, val_y, device)
        epoch_row = {
            "epoch": int(epoch),
            "loss": float(loss_sum / max(loss_weight, 1)),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_row)
        val_auc = val_metrics.get("auc")
        metric = float(val_auc) if val_auc is not None else -1.0
        if metric > best_metric:
            best_metric = metric
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
        print(
            "[train_v58_slot_quality_scorer] "
            f"epoch={epoch}/{args.epochs} loss={epoch_row['loss']:.6f} "
            f"train_auc={train_metrics.get('auc')} val_auc={val_metrics.get('auc')} "
            f"val_ap={val_metrics.get('ap')}"
        )
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_metric),
        "final_train": _classification_metrics(model, train_x, train_y, device),
        "final_val": _classification_metrics(model, val_x, val_y, device),
        "history": history,
    }
    return model, metrics


def _two_stage_metrics(
    model: V58SlotQualityScorer,
    x: torch.Tensor,
    rank_target: torch.Tensor,
    accept_y: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    if int(x.shape[0]) <= 0:
        return {"available": False}
    model.eval()
    rank_logits_list: List[torch.Tensor] = []
    accept_logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), 32768):
            xb = x[start : start + 32768].to(device=device)
            output = model(xb)
            rank_logits_list.append(output[..., int(model.config.rank_head_index)].detach().cpu())
            accept_logits_list.append(output[..., int(model.config.accept_head_index)].detach().cpu())
    rank_logits = torch.cat(rank_logits_list, dim=0)
    accept_logits = torch.cat(accept_logits_list, dim=0)
    rank_pred = rank_logits.argmax(dim=1)
    rank_target_cpu = rank_target.cpu().to(dtype=torch.long)
    accept_probs = torch.sigmoid(accept_logits).reshape(-1)
    accept_labels = accept_y.cpu().to(dtype=torch.float32).reshape(-1)
    selected_accept = torch.gather(torch.sigmoid(accept_logits), dim=1, index=rank_pred[:, None]).squeeze(1)
    selected_accept_label = torch.gather(accept_y.cpu().to(dtype=torch.float32), dim=1, index=rank_pred[:, None]).squeeze(1)
    return {
        "available": True,
        "groups": int(x.shape[0]),
        "num_slots": int(x.shape[1]),
        "rank_accuracy": float((rank_pred == rank_target_cpu).to(dtype=torch.float32).mean().item()),
        "accept_positive_rate": float((accept_labels > 0.5).to(dtype=torch.float32).mean().item()),
        "accept_auc": _auc(accept_probs, accept_labels),
        "accept_ap": _average_precision(accept_probs, accept_labels),
        "accept_accuracy_at_0p5": float(((accept_probs >= 0.5) == (accept_labels > 0.5)).to(dtype=torch.float32).mean().item()),
        "rank_selected_accept_positive_rate": float((selected_accept_label > 0.5).to(dtype=torch.float32).mean().item()),
        "rank_selected_accept_prob_mean": float(selected_accept.mean().item()),
    }


def _train_two_stage_model(
    args: argparse.Namespace,
    *,
    train_x: torch.Tensor,
    train_rank: torch.Tensor,
    train_accept: torch.Tensor,
    val_x: torch.Tensor,
    val_rank: torch.Tensor,
    val_accept: torch.Tensor,
    device: str,
) -> tuple[V58SlotQualityScorer, Dict[str, Any]]:
    order = torch.randperm(int(train_x.shape[0]))
    train_x = train_x[order]
    train_rank = train_rank[order]
    train_accept = train_accept[order]
    flat_train = train_x.reshape(-1, int(train_x.shape[-1]))
    mean = flat_train.mean(dim=0)
    std = flat_train.std(dim=0).clamp_min(1e-6)
    model = V58SlotQualityScorer(
        V58SlotQualityScorerConfig(
            feature_dim=int(train_x.shape[-1]),
            hidden_dim=int(args.hidden_dim),
            layers=int(args.layers),
            dropout=float(args.dropout),
            output_dim=2,
            rank_head_index=0,
            accept_head_index=1,
        )
    ).to(device)
    model.set_normalization(mean.to(device=device), std.to(device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_count = max(float((train_accept > 0.5).sum().item()), 1.0)
    neg_count = max(float((train_accept <= 0.5).sum().item()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], device=device, dtype=torch.float32)
    history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 4001)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_order = torch.randperm(int(train_x.shape[0]), generator=generator)
        loss_sum = 0.0
        loss_weight = 0
        for start in range(0, int(epoch_order.numel()), max(int(args.batch_size), 1)):
            index = epoch_order[start : start + max(int(args.batch_size), 1)]
            xb = train_x[index].to(device=device)
            rank_y = train_rank[index].to(device=device)
            accept_y = train_accept[index].to(device=device)
            output = model(xb)
            rank_logits = output[..., 0]
            accept_logits = output[..., 1]
            rank_loss = F.cross_entropy(rank_logits, rank_y)
            accept_loss = F.binary_cross_entropy_with_logits(accept_logits, accept_y, pos_weight=pos_weight)
            loss = float(args.lambda_rank_ce) * rank_loss + float(args.lambda_accept_bce) * accept_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(index.numel())
            loss_weight += int(index.numel())
        train_limit = min(int(train_x.shape[0]), 100000)
        train_metrics = _two_stage_metrics(
            model,
            train_x[:train_limit],
            train_rank[:train_limit],
            train_accept[:train_limit],
            device,
        )
        val_metrics = _two_stage_metrics(model, val_x, val_rank, val_accept, device)
        epoch_row = {
            "epoch": int(epoch),
            "loss": float(loss_sum / max(loss_weight, 1)),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_row)
        val_auc = val_metrics.get("accept_auc")
        rank_acc = val_metrics.get("rank_accuracy")
        metric = (float(val_auc) if val_auc is not None else 0.0) + 0.25 * (
            float(rank_acc) if rank_acc is not None else 0.0
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
        print(
            "[train_v58_slot_quality_scorer] "
            f"epoch={epoch}/{args.epochs} loss={epoch_row['loss']:.6f} "
            f"train_rank_acc={train_metrics.get('rank_accuracy')} val_rank_acc={val_metrics.get('rank_accuracy')} "
            f"val_accept_auc={val_metrics.get('accept_auc')} val_accept_ap={val_metrics.get('accept_ap')}"
        )
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_metric),
        "final_train": _two_stage_metrics(model, train_x, train_rank, train_accept, device),
        "final_val": _two_stage_metrics(model, val_x, val_rank, val_accept, device),
        "history": history,
    }
    return model, metrics


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    train_slots = _split_ints(str(args.train_slots))
    if any(slot <= 0 for slot in train_slots):
        raise SystemExit("--train-slots should contain nonzero slots only; slot0 is the fallback reference")
    if int(args.residual_slots) <= max(train_slots):
        raise SystemExit("--residual-slots must be larger than every --train-slots entry")
    if float(args.improve_margin) < 0.0 or float(args.hurt_margin) < 0.0:
        raise SystemExit("--improve-margin and --hurt-margin must be non-negative")
    if float(args.accept_flat_tolerance) < 0.0:
        raise SystemExit("--accept-flat-tolerance must be non-negative")
    if args.accept_strong_improve_margin is not None and float(args.accept_strong_improve_margin) < 0.0:
        raise SystemExit("--accept-strong-improve-margin must be non-negative")
    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)

    train_dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=args.subset,
            split=args.train_split,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
        )
    )
    train_samples = _select_samples(train_dataset, args.max_scenes)
    agents = _infer_agents(train_samples, args.sample_mode, args.agents)
    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=str(args.slow_cfg_path),
            checkpoint_path=str(args.slow_checkpoint),
        )
    )
    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[slow_predictor],
        samples=train_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)
    refiner_variant = _checkpoint_variant(str(args.refiner_checkpoint))
    refiner = load_social_cvae_teacher_refiner(str(args.refiner_checkpoint), map_location=device).to(device)
    refiner.eval()
    if not bool(getattr(refiner.config, "use_set_generator", False)):
        raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")
    max_slots = int(getattr(refiner.config, "max_residual_slots", 1))
    if int(args.residual_slots) > max_slots:
        raise SystemExit(f"--residual-slots {args.residual_slots} exceeds checkpoint max_residual_slots={max_slots}")
    print(
        "[train_v58_slot_quality_scorer] "
        f"device={device} refiner={Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots} train_slots={train_slots}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[train_v58_slot_quality_scorer] warning: selected_samples normalization is diagnostic only")

    if str(args.training_mode) == "two_stage_replacement":
        train_collector, feature_names = _collect_two_stage_split(
            args,
            split=str(args.train_split),
            protocol_settings=protocol_settings,
            device=device,
            slow_predictor=slow_predictor,
            refiner=refiner,
            normalization_stats=normalization_stats,
            data_root=data_root,
            max_scenes=args.max_scenes,
            max_groups=int(args.max_groups),
            train_slots=train_slots,
            seed_offset=101,
        )
        val_collector, val_feature_names = _collect_two_stage_split(
            args,
            split=str(args.val_split),
            protocol_settings=protocol_settings,
            device=device,
            slow_predictor=slow_predictor,
            refiner=refiner,
            normalization_stats=normalization_stats,
            data_root=data_root,
            max_scenes=args.max_val_scenes,
            max_groups=int(args.max_val_groups),
            train_slots=train_slots,
            seed_offset=202,
        )
        train_x, train_rank, train_accept = train_collector.tensors()
        val_x, val_rank, val_accept = val_collector.tensors()
        if int(train_x.shape[-1]) != int(val_x.shape[-1]):
            raise RuntimeError(f"Feature dim mismatch train={train_x.shape[-1]} val={val_x.shape[-1]}")
        if val_feature_names and len(val_feature_names) == int(train_x.shape[-1]):
            feature_names = val_feature_names if not feature_names else feature_names
        model, train_metrics = _train_two_stage_model(
            args,
            train_x=train_x,
            train_rank=train_rank,
            train_accept=train_accept,
            val_x=val_x,
            val_rank=val_rank,
            val_accept=val_accept,
            device=device,
        )
        feature_dim = int(train_x.shape[-1])
    else:
        train_collector, feature_names = _collect_split(
            args,
            split=str(args.train_split),
            protocol_settings=protocol_settings,
            device=device,
            slow_predictor=slow_predictor,
            refiner=refiner,
            normalization_stats=normalization_stats,
            data_root=data_root,
            max_scenes=args.max_scenes,
            max_per_class=int(args.max_samples_per_class),
            train_slots=train_slots,
            seed_offset=101,
        )
        val_collector, val_feature_names = _collect_split(
            args,
            split=str(args.val_split),
            protocol_settings=protocol_settings,
            device=device,
            slow_predictor=slow_predictor,
            refiner=refiner,
            normalization_stats=normalization_stats,
            data_root=data_root,
            max_scenes=args.max_val_scenes,
            max_per_class=int(args.max_val_samples_per_class),
            train_slots=train_slots,
            seed_offset=202,
        )
        train_x, train_y = train_collector.tensors()
        val_x, val_y = val_collector.tensors()
        if int(train_x.shape[1]) != int(val_x.shape[1]):
            raise RuntimeError(f"Feature dim mismatch train={train_x.shape[1]} val={val_x.shape[1]}")
        if val_feature_names and len(val_feature_names) == int(train_x.shape[1]):
            feature_names = val_feature_names if not feature_names else feature_names
        model, train_metrics = _train_model(
            args,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            device=device,
        )
        feature_dim = int(train_x.shape[1])

    output_path = (
        Path(str(args.output_checkpoint)).expanduser().resolve()
        if args.output_checkpoint
        else Path(str(args.output_dir)).expanduser().resolve() / str(args.run_name) / "v58_slot_quality_scorer_best.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_v58_slot_quality_scorer",
            "variant": "v58k_slot_quality_scorer",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "residual_slots": int(args.residual_slots),
            "train_slots": list(train_slots),
            "training_mode": str(args.training_mode),
            "rank_label_metric": str(args.rank_label_metric),
            "accept_improve_mode": str(args.accept_improve_mode),
            "accept_flat_tolerance": float(args.accept_flat_tolerance),
            "accept_strong_improve_margin": (
                None if args.accept_strong_improve_margin is None else float(args.accept_strong_improve_margin)
            ),
            "accept_require_improve_slow": bool(args.accept_require_improve_slow),
            "refiner_variant": refiner_variant,
            "feature_dim": int(feature_dim),
            "include_index_features": bool(args.include_index_features),
        },
        "config": model.config_dict,
        "model_state_dict": model.state_dict(),
        "feature_names": list(feature_names),
        "feature_mean": model.feature_mean.detach().cpu(),
        "feature_std": model.feature_std.detach().cpu(),
        "args": _coerce_jsonable(vars(args)),
        "dataset": {
            "data_root": data_root.as_posix(),
            "train": train_collector.summary(),
            "val": val_collector.summary(),
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(str(args.slow_checkpoint)).expanduser().resolve().as_posix(),
        "refiner_checkpoint": Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix(),
        "metrics": _coerce_jsonable(train_metrics),
    }
    torch.save(payload, output_path)
    metrics_path = output_path.with_suffix(".json")
    metrics_payload = dict(payload)
    metrics_payload.pop("model_state_dict", None)
    metrics_payload.pop("feature_mean", None)
    metrics_payload.pop("feature_std", None)
    metrics_path.write_text(json.dumps(_coerce_jsonable(metrics_payload), indent=2, ensure_ascii=False), encoding="utf-8")
    final_val = train_metrics.get("final_val", {})
    print(
        "[train_v58_slot_quality_scorer] done "
        f"best_epoch={train_metrics.get('best_epoch')} val_auc={final_val.get('auc')} val_ap={final_val.get('ap')}"
    )
    print(f"checkpoint={output_path.as_posix()}")
    print(f"metrics_json={metrics_path.as_posix()}")


if __name__ == "__main__":
    main()
