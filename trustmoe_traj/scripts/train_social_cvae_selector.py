"""Train SocialCVAE group-wise residual selectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trustmoe_traj.models import (
    SocialCVAEGroupSelector,
    SocialCVAEGroupSelectorConfig,
    TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM,
    compute_interaction_energy_features,
    compute_temporal_interaction_energy_features,
    compute_trajectory_aware_interaction_summary_features,
    load_social_cvae_teacher_refiner,
)
from trustmoe_traj.scripts.train_social_cvae_refiner import (
    DEFAULT_OUTPUT_DIR as REFINER_OUTPUT_DIR,
    _mean_metrics,
    _prepare_refiner_tensors,
    _selection_score,
    _summarize_refinement,
)
from trustmoe_traj.scripts.train_student_integrated_finetune import (
    DEFAULT_CACHE_PATH,
    CacheDataset,
    _jsonable,
    _load_cache,
    _masked_mean,
    _move_batch,
    _resolve_device,
    _select_indices,
    _set_seed,
)


DEFAULT_OUTPUT_DIR = REFINER_OUTPUT_DIR.parent / "social_cvae_selector_models"

ADVANCED_SELECTOR_VARIANTS = {
    "v27a",
    "v28a",
    "v28b",
    "v29a",
    "v29b",
    "v29c",
    "v29d",
    "v30a",
    "v31a",
    "v32a",
    "v33a",
    "v34a",
    "v35a",
}
UTILITY_SELECTOR_VARIANTS = {
    "v28a",
    "v28b",
    "v29a",
    "v29b",
    "v29c",
    "v29d",
    "v30a",
    "v31a",
    "v32a",
    "v33a",
    "v34a",
    "v35a",
    "v58h",
    "v58i",
    "v58j",
}
SOFT_UTILITY_SELECTOR_VARIANTS = {
    "v28b",
    "v29a",
    "v29b",
    "v29c",
    "v29d",
    "v30a",
    "v31a",
    "v32a",
    "v33a",
    "v34a",
    "v35a",
}
CANDIDATE_ENERGY_SUMMARY_VARIANTS = {"v29b", "v29c", "v29d", "v30a", "v31a", "v32a", "v33a", "v34a", "v35a"}
TRAJECTORY_AWARE_CANDIDATE_SUMMARY_VARIANTS = {"v30a", "v31a", "v32a", "v33a", "v34a", "v35a"}
ENERGY_GATED_FUSION_VARIANTS = {"v31a"}
CANDIDATE_SAFETY_PENALTY_VARIANTS = {"v32a", "v33a", "v34a", "v35a", "v58i", "v58j"}
RESIDUAL_ACCEPT_GATE_VARIANTS = {"v33a", "v58i", "v58j"}
BASE_BEST_GUARD_VARIANTS = {"v34a", "v35a"}
FDE_HURT_CANDIDATE_SAFETY_VARIANTS = {"v35a", "v58j"}
PER_BASE_SLOT_SELECTOR_VARIANTS = {"v55a", "v57b", "v57c", "v58g", "v58h", "v58i", "v58j"}
CONSERVATIVE_SLOT0_SELECTOR_VARIANTS = {"v57c", "v58i", "v58j"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SocialCVAE group-wise residual selector.")
    parser.add_argument(
        "--variant",
        type=str,
        default="v25a",
        choices=[
            "v25a",
            "v25b",
            "v25c",
            "v27a",
            "v28a",
            "v28b",
            "v29a",
            "v29b",
            "v29c",
            "v29d",
            "v30a",
            "v31a",
            "v32a",
            "v33a",
            "v34a",
            "v35a",
            "v55a",
            "v57b",
            "v57c",
            "v58g",
            "v58h",
            "v58i",
            "v58j",
        ],
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="social_cvae_selector")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--allow-energy-fallback", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--no-mode-embedding", action="store_true")
    parser.add_argument("--use-energy-risk-map", action="store_true")
    parser.add_argument("--energy-risk-distance-scale", type=float, default=0.5)
    parser.add_argument("--use-temporal-energy-encoder", action="store_true")
    parser.add_argument("--energy-temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--use-mean-candidate-comparison", action="store_true")
    parser.add_argument("--use-candidate-energy-context", action="store_true")
    parser.add_argument("--use-candidate-energy-summary-context", action="store_true")
    parser.add_argument(
        "--use-observable-feature-context",
        action="store_true",
        help="Concatenate diagnostic-style observable scalar features into the selector MLP.",
    )
    parser.add_argument("--residual-samples", type=int, default=10)
    parser.add_argument("--candidate-z-mode", type=str, default="sample", choices=["sample", "slots"])
    parser.add_argument(
        "--candidate-slot-start",
        type=int,
        default=0,
        help="For z_mode=slots, train on a contiguous slot window starting at this slot index.",
    )
    parser.add_argument("--include-mean-candidate", action="store_true")
    parser.add_argument("--label-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument(
        "--target-fallback-to-mean",
        action="store_true",
        help="Use mean-candidate labels unless the best sampled residual improves over z-mean by a margin.",
    )
    parser.add_argument(
        "--target-fallback-to-slot0",
        action="store_true",
        help="Use slot0 labels unless the best semantic slot improves over slot0 by a margin.",
    )
    parser.add_argument("--target-improvement-margin", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-expected-score", type=float, default=0.1)
    parser.add_argument("--lambda-margin-ranking", type=float, default=0.0)
    parser.add_argument("--selector-logit-margin", type=float, default=0.5)
    parser.add_argument("--lambda-soft-utility", type=float, default=0.0)
    parser.add_argument("--utility-soft-temperature", type=float, default=0.25)
    parser.add_argument("--lambda-pairwise-utility", type=float, default=0.0)
    parser.add_argument("--pairwise-utility-min-gap", type=float, default=0.05)
    parser.add_argument("--pairwise-utility-max-weight", type=float, default=2.0)
    parser.add_argument("--confidence-fallback-to-mean", action="store_true")
    parser.add_argument("--confidence-fallback-to-slot0", action="store_true")
    parser.add_argument("--fallback-prob-margin", type=float, default=0.05)
    parser.add_argument("--fallback-min-selected-prob", type=float, default=0.35)
    parser.add_argument("--utility-ade-weight", type=float, default=0.0)
    parser.add_argument("--utility-fde-weight", type=float, default=1.0)
    parser.add_argument("--utility-miss-penalty", type=float, default=0.0)
    parser.add_argument("--utility-mean-hurt-weight", type=float, default=0.0)
    parser.add_argument("--utility-mean-hurt-margin", type=float, default=0.0)
    parser.add_argument("--utility-base-best-hurt-weight", type=float, default=0.0)
    parser.add_argument("--utility-base-best-hurt-margin", type=float, default=0.0)
    parser.add_argument("--utility-slow-fde-hurt-weight", type=float, default=0.0)
    parser.add_argument("--utility-slow-fde-hurt-margin", type=float, default=0.0)
    parser.add_argument("--utility-slow-ade-hurt-weight", type=float, default=0.0)
    parser.add_argument("--utility-slow-ade-hurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-candidate-safety", type=float, default=0.0)
    parser.add_argument("--candidate-safety-margin", type=float, default=0.0)
    parser.add_argument("--lambda-residual-accept", type=float, default=0.0)
    parser.add_argument("--lambda-base-best-guard", type=float, default=0.0)
    parser.add_argument(
        "--top-base-rank-weights",
        type=str,
        default="",
        help="Comma-separated per-base-rank loss weights, e.g. 8,4,2,1.5,1. Ranks beyond the list use 1.",
    )
    parser.add_argument("--top-base-weight-metric", type=str, default="fde", choices=["fde", "ade_fde"])

    parser.add_argument("--selection-metric", type=str, default="safety", choices=["fde_min", "safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-nohurt-weight", type=float, default=2.0)
    parser.add_argument("--selection-diversity-weight", type=float, default=0.5)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def _variant_name(_args: argparse.Namespace) -> str:
    if str(_args.variant) in {"v58i", "v58j"}:
        samples_tag = f"slots{int(_args.residual_samples)}"
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        hurt_tag = f"hurt{int(round(float(_args.utility_slow_fde_hurt_weight) * 100)):03d}"
        rank_tag = "rank" + str(_args.top_base_rank_weights).replace(",", "x").replace(".", "p")
        if str(_args.variant) == "v58j":
            gate_tag = (
                f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
                f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
            )
            return (
                "v58j_slot0_front2_conservative_gate_"
                f"{samples_tag}_{margin_tag}_{ade_tag}_{fde_tag}_{hurt_tag}_{rank_tag}_{gate_tag}"
            )
        return f"v58i_slot0_front2_observable_gate_{samples_tag}_{margin_tag}_{ade_tag}_{fde_tag}_{hurt_tag}_{rank_tag}"
    if str(_args.variant) == "v58h":
        start_tag = f"start{int(_args.candidate_slot_start)}"
        samples_tag = f"slots{int(_args.residual_samples)}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        hurt_tag = f"hurt{int(round(float(_args.utility_slow_fde_hurt_weight) * 100)):03d}"
        return f"v58h_front_slot_utility_selector_{start_tag}_{samples_tag}_{ade_tag}_{fde_tag}_{miss_tag}_{hurt_tag}"
    if str(_args.variant) == "v58g":
        start_tag = f"start{int(_args.candidate_slot_start)}"
        return f"v58g_front_slot_selector_{start_tag}_slots{int(_args.residual_samples)}"
    if str(_args.variant) == "v57c":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        return f"v57c_conservative_semantic_slot_selector_slots{int(_args.residual_samples)}_{margin_tag}_{gate_tag}"
    if str(_args.variant) == "v57b":
        return f"v57b_per_base_semantic_slot_selector_slots{int(_args.residual_samples)}"
    if str(_args.variant) == "v55a":
        return f"v55a_quality_aware_per_base_slot_selector_slots{int(_args.residual_samples)}"
    if str(_args.variant) == "v35a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        safety_tag = f"safe{int(round(float(_args.lambda_candidate_safety) * 100)):03d}"
        guard_tag = f"guard{int(round(float(_args.lambda_base_best_guard) * 100)):03d}"
        return (
            "v35a_social_cvae_fdehurt_guard_safety_trajaware_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_"
            f"{gate_tag}_{safety_tag}_{guard_tag}"
        )
    if str(_args.variant) == "v34a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        safety_tag = f"safe{int(round(float(_args.lambda_candidate_safety) * 100)):03d}"
        guard_tag = f"guard{int(round(float(_args.lambda_base_best_guard) * 100)):03d}"
        return (
            "v34a_social_cvae_basebest_guard_safety_trajaware_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_"
            f"{gate_tag}_{safety_tag}_{guard_tag}"
        )
    if str(_args.variant) == "v33a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        safety_tag = f"safe{int(round(float(_args.lambda_candidate_safety) * 100)):03d}"
        accept_tag = f"accept{int(round(float(_args.lambda_residual_accept) * 100)):03d}"
        return (
            "v33a_social_cvae_accept_gated_safety_trajaware_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_"
            f"{gate_tag}_{safety_tag}_{accept_tag}"
        )
    if str(_args.variant) == "v32a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        safety_tag = f"safe{int(round(float(_args.lambda_candidate_safety) * 100)):03d}"
        return (
            "v32a_social_cvae_safety_penalty_trajaware_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_{gate_tag}_{safety_tag}"
        )
    if str(_args.variant) == "v31a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        return (
            "v31a_social_cvae_energy_gated_trajaware_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_{gate_tag}"
        )
    if str(_args.variant) == "v30a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        return (
            "v30a_social_cvae_trajaware_energy_summary_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_{gate_tag}"
        )
    if str(_args.variant) == "v29d":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        ade_tag = f"ade{int(round(float(_args.utility_ade_weight) * 100)):03d}"
        fde_tag = f"fde{int(round(float(_args.utility_fde_weight) * 100)):03d}"
        miss_tag = f"miss{int(round(float(_args.utility_miss_penalty) * 100)):03d}"
        gate_tag = (
            f"gate{int(round(float(_args.fallback_prob_margin) * 100)):03d}"
            f"p{int(round(float(_args.fallback_min_selected_prob) * 100)):03d}"
        )
        return (
            "v29d_social_cvae_adefde_energy_utility_selector_"
            f"r{int(_args.residual_samples)}_{margin_tag}_{ade_tag}_{fde_tag}_{miss_tag}_{gate_tag}"
        )
    if str(_args.variant) == "v29c":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v29c_social_cvae_batched_candidate_energy_summary_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v29b":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v29b_social_cvae_candidate_energy_summary_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v29a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v29a_social_cvae_candidate_energy_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v28b":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v28b_social_cvae_soft_utility_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v28a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v28a_social_cvae_conservative_utility_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v27a":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v27a_social_cvae_full_latent_selector_r{int(_args.residual_samples)}_{margin_tag}"
    if str(_args.variant) == "v25c":
        margin_tag = f"m{int(round(float(_args.target_improvement_margin) * 1000)):03d}"
        return f"v25c_social_cvae_group_selector_r3_mean_fallback_{margin_tag}"
    if str(_args.variant) == "v25b":
        return "v25b_social_cvae_group_selector_mean_fallback"
    return "v25a_social_cvae_group_selector"


def _split_floats(raw: str) -> List[float]:
    return [float(item) for item in str(raw).replace(",", " ").split() if item]


def _validate_variant_args(args: argparse.Namespace) -> None:
    if int(args.energy_temporal_hidden_dim) <= 0:
        raise SystemExit("--energy-temporal-hidden-dim must be positive")
    if float(args.energy_risk_distance_scale) <= 0.0:
        raise SystemExit("--energy-risk-distance-scale must be positive")
    if float(args.utility_soft_temperature) <= 0.0:
        raise SystemExit("--utility-soft-temperature must be positive")
    if float(args.pairwise_utility_min_gap) < 0.0:
        raise SystemExit("--pairwise-utility-min-gap must be non-negative")
    if float(args.pairwise_utility_max_weight) <= 0.0:
        raise SystemExit("--pairwise-utility-max-weight must be positive")
    if float(args.fallback_prob_margin) < 0.0:
        raise SystemExit("--fallback-prob-margin must be non-negative")
    if not (0.0 <= float(args.fallback_min_selected_prob) <= 1.0):
        raise SystemExit("--fallback-min-selected-prob must be in [0, 1]")
    if bool(args.use_candidate_energy_context) and bool(args.use_candidate_energy_summary_context):
        raise SystemExit("--use-candidate-energy-context and --use-candidate-energy-summary-context are mutually exclusive")
    if bool(args.target_fallback_to_mean) and bool(args.target_fallback_to_slot0):
        raise SystemExit("--target-fallback-to-mean and --target-fallback-to-slot0 are mutually exclusive")
    if bool(args.confidence_fallback_to_mean) and bool(args.confidence_fallback_to_slot0):
        raise SystemExit("--confidence-fallback-to-mean and --confidence-fallback-to-slot0 are mutually exclusive")
    if int(args.candidate_slot_start) < 0:
        raise SystemExit("--candidate-slot-start must be non-negative")
    if int(args.candidate_slot_start) > 0 and str(args.candidate_z_mode) != "slots":
        raise SystemExit("--candidate-slot-start > 0 requires --candidate-z-mode slots")
    for name in (
        "utility_ade_weight",
        "utility_fde_weight",
        "utility_miss_penalty",
        "utility_mean_hurt_weight",
        "utility_mean_hurt_margin",
        "utility_base_best_hurt_weight",
        "utility_base_best_hurt_margin",
        "utility_slow_fde_hurt_weight",
        "utility_slow_fde_hurt_margin",
        "utility_slow_ade_hurt_weight",
        "utility_slow_ade_hurt_margin",
        "lambda_soft_utility",
        "lambda_pairwise_utility",
        "lambda_candidate_safety",
        "candidate_safety_margin",
        "lambda_residual_accept",
        "lambda_base_best_guard",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if str(args.variant) in ADVANCED_SELECTOR_VARIANTS:
        if int(args.residual_samples) < 5:
            raise SystemExit(f"--variant {args.variant} requires --residual-samples >= 5")
        if not bool(args.include_mean_candidate):
            raise SystemExit(f"--variant {args.variant} requires --include-mean-candidate")
        if not bool(args.target_fallback_to_mean):
            raise SystemExit(f"--variant {args.variant} requires --target-fallback-to-mean")
        if float(args.target_improvement_margin) < 0.10:
            raise SystemExit(f"--variant {args.variant} requires --target-improvement-margin >= 0.10")
        if not bool(args.confidence_fallback_to_mean):
            raise SystemExit(f"--variant {args.variant} requires --confidence-fallback-to-mean")
        if not bool(args.use_energy_risk_map):
            raise SystemExit(f"--variant {args.variant} requires --use-energy-risk-map")
        if str(args.variant) not in CANDIDATE_ENERGY_SUMMARY_VARIANTS and not bool(args.use_temporal_energy_encoder):
            raise SystemExit(f"--variant {args.variant} requires --use-temporal-energy-encoder")
        if not bool(args.use_mean_candidate_comparison):
            raise SystemExit(f"--variant {args.variant} requires --use-mean-candidate-comparison")
        if str(args.variant) in UTILITY_SELECTOR_VARIANTS:
            if float(args.utility_fde_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-fde-weight > 0")
            if float(args.utility_ade_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-ade-weight > 0")
            if float(args.utility_miss_penalty) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-miss-penalty > 0")
            if float(args.utility_mean_hurt_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-mean-hurt-weight > 0")
            if float(args.utility_base_best_hurt_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-base-best-hurt-weight > 0")
        if str(args.variant) in SOFT_UTILITY_SELECTOR_VARIANTS:
            if float(args.lambda_soft_utility) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --lambda-soft-utility > 0")
            if float(args.lambda_pairwise_utility) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --lambda-pairwise-utility > 0")
        if str(args.variant) == "v29a" and not bool(args.use_candidate_energy_context):
            raise SystemExit("--variant v29a requires --use-candidate-energy-context")
        if str(args.variant) in CANDIDATE_ENERGY_SUMMARY_VARIANTS and not bool(args.use_candidate_energy_summary_context):
            raise SystemExit(f"--variant {args.variant} requires --use-candidate-energy-summary-context")
        if str(args.variant) in CANDIDATE_SAFETY_PENALTY_VARIANTS and float(args.lambda_candidate_safety) <= 0.0:
            raise SystemExit(f"--variant {args.variant} requires --lambda-candidate-safety > 0")
        if str(args.variant) in RESIDUAL_ACCEPT_GATE_VARIANTS and float(args.lambda_residual_accept) <= 0.0:
            raise SystemExit(f"--variant {args.variant} requires --lambda-residual-accept > 0")
        if str(args.variant) in BASE_BEST_GUARD_VARIANTS and float(args.lambda_base_best_guard) <= 0.0:
            raise SystemExit(f"--variant {args.variant} requires --lambda-base-best-guard > 0")
        return
    if str(args.variant) in PER_BASE_SLOT_SELECTOR_VARIANTS:
        if str(args.candidate_z_mode) != "slots":
            raise SystemExit(f"--variant {args.variant} requires --candidate-z-mode slots")
        if bool(args.include_mean_candidate):
            raise SystemExit(
                f"--variant {args.variant} keeps exactly one residual slot per base; do not pass --include-mean-candidate"
            )
        if bool(args.target_fallback_to_mean):
            raise SystemExit(f"--variant {args.variant} does not support --target-fallback-to-mean")
        if int(args.residual_samples) <= 1:
            raise SystemExit(f"--variant {args.variant} requires --residual-samples > 1")
        if str(args.variant) in {"v58i", "v58j"}:
            if int(args.candidate_slot_start) != 0 or int(args.residual_samples) != 3:
                raise SystemExit(
                    f"--variant {args.variant} is the slot0/slot1/slot2 gate; "
                    "use --candidate-slot-start 0 --residual-samples 3"
                )
            if float(args.utility_fde_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-fde-weight > 0")
            if float(args.utility_miss_penalty) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-miss-penalty > 0")
            if float(args.utility_slow_fde_hurt_weight) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --utility-slow-fde-hurt-weight > 0")
            if float(args.lambda_candidate_safety) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --lambda-candidate-safety > 0")
            if float(args.lambda_residual_accept) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --lambda-residual-accept > 0")
            if not bool(args.target_fallback_to_slot0):
                raise SystemExit(f"--variant {args.variant} requires --target-fallback-to-slot0")
            if not str(args.top_base_rank_weights).strip():
                raise SystemExit(f"--variant {args.variant} requires --top-base-rank-weights")
            if str(args.variant) == "v58j" and not bool(args.confidence_fallback_to_slot0):
                raise SystemExit("--variant v58j requires --confidence-fallback-to-slot0")
        if str(args.variant) == "v58h":
            if int(args.candidate_slot_start) != 1 or int(args.residual_samples) != 2:
                raise SystemExit("--variant v58h is the slot1/slot2 utility selector; use --candidate-slot-start 1 --residual-samples 2")
            if float(args.utility_fde_weight) <= 0.0:
                raise SystemExit("--variant v58h requires --utility-fde-weight > 0")
            if float(args.utility_miss_penalty) <= 0.0:
                raise SystemExit("--variant v58h requires --utility-miss-penalty > 0")
            if (
                float(args.lambda_expected_score) <= 0.0
                and float(args.lambda_soft_utility) <= 0.0
                and float(args.lambda_pairwise_utility) <= 0.0
            ):
                raise SystemExit(
                    "--variant v58h requires at least one utility-shaped loss: "
                    "--lambda-expected-score, --lambda-soft-utility, or --lambda-pairwise-utility"
                )
            if (
                float(args.utility_slow_fde_hurt_weight) <= 0.0
                and float(args.utility_base_best_hurt_weight) <= 0.0
            ):
                raise SystemExit(
                    "--variant v58h requires a no-hurt term: "
                    "--utility-slow-fde-hurt-weight or --utility-base-best-hurt-weight"
                )
        if str(args.variant) in CONSERVATIVE_SLOT0_SELECTOR_VARIANTS:
            if not bool(args.target_fallback_to_slot0):
                raise SystemExit(f"--variant {args.variant} requires --target-fallback-to-slot0")
            if str(args.variant) != "v58i" and not bool(args.confidence_fallback_to_slot0):
                raise SystemExit(f"--variant {args.variant} requires --confidence-fallback-to-slot0")
            if float(args.target_improvement_margin) <= 0.0:
                raise SystemExit(f"--variant {args.variant} requires --target-improvement-margin > 0")
        elif bool(args.target_fallback_to_slot0) or bool(args.confidence_fallback_to_slot0):
            raise SystemExit(
                f"--target-fallback-to-slot0/--confidence-fallback-to-slot0 are reserved for conservative slot variants"
            )
        return
    if str(args.variant) != "v25c":
        return
    if int(args.residual_samples) != 3:
        raise SystemExit("--variant v25c requires --residual-samples 3")
    if not bool(args.include_mean_candidate):
        raise SystemExit("--variant v25c requires --include-mean-candidate")
    if not bool(args.target_fallback_to_mean):
        raise SystemExit("--variant v25c requires --target-fallback-to-mean")
    allowed_margins = (0.10, 0.15)
    margin = float(args.target_improvement_margin)
    if not any(abs(margin - allowed) <= 1e-9 for allowed in allowed_margins):
        raise SystemExit("--variant v25c requires --target-improvement-margin 0.10 or 0.15")


@torch.no_grad()
def _sample_candidates(
    refiner: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    residual_samples: int,
    include_mean_candidate: bool,
    candidate_slot_start: int = 0,
    z_mode: str = "sample",
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    slot_start = int(candidate_slot_start)
    if slot_start < 0:
        raise ValueError(f"candidate_slot_start must be non-negative, got {slot_start}")
    if slot_start > 0 and str(z_mode) != "slots":
        raise ValueError("candidate_slot_start > 0 requires z_mode='slots'")
    if bool(include_mean_candidate):
        mean_outputs = refiner.refine(
            batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            num_samples=1,
            z_mode="mean",
        )
        pieces.append(mean_outputs["refined"])
    requested_samples = int(residual_samples)
    if str(z_mode) == "slots":
        requested_samples += slot_start
    sample_outputs = refiner.refine(
        batch["teacher_pred"],
        past_traj_original_scale=batch["past_traj_original_scale"],
        temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
        num_samples=requested_samples,
        z_mode=str(z_mode),
    )
    sampled = sample_outputs["refined"]
    if str(z_mode) == "slots" and slot_start > 0:
        sampled = sampled[:, slot_start : slot_start + int(residual_samples)]
    pieces.append(sampled)
    return torch.cat(pieces, dim=1)


def _candidate_temporal_energy(
    candidates: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    past_abs = batch["past_traj_original_scale"][..., :2].to(device=candidates.device, dtype=candidates.dtype)
    agent_mask = batch["agent_mask"].to(device=candidates.device).bool()
    if past_abs.ndim != 4 or int(past_abs.shape[-1]) != 2:
        raise ValueError(f"past_traj_original_scale must contain xy channels, got {tuple(past_abs.shape)}")
    batch_size, num_samples, num_modes, num_agents, num_steps, coord_dim = candidates.shape
    future_abs = candidates + past_abs[:, None, None, :, -1:, :]
    flat_future = future_abs.reshape(batch_size, num_samples * num_modes, num_agents, num_steps, coord_dim)
    flat_energy = compute_temporal_interaction_energy_features(
        flat_future,
        past_abs,
        agent_mask=agent_mask,
        collision_sigma=0.5,
        collision_radius=0.2,
        no_neighbor_distance=10.0,
    )
    return flat_energy.reshape(batch_size, num_samples, num_modes, num_agents, num_steps, -1).to(
        device=candidates.device,
        dtype=candidates.dtype,
    )


def _candidate_energy_summary(
    candidates: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    trajectory_aware: bool = False,
) -> torch.Tensor:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    past_abs = batch["past_traj_original_scale"][..., :2].to(device=candidates.device, dtype=candidates.dtype)
    agent_mask = batch["agent_mask"].to(device=candidates.device).bool()
    if past_abs.ndim != 4 or int(past_abs.shape[-1]) != 2:
        raise ValueError(f"past_traj_original_scale must contain xy channels, got {tuple(past_abs.shape)}")
    batch_size, num_samples, num_modes, num_agents, num_steps, coord_dim = candidates.shape
    future_abs = candidates + past_abs[:, None, None, :, -1:, :]
    flat_future = future_abs.reshape(batch_size, num_samples * num_modes, num_agents, num_steps, coord_dim)
    if bool(trajectory_aware):
        temporal_energy = compute_temporal_interaction_energy_features(
            flat_future,
            past_abs,
            agent_mask=agent_mask,
            collision_sigma=0.5,
            collision_radius=0.2,
            no_neighbor_distance=10.0,
        )
        temporal_energy = temporal_energy.reshape(batch_size, num_samples, num_modes, num_agents, num_steps, -1)
        base = batch["teacher_pred"].to(device=candidates.device, dtype=candidates.dtype)
        if base.ndim != 5:
            raise ValueError(f"teacher_pred must have shape [B,K,A,T,2], got {tuple(base.shape)}")
        base_abs = base + past_abs[:, None, :, -1:, :]
        return compute_trajectory_aware_interaction_summary_features(
            temporal_energy,
            future_abs,
            base_abs,
        ).to(device=candidates.device, dtype=candidates.dtype)
    flat_summary = compute_interaction_energy_features(
        flat_future,
        past_abs,
        agent_mask=agent_mask,
        collision_sigma=0.5,
        collision_radius=0.2,
        no_neighbor_distance=10.0,
    )
    return flat_summary.reshape(batch_size, num_samples, num_modes, num_agents, -1).to(
        device=candidates.device,
        dtype=candidates.dtype,
    )


def _use_trajectory_aware_candidate_summary(args: argparse.Namespace) -> bool:
    return str(args.variant) in TRAJECTORY_AWARE_CANDIDATE_SUMMARY_VARIANTS


def _use_energy_gated_fusion(args: argparse.Namespace) -> bool:
    return str(args.variant) in ENERGY_GATED_FUSION_VARIANTS


def _use_candidate_safety_penalty(args: argparse.Namespace) -> bool:
    return str(args.variant) in CANDIDATE_SAFETY_PENALTY_VARIANTS


def _use_residual_accept_gate(args: argparse.Namespace) -> bool:
    return str(args.variant) in RESIDUAL_ACCEPT_GATE_VARIANTS


def _use_base_best_guard(args: argparse.Namespace) -> bool:
    return str(args.variant) in BASE_BEST_GUARD_VARIANTS


def _use_fde_hurt_candidate_safety(args: argparse.Namespace) -> bool:
    return str(args.variant) in FDE_HURT_CANDIDATE_SAFETY_VARIANTS


def _candidate_score(candidates: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported label metric: {metric!r}")


def _candidate_error_components(
    candidates: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _base_error_components(
    base: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _candidate_utility_score(
    candidates: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Lower-is-better conservative utility used for V28-A selector labels."""

    cand_ade, cand_fde = _candidate_error_components(candidates, ground_truth)
    base_ade, base_fde = _base_error_components(base, ground_truth)
    primary = float(args.utility_ade_weight) * cand_ade + float(args.utility_fde_weight) * cand_fde
    score = primary
    if float(args.utility_miss_penalty) > 0.0:
        score = score + float(args.utility_miss_penalty) * (cand_fde > float(args.miss_threshold)).to(score.dtype)
    if float(args.utility_slow_fde_hurt_weight) > 0.0:
        fde_hurt = F.relu(cand_fde - base_fde[:, None, :, :] - float(args.utility_slow_fde_hurt_margin))
        score = score + float(args.utility_slow_fde_hurt_weight) * fde_hurt
    if float(args.utility_slow_ade_hurt_weight) > 0.0:
        ade_hurt = F.relu(cand_ade - base_ade[:, None, :, :] - float(args.utility_slow_ade_hurt_margin))
        score = score + float(args.utility_slow_ade_hurt_weight) * ade_hurt
    if float(args.utility_mean_hurt_weight) > 0.0:
        mean_primary = primary[:, :1, :, :]
        mean_hurt = F.relu(primary - mean_primary - float(args.utility_mean_hurt_margin))
        score = score + float(args.utility_mean_hurt_weight) * mean_hurt
    if float(args.utility_base_best_hurt_weight) > 0.0:
        base_primary = float(args.utility_ade_weight) * base_ade + float(args.utility_fde_weight) * base_fde
        base_best_index = base_fde.argmin(dim=1)
        base_best_mask = torch.zeros_like(base_fde, dtype=torch.bool)
        base_best_mask.scatter_(1, base_best_index[:, None, :], True)
        base_hurt = F.relu(primary - base_primary[:, None, :, :] - float(args.utility_base_best_hurt_margin))
        score = score + float(args.utility_base_best_hurt_weight) * base_hurt * base_best_mask[:, None, :, :].to(score.dtype)
    return score


def _training_score(
    candidates: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    if str(args.variant) in UTILITY_SELECTOR_VARIANTS:
        return _candidate_utility_score(candidates, base, ground_truth, args=args)
    return _candidate_score(candidates, ground_truth, metric=str(args.label_metric))


def _uses_target_slot0_fallback(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "target_fallback_to_slot0", False))


def _uses_target_reference_fallback(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "target_fallback_to_mean", False)) or _uses_target_slot0_fallback(args)


def _uses_confidence_slot0_fallback(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "confidence_fallback_to_slot0", False))


def _uses_confidence_reference_fallback(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "confidence_fallback_to_mean", False)) or _uses_confidence_slot0_fallback(args)


def _margin_adjusted_score(score: torch.Tensor, *, args: argparse.Namespace) -> torch.Tensor:
    if not _uses_target_reference_fallback(args):
        return score
    if bool(getattr(args, "target_fallback_to_mean", False)) and not bool(args.include_mean_candidate):
        raise ValueError("--target-fallback-to-mean requires --include-mean-candidate")
    adjusted = score.clone()
    if int(adjusted.shape[1]) > 1:
        adjusted[:, 1:, :, :] = adjusted[:, 1:, :, :] + float(args.target_improvement_margin)
    return adjusted


def _target_indices(score: torch.Tensor, *, args: argparse.Namespace) -> torch.Tensor:
    target = score.argmin(dim=1)
    if not _uses_target_reference_fallback(args):
        return target
    if bool(getattr(args, "target_fallback_to_mean", False)) and not bool(args.include_mean_candidate):
        raise ValueError("--target-fallback-to-mean requires --include-mean-candidate")
    best_score = torch.gather(score, dim=1, index=target[:, None, :, :]).squeeze(1)
    reference_score = score[:, 0]
    improve = best_score < (reference_score - float(args.target_improvement_margin))
    return torch.where(improve, target, torch.zeros_like(target))


def _select_candidates(candidates: torch.Tensor, sample_index: torch.Tensor) -> torch.Tensor:
    index = sample_index[:, None, :, :, None, None].expand(
        candidates.shape[0],
        1,
        candidates.shape[2],
        candidates.shape[3],
        candidates.shape[4],
        candidates.shape[5],
    )
    return torch.gather(candidates, dim=1, index=index).squeeze(1)


def _selector_indices(logits: torch.Tensor, *, args: argparse.Namespace) -> torch.Tensor:
    selected = logits.argmax(dim=1)
    if not _uses_confidence_reference_fallback(args):
        return selected
    if bool(getattr(args, "confidence_fallback_to_mean", False)) and not bool(getattr(args, "include_mean_candidate", False)):
        raise ValueError("--confidence-fallback-to-mean requires --include-mean-candidate")
    probs = F.softmax(logits, dim=1)
    selected_prob = torch.gather(probs, dim=1, index=selected[:, None, :, :]).squeeze(1)
    reference_prob = probs[:, 0]
    accept = (
        (selected != 0)
        & (selected_prob >= reference_prob + float(args.fallback_prob_margin))
        & (selected_prob >= float(args.fallback_min_selected_prob))
    )
    return torch.where(accept, selected, torch.zeros_like(selected))


def _rank_weight_tensor(base: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor, *, args: argparse.Namespace) -> Optional[torch.Tensor]:
    weights = _split_floats(str(getattr(args, "top_base_rank_weights", "")))
    if not weights:
        return None
    base_ade, base_fde = _base_error_components(base, ground_truth)
    if str(getattr(args, "top_base_weight_metric", "fde")) == "ade_fde":
        base_score = base_ade + base_fde
    else:
        base_score = base_fde
    order = torch.argsort(base_score, dim=1)
    rank = torch.empty_like(order)
    values = torch.arange(base_score.shape[1], device=base_score.device, dtype=torch.long)[None, :, None].expand_as(order)
    rank.scatter_(1, order, values)
    item_weight = torch.ones_like(base_score, dtype=torch.float32)
    for rank_index, weight in enumerate(weights):
        item_weight = torch.where(rank == int(rank_index), item_weight.new_tensor(float(weight)), item_weight)
    valid = mask[:, None, :].expand_as(item_weight).bool()
    if int(valid.sum().item()) > 0:
        item_weight = item_weight / item_weight[valid].mean().clamp_min(1.0e-6)
    return item_weight.to(device=base.device, dtype=base.dtype)


def _weighted_mode_agent_mean(values: torch.Tensor, mask: torch.Tensor, item_weight: Optional[torch.Tensor]) -> torch.Tensor:
    valid = mask[:, None, :].expand(values.shape[0], values.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return values.new_tensor(0.0)
    if item_weight is None:
        return values[valid].mean()
    weight = item_weight.to(device=values.device, dtype=values.dtype)
    weighted = values * weight
    return weighted[valid].sum() / weight[valid].sum().clamp_min(1.0e-6)


def _selector_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    item_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_samples = int(logits.shape[1])
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, num_samples)
    flat_target = target.reshape(-1)
    flat_mask = mask[:, None, :].expand(mask.shape[0], logits.shape[2], mask.shape[1]).reshape(-1).bool()
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none")
    if int(flat_mask.sum().item()) <= 0:
        return losses.new_tensor(0.0)
    if item_weight is None:
        return losses[flat_mask].mean()
    flat_weight = item_weight.reshape(-1).to(device=losses.device, dtype=losses.dtype)
    return (losses[flat_mask] * flat_weight[flat_mask]).sum() / flat_weight[flat_mask].sum().clamp_min(1.0e-6)


def _expected_score_loss(
    logits: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
    item_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    expected = (F.softmax(logits, dim=1) * score.detach()).sum(dim=1)
    return _weighted_mode_agent_mean(expected, mask, item_weight)


def _soft_utility_loss(
    logits: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
    item_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    adjusted_score = _margin_adjusted_score(score, args=args)
    temperature = max(float(args.utility_soft_temperature), 1e-6)
    target_probs = F.softmax(-adjusted_score.detach() / temperature, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    per_item = -(target_probs * log_probs).sum(dim=1)
    return _weighted_mode_agent_mean(per_item, mask, item_weight)


def _pairwise_utility_ranking_loss(
    logits: torch.Tensor,
    score: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
    item_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    adjusted_score = _margin_adjusted_score(score, args=args).detach()
    score_i = adjusted_score[:, :, None, :, :]
    score_j = adjusted_score[:, None, :, :, :]
    gap = score_j - score_i
    better = gap > float(args.pairwise_utility_min_gap)
    logit_i = logits[:, :, None, :, :]
    logit_j = logits[:, None, :, :, :]
    pair_loss = F.softplus(-(logit_i - logit_j))
    weight = gap.clamp_min(0.0).clamp_max(float(args.pairwise_utility_max_weight))
    eye = torch.eye(int(logits.shape[1]), dtype=torch.bool, device=logits.device)
    valid = (
        mask[:, None, None, None, :]
        .expand(mask.shape[0], logits.shape[1], logits.shape[1], logits.shape[2], mask.shape[1])
        .bool()
    )
    keep = better & valid & (~eye[None, :, :, None, None])
    if int(keep.sum().item()) <= 0:
        return logits.new_tensor(0.0)
    weighted = pair_loss * weight
    if item_weight is not None:
        weighted = weighted * item_weight[:, None, None, :, :].to(device=weighted.device, dtype=weighted.dtype)
        weight = weight * item_weight[:, None, None, :, :].to(device=weight.device, dtype=weight.dtype)
    return weighted[keep].sum() / weight[keep].sum().clamp_min(1e-6)


def _candidate_safety_targets(
    score: torch.Tensor,
    candidates: Optional[torch.Tensor] = None,
    ground_truth: Optional[torch.Tensor] = None,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    if bool(_use_fde_hurt_candidate_safety(args)):
        if candidates is None or ground_truth is None:
            raise ValueError("FDE-hurt candidate safety requires candidates and ground_truth")
        _cand_ade, cand_fde = _candidate_error_components(candidates, ground_truth)
        del _cand_ade
        unsafe = cand_fde > (cand_fde[:, :1, :, :] + float(args.candidate_safety_margin))
    else:
        if score.ndim != 4:
            raise ValueError(f"score must have shape [B,S,K,A], got {tuple(score.shape)}")
        unsafe = score > (score[:, :1, :, :] + float(args.candidate_safety_margin))
    if int(score.shape[1]) > 0:
        unsafe[:, 0, :, :] = False
    return unsafe.to(dtype=score.dtype)


def _candidate_safety_loss(
    aux_outputs: Mapping[str, torch.Tensor],
    score: torch.Tensor,
    candidates: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    safety_logits = aux_outputs.get("safety_logits")
    if safety_logits is None:
        return score.new_tensor(0.0)
    targets = _candidate_safety_targets(
        score.detach(),
        candidates=candidates.detach(),
        ground_truth=ground_truth.detach(),
        args=args,
    ).to(device=safety_logits.device, dtype=safety_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(safety_logits, targets, reduction="none")
    valid = mask[:, None, None, :].expand(mask.shape[0], safety_logits.shape[1], safety_logits.shape[2], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return losses.new_tensor(0.0)
    return losses[valid].mean()


def _candidate_safety_unsafe_ratio(
    score: torch.Tensor,
    candidates: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> float:
    targets = _candidate_safety_targets(
        score.detach(),
        candidates=candidates.detach(),
        ground_truth=ground_truth.detach(),
        args=args,
    )
    valid = mask[:, None, None, :].expand(mask.shape[0], targets.shape[1], targets.shape[2], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float(targets[valid].float().mean().detach().cpu())


def _residual_accept_targets(target: torch.Tensor) -> torch.Tensor:
    return (target != 0).to(dtype=torch.float32)


def _residual_accept_loss(
    aux_outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    accept_logits = aux_outputs.get("accept_logits")
    if accept_logits is None:
        return target.new_tensor(0.0, dtype=torch.float32)
    targets = _residual_accept_targets(target).to(device=accept_logits.device, dtype=accept_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(accept_logits, targets, reduction="none")
    valid = mask[:, None, :].expand(mask.shape[0], target.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return losses.new_tensor(0.0)
    return losses[valid].mean()


def _residual_accept_accuracy(
    aux_outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    accept_logits = aux_outputs.get("accept_logits")
    if accept_logits is None:
        return 0.0
    targets = _residual_accept_targets(target).to(device=accept_logits.device).bool()
    pred = accept_logits > 0.0
    valid = mask[:, None, :].expand(mask.shape[0], target.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float((pred[valid] == targets[valid]).float().mean().detach().cpu())


def _residual_accept_target_ratio(target: torch.Tensor, mask: torch.Tensor) -> float:
    targets = _residual_accept_targets(target)
    valid = mask[:, None, :].expand(mask.shape[0], target.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float(targets[valid].mean().detach().cpu())


def _base_best_guard_targets(base: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    base_ade, base_fde = _base_error_components(base, ground_truth)
    del base_ade
    best_index = base_fde.argmin(dim=1)
    targets = torch.zeros_like(base_fde)
    targets.scatter_(1, best_index[:, None, :], 1.0)
    return targets


def _base_best_guard_loss(
    aux_outputs: Mapping[str, torch.Tensor],
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    guard_logits = aux_outputs.get("base_best_guard_logits")
    if guard_logits is None:
        return base.new_tensor(0.0)
    targets = _base_best_guard_targets(base, ground_truth).to(device=guard_logits.device, dtype=guard_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(guard_logits, targets, reduction="none")
    valid = mask[:, None, :].expand(mask.shape[0], targets.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return losses.new_tensor(0.0)
    return losses[valid].mean()


def _base_best_guard_accuracy(
    aux_outputs: Mapping[str, torch.Tensor],
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    guard_logits = aux_outputs.get("base_best_guard_logits")
    if guard_logits is None:
        return 0.0
    targets = _base_best_guard_targets(base, ground_truth).to(device=guard_logits.device).bool()
    pred_index = guard_logits.argmax(dim=1)
    pred = torch.zeros_like(targets)
    pred.scatter_(1, pred_index[:, None, :], 1.0)
    pred = pred.bool()
    valid = mask[:, None, :].expand(mask.shape[0], targets.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float((pred[valid] == targets[valid]).float().mean().detach().cpu())


def _selector_margin_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    target_logits = torch.gather(logits, dim=1, index=target[:, None, :, :]).squeeze(1)
    violations = F.relu(float(margin) + logits - target_logits[:, None, :, :])
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask.scatter_(1, target[:, None, :, :], True)
    valid = mask[:, None, None, :].expand(mask.shape[0], logits.shape[1], logits.shape[2], mask.shape[1]).bool()
    keep = valid & (~target_mask)
    if int(keep.sum().item()) <= 0:
        return logits.new_tensor(0.0)
    return violations[keep].mean()


def _target_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    valid = mask[:, None, :].expand(mask.shape[0], pred.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float((pred[valid] == target[valid]).float().mean().detach().cpu())


def _target_mean_ratio(target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask[:, None, :].expand(mask.shape[0], target.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float((target[valid] == 0).float().mean().detach().cpu())


def _selected_mean_ratio(selected: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask[:, None, :].expand(mask.shape[0], selected.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        return 0.0
    return float((selected[valid] == 0).float().mean().detach().cpu())


def _target_utility_gain(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    target_score = torch.gather(score, dim=1, index=target[:, None, :, :]).squeeze(1)
    gain = score[:, 0] - target_score
    return float(_masked_mean(gain.mean(dim=1), mask).detach().cpu())


def _loss_step(
    selector: SocialCVAEGroupSelector,
    refiner: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    candidates = _sample_candidates(
        refiner,
        batch,
        residual_samples=int(args.residual_samples),
        include_mean_candidate=bool(args.include_mean_candidate),
        candidate_slot_start=int(args.candidate_slot_start),
        z_mode=str(args.candidate_z_mode),
    ).detach()
    score = _training_score(candidates, batch["teacher_pred"], batch["ground_truth"], args=args)
    target = _target_indices(score, args=args)
    item_weight = _rank_weight_tensor(
        batch["teacher_pred"],
        batch["ground_truth"],
        batch["agent_mask"].bool(),
        args=args,
    )
    candidate_energy = (
        _candidate_temporal_energy(candidates, batch)
        if bool(getattr(args, "use_candidate_energy_context", False))
        else None
    )
    candidate_energy_summary = (
        _candidate_energy_summary(
            candidates,
            batch,
            trajectory_aware=_use_trajectory_aware_candidate_summary(args),
        )
        if bool(getattr(args, "use_candidate_energy_summary_context", False))
        else None
    )
    selector_outputs = selector(
        candidates,
        base_trajectory=batch["teacher_pred"],
        past_traj_original_scale=batch["past_traj_original_scale"],
        temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
        candidate_temporal_energy_features=candidate_energy,
        candidate_energy_summary_features=candidate_energy_summary,
        return_aux=_use_candidate_safety_penalty(args) or _use_residual_accept_gate(args) or _use_base_best_guard(args),
    )
    if isinstance(selector_outputs, Mapping):
        aux_outputs = dict(selector_outputs)
        logits = aux_outputs["logits"]
    else:
        logits = selector_outputs
        aux_outputs = {"logits": logits}
    loss_ce = _selector_ce_loss(logits, target, batch["agent_mask"].bool(), item_weight=item_weight)
    loss_expected = _expected_score_loss(logits, score, batch["agent_mask"].bool(), item_weight=item_weight)
    loss_soft = _soft_utility_loss(logits, score, batch["agent_mask"].bool(), args=args, item_weight=item_weight)
    loss_pairwise = _pairwise_utility_ranking_loss(
        logits,
        score,
        batch["agent_mask"].bool(),
        args=args,
        item_weight=item_weight,
    )
    loss_safety = _candidate_safety_loss(
        aux_outputs,
        score,
        candidates,
        batch["ground_truth"],
        batch["agent_mask"].bool(),
        args=args,
    )
    loss_accept = _residual_accept_loss(aux_outputs, target, batch["agent_mask"].bool())
    loss_guard = _base_best_guard_loss(
        aux_outputs,
        batch["teacher_pred"],
        batch["ground_truth"],
        batch["agent_mask"].bool(),
    )
    loss_margin = _selector_margin_ranking_loss(
        logits,
        target,
        batch["agent_mask"].bool(),
        margin=float(args.selector_logit_margin),
    )
    loss = (
        float(args.lambda_ce) * loss_ce
        + float(args.lambda_expected_score) * loss_expected
        + float(args.lambda_soft_utility) * loss_soft
        + float(args.lambda_pairwise_utility) * loss_pairwise
        + float(args.lambda_candidate_safety) * loss_safety
        + float(args.lambda_residual_accept) * loss_accept
        + float(args.lambda_base_best_guard) * loss_guard
        + float(args.lambda_margin_ranking) * loss_margin
    )
    selected_index = _selector_indices(logits, args=args)
    selected = _select_candidates(candidates, selected_index)
    oracle = _select_candidates(candidates, target)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_expected_score": float(loss_expected.detach().cpu()),
        "loss_soft_utility": float(loss_soft.detach().cpu()),
        "loss_pairwise_utility": float(loss_pairwise.detach().cpu()),
        "loss_candidate_safety": float(loss_safety.detach().cpu()),
        "loss_residual_accept": float(loss_accept.detach().cpu()),
        "loss_base_best_guard": float(loss_guard.detach().cpu()),
        "loss_margin_ranking": float(loss_margin.detach().cpu()),
        "target_accuracy": _target_accuracy(logits, target, batch["agent_mask"].bool()),
        "target_mean_ratio": _target_mean_ratio(target, batch["agent_mask"].bool()),
        "selected_mean_ratio": _selected_mean_ratio(selected_index, batch["agent_mask"].bool()),
        "target_utility_gain": _target_utility_gain(score, target, batch["agent_mask"].bool()),
        "candidate_safety_unsafe_ratio": _candidate_safety_unsafe_ratio(
            score,
            candidates,
            batch["ground_truth"],
            batch["agent_mask"].bool(),
            args=args,
        ),
        "residual_accept_accuracy": _residual_accept_accuracy(aux_outputs, target, batch["agent_mask"].bool()),
        "residual_accept_target_ratio": _residual_accept_target_ratio(target, batch["agent_mask"].bool()),
        "base_best_guard_accuracy": _base_best_guard_accuracy(
            aux_outputs,
            batch["teacher_pred"],
            batch["ground_truth"],
            batch["agent_mask"].bool(),
        ),
    }
    return loss, metrics, {"selected": selected, "oracle": oracle, "candidates": candidates, "logits": logits}


@torch.no_grad()
def _eval_loader(
    selector: SocialCVAEGroupSelector,
    refiner: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    selector.eval()
    refiner.eval()
    summaries: List[Dict[str, float]] = []
    weights: List[int] = []
    accuracies: List[float] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        candidates = _sample_candidates(
            refiner,
            batch,
            residual_samples=int(args.residual_samples),
            include_mean_candidate=bool(args.include_mean_candidate),
            candidate_slot_start=int(args.candidate_slot_start),
            z_mode=str(args.candidate_z_mode),
        ).detach()
        score = _training_score(candidates, batch["teacher_pred"], batch["ground_truth"], args=args)
        target = _target_indices(score, args=args)
        candidate_energy = (
            _candidate_temporal_energy(candidates, batch)
            if bool(getattr(args, "use_candidate_energy_context", False))
            else None
        )
        candidate_energy_summary = (
            _candidate_energy_summary(
                candidates,
                batch,
                trajectory_aware=_use_trajectory_aware_candidate_summary(args),
            )
            if bool(getattr(args, "use_candidate_energy_summary_context", False))
            else None
        )
        outputs = selector.select(
            candidates,
            base_trajectory=batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            candidate_temporal_energy_features=candidate_energy,
            candidate_energy_summary_features=candidate_energy_summary,
        )
        selected_index = _selector_indices(outputs["logits"], args=args)
        selected = _select_candidates(candidates, selected_index)
        summary = _summarize_refinement(
            selected[:, None, ...],
            batch["teacher_pred"],
            batch["ground_truth"],
            batch["agent_mask"].bool(),
            miss_threshold=float(args.miss_threshold),
        )
        summaries.append(summary)
        weights.append(int(batch["agent_mask"].bool().sum().item()))
        accuracies.append(_target_accuracy(outputs["logits"], target, batch["agent_mask"].bool()))
        summary["target_mean_ratio"] = _target_mean_ratio(target, batch["agent_mask"].bool())
        summary["selected_mean_ratio"] = _selected_mean_ratio(selected_index, batch["agent_mask"].bool())
        summary["target_utility_gain"] = _target_utility_gain(score, target, batch["agent_mask"].bool())
    if not summaries:
        return {}
    total = max(sum(weights), 1)
    result: Dict[str, float] = {}
    for key in summaries[0].keys():
        result[key] = float(sum(summary[key] * weight for summary, weight in zip(summaries, weights)) / total)
    result["target_accuracy"] = float(sum(accuracies) / max(len(accuracies), 1))
    return result


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_samples) <= 0:
        raise SystemExit("--residual-samples must be positive")
    _validate_variant_args(args)
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    cache_path = Path(args.cache_path).expanduser().resolve()
    payload = _load_cache(cache_path)
    tensors = _prepare_refiner_tensors(payload, args=args)
    num_items = int(tensors["ground_truth"].shape[0])
    train_indices, val_indices = _select_indices(
        num_items,
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )
    train_loader = DataLoader(
        CacheDataset(tensors, train_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        CacheDataset(tensors, val_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    refiner_path = Path(args.refiner_checkpoint).expanduser().resolve()
    refiner = load_social_cvae_teacher_refiner(refiner_path, map_location=device).to(device)
    refiner.eval()
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)
    if str(args.candidate_z_mode) == "slots":
        max_slots = int(getattr(refiner.config, "max_residual_slots", 0))
        required_slots = int(args.candidate_slot_start) + int(args.residual_samples)
        if required_slots > max_slots:
            raise SystemExit(
                f"--candidate-slot-start + --residual-samples requires {required_slots} slots, "
                f"but checkpoint max_residual_slots={max_slots}"
            )

    teacher_shape = tensors["teacher_pred"].shape
    past_shape = tensors["past_traj_original_scale"].shape
    energy_shape = tensors["teacher_temporal_interaction_energy_features"].shape
    config = SocialCVAEGroupSelectorConfig(
        future_frames=int(teacher_shape[-2]),
        coord_dim=int(teacher_shape[-1]),
        past_frames=int(past_shape[-2]),
        past_feature_dim=int(past_shape[-1]),
        temporal_energy_dim=int(energy_shape[-1]),
        candidate_energy_summary_dim=(
            int(TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM)
            if _use_trajectory_aware_candidate_summary(args)
            else 0
        ),
        hidden_dim=int(args.hidden_dim),
        max_modes=int(teacher_shape[1]),
        use_mode_embedding=not bool(args.no_mode_embedding),
        use_energy_risk_map=bool(args.use_energy_risk_map),
        energy_risk_distance_scale=float(args.energy_risk_distance_scale),
        use_temporal_energy_encoder=bool(args.use_temporal_energy_encoder),
        energy_temporal_hidden_dim=int(args.energy_temporal_hidden_dim),
        use_mean_candidate_comparison=bool(args.use_mean_candidate_comparison),
        use_candidate_energy_context=bool(args.use_candidate_energy_context),
        use_candidate_energy_summary_context=bool(args.use_candidate_energy_summary_context),
        use_energy_gated_fusion=_use_energy_gated_fusion(args),
        use_candidate_safety_penalty=_use_candidate_safety_penalty(args),
        candidate_safety_penalty_strength=1.0,
        use_residual_accept_gate=_use_residual_accept_gate(args),
        residual_accept_gate_strength=1.0,
        use_base_best_guard=_use_base_best_guard(args),
        base_best_guard_strength=1.0,
        use_observable_feature_context=bool(args.use_observable_feature_context) or str(args.variant) in {"v58i", "v58j"},
    )
    selector = SocialCVAEGroupSelector(config).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_metrics: Dict[str, float] = {}
    best_checkpoint = output_dir / f"{args.run_name}_best.pt"
    latest_checkpoint = output_dir / f"{args.run_name}_latest.pt"

    print(
        "[train_social_cvae_selector] "
        f"variant={_variant_name(args)} cache={cache_path.as_posix()} refiner={refiner_path.as_posix()} "
        f"train_items={len(train_indices)} val_items={len(val_indices)} device={device} "
        f"residual_samples={args.residual_samples} candidate_z_mode={args.candidate_z_mode} "
        f"candidate_slot_start={args.candidate_slot_start} "
        f"include_mean={bool(args.include_mean_candidate)} "
        f"trainable_params={sum(p.numel() for p in selector.parameters() if p.requires_grad)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        selector.train()
        train_rows: List[Dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _outputs = _loss_step(selector, refiner, batch, args=args)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(selector.parameters(), float(args.grad_clip))
            optimizer.step()
            train_rows.append(metrics)
            if batch_index == 1 or batch_index % max(int(args.log_every), 1) == 0:
                print(
                    "[train_social_cvae_selector] "
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={metrics['loss']:.6f} ce={metrics['loss_ce']:.6f} "
                    f"acc={metrics['target_accuracy']:.4f}"
                )
        train_metrics = _mean_metrics(train_rows)
        val_metrics = _eval_loader(selector, refiner, val_loader, device=device, args=args)
        score = _selection_score(val_metrics, args)
        improved = best_score is None or score < best_score
        checkpoint_payload = {
            "model_state_dict": selector.state_dict(),
            "config": asdict(config),
            "meta": {
                "variant": _variant_name(args),
                "epoch": int(epoch),
                "selection_metric": args.selection_metric,
                "selection_score": float(score),
                "cache_path": cache_path.as_posix(),
                "refiner_checkpoint": refiner_path.as_posix(),
                "candidate_slot_start": int(args.candidate_slot_start),
            },
            "args": _jsonable(vars(args)),
            "train_metrics": _jsonable(train_metrics),
            "val_metrics": _jsonable(val_metrics),
        }
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_metrics = dict(val_metrics)
            torch.save(checkpoint_payload, best_checkpoint)
        torch.save(checkpoint_payload, latest_checkpoint)
        print(
            "[train_social_cvae_selector] "
            f"epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.6f} "
            f"val_FDE_min={val_metrics.get('refined_FDE_min', float('nan')):.6f} "
            f"val_dFDE={val_metrics.get('dFDE_min', float('nan')):+.6f} "
            f"val_acc={val_metrics.get('target_accuracy', float('nan')):.4f} "
            f"score={score:.6f} best={bool(improved)}"
        )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_social_cvae_selector",
            "variant": _variant_name(args),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_checkpoint": best_checkpoint.as_posix(),
            "selection_metric": args.selection_metric,
            "best_selection_score": best_score,
            "refiner_checkpoint": refiner_path.as_posix(),
            "candidate_slot_start": int(args.candidate_slot_start),
        },
        "args": _jsonable(vars(args)),
        "cache_path": cache_path.as_posix(),
        "model_config": asdict(config),
        "best_val_metrics": _jsonable(best_metrics),
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_epoch: {best_epoch}")
    print(f"best_checkpoint: {best_checkpoint.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
