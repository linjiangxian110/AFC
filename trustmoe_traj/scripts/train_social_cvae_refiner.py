"""Train a SocialCVAE-style residual refiner on teacher coarse trajectories.

V24-A uses the slow teacher output as the coarse trajectory set and learns a
latent residual distribution in trajectory time.  This is deliberately outside
the flow sampler: the supervised residual target is ``GT - teacher_pred``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trustmoe_traj.evaluation import displacement_errors
from trustmoe_traj.models import (
    SocialCVAETeacherRefiner,
    SocialCVAETeacherRefinerConfig,
    compute_temporal_interaction_energy_features,
    load_social_cvae_teacher_refiner,
)
from trustmoe_traj.scripts.analogical_future_coverage import AnalogicalFutureBank, _agent_features_from_batch
from trustmoe_traj.scripts.train_student_integrated_finetune import (
    DEFAULT_CACHE_PATH,
    CacheDataset,
    _diversity_loss,
    _good_nohurt_loss,
    _gt_min_loss,
    _jsonable,
    _load_cache,
    _masked_mean,
    _move_batch,
    _prepare_tensors,
    _resolve_device,
    _select_indices,
    _set_seed,
    _student_best_nohurt_loss,
    _trajectory_spread,
    _endpoint_spread,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "social_cvae_refiner_models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SocialCVAE-style teacher trajectory refiner.")
    parser.add_argument(
        "--variant",
        type=str,
        default="v24a",
        choices=[
            "v24a",
            "v24b",
            "v26a",
            "v37a",
            "v38a",
            "v56a",
            "v56b1",
            "v56c1",
            "v56c2",
            "v56c3",
            "v57a",
            "v58a1",
            "v58b1",
            "v58b2",
            "v58d1",
            "v58d2",
            "v58d3",
            "v58f1",
            "v58f2",
            "v59a",
            "v59b",
            "v59c",
            "v60a",
        ],
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="social_cvae_refiner")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--allow-energy-fallback",
        action="store_true",
        help=(
            "If the cache lacks teacher_temporal_interaction_energy_features, build an approximate local "
            "energy tensor from cached teacher_pred/past. For per-agent caches this has no scene-neighbor signal, "
            "so it should be used only for smoke tests."
        ),
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=1.0)
    parser.add_argument("--no-mode-embedding", action="store_true")
    parser.add_argument("--use-energy-risk-map", action="store_true")
    parser.add_argument("--energy-risk-distance-scale", type=float, default=0.5)
    parser.add_argument("--use-temporal-energy-encoder", action="store_true")
    parser.add_argument("--energy-temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=0)
    parser.add_argument("--decoder-layers", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--posterior-samples", type=int, default=1)
    parser.add_argument("--eval-z-mode", type=str, default="mean", choices=["mean", "sample", "slots"])
    parser.add_argument("--set-residual-slots", type=int, default=4)
    parser.add_argument("--set-slot-scale", type=float, default=1.0)
    parser.add_argument("--dynamic-slot-hidden-dim", type=int, default=0)
    parser.add_argument("--dynamic-slot-offset-scale", type=float, default=1.0)
    parser.add_argument(
        "--allow-dynamic-slot0",
        action="store_true",
        help="Allow the dynamic slot sampler to move slot0. By default slot0 stays static for safety.",
    )

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-recon-best", type=float, default=1.0)
    parser.add_argument("--lambda-gt-min", type=float, default=0.2)
    parser.add_argument("--lambda-set-coverage", type=float, default=1.0)
    parser.add_argument("--set-coverage-temperature", type=float, default=0.1)
    parser.add_argument("--lambda-energy-recon", type=float, default=0.2)
    parser.add_argument("--energy-risk-floor", type=float, default=0.05)
    parser.add_argument("--energy-distance-scale", type=float, default=0.5)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--lambda-base-best-nohurt", type=float, default=2.0)
    parser.add_argument("--base-best-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-good-nohurt", type=float, default=1.0)
    parser.add_argument("--good-nohurt-frac", type=float, default=0.25)
    parser.add_argument("--good-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-diversity-preserve", type=float, default=0.5)
    parser.add_argument("--diversity-preserve-target-ratio", type=float, default=0.98)
    parser.add_argument("--lambda-keep", type=float, default=0.05)
    parser.add_argument("--lambda-non-best-keep", type=float, default=0.0)
    parser.add_argument("--lambda-non-best-nohurt", type=float, default=0.0)
    parser.add_argument("--non-best-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-delta-l2", type=float, default=0.01)
    parser.add_argument("--lambda-slot-spread", type=float, default=0.05)
    parser.add_argument("--slot-endpoint-spread-target", type=float, default=0.05)
    parser.add_argument("--slot-trajectory-spread-target", type=float, default=0.02)
    parser.add_argument("--lambda-elite-soft-wta", type=float, default=0.6)
    parser.add_argument("--elite-soft-temperature", type=float, default=0.08)
    parser.add_argument("--elite-base-topk", type=int, default=1)
    parser.add_argument("--lambda-elite-improvement", type=float, default=0.8)
    parser.add_argument("--elite-improvement-margin", type=float, default=0.02)
    parser.add_argument("--lambda-elite-density", type=float, default=0.4)
    parser.add_argument("--elite-density-slots", type=int, default=2)
    parser.add_argument("--lambda-slot0-preserve", type=float, default=0.25)
    parser.add_argument("--lambda-residual-norm-band", type=float, default=0.05)
    parser.add_argument("--residual-endpoint-norm-max", type=float, default=0.6)
    parser.add_argument("--residual-trajectory-norm-max", type=float, default=0.35)
    parser.add_argument("--semantic-prototype-path", type=str, default=None)
    parser.add_argument("--lambda-semantic-prototype", type=float, default=0.2)
    parser.add_argument("--lambda-semantic-slot0-identity", type=float, default=0.5)
    parser.add_argument("--elite-teacher-checkpoint", type=str, default=None)
    parser.add_argument("--elite-teacher-slots", type=int, default=8)
    parser.add_argument("--lambda-elite-teacher-distill", type=float, default=0.0)
    parser.add_argument("--elite-distill-min-gain", type=float, default=0.05)
    parser.add_argument("--lambda-dynamic-slot-offset-l2", type=float, default=0.01)
    parser.add_argument("--lambda-elite-set-distill", type=float, default=0.0)
    parser.add_argument("--elite-set-topk", type=int, default=3)
    parser.add_argument("--elite-set-min-gain", type=float, default=0.10)
    parser.add_argument("--elite-set-student-to-teacher-weight", type=float, default=0.25)
    parser.add_argument("--lambda-front-elite-distill", type=float, default=0.0)
    parser.add_argument("--front-elite-slots", type=int, default=3)
    parser.add_argument("--front-elite-topk", type=int, default=3)
    parser.add_argument("--front-elite-min-gain", type=float, default=0.10)
    parser.add_argument("--front-elite-student-to-teacher-weight", type=float, default=0.10)
    parser.add_argument(
        "--include-slot0-in-elite-set-distill",
        action="store_true",
        help="Let slot0 participate in elite-set distillation. By default only semantic residual slots are distilled.",
    )
    parser.add_argument("--v59-anchor-modes", type=int, default=4)
    parser.add_argument("--v59-afc-top-m", type=int, default=20)
    parser.add_argument("--v59-afc-clusters", type=int, default=6)
    parser.add_argument("--v59-afc-loss-temperature", type=float, default=0.08)
    parser.add_argument("--v59-afc-precision-temperature", type=float, default=0.12)
    parser.add_argument("--v59-anchor-temperature", type=float, default=0.08)
    parser.add_argument("--v59-afc-max-bank-items", type=int, default=0)
    parser.add_argument("--lambda-v59-anchor-obs", type=float, default=1.0)
    parser.add_argument("--lambda-v59-afc", type=float, default=1.0)
    parser.add_argument("--lambda-v59-afc-precision", type=float, default=0.25)
    parser.add_argument("--lambda-v59-afc-entropy", type=float, default=0.2)
    parser.add_argument("--lambda-v59-base-preserve", type=float, default=0.5)
    parser.add_argument("--v59-base-preserve-corrected-weight", type=float, default=0.25)
    parser.add_argument("--lambda-v59-anchor-keep", type=float, default=1.0)
    parser.add_argument("--lambda-v59-spread-floor", type=float, default=1.0)
    parser.add_argument("--v59-spread-floor-endpoint-ratio", type=float, default=0.85)
    parser.add_argument("--v59-spread-floor-trajectory-ratio", type=float, default=0.85)
    parser.add_argument("--lambda-v59-diversity", type=float, default=0.2)
    parser.add_argument("--lambda-v59-risk", type=float, default=0.1)
    parser.add_argument("--lambda-v59-residual", type=float, default=0.02)
    parser.add_argument("--v59-velocity-delta-max", type=float, default=0.35)
    parser.add_argument("--v59-accel-delta-max", type=float, default=0.35)
    parser.add_argument("--v60-anchor-modes", type=int, default=4)
    parser.add_argument("--v60-afc-top-m", type=int, default=20)
    parser.add_argument("--v60-afc-clusters", type=int, default=8)
    parser.add_argument("--v60-afc-max-bank-items", type=int, default=0)
    parser.add_argument("--v60-role-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-v60-anchor-keep", type=float, default=2.0)
    parser.add_argument("--lambda-v60-afc-role", type=float, default=1.0)
    parser.add_argument("--lambda-v60-base-identity", type=float, default=0.05)
    parser.add_argument("--lambda-v60-spread-floor", type=float, default=2.0)
    parser.add_argument("--lambda-v60-risk", type=float, default=0.1)
    parser.add_argument("--lambda-v60-residual", type=float, default=0.005)
    parser.add_argument("--v60-base-identity-margin", type=float, default=0.75)
    parser.add_argument("--v60-spread-floor-endpoint-ratio", type=float, default=0.98)
    parser.add_argument("--v60-spread-floor-trajectory-ratio", type=float, default=0.98)

    parser.add_argument("--selection-metric", type=str, default="safety", choices=["fde_min", "safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-nohurt-weight", type=float, default=2.0)
    parser.add_argument("--selection-diversity-weight", type=float, default=0.5)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _variant_name(_args: argparse.Namespace) -> str:
    if str(_args.variant) == "v60a":
        anchor_tag = f"anchor{int(_args.v60_anchor_modes)}"
        afc_tag = f"afc{int(_args.v60_afc_top_m)}"
        cluster_tag = f"clusters{int(_args.v60_afc_clusters)}"
        floor_tag = f"floor{int(round(float(_args.v60_spread_floor_endpoint_ratio) * 100)):03d}"
        scale_tag = f"dyn{int(round(float(_args.dynamic_slot_offset_scale) * 100)):03d}"
        return (
            f"v60a_gt_decoupled_role_afc_transport_slots{int(_args.set_residual_slots)}_"
            f"{anchor_tag}_{afc_tag}_{cluster_tag}_{floor_tag}_{scale_tag}"
        )
    if str(_args.variant) in {"v59a", "v59b", "v59c"}:
        anchor_tag = f"anchor{int(_args.v59_anchor_modes)}"
        afc_tag = f"afc{int(_args.v59_afc_top_m)}"
        cluster_tag = f"clusters{int(_args.v59_afc_clusters)}"
        scale_tag = f"dyn{int(round(float(_args.dynamic_slot_offset_scale) * 100)):03d}"
        if str(_args.variant) == "v59c":
            floor_tag = f"floor{int(round(float(_args.v59_spread_floor_endpoint_ratio) * 100)):02d}"
            return (
                f"v59c_anchor_hard_afc_coverage_floor_slots{int(_args.set_residual_slots)}_"
                f"{anchor_tag}_{afc_tag}_{cluster_tag}_{floor_tag}_{scale_tag}"
            )
        if str(_args.variant) == "v59b":
            return (
                f"v59b_anchor_preserving_afc_mode_coverage_slots{int(_args.set_residual_slots)}_"
                f"{anchor_tag}_{afc_tag}_{cluster_tag}_{scale_tag}"
            )
        return (
            f"v59a_anchor_preserving_afc_residual_slots{int(_args.set_residual_slots)}_"
            f"{anchor_tag}_{afc_tag}_{scale_tag}"
        )
    if str(_args.variant) in {"v58d1", "v58d2", "v58d3", "v58f1", "v58f2"}:
        gain_tag = f"gain{int(round(float(_args.front_elite_min_gain) * 1000)):03d}"
        topk_tag = f"top{int(_args.front_elite_topk)}"
        front_tag = f"front{int(_args.front_elite_slots)}"
        scale_tag = f"dyn{int(round(float(_args.dynamic_slot_offset_scale) * 100)):03d}"
        mode_tag = {
            "v58d1": "slot1best",
            "v58d2": "topk",
            "v58d3": "proto",
            "v58f1": "slot1best",
            "v58f2": "front2proto",
        }[str(_args.variant)]
        return (
            f"{str(_args.variant)}_front_loaded_{mode_tag}_slots{int(_args.set_residual_slots)}_"
            f"{front_tag}_{topk_tag}_{gain_tag}_{scale_tag}"
        )
    if str(_args.variant) == "v58b2":
        gain_tag = f"gain{int(round(float(_args.elite_set_min_gain) * 1000)):03d}"
        topk_tag = f"top{int(_args.elite_set_topk)}"
        scale_tag = f"dyn{int(round(float(_args.dynamic_slot_offset_scale) * 100)):03d}"
        return (
            f"v58b2_cem_dlow_elite_set_slots{int(_args.set_residual_slots)}_"
            f"{topk_tag}_{gain_tag}_{scale_tag}"
        )
    if str(_args.variant) == "v58b1":
        scale_tag = f"dyn{int(round(float(_args.dynamic_slot_offset_scale) * 100)):03d}"
        return f"v58b1_dynamic_semantic_residual_slots{int(_args.set_residual_slots)}_{scale_tag}"
    if str(_args.variant) == "v58a1":
        margin_tag = f"gain{int(round(float(_args.elite_distill_min_gain) * 1000)):03d}"
        return f"v58a1_fair20_elite_distilled_residual_sampler_{margin_tag}"
    if str(_args.variant) == "v57a":
        return f"v57a_semantic_residual_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v56c1":
        return f"v56c1_quality_weighted_wta_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v56c2":
        return f"v56c2_quality_weighted_wta_slot0_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v56c3":
        return f"v56c3_quality_weighted_wta_gain_norm_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v56b1":
        return "v56b1_fair20_single_residual_slot"
    if str(_args.variant) == "v56a":
        return f"v56a_elite_guided_residual_density_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v38a":
        return f"v38a_social_cvae_set_generator_slots{int(_args.set_residual_slots)}"
    if str(_args.variant) == "v37a":
        return "v37a_social_cvae_energy_conditioned_generator"
    if str(_args.variant) == "v26a":
        return "v26a_social_cvae_energy_map_capacity_refiner"
    if str(_args.variant) == "v24b":
        return "v24b_social_cvae_nonbest_protected_refiner"
    return "v24a_social_cvae_teacher_trajectory_refiner"


def _validate_variant_args(args: argparse.Namespace) -> None:
    if int(args.decoder_layers) <= 0:
        raise SystemExit("--decoder-layers must be positive")
    if int(args.energy_temporal_hidden_dim) <= 0:
        raise SystemExit("--energy-temporal-hidden-dim must be positive")
    if float(args.energy_risk_distance_scale) <= 0.0:
        raise SystemExit("--energy-risk-distance-scale must be positive")
    if int(args.set_residual_slots) <= 0:
        raise SystemExit("--set-residual-slots must be positive")
    if float(args.set_slot_scale) <= 0.0:
        raise SystemExit("--set-slot-scale must be positive")
    if int(args.dynamic_slot_hidden_dim) < 0:
        raise SystemExit("--dynamic-slot-hidden-dim must be non-negative")
    if float(args.dynamic_slot_offset_scale) < 0.0:
        raise SystemExit("--dynamic-slot-offset-scale must be non-negative")
    if float(args.set_coverage_temperature) < 0.0:
        raise SystemExit("--set-coverage-temperature must be non-negative")
    if float(args.slot_endpoint_spread_target) < 0.0:
        raise SystemExit("--slot-endpoint-spread-target must be non-negative")
    if float(args.slot_trajectory_spread_target) < 0.0:
        raise SystemExit("--slot-trajectory-spread-target must be non-negative")
    if int(args.elite_base_topk) <= 0:
        raise SystemExit("--elite-base-topk must be positive")
    if int(args.elite_density_slots) <= 0:
        raise SystemExit("--elite-density-slots must be positive")
    if float(args.elite_soft_temperature) < 0.0:
        raise SystemExit("--elite-soft-temperature must be non-negative")
    if float(args.residual_endpoint_norm_max) < 0.0:
        raise SystemExit("--residual-endpoint-norm-max must be non-negative")
    if float(args.residual_trajectory_norm_max) < 0.0:
        raise SystemExit("--residual-trajectory-norm-max must be non-negative")
    if float(args.lambda_semantic_prototype) < 0.0:
        raise SystemExit("--lambda-semantic-prototype must be non-negative")
    if float(args.lambda_semantic_slot0_identity) < 0.0:
        raise SystemExit("--lambda-semantic-slot0-identity must be non-negative")
    if int(args.elite_teacher_slots) <= 0:
        raise SystemExit("--elite-teacher-slots must be positive")
    if float(args.lambda_elite_teacher_distill) < 0.0:
        raise SystemExit("--lambda-elite-teacher-distill must be non-negative")
    if float(args.elite_distill_min_gain) < 0.0:
        raise SystemExit("--elite-distill-min-gain must be non-negative")
    if float(args.lambda_dynamic_slot_offset_l2) < 0.0:
        raise SystemExit("--lambda-dynamic-slot-offset-l2 must be non-negative")
    if float(args.lambda_elite_set_distill) < 0.0:
        raise SystemExit("--lambda-elite-set-distill must be non-negative")
    if int(args.elite_set_topk) <= 0:
        raise SystemExit("--elite-set-topk must be positive")
    if float(args.elite_set_min_gain) < 0.0:
        raise SystemExit("--elite-set-min-gain must be non-negative")
    if float(args.elite_set_student_to_teacher_weight) < 0.0:
        raise SystemExit("--elite-set-student-to-teacher-weight must be non-negative")
    if float(args.lambda_front_elite_distill) < 0.0:
        raise SystemExit("--lambda-front-elite-distill must be non-negative")
    if int(args.front_elite_slots) <= 0:
        raise SystemExit("--front-elite-slots must be positive")
    if int(args.front_elite_topk) <= 0:
        raise SystemExit("--front-elite-topk must be positive")
    if float(args.front_elite_min_gain) < 0.0:
        raise SystemExit("--front-elite-min-gain must be non-negative")
    if float(args.front_elite_student_to_teacher_weight) < 0.0:
        raise SystemExit("--front-elite-student-to-teacher-weight must be non-negative")
    if int(args.v59_anchor_modes) < 0:
        raise SystemExit("--v59-anchor-modes must be non-negative")
    if int(args.v59_afc_top_m) <= 0:
        raise SystemExit("--v59-afc-top-m must be positive")
    if int(args.v59_afc_clusters) <= 0:
        raise SystemExit("--v59-afc-clusters must be positive")
    if float(args.v59_afc_loss_temperature) < 0.0:
        raise SystemExit("--v59-afc-loss-temperature must be non-negative")
    if float(args.v59_afc_precision_temperature) < 0.0:
        raise SystemExit("--v59-afc-precision-temperature must be non-negative")
    if float(args.v59_anchor_temperature) < 0.0:
        raise SystemExit("--v59-anchor-temperature must be non-negative")
    if int(args.v59_afc_max_bank_items) < 0:
        raise SystemExit("--v59-afc-max-bank-items must be non-negative")
    for name in (
        "lambda_v59_anchor_obs",
        "lambda_v59_afc",
        "lambda_v59_afc_precision",
        "lambda_v59_afc_entropy",
        "lambda_v59_base_preserve",
        "lambda_v59_anchor_keep",
        "lambda_v59_spread_floor",
        "lambda_v59_diversity",
        "lambda_v59_risk",
        "lambda_v59_residual",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.v59_velocity_delta_max) < 0.0:
        raise SystemExit("--v59-velocity-delta-max must be non-negative")
    if float(args.v59_accel_delta_max) < 0.0:
        raise SystemExit("--v59-accel-delta-max must be non-negative")
    if float(args.v59_base_preserve_corrected_weight) < 0.0:
        raise SystemExit("--v59-base-preserve-corrected-weight must be non-negative")
    if float(args.v59_spread_floor_endpoint_ratio) < 0.0:
        raise SystemExit("--v59-spread-floor-endpoint-ratio must be non-negative")
    if float(args.v59_spread_floor_trajectory_ratio) < 0.0:
        raise SystemExit("--v59-spread-floor-trajectory-ratio must be non-negative")
    if int(args.v60_anchor_modes) < 0:
        raise SystemExit("--v60-anchor-modes must be non-negative")
    if int(args.v60_afc_top_m) <= 0:
        raise SystemExit("--v60-afc-top-m must be positive")
    if int(args.v60_afc_clusters) <= 0:
        raise SystemExit("--v60-afc-clusters must be positive")
    if int(args.v60_afc_max_bank_items) < 0:
        raise SystemExit("--v60-afc-max-bank-items must be non-negative")
    if float(args.v60_role_temperature) < 0.0:
        raise SystemExit("--v60-role-temperature must be non-negative")
    for name in (
        "lambda_v60_anchor_keep",
        "lambda_v60_afc_role",
        "lambda_v60_base_identity",
        "lambda_v60_spread_floor",
        "lambda_v60_risk",
        "lambda_v60_residual",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.v60_base_identity_margin) < 0.0:
        raise SystemExit("--v60-base-identity-margin must be non-negative")
    if float(args.v60_spread_floor_endpoint_ratio) < 0.0:
        raise SystemExit("--v60-spread-floor-endpoint-ratio must be non-negative")
    if float(args.v60_spread_floor_trajectory_ratio) < 0.0:
        raise SystemExit("--v60-spread-floor-trajectory-ratio must be non-negative")
    if str(args.variant) not in {
        "v26a",
        "v37a",
        "v38a",
        "v56a",
        "v56b1",
        "v56c1",
        "v56c2",
        "v56c3",
        "v57a",
        "v58a1",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
        "v59a",
        "v59b",
        "v59c",
        "v60a",
    }:
        return
    if not bool(args.use_energy_risk_map):
        raise SystemExit(f"--variant {args.variant} requires --use-energy-risk-map")
    if not bool(args.use_temporal_energy_encoder):
        raise SystemExit(f"--variant {args.variant} requires --use-temporal-energy-encoder")
    if int(args.decoder_layers) < 3:
        raise SystemExit(f"--variant {args.variant} requires --decoder-layers >= 3")
    if str(args.variant) in {
        "v38a",
        "v56a",
        "v56c1",
        "v56c2",
        "v56c3",
        "v57a",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
        "v59a",
        "v59b",
        "v59c",
        "v60a",
    }:
        if int(args.set_residual_slots) <= 1:
            raise SystemExit(f"--variant {args.variant} requires --set-residual-slots > 1")
        if str(args.eval_z_mode) != "slots":
            raise SystemExit(f"--variant {args.variant} requires --eval-z-mode slots")
    if str(args.variant) in {"v57a", "v58b1"} and float(args.lambda_semantic_prototype) > 0.0:
        if not args.semantic_prototype_path:
            raise SystemExit(f"--variant {args.variant} requires --semantic-prototype-path when prototype loss is enabled")
    if str(args.variant) in {"v58d3", "v58f2"} and float(args.lambda_semantic_prototype) > 0.0:
        if not args.semantic_prototype_path:
            raise SystemExit(f"--variant {args.variant} requires --semantic-prototype-path when prototype loss is enabled")
    if str(args.variant) == "v56b1" and str(args.eval_z_mode) != "mean":
        raise SystemExit("--variant v56b1 requires --eval-z-mode mean")
    if str(args.variant) == "v58a1":
        if str(args.eval_z_mode) != "mean":
            raise SystemExit("--variant v58a1 requires --eval-z-mode mean")
        if float(args.lambda_elite_teacher_distill) > 0.0 and not args.elite_teacher_checkpoint:
            raise SystemExit("--variant v58a1 requires --elite-teacher-checkpoint when elite distill is enabled")
    if str(args.variant) == "v58b2":
        if float(args.lambda_elite_set_distill) > 0.0 and not args.elite_teacher_checkpoint:
            raise SystemExit("--variant v58b2 requires --elite-teacher-checkpoint when elite set distill is enabled")
    if str(args.variant) in {"v58d1", "v58d2", "v58d3", "v58f1", "v58f2"}:
        if float(args.lambda_elite_set_distill) > 0.0 and not args.elite_teacher_checkpoint:
            raise SystemExit(f"--variant {args.variant} requires --elite-teacher-checkpoint when elite set distill is enabled")
        if float(args.lambda_front_elite_distill) > 0.0 and not args.elite_teacher_checkpoint:
            raise SystemExit(f"--variant {args.variant} requires --elite-teacher-checkpoint when front elite distill is enabled")
        if int(args.front_elite_slots) >= int(args.set_residual_slots):
            raise SystemExit("--front-elite-slots must leave slot0 outside the front elite group")
        if str(args.variant) in {"v58d3", "v58f2"} and float(args.lambda_front_elite_distill) > 0.0 and not args.semantic_prototype_path:
            raise SystemExit(f"--variant {args.variant} requires --semantic-prototype-path when front elite distill is enabled")


def _is_set_generator_variant(args: argparse.Namespace) -> bool:
    return str(args.variant) in {
        "v38a",
        "v56a",
        "v56c1",
        "v56c2",
        "v56c3",
        "v57a",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
        "v59a",
        "v59b",
        "v59c",
        "v60a",
    }


def _is_prior_generator_variant(args: argparse.Namespace) -> bool:
    return _is_set_generator_variant(args) or str(args.variant) in {"v56b1", "v58a1"}


def _energy_key(raw_tensors: Mapping[str, Any]) -> Optional[str]:
    for candidate in (
        "teacher_temporal_interaction_energy_features",
        "temporal_interaction_energy_features",
    ):
        if candidate in raw_tensors:
            return candidate
    return None


def _fallback_temporal_energy(tensors: Mapping[str, torch.Tensor], *, args: argparse.Namespace) -> torch.Tensor:
    teacher = tensors["teacher_pred"]
    past = tensors["past_traj_original_scale"]
    past_abs = past[..., :2]
    future_abs = teacher + past_abs[:, None, :, -1:, :]
    return compute_temporal_interaction_energy_features(
        future_abs,
        past_abs,
        agent_mask=tensors["agent_mask"],
        collision_sigma=0.5,
        collision_radius=0.2,
        no_neighbor_distance=10.0,
    ).to(torch.float32)


def _prepare_refiner_tensors(payload: Mapping[str, Any], *, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    tensors = _prepare_tensors(payload)
    raw_tensors = payload.get("tensors", {})
    if not isinstance(raw_tensors, Mapping):
        raw_tensors = {}
    key = _energy_key(raw_tensors)
    if key is not None:
        tensors["teacher_temporal_interaction_energy_features"] = raw_tensors[key].to(torch.float32)
    elif bool(args.allow_energy_fallback):
        tensors["teacher_temporal_interaction_energy_features"] = _fallback_temporal_energy(tensors, args=args)
        print(
            "[train_social_cvae_refiner] warning: using approximate temporal energy fallback from cache tensors. "
            "For official V24 runs, re-export cache with --include-teacher-temporal-interaction-energy-features."
        )
    else:
        raise ValueError(
            "V24 training requires cache tensor `teacher_temporal_interaction_energy_features` "
            "(or `temporal_interaction_energy_features`). Re-export the teacher/student cache with "
            "--include-teacher-temporal-interaction-energy-features, or pass --allow-energy-fallback for smoke tests."
        )
    if "past_social_risk_features" in raw_tensors:
        tensors["past_social_risk_features"] = raw_tensors["past_social_risk_features"].to(torch.float32)
    return tensors


def _build_cache_afc_bank(
    tensors: Mapping[str, torch.Tensor],
    train_indices: Sequence[int],
    *,
    args: argparse.Namespace,
) -> Optional[AnalogicalFutureBank]:
    variant = str(args.variant)
    if variant not in {"v59a", "v59b", "v59c", "v60a"}:
        return None
    if not train_indices:
        raise ValueError("Cannot build V59 AFC bank from an empty train split")
    selected = [int(item) for item in train_indices]
    max_items = int(args.v60_afc_max_bank_items if variant == "v60a" else args.v59_afc_max_bank_items)
    if max_items > 0:
        selected = selected[: min(max_items, len(selected))]
    index = torch.as_tensor(selected, dtype=torch.long)
    batch: Dict[str, torch.Tensor] = {
        "past_traj_original_scale": tensors["past_traj_original_scale"].index_select(0, index),
        "fut_traj_original_scale": tensors["ground_truth"].index_select(0, index),
        "agent_mask": tensors["agent_mask"].index_select(0, index),
        "afc_source_id": tensors["afc_source_id"].index_select(0, index),
    }
    if "past_social_risk_features" in tensors:
        batch["past_social_risk_features"] = tensors["past_social_risk_features"].index_select(0, index)
    features = _agent_features_from_batch(batch)
    valid = batch["agent_mask"].detach().cpu().bool()
    futures = batch["fut_traj_original_scale"].detach().cpu().to(torch.float32)
    source_ids = batch["afc_source_id"].detach().cpu().to(torch.long)
    return AnalogicalFutureBank.from_tensors(
        features[valid],
        futures[valid],
        top_m=int(args.v60_afc_top_m if variant == "v60a" else args.v59_afc_top_m),
        eps_values=(0.5, 1.0),
        source_ids=source_ids[valid],
    )


def _load_semantic_prototypes(path: Optional[str], *, device: str, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if not path:
        return None
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, Mapping):
        raw = payload.get("prototypes", payload.get("semantic_prototypes"))
    else:
        raw = payload
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    prototypes = raw.to(device=device, dtype=dtype)
    if prototypes.ndim != 3 or int(prototypes.shape[-1]) != 2:
        raise ValueError(f"Semantic prototypes must have shape [P,T,2], got {tuple(prototypes.shape)}")
    if int(prototypes.shape[0]) <= 0:
        raise ValueError("Semantic prototypes must contain at least one prototype")
    return prototypes


def _flatten_draw_modes(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 6:
        raise ValueError(f"Expected [B,S,K,A,T,2], got {tuple(prediction.shape)}")
    b, s, k, a, t, d = prediction.shape
    return prediction.reshape(b, s * k, a, t, d)


def _base_best_index(base: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    fde = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)[..., -1]
    return fde.argmin(dim=1)


def _gather_modes(tensor: torch.Tensor, index: torch.Tensor, *, mode_dim: int) -> torch.Tensor:
    if mode_dim < 0:
        mode_dim = tensor.ndim + mode_dim
    if mode_dim >= tensor.ndim:
        raise ValueError(f"mode_dim={mode_dim} out of range for shape {tuple(tensor.shape)}")
    batch_size = int(tensor.shape[0])
    num_agents = int(index.shape[1])
    shape = [1] * tensor.ndim
    shape[0] = batch_size
    shape[mode_dim] = 1
    agent_dim = 3 if tensor.ndim == 6 else 2
    shape[agent_dim] = num_agents
    gather_index = index.reshape(batch_size, *([1] * (agent_dim - 1)), num_agents, *([1] * (tensor.ndim - agent_dim - 1)))
    gather_index = gather_index.expand(*[tensor.shape[dim] if dim != mode_dim else 1 for dim in range(tensor.ndim)])
    return torch.gather(tensor, dim=mode_dim, index=gather_index).squeeze(mode_dim)


def _kl_divergence(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    prior_var = torch.exp(prior_logvar).clamp_min(1e-8)
    posterior_var = torch.exp(posterior_logvar)
    return 0.5 * (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var
        - 1.0
    ).sum(dim=-1)


def _selected_kl_loss(outputs: Mapping[str, torch.Tensor], best_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if "posterior_mu" not in outputs or "posterior_logvar" not in outputs:
        return outputs["prior_mu"].new_tensor(0.0)
    kl = _kl_divergence(
        outputs["posterior_mu"],
        outputs["posterior_logvar"],
        outputs["prior_mu"],
        outputs["prior_logvar"],
    )
    selected_kl = torch.gather(kl, dim=1, index=best_index[:, None, :]).squeeze(1)
    return _masked_mean(selected_kl, mask)


def _score(prediction: torch.Tensor, ground_truth: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]


def _slot_score(refined: torch.Tensor, ground_truth: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    dist = torch.linalg.norm(refined - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]


def _elite_base_indices(base: torch.Tensor, ground_truth: torch.Tensor, *, topk: int) -> torch.Tensor:
    fde = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)[..., -1]
    keep_k = max(1, min(int(topk), int(fde.shape[1])))
    return fde.topk(k=keep_k, dim=1, largest=False).indices


def _gather_elite_slot_scores(slot_scores: torch.Tensor, elite_index: torch.Tensor) -> torch.Tensor:
    gather_index = elite_index[:, None, :, :].expand(-1, int(slot_scores.shape[1]), -1, -1)
    return torch.gather(slot_scores, dim=2, index=gather_index)


def _gather_elite_base_scores(base_scores: torch.Tensor, elite_index: torch.Tensor) -> torch.Tensor:
    return torch.gather(base_scores, dim=1, index=elite_index)


def _elite_soft_wta_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    temperature: float,
    base_topk: int,
) -> torch.Tensor:
    slot_scores = _slot_score(refined, ground_truth, fde_weight=fde_weight)
    elite_index = _elite_base_indices(base, ground_truth, topk=base_topk)
    elite_scores = _gather_elite_slot_scores(slot_scores, elite_index)
    if float(temperature) <= 0.0:
        selected = elite_scores.min(dim=1).values
    else:
        weights = torch.softmax(-elite_scores / max(float(temperature), 1e-6), dim=1).detach()
        selected = (weights * elite_scores).sum(dim=1)
    return _masked_mean(selected.mean(dim=1), mask)


def _elite_improvement_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    base_topk: int,
    margin: float,
) -> torch.Tensor:
    slot_scores = _slot_score(refined, ground_truth, fde_weight=fde_weight)
    base_scores = _score(base, ground_truth, fde_weight=fde_weight)
    elite_index = _elite_base_indices(base, ground_truth, topk=base_topk)
    elite_slot_scores = _gather_elite_slot_scores(slot_scores, elite_index)
    elite_base_scores = _gather_elite_base_scores(base_scores, elite_index)
    best_slot_score = elite_slot_scores.min(dim=1).values
    return _masked_mean(F.relu(best_slot_score - elite_base_scores + float(margin)).mean(dim=1), mask)


def _elite_density_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    base_topk: int,
    density_slots: int,
    margin: float,
) -> torch.Tensor:
    slot_scores = _slot_score(refined, ground_truth, fde_weight=fde_weight)
    base_scores = _score(base, ground_truth, fde_weight=fde_weight)
    elite_index = _elite_base_indices(base, ground_truth, topk=base_topk)
    elite_slot_scores = _gather_elite_slot_scores(slot_scores, elite_index)
    elite_base_scores = _gather_elite_base_scores(base_scores, elite_index)
    keep_slots = max(1, min(int(density_slots), int(elite_slot_scores.shape[1])))
    dense_score = elite_slot_scores.topk(k=keep_slots, dim=1, largest=False).values.mean(dim=1)
    return _masked_mean(F.relu(dense_score - elite_base_scores + float(margin)).mean(dim=1), mask)


def _residual_norm_band_loss(
    delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    endpoint_max: float,
    trajectory_max: float,
) -> torch.Tensor:
    if delta.ndim != 6:
        raise ValueError(f"Expected delta [B,S,K,A,T,2], got {tuple(delta.shape)}")
    penalty = delta.new_zeros(delta.shape[:-2])
    if float(endpoint_max) > 0.0:
        endpoint_norm = torch.linalg.norm(delta[..., -1, :], dim=-1)
        penalty = penalty + F.relu(endpoint_norm - float(endpoint_max)).pow(2)
    if float(trajectory_max) > 0.0:
        trajectory_norm = torch.linalg.norm(delta, dim=-1).mean(dim=-1)
        penalty = penalty + F.relu(trajectory_norm - float(trajectory_max)).pow(2)
    return _masked_mean(penalty.mean(dim=1).mean(dim=1), mask)


def _slot0_identity_loss(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 6:
        raise ValueError(f"Expected delta [B,S,K,A,T,2], got {tuple(delta.shape)}")
    slot0_norm = torch.linalg.norm(delta[:, 0], dim=-1).mean(dim=-1).mean(dim=1)
    return _masked_mean(slot0_norm, mask)


def _semantic_prototype_alignment_loss(
    delta: torch.Tensor,
    prototypes: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    if prototypes is None:
        return delta.new_tensor(0.0)
    if delta.ndim != 6:
        raise ValueError(f"Expected delta [B,S,K,A,T,2], got {tuple(delta.shape)}")
    if int(delta.shape[1]) <= 1:
        return delta.new_tensor(0.0)
    if int(prototypes.shape[-2]) != int(delta.shape[-2]):
        raise ValueError(
            f"Prototype future length {int(prototypes.shape[-2])} does not match delta length {int(delta.shape[-2])}"
        )
    align_slots = min(int(delta.shape[1]) - 1, int(prototypes.shape[0]))
    semantic_delta = delta[:, 1 : 1 + align_slots]
    target = prototypes[:align_slots].to(device=delta.device, dtype=delta.dtype)
    error = torch.linalg.norm(semantic_delta - target[None, :, None, None, :, :], dim=-1).mean(dim=-1)
    return _masked_mean(error.mean(dim=1).mean(dim=1), mask)


def _elite_teacher_distill_loss(
    student_refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    temporal_energy_features: torch.Tensor,
    past_traj_original_scale: torch.Tensor,
    elite_teacher: Optional[SocialCVAETeacherRefiner],
    *,
    teacher_slots: int,
    fde_weight: float,
    min_gain: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if elite_teacher is None:
        return student_refined.new_tensor(0.0), {
            "elite_teacher_accept_ratio": 0.0,
            "elite_teacher_target_delta_l2": 0.0,
        }
    if student_refined.ndim != 6 or int(student_refined.shape[1]) != 1:
        raise ValueError(f"Expected student_refined [B,1,K,A,T,2], got {tuple(student_refined.shape)}")
    with torch.no_grad():
        teacher_outputs = elite_teacher.refine(
            base,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            num_samples=int(teacher_slots),
            z_mode="slots",
        )
        teacher_refined = teacher_outputs["refined"]
        teacher_scores = _slot_score(teacher_refined, ground_truth, fde_weight=fde_weight)
        best_slot = teacher_scores.argmin(dim=1)
        gather_index = best_slot[:, None, :, :, None, None].expand(
            teacher_refined.shape[0],
            1,
            teacher_refined.shape[2],
            teacher_refined.shape[3],
            teacher_refined.shape[4],
            teacher_refined.shape[5],
        )
        teacher_best = torch.gather(teacher_refined, dim=1, index=gather_index).squeeze(1)
        best_score = torch.gather(teacher_scores, dim=1, index=best_slot[:, None, :, :]).squeeze(1)
        base_score = _score(base, ground_truth, fde_weight=fde_weight)
        accept = best_score < (base_score - float(min_gain))
        target = torch.where(accept[..., None, None], teacher_best, base)
    student = student_refined[:, 0]
    dist = torch.linalg.norm(student - target, dim=-1)
    per_mode = dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]
    loss = _masked_mean(per_mode.mean(dim=1), mask)
    valid = mask[:, None, :].expand(mask.shape[0], base.shape[1], mask.shape[1]).bool()
    if int(valid.sum().item()) <= 0:
        accept_ratio = 0.0
        target_delta_l2 = 0.0
    else:
        accept_ratio = float(accept[valid].to(dtype=torch.float32).mean().detach().cpu())
        target_delta_l2 = float(
            torch.linalg.norm((target - base), dim=-1).mean(dim=-1)[valid].mean().detach().cpu()
        )
    return loss, {
        "elite_teacher_accept_ratio": accept_ratio,
        "elite_teacher_target_delta_l2": target_delta_l2,
    }


def _masked_element_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(device=values.device, dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def _gather_slot_dim(tensor: torch.Tensor, slot_index: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 4:
        raise ValueError(f"Expected tensor with slot dim, got {tuple(tensor.shape)}")
    batch_size, num_keep, num_modes, num_agents = slot_index.shape
    tail_shape = tuple(tensor.shape[4:])
    index = slot_index.reshape(batch_size, num_keep, num_modes, num_agents, *([1] * len(tail_shape)))
    index = index.expand(batch_size, num_keep, num_modes, num_agents, *tail_shape)
    return torch.gather(tensor, dim=1, index=index)


def _elite_residual_set_distill_loss(
    student_refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    temporal_energy_features: torch.Tensor,
    past_traj_original_scale: torch.Tensor,
    elite_teacher: Optional[SocialCVAETeacherRefiner],
    *,
    teacher_slots: int,
    target_topk: int,
    min_gain: float,
    fde_weight: float,
    student_to_teacher_weight: float,
    include_slot0: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    zero = student_refined.new_tensor(0.0)
    empty_metrics = {
        "elite_set_accept_ratio": 0.0,
        "elite_set_targets_per_agent": 0.0,
        "elite_set_target_delta_l2": 0.0,
        "elite_set_target_to_student": 0.0,
        "elite_set_student_to_target": 0.0,
    }
    if elite_teacher is None:
        return zero, empty_metrics
    if student_refined.ndim != 6:
        raise ValueError(f"Expected student_refined [B,S,K,A,T,2], got {tuple(student_refined.shape)}")
    if int(student_refined.shape[1]) <= 1 and not bool(include_slot0):
        return zero, empty_metrics

    with torch.no_grad():
        teacher_outputs = elite_teacher.refine(
            base,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            num_samples=int(teacher_slots),
            z_mode="slots",
        )
        teacher_refined = teacher_outputs["refined"]
        teacher_scores = _slot_score(teacher_refined, ground_truth, fde_weight=fde_weight)
        base_scores = _score(base, ground_truth, fde_weight=fde_weight)
        teacher_gain = base_scores[:, None, :, :] - teacher_scores
        keep_targets = max(1, min(int(target_topk), int(teacher_scores.shape[1])))
        _top_scores, top_index = teacher_scores.topk(k=keep_targets, dim=1, largest=False)
        top_gain = torch.gather(teacher_gain, dim=1, index=top_index)
        accepted = top_gain >= float(min_gain)
        teacher_delta = teacher_refined - base[:, None, ...]
        target_delta = _gather_slot_dim(teacher_delta, top_index)

    student_delta = student_refined - base[:, None, ...]
    if bool(include_slot0):
        student_candidates = student_delta
    else:
        student_candidates = student_delta[:, 1:]
    diff = student_candidates[:, :, None, ...] - target_delta[:, None, ...]
    point_dist = torch.linalg.norm(diff, dim=-1)
    pair_dist = point_dist.mean(dim=-1) + float(fde_weight) * point_dist[..., -1]

    valid_targets = accepted & mask[:, None, None, :].bool()
    target_to_student = pair_dist.min(dim=1).values
    loss_target_to_student = _masked_element_mean(target_to_student, valid_targets)

    large = torch.tensor(1.0e6, device=pair_dist.device, dtype=pair_dist.dtype)
    masked_pair_dist = pair_dist.masked_fill(~valid_targets[:, None, ...], large)
    student_to_target = masked_pair_dist.min(dim=2).values
    has_target = valid_targets.any(dim=1)
    valid_students = has_target[:, None, :, :].expand_as(student_to_target)
    loss_student_to_target = _masked_element_mean(student_to_target, valid_students)

    loss = loss_target_to_student + float(student_to_teacher_weight) * loss_student_to_target
    target_norm = torch.linalg.norm(target_delta, dim=-1).mean(dim=-1)
    valid_target_count = int(valid_targets.sum().detach().cpu().item())
    possible_targets = int((mask[:, None, None, :].expand_as(valid_targets)).sum().detach().cpu().item())
    if valid_target_count <= 0:
        target_delta_l2 = 0.0
    else:
        target_delta_l2 = float(target_norm[valid_targets].mean().detach().cpu())
    valid_agents = max(int(mask.sum().detach().cpu().item()), 1)
    metrics = {
        "elite_set_accept_ratio": float(valid_target_count / max(possible_targets, 1)),
        "elite_set_targets_per_agent": float(valid_target_count / valid_agents),
        "elite_set_target_delta_l2": target_delta_l2,
        "elite_set_target_to_student": float(loss_target_to_student.detach().cpu()),
        "elite_set_student_to_target": float(loss_student_to_target.detach().cpu()),
    }
    return loss, metrics


def _front_loaded_elite_distill_loss(
    student_refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    temporal_energy_features: torch.Tensor,
    past_traj_original_scale: torch.Tensor,
    elite_teacher: Optional[SocialCVAETeacherRefiner],
    semantic_prototypes: Optional[torch.Tensor],
    *,
    mode: str,
    teacher_slots: int,
    front_slots: int,
    target_topk: int,
    min_gain: float,
    fde_weight: float,
    student_to_teacher_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    zero = student_refined.new_tensor(0.0)
    empty_metrics = {
        "front_elite_accept_ratio": 0.0,
        "front_elite_targets_per_agent": 0.0,
        "front_elite_target_delta_l2": 0.0,
        "front_elite_target_to_student": 0.0,
        "front_elite_student_to_target": 0.0,
        "front_elite_assigned_slot_mean": 0.0,
        "front_elite_slot1_target_ratio": 0.0,
    }
    if elite_teacher is None:
        return zero, empty_metrics
    if student_refined.ndim != 6:
        raise ValueError(f"Expected student_refined [B,S,K,A,T,2], got {tuple(student_refined.shape)}")
    if int(student_refined.shape[1]) <= 1:
        return zero, empty_metrics
    if str(mode) not in {"best", "topk", "prototype"}:
        raise ValueError(f"Unsupported front elite mode: {mode!r}")

    max_front_slots = int(student_refined.shape[1]) - 1
    used_front_slots = max(1, min(int(front_slots), max_front_slots))
    if str(mode) == "best":
        used_front_slots = 1
        used_targets = 1
    else:
        used_targets = max(1, min(int(target_topk), int(teacher_slots)))
        used_targets = min(used_targets, used_front_slots) if str(mode) == "topk" else used_targets

    prototypes: Optional[torch.Tensor] = None
    if str(mode) == "prototype":
        if semantic_prototypes is None:
            return zero, empty_metrics
        if int(semantic_prototypes.shape[-2]) != int(student_refined.shape[-2]):
            raise ValueError(
                f"Prototype future length {int(semantic_prototypes.shape[-2])} "
                f"does not match student future length {int(student_refined.shape[-2])}"
            )
        used_front_slots = min(used_front_slots, int(semantic_prototypes.shape[0]))
        if used_front_slots <= 0:
            return zero, empty_metrics
        prototypes = semantic_prototypes[:used_front_slots].to(device=student_refined.device, dtype=student_refined.dtype)

    with torch.no_grad():
        teacher_outputs = elite_teacher.refine(
            base,
            past_traj_original_scale=past_traj_original_scale,
            temporal_energy_features=temporal_energy_features,
            num_samples=int(teacher_slots),
            z_mode="slots",
        )
        teacher_refined = teacher_outputs["refined"]
        teacher_scores = _slot_score(teacher_refined, ground_truth, fde_weight=fde_weight)
        base_scores = _score(base, ground_truth, fde_weight=fde_weight)
        teacher_gain = base_scores[:, None, :, :] - teacher_scores
        keep_targets = max(1, min(int(used_targets), int(teacher_scores.shape[1])))
        _top_scores, top_index = teacher_scores.topk(k=keep_targets, dim=1, largest=False)
        top_gain = torch.gather(teacher_gain, dim=1, index=top_index)
        accepted = top_gain >= float(min_gain)
        teacher_delta = teacher_refined - base[:, None, ...]
        target_delta = _gather_slot_dim(teacher_delta, top_index)

        if str(mode) == "prototype":
            if prototypes is None:
                assignment = torch.zeros_like(top_index)
            else:
                proto_diff = target_delta[:, None, ...] - prototypes[None, :, None, None, None, :, :]
                proto_point = torch.linalg.norm(proto_diff, dim=-1)
                proto_dist = proto_point.mean(dim=-1) + float(fde_weight) * proto_point[..., -1]
                assignment = proto_dist.argmin(dim=1)
        else:
            rank_assignment = torch.arange(keep_targets, device=student_refined.device, dtype=torch.long)
            assignment = rank_assignment[None, :, None, None].expand_as(top_index)

    student_delta = student_refined - base[:, None, ...]
    student_front = student_delta[:, 1 : 1 + used_front_slots]
    diff = student_front[:, :, None, ...] - target_delta[:, None, ...]
    point_dist = torch.linalg.norm(diff, dim=-1)
    pair_dist = point_dist.mean(dim=-1) + float(fde_weight) * point_dist[..., -1]

    valid_targets = accepted & mask[:, None, None, :].bool()
    slot_axis = torch.arange(used_front_slots, device=student_refined.device, dtype=torch.long)[None, :, None, None, None]
    valid_pair = valid_targets[:, None, ...] & (slot_axis == assignment[:, None, ...])
    large = torch.tensor(1.0e6, device=pair_dist.device, dtype=pair_dist.dtype)

    target_to_student = pair_dist.masked_fill(~valid_pair, large).min(dim=1).values
    loss_target_to_student = _masked_element_mean(target_to_student, valid_targets)

    student_to_target = pair_dist.masked_fill(~valid_pair, large).min(dim=2).values
    valid_students = valid_pair.any(dim=2)
    loss_student_to_target = _masked_element_mean(student_to_target, valid_students)

    loss = loss_target_to_student + float(student_to_teacher_weight) * loss_student_to_target
    target_norm = torch.linalg.norm(target_delta, dim=-1).mean(dim=-1)
    valid_target_count = int(valid_targets.sum().detach().cpu().item())
    possible_targets = int((mask[:, None, None, :].expand_as(valid_targets)).sum().detach().cpu().item())
    valid_agents = max(int(mask.sum().detach().cpu().item()), 1)
    if valid_target_count <= 0:
        target_delta_l2 = 0.0
        assigned_slot_mean = 0.0
        slot1_target_ratio = 0.0
    else:
        target_delta_l2 = float(target_norm[valid_targets].mean().detach().cpu())
        assigned = assignment[valid_targets].to(dtype=torch.float32)
        assigned_slot_mean = float((assigned + 1.0).mean().detach().cpu())
        slot1_target_ratio = float((assigned == 0).to(dtype=torch.float32).mean().detach().cpu())
    metrics = {
        "front_elite_accept_ratio": float(valid_target_count / max(possible_targets, 1)),
        "front_elite_targets_per_agent": float(valid_target_count / valid_agents),
        "front_elite_target_delta_l2": target_delta_l2,
        "front_elite_target_to_student": float(loss_target_to_student.detach().cpu()),
        "front_elite_student_to_target": float(loss_student_to_target.detach().cpu()),
        "front_elite_assigned_slot_mean": assigned_slot_mean,
        "front_elite_slot1_target_ratio": slot1_target_ratio,
    }
    return loss, metrics


def _base_best_recon_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
) -> torch.Tensor:
    best_index = _base_best_index(base, ground_truth)
    selected = _gather_modes(refined, best_index, mode_dim=2)
    error = torch.linalg.norm(selected - ground_truth[:, None, ...], dim=-1)
    score = error.mean(dim=-1) + float(fde_weight) * error[..., -1]
    return _masked_mean(score.min(dim=1).values, mask)


def _gt_softmin_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    temperature: float,
) -> torch.Tensor:
    scores = _score(_flatten_draw_modes(prediction), ground_truth, fde_weight=fde_weight)
    if float(temperature) <= 0.0:
        return _masked_mean(scores.min(dim=1).values, mask)
    weights = torch.softmax(-scores / max(float(temperature), 1e-6), dim=1)
    return _masked_mean((weights * scores).sum(dim=1), mask)


def _masked_softmin(values: torch.Tensor, valid: torch.Tensor, *, dim: int, temperature: float) -> torch.Tensor:
    large = torch.tensor(1.0e6, device=values.device, dtype=values.dtype)
    masked = values.masked_fill(~valid, large)
    if float(temperature) <= 0.0:
        return masked.min(dim=dim).values
    return -float(temperature) * torch.logsumexp(-masked / max(float(temperature), 1e-6), dim=dim)


def _v59_anchor_mode_mask(
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    topk: int,
    fde_weight: float,
) -> torch.Tensor:
    batch_size, num_modes, num_agents = [int(item) for item in base.shape[:3]]
    keep = max(0, min(int(topk), num_modes))
    if keep <= 0:
        return torch.zeros((batch_size, num_modes, num_agents), device=base.device, dtype=torch.bool)
    scores = _score(base, ground_truth, fde_weight=float(fde_weight))
    scores = scores.masked_fill(~mask[:, None, :].bool(), float("inf"))
    indices = scores.topk(k=keep, dim=1, largest=False).indices
    anchor = torch.zeros_like(scores, dtype=torch.bool)
    anchor.scatter_(dim=1, index=indices, value=True)
    return anchor & mask[:, None, :].bool()


def _v59_anchor_obs_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchor_modes: int,
    fde_weight: float,
    temperature: float,
) -> torch.Tensor:
    scores = _slot_score(refined, ground_truth, fde_weight=float(fde_weight))
    anchor = _v59_anchor_mode_mask(
        base,
        ground_truth,
        mask,
        topk=int(anchor_modes),
        fde_weight=float(fde_weight),
    )
    valid = anchor[:, None, :, :].expand_as(scores)
    flat_scores = scores.permute(0, 3, 1, 2).reshape(scores.shape[0], scores.shape[3], -1)
    flat_valid = valid.permute(0, 3, 1, 2).reshape(scores.shape[0], scores.shape[3], -1)
    loss = _masked_softmin(flat_scores, flat_valid, dim=-1, temperature=float(temperature))
    return _masked_mean(loss, mask)


def _v59_afc_proxy_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    afc_bank: Optional[AnalogicalFutureBank],
    *,
    anchor_modes: int,
    fde_weight: float,
    temperature: float,
) -> torch.Tensor:
    if afc_bank is None:
        return refined.new_tensor(0.0)
    if int(refined.shape[1]) <= 1:
        return refined.new_tensor(0.0)
    _features, _valid_cpu, top_indices = afc_bank._query(batch)
    query_count = int(top_indices.shape[0])
    if query_count <= 0:
        return refined.new_tensor(0.0)
    valid = mask.detach().bool()
    if int(valid.sum().detach().cpu().item()) != query_count:
        raise ValueError(
            f"V59 AFC query mismatch: mask has {int(valid.sum().detach().cpu().item())} valid agents, "
            f"bank returned {query_count}"
        )
    proxies = afc_bank.futures[top_indices].to(device=refined.device, dtype=refined.dtype)
    corrected = refined[:, 1:, ...]
    candidates = corrected.permute(0, 3, 1, 2, 4, 5)[valid]
    anchor = _v59_anchor_mode_mask(
        base,
        ground_truth,
        mask,
        topk=int(anchor_modes),
        fde_weight=float(fde_weight),
    )
    diversity_modes = (~anchor.permute(0, 2, 1)[valid]).bool()
    if int(diversity_modes.sum().detach().cpu().item()) <= 0:
        return refined.new_tensor(0.0)
    num_slots = int(candidates.shape[1])
    num_modes = int(candidates.shape[2])
    candidates_flat = candidates.reshape(query_count, num_slots * num_modes, *candidates.shape[-2:])
    candidate_valid = diversity_modes[:, None, :].expand(query_count, num_slots, num_modes).reshape(query_count, -1)
    query_has_candidate = candidate_valid.any(dim=1)
    if not bool(query_has_candidate.any().detach().cpu().item()):
        return refined.new_tensor(0.0)
    ade_pairwise = torch.linalg.norm(
        candidates_flat[:, :, None, :, :] - proxies[:, None, :, :, :],
        dim=-1,
    ).mean(dim=-1)
    softmin = _masked_softmin(
        ade_pairwise,
        candidate_valid[:, :, None].expand_as(ade_pairwise),
        dim=1,
        temperature=float(temperature),
    )
    finite = torch.isfinite(softmin) & query_has_candidate[:, None]
    if not bool(finite.any().detach().cpu().item()):
        return refined.new_tensor(0.0)
    return softmin[finite].mean()


def _v59_proxy_cluster_centers(proxies: torch.Tensor, *, max_clusters: int) -> torch.Tensor:
    """Greedy farthest-endpoint proxy centers for AFC mode coverage."""
    query_count, proxy_count = [int(item) for item in proxies.shape[:2]]
    keep = max(1, min(int(max_clusters), proxy_count))
    if proxy_count <= keep:
        return proxies[:, :keep]
    endpoints = proxies[:, :, -1, :]
    selected_rows: List[torch.Tensor] = []
    for query_index in range(query_count):
        chosen = [0]
        min_dist = torch.linalg.norm(endpoints[query_index] - endpoints[query_index, 0:1], dim=-1)
        for _ in range(1, keep):
            next_index = int(min_dist.argmax().detach().cpu().item())
            chosen.append(next_index)
            dist = torch.linalg.norm(endpoints[query_index] - endpoints[query_index, next_index : next_index + 1], dim=-1)
            min_dist = torch.minimum(min_dist, dist)
        selected_rows.append(proxies[query_index, torch.as_tensor(chosen, device=proxies.device, dtype=torch.long)])
    return torch.stack(selected_rows, dim=0)


def _v59b_afc_mode_coverage_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    afc_bank: Optional[AnalogicalFutureBank],
    *,
    anchor_modes: int,
    fde_weight: float,
    coverage_temperature: float,
    precision_temperature: float,
    cluster_count: int,
    precision_weight: float,
    entropy_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    zero = refined.new_tensor(0.0)
    empty_metrics = {
        "loss_v59_afc_coverage": 0.0,
        "loss_v59_afc_precision": 0.0,
        "loss_v59_afc_entropy": 0.0,
        "v59_afc_cluster_count": 0.0,
    }
    if afc_bank is None or int(refined.shape[1]) <= 1:
        return zero, empty_metrics

    _features, _valid_cpu, top_indices = afc_bank._query(batch)
    query_count = int(top_indices.shape[0])
    if query_count <= 0:
        return zero, empty_metrics
    valid = mask.detach().bool()
    if int(valid.sum().detach().cpu().item()) != query_count:
        raise ValueError(
            f"V59B AFC query mismatch: mask has {int(valid.sum().detach().cpu().item())} valid agents, "
            f"bank returned {query_count}"
        )

    proxies = afc_bank.futures[top_indices].to(device=refined.device, dtype=refined.dtype)
    centers = _v59_proxy_cluster_centers(proxies, max_clusters=int(cluster_count))
    center_count = int(centers.shape[1])
    if center_count <= 0:
        return zero, empty_metrics

    corrected = refined[:, 1:, ...]
    candidates = corrected.permute(0, 3, 1, 2, 4, 5)[valid]
    anchor = _v59_anchor_mode_mask(
        base,
        ground_truth,
        mask,
        topk=int(anchor_modes),
        fde_weight=float(fde_weight),
    )
    diversity_modes = (~anchor.permute(0, 2, 1)[valid]).bool()
    if int(diversity_modes.sum().detach().cpu().item()) <= 0:
        return zero, empty_metrics

    num_slots = int(candidates.shape[1])
    num_modes = int(candidates.shape[2])
    candidates_flat = candidates.reshape(query_count, num_slots * num_modes, *candidates.shape[-2:])
    candidate_valid = diversity_modes[:, None, :].expand(query_count, num_slots, num_modes).reshape(query_count, -1)
    query_has_candidate = candidate_valid.any(dim=1)
    if not bool(query_has_candidate.any().detach().cpu().item()):
        return zero, empty_metrics

    ade_pairwise = torch.linalg.norm(
        candidates_flat[:, :, None, :, :] - centers[:, None, :, :, :],
        dim=-1,
    ).mean(dim=-1)

    coverage_scores = _masked_softmin(
        ade_pairwise,
        candidate_valid[:, :, None].expand_as(ade_pairwise),
        dim=1,
        temperature=float(coverage_temperature),
    )
    coverage_loss = coverage_scores[query_has_candidate].mean()

    precision_scores = _masked_softmin(
        ade_pairwise,
        torch.ones_like(ade_pairwise, dtype=torch.bool),
        dim=2,
        temperature=float(precision_temperature),
    )
    precision_loss = precision_scores[candidate_valid].mean() if bool(candidate_valid.any().detach().cpu().item()) else zero

    assignment_temperature = max(float(precision_temperature), 1e-6)
    assignments = torch.softmax(-ade_pairwise / assignment_temperature, dim=2)
    assignments = assignments * candidate_valid[:, :, None].to(dtype=assignments.dtype)
    candidate_count = candidate_valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=assignments.dtype)
    mass = assignments.sum(dim=1) / candidate_count
    target = torch.full_like(mass, fill_value=1.0 / max(center_count, 1))
    entropy_loss = ((mass - target) ** 2).sum(dim=1)[query_has_candidate].mean()

    loss = coverage_loss + float(precision_weight) * precision_loss + float(entropy_weight) * entropy_loss
    metrics = {
        "loss_v59_afc_coverage": float(coverage_loss.detach().cpu()),
        "loss_v59_afc_precision": float(precision_loss.detach().cpu()),
        "loss_v59_afc_entropy": float(entropy_loss.detach().cpu()),
        "v59_afc_cluster_count": float(center_count),
    }
    return loss, metrics


def _v59_base_preserve_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    *,
    corrected_weight: float = 0.0,
) -> torch.Tensor:
    slot0 = refined[:, 0]
    error = torch.linalg.norm(slot0 - base, dim=-1).mean(dim=-1).mean(dim=1)
    if float(corrected_weight) > 0.0 and int(refined.shape[1]) > 1:
        corrected_error = torch.linalg.norm(refined[:, 1:] - base[:, None, ...], dim=-1).mean(dim=-1).mean(dim=2).mean(dim=1)
        error = error + float(corrected_weight) * corrected_error
    return _masked_mean(error, mask)


def _v59_anchor_keep_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchor_modes: int,
    fde_weight: float,
) -> torch.Tensor:
    anchor = _v59_anchor_mode_mask(
        base,
        ground_truth,
        mask,
        topk=int(anchor_modes),
        fde_weight=float(fde_weight),
    )
    if not bool(anchor.any().detach().cpu().item()):
        return refined.new_tensor(0.0)
    error = torch.linalg.norm(refined - base[:, None, ...], dim=-1).mean(dim=-1)
    valid = anchor[:, None, :, :].expand_as(error)
    return error[valid].mean()


def _v59_spread_floor_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    *,
    endpoint_ratio: float,
    trajectory_ratio: float,
) -> torch.Tensor:
    valid = mask.bool()
    if not bool(valid.any().detach().cpu().item()):
        return refined.new_tensor(0.0)
    base_endpoint = _endpoint_spread(base).detach()
    base_trajectory = _trajectory_spread(base).detach()
    losses: List[torch.Tensor] = []
    for slot_index in range(int(refined.shape[1])):
        slot_pred = refined[:, slot_index]
        if float(endpoint_ratio) > 0.0:
            endpoint_floor = float(endpoint_ratio) * base_endpoint
            endpoint_loss = F.relu(endpoint_floor - _endpoint_spread(slot_pred))
            losses.append(_masked_mean(endpoint_loss, mask))
        if float(trajectory_ratio) > 0.0:
            trajectory_floor = float(trajectory_ratio) * base_trajectory
            trajectory_loss = F.relu(trajectory_floor - _trajectory_spread(slot_pred))
            losses.append(_masked_mean(trajectory_loss, mask))
    if not losses:
        return refined.new_tensor(0.0)
    return torch.stack(losses).mean()


def _set_endpoint_spread_flat(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 4:
        raise ValueError(f"Expected prediction [Q,K,T,2], got {tuple(prediction.shape)}")
    endpoints = prediction[..., -1, :]
    pairwise = torch.cdist(endpoints, endpoints, p=2)
    return _slot_offdiag_mean(pairwise)


def _set_trajectory_spread_flat(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 4:
        raise ValueError(f"Expected prediction [Q,K,T,2], got {tuple(prediction.shape)}")
    pairwise = torch.linalg.norm(
        prediction[:, :, None, :, :] - prediction[:, None, :, :, :],
        dim=-1,
    ).mean(dim=-1)
    return _slot_offdiag_mean(pairwise)


def _v60_gather_assigned_centers(centers: torch.Tensor, num_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    query_count, center_count, future_frames, coord_dim = [int(item) for item in centers.shape]
    role_ids = torch.arange(num_modes, device=centers.device, dtype=torch.long) % max(center_count, 1)
    role_ids = role_ids[None, :].expand(query_count, num_modes)
    gather_index = role_ids[:, :, None, None].expand(query_count, num_modes, future_frames, coord_dim)
    assigned = torch.gather(centers, dim=1, index=gather_index)
    return assigned, role_ids


def _v60_afc_role_transport_losses(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    afc_bank: Optional[AnalogicalFutureBank],
    *,
    anchor_modes: int,
    fde_weight: float,
    cluster_count: int,
    role_temperature: float,
    base_identity_margin: float,
    spread_floor_endpoint_ratio: float,
    spread_floor_trajectory_ratio: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    zero = refined.new_tensor(0.0)
    losses = {
        "loss_v60_afc_role": zero,
        "loss_v60_base_identity": zero,
        "loss_v60_spread_floor": zero,
    }
    metrics = {
        "v60_role_center_count": 0.0,
        "v60_diversity_mode_ratio": 0.0,
        "v60_selected_endpoint_ratio": 0.0,
        "v60_selected_trajectory_ratio": 0.0,
    }
    if afc_bank is None or int(refined.shape[1]) <= 1:
        return losses, metrics

    _features, _valid_cpu, top_indices = afc_bank._query(batch)
    query_count = int(top_indices.shape[0])
    if query_count <= 0:
        return losses, metrics
    valid = mask.detach().bool()
    if int(valid.sum().detach().cpu().item()) != query_count:
        raise ValueError(
            f"V60 AFC query mismatch: mask has {int(valid.sum().detach().cpu().item())} valid agents, "
            f"bank returned {query_count}"
        )

    proxies = afc_bank.futures[top_indices].to(device=refined.device, dtype=refined.dtype)
    centers = _v59_proxy_cluster_centers(proxies, max_clusters=int(cluster_count))
    center_count = int(centers.shape[1])
    if center_count <= 0:
        return losses, metrics

    corrected = refined[:, 1:, ...]
    candidates = corrected.permute(0, 3, 1, 2, 4, 5)[valid]
    base_by_agent = base.permute(0, 2, 1, 3, 4)[valid]
    num_slots = int(candidates.shape[1])
    num_modes = int(candidates.shape[2])
    assigned_centers, _role_ids = _v60_gather_assigned_centers(centers, num_modes=num_modes)

    anchor = _v59_anchor_mode_mask(
        base,
        ground_truth,
        mask,
        topk=int(anchor_modes),
        fde_weight=float(fde_weight),
    )
    diversity_modes = (~anchor.permute(0, 2, 1)[valid]).bool()
    diversity_count = int(diversity_modes.sum().detach().cpu().item())
    if diversity_count <= 0:
        return losses, metrics

    ade_to_role = torch.linalg.norm(
        candidates - assigned_centers[:, None, ...],
        dim=-1,
    ).mean(dim=-1)
    if float(role_temperature) <= 0.0:
        role_per_mode = ade_to_role.min(dim=1).values
    else:
        role_per_mode = _masked_softmin(
            ade_to_role,
            torch.ones_like(ade_to_role, dtype=torch.bool),
            dim=1,
            temperature=float(role_temperature),
        )
    loss_role = role_per_mode[diversity_modes].mean()

    best_slot = ade_to_role.argmin(dim=1)
    gather_index = best_slot[:, None, :, None, None].expand(
        query_count,
        1,
        num_modes,
        int(candidates.shape[-2]),
        int(candidates.shape[-1]),
    )
    selected_corr = torch.gather(candidates, dim=1, index=gather_index).squeeze(1)
    selected_set = torch.where(diversity_modes[:, :, None, None], selected_corr, base_by_agent)

    base_identity = torch.linalg.norm(selected_corr - base_by_agent, dim=-1).mean(dim=-1)
    loss_base_identity = F.relu(base_identity - float(base_identity_margin))[diversity_modes].mean()

    spread_losses: List[torch.Tensor] = []
    endpoint_ratio = torch.zeros((query_count,), device=refined.device, dtype=refined.dtype)
    trajectory_ratio = torch.zeros((query_count,), device=refined.device, dtype=refined.dtype)
    base_endpoint = _set_endpoint_spread_flat(base_by_agent).detach()
    base_trajectory = _set_trajectory_spread_flat(base_by_agent).detach()
    selected_endpoint = _set_endpoint_spread_flat(selected_set)
    selected_trajectory = _set_trajectory_spread_flat(selected_set)
    endpoint_ratio = selected_endpoint / base_endpoint.abs().clamp_min(1e-8)
    trajectory_ratio = selected_trajectory / base_trajectory.abs().clamp_min(1e-8)
    if float(spread_floor_endpoint_ratio) > 0.0:
        spread_losses.append(F.relu(float(spread_floor_endpoint_ratio) * base_endpoint - selected_endpoint).mean())
    if float(spread_floor_trajectory_ratio) > 0.0:
        spread_losses.append(F.relu(float(spread_floor_trajectory_ratio) * base_trajectory - selected_trajectory).mean())
    loss_spread_floor = torch.stack(spread_losses).mean() if spread_losses else zero

    losses = {
        "loss_v60_afc_role": loss_role,
        "loss_v60_base_identity": loss_base_identity,
        "loss_v60_spread_floor": loss_spread_floor,
    }
    metrics = {
        "v60_role_center_count": float(center_count),
        "v60_diversity_mode_ratio": float(diversity_modes.to(dtype=torch.float32).mean().detach().cpu()),
        "v60_selected_endpoint_ratio": float(endpoint_ratio.mean().detach().cpu()),
        "v60_selected_trajectory_ratio": float(trajectory_ratio.mean().detach().cpu()),
    }
    return losses, metrics


def _v59_motion_risk_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    *,
    velocity_delta_max: float,
    accel_delta_max: float,
) -> torch.Tensor:
    corrected = refined[:, 1:] if int(refined.shape[1]) > 1 else refined
    base_expanded = base[:, None, ...]
    terms: List[torch.Tensor] = []
    if float(velocity_delta_max) > 0.0 and int(corrected.shape[-2]) > 1:
        corr_vel = corrected[..., 1:, :] - corrected[..., :-1, :]
        base_vel = base_expanded[..., 1:, :] - base_expanded[..., :-1, :]
        vel_delta = torch.linalg.norm(corr_vel - base_vel, dim=-1).mean(dim=-1).mean(dim=2).mean(dim=1)
        terms.append(F.relu(vel_delta - float(velocity_delta_max)))
    if float(accel_delta_max) > 0.0 and int(corrected.shape[-2]) > 2:
        corr_vel = corrected[..., 1:, :] - corrected[..., :-1, :]
        base_vel = base_expanded[..., 1:, :] - base_expanded[..., :-1, :]
        corr_accel = corr_vel[..., 1:, :] - corr_vel[..., :-1, :]
        base_accel = base_vel[..., 1:, :] - base_vel[..., :-1, :]
        accel_delta = torch.linalg.norm(corr_accel - base_accel, dim=-1).mean(dim=-1).mean(dim=2).mean(dim=1)
        terms.append(F.relu(accel_delta - float(accel_delta_max)))
    if not terms:
        return refined.new_tensor(0.0)
    return _masked_mean(sum(terms), mask)


def _slot_offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    num_slots = int(pairwise.shape[-1])
    if num_slots <= 1:
        return pairwise.new_zeros((pairwise.shape[0],))
    keep = ~torch.eye(num_slots, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _slot_spread_loss(
    refined: torch.Tensor,
    mask: torch.Tensor,
    *,
    endpoint_target: float,
    trajectory_target: float,
) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    batch_size, num_slots, num_modes, num_agents, num_steps, coord_dim = refined.shape
    if int(num_slots) <= 1:
        return refined.new_tensor(0.0)

    endpoint_loss = refined.new_zeros((batch_size, num_agents))
    if float(endpoint_target) > 0.0:
        endpoints = refined[..., -1, :].permute(0, 2, 3, 1, 4).reshape(
            batch_size * num_modes * num_agents,
            num_slots,
            coord_dim,
        )
        endpoint_spread = _slot_offdiag_mean(torch.cdist(endpoints, endpoints, p=2)).reshape(
            batch_size,
            num_modes,
            num_agents,
        )
        endpoint_loss = F.relu(float(endpoint_target) - endpoint_spread).mean(dim=1)

    trajectory_loss = refined.new_zeros((batch_size, num_agents))
    if float(trajectory_target) > 0.0:
        trajectories = refined.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size * num_modes * num_agents,
            num_slots,
            num_steps,
            coord_dim,
        )
        pairwise = torch.linalg.norm(
            trajectories[:, :, None, :, :] - trajectories[:, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)
        trajectory_spread = _slot_offdiag_mean(pairwise).reshape(batch_size, num_modes, num_agents)
        trajectory_loss = F.relu(float(trajectory_target) - trajectory_spread).mean(dim=1)

    return _masked_mean(endpoint_loss + trajectory_loss, mask)


def _temporal_energy_risk(temporal_energy: torch.Tensor, *, distance_scale: float) -> torch.Tensor:
    if temporal_energy.ndim != 5 or int(temporal_energy.shape[-1]) < 5:
        raise ValueError(f"temporal energy must have shape [B,K,A,T,C>=5], got {tuple(temporal_energy.shape)}")
    min_neighbor_distance = temporal_energy[..., 0].clamp_min(0.0)
    soft_collision_energy = temporal_energy[..., 1].clamp_min(0.0)
    close_neighbor_count = temporal_energy[..., 2].clamp_min(0.0)
    approaching_score = temporal_energy[..., 3].clamp_min(0.0)
    endpoint_crowding_energy = temporal_energy[..., 4].clamp_min(0.0)
    distance_risk = torch.exp(-min_neighbor_distance / max(float(distance_scale), 1e-6))
    soft_risk = soft_collision_energy / (1.0 + soft_collision_energy)
    close_risk = close_neighbor_count / (1.0 + close_neighbor_count)
    endpoint_risk = endpoint_crowding_energy / (1.0 + endpoint_crowding_energy)
    risk = torch.stack(
        [distance_risk, soft_risk, close_risk, approaching_score.clamp(0.0, 1.0), endpoint_risk],
        dim=0,
    ).amax(dim=0)
    return torch.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _energy_recon_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    risk_floor: float,
    distance_scale: float,
) -> torch.Tensor:
    best_index = _base_best_index(base, ground_truth)
    selected = _gather_modes(refined, best_index, mode_dim=2)
    risk = _temporal_energy_risk(
        temporal_energy.to(device=refined.device, dtype=refined.dtype),
        distance_scale=float(distance_scale),
    )
    selected_risk = torch.gather(risk, dim=1, index=best_index[:, None, :, None].expand(-1, 1, -1, risk.shape[-1])).squeeze(1)
    error = torch.linalg.norm(selected - ground_truth[:, None, ...], dim=-1)
    weighted = error * selected_risk[:, None, :, :].clamp_min(float(risk_floor))
    score = weighted.mean(dim=-1) + float(fde_weight) * weighted[..., -1]
    return _masked_mean(score.min(dim=1).values, mask)


def _keep_loss(refined: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = refined[:, 0]
    error = torch.linalg.norm(first - base, dim=-1).mean(dim=-1).mean(dim=1)
    return _masked_mean(error, mask)


def _non_best_mode_mask(base: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num_modes = int(base.shape[1])
    if num_modes <= 1:
        return torch.zeros(base.shape[0], num_modes, base.shape[2], dtype=torch.bool, device=base.device)
    best_index = _base_best_index(base, ground_truth)
    mode_mask = torch.ones(base.shape[0], num_modes, base.shape[2], dtype=torch.bool, device=base.device)
    mode_mask.scatter_(1, best_index[:, None, :], False)
    return mode_mask & mask[:, None, :].bool()


def _mode_masked_mean(values: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    weight = mode_mask.to(device=values.device, dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def _non_best_keep_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    first = refined[:, 0]
    mode_mask = _non_best_mode_mask(base, ground_truth, mask)
    error = torch.linalg.norm(first - base, dim=-1).mean(dim=-1)
    return _mode_masked_mean(error, mode_mask)


def _non_best_nohurt_loss(
    refined: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fde_weight: float,
    margin: float,
) -> torch.Tensor:
    first = refined[:, 0]
    mode_mask = _non_best_mode_mask(base, ground_truth, mask)
    base_score = _score(base, ground_truth, fde_weight=fde_weight)
    refined_score = _score(first, ground_truth, fde_weight=fde_weight)
    hurt = F.relu(refined_score - base_score - float(margin))
    return _mode_masked_mean(hurt, mode_mask)


def _delta_l2_loss(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l2 = torch.linalg.norm(delta, dim=-1).mean(dim=-1).mean(dim=2).mean(dim=1)
    return _masked_mean(l2, mask)


def _dynamic_slot_offset_l2_loss(outputs: Mapping[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    offset = outputs.get("dynamic_slot_offset")
    if not torch.is_tensor(offset):
        delta = outputs.get("delta")
        if torch.is_tensor(delta):
            return delta.new_tensor(0.0)
        return mask.new_tensor(0.0, dtype=torch.float32)
    offset_norm = torch.linalg.norm(offset, dim=-1).mean(dim=1).mean(dim=1)
    return _masked_mean(offset_norm, mask)


def _summarize_refinement(
    prediction: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    miss_threshold: float,
    outputs: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    pred_set = _flatten_draw_modes(prediction)
    pred_first = prediction[:, 0]
    pred_errors = displacement_errors(pred_set, ground_truth, agent_mask=mask)
    first_errors = displacement_errors(pred_first, ground_truth, agent_mask=mask)
    base_errors = displacement_errors(base, ground_truth, agent_mask=mask)
    valid = pred_errors["valid_agents"].bool()
    valid_expanded = valid[:, None, :]
    inf = torch.tensor(float("inf"), device=prediction.device, dtype=prediction.dtype)
    pred_ade = pred_errors["ade_per_mode_agent"]
    pred_fde = pred_errors["fde_per_mode_agent"]
    first_ade = first_errors["ade_per_mode_agent"]
    first_fde = first_errors["fde_per_mode_agent"]
    base_fde = base_errors["fde_per_mode_agent"]
    pred_fde_min = pred_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    base_fde_min = base_fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    base_best_index = base_fde.argmin(dim=1)
    base_best_fde = torch.gather(base_fde, dim=1, index=base_best_index[:, None, :]).squeeze(1)
    pred_at_base_best_fde = torch.gather(first_fde, dim=1, index=base_best_index[:, None, :]).squeeze(1)
    base_best_delta = pred_at_base_best_fde - base_best_fde
    base_miss = base_fde_min > float(miss_threshold)
    pred_miss = pred_fde_min > float(miss_threshold)
    result = {
        "refined_ADE_min": float(pred_ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu()),
        "refined_FDE_min": float(pred_fde_min[valid].mean().cpu()),
        "refined_ADE_avg": float(pred_ade.mean(dim=1)[valid].mean().cpu()),
        "refined_FDE_avg": float(pred_fde.mean(dim=1)[valid].mean().cpu()),
        "refined_MissRate": float(pred_miss[valid].float().mean().cpu()),
        "base_FDE_min": float(base_fde_min[valid].mean().cpu()),
        "base_MissRate": float(base_miss[valid].float().mean().cpu()),
        "dFDE_min": float((pred_fde_min - base_fde_min)[valid].mean().cpu()),
        "dMissRate": float((pred_miss[valid].float() - base_miss[valid].float()).mean().cpu()),
        "base_best_hurt_mean": float(F.relu(base_best_delta[valid]).mean().cpu()),
        "base_best_worse_rate": float((base_best_delta[valid] > 0.0).float().mean().cpu()),
        "endpoint_ratio": float((_endpoint_spread(pred_first) / _endpoint_spread(base).abs().clamp_min(1e-8))[valid].mean().cpu()),
        "trajectory_ratio": float((_trajectory_spread(pred_first) / _trajectory_spread(base).abs().clamp_min(1e-8))[valid].mean().cpu()),
        "evaluated_modes": float(pred_set.shape[1]),
    }
    if int(prediction.shape[1]) > 1:
        result["slot0_ADE_min"] = float(first_ade.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu())
        result["slot0_FDE_min"] = float(first_fde.masked_fill(~valid_expanded, inf).min(dim=1).values[valid].mean().cpu())
        result["slot0_ADE_avg"] = float(first_ade.mean(dim=1)[valid].mean().cpu())
        result["slot0_FDE_avg"] = float(first_fde.mean(dim=1)[valid].mean().cpu())
    if outputs is not None:
        delta = outputs.get("delta")
        if torch.is_tensor(delta):
            result["delta_l2_mean"] = float(torch.linalg.norm(delta[:, 0], dim=-1).mean(dim=-1).mean(dim=1)[valid].mean().cpu())
        prior_logvar = outputs.get("prior_logvar")
        if torch.is_tensor(prior_logvar):
            prior_std = torch.exp(0.5 * prior_logvar).mean(dim=-1)
            result["prior_std_mean"] = float(prior_std[valid[:, None, :].expand_as(prior_std)].mean().cpu())
        dynamic_slot_offset = outputs.get("dynamic_slot_offset")
        if torch.is_tensor(dynamic_slot_offset):
            offset_norm = torch.linalg.norm(dynamic_slot_offset, dim=-1).mean(dim=1).mean(dim=1)
            result["dynamic_slot_offset_l2_mean"] = float(offset_norm[valid].mean().cpu())
    return result


def _selection_score(metrics: Mapping[str, float], args: argparse.Namespace) -> float:
    if args.selection_metric == "fde_min":
        return float(metrics["refined_FDE_min"])
    endpoint_penalty = max(0.0, 1.0 - float(metrics.get("endpoint_ratio", 1.0)))
    trajectory_penalty = max(0.0, 1.0 - float(metrics.get("trajectory_ratio", 1.0)))
    return (
        float(metrics["refined_FDE_min"])
        + float(args.selection_miss_weight) * max(0.0, float(metrics.get("dMissRate", 0.0)))
        + float(args.selection_nohurt_weight) * float(metrics.get("base_best_hurt_mean", 0.0))
        + float(args.selection_diversity_weight) * (endpoint_penalty + trajectory_penalty)
    )


def _loss_step(
    model: SocialCVAETeacherRefiner,
    batch: Mapping[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    semantic_prototypes: Optional[torch.Tensor] = None,
    elite_teacher: Optional[SocialCVAETeacherRefiner] = None,
    afc_bank: Optional[AnalogicalFutureBank] = None,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    base = batch["teacher_pred"]
    gt = batch["ground_truth"]
    mask = batch["agent_mask"].bool()
    energy = batch["teacher_temporal_interaction_energy_features"]
    is_set_generator = _is_set_generator_variant(args)
    is_prior_generator = _is_prior_generator_variant(args)
    variant = str(args.variant)
    use_quality_wta = variant in {
        "v56a",
        "v56c1",
        "v56c2",
        "v56c3",
        "v57a",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
    }
    use_slot0_preserve = variant in {"v56a", "v56c2"}
    use_margin_gain = variant in {"v56a", "v56c3"}
    use_density = variant == "v56a"
    use_residual_norm_band = variant in {
        "v56a",
        "v56c3",
        "v57a",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
    }
    use_semantic_prototype = variant in {"v57a", "v58b1"} or (
        variant in {"v58d3", "v58f2"} and float(args.lambda_semantic_prototype) > 0.0
    )
    use_semantic_slot0_identity = variant in {
        "v57a",
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
    }
    use_dynamic_slot_offset = variant in {
        "v58b1",
        "v58b2",
        "v58d1",
        "v58d2",
        "v58d3",
        "v58f1",
        "v58f2",
        "v59a",
        "v59b",
        "v59c",
        "v60a",
    }
    use_elite_teacher_distill = variant == "v58a1" and float(args.lambda_elite_teacher_distill) > 0.0
    use_elite_set_distill = variant in {"v58b2", "v58d1", "v58d2", "v58d3", "v58f1", "v58f2"} and float(args.lambda_elite_set_distill) > 0.0
    use_front_elite_distill = variant in {"v58d1", "v58d2", "v58d3", "v58f1", "v58f2"} and float(args.lambda_front_elite_distill) > 0.0
    if is_set_generator:
        num_training_samples = int(args.set_residual_slots)
    elif is_prior_generator:
        num_training_samples = 1
    else:
        num_training_samples = int(args.posterior_samples)
    training_z_mode = "slots" if is_set_generator else "mean" if is_prior_generator else "sample"
    train_with_prior = is_prior_generator or variant == "v60a"
    outputs = model(
        base,
        past_traj_original_scale=batch["past_traj_original_scale"],
        temporal_energy_features=energy,
        ground_truth=None if train_with_prior else gt,
        num_samples=num_training_samples,
        z_source="prior" if train_with_prior else "posterior",
        z_mode=training_z_mode,
    )
    refined = outputs["refined"]
    best_index = _base_best_index(base, gt)
    loss_recon = _base_best_recon_loss(refined, base, gt, mask, fde_weight=float(args.fde_weight))
    loss_gt = _gt_min_loss(refined, gt, mask, fde_weight=float(args.fde_weight))
    loss_set_coverage = _gt_softmin_loss(
        refined,
        gt,
        mask,
        fde_weight=float(args.fde_weight),
        temperature=float(args.set_coverage_temperature),
    )
    loss_energy = _energy_recon_loss(
        refined,
        base,
        gt,
        energy,
        mask,
        fde_weight=float(args.fde_weight),
        risk_floor=float(args.energy_risk_floor),
        distance_scale=float(args.energy_distance_scale),
    )
    loss_kl = _selected_kl_loss(outputs, best_index, mask)
    loss_base_best = _student_best_nohurt_loss(
        refined,
        base,
        gt,
        mask,
        margin=float(args.base_best_nohurt_margin),
    )
    loss_good = _good_nohurt_loss(
        refined,
        base,
        gt,
        mask,
        fde_weight=float(args.fde_weight),
        good_frac=float(args.good_nohurt_frac),
        margin=float(args.good_nohurt_margin),
    )
    loss_diversity = _diversity_loss(
        refined,
        base,
        mask,
        target_ratio=float(args.diversity_preserve_target_ratio),
    )
    loss_keep = _keep_loss(refined, base, mask)
    loss_non_best_keep = _non_best_keep_loss(refined, base, gt, mask)
    loss_non_best_nohurt = _non_best_nohurt_loss(
        refined,
        base,
        gt,
        mask,
        fde_weight=float(args.fde_weight),
        margin=float(args.non_best_nohurt_margin),
    )
    loss_delta = _delta_l2_loss(outputs["delta"], mask)
    loss_dynamic_slot_offset_l2 = _dynamic_slot_offset_l2_loss(outputs, mask)
    loss_slot_spread = _slot_spread_loss(
        refined,
        mask,
        endpoint_target=float(args.slot_endpoint_spread_target),
        trajectory_target=float(args.slot_trajectory_spread_target),
    )
    if variant in {"v59a", "v59b", "v59c"}:
        loss_v59_anchor_obs = _v59_anchor_obs_loss(
            refined,
            base,
            gt,
            mask,
            anchor_modes=int(args.v59_anchor_modes),
            fde_weight=float(args.fde_weight),
            temperature=float(args.v59_anchor_temperature),
        )
        if variant in {"v59b", "v59c"}:
            loss_v59_afc, v59_afc_metrics = _v59b_afc_mode_coverage_loss(
                refined,
                base,
                gt,
                batch,
                mask,
                afc_bank,
                anchor_modes=int(args.v59_anchor_modes),
                fde_weight=float(args.fde_weight),
                coverage_temperature=float(args.v59_afc_loss_temperature),
                precision_temperature=float(args.v59_afc_precision_temperature),
                cluster_count=int(args.v59_afc_clusters),
                precision_weight=float(args.lambda_v59_afc_precision),
                entropy_weight=float(args.lambda_v59_afc_entropy),
            )
        else:
            loss_v59_afc = _v59_afc_proxy_loss(
                refined,
                base,
                gt,
                batch,
                mask,
                afc_bank,
                anchor_modes=int(args.v59_anchor_modes),
                fde_weight=float(args.fde_weight),
                temperature=float(args.v59_afc_loss_temperature),
            )
            v59_afc_metrics = {
                "loss_v59_afc_coverage": 0.0,
                "loss_v59_afc_precision": 0.0,
                "loss_v59_afc_entropy": 0.0,
                "v59_afc_cluster_count": 0.0,
            }
        loss_v59_base_preserve = _v59_base_preserve_loss(
            refined,
            base,
            mask,
            corrected_weight=float(args.v59_base_preserve_corrected_weight) if variant in {"v59b", "v59c"} else 0.0,
        )
        if variant == "v59c":
            loss_v59_anchor_keep = _v59_anchor_keep_loss(
                refined,
                base,
                gt,
                mask,
                anchor_modes=int(args.v59_anchor_modes),
                fde_weight=float(args.fde_weight),
            )
            loss_v59_spread_floor = _v59_spread_floor_loss(
                refined,
                base,
                mask,
                endpoint_ratio=float(args.v59_spread_floor_endpoint_ratio),
                trajectory_ratio=float(args.v59_spread_floor_trajectory_ratio),
            )
        else:
            loss_v59_anchor_keep = refined.new_tensor(0.0)
            loss_v59_spread_floor = refined.new_tensor(0.0)
        loss_v59_diversity = loss_diversity
        loss_v59_risk = _v59_motion_risk_loss(
            refined,
            base,
            mask,
            velocity_delta_max=float(args.v59_velocity_delta_max),
            accel_delta_max=float(args.v59_accel_delta_max),
        )
        loss_v59_residual = loss_delta + loss_dynamic_slot_offset_l2
    else:
        loss_v59_anchor_obs = refined.new_tensor(0.0)
        loss_v59_afc = refined.new_tensor(0.0)
        loss_v59_base_preserve = refined.new_tensor(0.0)
        loss_v59_anchor_keep = refined.new_tensor(0.0)
        loss_v59_spread_floor = refined.new_tensor(0.0)
        loss_v59_diversity = refined.new_tensor(0.0)
        loss_v59_risk = refined.new_tensor(0.0)
        loss_v59_residual = refined.new_tensor(0.0)
        v59_afc_metrics = {
            "loss_v59_afc_coverage": 0.0,
            "loss_v59_afc_precision": 0.0,
            "loss_v59_afc_entropy": 0.0,
            "v59_afc_cluster_count": 0.0,
        }
    if variant == "v60a":
        loss_v60_anchor_keep = _v59_anchor_keep_loss(
            refined,
            base,
            gt,
            mask,
            anchor_modes=int(args.v60_anchor_modes),
            fde_weight=float(args.fde_weight),
        )
        v60_losses, v60_metrics = _v60_afc_role_transport_losses(
            refined,
            base,
            gt,
            batch,
            mask,
            afc_bank,
            anchor_modes=int(args.v60_anchor_modes),
            fde_weight=float(args.fde_weight),
            cluster_count=int(args.v60_afc_clusters),
            role_temperature=float(args.v60_role_temperature),
            base_identity_margin=float(args.v60_base_identity_margin),
            spread_floor_endpoint_ratio=float(args.v60_spread_floor_endpoint_ratio),
            spread_floor_trajectory_ratio=float(args.v60_spread_floor_trajectory_ratio),
        )
        loss_v60_afc_role = v60_losses["loss_v60_afc_role"]
        loss_v60_base_identity = v60_losses["loss_v60_base_identity"]
        loss_v60_spread_floor = v60_losses["loss_v60_spread_floor"]
        loss_v60_risk = _v59_motion_risk_loss(
            refined,
            base,
            mask,
            velocity_delta_max=float(args.v59_velocity_delta_max),
            accel_delta_max=float(args.v59_accel_delta_max),
        )
        loss_v60_residual = loss_dynamic_slot_offset_l2
    else:
        loss_v60_anchor_keep = refined.new_tensor(0.0)
        loss_v60_afc_role = refined.new_tensor(0.0)
        loss_v60_base_identity = refined.new_tensor(0.0)
        loss_v60_spread_floor = refined.new_tensor(0.0)
        loss_v60_risk = refined.new_tensor(0.0)
        loss_v60_residual = refined.new_tensor(0.0)
        v60_metrics = {
            "v60_role_center_count": 0.0,
            "v60_diversity_mode_ratio": 0.0,
            "v60_selected_endpoint_ratio": 0.0,
            "v60_selected_trajectory_ratio": 0.0,
        }
    if use_quality_wta:
        loss_elite_soft_wta = _elite_soft_wta_loss(
            refined,
            base,
            gt,
            mask,
            fde_weight=float(args.fde_weight),
            temperature=float(args.elite_soft_temperature),
            base_topk=int(args.elite_base_topk),
        )
    else:
        loss_elite_soft_wta = refined.new_tensor(0.0)
    if use_margin_gain:
        loss_elite_improvement = _elite_improvement_loss(
            refined,
            base,
            gt,
            mask,
            fde_weight=float(args.fde_weight),
            base_topk=int(args.elite_base_topk),
            margin=float(args.elite_improvement_margin),
        )
    else:
        loss_elite_improvement = refined.new_tensor(0.0)
    if use_density:
        loss_elite_density = _elite_density_loss(
            refined,
            base,
            gt,
            mask,
            fde_weight=float(args.fde_weight),
            base_topk=int(args.elite_base_topk),
            density_slots=int(args.elite_density_slots),
            margin=float(args.elite_improvement_margin),
        )
    else:
        loss_elite_density = refined.new_tensor(0.0)
    if use_slot0_preserve:
        loss_slot0_preserve = loss_keep
    else:
        loss_slot0_preserve = refined.new_tensor(0.0)
    if use_residual_norm_band:
        loss_residual_norm_band = _residual_norm_band_loss(
            outputs["delta"],
            mask,
            endpoint_max=float(args.residual_endpoint_norm_max),
            trajectory_max=float(args.residual_trajectory_norm_max),
        )
    else:
        loss_residual_norm_band = refined.new_tensor(0.0)
    if use_semantic_prototype:
        loss_semantic_prototype = _semantic_prototype_alignment_loss(outputs["delta"], semantic_prototypes, mask)
    else:
        loss_semantic_prototype = refined.new_tensor(0.0)
    if use_semantic_slot0_identity:
        loss_semantic_slot0_identity = _slot0_identity_loss(outputs["delta"], mask)
    else:
        loss_semantic_slot0_identity = refined.new_tensor(0.0)
    if use_elite_teacher_distill:
        loss_elite_teacher_distill, elite_teacher_metrics = _elite_teacher_distill_loss(
            refined,
            base,
            gt,
            mask,
            energy,
            batch["past_traj_original_scale"],
            elite_teacher,
            teacher_slots=int(args.elite_teacher_slots),
            fde_weight=float(args.fde_weight),
            min_gain=float(args.elite_distill_min_gain),
        )
    else:
        loss_elite_teacher_distill = refined.new_tensor(0.0)
        elite_teacher_metrics = {
            "elite_teacher_accept_ratio": 0.0,
            "elite_teacher_target_delta_l2": 0.0,
        }
    if use_elite_set_distill:
        loss_elite_set_distill, elite_set_metrics = _elite_residual_set_distill_loss(
            refined,
            base,
            gt,
            mask,
            energy,
            batch["past_traj_original_scale"],
            elite_teacher,
            teacher_slots=int(args.elite_teacher_slots),
            target_topk=int(args.elite_set_topk),
            min_gain=float(args.elite_set_min_gain),
            fde_weight=float(args.fde_weight),
            student_to_teacher_weight=float(args.elite_set_student_to_teacher_weight),
            include_slot0=bool(args.include_slot0_in_elite_set_distill),
        )
    else:
        loss_elite_set_distill = refined.new_tensor(0.0)
        elite_set_metrics = {
            "elite_set_accept_ratio": 0.0,
            "elite_set_targets_per_agent": 0.0,
            "elite_set_target_delta_l2": 0.0,
            "elite_set_target_to_student": 0.0,
            "elite_set_student_to_target": 0.0,
        }
    if use_front_elite_distill:
        front_mode = {
            "v58d1": "best",
            "v58d2": "topk",
            "v58d3": "prototype",
            "v58f1": "best",
            "v58f2": "prototype",
        }[variant]
        loss_front_elite_distill, front_elite_metrics = _front_loaded_elite_distill_loss(
            refined,
            base,
            gt,
            mask,
            energy,
            batch["past_traj_original_scale"],
            elite_teacher,
            semantic_prototypes,
            mode=front_mode,
            teacher_slots=int(args.elite_teacher_slots),
            front_slots=int(args.front_elite_slots),
            target_topk=int(args.front_elite_topk),
            min_gain=float(args.front_elite_min_gain),
            fde_weight=float(args.fde_weight),
            student_to_teacher_weight=float(args.front_elite_student_to_teacher_weight),
        )
    else:
        loss_front_elite_distill = refined.new_tensor(0.0)
        front_elite_metrics = {
            "front_elite_accept_ratio": 0.0,
            "front_elite_targets_per_agent": 0.0,
            "front_elite_target_delta_l2": 0.0,
            "front_elite_target_to_student": 0.0,
            "front_elite_student_to_target": 0.0,
            "front_elite_assigned_slot_mean": 0.0,
            "front_elite_slot1_target_ratio": 0.0,
        }
    gt_weight = 0.0 if is_set_generator else float(args.lambda_gt_min)
    set_coverage_weight = float(args.lambda_set_coverage) if is_set_generator else 0.0
    kl_weight = 0.0 if is_set_generator else float(args.lambda_kl)
    slot_spread_weight = float(args.lambda_slot_spread) if is_set_generator else 0.0
    quality_wta_weight = 1.0 if use_quality_wta else 0.0
    margin_gain_weight = 1.0 if use_margin_gain else 0.0
    density_weight = 1.0 if use_density else 0.0
    slot0_preserve_weight = 1.0 if use_slot0_preserve else 0.0
    residual_norm_band_weight = 1.0 if use_residual_norm_band else 0.0
    semantic_prototype_weight = 1.0 if use_semantic_prototype else 0.0
    semantic_slot0_identity_weight = 1.0 if use_semantic_slot0_identity else 0.0
    dynamic_slot_offset_weight = 1.0 if use_dynamic_slot_offset else 0.0
    if variant == "v60a":
        loss = (
            float(args.lambda_v60_anchor_keep) * loss_v60_anchor_keep
            + float(args.lambda_v60_afc_role) * loss_v60_afc_role
            + float(args.lambda_v60_base_identity) * loss_v60_base_identity
            + float(args.lambda_v60_spread_floor) * loss_v60_spread_floor
            + float(args.lambda_v60_risk) * loss_v60_risk
            + float(args.lambda_v60_residual) * loss_v60_residual
        )
    elif variant in {"v59a", "v59b", "v59c"}:
        loss = (
            float(args.lambda_v59_anchor_obs) * loss_v59_anchor_obs
            + float(args.lambda_v59_afc) * loss_v59_afc
            + float(args.lambda_v59_base_preserve) * loss_v59_base_preserve
            + float(args.lambda_v59_anchor_keep) * loss_v59_anchor_keep
            + float(args.lambda_v59_spread_floor) * loss_v59_spread_floor
            + float(args.lambda_v59_diversity) * loss_v59_diversity
            + float(args.lambda_v59_risk) * loss_v59_risk
            + float(args.lambda_v59_residual) * loss_v59_residual
        )
    else:
        loss = (
            float(args.lambda_recon_best) * loss_recon
            + gt_weight * loss_gt
            + set_coverage_weight * loss_set_coverage
            + float(args.lambda_energy_recon) * loss_energy
            + kl_weight * loss_kl
            + float(args.lambda_base_best_nohurt) * loss_base_best
            + float(args.lambda_good_nohurt) * loss_good
            + float(args.lambda_diversity_preserve) * loss_diversity
            + float(args.lambda_keep) * loss_keep
            + float(args.lambda_non_best_keep) * loss_non_best_keep
            + float(args.lambda_non_best_nohurt) * loss_non_best_nohurt
            + float(args.lambda_delta_l2) * loss_delta
            + slot_spread_weight * loss_slot_spread
            + quality_wta_weight * float(args.lambda_elite_soft_wta) * loss_elite_soft_wta
            + margin_gain_weight * float(args.lambda_elite_improvement) * loss_elite_improvement
            + density_weight * float(args.lambda_elite_density) * loss_elite_density
            + slot0_preserve_weight * float(args.lambda_slot0_preserve) * loss_slot0_preserve
            + residual_norm_band_weight * float(args.lambda_residual_norm_band) * loss_residual_norm_band
            + semantic_prototype_weight * float(args.lambda_semantic_prototype) * loss_semantic_prototype
            + semantic_slot0_identity_weight
            * float(args.lambda_semantic_slot0_identity)
            * loss_semantic_slot0_identity
            + float(args.lambda_elite_teacher_distill) * loss_elite_teacher_distill
            + float(args.lambda_elite_set_distill) * loss_elite_set_distill
            + float(args.lambda_front_elite_distill) * loss_front_elite_distill
            + dynamic_slot_offset_weight * float(args.lambda_dynamic_slot_offset_l2) * loss_dynamic_slot_offset_l2
        )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_gt_min": float(loss_gt.detach().cpu()),
        "loss_set_coverage": float(loss_set_coverage.detach().cpu()),
        "loss_energy": float(loss_energy.detach().cpu()),
        "loss_kl": float(loss_kl.detach().cpu()),
        "loss_base_best_nohurt": float(loss_base_best.detach().cpu()),
        "loss_good_nohurt": float(loss_good.detach().cpu()),
        "loss_diversity": float(loss_diversity.detach().cpu()),
        "loss_keep": float(loss_keep.detach().cpu()),
        "loss_non_best_keep": float(loss_non_best_keep.detach().cpu()),
        "loss_non_best_nohurt": float(loss_non_best_nohurt.detach().cpu()),
        "loss_delta_l2": float(loss_delta.detach().cpu()),
        "loss_slot_spread": float(loss_slot_spread.detach().cpu()),
        "loss_v59_anchor_obs": float(loss_v59_anchor_obs.detach().cpu()),
        "loss_v59_afc": float(loss_v59_afc.detach().cpu()),
        "loss_v59_afc_coverage": float(v59_afc_metrics["loss_v59_afc_coverage"]),
        "loss_v59_afc_precision": float(v59_afc_metrics["loss_v59_afc_precision"]),
        "loss_v59_afc_entropy": float(v59_afc_metrics["loss_v59_afc_entropy"]),
        "v59_afc_cluster_count": float(v59_afc_metrics["v59_afc_cluster_count"]),
        "loss_v59_base_preserve": float(loss_v59_base_preserve.detach().cpu()),
        "loss_v59_anchor_keep": float(loss_v59_anchor_keep.detach().cpu()),
        "loss_v59_spread_floor": float(loss_v59_spread_floor.detach().cpu()),
        "loss_v59_diversity": float(loss_v59_diversity.detach().cpu()),
        "loss_v59_risk": float(loss_v59_risk.detach().cpu()),
        "loss_v59_residual": float(loss_v59_residual.detach().cpu()),
        "loss_v60_anchor_keep": float(loss_v60_anchor_keep.detach().cpu()),
        "loss_v60_afc_role": float(loss_v60_afc_role.detach().cpu()),
        "loss_v60_base_identity": float(loss_v60_base_identity.detach().cpu()),
        "loss_v60_spread_floor": float(loss_v60_spread_floor.detach().cpu()),
        "loss_v60_risk": float(loss_v60_risk.detach().cpu()),
        "loss_v60_residual": float(loss_v60_residual.detach().cpu()),
        "v60_role_center_count": float(v60_metrics["v60_role_center_count"]),
        "v60_diversity_mode_ratio": float(v60_metrics["v60_diversity_mode_ratio"]),
        "v60_selected_endpoint_ratio": float(v60_metrics["v60_selected_endpoint_ratio"]),
        "v60_selected_trajectory_ratio": float(v60_metrics["v60_selected_trajectory_ratio"]),
        "loss_elite_soft_wta": float(loss_elite_soft_wta.detach().cpu()),
        "loss_elite_improvement": float(loss_elite_improvement.detach().cpu()),
        "loss_elite_density": float(loss_elite_density.detach().cpu()),
        "loss_slot0_preserve": float(loss_slot0_preserve.detach().cpu()),
        "loss_residual_norm_band": float(loss_residual_norm_band.detach().cpu()),
        "loss_semantic_prototype": float(loss_semantic_prototype.detach().cpu()),
        "loss_semantic_slot0_identity": float(loss_semantic_slot0_identity.detach().cpu()),
        "loss_elite_teacher_distill": float(loss_elite_teacher_distill.detach().cpu()),
        "loss_elite_set_distill": float(loss_elite_set_distill.detach().cpu()),
        "loss_front_elite_distill": float(loss_front_elite_distill.detach().cpu()),
        "loss_dynamic_slot_offset_l2": float(loss_dynamic_slot_offset_l2.detach().cpu()),
        **elite_teacher_metrics,
        **elite_set_metrics,
        **front_elite_metrics,
    }
    return loss, metrics, outputs


@torch.no_grad()
def _eval_loader(
    model: SocialCVAETeacherRefiner,
    loader: DataLoader,
    *,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    summaries: List[Dict[str, float]] = []
    weights: List[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        is_set_generator = _is_set_generator_variant(args)
        outputs = model.refine(
            batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            num_samples=int(args.set_residual_slots if is_set_generator else 1),
            z_mode=str(args.eval_z_mode),
        )
        summary = _summarize_refinement(
            outputs["refined"],
            batch["teacher_pred"],
            batch["ground_truth"],
            batch["agent_mask"].bool(),
            miss_threshold=float(args.miss_threshold),
            outputs=outputs,
        )
        summaries.append(summary)
        weights.append(int(batch["agent_mask"].bool().sum().item()))
    if not summaries:
        return {}
    total = max(sum(weights), 1)
    result: Dict[str, float] = {}
    for key in summaries[0].keys():
        result[key] = float(sum(summary[key] * weight for summary, weight in zip(summaries, weights)) / total)
    return result


def _mean_metrics(items: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(items)
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}


def main() -> None:
    args = build_parser().parse_args()
    _validate_variant_args(args)
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    cache_path = Path(args.cache_path).expanduser().resolve()
    payload = _load_cache(cache_path)
    tensors = _prepare_refiner_tensors(payload, args=args)
    num_items = int(tensors["ground_truth"].shape[0])
    num_agents = int(tensors["agent_mask"].shape[1])
    tensors["afc_source_id"] = torch.arange(num_items, dtype=torch.long)[:, None].expand(num_items, num_agents).clone()
    train_indices, val_indices = _select_indices(
        num_items,
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )
    afc_bank = _build_cache_afc_bank(tensors, train_indices, args=args)
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
    teacher_shape = tensors["teacher_pred"].shape
    past_shape = tensors["past_traj_original_scale"].shape
    energy_shape = tensors["teacher_temporal_interaction_energy_features"].shape
    config = SocialCVAETeacherRefinerConfig(
        future_frames=int(teacher_shape[-2]),
        coord_dim=int(teacher_shape[-1]),
        past_frames=int(past_shape[-2]),
        past_feature_dim=int(past_shape[-1]),
        temporal_energy_dim=int(energy_shape[-1]),
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        max_modes=int(teacher_shape[1]),
        use_mode_embedding=not bool(args.no_mode_embedding),
        residual_scale=float(args.residual_scale),
        max_delta=None if float(args.max_delta) <= 0.0 else float(args.max_delta),
        use_energy_risk_map=bool(args.use_energy_risk_map),
        energy_risk_distance_scale=float(args.energy_risk_distance_scale),
        use_temporal_energy_encoder=bool(args.use_temporal_energy_encoder),
        energy_temporal_hidden_dim=int(args.energy_temporal_hidden_dim),
        decoder_hidden_dim=int(args.decoder_hidden_dim),
        decoder_layers=int(args.decoder_layers),
        use_energy_conditioned_generator=str(args.variant) == "v37a",
        use_set_generator=_is_set_generator_variant(args),
        max_residual_slots=int(args.set_residual_slots if _is_set_generator_variant(args) else 1),
        set_slot_scale=float(args.set_slot_scale),
        use_dynamic_slot_offsets=str(args.variant)
        in {"v58b1", "v58b2", "v58d1", "v58d2", "v58d3", "v58f1", "v58f2", "v59a", "v59b", "v59c", "v60a"},
        dynamic_slot_hidden_dim=int(args.dynamic_slot_hidden_dim),
        dynamic_slot_offset_scale=float(args.dynamic_slot_offset_scale),
        dynamic_slot0_zero=not bool(args.allow_dynamic_slot0),
    )
    model = SocialCVAETeacherRefiner(config).to(device)
    semantic_prototypes = _load_semantic_prototypes(args.semantic_prototype_path, device=device)
    if semantic_prototypes is not None:
        if str(args.variant) in {"v58d3", "v58f2"}:
            expected_min = int(args.front_elite_slots)
            if int(semantic_prototypes.shape[0]) < expected_min:
                raise SystemExit(
                    f"--semantic-prototype-path contains {int(semantic_prototypes.shape[0])} prototypes, "
                    f"but --front-elite-slots {expected_min} requires at least {expected_min}"
                )
        else:
            expected = int(args.set_residual_slots) - 1
            if int(semantic_prototypes.shape[0]) != expected:
                raise SystemExit(
                    f"--semantic-prototype-path contains {int(semantic_prototypes.shape[0])} prototypes, "
                    f"but --set-residual-slots {int(args.set_residual_slots)} expects {expected}"
                )
        if int(semantic_prototypes.shape[1]) != int(teacher_shape[-2]):
            raise SystemExit(
                f"--semantic-prototype-path future length {int(semantic_prototypes.shape[1])} "
                f"does not match cache future length {int(teacher_shape[-2])}"
            )
    elite_teacher: Optional[SocialCVAETeacherRefiner] = None
    elite_teacher_path: Optional[Path] = None
    if args.elite_teacher_checkpoint:
        elite_teacher_path = Path(args.elite_teacher_checkpoint).expanduser().resolve()
        elite_teacher = load_social_cvae_teacher_refiner(elite_teacher_path, map_location=device).to(device)
        elite_teacher.eval()
        for parameter in elite_teacher.parameters():
            parameter.requires_grad_(False)
        if not bool(getattr(elite_teacher.config, "use_set_generator", False)):
            raise SystemExit("--elite-teacher-checkpoint must be a set-generator checkpoint")
        if int(args.elite_teacher_slots) > int(getattr(elite_teacher.config, "max_residual_slots", 1)):
            raise SystemExit(
                f"--elite-teacher-slots {int(args.elite_teacher_slots)} exceeds teacher max_residual_slots="
                f"{int(getattr(elite_teacher.config, 'max_residual_slots', 1))}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_metrics: Dict[str, float] = {}
    best_checkpoint = output_dir / f"{args.run_name}_best.pt"
    latest_checkpoint = output_dir / f"{args.run_name}_latest.pt"

    print(
        "[train_social_cvae_refiner] "
        f"variant={_variant_name(args)} cache={cache_path.as_posix()} train_items={len(train_indices)} "
        f"val_items={len(val_indices)} device={device} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if afc_bank is not None:
        print(
            "[train_social_cvae_refiner] "
            f"v59_afc_bank_size={afc_bank.bank_size} top_m={int(args.v59_afc_top_m)} "
            f"anchor_modes={int(args.v59_anchor_modes)}"
        )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_metric_rows: List[Dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _outputs = _loss_step(
                model,
                batch,
                args=args,
                semantic_prototypes=semantic_prototypes,
                elite_teacher=elite_teacher,
                afc_bank=afc_bank,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            train_metric_rows.append(metrics)
            if batch_index == 1 or batch_index % max(int(args.log_every), 1) == 0:
                if str(args.variant) == "v60a":
                    print(
                        "[train_social_cvae_refiner] "
                        f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                        f"loss={metrics['loss']:.6f} anchor_keep={metrics['loss_v60_anchor_keep']:.6f} "
                        f"afc_role={metrics['loss_v60_afc_role']:.6f} "
                        f"base_id={metrics['loss_v60_base_identity']:.6f} "
                        f"spread_floor={metrics['loss_v60_spread_floor']:.6f} "
                        f"risk={metrics['loss_v60_risk']:.6f} residual={metrics['loss_v60_residual']:.6f} "
                        f"role_centers={metrics['v60_role_center_count']:.1f} "
                        f"div_ratio={metrics['v60_diversity_mode_ratio']:.3f} "
                        f"end_ratio={metrics['v60_selected_endpoint_ratio']:.3f} "
                        f"traj_ratio={metrics['v60_selected_trajectory_ratio']:.3f}"
                    )
                elif str(args.variant) in {"v59a", "v59b", "v59c"}:
                    print(
                        "[train_social_cvae_refiner] "
                        f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                        f"loss={metrics['loss']:.6f} anchor={metrics['loss_v59_anchor_obs']:.6f} "
                        f"afc={metrics['loss_v59_afc']:.6f} base={metrics['loss_v59_base_preserve']:.6f} "
                        f"anchor_keep={metrics['loss_v59_anchor_keep']:.6f} "
                        f"spread_floor={metrics['loss_v59_spread_floor']:.6f} "
                        f"risk={metrics['loss_v59_risk']:.6f} "
                        f"afc_cov={metrics['loss_v59_afc_coverage']:.6f} "
                        f"afc_ent={metrics['loss_v59_afc_entropy']:.6f}"
                    )
                else:
                    print(
                        "[train_social_cvae_refiner] "
                        f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                        f"loss={metrics['loss']:.6f} recon={metrics['loss_recon']:.6f} "
                        f"set={metrics['loss_set_coverage']:.6f} kl={metrics['loss_kl']:.6f}"
                    )
        val_metrics = _eval_loader(model, val_loader, device=device, args=args)
        score = _selection_score(val_metrics, args)
        train_metrics = _mean_metrics(train_metric_rows)
        improved = best_score is None or score < best_score
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "meta": {
                        "variant": _variant_name(args),
                        "epoch": int(epoch),
                        "selection_metric": args.selection_metric,
                        "selection_score": float(score),
                        "cache_path": cache_path.as_posix(),
                        "elite_teacher_checkpoint": None if elite_teacher_path is None else elite_teacher_path.as_posix(),
                        "v59_afc_bank_size": None if afc_bank is None else int(afc_bank.bank_size),
                    },
                    "args": _jsonable(vars(args)),
                    "train_metrics": _jsonable(train_metrics),
                    "val_metrics": _jsonable(val_metrics),
                },
                best_checkpoint,
            )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "meta": {
                    "variant": _variant_name(args),
                    "epoch": int(epoch),
                    "cache_path": cache_path.as_posix(),
                    "v59_afc_bank_size": None if afc_bank is None else int(afc_bank.bank_size),
                },
                "args": _jsonable(vars(args)),
                "train_metrics": _jsonable(train_metrics),
                "val_metrics": _jsonable(val_metrics),
            },
            latest_checkpoint,
        )
        print(
            "[train_social_cvae_refiner] "
            f"epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.6f} "
            f"val_FDE_min={val_metrics.get('refined_FDE_min', float('nan')):.6f} "
            f"val_dFDE={val_metrics.get('dFDE_min', float('nan')):+.6f} "
            f"val_base_hurt={val_metrics.get('base_best_hurt_mean', float('nan')):.6f} "
            f"score={score:.6f} best={bool(improved)}"
        )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_social_cvae_refiner",
            "variant": _variant_name(args),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_checkpoint": best_checkpoint.as_posix(),
            "selection_metric": args.selection_metric,
            "best_selection_score": best_score,
            "elite_teacher_checkpoint": None if elite_teacher_path is None else elite_teacher_path.as_posix(),
            "v59_afc_bank_size": None if afc_bank is None else int(afc_bank.bank_size),
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
