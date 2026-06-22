"""V19-A: train a readout-token hidden adapter for the fast student.

This variant freezes the MoFlow IMLE student and trains only a lightweight
adapter inserted between ``motion_decoder`` and ``reg_head``.  It keeps the
same safety-oriented losses used by V18-B2, but avoids drifting the original
decoder/head parameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trustmoe_traj.data.transforms import DEFAULT_PAST_SOCIAL_RISK_DIM
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    build_student_hidden_adapter_for_model,
)
from trustmoe_traj.scripts.train_student_integrated_finetune import (
    DEFAULT_CACHE_PATH,
    DEFAULT_OUTPUT_DIR,
    CacheDataset,
    _diversity_loss,
    _energy_gt_loss,
    _generate_student,
    _good_nohurt_loss,
    _gt_min_loss,
    _jsonable,
    _load_cache,
    _masked_mean,
    _move_batch,
    _prepare_tensors,
    _resolve_device,
    _select_indices,
    _selection_score,
    _set_chamfer_loss,
    _set_seed,
    _student_best_nohurt_loss,
    _summarize_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a hidden-token adapter for the MoFlow fast student.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="student_hidden_adapter_v19a")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--fast-cfg-path", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate", dest="rotate", action="store_true")
    rotate_group.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.set_defaults(rotate=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)

    parser.add_argument(
        "--adapter-site",
        type=str,
        default="readout",
        choices=["readout", "query"],
        help="Use a readout-token adapter after motion_decoder, or a query-token adapter before motion_decoder.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mode-context-num-layers", type=int, default=1)
    parser.add_argument("--mode-context-num-heads", type=int, default=4)
    parser.add_argument("--mode-context-dropout", type=float, default=0.0)
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-token-delta", type=float, default=0.5)
    parser.add_argument("--gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument(
        "--use-past-social-risk",
        action="store_true",
        help="Condition the hidden adapter on observed-past social-risk features for V19-B.",
    )
    parser.add_argument("--social-risk-dim", type=int, default=DEFAULT_PAST_SOCIAL_RISK_DIM)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-set", type=float, default=1.0)
    parser.add_argument("--lambda-gt-min", type=float, default=0.3)
    parser.add_argument("--lambda-energy-gt", type=float, default=0.2)
    parser.add_argument("--energy-gt-top-k", type=int, default=2)
    parser.add_argument("--energy-risk-floor", type=float, default=0.05)
    parser.add_argument("--lambda-good-nohurt", type=float, default=1.0)
    parser.add_argument("--good-nohurt-frac", type=float, default=0.25)
    parser.add_argument("--good-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-student-best-nohurt", type=float, default=1.0)
    parser.add_argument("--student-best-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-diversity-preserve", type=float, default=0.2)
    parser.add_argument("--diversity-preserve-target-ratio", type=float, default=0.98)
    parser.add_argument("--lambda-keep-set", type=float, default=0.1)
    parser.add_argument("--lambda-hidden-delta", type=float, default=0.001)
    parser.add_argument("--lambda-hidden-gate", type=float, default=0.001)
    parser.add_argument(
        "--lambda-gate-target",
        type=float,
        default=0.0,
        help=(
            "V20-B: explicitly calibrate the adapter gate with a correction-need target. "
            "Default 0 keeps the V20-A objective unchanged."
        ),
    )
    parser.add_argument(
        "--gate-target-kind",
        type=str,
        default="student_error_teacher_social",
        choices=["student_error", "student_error_teacher", "student_error_social", "student_error_teacher_social"],
        help=(
            "How to build the gate target. Student error is always the base correction-need signal; "
            "teacher/social terms are reliability evidence, not a teacher ceiling."
        ),
    )
    parser.add_argument("--gate-target-loss", type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument("--gate-target-student-bad-quantile", type=float, default=0.80)
    parser.add_argument("--gate-target-student-bad-threshold", type=float, default=None)
    parser.add_argument("--gate-target-temperature", type=float, default=0.20)
    parser.add_argument("--gate-target-mode-threshold-ratio", type=float, default=0.75)
    parser.add_argument("--gate-target-teacher-margin", type=float, default=0.02)
    parser.add_argument("--gate-target-evidence-floor", type=float, default=0.25)
    parser.add_argument("--gate-target-min", type=float, default=0.02)
    parser.add_argument("--gate-target-max", type=float, default=0.85)
    parser.add_argument("--selection-metric", type=str, default="safety", choices=["fde_min", "safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--selection-nohurt-weight", type=float, default=1.0)
    parser.add_argument("--selection-student-best-weight", type=float, default=1.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _maybe_add_social_risk_tensors(
    tensors: Dict[str, torch.Tensor],
    payload: Mapping[str, Any],
    *,
    use_past_social_risk: bool,
) -> None:
    if not bool(use_past_social_risk):
        return
    raw_tensors = payload.get("tensors", {})
    if not isinstance(raw_tensors, Mapping) or "past_social_risk_features" not in raw_tensors:
        raise ValueError(
            "V19-B training requires cache tensor `past_social_risk_features`. "
            "Re-export the teacher/student cache with the current data transform code."
        )
    tensors["past_social_risk_features"] = raw_tensors["past_social_risk_features"].to(torch.float32)


def _variant_name(args: argparse.Namespace) -> str:
    if float(getattr(args, "lambda_gate_target", 0.0)) > 0.0:
        if args.adapter_site == "query":
            return (
                "v20b_query_past_social_risk_correction_need_gate_hidden_adapter"
                if bool(args.use_past_social_risk)
                else "v20b_query_correction_need_gate_hidden_adapter"
            )
        return (
            "v20b_readout_past_social_risk_correction_need_gate_hidden_adapter"
            if bool(args.use_past_social_risk)
            else "v20b_readout_correction_need_gate_hidden_adapter"
        )
    if args.adapter_site == "query":
        return (
            "v20a_query_past_social_risk_hidden_adapter"
            if bool(args.use_past_social_risk)
            else "v20a_query_hidden_adapter"
        )
    return (
        "v19b_past_social_risk_readout_hidden_adapter"
        if bool(args.use_past_social_risk)
        else "v19a_readout_hidden_adapter"
    )


def _attached_adapter(predictor: MoFlowFastPredictor, adapter_site: str) -> torch.nn.Module:
    attr_name = "student_query_adapter" if adapter_site == "query" else "student_hidden_adapter"
    adapter = getattr(predictor.engine.model, attr_name, None)
    if adapter is None:
        raise RuntimeError(f"{attr_name} is not attached")
    return adapter


def _set_model_modes(predictor: MoFlowFastPredictor, *, adapter_train: bool) -> None:
    predictor.engine.eval()
    predictor.engine.model.eval()
    adapter = _attached_adapter(predictor, getattr(predictor.engine.model, "student_adapter_site", "readout"))
    adapter.train(bool(adapter_train))


def _fde_by_mode(prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)[..., -1]


def _valid_fde_min(prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fde = _fde_by_mode(prediction, ground_truth)
    valid = mask.bool()
    inf = torch.tensor(float("inf"), device=fde.device, dtype=fde.dtype)
    return fde.masked_fill(~valid[:, None, :], inf).min(dim=1).values


def _resolve_gate_target_settings(
    tensors: Mapping[str, torch.Tensor],
    train_indices: Sequence[int],
    args: argparse.Namespace,
) -> None:
    if float(args.lambda_gate_target) <= 0.0:
        args.gate_target_student_bad_threshold_resolved = None
        args.gate_target_mode_bad_threshold_resolved = None
        return

    if args.gate_target_student_bad_threshold is not None:
        threshold = float(args.gate_target_student_bad_threshold)
    else:
        if not train_indices:
            raise ValueError("Cannot resolve gate target threshold from an empty train split")
        index = torch.as_tensor(list(train_indices), dtype=torch.long)
        student = tensors["student_pred"][index]
        gt = tensors["ground_truth"][index]
        mask = tensors["agent_mask"][index].bool()
        fde_min = _valid_fde_min(student, gt, mask)
        valid_values = fde_min[mask]
        if int(valid_values.numel()) <= 0:
            raise ValueError("Cannot resolve gate target threshold because train split has no valid agents")
        quantile = min(max(float(args.gate_target_student_bad_quantile), 0.0), 1.0)
        threshold = float(torch.quantile(valid_values.detach().cpu(), quantile).item())

    mode_threshold = float(threshold) * max(float(args.gate_target_mode_threshold_ratio), 1e-6)
    args.gate_target_student_bad_threshold_resolved = float(threshold)
    args.gate_target_mode_bad_threshold_resolved = float(mode_threshold)


def _gate_by_mode(gate: torch.Tensor) -> torch.Tensor:
    gate = gate.squeeze(-1)
    if gate.ndim == 4:
        return gate.mean(dim=1)
    if gate.ndim == 3:
        return gate
    raise ValueError(f"adapter gate must have shape [B,K,A,1] or [B,M,K,A,1], got {tuple(gate.shape)}")


def _past_social_gate_evidence(
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    social_risk = batch.get("past_social_risk_features")
    if social_risk is None:
        return None
    risk = social_risk.to(device=device, dtype=dtype)
    if risk.ndim != 3 or risk.shape[-1] <= 0:
        return None

    parts: List[torch.Tensor] = []
    for index in (1, 2, 3, 4, 8, 9):
        if int(risk.shape[-1]) > index:
            parts.append(risk[..., index].clamp(0.0, 1.0))
    for index in (5, 6, 7):
        if int(risk.shape[-1]) > index:
            positive = risk[..., index].clamp_min(0.0)
            parts.append(positive / (1.0 + positive))
    if not parts:
        return None
    return torch.stack(parts, dim=-1).amax(dim=-1).clamp(0.0, 1.0)


def _correction_need_gate_target(
    *,
    student: torch.Tensor,
    teacher: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    student_fde = _fde_by_mode(student, ground_truth)
    teacher_fde_min = _valid_fde_min(teacher, ground_truth, mask)
    student_fde_min = _valid_fde_min(student, ground_truth, mask)
    valid_agent = mask.bool()
    student_fde = student_fde.masked_fill(~valid_agent[:, None, :], 0.0)
    teacher_fde_min = teacher_fde_min.masked_fill(~valid_agent, 0.0)
    student_fde_min = student_fde_min.masked_fill(~valid_agent, 0.0)

    threshold = getattr(args, "gate_target_student_bad_threshold_resolved", None)
    if threshold is None:
        threshold = args.gate_target_student_bad_threshold
    if threshold is None:
        raise ValueError("gate target threshold is not resolved; call _resolve_gate_target_settings first")

    temperature = max(float(args.gate_target_temperature), 1e-6)
    threshold_tensor = student_fde.new_tensor(float(threshold))
    mode_threshold = getattr(args, "gate_target_mode_bad_threshold_resolved", None)
    if mode_threshold is None:
        mode_threshold = float(threshold) * float(args.gate_target_mode_threshold_ratio)
    mode_threshold_tensor = student_fde.new_tensor(float(mode_threshold))

    sample_need = torch.sigmoid((student_fde_min - threshold_tensor) / temperature)
    mode_need = torch.sigmoid((student_fde - mode_threshold_tensor) / temperature)
    target = sample_need[:, None, :] * mode_need

    evidence_parts: List[torch.Tensor] = []
    if "teacher" in str(args.gate_target_kind):
        margin = float(args.gate_target_teacher_margin)
        teacher_good = torch.sigmoid((threshold_tensor - teacher_fde_min) / temperature)
        teacher_better = torch.sigmoid((student_fde_min - teacher_fde_min - margin) / temperature)
        evidence_parts.append(torch.maximum(teacher_good, teacher_better))
    if "social" in str(args.gate_target_kind):
        social_evidence = _past_social_gate_evidence(
            batch,
            device=student_fde.device,
            dtype=student_fde.dtype,
        )
        if social_evidence is not None:
            evidence_parts.append(social_evidence)

    if evidence_parts:
        evidence = torch.stack(evidence_parts, dim=0).amax(dim=0)
        evidence_floor = min(max(float(args.gate_target_evidence_floor), 0.0), 1.0)
        target = target * (evidence_floor + (1.0 - evidence_floor) * evidence[:, None, :])

    target = target.clamp(0.0, 1.0)
    target_min = min(max(float(args.gate_target_min), 0.0), 1.0)
    target_max = min(max(float(args.gate_target_max), target_min), 1.0)
    return target_min + (target_max - target_min) * target


def _gate_target_loss(
    gate: torch.Tensor,
    student: torch.Tensor,
    teacher: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    gate_mode = _gate_by_mode(gate)
    if float(args.lambda_gate_target) <= 0.0:
        zero = gate_mode.new_tensor(0.0)
        return zero, {
            "loss_gate_target": 0.0,
            "hidden_gate_target_mean": 0.0,
            "hidden_gate_target_high_rate": 0.0,
            "hidden_gate_target_mse": 0.0,
        }

    target = _correction_need_gate_target(
        student=student,
        teacher=teacher,
        ground_truth=ground_truth,
        mask=mask,
        batch=batch,
        args=args,
    ).detach()
    valid = mask.bool()[:, None, :].expand_as(gate_mode)
    if int(valid.sum().item()) <= 0:
        zero = gate_mode.new_tensor(0.0)
        return zero, {
            "loss_gate_target": 0.0,
            "hidden_gate_target_mean": 0.0,
            "hidden_gate_target_high_rate": 0.0,
            "hidden_gate_target_mse": 0.0,
        }

    if args.gate_target_loss == "bce":
        loss_map = F.binary_cross_entropy(gate_mode.clamp(1e-5, 1.0 - 1e-5), target, reduction="none")
    else:
        loss_map = F.mse_loss(gate_mode, target, reduction="none")
    loss = _masked_mean(loss_map, valid)
    mse = F.mse_loss(gate_mode, target, reduction="none")
    metrics = {
        "loss_gate_target": float(loss.detach().cpu()),
        "hidden_gate_target_mean": float(target[valid].mean().detach().cpu()),
        "hidden_gate_target_high_rate": float((target[valid] > 0.5).to(torch.float32).mean().detach().cpu()),
        "hidden_gate_target_mse": float(mse[valid].mean().detach().cpu()),
    }
    return loss, metrics


def _hidden_loss_step(
    predictor: MoFlowFastPredictor,
    batch: Mapping[str, torch.Tensor],
    *,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    _generated_norm, generated = _generate_student(predictor, batch, normalization_stats=normalization_stats)
    student = batch["student_pred"]
    teacher = batch["teacher_pred"]
    gt = batch["ground_truth"]
    mask = batch["agent_mask"].bool()

    loss_teacher = _set_chamfer_loss(generated, teacher, mask, fde_weight=args.fde_weight)
    loss_gt = _gt_min_loss(generated, gt, mask, fde_weight=args.fde_weight)
    loss_energy = _energy_gt_loss(
        generated,
        student,
        gt,
        batch.get("temporal_interaction_energy_features"),
        mask,
        fde_weight=args.fde_weight,
        top_k=args.energy_gt_top_k,
        risk_floor=args.energy_risk_floor,
    )
    loss_nohurt = _good_nohurt_loss(
        generated,
        student,
        gt,
        mask,
        fde_weight=args.fde_weight,
        good_frac=args.good_nohurt_frac,
        margin=args.good_nohurt_margin,
    )
    loss_student_best_nohurt = _student_best_nohurt_loss(
        generated,
        student,
        gt,
        mask,
        margin=args.student_best_nohurt_margin,
    )
    loss_div = _diversity_loss(generated, student, mask, target_ratio=args.diversity_preserve_target_ratio)
    loss_keep = _set_chamfer_loss(generated, student, mask, fde_weight=args.fde_weight)

    adapter = _attached_adapter(predictor, getattr(predictor.engine.model, "student_adapter_site", "readout"))
    if adapter is None or adapter.last_delta is None or adapter.last_gate is None:
        raise RuntimeError("student adapter did not expose hidden diagnostics")
    loss_hidden_delta = adapter.last_delta.pow(2).mean()
    loss_hidden_gate = adapter.last_gate.mean()
    loss_gate_target, gate_target_metrics = _gate_target_loss(
        adapter.last_gate,
        student,
        teacher,
        gt,
        mask,
        batch,
        args,
    )

    loss = (
        float(args.lambda_teacher_set) * loss_teacher
        + float(args.lambda_gt_min) * loss_gt
        + float(args.lambda_energy_gt) * loss_energy
        + float(args.lambda_good_nohurt) * loss_nohurt
        + float(args.lambda_student_best_nohurt) * loss_student_best_nohurt
        + float(args.lambda_diversity_preserve) * loss_div
        + float(args.lambda_keep_set) * loss_keep
        + float(args.lambda_hidden_delta) * loss_hidden_delta
        + float(args.lambda_hidden_gate) * loss_hidden_gate
        + float(args.lambda_gate_target) * loss_gate_target
    )
    components = {
        "loss": float(loss.detach().cpu()),
        "loss_teacher_set": float(loss_teacher.detach().cpu()),
        "loss_gt_min": float(loss_gt.detach().cpu()),
        "loss_energy_gt": float(loss_energy.detach().cpu()),
        "loss_good_nohurt": float(loss_nohurt.detach().cpu()),
        "loss_student_best_nohurt": float(loss_student_best_nohurt.detach().cpu()),
        "loss_diversity": float(loss_div.detach().cpu()),
        "loss_keep_set": float(loss_keep.detach().cpu()),
        "loss_hidden_delta": float(loss_hidden_delta.detach().cpu()),
        "loss_hidden_gate": float(loss_hidden_gate.detach().cpu()),
        "hidden_gate_mean": float(adapter.last_gate.detach().mean().cpu()),
        "hidden_delta_l2_mean": float(adapter.last_delta.detach().pow(2).mean().sqrt().cpu()),
    }
    components.update(gate_target_metrics)
    return loss, components, {"hidden_adapter_pred": generated[:, 0].detach()}


@torch.no_grad()
def _evaluate(
    predictor: MoFlowFastPredictor,
    loader: DataLoader,
    *,
    device: str,
    normalization_stats: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, float]:
    _set_model_modes(predictor, adapter_train=False)
    sums: Dict[str, float] = {}
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        _loss, components, output = _hidden_loss_step(
            predictor,
            batch,
            normalization_stats=normalization_stats,
            args=args,
        )
        valid = int(batch["agent_mask"].bool().sum().item())
        metrics = _summarize_prediction(
            output["hidden_adapter_pred"].unsqueeze(1),
            batch["student_pred"],
            batch["ground_truth"],
            batch["agent_mask"].bool(),
            miss_threshold=args.miss_threshold,
        )
        for key, value in {**components, **metrics}.items():
            sums[key] = sums.get(key, 0.0) + float(value) * valid
        total += valid
    if total <= 0:
        raise ValueError("Validation loader has no valid agents")
    return {key: value / total for key, value in sums.items()}


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_cache(cache_path)
    tensors = _prepare_tensors(payload)
    _maybe_add_social_risk_tensors(
        tensors,
        payload,
        use_past_social_risk=bool(args.use_past_social_risk),
    )
    train_indices, val_indices = _select_indices(
        int(tensors["ground_truth"].shape[0]),
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )
    _resolve_gate_target_settings(tensors, train_indices, args)

    predictor = MoFlowFastPredictor(
        MoFlowPredictorConfig(
            subset=args.subset,
            sample_mode=args.sample_mode,
            agents=1,
            data_norm=args.data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            device=device,
            cfg_path=args.fast_cfg_path,
            checkpoint_path=args.fast_checkpoint,
            num_to_gen=int(args.num_to_gen),
        )
    )
    predictor._set_normalization_stats(payload["normalization_stats"])

    past_shape = list(tensors["past_traj_original_scale"].shape)
    adapter = build_student_hidden_adapter_for_model(
        predictor.engine.model,
        past_frames=int(past_shape[2]),
        past_feature_dim=int(past_shape[3]),
        adapter_site=args.adapter_site,
        hidden_dim=int(args.hidden_dim),
        max_modes=args.max_modes,
        use_noise=not bool(args.no_noise),
        use_past_social_risk=bool(args.use_past_social_risk),
        social_risk_dim=int(args.social_risk_dim),
        num_mode_context_layers=int(args.mode_context_num_layers),
        num_mode_context_heads=int(args.mode_context_num_heads),
        mode_context_dropout=float(args.mode_context_dropout),
        residual_scale=float(args.residual_scale),
        max_token_delta=args.max_token_delta,
        gate_init_bias=float(args.gate_init_bias),
    ).to(device)
    if args.adapter_site == "query":
        predictor.attach_student_query_adapter(adapter)
    else:
        predictor.attach_student_hidden_adapter(adapter)
    predictor.engine.model.student_adapter_site = args.adapter_site
    for param in predictor.engine.model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)
    adapter_prefix = "student_query_adapter" if args.adapter_site == "query" else "student_hidden_adapter"
    trainable_names = [f"{adapter_prefix}.{name}" for name, param in adapter.named_parameters() if param.requires_grad]

    train_loader = DataLoader(
        CacheDataset(tensors, train_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    val_loader = DataLoader(
        CacheDataset(tensors, val_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / f"{args.run_name}_best.pt"
    last_path = output_dir / f"{args.run_name}_last.pt"

    print(
        "[train_student_hidden_adapter] "
        f"cache={cache_path.as_posix()} train_items={len(train_indices)} val_items={len(val_indices)} "
        f"device={device} trainable_params={len(trainable_names)} "
        f"gate_target_lambda={float(args.lambda_gate_target):.4f} "
        f"gate_target_bad_fde={getattr(args, 'gate_target_student_bad_threshold_resolved', None)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        _set_model_modes(predictor, adapter_train=True)
        train_sums: Dict[str, float] = {}
        train_total = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, _output = _hidden_loss_step(
                predictor,
                batch,
                normalization_stats=payload["normalization_stats"],
                args=args,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            optimizer.step()

            valid = int(batch["agent_mask"].bool().sum().item())
            for key, value in components.items():
                train_sums[key] = train_sums.get(key, 0.0) + float(value) * valid
            train_total += valid

        train_metrics = {key: value / max(train_total, 1) for key, value in train_sums.items()}
        val_metrics = _evaluate(
            predictor,
            val_loader,
            device=device,
            normalization_stats=payload["normalization_stats"],
            args=args,
        )
        selection_score = _selection_score(val_metrics, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": selection_score}
        history.append(row)
        if float(selection_score) < best_metric:
            best_metric = float(selection_score)
            best_epoch = epoch
            torch.save(
                {
                    "model_state": adapter.state_dict(),
                    "config": asdict(adapter.config),
                    "meta": {
                        "script": "trustmoe_traj.scripts.train_student_hidden_adapter",
                        "variant": _variant_name(args),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "base_fast_checkpoint": str(Path(args.fast_checkpoint).expanduser().resolve()),
                        "base_fast_cfg_path": str(Path(args.fast_cfg_path).expanduser().resolve()),
                        "cache_path": cache_path.as_posix(),
                        "run_name": args.run_name,
                        "seed": int(args.seed),
                        "best_epoch": int(best_epoch),
                        "selection_metric": args.selection_metric,
                        "selection_score": float(selection_score),
                    },
                    "args": _jsonable(vars(args)),
                    "normalization_stats": _jsonable(payload["normalization_stats"]),
                    "best_val_metrics": _jsonable(val_metrics),
                    "best_selection_score": float(selection_score),
                    "trainable_names": trainable_names,
                },
                best_path,
            )

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(int(args.log_every), 1) == 0:
            print(
                "[train_student_hidden_adapter] "
                f"epoch={epoch:03d} train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_FDE_min={val_metrics['finetuned_FDE_min']:.6f} "
                f"val_dFDE={val_metrics['dFDE_min']:+.6f} "
                f"val_dMiss={val_metrics.get('dMissRate', 0.0):+.6f} "
                f"student_best_hurt={val_metrics.get('student_best_hurt_mean', 0.0):.6f} "
                f"gate={val_metrics.get('hidden_gate_mean', 0.0):.4f} "
                f"gate_target={val_metrics.get('hidden_gate_target_mean', 0.0):.4f} "
                f"delta_l2={val_metrics.get('hidden_delta_l2_mean', 0.0):.4f} "
                f"select={selection_score:.6f}"
            )

    torch.save(
        {
            "model_state": adapter.state_dict(),
            "config": asdict(adapter.config),
            "meta": {
                "script": "trustmoe_traj.scripts.train_student_hidden_adapter",
                "variant": _variant_name(args),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_fast_checkpoint": str(Path(args.fast_checkpoint).expanduser().resolve()),
                "base_fast_cfg_path": str(Path(args.fast_cfg_path).expanduser().resolve()),
                "cache_path": cache_path.as_posix(),
                "run_name": args.run_name,
                "seed": int(args.seed),
                "best_epoch": int(best_epoch),
                "best_checkpoint": best_path.as_posix(),
                "selection_metric": args.selection_metric,
                "best_selection_score": float(best_metric),
            },
            "args": _jsonable(vars(args)),
            "normalization_stats": _jsonable(payload["normalization_stats"]),
            "history": _jsonable(history),
            "trainable_names": trainable_names,
        },
        last_path,
    )
    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_student_hidden_adapter",
            "variant": _variant_name(args),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cache_path": cache_path.as_posix(),
            "run_name": args.run_name,
            "seed": int(args.seed),
            "best_epoch": int(best_epoch),
            "best_checkpoint": best_path.as_posix(),
            "last_checkpoint": last_path.as_posix(),
            "selection_metric": args.selection_metric,
            "best_selection_score": float(best_metric),
        },
        "args": _jsonable(vars(args)),
        "normalization_stats": _jsonable(payload["normalization_stats"]),
        "train_items": len(train_indices),
        "val_items": len(val_indices),
        "best_val_metrics": _jsonable(history[best_epoch - 1]["val"] if best_epoch else {}),
        "history": _jsonable(history),
        "trainable_names": trainable_names,
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")


if __name__ == "__main__":
    main()
