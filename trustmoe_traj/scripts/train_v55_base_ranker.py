"""Train V55-D high-potential base ranker for adaptive V38 slot budgets."""

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

from trustmoe_traj.evaluation import summarize_accuracy_metrics
from trustmoe_traj.models import V55BaseRanker, V55BaseRankerConfig, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import _flatten_refined, _gather_candidates
from trustmoe_traj.scripts.diagnose_v55_adaptive_base_budget import _budget_indices
from trustmoe_traj.scripts.train_social_cvae_refiner import DEFAULT_OUTPUT_DIR as REFINER_OUTPUT_DIR, _prepare_refiner_tensors
from trustmoe_traj.scripts.train_student_integrated_finetune import (
    DEFAULT_CACHE_PATH,
    CacheDataset,
    _jsonable,
    _load_cache,
    _move_batch,
    _resolve_device,
    _select_indices,
    _set_seed,
)


DEFAULT_OUTPUT_DIR = REFINER_OUTPUT_DIR.parent / "v55_base_ranker_models"
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
BUDGETS: Sequence[str] = ("top5x4", "top4x4_next4slot0", "top3x4_next8slot0", "top10x2_slot01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V55-D high-potential base ranker.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", type=str, default="v55_base_ranker")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--allow-energy-fallback", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-mode-embedding", action="store_true")
    parser.add_argument("--use-energy-risk-map", action="store_true")
    parser.add_argument("--energy-risk-distance-scale", type=float, default=0.5)
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--target-top-k", type=int, default=5)
    parser.add_argument("--label-metric", type=str, default="fde", choices=["fde", "ade_fde"])

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-soft-rank", type=float, default=0.5)
    parser.add_argument("--lambda-pairwise-rank", type=float, default=0.1)
    parser.add_argument("--soft-rank-temperature", type=float, default=0.25)
    parser.add_argument("--pairwise-margin", type=float, default=0.02)
    parser.add_argument("--pairwise-max-weight", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--selection-metric", type=str, default="top5x4_fde_min", choices=["top5x4_fde_min", "top5x4_safety"])
    parser.add_argument("--selection-miss-weight", type=float, default=2.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def _variant_name(args: argparse.Namespace) -> str:
    return f"v55d_high_potential_base_ranker_top{int(args.target_top_k)}_slots{int(args.residual_slots)}"


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.target_top_k) <= 0:
        raise SystemExit("--target-top-k must be positive")
    if int(args.target_top_k) > 20:
        raise SystemExit("--target-top-k must not exceed max modes")
    if float(args.dropout) < 0.0 or float(args.dropout) >= 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    for name in (
        "lambda_bce",
        "lambda_soft_rank",
        "lambda_pairwise_rank",
        "soft_rank_temperature",
        "pairwise_margin",
        "pairwise_max_weight",
        "positive_weight",
        "selection_miss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.soft_rank_temperature) <= 0.0:
        raise SystemExit("--soft-rank-temperature must be positive")
    if float(args.pairwise_max_weight) <= 0.0:
        raise SystemExit("--pairwise-max-weight must be positive")


def _base_score(base: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported label metric: {metric!r}")


def _target_mask(base_score: torch.Tensor, *, top_k: int) -> torch.Tensor:
    keep = min(int(top_k), int(base_score.shape[1]))
    indices = torch.topk(base_score, k=keep, dim=1, largest=False).indices
    target = torch.zeros_like(base_score)
    target.scatter_(1, indices, 1.0)
    return target


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.to(device=values.device, dtype=torch.bool)
    if int(keep.sum().item()) <= 0:
        return values.new_tensor(0.0)
    return values[keep].mean()


def _bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, positive_weight: float) -> torch.Tensor:
    pos_weight = logits.new_tensor(max(float(positive_weight), 1e-6))
    losses = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    valid = mask[:, None, :].expand_as(losses).bool()
    return _masked_mean(losses, valid)


def _soft_rank_loss(logits: torch.Tensor, base_score: torch.Tensor, mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    target_prob = F.softmax(-base_score.detach() / max(float(temperature), 1e-6), dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    per_agent = -(target_prob * log_prob).sum(dim=1)
    return _masked_mean(per_agent, mask.bool())


def _pairwise_rank_loss(
    logits: torch.Tensor,
    base_score: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
    max_weight: float,
) -> torch.Tensor:
    score_i = base_score[:, :, None, :]
    score_j = base_score[:, None, :, :]
    gap = score_j - score_i
    better = gap > float(margin)
    logit_i = logits[:, :, None, :]
    logit_j = logits[:, None, :, :]
    pair_loss = F.softplus(-(logit_i - logit_j))
    weight = gap.clamp_min(0.0).clamp_max(float(max_weight))
    eye = torch.eye(int(logits.shape[1]), dtype=torch.bool, device=logits.device)
    valid = mask[:, None, None, :].expand(logits.shape[0], logits.shape[1], logits.shape[1], logits.shape[2]).bool()
    keep = better & valid & (~eye[None, :, :, None])
    if int(keep.sum().item()) <= 0:
        return logits.new_tensor(0.0)
    weighted = pair_loss * weight
    return weighted[keep].sum() / weight[keep].sum().clamp_min(1e-6)


def _rank_metrics(logits: torch.Tensor, base_score: torch.Tensor, mask: torch.Tensor, *, top_k: int) -> Dict[str, float]:
    keep = min(int(top_k), int(logits.shape[1]))
    pred_order = torch.argsort(logits.detach(), dim=1, descending=True)
    target_order = torch.argsort(base_score.detach(), dim=1)
    pred_top = pred_order[:, :keep, :]
    target_top = target_order[:, :keep, :]
    valid = mask.to(device=logits.device, dtype=torch.bool)
    if int(valid.sum().item()) <= 0:
        return {"top1_acc": 0.0, "topk_best_hit": 0.0, "topk_recall": 0.0, "target_topk": float(keep)}
    top1_acc = pred_order[:, 0, :] == target_order[:, 0, :]
    best_hit = (pred_top == target_order[:, :1, :]).any(dim=1)
    intersection = (pred_top[:, :, None, :] == target_top[:, None, :, :]).any(dim=1).sum(dim=1).to(dtype=torch.float32)
    recall = intersection / float(keep)
    return {
        "top1_acc": float(top1_acc[valid].float().mean().detach().cpu()),
        "topk_best_hit": float(best_hit[valid].float().mean().detach().cpu()),
        "topk_recall": float(recall[valid].mean().detach().cpu()),
        "target_topk": float(keep),
    }


def _mean_metrics(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    items = list(rows)
    if not items:
        return {}
    keys = list(items[0].keys())
    return {key: float(sum(float(row[key]) for row in items) / len(items)) for key in keys}


def _weighted_mean(rows: Sequence[Mapping[str, float]], weights: Sequence[int]) -> Dict[str, float]:
    if not rows:
        return {}
    total = max(int(sum(weights)), 1)
    result: Dict[str, float] = {}
    for key in rows[0].keys():
        result[key] = float(sum(float(row[key]) * int(weight) for row, weight in zip(rows, weights)) / total)
    return result


def _summarize_prediction(
    prediction: torch.Tensor,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    prefix: str,
    miss_threshold: float,
) -> Dict[str, float]:
    pred = summarize_accuracy_metrics(prediction, ground_truth, agent_mask=mask, miss_threshold=float(miss_threshold))
    base_summary = summarize_accuracy_metrics(base, ground_truth, agent_mask=mask, miss_threshold=float(miss_threshold))
    result: Dict[str, float] = {}
    for metric in METRICS:
        result[f"{prefix}_{metric}"] = float(pred[metric])
        result[f"{prefix}_d{metric}"] = float(pred[metric]) - float(base_summary[metric])
    result[f"{prefix}_num_valid_agents"] = float(pred["num_valid_agents"])
    return result


def _predict_budget(refined: torch.Tensor, base_order: torch.Tensor, *, budget: str) -> torch.Tensor:
    flat = _flatten_refined(refined)
    dummy_scores = flat.new_zeros(flat.shape[0], flat.shape[1], flat.shape[2])
    indices = _budget_indices(
        base_order,
        budget=budget,
        flat_scores=dummy_scores,
        num_slots=int(refined.shape[1]),
        num_base_modes=int(refined.shape[2]),
    )
    return _gather_candidates(flat, indices)


def _loss_step(
    model: V55BaseRanker,
    refiner: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        refiner_outputs = refiner.refine(
            batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            num_samples=int(args.residual_slots),
            z_mode="slots",
        )
        refined = refiner_outputs["refined"].detach()
        base_score = _base_score(batch["teacher_pred"], batch["ground_truth"], metric=str(args.label_metric)).detach()
        target = _target_mask(base_score, top_k=int(args.target_top_k))
    logits = model(
        batch["teacher_pred"],
        refined_trajectory=refined,
        past_traj_original_scale=batch["past_traj_original_scale"],
        temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
    )
    loss_bce = _bce_loss(logits, target, batch["agent_mask"].bool(), positive_weight=float(args.positive_weight))
    loss_soft = _soft_rank_loss(
        logits,
        base_score,
        batch["agent_mask"].bool(),
        temperature=float(args.soft_rank_temperature),
    )
    loss_pairwise = _pairwise_rank_loss(
        logits,
        base_score,
        batch["agent_mask"].bool(),
        margin=float(args.pairwise_margin),
        max_weight=float(args.pairwise_max_weight),
    )
    loss = (
        float(args.lambda_bce) * loss_bce
        + float(args.lambda_soft_rank) * loss_soft
        + float(args.lambda_pairwise_rank) * loss_pairwise
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_bce": float(loss_bce.detach().cpu()),
        "loss_soft_rank": float(loss_soft.detach().cpu()),
        "loss_pairwise_rank": float(loss_pairwise.detach().cpu()),
        **_rank_metrics(logits, base_score, batch["agent_mask"].bool(), top_k=int(args.target_top_k)),
    }
    return loss, metrics


@torch.no_grad()
def _eval_loader(
    model: V55BaseRanker,
    refiner: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    refiner.eval()
    rows: List[Dict[str, float]] = []
    weights: List[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        refiner_outputs = refiner.refine(
            batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            num_samples=int(args.residual_slots),
            z_mode="slots",
        )
        refined = refiner_outputs["refined"]
        base_score = _base_score(batch["teacher_pred"], batch["ground_truth"], metric=str(args.label_metric))
        logits = model(
            batch["teacher_pred"],
            refined_trajectory=refined,
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
        )
        order = torch.argsort(logits, dim=1, descending=True)
        summary: Dict[str, float] = {
            **_rank_metrics(logits, base_score, batch["agent_mask"].bool(), top_k=int(args.target_top_k)),
        }
        for budget in BUDGETS:
            prediction = _predict_budget(refined, order, budget=budget)
            summary.update(
                _summarize_prediction(
                    prediction,
                    batch["teacher_pred"],
                    batch["ground_truth"],
                    batch["agent_mask"].bool(),
                    prefix=budget,
                    miss_threshold=float(args.miss_threshold),
                )
            )
        rows.append(summary)
        weights.append(int(batch["agent_mask"].bool().sum().item()))
    return _weighted_mean(rows, weights)


def _selection_score(metrics: Mapping[str, float], args: argparse.Namespace) -> float:
    if str(args.selection_metric) == "top5x4_fde_min":
        return float(metrics["top5x4_FDE_min"])
    return float(metrics["top5x4_FDE_min"]) + float(args.selection_miss_weight) * max(
        0.0,
        float(metrics.get("top5x4_dMissRate", 0.0)),
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
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
    if not bool(getattr(refiner.config, "use_set_generator", False)):
        raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")

    teacher_shape = tensors["teacher_pred"].shape
    past_shape = tensors["past_traj_original_scale"].shape
    energy_shape = tensors["teacher_temporal_interaction_energy_features"].shape
    config = V55BaseRankerConfig(
        future_frames=int(teacher_shape[-2]),
        coord_dim=int(teacher_shape[-1]),
        past_frames=int(past_shape[-2]),
        past_feature_dim=int(past_shape[-1]),
        temporal_energy_dim=int(energy_shape[-1]),
        residual_slots=int(args.residual_slots),
        hidden_dim=int(args.hidden_dim),
        max_modes=int(teacher_shape[1]),
        use_mode_embedding=not bool(args.no_mode_embedding),
        use_energy_risk_map=bool(args.use_energy_risk_map),
        energy_risk_distance_scale=float(args.energy_risk_distance_scale),
        dropout=float(args.dropout),
    )
    model = V55BaseRanker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_metrics: Dict[str, float] = {}
    best_checkpoint = output_dir / f"{args.run_name}_best.pt"
    latest_checkpoint = output_dir / f"{args.run_name}_latest.pt"

    print(
        "[train_v55_base_ranker] "
        f"variant={_variant_name(args)} cache={cache_path.as_posix()} refiner={refiner_path.as_posix()} "
        f"train_items={len(train_indices)} val_items={len(val_indices)} device={device} "
        f"target_top_k={args.target_top_k} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_rows: List[Dict[str, float]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss_step(model, refiner, batch, args=args)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            train_rows.append(metrics)
            if batch_index == 1 or batch_index % max(int(args.log_every), 1) == 0:
                print(
                    "[train_v55_base_ranker] "
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={metrics['loss']:.6f} bce={metrics['loss_bce']:.6f} "
                    f"top5_hit={metrics['topk_best_hit']:.4f} recall={metrics['topk_recall']:.4f}"
                )
        train_metrics = _mean_metrics(train_rows)
        val_metrics = _eval_loader(model, refiner, val_loader, device=device, args=args)
        score = _selection_score(val_metrics, args)
        improved = best_score is None or score < best_score
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "meta": {
                "variant": _variant_name(args),
                "epoch": int(epoch),
                "selection_metric": args.selection_metric,
                "selection_score": float(score),
                "cache_path": cache_path.as_posix(),
                "refiner_checkpoint": refiner_path.as_posix(),
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
            "[train_v55_base_ranker] "
            f"epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.6f} "
            f"val_top5x4_FDE_min={val_metrics.get('top5x4_FDE_min', float('nan')):.6f} "
            f"val_top5x4_dFDE={val_metrics.get('top5x4_dFDE_min', float('nan')):+.6f} "
            f"val_hit={val_metrics.get('topk_best_hit', float('nan')):.4f} "
            f"score={score:.6f} best={bool(improved)}"
        )

    summary = {
        "meta": {
            "script": "trustmoe_traj.scripts.train_v55_base_ranker",
            "variant": _variant_name(args),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_checkpoint": best_checkpoint.as_posix(),
            "selection_metric": args.selection_metric,
            "best_selection_score": best_score,
            "refiner_checkpoint": refiner_path.as_posix(),
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
