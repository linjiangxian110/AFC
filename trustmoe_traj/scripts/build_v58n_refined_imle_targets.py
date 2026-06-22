"""Build V58N residual-enhanced IMLE teacher targets.

The output is MoFlow IMLE-compatible train pickle data.  It keeps the final
teacher budget at K=20 by replacing each slow/base mode with at most one
Pareto-safe residual slot instead of distilling the full K*slots pool.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MoFlow" / "data" / "eth_ucy" / "imle_v58n_a"
EPS = 1e-8
ACCURACY_METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V58N refined K=20 IMLE teacher targets.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="train", choices=["train"])
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
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--candidate-slots", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="slot0_preserving_pareto",
        choices=["slot0_preserving_pareto", "slot0_preserving_best", "slot0_heavy_pareto"],
        help=(
            "Pareto mode replaces only safe/improving residuals; best mode is an oracle upper-bound target; "
            "slot0_heavy_pareto preserves the best slow modes and caps residual replacements."
        ),
    )
    parser.add_argument("--rank-label-metric", type=str, default="ade_fde", choices=["fde", "ade_fde"])
    parser.add_argument("--improve-margin", type=float, default=0.0)
    parser.add_argument("--hurt-margin", type=float, default=0.0)
    parser.add_argument("--accept-flat-tolerance", type=float, default=0.005)
    parser.add_argument("--accept-strong-improve-margin", type=float, default=0.03)
    parser.add_argument(
        "--keep-slow-top-k",
        type=int,
        default=0,
        help="For slot0_heavy_pareto, force the best N slow modes per agent to remain slot0.",
    )
    parser.add_argument(
        "--max-replacement-ratio",
        type=float,
        default=1.0,
        help="For slot0_heavy_pareto, cap residual replacements per agent to floor(ratio * keep_k).",
    )
    parser.add_argument(
        "--max-replacements-per-agent",
        type=int,
        default=None,
        help="Optional stricter absolute cap for residual replacements per agent.",
    )

    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", type=str, default="v58n_a_refined")
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--summary-txt", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item]
    if not values:
        raise SystemExit("Expected at least one candidate slot")
    return values


def _normalize_min_max(tensor: torch.Tensor, stats: Mapping[str, float]) -> torch.Tensor:
    min_val = float(stats["fut_traj_min"])
    max_val = float(stats["fut_traj_max"])
    if abs(max_val - min_val) <= EPS:
        raise ValueError("Invalid fut_traj min/max normalization stats")
    return (2.0 * (tensor - min_val) / (max_val - min_val) - 1.0).to(torch.float32)


def _candidate_ade_fde(candidates: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _score_from_ade_fde(ade: torch.Tensor, fde: torch.Tensor, *, metric: str) -> torch.Tensor:
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return ade + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _pareto_good(
    dade: torch.Tensor,
    dfde: torch.Tensor,
    *,
    improve_margin: float,
    hurt_margin: float,
    flat_tolerance: float,
    strong_improve_margin: float,
) -> torch.Tensor:
    safe = (dade <= float(hurt_margin) + EPS) & (dfde <= float(hurt_margin) + EPS)
    both_improve = (dade < -float(improve_margin) - EPS) & (dfde < -float(improve_margin) - EPS)
    ade_flat_fde_strong = (dade <= float(flat_tolerance) + EPS) & (
        dfde < -float(strong_improve_margin) - EPS
    )
    fde_flat_ade_strong = (dfde <= float(flat_tolerance) + EPS) & (
        dade < -float(strong_improve_margin) - EPS
    )
    return safe & (both_improve | ade_flat_fde_strong | fde_flat_ade_strong)


def _select_slots(
    refined: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    candidate_slots: Sequence[int],
    target_mode: str,
    rank_label_metric: str,
    improve_margin: float,
    hurt_margin: float,
    flat_tolerance: float,
    strong_improve_margin: float,
    keep_slow_top_k: int,
    max_replacement_ratio: float,
    max_replacements_per_agent: Optional[int],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    slot_ids = torch.tensor(list(candidate_slots), device=refined.device, dtype=torch.long)
    if 0 not in [int(item) for item in candidate_slots]:
        raise ValueError("candidate_slots must include slot0")
    slot_candidates = refined.index_select(dim=1, index=slot_ids)
    local_slot0 = int((slot_ids == 0).nonzero(as_tuple=False).reshape(-1)[0].item())

    ade, fde = _candidate_ade_fde(slot_candidates, ground_truth)
    score = _score_from_ade_fde(ade, fde, metric=rank_label_metric)
    slot0_ade = ade[:, local_slot0 : local_slot0 + 1]
    slot0_fde = fde[:, local_slot0 : local_slot0 + 1]
    dade = ade - slot0_ade
    dfde = fde - slot0_fde

    if target_mode in {"slot0_preserving_pareto", "slot0_heavy_pareto"}:
        good = _pareto_good(
            dade,
            dfde,
            improve_margin=improve_margin,
            hurt_margin=hurt_margin,
            flat_tolerance=flat_tolerance,
            strong_improve_margin=strong_improve_margin,
        )
        good[:, local_slot0] = True
        masked_score = score.masked_fill(~good, float("inf"))
    elif target_mode == "slot0_preserving_best":
        masked_score = score.clone()
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")

    selected_local = masked_score.argmin(dim=1).to(dtype=torch.long)
    protected_top = torch.zeros_like(selected_local, dtype=torch.bool)
    cap_dropped = torch.zeros_like(selected_local, dtype=torch.bool)
    selected_nonzero_before_cap = slot_ids[selected_local] != 0
    replacement_cap = int(refined.shape[2])

    if target_mode == "slot0_heavy_pareto":
        _b, _num_candidate_slots, k, _a = score.shape
        slot0_score = score[:, local_slot0]
        local_slot0_tensor = torch.full_like(selected_local, local_slot0)

        keep_top = min(max(int(keep_slow_top_k), 0), k)
        if keep_top > 0:
            top_slow_idx = slot0_score.argsort(dim=1)[:, :keep_top, :]
            protected_top.scatter_(dim=1, index=top_slow_idx, value=True)
            selected_local = torch.where(protected_top, local_slot0_tensor, selected_local)

        selected_score = torch.gather(score, dim=1, index=selected_local[:, None, :, :]).squeeze(1)
        selected_nonzero_before_cap = slot_ids[selected_local] != 0
        replace_candidate = selected_nonzero_before_cap & ~protected_top

        ratio_cap = int(np.floor(max(float(max_replacement_ratio), 0.0) * float(k) + EPS))
        replacement_cap = min(k, ratio_cap)
        if max_replacements_per_agent is not None:
            replacement_cap = min(replacement_cap, max(int(max_replacements_per_agent), 0))

        if replacement_cap <= 0:
            cap_dropped = replace_candidate
            selected_local = torch.where(cap_dropped, local_slot0_tensor, selected_local)
        elif replacement_cap < k:
            improvement = (slot0_score - selected_score).masked_fill(~replace_candidate, float("-inf"))
            keep_replacement = torch.zeros_like(replace_candidate, dtype=torch.bool)
            top_replace_idx = improvement.argsort(dim=1, descending=True)[:, :replacement_cap, :]
            keep_replacement.scatter_(dim=1, index=top_replace_idx, value=True)
            keep_replacement = keep_replacement & replace_candidate
            cap_dropped = replace_candidate & ~keep_replacement
            selected_local = torch.where(cap_dropped, local_slot0_tensor, selected_local)

    selected_actual = slot_ids[selected_local]
    aux = {
        "ade": ade,
        "fde": fde,
        "dade": dade,
        "dfde": dfde,
        "selected_local": selected_local,
        "selected_actual": selected_actual,
        "protected_top": protected_top,
        "cap_dropped": cap_dropped,
        "selected_nonzero_before_cap": selected_nonzero_before_cap,
        "replacement_cap": torch.tensor(replacement_cap, device=refined.device, dtype=torch.long),
    }
    if target_mode in {"slot0_preserving_pareto", "slot0_heavy_pareto"}:
        aux["good"] = good
    return selected_actual, aux


def _gather_selected(refined: torch.Tensor, selected_slots: torch.Tensor) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    b, _s, k, a, t, d = refined.shape
    if tuple(selected_slots.shape) != (b, k, a):
        raise ValueError(f"selected_slots shape {tuple(selected_slots.shape)} does not match refined")
    index = selected_slots.to(device=refined.device, dtype=torch.long)[:, None, :, :, None, None]
    index = index.expand(b, 1, k, a, t, d)
    return torch.gather(refined, dim=1, index=index).squeeze(1)


def _prediction_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    miss_threshold: float,
) -> Dict[str, float]:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, ...], dim=-1)
    ade = dist.mean(dim=-1)
    fde = dist[..., -1]
    valid = agent_mask.bool()
    valid_expanded = valid[:, None, :].expand_as(ade)
    inf = torch.tensor(float("inf"), device=prediction.device, dtype=prediction.dtype)
    ade_min = ade.masked_fill(~valid_expanded, inf).min(dim=1).values
    fde_min = fde.masked_fill(~valid_expanded, inf).min(dim=1).values
    return {
        "ADE_min": float(ade_min[valid].mean().detach().cpu()),
        "FDE_min": float(fde_min[valid].mean().detach().cpu()),
        "ADE_avg": float(ade.mean(dim=1)[valid].mean().detach().cpu()),
        "FDE_avg": float(fde.mean(dim=1)[valid].mean().detach().cpu()),
        "MissRate": float((fde_min > float(miss_threshold))[valid].to(torch.float32).mean().detach().cpu()),
        "num_valid_agents": float(valid.sum().detach().cpu()),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class WeightedMetricAccumulator:
    def __init__(self) -> None:
        self.weight = 0.0
        self.sums: Dict[str, float] = {}

    def add(self, metrics: Mapping[str, float], weight: float) -> None:
        weight_f = float(weight)
        self.weight += weight_f
        for key, value in metrics.items():
            if key == "num_valid_agents":
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * weight_f

    def finalize(self) -> Dict[str, float]:
        if self.weight <= 0:
            return {}
        return {key: value / self.weight for key, value in self.sums.items()}


def _write_imle_chunk(
    *,
    output_dir: Path,
    run_name: str,
    chunk_index: int,
    batch: Mapping[str, Any],
    target_metric: torch.Tensor,
    target_normalized: torch.Tensor,
    meta_data: Mapping[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_datapoints = int(target_metric.shape[0])
    payload = {
        "past_traj": _to_numpy(batch["past_traj"]).astype(np.float32, copy=False),
        "fut_traj": _to_numpy(batch["fut_traj"]).astype(np.float32, copy=False),
        "past_traj_original_scale": _to_numpy(batch["past_traj_original_scale"]).astype(np.float32, copy=False),
        "fut_traj_original_scale": _to_numpy(batch["fut_traj_original_scale"]).astype(np.float32, copy=False),
        "fut_traj_vel": _to_numpy(batch["fut_traj_vel"]).astype(np.float32, copy=False),
        "y_t": _to_numpy(target_normalized[:, None, ...]).astype(np.float32, copy=False),
        "y_pred_data": _to_numpy(target_metric).astype(np.float32, copy=False),
        "meta_data": dict(meta_data),
    }
    output_path = output_dir / f"{run_name}_train_batch_{chunk_index:05d}_{num_datapoints}.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    if int(selected.numel()) <= 0:
        return 0.0
    return float(selected.to(dtype=torch.float32).mean().detach().cpu())


def _build_summary_lines(payload: Mapping[str, Any]) -> List[str]:
    lines = [
        "===== V58N REFINED IMLE TARGETS =====",
        f"subset={payload['args']['subset']} split={payload['args']['split']} target_mode={payload['args']['target_mode']}",
        (
            f"keep_slow_top_k={payload['args'].get('keep_slow_top_k', 0)} "
            f"max_replacement_ratio={payload['args'].get('max_replacement_ratio', 1.0)} "
            f"max_replacements_per_agent={payload['args'].get('max_replacements_per_agent')}"
        ),
        f"output_dir={payload['output']['output_dir']}",
        f"files={len(payload['output']['files'])} num_eval_items={payload['dataset']['num_selected_eval_items']}",
        "",
        "-- target - slow deltas --",
    ]
    metrics = payload["metrics"]
    for metric in ACCURACY_METRICS:
        lines.append(
            f"d{metric}: {metrics[f'd{metric}']:+.6f} "
            f"target={metrics[f'target_{metric}']:.6f} slow={metrics[f'slow_{metric}']:.6f}"
        )
    lines.extend(
        [
            "",
            "-- selection --",
            f"selected_nonzero_ratio={payload['selection']['selected_nonzero_ratio']:.6f}",
            f"selected_slot0_ratio={payload['selection']['selected_slot0_ratio']:.6f}",
            f"selected_nonzero_before_cap_ratio={payload['selection']['selected_nonzero_before_cap_ratio']:.6f}",
            f"protected_slow_top_ratio={payload['selection']['protected_slow_top_ratio']:.6f}",
            f"cap_dropped_nonzero_ratio={payload['selection']['cap_dropped_nonzero_ratio']:.6f}",
            f"accepted_nonzero_better_slot0_ade_ratio={payload['selection']['accepted_nonzero_better_slot0_ade_ratio']:.6f}",
            f"accepted_nonzero_better_slot0_fde_ratio={payload['selection']['accepted_nonzero_better_slot0_fde_ratio']:.6f}",
            f"accepted_nonzero_hurt_slot0_ade_ratio={payload['selection']['accepted_nonzero_hurt_slot0_ade_ratio']:.6f}",
            f"accepted_nonzero_hurt_slot0_fde_ratio={payload['selection']['accepted_nonzero_hurt_slot0_fde_ratio']:.6f}",
        ]
    )
    return lines


def main() -> None:
    args = build_parser().parse_args()
    candidate_slots = _split_ints(str(args.candidate_slots))
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if any(slot < 0 or slot >= int(args.residual_slots) for slot in candidate_slots):
        raise SystemExit("--candidate-slots entries must be within [0, residual_slots)")
    if 0 not in candidate_slots:
        raise SystemExit("--candidate-slots must include 0")
    if int(args.keep_slow_top_k) < 0:
        raise SystemExit("--keep-slow-top-k must be >= 0")
    if float(args.max_replacement_ratio) < 0:
        raise SystemExit("--max-replacement-ratio must be >= 0")
    if args.max_replacements_per_agent is not None and int(args.max_replacements_per_agent) < 0:
        raise SystemExit("--max-replacements-per-agent must be >= 0")

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
    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=(slow_predictor,),
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[build_v58n_refined_imle_targets] warning: selected_samples normalization is diagnostic only")

    refiner = load_social_cvae_teacher_refiner(str(args.refiner_checkpoint), map_location=device).to(device)
    refiner.eval()
    refiner_variant = _checkpoint_variant(str(args.refiner_checkpoint))
    max_slots = int(getattr(refiner.config, "max_residual_slots", 1))
    if int(args.residual_slots) > max_slots:
        raise SystemExit(f"--residual-slots {args.residual_slots} exceeds checkpoint max_residual_slots={max_slots}")

    output_dir = Path(args.output_root).expanduser().resolve() / str(args.subset)
    existing = sorted(output_dir.glob(f"{args.run_name}_train_batch_*.pkl"))
    if existing:
        print(
            "[build_v58n_refined_imle_targets] warning: existing files with this run-name will be overwritten; "
            f"count={len(existing)} output_dir={output_dir.as_posix()}"
        )

    print(
        "[build_v58n_refined_imle_targets] "
        f"subset={args.subset} split={args.split} variant={refiner_variant} "
        f"slots={args.residual_slots} candidate_slots={candidate_slots} keep_k={args.keep_k} "
        f"target_mode={args.target_mode} output_dir={output_dir.as_posix()}"
    )

    slow_acc = WeightedMetricAccumulator()
    target_acc = WeightedMetricAccumulator()
    selection_sums = {
        "selected_nonzero": 0.0,
        "selected_slot0": 0.0,
        "selected_nonzero_before_cap": 0.0,
        "protected_slow_top": 0.0,
        "cap_dropped_nonzero": 0.0,
        "accepted_nonzero_better_slot0_ade": 0.0,
        "accepted_nonzero_better_slot0_fde": 0.0,
        "accepted_nonzero_hurt_slot0_ade": 0.0,
        "accepted_nonzero_hurt_slot0_fde": 0.0,
        "accepted_nonzero_count": 0.0,
        "valid_base_count": 0.0,
    }
    output_files: List[str] = []

    meta_data = {
        "script": "trustmoe_traj.scripts.build_v58n_refined_imle_targets",
        "variant": "v58n_residual_enhanced_imle_target",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_mode": str(args.target_mode),
        "candidate_slots": list(candidate_slots),
        "residual_slots": int(args.residual_slots),
        "keep_k": int(args.keep_k),
        "rank_label_metric": str(args.rank_label_metric),
        "keep_slow_top_k": int(args.keep_slow_top_k),
        "max_replacement_ratio": float(args.max_replacement_ratio),
        "max_replacements_per_agent": (
            None if args.max_replacements_per_agent is None else int(args.max_replacements_per_agent)
        ),
        "normalization_stats": _coerce_jsonable(normalization_stats),
    }

    chunks = list(_iter_chunks(list(enumerate(selected_samples)), int(args.batch_scenes)))
    with torch.no_grad():
        for chunk_index, chunk_pairs in enumerate(chunks, start=1):
            chunk = [sample for _scene_index, sample in chunk_pairs]
            batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
            slow_output = slow_predictor.predict(batch, return_all_states=False)
            if int(slow_output.slow_pred.shape[1]) != int(args.keep_k):
                raise SystemExit(
                    f"Expected slow/base modes == keep_k, got {slow_output.slow_pred.shape[1]} vs {args.keep_k}"
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

            refiner_outputs = refiner.refine(
                slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                num_samples=int(args.residual_slots),
                z_mode="slots",
            )
            refined = refiner_outputs["refined"]
            ground_truth = batch["fut_traj_original_scale"].to(device=device)
            agent_mask = batch["agent_mask"].to(device=device)

            selected_slots, select_aux = _select_slots(
                refined,
                ground_truth,
                candidate_slots=candidate_slots,
                target_mode=str(args.target_mode),
                rank_label_metric=str(args.rank_label_metric),
                improve_margin=float(args.improve_margin),
                hurt_margin=float(args.hurt_margin),
                flat_tolerance=float(args.accept_flat_tolerance),
                strong_improve_margin=float(args.accept_strong_improve_margin),
                keep_slow_top_k=int(args.keep_slow_top_k),
                max_replacement_ratio=float(args.max_replacement_ratio),
                max_replacements_per_agent=args.max_replacements_per_agent,
            )
            target_metric = _gather_selected(refined, selected_slots)
            target_normalized = _normalize_min_max(target_metric, normalization_stats)

            valid_base = agent_mask.bool()[:, None, :].expand_as(selected_slots)
            selected_nonzero = selected_slots != 0
            accepted_nonzero = selected_nonzero & valid_base
            dade = torch.gather(
                select_aux["dade"],
                dim=1,
                index=select_aux["selected_local"][:, None, :, :],
            ).squeeze(1)
            dfde = torch.gather(
                select_aux["dfde"],
                dim=1,
                index=select_aux["selected_local"][:, None, :, :],
            ).squeeze(1)
            valid_base_count = float(valid_base.to(dtype=torch.float32).sum().detach().cpu())
            accepted_count = float(accepted_nonzero.to(dtype=torch.float32).sum().detach().cpu())
            selection_sums["valid_base_count"] += valid_base_count
            selection_sums["accepted_nonzero_count"] += accepted_count
            selection_sums["selected_nonzero"] += float(
                (selected_nonzero & valid_base).to(dtype=torch.float32).sum().detach().cpu()
            )
            selection_sums["selected_slot0"] += float(
                ((selected_slots == 0) & valid_base).to(dtype=torch.float32).sum().detach().cpu()
            )
            selection_sums["selected_nonzero_before_cap"] += float(
                (select_aux["selected_nonzero_before_cap"] & valid_base).to(dtype=torch.float32).sum().detach().cpu()
            )
            selection_sums["protected_slow_top"] += float(
                (select_aux["protected_top"] & valid_base).to(dtype=torch.float32).sum().detach().cpu()
            )
            selection_sums["cap_dropped_nonzero"] += float(
                (select_aux["cap_dropped"] & valid_base).to(dtype=torch.float32).sum().detach().cpu()
            )
            if accepted_count > 0:
                selection_sums["accepted_nonzero_better_slot0_ade"] += float(
                    ((dade < -EPS) & accepted_nonzero).to(dtype=torch.float32).sum().detach().cpu()
                )
                selection_sums["accepted_nonzero_better_slot0_fde"] += float(
                    ((dfde < -EPS) & accepted_nonzero).to(dtype=torch.float32).sum().detach().cpu()
                )
                selection_sums["accepted_nonzero_hurt_slot0_ade"] += float(
                    ((dade > EPS) & accepted_nonzero).to(dtype=torch.float32).sum().detach().cpu()
                )
                selection_sums["accepted_nonzero_hurt_slot0_fde"] += float(
                    ((dfde > EPS) & accepted_nonzero).to(dtype=torch.float32).sum().detach().cpu()
                )

            slow_metrics = _prediction_metrics(
                slow_output.slow_pred,
                ground_truth,
                agent_mask,
                miss_threshold=float(args.miss_threshold),
            )
            target_metrics = _prediction_metrics(
                target_metric,
                ground_truth,
                agent_mask,
                miss_threshold=float(args.miss_threshold),
            )
            weight = float(target_metrics["num_valid_agents"])
            slow_acc.add(slow_metrics, weight)
            target_acc.add(target_metrics, weight)

            output_path = _write_imle_chunk(
                output_dir=output_dir,
                run_name=str(args.run_name),
                chunk_index=chunk_index,
                batch=batch,
                target_metric=target_metric,
                target_normalized=target_normalized,
                meta_data=meta_data,
            )
            output_files.append(output_path.as_posix())

            should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
            if should_log:
                print(
                    f"[build_v58n_refined_imle_targets] chunks={chunk_index}/{len(chunks)} "
                    f"raw_scenes={min(chunk_index * int(args.batch_scenes), len(selected_samples))}/{len(selected_samples)} "
                    f"last_file={output_path.name}"
                )

    slow_summary = slow_acc.finalize()
    target_summary = target_acc.finalize()
    metrics: Dict[str, float] = {}
    for metric_name in ACCURACY_METRICS:
        metrics[f"slow_{metric_name}"] = float(slow_summary[metric_name])
        metrics[f"target_{metric_name}"] = float(target_summary[metric_name])
        metrics[f"d{metric_name}"] = float(target_summary[metric_name] - slow_summary[metric_name])

    valid_base_total = max(selection_sums["valid_base_count"], 1.0)
    accepted_total = max(selection_sums["accepted_nonzero_count"], 1.0)
    selection = {
        "selected_nonzero_ratio": selection_sums["selected_nonzero"] / valid_base_total,
        "selected_slot0_ratio": selection_sums["selected_slot0"] / valid_base_total,
        "selected_nonzero_before_cap_ratio": selection_sums["selected_nonzero_before_cap"] / valid_base_total,
        "protected_slow_top_ratio": selection_sums["protected_slow_top"] / valid_base_total,
        "cap_dropped_nonzero_ratio": selection_sums["cap_dropped_nonzero"] / valid_base_total,
        "accepted_nonzero_better_slot0_ade_ratio": selection_sums["accepted_nonzero_better_slot0_ade"] / accepted_total,
        "accepted_nonzero_better_slot0_fde_ratio": selection_sums["accepted_nonzero_better_slot0_fde"] / accepted_total,
        "accepted_nonzero_hurt_slot0_ade_ratio": selection_sums["accepted_nonzero_hurt_slot0_ade"] / accepted_total,
        "accepted_nonzero_hurt_slot0_fde_ratio": selection_sums["accepted_nonzero_hurt_slot0_fde"] / accepted_total,
        "valid_base_count": selection_sums["valid_base_count"],
        "accepted_nonzero_count": selection_sums["accepted_nonzero_count"],
    }

    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.build_v58n_refined_imle_targets",
            "generated_at": datetime.now(timezone.utc).isoformat(),
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
            "sample_mode": args.sample_mode,
            "agents": agents,
            "data_norm": args.data_norm,
            "rotate": bool(args.rotate),
            "rotate_time_frame": int(args.rotate_time_frame),
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
            "refiner_variant": refiner_variant,
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "metrics": metrics,
        "selection": selection,
        "output": {
            "output_dir": output_dir.as_posix(),
            "files": output_files,
        },
    }

    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / f"{args.run_name}_summary.json"
    summary_txt = Path(args.summary_txt).expanduser().resolve() if args.summary_txt else output_dir / f"{args.run_name}_summary.txt"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_coerce_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = _build_summary_lines(payload)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"summary_json={summary_json.as_posix()}")
    print(f"summary_txt={summary_txt.as_posix()}")


if __name__ == "__main__":
    main()
