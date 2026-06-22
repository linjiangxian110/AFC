"""Official-style evaluation for V55-D high-potential base rankers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner, load_v55_base_ranker
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _add_branch,
    _all_indices,
    _base_for_flat,
    _flatten_refined,
    _gather_candidates,
    _oracle_indices,
    _predictor_cfg,
    _set_seed,
    _slot0_indices,
)
from trustmoe_traj.scripts.diagnose_v55_adaptive_base_budget import (
    BUDGETS,
    _base_order,
    _budget_indices,
    _candidate_score,
)
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
EVAL_BUDGETS: Sequence[str] = ("top5x4", "top4x4_next4slot0", "top3x4_next8slot0", "top10x2_slot01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V55-D high-potential base ranker.")
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
    parser.add_argument("--ranker-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _base_score(base: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _rank_aux(base_order: torch.Tensor, base_score: torch.Tensor, mask: torch.Tensor, *, top_k: int) -> Dict[str, float]:
    keep = min(int(top_k), int(base_order.shape[1]))
    target_order = torch.argsort(base_score, dim=1)
    pred_top = base_order[:, :keep, :]
    target_top = target_order[:, :keep, :]
    valid = mask.to(device=base_order.device, dtype=torch.bool)
    if int(valid.sum().item()) <= 0:
        return {"ranker_top1_acc": 0.0, "ranker_topk_best_hit": 0.0, "ranker_topk_recall": 0.0}
    top1_acc = base_order[:, 0, :] == target_order[:, 0, :]
    best_hit = (pred_top == target_order[:, :1, :]).any(dim=1)
    intersection = (pred_top[:, :, None, :] == target_top[:, None, :, :]).any(dim=1).sum(dim=1).to(dtype=torch.float32)
    recall = intersection / float(keep)
    return {
        "ranker_top1_acc": float(top1_acc[valid].float().mean().detach().cpu()),
        "ranker_topk_best_hit": float(best_hit[valid].float().mean().detach().cpu()),
        "ranker_topk_recall": float(recall[valid].mean().detach().cpu()),
    }


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "None"
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.6f}"


def _print_summary(metrics: Mapping[str, float], *, branches: Sequence[str]) -> None:
    print("\n[eval_v55_base_ranker] branch - slow deltas")
    for field_name in branches:
        print(f"\n-- {field_name} --")
        for metric_name in METRICS:
            branch = _metric(metrics, field_name, metric_name)
            slow = _metric(metrics, "slow_pred", metric_name)
            delta = None if branch is None or slow is None else branch - slow
            print(f"d{metric_name}: {_fmt(delta, signed=True)}  branch={_fmt(branch)}  slow={_fmt(slow)}")
        for aux_name in (
            "delta_l2_mean",
            "endpoint_ratio",
            "trajectory_ratio",
            "unique_base_mode_ratio",
            "ranker_top1_acc",
            "ranker_topk_best_hit",
            "ranker_topk_recall",
        ):
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(float(metrics[key]))}")


def _budget_prediction(
    flat: torch.Tensor,
    order: torch.Tensor,
    *,
    budget: str,
    flat_scores: torch.Tensor,
    num_slots: int,
    num_base_modes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _budget_indices(
        order,
        budget=budget,
        flat_scores=flat_scores,
        num_slots=int(num_slots),
        num_base_modes=int(num_base_modes),
    )
    return _gather_candidates(flat, indices), indices


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.keep_k) != 20:
        raise SystemExit("V55-D built-in budgets currently require --keep-k 20")
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
    ranker_variant = _checkpoint_variant(args.ranker_checkpoint)
    refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    ranker = load_v55_base_ranker(args.ranker_checkpoint, map_location=device).to(device)
    refiner.eval()
    ranker.eval()
    if not bool(getattr(refiner.config, "use_set_generator", False)):
        raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")
    if int(args.residual_slots) > int(getattr(refiner.config, "max_residual_slots", 1)):
        raise SystemExit("--residual-slots exceeds checkpoint max_residual_slots")

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

    deterministic_branches = [
        "v38_slot0_20_pred",
        "v38_oracle20_from80_pred",
        "v38_full80_pred",
        "v55c_oracle_base_top5x4_pred",
        "v55c_teacher_order_top5x4_pred",
    ]
    for budget in EVAL_BUDGETS:
        deterministic_branches.append(f"v55d_ranker_{budget}_pred")
    branches = ["slow_pred", *deterministic_branches]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[eval_v55_base_ranker] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} ranker={Path(args.ranker_checkpoint).expanduser().resolve().as_posix()} "
        f"ranker_variant={ranker_variant} refiner_variant={refiner_variant} slots={args.residual_slots}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_v55_base_ranker] warning: selected_samples normalization is diagnostic only")

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
        flat_scores = _candidate_score(flat, ground_truth, metric=str(args.oracle_select_metric))
        base_score = _base_score(slow_output.slow_pred, ground_truth, metric=str(args.oracle_select_metric))
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])
        num_slots = int(refined.shape[1])

        ranker_latencies, ranker_outputs = _measure_predict_latency_ms(
            lambda: ranker.rank(
                slow_output.slow_pred,
                refined_trajectory=refined,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        ranker_order = ranker_outputs["base_order"]
        ranker_branch_latencies = [
            float(refiner_ms) + float(ranker_ms)
            for refiner_ms, ranker_ms in zip(refiner_latencies, ranker_latencies)
        ]
        ranker_aux = _rank_aux(ranker_order, base_score, batch["agent_mask"].to(device=device).bool(), top_k=5)
        valid_count = int(batch["agent_mask"].bool().sum().item())

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

        oracle_base_order = _base_order(
            ranker="oracle_base",
            base=slow_output.slow_pred,
            ground_truth=ground_truth,
            temporal_energy=temporal_energy,
            metric=str(args.oracle_select_metric),
        )
        teacher_order = _base_order(
            ranker="teacher_order",
            base=slow_output.slow_pred,
            ground_truth=ground_truth,
            temporal_energy=temporal_energy,
            metric=str(args.oracle_select_metric),
        )
        for field_name, order in (
            ("v55c_oracle_base_top5x4_pred", oracle_base_order),
            ("v55c_teacher_order_top5x4_pred", teacher_order),
        ):
            prediction, indices = _budget_prediction(
                flat,
                order,
                budget="top5x4",
                flat_scores=flat_scores,
                num_slots=num_slots,
                num_base_modes=num_base_modes,
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=prediction,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=_gather_candidates(base_flat, indices),
                spread_base=slow_output.slow_pred,
                selected_flat_indices=indices,
                num_base_modes=num_base_modes,
            )

        for budget in EVAL_BUDGETS:
            field_name = f"v55d_ranker_{budget}_pred"
            prediction, indices = _budget_prediction(
                flat,
                ranker_order,
                budget=budget,
                flat_scores=flat_scores,
                num_slots=num_slots,
                num_base_modes=num_base_modes,
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=prediction,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=ranker_branch_latencies,
                base_for_delta=_gather_candidates(base_flat, indices),
                spread_base=slow_output.slow_pred,
                selected_flat_indices=indices,
                num_base_modes=num_base_modes,
            )
            aux_accumulators[field_name].add(ranker_aux, weight=valid_count)

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_v55_base_ranker] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[field_name].finalize(field_name))

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_v55_base_ranker",
            "variant": "v55d_high_potential_base_ranker",
            "ranker_variant": ranker_variant,
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "oracle_select_metric": args.oracle_select_metric,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "deterministic_branches": list(deterministic_branches),
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
        "ranker_checkpoint": Path(args.ranker_checkpoint).expanduser().resolve().as_posix(),
        "metrics": _coerce_jsonable(metrics),
    }
    _print_summary(metrics, branches=deterministic_branches)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
