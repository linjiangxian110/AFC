"""Official-style evaluation for per-base residual slot selectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_group_selector, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _add_branch,
    _all_indices,
    _base_for_flat,
    _flatten_refined,
    _gather_candidates,
    _oracle_indices,
    _predictor_cfg,
    _slot0_indices,
    _set_seed,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import _checkpoint_variant, _local_temporal_energy
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.train_social_cvae_selector import _select_candidates, _selector_indices
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


def _branch_names(branch_family: str) -> Dict[str, str]:
    if branch_family == "v57c":
        return {
            "selector": "v57c_conservative_semantic_slot_selector20_pred",
            "per_base_oracle": "v57c_per_base_oracle20_pred",
            "slot0": "v57a_slot0_20_pred",
            "oracle_pool": "v57a_oracle20_from_semantic_pool_pred",
            "full_pool": "v57a_full_semantic_pool_pred",
        }
    if branch_family == "v57b":
        return {
            "selector": "v57b_semantic_slot_selector20_pred",
            "per_base_oracle": "v57b_per_base_oracle20_pred",
            "slot0": "v57a_slot0_20_pred",
            "oracle_pool": "v57a_oracle20_from_semantic_pool_pred",
            "full_pool": "v57a_full_semantic_pool_pred",
        }
    return {
        "selector": "v55_slot_selector20_pred",
        "per_base_oracle": "v55_per_base_oracle20_pred",
        "slot0": "v38_slot0_20_pred",
        "oracle_pool": "v38_oracle20_from80_pred",
        "full_pool": "v38_full80_pred",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate per-base residual slot selectors.")
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
    parser.add_argument("--selector-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--branch-family", type=str, default="v55a", choices=["v55a", "v57b", "v57c"])
    parser.add_argument("--confidence-fallback-to-mean", action="store_true")
    parser.add_argument("--confidence-fallback-to-slot0", action="store_true")
    parser.add_argument("--fallback-prob-margin", type=float, default=0.05)
    parser.add_argument("--fallback-min-selected-prob", type=float, default=0.35)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _candidate_score(candidates: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _per_base_oracle_slots(candidates: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    return _candidate_score(candidates, ground_truth, metric=metric).argmin(dim=1)


def _slot_flat_indices(slot_index: torch.Tensor, *, num_base_modes: int) -> torch.Tensor:
    modes = torch.arange(num_base_modes, device=slot_index.device, dtype=torch.long)[None, :, None].expand_as(slot_index)
    return slot_index.to(dtype=torch.long) * int(num_base_modes) + modes


def _valid_base_mask(agent_mask: torch.Tensor, num_base_modes: int) -> torch.Tensor:
    return agent_mask.bool()[:, None, :].expand(agent_mask.shape[0], int(num_base_modes), agent_mask.shape[1])


def _mean_on_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    keep = valid.to(device=values.device, dtype=torch.bool)
    if int(keep.sum().item()) <= 0:
        return 0.0
    return float(values[keep].to(dtype=torch.float32).mean().detach().cpu())


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "None"
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.6f}"


def _print_summary(metrics: Mapping[str, float], *, branches: Sequence[str]) -> None:
    print("\n[eval_v55_slot_selector] branch - slow deltas")
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
            "selected_slot_mean",
            "selected_slot0_ratio",
            "per_base_oracle_slot_accuracy",
        ):
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(float(metrics[key]))}")


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if bool(args.confidence_fallback_to_mean) and bool(args.confidence_fallback_to_slot0):
        raise SystemExit("--confidence-fallback-to-mean and --confidence-fallback-to-slot0 are mutually exclusive")
    if not (0.0 <= float(args.fallback_min_selected_prob) <= 1.0):
        raise SystemExit("--fallback-min-selected-prob must be in [0, 1]")
    if str(args.branch_family) == "v57c" and not bool(args.confidence_fallback_to_slot0):
        raise SystemExit("--branch-family v57c requires --confidence-fallback-to-slot0")
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
    selector_variant = _checkpoint_variant(args.selector_checkpoint)
    refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    selector = load_social_cvae_group_selector(args.selector_checkpoint, map_location=device).to(device)
    refiner.eval()
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

    branch_names = _branch_names(str(args.branch_family))
    deterministic_branches = [
        branch_names["selector"],
        branch_names["per_base_oracle"],
        branch_names["slot0"],
        branch_names["oracle_pool"],
        branch_names["full_pool"],
    ]
    branches = ["slow_pred", *deterministic_branches]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[eval_v55_slot_selector] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} selector={Path(args.selector_checkpoint).expanduser().resolve().as_posix()} "
        f"selector_variant={selector_variant} refiner_variant={refiner_variant} "
        f"slots={args.residual_slots} keep_k={args.keep_k} oracle_metric={args.oracle_select_metric}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_v55_slot_selector] warning: selected_samples normalization is diagnostic only")

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
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])

        selector_latencies, selector_outputs = _measure_predict_latency_ms(
            lambda: selector.select(
                refined,
                base_trajectory=slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        selected_slots = _selector_indices(selector_outputs["logits"], args=args).to(dtype=torch.long)
        selected_pred = _select_candidates(refined, selected_slots)
        selected_flat_indices = _slot_flat_indices(selected_slots, num_base_modes=num_base_modes)
        v55_latencies = [
            float(refiner_ms) + float(selector_ms)
            for refiner_ms, selector_ms in zip(refiner_latencies, selector_latencies)
        ]
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=branch_names["selector"],
            prediction=selected_pred,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=v55_latencies,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=selected_flat_indices,
            num_base_modes=num_base_modes,
        )

        oracle_slots = _per_base_oracle_slots(refined, ground_truth, metric=str(args.oracle_select_metric))
        oracle_per_base_pred = _select_candidates(refined, oracle_slots)
        oracle_per_base_indices = _slot_flat_indices(oracle_slots, num_base_modes=num_base_modes)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=branch_names["per_base_oracle"],
            prediction=oracle_per_base_pred,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=oracle_per_base_indices,
            num_base_modes=num_base_modes,
        )

        valid_base = _valid_base_mask(batch["agent_mask"].to(device=device), num_base_modes)
        valid_count = int(batch["agent_mask"].bool().sum().item())
        aux_accumulators[branch_names["selector"]].add(
            {
                "selected_slot_mean": _mean_on_valid(selected_slots, valid_base),
                "selected_slot0_ratio": _mean_on_valid((selected_slots == 0).to(dtype=torch.float32), valid_base),
                "per_base_oracle_slot_accuracy": _mean_on_valid(
                    (selected_slots == oracle_slots).to(dtype=torch.float32),
                    valid_base,
                ),
            },
            weight=valid_count,
        )

        slot0_indices = _slot0_indices(batch_size, num_base_modes, num_agents, device=flat.device)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=branch_names["slot0"],
            prediction=refined[:, 0],
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=slot0_indices,
            num_base_modes=num_base_modes,
        )

        oracle80_indices = _oracle_indices(flat, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=branch_names["oracle_pool"],
            prediction=_gather_candidates(flat, oracle80_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=_gather_candidates(base_flat, oracle80_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=oracle80_indices,
            num_base_modes=num_base_modes,
        )

        full_indices = _all_indices(batch_size, num_candidates, num_agents, device=flat.device)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=branch_names["full_pool"],
            prediction=flat,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=base_flat,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=full_indices,
            num_base_modes=num_base_modes,
        )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_v55_slot_selector] "
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
            "script": "trustmoe_traj.scripts.eval_v55_slot_selector",
            "variant": str(args.branch_family),
            "selector_variant": selector_variant,
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
        "selector_checkpoint": Path(args.selector_checkpoint).expanduser().resolve().as_posix(),
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
