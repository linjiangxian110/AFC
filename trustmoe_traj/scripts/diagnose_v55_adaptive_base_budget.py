"""V55-C adaptive base-budget diagnosis for V38-A residual slots.

The script keeps the V38-A slots4 proposal pool frozen and tests whether a
fair K=20 set can be built by allocating more residual slots to high-potential
base modes.  It is a diagnosis script, not a training script.

Candidate layout follows V38: ``flat_index = slot_id * num_base_modes + base_id``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _add_branch,
    _add_random_aggregates,
    _all_indices,
    _base_for_flat,
    _flatten_refined,
    _gather_candidates,
    _oracle_indices,
    _predictor_cfg,
    _set_seed,
    _slot0_indices,
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
BUDGETS: Sequence[str] = (
    "top5x4",
    "top4x4_next4slot0",
    "top3x4_next8slot0",
    "top10x2_slot01",
    "top10x2_oracle_slots",
)
BASE_RANKERS: Sequence[str] = ("oracle_base", "teacher_order", "energy_risk")
RANDOM_GROUPS: Sequence[str] = ("top5x4", "top10x2_slot01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V55-C adaptive base-budget K=20 reductions.")
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
    parser.add_argument("--residual-slots", type=int, default=4)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--random-trials", type=int, default=20)
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


def _candidate_score(flat: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(flat - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported metric: {metric!r}")


def _energy_risk_score(temporal_energy: torch.Tensor) -> torch.Tensor:
    if temporal_energy.ndim != 5:
        raise ValueError(f"Expected temporal energy [B,K,A,T,C], got {tuple(temporal_energy.shape)}")
    energy = torch.nan_to_num(temporal_energy, nan=0.0, posinf=0.0, neginf=0.0)
    if int(energy.shape[-1]) < 5:
        return energy.mean(dim=-1).mean(dim=-1)
    min_neighbor_distance = energy[..., 0:1].clamp_min(0.0)
    soft_collision_energy = energy[..., 1:2].clamp_min(0.0)
    close_neighbor_count = energy[..., 2:3].clamp_min(0.0)
    approaching_score = energy[..., 3:4].clamp(0.0, 1.0)
    endpoint_crowding_energy = energy[..., 4:5].clamp_min(0.0)
    risk = torch.cat(
        [
            torch.exp(-min_neighbor_distance / 0.5),
            soft_collision_energy / (1.0 + soft_collision_energy),
            close_neighbor_count / (1.0 + close_neighbor_count),
            approaching_score,
            endpoint_crowding_energy / (1.0 + endpoint_crowding_energy),
        ],
        dim=-1,
    )
    return risk.mean(dim=-1).mean(dim=-1)


def _base_order(
    *,
    ranker: str,
    base: torch.Tensor,
    ground_truth: torch.Tensor,
    temporal_energy: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    batch_size, num_modes, num_agents = int(base.shape[0]), int(base.shape[1]), int(base.shape[2])
    if ranker == "oracle_base":
        return torch.argsort(_base_score(base, ground_truth, metric=metric), dim=1)
    if ranker == "teacher_order":
        return torch.arange(num_modes, device=base.device, dtype=torch.long)[None, :, None].expand(
            batch_size,
            num_modes,
            num_agents,
        )
    if ranker == "energy_risk":
        return torch.argsort(_energy_risk_score(temporal_energy.to(device=base.device, dtype=base.dtype)), dim=1)
    raise ValueError(f"Unsupported ranker: {ranker!r}")


def _random_base_order(batch_size: int, num_modes: int, num_agents: int, *, device: torch.device) -> torch.Tensor:
    score = torch.rand(batch_size, num_modes, num_agents, device=device)
    return torch.argsort(score, dim=1)


def _flat_from_base_slots(base_ids: torch.Tensor, slot_ids: torch.Tensor, *, num_base_modes: int) -> torch.Tensor:
    if base_ids.ndim != 3:
        raise ValueError(f"base_ids must have shape [B,M,A], got {tuple(base_ids.shape)}")
    slots = slot_ids.to(device=base_ids.device, dtype=torch.long)
    flat = base_ids[:, :, None, :] + slots[None, None, :, None] * int(num_base_modes)
    return flat.reshape(int(base_ids.shape[0]), int(base_ids.shape[1]) * int(slots.numel()), int(base_ids.shape[2]))


def _budget_indices(
    base_order: torch.Tensor,
    *,
    budget: str,
    flat_scores: torch.Tensor,
    num_slots: int,
    num_base_modes: int,
) -> torch.Tensor:
    if budget == "top5x4":
        return _flat_from_base_slots(base_order[:, :5, :], torch.arange(num_slots), num_base_modes=num_base_modes)
    if budget == "top4x4_next4slot0":
        first = _flat_from_base_slots(base_order[:, :4, :], torch.arange(num_slots), num_base_modes=num_base_modes)
        second = base_order[:, 4:8, :]
        return torch.cat([first, second], dim=1)
    if budget == "top3x4_next8slot0":
        first = _flat_from_base_slots(base_order[:, :3, :], torch.arange(num_slots), num_base_modes=num_base_modes)
        second = base_order[:, 3:11, :]
        return torch.cat([first, second], dim=1)
    if budget == "top10x2_slot01":
        return _flat_from_base_slots(base_order[:, :10, :], torch.arange(2), num_base_modes=num_base_modes)
    if budget == "top10x2_oracle_slots":
        top_bases = base_order[:, :10, :]
        score_by_slot = flat_scores.reshape(flat_scores.shape[0], int(num_slots), int(num_base_modes), flat_scores.shape[2])
        gather_index = top_bases[:, None, :, :].expand(
            int(flat_scores.shape[0]),
            int(num_slots),
            int(top_bases.shape[1]),
            int(flat_scores.shape[2]),
        )
        selected_scores = torch.gather(score_by_slot, dim=2, index=gather_index)
        best_slots = torch.topk(selected_scores, k=2, dim=1, largest=False).indices.permute(0, 2, 1, 3)
        flat = top_bases[:, :, None, :] + best_slots * int(num_base_modes)
        return flat.reshape(int(flat_scores.shape[0]), 20, int(flat_scores.shape[2]))
    raise ValueError(f"Unsupported budget: {budget!r}")


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "None"
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{float(value):.6f}"


def _print_summary(
    metrics: Mapping[str, float],
    *,
    deterministic_branches: Sequence[str],
    random_groups: Mapping[str, Sequence[str]],
) -> None:
    print("\n[diagnose_v55_adaptive_base_budget] branch - slow deltas")
    for field_name in deterministic_branches:
        print(f"\n-- {field_name} --")
        for metric_name in METRICS:
            branch = _metric(metrics, field_name, metric_name)
            slow = _metric(metrics, "slow_pred", metric_name)
            delta = None if branch is None or slow is None else branch - slow
            print(f"d{metric_name}: {_fmt(delta, signed=True)}  branch={_fmt(branch)}  slow={_fmt(slow)}")
        for aux_name in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(float(metrics[key]))}")
    for group_name, trial_branches in random_groups.items():
        print(f"\n-- {group_name} ({len(trial_branches)} trials) --")
        for metric_name in METRICS:
            mean_key = f"{group_name}_mean_{metric_name}"
            std_key = f"{group_name}_std_{metric_name}"
            slow = _metric(metrics, "slow_pred", metric_name)
            mean_value = metrics.get(mean_key)
            delta = None if mean_value is None or slow is None else float(mean_value) - slow
            print(
                f"mean d{metric_name}: {_fmt(delta, signed=True)}  "
                f"mean={_fmt(mean_value)}  std={_fmt(metrics.get(std_key))}  slow={_fmt(slow)}"
            )


def main() -> None:
    args = build_parser().parse_args()
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.keep_k) != 20:
        raise SystemExit("V55-C built-in budgets currently require --keep-k 20")
    if int(args.random_trials) < 0:
        raise SystemExit("--random-trials must be non-negative")

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
    ]
    for ranker in BASE_RANKERS:
        for budget in BUDGETS:
            deterministic_branches.append(f"v55c_{ranker}_{budget}_pred")
    random_groups = {
        f"v55c_random_base_{budget}": [
            f"v55c_random_base_{budget}_t{trial:02d}_pred" for trial in range(int(args.random_trials))
        ]
        for budget in RANDOM_GROUPS
    }
    random_branches = [branch for branches in random_groups.values() for branch in branches]
    branches = ["slow_pred", *deterministic_branches, *random_branches]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[diagnose_v55_adaptive_base_budget] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots} random_trials={args.random_trials}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_v55_adaptive_base_budget] warning: selected_samples normalization is diagnostic only")

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
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])
        num_slots = int(refined.shape[1])

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

        base_orders: Dict[str, torch.Tensor] = {
            ranker: _base_order(
                ranker=ranker,
                base=slow_output.slow_pred,
                ground_truth=ground_truth,
                temporal_energy=temporal_energy,
                metric=str(args.oracle_select_metric),
            )
            for ranker in BASE_RANKERS
        }
        for ranker, order in base_orders.items():
            for budget in BUDGETS:
                field_name = f"v55c_{ranker}_{budget}_pred"
                indices = _budget_indices(
                    order,
                    budget=budget,
                    flat_scores=flat_scores,
                    num_slots=num_slots,
                    num_base_modes=num_base_modes,
                )
                _add_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=field_name,
                    prediction=_gather_candidates(flat, indices),
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=refiner_latencies,
                    base_for_delta=_gather_candidates(base_flat, indices),
                    spread_base=slow_output.slow_pred,
                    selected_flat_indices=indices,
                    num_base_modes=num_base_modes,
                )

        for trial_index in range(int(args.random_trials)):
            order = _random_base_order(batch_size, num_base_modes, num_agents, device=flat.device)
            for budget in RANDOM_GROUPS:
                field_name = f"v55c_random_base_{budget}_t{trial_index:02d}_pred"
                indices = _budget_indices(
                    order,
                    budget=budget,
                    flat_scores=flat_scores,
                    num_slots=num_slots,
                    num_base_modes=num_base_modes,
                )
                _add_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=field_name,
                    prediction=_gather_candidates(flat, indices),
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=refiner_latencies,
                    base_for_delta=_gather_candidates(base_flat, indices),
                    spread_base=slow_output.slow_pred,
                    selected_flat_indices=indices,
                    num_base_modes=num_base_modes,
                )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_v55_adaptive_base_budget] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[field_name].finalize(field_name))
    for group_name, trial_branches in random_groups.items():
        _add_random_aggregates(metrics, group_name=group_name, trial_branches=trial_branches)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_v55_adaptive_base_budget",
            "variant": "v55c_adaptive_base_budget_diagnosis",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "oracle_select_metric": args.oracle_select_metric,
            "random_trials": int(args.random_trials),
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "deterministic_branches": list(deterministic_branches),
        "random_groups": {key: list(value) for key, value in random_groups.items()},
        "budgets": list(BUDGETS),
        "base_rankers": list(BASE_RANKERS),
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
        "metrics": _coerce_jsonable(metrics),
    }
    _print_summary(metrics, deterministic_branches=deterministic_branches, random_groups=random_groups)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
