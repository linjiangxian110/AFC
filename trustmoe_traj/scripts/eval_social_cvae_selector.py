"""Official-style evaluation for SocialCVAE group selectors."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
    load_social_cvae_group_selector,
    load_social_cvae_teacher_refiner,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import (
    _capture_rng,
    _checkpoint_variant,
    _energy_risk_mean,
    _local_temporal_energy,
    _restore_rng,
)
from trustmoe_traj.scripts.interaction_energy_features import (
    build_per_agent_scene_candidate_interaction_features,
    build_per_agent_scene_candidate_trajectory_aware_interaction_features,
    build_per_agent_scene_temporal_interaction_features,
)
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
from trustmoe_traj.scripts.train_social_cvae_selector import (
    _candidate_energy_summary,
    _candidate_temporal_energy,
    _sample_candidates,
    _select_candidates,
    _selector_indices,
)


METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a SocialCVAE group-wise selector.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument("--residual-samples", type=int, default=10)
    parser.add_argument("--candidate-z-mode", type=str, default="sample", choices=["sample", "slots"])
    parser.add_argument("--include-mean-candidate", action="store_true")
    parser.add_argument("--confidence-fallback-to-mean", action="store_true")
    parser.add_argument("--fallback-prob-margin", type=float, default=0.05)
    parser.add_argument("--fallback-min-selected-prob", type=float, default=0.35)
    parser.add_argument("--include-fast", action="store_true")
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--fast-checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _predictor_cfg(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    cfg_path: str,
    checkpoint_path: str,
) -> MoFlowPredictorConfig:
    return MoFlowPredictorConfig(
        subset=args.subset,
        sample_mode=args.sample_mode,
        agents=agents,
        data_norm=args.data_norm,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        device=device,
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
        num_to_gen=int(args.num_to_gen),
    )


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _print_delta_summary(metrics: Mapping[str, float]) -> None:
    print("\n[eval_social_cvae_selector] SocialCVAESelector - Slow")
    for name in METRICS:
        selected = _metric(metrics, "social_cvae_selector_pred", name)
        slow = _metric(metrics, "slow_pred", name)
        delta = None if selected is None or slow is None else selected - slow
        delta_text = "None" if delta is None else f"{delta:+.6f}"
        selected_text = "None" if selected is None else f"{selected:.6f}"
        slow_text = "None" if slow is None else f"{slow:.6f}"
        print(f"d{name}: {delta_text}  social_cvae_selector={selected_text}  slow={slow_text}")
    for key in (
        "social_cvae_selector_delta_l2_mean",
        "social_cvae_selector_selected_index_mean",
        "social_cvae_selector_selected_mean_ratio",
        "social_cvae_selector_energy_risk_mean",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]}")


def _candidate_scene_temporal_energy(
    candidates: torch.Tensor,
    *,
    args: argparse.Namespace,
    chunk: list[Mapping[str, Any]],
    selector_batch: Mapping[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    if args.sample_mode != "per_agent":
        return _candidate_temporal_energy(candidates, selector_batch)
    pieces = []
    for sample_index in range(int(candidates.shape[1])):
        pieces.append(
            build_per_agent_scene_temporal_interaction_features(
                chunk,
                candidates[:, sample_index, ...],
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=0.5,
                collision_radius=0.2,
                no_neighbor_distance=10.0,
            )
        )
    return torch.stack(pieces, dim=1).to(device=device, dtype=candidates.dtype)


def _candidate_scene_energy_summary(
    candidates: torch.Tensor,
    *,
    args: argparse.Namespace,
    chunk: list[Mapping[str, Any]],
    selector_batch: Mapping[str, torch.Tensor],
    device: str,
    trajectory_aware: bool,
) -> torch.Tensor:
    if args.sample_mode != "per_agent":
        return _candidate_energy_summary(candidates, selector_batch, trajectory_aware=bool(trajectory_aware))
    if bool(trajectory_aware):
        return build_per_agent_scene_candidate_trajectory_aware_interaction_features(
            chunk,
            candidates,
            selector_batch["teacher_pred"],
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            collision_sigma=0.5,
            collision_radius=0.2,
            no_neighbor_distance=10.0,
        ).to(device=device, dtype=candidates.dtype)
    return build_per_agent_scene_candidate_interaction_features(
        chunk,
        candidates,
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        collision_sigma=0.5,
        collision_radius=0.2,
        no_neighbor_distance=10.0,
    ).to(device=device, dtype=candidates.dtype)


def main() -> None:
    args = build_parser().parse_args()
    if args.include_fast and (not args.fast_cfg_path or not args.fast_checkpoint):
        raise SystemExit("--fast-cfg-path and --fast-checkpoint are required with --include-fast")
    if int(args.residual_samples) <= 0:
        raise SystemExit("--residual-samples must be positive")
    _set_seed(args.seed)
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
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    fast_predictor = None
    if args.include_fast:
        fast_predictor = MoFlowFastPredictor(
            _predictor_cfg(
                args=args,
                agents=agents,
                device=device,
                cfg_path=args.fast_cfg_path,
                checkpoint_path=args.fast_checkpoint,
            )
        )
    refiner_variant = _checkpoint_variant(args.refiner_checkpoint)
    selector_variant = _checkpoint_variant(args.selector_checkpoint)
    refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    selector = load_social_cvae_group_selector(args.selector_checkpoint, map_location=device).to(device)
    refiner.eval()
    selector.eval()
    use_candidate_energy_context = bool(getattr(selector.config, "use_candidate_energy_context", False))
    use_candidate_energy_summary_context = bool(
        getattr(selector.config, "use_candidate_energy_summary_context", False)
    )
    use_trajectory_aware_candidate_summary = bool(
        int(getattr(selector.config, "candidate_energy_summary_dim", 0) or 0)
        > int(getattr(selector.config, "temporal_energy_dim", 0) or 0)
    )
    use_energy_gated_fusion = bool(getattr(selector.config, "use_energy_gated_fusion", False))
    use_candidate_safety_penalty = bool(getattr(selector.config, "use_candidate_safety_penalty", False))
    use_residual_accept_gate = bool(getattr(selector.config, "use_residual_accept_gate", False))
    use_base_best_guard = bool(getattr(selector.config, "use_base_best_guard", False))

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[item for item in (slow_predictor, fast_predictor) if item is not None],
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)
    if fast_predictor is not None:
        fast_predictor._set_normalization_stats(normalization_stats)

    accumulators: Dict[str, BranchAccumulator] = {
        "slow_pred": BranchAccumulator("slow_pred", args.miss_threshold),
        "social_cvae_selector_pred": BranchAccumulator("social_cvae_selector_pred", args.miss_threshold),
    }
    if fast_predictor is not None:
        accumulators["fast_pred"] = BranchAccumulator("fast_pred", args.miss_threshold)

    print(
        "[eval_social_cvae_selector] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} selector={Path(args.selector_checkpoint).expanduser().resolve().as_posix()} "
        f"selector_variant={selector_variant} refiner_variant={refiner_variant} "
        f"residual_samples={args.residual_samples} candidate_z_mode={args.candidate_z_mode} "
        f"include_mean={bool(args.include_mean_candidate)} "
        f"candidate_energy={use_candidate_energy_context} "
        f"candidate_energy_summary={use_candidate_energy_summary_context} "
        f"trajectory_aware_summary={use_trajectory_aware_candidate_summary} "
        f"energy_gated_fusion={use_energy_gated_fusion} "
        f"candidate_safety_penalty={use_candidate_safety_penalty} "
        f"residual_accept_gate={use_residual_accept_gate} "
        f"base_best_guard={use_base_best_guard}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_social_cvae_selector] warning: selected_samples normalization is diagnostic only")

    aux_weight = 0
    delta_sum = 0.0
    selected_index_sum = 0.0
    selected_mean_ratio_sum = 0.0
    risk_sum = 0.0
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        rng_state = _capture_rng(device)
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

        selector_batch = dict(batch)
        selector_batch["teacher_pred"] = slow_output.slow_pred
        selector_batch["teacher_temporal_interaction_energy_features"] = temporal_energy.to(device=device)
        selector_batch = {key: value.to(device=device) if torch.is_tensor(value) else value for key, value in selector_batch.items()}
        def _select_with_optional_fallback() -> Dict[str, torch.Tensor]:
            candidates = _sample_candidates(
                refiner,
                selector_batch,
                residual_samples=int(args.residual_samples),
                include_mean_candidate=bool(args.include_mean_candidate),
                z_mode=str(args.candidate_z_mode),
            )
            candidate_energy = (
                _candidate_scene_temporal_energy(
                    candidates,
                    args=args,
                    chunk=chunk,
                    selector_batch=selector_batch,
                    device=device,
                )
                if use_candidate_energy_context
                else None
            )
            candidate_energy_summary = (
                _candidate_scene_energy_summary(
                    candidates,
                    args=args,
                    chunk=chunk,
                    selector_batch=selector_batch,
                    device=device,
                    trajectory_aware=use_trajectory_aware_candidate_summary,
                )
                if use_candidate_energy_summary_context
                else None
            )
            logits = selector(
                candidates,
                base_trajectory=slow_output.slow_pred,
                past_traj_original_scale=selector_batch["past_traj_original_scale"],
                temporal_energy_features=selector_batch["teacher_temporal_interaction_energy_features"],
                candidate_temporal_energy_features=candidate_energy,
                candidate_energy_summary_features=candidate_energy_summary,
            )
            selected_index = _selector_indices(logits, args=args)
            selected = _select_candidates(candidates, selected_index)
            return {"selected": selected, "logits": logits, "selected_index": selected_index}

        selector_latencies, selector_outputs = _measure_predict_latency_ms(
            _select_with_optional_fallback,
            runs=int(args.latency_runs),
            device=device,
        )
        selected_pred = selector_outputs["selected"]
        selector_summary = evaluate_model_output(
            {"social_cvae_selector_pred": selected_pred},
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("social_cvae_selector_pred",),
        )
        accumulators["social_cvae_selector_pred"].add_chunk(selector_summary.metrics, selector_latencies)
        valid_count = int(batch["agent_mask"].bool().sum().item())
        if valid_count > 0:
            delta = torch.linalg.norm(selected_pred - slow_output.slow_pred, dim=-1).mean(dim=-1).mean().detach().cpu()
            delta_sum += float(delta) * valid_count
            selected_valid = batch["agent_mask"].bool()[:, None, :].expand_as(selector_outputs["selected_index"])
            selected_index_sum += (
                float(selector_outputs["selected_index"][selected_valid].float().mean().detach().cpu())
                * valid_count
            )
            selected_mean_ratio_sum += (
                float((selector_outputs["selected_index"][selected_valid] == 0).float().mean().detach().cpu())
                * valid_count
            )
            risk_mean, _risk_count = _energy_risk_mean(temporal_energy, batch["agent_mask"].bool())
            risk_sum += float(risk_mean) * valid_count
            aux_weight += valid_count

        if fast_predictor is not None:
            _restore_rng(rng_state, device)
            fast_batch = fast_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
            fast_latencies, fast_output = _measure_predict_latency_ms(
                lambda: fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen),
                runs=int(args.latency_runs),
                device=device,
            )
            fast_summary = evaluate_model_output(
                fast_output,
                fast_batch,
                miss_threshold=float(args.miss_threshold),
                prediction_fields=("fast_pred",),
            )
            accumulators["fast_pred"].add_chunk(fast_summary.metrics, fast_latencies)

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_social_cvae_selector] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for _field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
    if aux_weight > 0:
        metrics["social_cvae_selector_delta_l2_mean"] = float(delta_sum / aux_weight)
        metrics["social_cvae_selector_selected_index_mean"] = float(selected_index_sum / aux_weight)
        metrics["social_cvae_selector_selected_mean_ratio"] = float(selected_mean_ratio_sum / aux_weight)
        metrics["social_cvae_selector_energy_risk_mean"] = float(risk_sum / aux_weight)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_social_cvae_selector",
            "variant": selector_variant,
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
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
    _print_delta_summary(metrics)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
