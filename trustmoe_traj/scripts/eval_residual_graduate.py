"""Evaluate Residual Graduate checkpoints on ETH official/internal splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output, summarize_latency_ms
from trustmoe_traj.models import MoFlowFastPredictor, MoFlowSlowPredictor, ResidualGraduateModel
from trustmoe_traj.scripts.interaction_energy_features import (
    build_per_agent_scene_interaction_features,
    build_per_agent_scene_temporal_interaction_features,
)
from trustmoe_traj.scripts.run_eval import (
    BranchAccumulator,
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _build_predictor_config,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Residual Graduate checkpoint on ETH splits.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate", dest="rotate", action="store_true")
    rotate_group.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.set_defaults(rotate=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)

    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--graduate-checkpoint", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-sampling-steps", type=int, default=None)
    parser.add_argument("--slow-solver", type=str, default=None, choices=["euler", "lin_poly"])
    parser.add_argument("--slow-lin-poly-p", type=int, default=None)
    parser.add_argument("--slow-lin-poly-long-step", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _load_graduate_model(checkpoint_path: str | Path, *, device: str) -> tuple[ResidualGraduateModel, Dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid graduate checkpoint payload type: {type(payload)!r}")
    if "model_config" not in payload or "model_state" not in payload:
        raise ValueError("Graduate checkpoint must contain `model_config` and `model_state`")

    model = ResidualGraduateModel(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, dict(payload)


def _mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(item) for item in values) / len(values))


def main() -> None:
    args = build_parser().parse_args()
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    if args.include_slow and not args.slow_checkpoint:
        raise SystemExit("--include-slow requires --slow-checkpoint")

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

    fast_predictor = MoFlowFastPredictor(
        _build_predictor_config(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.fast_cfg_path,
            checkpoint_path=args.fast_checkpoint,
        )
    )
    slow_predictor = None
    if args.include_slow:
        slow_predictor = MoFlowSlowPredictor(
            _build_predictor_config(
                args=args,
                agents=agents,
                device=device,
                cfg_path=args.slow_cfg_path,
                checkpoint_path=args.slow_checkpoint,
                sampling_steps=args.slow_sampling_steps,
                solver=args.slow_solver,
                lin_poly_p=args.slow_lin_poly_p,
                lin_poly_long_step=args.slow_lin_poly_long_step,
            )
        )

    graduate_model, graduate_payload = _load_graduate_model(args.graduate_checkpoint, device=device)
    stats_owner = slow_predictor or fast_predictor
    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=(slow_predictor, fast_predictor),
        samples=selected_samples,
        stats_owner=stats_owner,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    diagnostic_normalization = _is_diagnostic_normalization_source(protocol_settings.normalization_source)
    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )

    accumulators: Dict[str, BranchAccumulator] = {
        "fast_pred": BranchAccumulator("fast_pred", miss_threshold=args.miss_threshold),
        "graduate_pred": BranchAccumulator("graduate_pred", miss_threshold=args.miss_threshold),
    }
    if slow_predictor is not None:
        accumulators["slow_pred"] = BranchAccumulator("slow_pred", miss_threshold=args.miss_threshold)

    graduate_head_latencies: List[float] = []
    graduate_gate_means: List[float] = []
    graduate_delta_l2_means: List[float] = []
    graduate_selector_prob_means: List[float] = []
    graduate_best_refine_l2_means: List[float] = []
    graduate_temporal_gate_means: List[float] = []
    graduate_temporal_refine_l2_means: List[float] = []

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]

        if slow_predictor is not None:
            slow_batch = slow_predictor.build_moflow_batch(
                chunk,
                normalization_stats=normalization_stats,
                as_torch=True,
            )
            slow_latencies, slow_output = _measure_predict_latency_ms(
                lambda: slow_predictor.predict(slow_batch, return_all_states=False),
                runs=args.latency_runs,
                device=device,
            )
            slow_summary = evaluate_model_output(
                slow_output,
                slow_batch,
                miss_threshold=args.miss_threshold,
                prediction_fields=("slow_pred",),
            )
            accumulators["slow_pred"].add_chunk(slow_summary.metrics, slow_latencies)

        fast_batch = fast_predictor.build_moflow_batch(
            chunk,
            normalization_stats=normalization_stats,
            as_torch=True,
        )
        fast_latencies, fast_output = _measure_predict_latency_ms(
            lambda: fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen),
            runs=args.latency_runs,
            device=device,
        )
        fast_summary = evaluate_model_output(
            fast_output,
            fast_batch,
            miss_threshold=args.miss_threshold,
            prediction_fields=("fast_pred",),
        )
        accumulators["fast_pred"].add_chunk(fast_summary.metrics, fast_latencies)

        def _run_graduate_model() -> Dict[str, torch.Tensor]:
            interaction_energy_features = None
            if bool(getattr(graduate_model.config, "use_interaction_energy", False)):
                interaction_energy_features = build_per_agent_scene_interaction_features(
                    chunk,
                    fast_output.fast_pred,
                    rotate=bool(args.rotate),
                    rotate_time_frame=int(args.rotate_time_frame),
                    collision_sigma=float(getattr(graduate_model.config, "collision_sigma", 0.5)),
                    collision_radius=float(getattr(graduate_model.config, "collision_radius", 0.2)),
                    no_neighbor_distance=float(getattr(graduate_model.config, "no_neighbor_distance", 10.0)),
                    temporal_stride=int(getattr(graduate_model.config, "interaction_energy_temporal_stride", 1)),
                )
            temporal_interaction_energy_features = None
            if bool(getattr(graduate_model.config, "use_temporal_energy_refiner", False)):
                temporal_interaction_energy_features = build_per_agent_scene_temporal_interaction_features(
                    chunk,
                    fast_output.fast_pred,
                    rotate=bool(args.rotate),
                    rotate_time_frame=int(args.rotate_time_frame),
                    collision_sigma=float(getattr(graduate_model.config, "collision_sigma", 0.5)),
                    collision_radius=float(getattr(graduate_model.config, "collision_radius", 0.2)),
                    no_neighbor_distance=float(getattr(graduate_model.config, "no_neighbor_distance", 10.0)),
                )
            return graduate_model(
                fast_output.fast_pred,
                fast_batch["past_traj_original_scale"],
                agent_mask=fast_batch.get("agent_mask"),
                interaction_energy_features=interaction_energy_features,
                temporal_interaction_energy_features=temporal_interaction_energy_features,
            )

        head_latencies, graduate_output = _measure_predict_latency_ms(
            _run_graduate_model,
            runs=args.latency_runs,
            device=device,
        )
        graduate_metrics_payload = {"graduate_pred": graduate_output["graduate_pred"]}
        graduate_summary = evaluate_model_output(
            graduate_metrics_payload,
            fast_batch,
            miss_threshold=args.miss_threshold,
            prediction_fields=("graduate_pred",),
        )
        graduate_total_latencies = [
            float(fast_latency) + float(head_latency)
            for fast_latency, head_latency in zip(fast_latencies, head_latencies)
        ]
        accumulators["graduate_pred"].add_chunk(graduate_summary.metrics, graduate_total_latencies)
        graduate_head_latencies.extend(float(item) for item in head_latencies)
        graduate_gate_means.append(float(graduate_output["gate"].detach().mean().cpu()))
        graduate_delta_l2_means.append(float(graduate_output["delta_pred"].detach().pow(2).mean().sqrt().cpu()))
        if "best_mode_selector_prob" in graduate_output:
            graduate_selector_prob_means.append(
                float(graduate_output["best_mode_selector_prob"].detach().mean().cpu())
            )
        if "best_mode_refine_delta" in graduate_output:
            graduate_best_refine_l2_means.append(
                float(graduate_output["best_mode_refine_delta"].detach().pow(2).mean().sqrt().cpu())
            )
        if "temporal_repair_gate" in graduate_output:
            graduate_temporal_gate_means.append(float(graduate_output["temporal_repair_gate"].detach().mean().cpu()))
        if "temporal_refine_delta" in graduate_output:
            graduate_temporal_refine_l2_means.append(
                float(graduate_output["temporal_refine_delta"].detach().pow(2).mean().sqrt().cpu())
            )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(args.log_every, 1) == 0
        if should_log:
            print(
                f"[eval_residual_graduate] processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    aggregated_metrics: Dict[str, float] = {}
    for accumulator in accumulators.values():
        aggregated_metrics.update(accumulator.finalize())
    aggregated_metrics.update(
        {
            f"graduate_head_{key}": value
            for key, value in summarize_latency_ms(graduate_head_latencies).items()
        }
    )
    aggregated_metrics["graduate_gate_mean"] = _mean_float(graduate_gate_means)
    aggregated_metrics["graduate_delta_l2_mean"] = _mean_float(graduate_delta_l2_means)
    if graduate_selector_prob_means:
        aggregated_metrics["graduate_best_selector_prob_mean"] = _mean_float(graduate_selector_prob_means)
    if graduate_best_refine_l2_means:
        aggregated_metrics["graduate_best_refine_delta_l2_mean"] = _mean_float(graduate_best_refine_l2_means)
    if graduate_temporal_gate_means:
        aggregated_metrics["graduate_temporal_gate_mean"] = _mean_float(graduate_temporal_gate_means)
    if graduate_temporal_refine_l2_means:
        aggregated_metrics["graduate_temporal_refine_delta_l2_mean"] = _mean_float(graduate_temporal_refine_l2_means)

    result = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_residual_graduate",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "protocol": protocol_settings.protocol,
            "normalization_source": protocol_settings.normalization_source,
            "diagnostic_normalization": diagnostic_normalization,
            "benchmark_comparable": benchmark_comparable,
            "include_slow": bool(args.include_slow),
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
            "num_to_gen": int(args.num_to_gen),
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
        },
        "protocol_settings": {
            "protocol": protocol_settings.protocol,
            "min_agents": int(protocol_settings.min_agents),
            "prefer_cache": bool(protocol_settings.prefer_cache),
            "normalization_source": protocol_settings.normalization_source,
        },
        "checkpoints": {
            "graduate_checkpoint": Path(args.graduate_checkpoint).expanduser().as_posix(),
            "graduate_epoch": graduate_payload.get("epoch"),
            "graduate_model_config": _coerce_jsonable(graduate_payload.get("model_config", {})),
            "fast_checkpoint": args.fast_checkpoint,
            "fast_cfg_path": args.fast_cfg_path,
            "slow_checkpoint": args.slow_checkpoint,
            "slow_cfg_path": args.slow_cfg_path,
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "available_predictions": list(accumulators.keys()),
        "metrics": aggregated_metrics,
    }

    print("[eval_residual_graduate] completed")
    print(
        f"subset={args.subset} split={args.split} protocol={protocol_settings.protocol} "
        f"sample_mode={args.sample_mode}"
    )
    print(
        f"selected_scenes={len(selected_samples)} selected_eval_items={selected_eval_items} "
        f"batch_scenes={args.batch_scenes} min_agents={protocol_settings.min_agents}"
    )
    print(
        f"loaded_from_cache={dataset.loaded_from_cache} prefer_cache={protocol_settings.prefer_cache} "
        f"cache_compatible={dataset.cache_compatible}"
    )
    print(f"normalization_source={protocol_settings.normalization_source}")
    print(f"benchmark_comparable={benchmark_comparable}")
    print(f"rotate={bool(args.rotate)} rotate_time_frame={int(args.rotate_time_frame)}")
    print(f"graduate_checkpoint={Path(args.graduate_checkpoint).expanduser().as_posix()}")
    for key in sorted(aggregated_metrics):
        print(f"{key}={aggregated_metrics[key]}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_coerce_jsonable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
