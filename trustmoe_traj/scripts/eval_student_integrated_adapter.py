"""Official-style evaluation for the V18-A student-integrated adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
    load_student_integrated_adapter,
)
from trustmoe_traj.scripts.interaction_energy_features import build_per_agent_scene_temporal_interaction_features
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    BranchAccumulator,
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


ACCURACY_METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V18-A student-integrated adapter on ETH splits.")
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

    parser.add_argument("--adapter-checkpoint", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument(
        "--collision-sigma",
        type=float,
        default=0.5,
        help="Temporal energy feature collision sigma, matching cache export when possible.",
    )
    parser.add_argument("--collision-radius", type=float, default=0.2)
    parser.add_argument("--no-neighbor-distance", type=float, default=10.0)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _validate_checkpoint_requirements(args: argparse.Namespace) -> None:
    if not args.fast_checkpoint:
        raise SystemExit("--fast-checkpoint is required")
    if args.include_slow and not args.slow_checkpoint:
        raise SystemExit("--slow-checkpoint is required when --include-slow is set")


def _normalization_mismatch(
    adapter_stats: Mapping[str, Any],
    eval_stats: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    mismatches: Dict[str, Dict[str, float]] = {}
    for key in ("fut_traj_min", "fut_traj_max"):
        if key not in adapter_stats or key not in eval_stats:
            continue
        adapter_value = float(adapter_stats[key])
        eval_value = float(eval_stats[key])
        if abs(adapter_value - eval_value) > float(tolerance):
            mismatches[key] = {"adapter": adapter_value, "eval": eval_value}
    return mismatches


def _adapter_predict(
    *,
    adapter: torch.nn.Module,
    fast_predictor: MoFlowFastPredictor,
    fast_output: Any,
    fast_batch: Mapping[str, Any],
    temporal_features: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if fast_output.fast_pred is None:
        raise RuntimeError("Fast output is missing fast_pred")
    fast_norm = fast_output.extras.get("fast_pred_normalized")
    if fast_norm is None:
        raise RuntimeError("Fast output is missing fast_pred_normalized")
    result = adapter(
        fast_norm,
        past_traj_original_scale=fast_batch["past_traj_original_scale"],
        temporal_interaction_energy_features=temporal_features,
        return_dict=True,
    )
    adapted_metric = fast_predictor._to_metric_scale(result["adapted_future_normalized"])
    return {
        "student_integrated_pred": adapted_metric,
        "gate": result["gate"],
        "delta_normalized": result["delta_normalized"],
    }


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _print_delta_summary(metrics: Mapping[str, float]) -> None:
    print("\n[eval_student_integrated_adapter] StudentIntegrated - Fast")
    for name in ACCURACY_METRICS:
        adapted = _metric(metrics, "student_integrated_pred", name)
        fast = _metric(metrics, "fast_pred", name)
        delta = None if adapted is None or fast is None else adapted - fast
        signed = "None" if delta is None else f"{delta:+.6f}"
        adapted_text = "None" if adapted is None else f"{adapted:.6f}"
        fast_text = "None" if fast is None else f"{fast:.6f}"
        print(f"d{name}: {signed}  student_integrated={adapted_text}  fast={fast_text}")
    if "student_integrated_gate_mean" in metrics:
        print(f"student_integrated_gate_mean={metrics['student_integrated_gate_mean']:.6f}")
    if "student_integrated_delta_l2_mean" in metrics:
        print(f"student_integrated_delta_l2_mean={metrics['student_integrated_delta_l2_mean']:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_checkpoint_requirements(args)
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
            MoFlowPredictorConfig(
                subset=args.subset,
                sample_mode=args.sample_mode,
                agents=agents,
                data_norm=args.data_norm,
                rotate=args.rotate,
                rotate_time_frame=args.rotate_time_frame,
                device=device,
                cfg_path=args.slow_cfg_path,
                checkpoint_path=args.slow_checkpoint,
                num_to_gen=args.num_to_gen,
            )
        )

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[item for item in (fast_predictor, slow_predictor) if item is not None],
        samples=selected_samples,
        stats_owner=fast_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )

    adapter, adapter_payload = load_student_integrated_adapter(args.adapter_checkpoint, map_location=device)
    adapter.to(device)
    adapter.eval()
    adapter_stats = adapter_payload.get("normalization_stats", {})
    norm_mismatch = _normalization_mismatch(adapter_stats, normalization_stats)
    if norm_mismatch:
        print(
            "[eval_student_integrated_adapter] warning: adapter/eval normalization stats differ: "
            f"{norm_mismatch}"
        )

    accumulators: Dict[str, BranchAccumulator] = {
        "fast_pred": BranchAccumulator("fast_pred", args.miss_threshold),
        "student_integrated_pred": BranchAccumulator("student_integrated_pred", args.miss_threshold),
    }
    if slow_predictor is not None:
        accumulators["slow_pred"] = BranchAccumulator("slow_pred", args.miss_threshold)

    head_latencies: List[float] = []
    gate_means: List[float] = []
    delta_l2_means: List[float] = []
    chunks = list(_iter_chunks(selected_samples, args.batch_scenes))

    print(
        "[eval_student_integrated_adapter] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} adapter={Path(args.adapter_checkpoint).expanduser().resolve().as_posix()}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_student_integrated_adapter] warning: selected_samples normalization is diagnostic only")

    for chunk_index, chunk in enumerate(chunks, start=1):
        fast_batch = fast_predictor.build_moflow_batch(
            chunk,
            normalization_stats=normalization_stats,
            as_torch=True,
        )
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

        if slow_predictor is not None:
            slow_batch = slow_predictor.build_moflow_batch(
                chunk,
                normalization_stats=normalization_stats,
                as_torch=True,
            )
            slow_latencies, slow_output = _measure_predict_latency_ms(
                lambda: slow_predictor.predict(slow_batch, return_all_states=False),
                runs=int(args.latency_runs),
                device=device,
            )
            slow_summary = evaluate_model_output(
                slow_output,
                slow_batch,
                miss_threshold=float(args.miss_threshold),
                prediction_fields=("slow_pred",),
            )
            accumulators["slow_pred"].add_chunk(slow_summary.metrics, slow_latencies)

        temporal_features = None
        if bool(adapter.config.use_temporal_energy):
            temporal_features = build_per_agent_scene_temporal_interaction_features(
                chunk,
                fast_output.fast_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=float(args.collision_sigma),
                collision_radius=float(args.collision_radius),
                no_neighbor_distance=float(args.no_neighbor_distance),
            )

        with torch.no_grad():
            adapter_latencies, adapter_output = _measure_predict_latency_ms(
                lambda: _adapter_predict(
                    adapter=adapter,
                    fast_predictor=fast_predictor,
                    fast_output=fast_output,
                    fast_batch=fast_batch,
                    temporal_features=temporal_features,
                ),
                runs=int(args.latency_runs),
                device=device,
            )
        adapted_summary = evaluate_model_output(
            adapter_output,
            fast_batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("student_integrated_pred",),
        )
        total_latencies = [
            float(fast_latency) + float(adapter_latency)
            for fast_latency, adapter_latency in zip(fast_latencies, adapter_latencies)
        ]
        accumulators["student_integrated_pred"].add_chunk(adapted_summary.metrics, total_latencies)
        head_latencies.extend(float(item) for item in adapter_latencies)
        gate_means.append(float(adapter_output["gate"].detach().mean().cpu()))
        delta_l2_means.append(float(adapter_output["delta_normalized"].detach().pow(2).mean().sqrt().cpu()))

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_student_integrated_adapter] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
    metrics["student_integrated_head_latency_avg_ms"] = (
        float(sum(head_latencies) / len(head_latencies)) if head_latencies else 0.0
    )
    metrics["student_integrated_gate_mean"] = float(sum(gate_means) / len(gate_means)) if gate_means else 0.0
    metrics["student_integrated_delta_l2_mean"] = (
        float(sum(delta_l2_means) / len(delta_l2_means)) if delta_l2_means else 0.0
    )

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_student_integrated_adapter",
            "variant": "v18a_student_integrated_adapter",
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
        "adapter_checkpoint": Path(args.adapter_checkpoint).expanduser().resolve().as_posix(),
        "adapter_meta": _coerce_jsonable(adapter_payload.get("meta", {})),
        "adapter_normalization_mismatch": _coerce_jsonable(norm_mismatch),
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
