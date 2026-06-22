"""Official-style evaluation for V18-B fine-tuned fast students."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowFastPredictor, MoFlowPredictorConfig, MoFlowSlowPredictor
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


METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a V18-B fine-tuned fast student.")
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
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

    parser.add_argument("--fast-cfg-path", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--finetuned-cfg-path", type=str, default=None)
    parser.add_argument("--finetuned-checkpoint", type=str, required=True)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _validate_checkpoint_requirements(args: argparse.Namespace) -> None:
    if args.include_slow and not args.slow_checkpoint:
        raise SystemExit("--slow-checkpoint is required when --include-slow is set")


def _predictor(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    cfg_path: Optional[str],
    checkpoint_path: str,
) -> MoFlowFastPredictor:
    return MoFlowFastPredictor(
        MoFlowPredictorConfig(
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
    )


def _slow_predictor(args: argparse.Namespace, *, agents: int, device: str) -> MoFlowSlowPredictor:
    return MoFlowSlowPredictor(
        MoFlowPredictorConfig(
            subset=args.subset,
            sample_mode=args.sample_mode,
            agents=agents,
            data_norm=args.data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
            num_to_gen=int(args.num_to_gen),
        )
    )


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _print_delta_summary(metrics: Mapping[str, float]) -> None:
    print("\n[eval_student_integrated_finetune] FinetunedFast - Fast")
    for name in METRICS:
        tuned = _metric(metrics, "finetuned_fast_pred", name)
        fast = _metric(metrics, "fast_pred", name)
        delta = None if tuned is None or fast is None else tuned - fast
        delta_text = "None" if delta is None else f"{delta:+.6f}"
        tuned_text = "None" if tuned is None else f"{tuned:.6f}"
        fast_text = "None" if fast is None else f"{fast:.6f}"
        print(f"d{name}: {delta_text}  finetuned_fast={tuned_text}  fast={fast_text}")


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

    fast_predictor = _predictor(
        args=args,
        agents=agents,
        device=device,
        cfg_path=args.fast_cfg_path,
        checkpoint_path=args.fast_checkpoint,
    )
    finetuned_predictor = _predictor(
        args=args,
        agents=agents,
        device=device,
        cfg_path=args.finetuned_cfg_path or args.fast_cfg_path,
        checkpoint_path=args.finetuned_checkpoint,
    )
    slow_predictor = _slow_predictor(args, agents=agents, device=device) if args.include_slow else None

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[item for item in (fast_predictor, finetuned_predictor, slow_predictor) if item is not None],
        samples=selected_samples,
        stats_owner=fast_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    fast_predictor._set_normalization_stats(normalization_stats)
    finetuned_predictor._set_normalization_stats(normalization_stats)
    if slow_predictor is not None:
        slow_predictor._set_normalization_stats(normalization_stats)

    accumulators: Dict[str, BranchAccumulator] = {
        "fast_pred": BranchAccumulator("fast_pred", args.miss_threshold),
        "finetuned_fast_pred": BranchAccumulator("finetuned_fast_pred", args.miss_threshold),
    }
    if slow_predictor is not None:
        accumulators["slow_pred"] = BranchAccumulator("slow_pred", args.miss_threshold)

    print(
        "[eval_student_integrated_finetune] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} finetuned={Path(args.finetuned_checkpoint).expanduser().resolve().as_posix()}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_student_integrated_finetune] warning: selected_samples normalization is diagnostic only")

    chunks = list(_iter_chunks(selected_samples, args.batch_scenes))
    for chunk_index, chunk in enumerate(chunks, start=1):
        batch = fast_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        fast_latencies, fast_output = _measure_predict_latency_ms(
            lambda: fast_predictor.predict(batch, num_to_gen=args.num_to_gen),
            runs=int(args.latency_runs),
            device=device,
        )
        fast_summary = evaluate_model_output(
            fast_output,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("fast_pred",),
        )
        accumulators["fast_pred"].add_chunk(fast_summary.metrics, fast_latencies)

        tuned_latencies, tuned_output = _measure_predict_latency_ms(
            lambda: finetuned_predictor.predict(batch, num_to_gen=args.num_to_gen),
            runs=int(args.latency_runs),
            device=device,
        )
        tuned_payload = {"finetuned_fast_pred": tuned_output.fast_pred}
        tuned_summary = evaluate_model_output(
            tuned_payload,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("finetuned_fast_pred",),
        )
        accumulators["finetuned_fast_pred"].add_chunk(tuned_summary.metrics, tuned_latencies)

        if slow_predictor is not None:
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

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_student_integrated_finetune] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_student_integrated_finetune",
            "variant": "v18b_fast_student_decoder_finetune",
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
