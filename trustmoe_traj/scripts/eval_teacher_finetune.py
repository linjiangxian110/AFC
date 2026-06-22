"""Official-style evaluation for V23 teacher fine-tuned checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

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
    parser = argparse.ArgumentParser(description="Evaluate a V23 fine-tuned slow teacher checkpoint.")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--finetuned-checkpoint", type=str, required=True)
    parser.add_argument("--finetuned-cfg-path", type=str, default=None)
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


def _capture_rng(device: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any], device: str) -> None:
    torch.random.set_rng_state(state["cpu"])
    if torch.device(device).type == "cuda" and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _print_delta_summary(metrics: Mapping[str, float]) -> None:
    print("\n[eval_teacher_finetune] TeacherFinetune - Slow")
    for name in METRICS:
        finetuned = _metric(metrics, "teacher_finetune_pred", name)
        slow = _metric(metrics, "slow_pred", name)
        delta = None if finetuned is None or slow is None else finetuned - slow
        delta_text = "None" if delta is None else f"{delta:+.6f}"
        finetuned_text = "None" if finetuned is None else f"{finetuned:.6f}"
        slow_text = "None" if slow is None else f"{slow:.6f}"
        print(f"d{name}: {delta_text}  teacher_finetune={finetuned_text}  slow={slow_text}")


def main() -> None:
    args = build_parser().parse_args()
    if args.include_fast and (not args.fast_cfg_path or not args.fast_checkpoint):
        raise SystemExit("--fast-cfg-path and --fast-checkpoint are required with --include-fast")
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
    finetuned_cfg_path = args.finetuned_cfg_path or args.slow_cfg_path

    slow_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    finetuned_predictor = MoFlowSlowPredictor(
        _predictor_cfg(
            args=args,
            agents=agents,
            device=device,
            cfg_path=finetuned_cfg_path,
            checkpoint_path=args.finetuned_checkpoint,
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

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[item for item in (slow_predictor, finetuned_predictor, fast_predictor) if item is not None],
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)
    finetuned_predictor._set_normalization_stats(normalization_stats)
    if fast_predictor is not None:
        fast_predictor._set_normalization_stats(normalization_stats)

    accumulators: Dict[str, BranchAccumulator] = {
        "slow_pred": BranchAccumulator("slow_pred", args.miss_threshold),
        "teacher_finetune_pred": BranchAccumulator("teacher_finetune_pred", args.miss_threshold),
    }
    if fast_predictor is not None:
        accumulators["fast_pred"] = BranchAccumulator("fast_pred", args.miss_threshold)

    print(
        "[eval_teacher_finetune] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} finetuned={Path(args.finetuned_checkpoint).expanduser().resolve().as_posix()}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_teacher_finetune] warning: selected_samples normalization is diagnostic only")

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

        _restore_rng(rng_state, device)
        finetuned_latencies, finetuned_output = _measure_predict_latency_ms(
            lambda: finetuned_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
        finetuned_payload = {"teacher_finetune_pred": finetuned_output.slow_pred}
        finetuned_summary = evaluate_model_output(
            finetuned_payload,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("teacher_finetune_pred",),
        )
        accumulators["teacher_finetune_pred"].add_chunk(finetuned_summary.metrics, finetuned_latencies)

        if fast_predictor is not None:
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
                "[eval_teacher_finetune] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for _field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_teacher_finetune",
            "variant": "v23a_teacher_flow_temporal_energy_finetune",
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
        "finetuned_checkpoint": Path(args.finetuned_checkpoint).expanduser().resolve().as_posix(),
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
