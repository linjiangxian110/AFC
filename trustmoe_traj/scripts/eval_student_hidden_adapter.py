"""Official-style evaluation for student hidden-token adapters."""

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
from trustmoe_traj.models import (
    MoFlowFastPredictor,
    MoFlowPredictorConfig,
    MoFlowSlowPredictor,
    load_student_hidden_adapter,
)
from trustmoe_traj.scripts.eval_student_integrated_adapter import _normalization_mismatch
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    BranchAccumulator,
    _build_base_per_sample_records,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_benchmark_comparable_run,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _measure_predict_latency_ms,
    _ordered_per_sample_records,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _update_per_sample_records_for_branch,
    _validate_protocol_assumptions,
)


METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a student hidden-token adapter.")
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
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic per-sample diagnosis.")

    rotate_group = parser.add_mutually_exclusive_group()
    rotate_group.add_argument("--rotate", dest="rotate", action="store_true")
    rotate_group.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.set_defaults(rotate=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)

    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.2,
        help="Distance threshold for prediction collision proxy in dataset coordinate units.",
    )
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--fast-cfg-path", type=str, required=True)
    parser.add_argument("--fast-checkpoint", type=str, required=True)
    parser.add_argument("--adapter-checkpoint", type=str, required=True)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-per-sample-json",
        type=str,
        default=None,
        help="Optional JSON path for saving per eval-item fast/hidden/slow ADE/FDE details.",
    )

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


def _variant_name(adapter: torch.nn.Module, checkpoint_payload: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(checkpoint_payload, Mapping):
        meta = checkpoint_payload.get("meta", {})
        if isinstance(meta, Mapping) and meta.get("variant"):
            return str(meta["variant"])
    site = str(getattr(adapter.config, "adapter_site", "readout"))
    uses_social = bool(getattr(adapter.config, "use_past_social_risk", False))
    if site == "query":
        return "v20a_query_past_social_risk_hidden_adapter" if uses_social else "v20a_query_hidden_adapter"
    return "v19b_past_social_risk_readout_hidden_adapter" if uses_social else "v19a_readout_hidden_adapter"


def _validate_checkpoint_requirements(args: argparse.Namespace) -> None:
    if args.include_slow and not args.slow_checkpoint:
        raise SystemExit("--slow-checkpoint is required when --include-slow is set")


def _predictor(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
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
            cfg_path=args.fast_cfg_path,
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
    print("\n[eval_student_hidden_adapter] HiddenAdapter - Fast")
    for name in METRICS:
        hidden = _metric(metrics, "hidden_adapter_pred", name)
        fast = _metric(metrics, "fast_pred", name)
        delta = None if hidden is None or fast is None else hidden - fast
        delta_text = "None" if delta is None else f"{delta:+.6f}"
        hidden_text = "None" if hidden is None else f"{hidden:.6f}"
        fast_text = "None" if fast is None else f"{fast:.6f}"
        print(f"d{name}: {delta_text}  hidden_adapter={hidden_text}  fast={fast_text}")
    if "hidden_adapter_gate_mean" in metrics:
        print(f"hidden_adapter_gate_mean={metrics['hidden_adapter_gate_mean']:.6f}")
    if "hidden_adapter_delta_l2_mean" in metrics:
        print(f"hidden_adapter_delta_l2_mean={metrics['hidden_adapter_delta_l2_mean']:.6f}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_checkpoint_requirements(args)
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

    fast_predictor = _predictor(args=args, agents=agents, device=device, checkpoint_path=args.fast_checkpoint)
    hidden_predictor = _predictor(args=args, agents=agents, device=device, checkpoint_path=args.fast_checkpoint)
    adapter, adapter_payload = load_student_hidden_adapter(args.adapter_checkpoint, map_location=device)
    adapter_site = str(getattr(adapter.config, "adapter_site", "readout"))
    if adapter_site == "query":
        hidden_predictor.attach_student_query_adapter(adapter)
    else:
        hidden_predictor.attach_student_hidden_adapter(adapter)
    slow_predictor = _slow_predictor(args, agents=agents, device=device) if args.include_slow else None

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[item for item in (fast_predictor, hidden_predictor, slow_predictor) if item is not None],
        samples=selected_samples,
        stats_owner=fast_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    fast_predictor._set_normalization_stats(normalization_stats)
    hidden_predictor._set_normalization_stats(normalization_stats)
    if slow_predictor is not None:
        slow_predictor._set_normalization_stats(normalization_stats)

    adapter_stats = adapter_payload.get("normalization_stats", {})
    norm_mismatch = _normalization_mismatch(adapter_stats, normalization_stats)
    if norm_mismatch:
        print(f"[eval_student_hidden_adapter] warning: adapter/eval normalization stats differ: {norm_mismatch}")

    accumulators: Dict[str, BranchAccumulator] = {
        "fast_pred": BranchAccumulator("fast_pred", args.miss_threshold),
        "hidden_adapter_pred": BranchAccumulator("hidden_adapter_pred", args.miss_threshold),
    }
    if slow_predictor is not None:
        accumulators["slow_pred"] = BranchAccumulator("slow_pred", args.miss_threshold)

    print(
        "[eval_student_hidden_adapter] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} adapter={Path(args.adapter_checkpoint).expanduser().resolve().as_posix()}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_student_hidden_adapter] warning: selected_samples normalization is diagnostic only")

    gate_means: List[float] = []
    delta_l2_means: List[float] = []
    per_sample_records: List[Dict[str, Any]] = []
    next_eval_item_index = 0
    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        global_scene_indices = [int(scene_index) for scene_index, _sample in chunk_pairs]
        chunk = [sample for _scene_index, sample in chunk_pairs]
        chunk_per_sample_records: Dict[tuple[int, int], Dict[str, Any]] = {}
        if args.output_per_sample_json:
            chunk_per_sample_records, next_eval_item_index = _build_base_per_sample_records(
                samples=chunk,
                global_scene_indices=global_scene_indices,
                sample_mode=args.sample_mode,
                eval_item_offset=next_eval_item_index,
            )

        batch = fast_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        rng_state = _capture_rng(device)
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
        if args.output_per_sample_json:
            _update_per_sample_records_for_branch(
                chunk_per_sample_records,
                branch_name="fast_pred",
                output=fast_output,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                collision_threshold=float(args.collision_threshold),
            )

        _restore_rng(rng_state, device)
        hidden_latencies, hidden_output = _measure_predict_latency_ms(
            lambda: hidden_predictor.predict(batch, num_to_gen=args.num_to_gen),
            runs=int(args.latency_runs),
            device=device,
        )
        hidden_payload = {"hidden_adapter_pred": hidden_output.fast_pred}
        hidden_summary = evaluate_model_output(
            hidden_payload,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("hidden_adapter_pred",),
        )
        accumulators["hidden_adapter_pred"].add_chunk(hidden_summary.metrics, hidden_latencies)
        if adapter.last_gate is not None:
            gate_means.append(float(adapter.last_gate.detach().mean().cpu()))
        if adapter.last_delta is not None:
            delta_l2_means.append(float(adapter.last_delta.detach().pow(2).mean().sqrt().cpu()))
        if args.output_per_sample_json:
            _update_per_sample_records_for_branch(
                chunk_per_sample_records,
                branch_name="hidden_adapter_pred",
                output=hidden_payload,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                collision_threshold=float(args.collision_threshold),
            )

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
            if args.output_per_sample_json:
                _update_per_sample_records_for_branch(
                    chunk_per_sample_records,
                    branch_name="slow_pred",
                    output=slow_output,
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    collision_threshold=float(args.collision_threshold),
                )

        if args.output_per_sample_json:
            per_sample_records.extend(_ordered_per_sample_records(chunk_per_sample_records))

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_student_hidden_adapter] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for _field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
    metrics["hidden_adapter_gate_mean"] = float(sum(gate_means) / len(gate_means)) if gate_means else 0.0
    metrics["hidden_adapter_delta_l2_mean"] = float(sum(delta_l2_means) / len(delta_l2_means)) if delta_l2_means else 0.0

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_student_hidden_adapter",
            "variant": _variant_name(adapter, adapter_payload),
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
    if args.output_per_sample_json:
        payload["per_sample"] = {
            "output_path": Path(args.output_per_sample_json).expanduser().as_posix(),
            "num_records": len(per_sample_records),
            "record_granularity": "eval_item_agent",
        }
    _print_delta_summary(metrics)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")
    if args.output_per_sample_json:
        per_sample_path = Path(args.output_per_sample_json).expanduser().resolve()
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_payload = {
            "meta": {
                **payload["meta"],
                "subset": args.subset,
                "split": args.split,
                "sample_mode": args.sample_mode,
                "adapter_checkpoint": payload["adapter_checkpoint"],
            },
            "args": _coerce_jsonable(vars(args)),
            "dataset": payload["dataset"],
            "normalization_meta": payload["normalization_meta"],
            "adapter_meta": payload["adapter_meta"],
            "records": per_sample_records,
        }
        per_sample_path.write_text(
            json.dumps(_coerce_jsonable(per_sample_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"output_per_sample_json={per_sample_path.as_posix()}")
        print(f"per_sample_records={len(per_sample_records)}")


if __name__ == "__main__":
    main()
