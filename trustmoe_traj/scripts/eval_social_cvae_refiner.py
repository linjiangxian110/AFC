"""Official-style evaluation for V24 SocialCVAE teacher trajectory refiners."""

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
    compute_temporal_interaction_energy_features,
    load_social_cvae_teacher_refiner,
)
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


METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a V24 SocialCVAE teacher-output residual refiner.")
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
    parser.add_argument("--num-residual-samples", type=int, default=1)
    parser.add_argument("--z-mode", type=str, default="mean", choices=["mean", "sample", "slots"])
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
    print("\n[eval_social_cvae_refiner] SocialCVAERefiner - Slow")
    for name in METRICS:
        refined = _metric(metrics, "social_cvae_refiner_pred", name)
        slow = _metric(metrics, "slow_pred", name)
        delta = None if refined is None or slow is None else refined - slow
        delta_text = "None" if delta is None else f"{delta:+.6f}"
        refined_text = "None" if refined is None else f"{refined:.6f}"
        slow_text = "None" if slow is None else f"{slow:.6f}"
        print(f"d{name}: {delta_text}  social_cvae_refiner={refined_text}  slow={slow_text}")
    if "social_cvae_refiner_delta_l2_mean" in metrics:
        print(f"delta_l2_mean: {metrics['social_cvae_refiner_delta_l2_mean']}")
    if "social_cvae_refiner_dynamic_slot_offset_l2_mean" in metrics:
        print(f"dynamic_slot_offset_l2_mean: {metrics['social_cvae_refiner_dynamic_slot_offset_l2_mean']}")
    if "social_cvae_refiner_energy_risk_mean" in metrics:
        print(f"energy_risk_mean: {metrics['social_cvae_refiner_energy_risk_mean']}")


def _flatten_refined(refined: torch.Tensor) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected refined [B,S,K,A,T,2], got {tuple(refined.shape)}")
    b, s, k, a, t, d = refined.shape
    return refined.reshape(b, s * k, a, t, d)


def _checkpoint_variant(path: str) -> str:
    try:
        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    except Exception:
        return "social_cvae_teacher_trajectory_refiner"
    if isinstance(checkpoint, Mapping):
        meta = checkpoint.get("meta", {})
        if isinstance(meta, Mapping) and meta.get("variant"):
            return str(meta["variant"])
    return "social_cvae_teacher_trajectory_refiner"


def _local_temporal_energy(batch: Mapping[str, torch.Tensor], prediction: torch.Tensor) -> torch.Tensor:
    past = batch["past_traj_original_scale"].to(device=prediction.device, dtype=prediction.dtype)
    past_abs = past[..., :2]
    future_abs = prediction + past_abs[:, None, :, -1:, :]
    return compute_temporal_interaction_energy_features(
        future_abs,
        past_abs,
        agent_mask=batch.get("agent_mask"),
        collision_sigma=0.5,
        collision_radius=0.2,
        no_neighbor_distance=10.0,
    )


def _energy_risk_mean(energy: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    if energy.ndim != 5 or int(energy.shape[-1]) < 5:
        return 0.0, 0
    min_neighbor_distance = energy[..., 0].clamp_min(0.0)
    soft_collision_energy = energy[..., 1].clamp_min(0.0)
    close_neighbor_count = energy[..., 2].clamp_min(0.0)
    approaching_score = energy[..., 3].clamp_min(0.0)
    endpoint_crowding_energy = energy[..., 4].clamp_min(0.0)
    risk = torch.stack(
        [
            torch.exp(-min_neighbor_distance / 0.5),
            soft_collision_energy / (1.0 + soft_collision_energy),
            close_neighbor_count / (1.0 + close_neighbor_count),
            approaching_score.clamp(0.0, 1.0),
            endpoint_crowding_energy / (1.0 + endpoint_crowding_energy),
        ],
        dim=0,
    ).amax(dim=0)
    valid = mask.to(device=risk.device, dtype=torch.bool)
    expanded = valid[:, None, :, None].expand_as(risk)
    count = int(expanded.sum().item())
    if count <= 0:
        return 0.0, 0
    return float(risk[expanded].mean().detach().cpu()), count


def main() -> None:
    args = build_parser().parse_args()
    if args.include_fast and (not args.fast_cfg_path or not args.fast_checkpoint):
        raise SystemExit("--fast-cfg-path and --fast-checkpoint are required with --include-fast")
    if int(args.num_residual_samples) <= 0:
        raise SystemExit("--num-residual-samples must be positive")
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
    refiner = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    refiner.eval()
    use_energy_conditioned_generator = bool(getattr(refiner.config, "use_energy_conditioned_generator", False))
    use_set_generator = bool(getattr(refiner.config, "use_set_generator", False))
    if str(args.z_mode) == "slots" and not use_set_generator:
        raise SystemExit("--z-mode slots requires a refiner checkpoint trained with use_set_generator=True")

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
        "social_cvae_refiner_pred": BranchAccumulator("social_cvae_refiner_pred", args.miss_threshold),
    }
    if fast_predictor is not None:
        accumulators["fast_pred"] = BranchAccumulator("fast_pred", args.miss_threshold)

    print(
        "[eval_social_cvae_refiner] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} residual_samples={args.num_residual_samples} z_mode={args.z_mode} "
        f"energy_conditioned_generator={use_energy_conditioned_generator} set_generator={use_set_generator}"
    )
    if int(args.num_residual_samples) != 1:
        print("[eval_social_cvae_refiner] warning: num_residual_samples != 1 changes the number of evaluated modes.")
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_social_cvae_refiner] warning: selected_samples normalization is diagnostic only")

    aux_weight = 0
    aux_delta_sum = 0.0
    aux_dynamic_slot_offset_sum = 0.0
    aux_dynamic_slot_offset_seen = False
    aux_risk_sum = 0.0
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

        refiner_latencies, refiner_outputs = _measure_predict_latency_ms(
            lambda: refiner.refine(
                slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                num_samples=int(args.num_residual_samples),
                z_mode=str(args.z_mode),
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        refined_pred = _flatten_refined(refiner_outputs["refined"])
        refiner_payload = {"social_cvae_refiner_pred": refined_pred}
        refiner_summary = evaluate_model_output(
            refiner_payload,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=("social_cvae_refiner_pred",),
        )
        accumulators["social_cvae_refiner_pred"].add_chunk(refiner_summary.metrics, refiner_latencies)
        valid_count = int(batch["agent_mask"].bool().sum().item())
        if valid_count > 0:
            delta_l2 = torch.linalg.norm(refiner_outputs["delta"], dim=-1).mean(dim=-1).mean().detach().cpu()
            aux_delta_sum += float(delta_l2) * valid_count
            dynamic_slot_offset = refiner_outputs.get("dynamic_slot_offset")
            if torch.is_tensor(dynamic_slot_offset):
                aux_dynamic_slot_offset_seen = True
                offset_l2 = torch.linalg.norm(dynamic_slot_offset, dim=-1).mean().detach().cpu()
                aux_dynamic_slot_offset_sum += float(offset_l2) * valid_count
            risk_mean, _risk_count = _energy_risk_mean(temporal_energy, batch["agent_mask"].bool())
            aux_risk_sum += float(risk_mean) * valid_count
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
                "[eval_social_cvae_refiner] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for _field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
    if aux_weight > 0:
        metrics["social_cvae_refiner_delta_l2_mean"] = float(aux_delta_sum / aux_weight)
        if aux_dynamic_slot_offset_seen:
            metrics["social_cvae_refiner_dynamic_slot_offset_l2_mean"] = float(
                aux_dynamic_slot_offset_sum / aux_weight
            )
        metrics["social_cvae_refiner_energy_risk_mean"] = float(aux_risk_sum / aux_weight)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_social_cvae_refiner",
            "variant": refiner_variant,
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
