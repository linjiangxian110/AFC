"""Export teacher/student prediction caches for Residual Graduate training.

The first Residual Graduate stage needs fixed slow-teacher and fast-student
outputs on the train split. This script reuses the same MoFlow wrapper and
protocol path as ``run_eval.py``, but saves tensors instead of aggregate
metrics so the next training step can learn residual corrections offline.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

try:  # pragma: no cover - numpy is present in normal project environments.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import displacement_errors, infer_ground_truth_from_batch
from trustmoe_traj.models import MoFlowFastPredictor, MoFlowSlowPredictor
from trustmoe_traj.scripts.interaction_energy_features import (
    build_per_agent_scene_interaction_features,
    build_per_agent_scene_temporal_interaction_features,
)
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _build_base_per_sample_records,
    _build_predictor_config,
    _coerce_jsonable,
    _count_selected_eval_items,
    _infer_agents,
    _is_benchmark_comparable_run,
    _is_diagnostic_normalization_source,
    _iter_chunks,
    _ordered_per_sample_records,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "teacher_student_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export fixed fast-student / slow-teacher predictions for Residual Graduate training."
    )
    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument(
        "--normalization-source",
        type=str,
        default="auto",
        choices=NORMALIZATION_SOURCES,
        help="auto / selected_samples / train_split / predictor_cfg",
    )
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Seed used before exporting stochastic predictions")

    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--fast-checkpoint", type=str, default=None)
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--slow-sampling-steps", type=int, default=None)
    parser.add_argument("--slow-solver", type=str, default=None, choices=["euler", "lin_poly"])
    parser.add_argument("--slow-lin-poly-p", type=int, default=None)
    parser.add_argument("--slow-lin-poly-long-step", type=int, default=None)
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow export without checkpoints. Intended only for smoke tests.",
    )

    parser.add_argument(
        "--output-cache",
        type=str,
        default=None,
        help="Output .pt path. Defaults to analysis/teacher_student_cache/<protocol>_<subset>_<split>.pt",
    )
    parser.add_argument(
        "--output-summary-json",
        type=str,
        default=None,
        help="Optional lightweight JSON summary path for quick inspection.",
    )
    parser.add_argument(
        "--include-interaction-energy-features",
        action="store_true",
        help="Precompute V14 scene-aware interaction energy features for each per-agent cache row.",
    )
    parser.add_argument(
        "--include-temporal-interaction-energy-features",
        action="store_true",
        help="Precompute V17-B per-timestep scene-aware interaction energy features.",
    )
    parser.add_argument(
        "--include-teacher-interaction-energy-features",
        action="store_true",
        help="Precompute scene-aware interaction energy features from slow-teacher predictions.",
    )
    parser.add_argument(
        "--include-teacher-temporal-interaction-energy-features",
        action="store_true",
        help="Precompute per-timestep scene-aware interaction energy features from slow-teacher predictions.",
    )
    parser.add_argument("--collision-sigma", type=float, default=0.5)
    parser.add_argument("--collision-radius", type=float, default=0.2)
    parser.add_argument("--no-neighbor-distance", type=float, default=10.0)
    parser.add_argument(
        "--interaction-energy-temporal-stride",
        type=int,
        default=1,
        help="Use every Nth future step when precomputing interaction energy. The endpoint is always included.",
    )

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _default_output_cache(args: argparse.Namespace) -> Path:
    filename = f"{args.protocol}_{args.subset}_{args.split}_teacher_student_predictions.pt"
    return DEFAULT_OUTPUT_DIR / filename


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.slow_checkpoint:
        missing.append("slow")
    if not args.fast_checkpoint:
        missing.append("fast")
    if missing and not args.allow_random_init:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing checkpoint path for baseline(s): {joined}. "
            "Provide --slow-checkpoint/--fast-checkpoint or pass --allow-random-init for smoke tests."
        )

    if args.sample_mode != "per_agent":
        raise SystemExit("The first Residual Graduate export version supports --sample-mode per_agent only.")

    if args.protocol == "official_align" and not args.rotate:
        print(
            "[export_teacher_student_predictions] warning: official_align exports usually require "
            "--rotate --rotate-time-frame 6 for current ETH checkpoints."
        )


def _as_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _as_cpu_bool(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").bool().contiguous()


def _coerce_prediction_shape(prediction: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(prediction, dtype=torch.float32)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape [B, A, T, 2] or [B, K, A, T, 2], got {tuple(tensor.shape)}")
    return tensor


def _cat_tensors(chunks: Iterable[torch.Tensor], *, name: str) -> torch.Tensor:
    tensors = list(chunks)
    if not tensors:
        raise ValueError(f"No tensors collected for {name}")
    return torch.cat(tensors, dim=0).contiguous()


def _gather_best_mode_prediction(prediction: torch.Tensor, best_modes: torch.Tensor) -> torch.Tensor:
    pred = _coerce_prediction_shape(prediction, name="prediction")
    mode_index = best_modes.to(device=pred.device, dtype=torch.long)
    if mode_index.shape != (pred.shape[0], pred.shape[2]):
        raise ValueError(
            "best_modes must have shape [B, A], got "
            f"{tuple(mode_index.shape)} for prediction shape {tuple(pred.shape)}"
        )
    gather_index = mode_index[:, None, :, None, None].expand(pred.shape[0], 1, pred.shape[2], pred.shape[3], pred.shape[4])
    return pred.gather(dim=1, index=gather_index).squeeze(1).contiguous()


def _prediction_error_payload(
    prediction: Any,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    prefix: str,
    miss_threshold: float,
) -> Dict[str, torch.Tensor]:
    pred = _coerce_prediction_shape(prediction, name=f"{prefix}_pred")
    errors = displacement_errors(pred, ground_truth, agent_mask=agent_mask)
    ade = errors["ade_per_mode_agent"].detach().cpu().to(torch.float32).contiguous()
    fde = errors["fde_per_mode_agent"].detach().cpu().to(torch.float32).contiguous()
    valid = errors["valid_agents"].detach().cpu().bool().contiguous()

    ade_min, ade_best_modes = ade.min(dim=1)
    fde_min, fde_best_modes = fde.min(dim=1)
    return {
        f"{prefix}_ADE_per_mode_agent": ade,
        f"{prefix}_FDE_per_mode_agent": fde,
        f"{prefix}_ADE_min_per_agent": ade_min.contiguous(),
        f"{prefix}_FDE_min_per_agent": fde_min.contiguous(),
        f"{prefix}_ADE_avg_per_agent": ade.mean(dim=1).contiguous(),
        f"{prefix}_FDE_avg_per_agent": fde.mean(dim=1).contiguous(),
        f"{prefix}_best_ADE_mode": ade_best_modes.to(torch.long).contiguous(),
        f"{prefix}_best_FDE_mode": fde_best_modes.to(torch.long).contiguous(),
        f"{prefix}_Miss_per_agent": (fde_min > float(miss_threshold)).bool().contiguous(),
        "valid_agents": valid,
    }


def _tensor_shapes(tensors: Mapping[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {name: [int(dim) for dim in tensor.shape] for name, tensor in tensors.items()}


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    output_cache = Path(args.output_cache).expanduser().resolve() if args.output_cache else _default_output_cache(args)

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
    fast_predictor = MoFlowFastPredictor(
        _build_predictor_config(
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
        predictors=(slow_predictor, fast_predictor),
        samples=selected_samples,
        stats_owner=slow_predictor,
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

    if diagnostic_normalization:
        print(
            "[export_teacher_student_predictions] warning: normalization_source=selected_samples is diagnostic. "
            "Use train_split for checkpoint-backed graduate training unless you are reproducing a debug run."
        )

    tensor_chunks: Dict[str, List[torch.Tensor]] = {
        "student_pred": [],
        "teacher_pred": [],
        "ground_truth": [],
        "agent_mask": [],
        "past_traj_original_scale": [],
        "past_social_risk_features": [],
        "target_agent_indices": [],
        "indexes": [],
    }
    if args.include_interaction_energy_features:
        tensor_chunks["interaction_energy_features"] = []
    if args.include_temporal_interaction_energy_features:
        tensor_chunks["temporal_interaction_energy_features"] = []
    if args.include_teacher_interaction_energy_features:
        tensor_chunks["teacher_interaction_energy_features"] = []
    if args.include_teacher_temporal_interaction_energy_features:
        tensor_chunks["teacher_temporal_interaction_energy_features"] = []
    error_chunks: Dict[str, List[torch.Tensor]] = {}
    records: List[Dict[str, Any]] = []
    next_eval_item_index = 0

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))

    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        global_scene_indices = [int(scene_index) for scene_index, _sample in chunk_pairs]
        chunk = [sample for _scene_index, sample in chunk_pairs]
        chunk_records, next_eval_item_index = _build_base_per_sample_records(
            samples=chunk,
            global_scene_indices=global_scene_indices,
            sample_mode=args.sample_mode,
            eval_item_offset=next_eval_item_index,
        )

        slow_batch = slow_predictor.build_moflow_batch(
            chunk,
            normalization_stats=normalization_stats,
            as_torch=True,
        )
        fast_batch = fast_predictor.build_moflow_batch(
            chunk,
            normalization_stats=normalization_stats,
            as_torch=True,
        )

        slow_output = slow_predictor.predict(slow_batch, return_all_states=False)
        fast_output = fast_predictor.predict(fast_batch, num_to_gen=args.num_to_gen)

        if slow_output.slow_pred is None:
            raise RuntimeError("Slow predictor did not return slow_pred")
        if fast_output.fast_pred is None:
            raise RuntimeError("Fast predictor did not return fast_pred")

        if args.include_interaction_energy_features:
            interaction_energy_features = build_per_agent_scene_interaction_features(
                chunk,
                fast_output.fast_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=float(args.collision_sigma),
                collision_radius=float(args.collision_radius),
                no_neighbor_distance=float(args.no_neighbor_distance),
                temporal_stride=int(args.interaction_energy_temporal_stride),
            )
            tensor_chunks["interaction_energy_features"].append(_as_cpu_float(interaction_energy_features))
        if args.include_temporal_interaction_energy_features:
            temporal_interaction_energy_features = build_per_agent_scene_temporal_interaction_features(
                chunk,
                fast_output.fast_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=float(args.collision_sigma),
                collision_radius=float(args.collision_radius),
                no_neighbor_distance=float(args.no_neighbor_distance),
            )
            tensor_chunks["temporal_interaction_energy_features"].append(
                _as_cpu_float(temporal_interaction_energy_features)
            )
        if args.include_teacher_interaction_energy_features:
            teacher_interaction_energy_features = build_per_agent_scene_interaction_features(
                chunk,
                slow_output.slow_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=float(args.collision_sigma),
                collision_radius=float(args.collision_radius),
                no_neighbor_distance=float(args.no_neighbor_distance),
                temporal_stride=int(args.interaction_energy_temporal_stride),
            )
            tensor_chunks["teacher_interaction_energy_features"].append(
                _as_cpu_float(teacher_interaction_energy_features)
            )
        if args.include_teacher_temporal_interaction_energy_features:
            teacher_temporal_interaction_energy_features = build_per_agent_scene_temporal_interaction_features(
                chunk,
                slow_output.slow_pred,
                rotate=bool(args.rotate),
                rotate_time_frame=int(args.rotate_time_frame),
                collision_sigma=float(args.collision_sigma),
                collision_radius=float(args.collision_radius),
                no_neighbor_distance=float(args.no_neighbor_distance),
            )
            tensor_chunks["teacher_temporal_interaction_energy_features"].append(
                _as_cpu_float(teacher_temporal_interaction_energy_features)
            )

        slow_gt_payload = infer_ground_truth_from_batch(slow_batch)
        fast_gt_payload = infer_ground_truth_from_batch(fast_batch)
        ground_truth = slow_gt_payload["ground_truth"]
        agent_mask = slow_gt_payload["agent_mask"]
        if not torch.allclose(ground_truth.cpu(), fast_gt_payload["ground_truth"].cpu(), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Slow and fast batches produced different ground-truth tensors")
        if not torch.equal(agent_mask.cpu().bool(), fast_gt_payload["agent_mask"].cpu().bool()):
            raise RuntimeError("Slow and fast batches produced different agent masks")

        student_pred = _coerce_prediction_shape(fast_output.fast_pred, name="student_pred")
        teacher_pred = _coerce_prediction_shape(slow_output.slow_pred, name="teacher_pred")
        student_errors = _prediction_error_payload(
            student_pred,
            ground_truth,
            agent_mask,
            prefix="student",
            miss_threshold=args.miss_threshold,
        )
        teacher_errors = _prediction_error_payload(
            teacher_pred,
            ground_truth,
            agent_mask,
            prefix="teacher",
            miss_threshold=args.miss_threshold,
        )

        tensor_chunks["student_pred"].append(_as_cpu_float(student_pred))
        tensor_chunks["teacher_pred"].append(_as_cpu_float(teacher_pred))
        tensor_chunks["ground_truth"].append(_as_cpu_float(ground_truth))
        tensor_chunks["agent_mask"].append(_as_cpu_bool(agent_mask))
        tensor_chunks["past_traj_original_scale"].append(_as_cpu_float(slow_batch["past_traj_original_scale"]))
        tensor_chunks["past_social_risk_features"].append(_as_cpu_float(slow_batch["past_social_risk_features"]))
        tensor_chunks["target_agent_indices"].append(slow_batch["target_agent_indices"].detach().cpu().to(torch.long))
        tensor_chunks["indexes"].append(slow_batch["indexes"].detach().cpu().to(torch.long))

        for name, tensor in {**student_errors, **teacher_errors}.items():
            if name == "valid_agents":
                continue
            error_chunks.setdefault(name, []).append(tensor.detach().cpu().contiguous())

        ordered_chunk_records = _ordered_per_sample_records(chunk_records)
        for record in ordered_chunk_records:
            record["cache_row_index"] = len(records)
            records.append(record)

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(args.log_every, 1) == 0
        if should_log:
            print(
                f"[export_teacher_student_predictions] processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    tensors = {name: _cat_tensors(chunks, name=name) for name, chunks in tensor_chunks.items()}
    tensors.update({name: _cat_tensors(chunks, name=name) for name, chunks in error_chunks.items()})
    tensors["teacher_advantage_ADE_min"] = (
        tensors["student_ADE_min_per_agent"] - tensors["teacher_ADE_min_per_agent"]
    ).contiguous()
    tensors["teacher_advantage_FDE_min"] = (
        tensors["student_FDE_min_per_agent"] - tensors["teacher_FDE_min_per_agent"]
    ).contiguous()
    tensors["student_best_FDE_pred"] = _gather_best_mode_prediction(
        tensors["student_pred"],
        tensors["student_best_FDE_mode"],
    )
    tensors["teacher_best_FDE_pred"] = _gather_best_mode_prediction(
        tensors["teacher_pred"],
        tensors["teacher_best_FDE_mode"],
    )

    num_rows = int(tensors["ground_truth"].shape[0])
    if len(records) != num_rows:
        raise RuntimeError(f"Record count mismatch: records={len(records)} tensor_rows={num_rows}")

    missing_checkpoint_branches = [
        name
        for name, path in (("slow", args.slow_checkpoint), ("fast", args.fast_checkpoint))
        if path is None
    ]
    cache_payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_teacher_student_predictions",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "residual_graduate_teacher_student_prediction_cache",
            "formal_baseline": len(missing_checkpoint_branches) == 0,
            "device": device,
            "seed": args.seed,
            "missing_checkpoint_branches": missing_checkpoint_branches,
            "protocol": protocol_settings.protocol,
            "normalization_source": protocol_settings.normalization_source,
            "diagnostic_normalization": diagnostic_normalization,
            "benchmark_comparable": benchmark_comparable,
            "record_granularity": "eval_item_agent",
            "tensor_coordinate_frame": "MoFlow ETH metric relative future frame",
            "includes_interaction_energy_features": bool(args.include_interaction_energy_features),
            "includes_temporal_interaction_energy_features": bool(args.include_temporal_interaction_energy_features),
            "includes_teacher_interaction_energy_features": bool(args.include_teacher_interaction_energy_features),
            "includes_teacher_temporal_interaction_energy_features": bool(
                args.include_teacher_temporal_interaction_energy_features
            ),
            "includes_past_social_risk_features": True,
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
            "split": args.split,
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
            "slow_checkpoint": args.slow_checkpoint,
            "fast_checkpoint": args.fast_checkpoint,
            "slow_cfg_path": args.slow_cfg_path,
            "fast_cfg_path": args.fast_cfg_path,
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "tensor_shapes": _tensor_shapes(tensors),
        "tensors": tensors,
        "records": records,
    }

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, output_cache)

    summary = {
        "meta": cache_payload["meta"],
        "dataset": cache_payload["dataset"],
        "predictor": cache_payload["predictor"],
        "protocol_settings": cache_payload["protocol_settings"],
        "checkpoints": cache_payload["checkpoints"],
        "normalization_meta": cache_payload["normalization_meta"],
        "tensor_shapes": cache_payload["tensor_shapes"],
        "output_cache": output_cache.as_posix(),
        "num_records": len(records),
        "teacher_advantage": {
            "ADE_min_mean": float(tensors["teacher_advantage_ADE_min"][tensors["agent_mask"]].mean().item()),
            "FDE_min_mean": float(tensors["teacher_advantage_FDE_min"][tensors["agent_mask"]].mean().item()),
            "teacher_better_FDE_ratio": float((tensors["teacher_advantage_FDE_min"][tensors["agent_mask"]] > 0).float().mean().item()),
        },
    }

    if args.output_summary_json:
        summary_path = Path(args.output_summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(_coerce_jsonable(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("[export_teacher_student_predictions] completed")
    print(
        f"subset={args.subset} split={args.split} protocol={protocol_settings.protocol} "
        f"sample_mode={args.sample_mode}"
    )
    print(
        f"selected_scenes={len(selected_samples)} selected_eval_items={selected_eval_items} "
        f"tensor_rows={num_rows}"
    )
    print(
        f"loaded_from_cache={dataset.loaded_from_cache} prefer_cache={protocol_settings.prefer_cache} "
        f"cache_compatible={dataset.cache_compatible}"
    )
    print(f"normalization_source={protocol_settings.normalization_source}")
    print(f"rotate={bool(args.rotate)} rotate_time_frame={int(args.rotate_time_frame)}")
    print(f"teacher_better_FDE_ratio={summary['teacher_advantage']['teacher_better_FDE_ratio']}")
    print(f"output_cache={output_cache.as_posix()}")
    if args.output_summary_json:
        print(f"output_summary_json={Path(args.output_summary_json).expanduser().resolve().as_posix()}")


if __name__ == "__main__":
    main()
