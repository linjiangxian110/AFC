"""SDD controlled diagnostic for MoFlow slow sampling and AFC coverage.

This is the SDD counterpart of ``diagnose_headroom_analysis``.  It evaluates an
existing MoFlow SDD slow checkpoint and produces the same branch/metric schema
used by the ETH-UCY headroom summaries:

* slow20: native K=20 slow teacher samples;
* cv_linear20 and random_spread*: weak / fake-diversity controls;
* slow{K}_full, slow{K}_gt_oracle20, slow{K}_afc_greedy20,
  slow{K}_endpoint_fps20, and slow{K}_random20/mean{T}: larger slow-pool
  reductions under the standard AFC protocol.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.sdd import DEFAULT_SDD_DATA_ROOT, SDDAdapterConfig, SDDTrajectoryDataset
from trustmoe_traj.models import MoFlowPredictorConfig, MoFlowSDDSlowPredictor
from trustmoe_traj.scripts.analogical_future_coverage import (
    AFC_FEATURE_VARIANTS,
    attach_afc_metadata_to_batch,
    build_sdd_analogical_future_bank,
    split_float_list,
)
from trustmoe_traj.scripts.diagnose_headroom_analysis import (
    _add_headroom_branch,
    _afc_greedy_indices,
    _constant_velocity_prediction,
    _print_summary,
    _predict_slow_repeated_pool,
    _random_pool_mean_branch_name,
    _random_spread_branch_name,
    _random_spread_prediction,
    _split_ints,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _gather_candidates,
    _oracle_indices,
    _random_global_indices,
    _set_seed,
    _structured_fps_indices,
)
from trustmoe_traj.scripts.run_eval import (
    BranchAccumulator,
    _coerce_jsonable,
    _count_selected_eval_items,
    _iter_chunks,
    _measure_predict_latency_ms,
    _resolve_device,
)


NORMALIZATION_SOURCES: Sequence[str] = ("auto", "selected_samples", "train_split", "predictor_cfg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SDD AFC controlled diagnostic with a MoFlow slow checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_SDD_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_scene", choices=["per_scene", "per_agent"])
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max", "original"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--normalization-max-train-records", type=int, default=None)
    parser.add_argument("--batch-records", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--slow-sampling-steps", type=int, default=None)
    parser.add_argument("--slow-solver", type=str, default=None, choices=["euler", "lin_poly"])
    parser.add_argument("--slow-lin-poly-p", type=int, default=None)
    parser.add_argument("--slow-lin-poly-long-step", type=int, default=None)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--slow-pool-ks", type=str, default="20,50,100,200")
    parser.add_argument("--oracle-select-metric", type=str, default="ade_fde", choices=["fde", "ade_fde"])
    parser.add_argument("--afc-selection-tau", type=float, default=1.0)
    parser.add_argument("--disable-random-pool-selection", action="store_true")
    parser.add_argument("--random-pool-trials", type=int, default=1)
    parser.add_argument("--random-pool-emit-trials", action="store_true")
    parser.add_argument("--disable-cv-linear", action="store_true")
    parser.add_argument("--disable-random-spread", action="store_true")
    parser.add_argument("--random-spread-source", type=str, default="slow_radial", choices=["slow_radial", "cv"])
    parser.add_argument("--random-spread-endpoint-scale", type=float, default=2.0)
    parser.add_argument("--random-spread-endpoint-scales", type=str, default="")
    parser.add_argument("--random-spread-noise-scale", type=float, default=0.05)

    parser.add_argument("--afc-train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.5,1.0")
    parser.add_argument("--afc-feature-variant", type=str, default="full_past_social", choices=AFC_FEATURE_VARIANTS)
    parser.add_argument("--afc-max-train-records", type=int, default=None)
    parser.add_argument("--afc-batch-records", type=int, default=256)
    parser.add_argument("--afc-use-source-metadata", action="store_true")
    parser.add_argument("--afc-source-id-field", type=str, default="source_file")
    parser.add_argument("--afc-filter-same-source", action="store_true")
    parser.add_argument("--afc-temporal-gap-frames", type=int, default=0)
    parser.add_argument("--afc-randomize-bank-seed", type=int, default=None)

    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _select_samples(dataset: SDDTrajectoryDataset, max_records: Optional[int]) -> List[Mapping[str, Any]]:
    limit = len(dataset) if max_records is None else min(int(max_records), len(dataset))
    if limit <= 0:
        raise ValueError("SDD dataset is empty after applying max_records")
    return [dataset[index] for index in range(limit)]


def _build_predictor_config(
    *,
    args: argparse.Namespace,
    agents: int,
    device: str,
    cfg_path: Optional[str],
    checkpoint_path: str,
) -> MoFlowPredictorConfig:
    return MoFlowPredictorConfig(
        dataset="sdd",
        subset="sdd",
        sample_mode=str(args.sample_mode),
        agents=int(agents),
        data_norm=str(args.data_norm),
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        device=device,
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
        num_to_gen=int(args.num_to_gen),
        sampling_steps=args.slow_sampling_steps,
        solver=args.slow_solver,
        lin_poly_p=args.slow_lin_poly_p,
        lin_poly_long_step=args.slow_lin_poly_long_step,
    )


def _resolve_sdd_normalization_stats(
    *,
    args: argparse.Namespace,
    data_root: Path,
    slow_predictor: MoFlowSDDSlowPredictor,
    selected_samples: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    if str(args.data_norm) != "min_max":
        return {}, {"source": "disabled", "reason": f"data_norm={args.data_norm}"}

    requested = str(args.normalization_source)
    if requested == "auto":
        cfg_stats = slow_predictor.get_cfg_normalization_stats()
        if cfg_stats:
            return cfg_stats, {
                "source": "predictor_cfg",
                "source_predictor": slow_predictor.predictor_name,
                "source_cfg_path": slow_predictor.cfg.cfg_path,
                "auto_selected": True,
            }
        requested = "train_split"

    if requested == "predictor_cfg":
        cfg_stats = slow_predictor.get_cfg_normalization_stats()
        if not cfg_stats:
            raise ValueError(
                "normalization_source=predictor_cfg requested, but the SDD slow config has no "
                "past_traj_min/past_traj_max/fut_traj_min/fut_traj_max fields"
            )
        return cfg_stats, {
            "source": "predictor_cfg",
            "source_predictor": slow_predictor.predictor_name,
            "source_cfg_path": slow_predictor.cfg.cfg_path,
        }

    if requested == "selected_samples":
        return slow_predictor.infer_normalization_stats(selected_samples), {
            "source": "selected_samples",
            "source_predictor": slow_predictor.predictor_name,
            "num_source_records": len(selected_samples),
        }

    if requested == "train_split":
        train_dataset = SDDTrajectoryDataset(
            SDDAdapterConfig(
                data_root=data_root,
                split="train",
                max_samples=args.normalization_max_train_records,
            )
        )
        train_samples = _select_samples(train_dataset, args.normalization_max_train_records)
        return slow_predictor.infer_normalization_stats(train_samples), {
            "source": "train_split",
            "source_predictor": slow_predictor.predictor_name,
            "train_num_source_records": len(train_samples),
            "normalization_max_train_records": args.normalization_max_train_records,
        }

    raise ValueError(f"Unsupported normalization_source={requested!r}")


def _is_diagnostic_normalization_source(source: str) -> bool:
    return str(source) == "selected_samples"


def main() -> None:
    args = build_parser().parse_args()
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.random_pool_trials) <= 0:
        raise SystemExit("--random-pool-trials must be positive")
    if int(args.afc_temporal_gap_frames) < 0:
        raise SystemExit("--afc-temporal-gap-frames must be non-negative")

    pool_ks = sorted(set(_split_ints(str(args.slow_pool_ks))))
    if int(args.keep_k) not in pool_ks:
        pool_ks.insert(0, int(args.keep_k))
    if any(item < int(args.keep_k) for item in pool_ks):
        raise SystemExit("--slow-pool-ks entries must be >= keep_k")
    random_spread_scales = (
        split_float_list(str(args.random_spread_endpoint_scales))
        if str(args.random_spread_endpoint_scales).strip()
        else [float(args.random_spread_endpoint_scale)]
    )
    if any(float(item) <= 0 for item in random_spread_scales):
        raise SystemExit("--random-spread endpoint scales must be positive")

    _set_seed(args.seed)
    device = _resolve_device(str(args.device))
    data_root = Path(args.data_root).expanduser().resolve()
    dataset = SDDTrajectoryDataset(
        SDDAdapterConfig(
            data_root=data_root,
            split=str(args.split),
            max_samples=args.max_records,
        )
    )
    selected_samples = _select_samples(dataset, args.max_records)
    agents = 1 if str(args.sample_mode) == "per_agent" else int(args.agents)
    selected_eval_items = _count_selected_eval_items(selected_samples, str(args.sample_mode))

    slow_predictor = MoFlowSDDSlowPredictor(
        _build_predictor_config(
            args=args,
            agents=agents,
            device=device,
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
    normalization_stats, normalization_meta = _resolve_sdd_normalization_stats(
        args=args,
        data_root=data_root,
        slow_predictor=slow_predictor,
        selected_samples=selected_samples,
    )
    slow_predictor._set_normalization_stats(normalization_stats)

    afc_temporal_gap = int(args.afc_temporal_gap_frames) if int(args.afc_temporal_gap_frames) > 0 else None
    afc_needs_metadata = bool(args.afc_use_source_metadata or args.afc_filter_same_source or afc_temporal_gap is not None)
    afc_bank = build_sdd_analogical_future_bank(
        data_root=data_root,
        train_split=str(args.afc_train_split),
        sample_mode=str(args.sample_mode),
        data_norm=str(args.data_norm),
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        normalization_stats=normalization_stats,
        max_train_scenes=args.afc_max_train_records,
        batch_scenes=int(args.afc_batch_records),
        top_m=int(args.afc_top_m),
        eps_values=split_float_list(str(args.afc_eps)),
        feature_variant=str(args.afc_feature_variant),
        include_source_metadata=afc_needs_metadata,
        source_id_field=str(args.afc_source_id_field),
        filter_same_source=bool(args.afc_filter_same_source),
        temporal_gap_frames=afc_temporal_gap,
        randomize_futures_seed=args.afc_randomize_bank_seed,
    )

    branches: List[str] = ["slow20_pred"]
    if not bool(args.disable_cv_linear):
        branches.append("cv_linear20_pred")
    if not bool(args.disable_random_spread):
        for scale in random_spread_scales:
            branches.append(_random_spread_branch_name(float(scale), num_scales=len(random_spread_scales)))
    for pool_k in pool_ks:
        if int(pool_k) == int(args.keep_k):
            continue
        branches.extend(
            [
                f"slow{pool_k}_full_pred",
                f"slow{pool_k}_gt_oracle20_pred",
                f"slow{pool_k}_afc_greedy20_pred",
                f"slow{pool_k}_endpoint_fps20_pred",
            ]
        )
        if not bool(args.disable_random_pool_selection):
            branches.append(_random_pool_mean_branch_name(int(pool_k), int(args.random_pool_trials)))
            if bool(args.random_pool_emit_trials) and int(args.random_pool_trials) > 1:
                for trial_index in range(int(args.random_pool_trials)):
                    branches.append(f"slow{int(pool_k)}_random20_trial{trial_index}_pred")

    accumulators = {branch: BranchAccumulator(branch, args.miss_threshold) for branch in branches}
    aux_accumulators = {branch: AuxAccumulator() for branch in branches}

    print(
        "[diagnose_sdd_headroom_analysis] "
        f"split={args.split} records={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} keep_k={args.keep_k} slow_pool_ks={pool_ks} afc_bank={afc_bank.bank_size} "
        f"afc_feature={args.afc_feature_variant} afc_filter_same_source={args.afc_filter_same_source} "
        f"afc_temporal_gap={afc_temporal_gap or 0} afc_randomize={args.afc_randomize_bank_seed or 'none'} "
        f"normalization_source={normalization_meta.get('source')}"
    )
    if _is_diagnostic_normalization_source(str(normalization_meta.get("source", ""))):
        print("[diagnose_sdd_headroom_analysis] warning: selected_samples normalization is diagnostic only")

    chunks = list(_iter_chunks(list(enumerate(selected_samples)), int(args.batch_records)))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _record_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        if afc_needs_metadata:
            attach_afc_metadata_to_batch(
                batch,
                samples=chunk,
                sample_mode=str(args.sample_mode),
                source_id_field=str(args.afc_source_id_field),
            )
        base_latencies, slow20_output = _measure_predict_latency_ms(
            lambda: slow_predictor.predict(batch, return_all_states=False),
            runs=int(args.latency_runs),
            device=device,
        )
        slow20 = slow20_output.slow_pred
        if int(slow20.shape[1]) != int(args.keep_k):
            raise SystemExit(f"Expected slow20 modes == keep_k, got {slow20.shape[1]} vs {args.keep_k}")

        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        batch_size, _base_modes, num_agents = [int(item) for item in slow20.shape[:3]]
        base_indices = torch.arange(int(args.keep_k), device=slow20.device, dtype=torch.long)[None, :, None].expand(
            batch_size,
            int(args.keep_k),
            num_agents,
        )
        _add_headroom_branch(
            accumulators,
            aux_accumulators,
            field_name="slow20_pred",
            prediction=slow20,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=base_latencies,
            afc_bank=afc_bank,
            spread_base=slow20,
            selected_flat_indices=base_indices,
            num_base_modes=int(args.keep_k),
        )

        if not bool(args.disable_cv_linear):
            cv_linear = _constant_velocity_prediction(batch, keep_k=int(args.keep_k), device=torch.device(device))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name="cv_linear20_pred",
                prediction=cv_linear,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=[0.0],
                afc_bank=afc_bank,
                spread_base=slow20,
            )

        if not bool(args.disable_random_spread):
            for scale in random_spread_scales:
                random_spread = _random_spread_prediction(
                    batch,
                    keep_k=int(args.keep_k),
                    device=torch.device(device),
                    endpoint_scale=float(scale),
                    noise_scale=float(args.random_spread_noise_scale),
                    source=str(args.random_spread_source),
                    base_prediction=slow20,
                )
                _add_headroom_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=_random_spread_branch_name(float(scale), num_scales=len(random_spread_scales)),
                    prediction=random_spread,
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=[0.0],
                    afc_bank=afc_bank,
                    spread_base=slow20,
                )

        for pool_k in pool_ks:
            if int(pool_k) == int(args.keep_k):
                continue
            pool_latencies, pool_output = _measure_predict_latency_ms(
                lambda pool_k=pool_k: _predict_slow_repeated_pool(
                    slow_predictor,
                    batch,
                    pool_k=int(pool_k),
                    first_prediction=slow20,
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            pool_pred = pool_output
            if int(pool_pred.shape[1]) != int(pool_k):
                raise SystemExit(f"Expected slow pool modes == {pool_k}, got {pool_pred.shape[1]}")
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_full_pred",
                prediction=pool_pred,
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            gt_indices = _oracle_indices(pool_pred, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_gt_oracle20_pred",
                prediction=_gather_candidates(pool_pred, gt_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            afc_indices = _afc_greedy_indices(
                pool_pred,
                batch,
                afc_bank,
                keep_k=int(args.keep_k),
                tau=float(args.afc_selection_tau),
            )
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_afc_greedy20_pred",
                prediction=_gather_candidates(pool_pred, afc_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            fps_indices = _structured_fps_indices(pool_pred[..., -1, :], keep_k=int(args.keep_k))
            _add_headroom_branch(
                accumulators,
                aux_accumulators,
                field_name=f"slow{pool_k}_endpoint_fps20_pred",
                prediction=_gather_candidates(pool_pred, fps_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=pool_latencies,
                afc_bank=afc_bank,
                spread_base=slow20,
            )
            if not bool(args.disable_random_pool_selection):
                mean_branch = _random_pool_mean_branch_name(int(pool_k), int(args.random_pool_trials))
                for trial_index in range(int(args.random_pool_trials)):
                    random_indices = _random_global_indices(
                        batch_size,
                        int(pool_k),
                        num_agents,
                        keep_k=int(args.keep_k),
                        device=pool_pred.device,
                    )
                    random_prediction = _gather_candidates(pool_pred, random_indices)
                    _add_headroom_branch(
                        accumulators,
                        aux_accumulators,
                        field_name=mean_branch,
                        prediction=random_prediction,
                        batch=batch,
                        miss_threshold=float(args.miss_threshold),
                        latencies_ms=pool_latencies,
                        afc_bank=afc_bank,
                        spread_base=slow20,
                    )
                    if bool(args.random_pool_emit_trials) and int(args.random_pool_trials) > 1:
                        _add_headroom_branch(
                            accumulators,
                            aux_accumulators,
                            field_name=f"slow{int(pool_k)}_random20_trial{trial_index}_pred",
                            prediction=random_prediction,
                            batch=batch,
                            miss_threshold=float(args.miss_threshold),
                            latencies_ms=pool_latencies,
                            afc_bank=afc_bank,
                            spread_base=slow20,
                        )

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[diagnose_sdd_headroom_analysis] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"records={min(chunk_index * int(args.batch_records), len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for branch, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[branch].finalize(branch))

    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_sdd_headroom_analysis",
            "variant": "sdd_exp1_headroom_afc_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "sdd",
            "split": str(args.split),
            "keep_k": int(args.keep_k),
            "slow_pool_ks": [int(item) for item in pool_ks],
            "oracle_select_metric": str(args.oracle_select_metric),
            "afc_selection_tau": float(args.afc_selection_tau),
            "afc_feature_variant": str(args.afc_feature_variant),
            "afc_bank_size": int(afc_bank.bank_size),
            "afc_bank_feature_dim": int(afc_bank.feature_dim),
            "afc_use_source_metadata": bool(afc_needs_metadata),
            "afc_source_id_field": str(args.afc_source_id_field),
            "afc_filter_same_source": bool(args.afc_filter_same_source),
            "afc_temporal_gap_frames": int(args.afc_temporal_gap_frames),
            "afc_randomize_bank_seed": args.afc_randomize_bank_seed,
            "benchmark_comparable": False,
            "diagnostic_normalization": _is_diagnostic_normalization_source(str(normalization_meta.get("source", ""))),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_records": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(args.slow_checkpoint).expanduser().resolve().as_posix(),
        "metrics": _coerce_jsonable(metrics),
    }
    _print_summary(metrics, branches)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
