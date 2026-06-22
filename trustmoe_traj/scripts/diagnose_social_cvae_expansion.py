"""V25 precheck: oracle diagnosis for SocialCVAE residual expansion.

The script evaluates whether a trained V24/V24-B refiner already places better
trajectories in its latent residual distribution.  It expands each slow-teacher
mode with R sampled residuals, then compares random sampling against two GT
oracles:

* group oracle: choose one residual sample per original teacher mode;
* pool oracle: choose K trajectories globally from the K * R candidate pool.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
from trustmoe_traj.scripts.eval_social_cvae_refiner import (
    _capture_rng,
    _checkpoint_variant,
    _energy_risk_mean,
    _local_temporal_energy,
    _restore_rng,
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


METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


@dataclass
class AuxAccumulator:
    """Weighted split-level averages for diagnostic metrics."""

    total_valid_agents: float = 0.0
    sums: Dict[str, float] = field(default_factory=dict)

    def add(self, values: Mapping[str, float], *, weight: int) -> None:
        if int(weight) <= 0:
            return
        self.total_valid_agents += float(weight)
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * float(weight)

    def finalize(self, prefix: str) -> Dict[str, float]:
        if self.total_valid_agents <= 0:
            return {}
        return {f"{prefix}_{key}": float(value / self.total_valid_agents) for key, value in self.sums.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose V25 SocialCVAE multi-sample residual expansion.")
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
    parser.add_argument("--residual-samples-list", type=str, default="3,5,10")
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--include-fast", action="store_true")
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--fast-checkpoint", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item]
    if not values:
        raise ValueError("--residual-samples-list must contain at least one positive integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"residual sample counts must be positive: {values}")
    return sorted(dict.fromkeys(values))


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


def _flatten_refined(refined: torch.Tensor) -> torch.Tensor:
    if refined.ndim != 6:
        raise ValueError(f"Expected [B,R,K,A,T,2], got {tuple(refined.shape)}")
    b, r, k, a, t, d = refined.shape
    return refined.reshape(b, r * k, a, t, d)


def _candidate_score(prediction: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    dist = torch.linalg.norm(prediction - ground_truth[:, None, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported oracle metric: {metric!r}")


def _group_oracle(refined: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    score = _candidate_score(refined, ground_truth, metric=metric)
    best_sample = score.argmin(dim=1)
    index = best_sample[:, None, :, :, None, None].expand(
        refined.shape[0],
        1,
        refined.shape[2],
        refined.shape[3],
        refined.shape[4],
        refined.shape[5],
    )
    return torch.gather(refined, dim=1, index=index).squeeze(1)


def _pool_oracle(
    refined: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = _flatten_refined(refined)
    dist = torch.linalg.norm(flat - ground_truth[:, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        score = fde
    elif metric == "ade_fde":
        score = dist.mean(dim=-1) + fde
    else:
        raise ValueError(f"Unsupported oracle metric: {metric!r}")
    num_modes = int(refined.shape[2])
    keep_k = min(num_modes, int(score.shape[1]))
    top_index = torch.topk(score, k=keep_k, dim=1, largest=False).indices
    gather_index = top_index[:, :, :, None, None].expand(
        flat.shape[0],
        keep_k,
        flat.shape[2],
        flat.shape[3],
        flat.shape[4],
    )
    return torch.gather(flat, dim=1, index=gather_index), top_index


def _offdiag_mean(pairwise: torch.Tensor) -> torch.Tensor:
    num_modes = int(pairwise.shape[-1])
    if num_modes <= 1:
        return pairwise.new_zeros((pairwise.shape[0],))
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[:, keep].mean(dim=-1)


def _endpoint_spread(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, _t, d = prediction.shape
    endpoints = prediction[..., -1, :].permute(0, 2, 1, 3).reshape(b * a, k, d)
    return _offdiag_mean(torch.cdist(endpoints, endpoints, p=2)).reshape(b, a)


def _trajectory_spread(prediction: torch.Tensor) -> torch.Tensor:
    b, k, a, t, d = prediction.shape
    traj = prediction.permute(0, 2, 1, 3, 4).reshape(b * a, k, t, d)
    pairwise = torch.linalg.norm(traj[:, :, None, :, :] - traj[:, None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean(pairwise).reshape(b, a)


def _unique_base_mode_ratio(selected_flat_indices: Optional[torch.Tensor], *, num_modes: int, mask: torch.Tensor) -> Optional[float]:
    if selected_flat_indices is None:
        return None
    valid = mask.detach().cpu().bool()
    modes = (selected_flat_indices.detach().cpu() % int(num_modes)).to(torch.long)
    values: List[float] = []
    for batch_index in range(int(valid.shape[0])):
        for agent_index in range(int(valid.shape[1])):
            if not bool(valid[batch_index, agent_index].item()):
                continue
            selected = modes[batch_index, :, agent_index].tolist()
            values.append(float(len(set(int(item) for item in selected)) / max(int(num_modes), 1)))
    if not values:
        return None
    return float(sum(values) / len(values))


def _branch_aux(
    prediction: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    *,
    selected_flat_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    valid = mask.to(device=prediction.device, dtype=torch.bool)
    if int(valid.sum().item()) <= 0:
        return {}
    delta_l2 = torch.linalg.norm(prediction - base, dim=-1).mean(dim=-1).mean(dim=1)
    endpoint_ratio = _endpoint_spread(prediction) / _endpoint_spread(base).abs().clamp_min(1e-8)
    trajectory_ratio = _trajectory_spread(prediction) / _trajectory_spread(base).abs().clamp_min(1e-8)
    result = {
        "delta_l2_mean": float(delta_l2[valid].mean().detach().cpu()),
        "endpoint_ratio": float(endpoint_ratio[valid].mean().detach().cpu()),
        "trajectory_ratio": float(trajectory_ratio[valid].mean().detach().cpu()),
    }
    unique_ratio = _unique_base_mode_ratio(selected_flat_indices, num_modes=int(base.shape[1]), mask=valid)
    if unique_ratio is not None:
        result["unique_base_mode_ratio"] = float(unique_ratio)
    return result


def _add_branch(
    accumulators: Mapping[str, BranchAccumulator],
    aux_accumulators: Mapping[str, AuxAccumulator],
    *,
    field_name: str,
    prediction: torch.Tensor,
    base: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    miss_threshold: float,
    latencies_ms: Iterable[float],
    selected_flat_indices: Optional[torch.Tensor] = None,
) -> None:
    summary = evaluate_model_output(
        {field_name: prediction},
        batch,
        miss_threshold=float(miss_threshold),
        prediction_fields=(field_name,),
    )
    accumulators[field_name].add_chunk(summary.metrics, latencies_ms)
    valid_count = int(batch["agent_mask"].bool().sum().item())
    aux_accumulators[field_name].add(
        _branch_aux(
            prediction,
            base,
            batch["agent_mask"].bool(),
            selected_flat_indices=selected_flat_indices,
        ),
        weight=valid_count,
    )


def _metric(metrics: Mapping[str, float], field: str, name: str) -> Optional[float]:
    value = metrics.get(f"{field}_{name}")
    return None if value is None else float(value)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "None"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _print_delta_summary(metrics: Mapping[str, float], branches: Sequence[str]) -> None:
    print("\n[diagnose_social_cvae_expansion] branch - slow deltas")
    for field_name in branches:
        if field_name == "slow_pred":
            continue
        print(f"\n-- {field_name} --")
        for name in METRICS:
            branch = _metric(metrics, field_name, name)
            slow = _metric(metrics, "slow_pred", name)
            delta = None if branch is None or slow is None else branch - slow
            print(f"d{name}: {_fmt(delta, signed=True)}  branch={_fmt(branch)}  slow={_fmt(slow)}")
        for aux_name in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(float(metrics[key]))}")


def main() -> None:
    args = build_parser().parse_args()
    if args.include_fast and (not args.fast_cfg_path or not args.fast_checkpoint):
        raise SystemExit("--fast-cfg-path and --fast-checkpoint are required with --include-fast")
    residual_counts = _split_ints(args.residual_samples_list)
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

    branches = ["slow_pred", "v25_z_mean_pred"]
    for sample_count in residual_counts:
        branches.extend(
            [
                f"v25_r{sample_count}_random_pred",
                f"v25_r{sample_count}_oracle_group_pred",
                f"v25_r{sample_count}_oracle_pool_pred",
            ]
        )
    if fast_predictor is not None:
        branches.append("fast_pred")

    accumulators = {
        field_name: BranchAccumulator(field_name, args.miss_threshold)
        for field_name in branches
    }
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[diagnose_social_cvae_expansion] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(args.refiner_checkpoint).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} residual_counts={residual_counts} oracle_metric={args.oracle_select_metric}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[diagnose_social_cvae_expansion] warning: selected_samples normalization is diagnostic only")

    energy_risk_sum = 0.0
    aux_weight = 0
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

        valid_count = int(batch["agent_mask"].bool().sum().item())
        if valid_count > 0:
            risk_mean, _risk_count = _energy_risk_mean(temporal_energy, batch["agent_mask"].bool())
            energy_risk_sum += float(risk_mean) * valid_count
            aux_weight += valid_count

        mean_latencies, mean_outputs = _measure_predict_latency_ms(
            lambda: refiner.refine(
                slow_output.slow_pred,
                past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                temporal_energy_features=temporal_energy.to(device=device),
                num_samples=1,
                z_mode="mean",
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        mean_pred = mean_outputs["refined"].squeeze(1)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name="v25_z_mean_pred",
            prediction=mean_pred,
            base=slow_output.slow_pred,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=mean_latencies,
        )

        for sample_count in residual_counts:
            sample_latencies, sample_outputs = _measure_predict_latency_ms(
                lambda: refiner.refine(
                    slow_output.slow_pred,
                    past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                    temporal_energy_features=temporal_energy.to(device=device),
                    num_samples=int(sample_count),
                    z_mode="sample",
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            refined = sample_outputs["refined"]
            random_pred = refined[:, 0]
            group_pred = _group_oracle(
                refined,
                batch["fut_traj_original_scale"].to(device=device),
                metric=str(args.oracle_select_metric),
            )
            pool_pred, pool_index = _pool_oracle(
                refined,
                batch["fut_traj_original_scale"].to(device=device),
                metric=str(args.oracle_select_metric),
            )
            for field_name, prediction, selected_index in (
                (f"v25_r{sample_count}_random_pred", random_pred, None),
                (f"v25_r{sample_count}_oracle_group_pred", group_pred, None),
                (f"v25_r{sample_count}_oracle_pool_pred", pool_pred, pool_index),
            ):
                _add_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=field_name,
                    prediction=prediction,
                    base=slow_output.slow_pred,
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=sample_latencies,
                    selected_flat_indices=selected_index,
                )

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
                "[diagnose_social_cvae_expansion] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[field_name].finalize(field_name))
    if aux_weight > 0:
        metrics["temporal_energy_risk_mean"] = float(energy_risk_sum / aux_weight)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_social_cvae_expansion",
            "variant": "v25_precheck_social_cvae_expansion_oracle",
            "refiner_variant": refiner_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_sample_counts": residual_counts,
            "oracle_select_metric": args.oracle_select_metric,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
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
    _print_delta_summary(metrics, branches)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
