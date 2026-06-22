"""V58-C fair-20 diagnostics for residual slot refiners.

This script evaluates one trained set-generator refiner checkpoint and derives
several fair K=20 branches from the same full residual pool:

* fixed slot20: slot0, slot1, ... each keeps all base modes for one slot;
* per-base oracle20: GT chooses the best slot for each base mode;
* global oracle20: GT chooses the best 20 candidates from the full pool;
* full pool: all slot x base candidates, for reference.

It also has a summary mode so the three diagnostic families can stay in one
code file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_group_selector, load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _add_branch,
    _all_indices,
    _base_for_flat,
    _flatten_refined,
    _gather_candidates,
    _oracle_indices,
    _predictor_cfg,
    _set_seed,
)
from trustmoe_traj.scripts.eval_social_cvae_refiner import (
    _checkpoint_variant,
    _energy_risk_mean,
    _local_temporal_energy,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
AUX_METRICS: Sequence[str] = (
    "latency_avg_ms",
    "delta_l2_mean",
    "endpoint_ratio",
    "trajectory_ratio",
    "unique_base_mode_ratio",
    "selected_slot_mean",
    "selected_slot0_ratio",
    "raw_selected_slot_mean",
    "raw_selected_slot0_ratio",
    "selector_fallback_to_slot0_ratio",
    "front_oracle_slot_accuracy",
    "selected_nonzero_ratio",
    "raw_selected_nonzero_ratio",
    "selected_slot1_ratio",
    "selected_slot2_ratio",
    "selected_slot3_ratio",
    "raw_selected_slot1_ratio",
    "raw_selected_slot2_ratio",
    "raw_selected_slot3_ratio",
    "front_oracle_nonzero_ratio",
    "front_slot0_good_vs_slow_ratio",
    "front_all_bad_vs_slow_ratio",
    "selector_mean_dade_vs_slot0",
    "selector_mean_dfde_vs_slot0",
    "selector_mean_dscore_vs_slot0",
    "selector_mean_dscore_vs_slow",
    "selector_raw_mean_dade_vs_slot0",
    "selector_raw_mean_dfde_vs_slot0",
    "selector_raw_mean_dscore_vs_slot0",
    "selector_selected_prob_mean",
    "selector_raw_prob_mean",
    "selector_slot0_prob_mean",
    "selector_raw_prob_margin_mean",
    "accepted_nonzero_better_slot0_ade_ratio",
    "accepted_nonzero_better_slot0_fde_ratio",
    "accepted_nonzero_hurt_slot0_ade_ratio",
    "accepted_nonzero_hurt_slot0_fde_ratio",
    "accepted_nonzero_improves_slow_score_ratio",
    "accepted_nonzero_hurts_slow_score_ratio",
    "accepted_nonzero_mean_dade_vs_slot0",
    "accepted_nonzero_mean_dfde_vs_slot0",
    "accepted_nonzero_mean_dscore_vs_slot0",
    "accepted_nonzero_prob_mean",
    "accepted_nonzero_prob_margin_mean",
    "raw_nonzero_hurt_slot0_ade_ratio",
    "raw_nonzero_hurt_slot0_fde_ratio",
    "fallback_raw_hurt_slot0_ade_ratio",
    "fallback_raw_hurt_slot0_fde_ratio",
    "fallback_raw_prob_mean",
    "fallback_raw_prob_margin_mean",
    "missed_oracle_nonzero_ratio",
    "oracle_nonzero_recall_ratio",
    "oracle_slot0_recall_ratio",
    "all_bad_fallback_to_slot0_ratio",
    "all_bad_nonzero_accept_ratio",
)
EPS = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate/summarize V58-C fair-20 residual slot diagnostics.")
    parser.add_argument("--summarize-only", action="store_true", help="Aggregate saved JSON files instead of evaluating.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, default=None)
    parser.add_argument("--eval-file-prefix", type=str, default="v58c_fair20")
    parser.add_argument("--diagnostic-prefix", type=str, default="v58c")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)

    parser.add_argument("--protocol", type=str, default="official_align", choices=EVAL_PROTOCOLS)
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sample-mode", type=str, default="per_agent", choices=["per_agent", "per_scene"])
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=None)
    parser.add_argument("--data-norm", type=str, default="min_max", choices=["min_max"])
    parser.add_argument("--normalization-source", type=str, default="auto", choices=NORMALIZATION_SOURCES)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, default=None)
    parser.add_argument("--refiner-checkpoint", type=str, default=None)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--front-slot-start", type=int, default=1)
    parser.add_argument("--front-slots", type=int, default=0)
    parser.add_argument("--selector-checkpoint", type=str, default=None)
    parser.add_argument(
        "--include-selector",
        action="store_true",
        help="Include the front-slot selector branch in evaluation summaries.",
    )
    parser.add_argument("--selector-branch-name", type=str, default=None)
    parser.add_argument(
        "--selector-confidence-fallback-to-slot0",
        action="store_true",
        help="Use slot0 unless the selector prefers slot1/2 with sufficient confidence.",
    )
    parser.add_argument("--selector-fallback-prob-margin", type=float, default=0.05)
    parser.add_argument("--selector-fallback-min-selected-prob", type=float, default=0.35)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in _split_items(raw)]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [item for item in (_num(value) for value in values) if item is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0.0 else ""
    return f"{prefix}{numeric:.6f}"


def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: Mapping[str, Any], field: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{field}_{name}"))


def _full_branch_name(prefix: str, residual_slots: int, keep_k: int) -> str:
    return f"{prefix}_full{int(residual_slots) * int(keep_k)}_pred"


def _front_oracle_branch_name(prefix: str, front_slot_start: int, front_slots: int) -> str:
    end_slot = int(front_slot_start) + int(front_slots) - 1
    if int(front_slot_start) == 1:
        return f"{prefix}_front{int(front_slots)}_oracle20_pred"
    return f"{prefix}_slots{int(front_slot_start)}to{end_slot}_oracle20_pred"


def _front_random_branch_name(prefix: str, front_slot_start: int, front_slots: int) -> str:
    end_slot = int(front_slot_start) + int(front_slots) - 1
    if int(front_slot_start) == 1:
        return f"{prefix}_front{int(front_slots)}_random20_pred"
    return f"{prefix}_slots{int(front_slot_start)}to{end_slot}_random20_pred"


def _front_roundrobin_branch_name(prefix: str, front_slot_start: int, front_slots: int) -> str:
    end_slot = int(front_slot_start) + int(front_slots) - 1
    if int(front_slot_start) == 1:
        return f"{prefix}_front{int(front_slots)}_roundrobin20_pred"
    return f"{prefix}_slots{int(front_slot_start)}to{end_slot}_roundrobin20_pred"


def _front_selector_branch_name(prefix: str, front_slot_start: int, front_slots: int) -> str:
    end_slot = int(front_slot_start) + int(front_slots) - 1
    if int(front_slot_start) == 1:
        return f"{prefix}_front{int(front_slots)}_selector20_pred"
    return f"{prefix}_slots{int(front_slot_start)}to{end_slot}_selector20_pred"


def _front_global_oracle_branch_name(prefix: str, front_slot_start: int, front_slots: int) -> str:
    end_slot = int(front_slot_start) + int(front_slots) - 1
    if int(front_slot_start) == 1:
        return f"{prefix}_front{int(front_slots)}_global_oracle20_pred"
    return f"{prefix}_slots{int(front_slot_start)}to{end_slot}_global_oracle20_pred"


def _branches(
    prefix: str,
    residual_slots: int,
    keep_k: int,
    *,
    front_slot_start: int = 1,
    front_slots: int = 0,
    include_selector: bool = False,
    selector_branch_name: Optional[str] = None,
) -> List[str]:
    fixed = [f"{prefix}_slot{slot}_20_pred" for slot in range(int(residual_slots))]
    front: List[str] = []
    if int(front_slots) > 0:
        front = [
            _front_random_branch_name(prefix, front_slot_start, front_slots),
            _front_roundrobin_branch_name(prefix, front_slot_start, front_slots),
        ]
        if bool(include_selector):
            front.append(selector_branch_name or _front_selector_branch_name(prefix, front_slot_start, front_slots))
        front.extend(
            [
                _front_oracle_branch_name(prefix, front_slot_start, front_slots),
                _front_global_oracle_branch_name(prefix, front_slot_start, front_slots),
            ]
        )
    return [
        *fixed,
        *front,
        f"{prefix}_per_base_oracle20_pred",
        f"{prefix}_global_oracle20_pred",
        _full_branch_name(prefix, residual_slots, keep_k),
    ]


def _fixed_slot_indices(
    *,
    batch_size: int,
    slot_index: int,
    num_base_modes: int,
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    modes = torch.arange(num_base_modes, device=device, dtype=torch.long)
    indices = int(slot_index) * int(num_base_modes) + modes
    return indices[None, :, None].expand(int(batch_size), int(num_base_modes), int(num_agents))


def _slot_flat_indices(slot_index: torch.Tensor, *, num_base_modes: int) -> torch.Tensor:
    modes = torch.arange(num_base_modes, device=slot_index.device, dtype=torch.long)[None, :, None].expand_as(slot_index)
    return slot_index.to(dtype=torch.long) * int(num_base_modes) + modes


def _candidate_score_slots(candidates: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    fde = dist[..., -1]
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return dist.mean(dim=-1) + fde
    raise ValueError(f"Unsupported oracle metric: {metric!r}")


def _candidate_ade_fde_slots(candidates: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.ndim != 6:
        raise ValueError(f"Expected candidates [B,S,K,A,T,2], got {tuple(candidates.shape)}")
    dist = torch.linalg.norm(candidates - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _base_ade_fde(base: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if base.ndim != 5:
        raise ValueError(f"Expected base [B,K,A,T,2], got {tuple(base.shape)}")
    dist = torch.linalg.norm(base - ground_truth[:, None, ...], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def _score_from_ade_fde(ade: torch.Tensor, fde: torch.Tensor, *, metric: str) -> torch.Tensor:
    if metric == "fde":
        return fde
    if metric == "ade_fde":
        return ade + fde
    raise ValueError(f"Unsupported oracle metric: {metric!r}")


def _gather_slot_values(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"Expected values [B,S,K,A], got {tuple(values.shape)}")
    if slots.ndim != 3:
        raise ValueError(f"Expected slots [B,K,A], got {tuple(slots.shape)}")
    index = slots.to(device=values.device, dtype=torch.long)[:, None, :, :]
    return torch.gather(values, dim=1, index=index).squeeze(1)


def _per_base_oracle_slots(candidates: torch.Tensor, ground_truth: torch.Tensor, *, metric: str) -> torch.Tensor:
    return _candidate_score_slots(candidates, ground_truth, metric=metric).argmin(dim=1)


def _valid_base_mask(agent_mask: torch.Tensor, num_base_modes: int) -> torch.Tensor:
    return agent_mask.bool()[:, None, :].expand(agent_mask.shape[0], int(num_base_modes), agent_mask.shape[1])


def _mean_on_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    keep = valid.to(device=values.device, dtype=torch.bool)
    if int(keep.sum().item()) <= 0:
        return 0.0
    return float(values[keep].to(dtype=torch.float32).mean().detach().cpu())


def _count_on_valid(mask: torch.Tensor, valid: torch.Tensor) -> int:
    keep = mask.to(dtype=torch.bool) & valid.to(device=mask.device, dtype=torch.bool)
    return int(keep.sum().detach().cpu().item())


def _mean_on_mask(values: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> Optional[float]:
    keep = mask.to(device=values.device, dtype=torch.bool) & valid.to(device=values.device, dtype=torch.bool)
    if int(keep.sum().item()) <= 0:
        return None
    return float(values[keep].to(dtype=torch.float32).mean().detach().cpu())


def _ratio_on_mask(event: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> Optional[float]:
    return _mean_on_mask(event.to(dtype=torch.float32), mask, valid)


def _selector_indices_with_optional_slot0_fallback(
    logits: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_selected = logits.argmax(dim=1).to(dtype=torch.long)
    if not bool(getattr(args, "selector_confidence_fallback_to_slot0", False)):
        fallback = torch.zeros_like(raw_selected, dtype=torch.bool)
        return raw_selected, raw_selected, fallback
    probs = torch.softmax(logits, dim=1)
    selected_prob = torch.gather(probs, dim=1, index=raw_selected[:, None, :, :]).squeeze(1)
    slot0_prob = probs[:, 0]
    accept = (
        (raw_selected != 0)
        & (selected_prob >= slot0_prob + float(args.selector_fallback_prob_margin))
        & (selected_prob >= float(args.selector_fallback_min_selected_prob))
    )
    selected = torch.where(accept, raw_selected, torch.zeros_like(raw_selected))
    fallback = (raw_selected != 0) & (selected == 0)
    return selected.to(dtype=torch.long), raw_selected, fallback


def _merge_aux_values(accumulator: AuxAccumulator, values: Mapping[str, float], *, weight: int) -> None:
    """Add extra aux keys using the existing denominator from _add_branch."""

    if int(weight) <= 0:
        return
    for key, value in values.items():
        accumulator.sums[key] = accumulator.sums.get(key, 0.0) + float(value) * float(weight)


def _merge_aux_metric(accumulator: AuxAccumulator, key: str, value: Optional[float], *, weight: int) -> None:
    if value is None or int(weight) <= 0:
        return
    add_metric = getattr(accumulator, "add_metric", None)
    if callable(add_metric):
        add_metric(key, value, weight=int(weight))
        return
    accumulator.sums[key] = accumulator.sums.get(key, 0.0) + float(value) * float(weight)


def _merge_conditional_ratio(
    accumulator: AuxAccumulator,
    key: str,
    event: torch.Tensor,
    denominator_mask: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    weight = _count_on_valid(denominator_mask, valid)
    _merge_aux_metric(accumulator, key, _ratio_on_mask(event, denominator_mask, valid), weight=weight)


def _merge_conditional_mean(
    accumulator: AuxAccumulator,
    key: str,
    values: torch.Tensor,
    denominator_mask: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    weight = _count_on_valid(denominator_mask, valid)
    _merge_aux_metric(accumulator, key, _mean_on_mask(values, denominator_mask, valid), weight=weight)


def _require_eval_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("slow_cfg_path", "slow_checkpoint", "refiner_checkpoint", "output_json")
        if not getattr(args, name)
    ]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(f"Missing required evaluation arguments: {joined}")
    if int(args.residual_slots) <= 1:
        raise SystemExit("--residual-slots must be > 1")
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.front_slots) < 0:
        raise SystemExit("--front-slots must be non-negative")
    if int(args.front_slots) > 0:
        if int(args.front_slot_start) < 0:
            raise SystemExit("--front-slot-start must be non-negative")
        if int(args.front_slot_start) + int(args.front_slots) > int(args.residual_slots):
            raise SystemExit("--front-slot-start + --front-slots must not exceed --residual-slots")
    include_selector = bool(args.include_selector or args.selector_checkpoint)
    if include_selector and int(args.front_slots) <= 0:
        raise SystemExit("--include-selector/--selector-checkpoint requires --front-slots > 0")
    if include_selector and not args.selector_checkpoint and not bool(args.summarize_only):
        raise SystemExit("--include-selector requires --selector-checkpoint during evaluation")
    if float(args.selector_fallback_prob_margin) < 0.0:
        raise SystemExit("--selector-fallback-prob-margin must be non-negative")
    if not (0.0 <= float(args.selector_fallback_min_selected_prob) <= 1.0):
        raise SystemExit("--selector-fallback-min-selected-prob must be in [0, 1]")
    if bool(args.selector_confidence_fallback_to_slot0) and int(args.front_slot_start) != 0:
        raise SystemExit("--selector-confidence-fallback-to-slot0 requires --front-slot-start 0")


def _print_eval_summary(metrics: Mapping[str, float], *, branches: Sequence[str]) -> None:
    print("\n[eval_v58c_fair20_residual_slots] branch - slow deltas")
    for field_name in branches:
        print(f"\n-- {field_name} --")
        for metric_name in METRICS:
            branch = _metric(metrics, field_name, metric_name)
            slow = _metric(metrics, "slow_pred", metric_name)
            delta = None if branch is None or slow is None else branch - slow
            print(f"d{metric_name}: {_fmt(delta, signed=True)}  branch={_fmt(branch)}  slow={_fmt(slow)}")
        for aux_name in AUX_METRICS:
            key = f"{field_name}_{aux_name}"
            if key in metrics:
                print(f"{aux_name}: {_fmt(metrics[key])}")
    prefix = ""
    for branch in branches:
        if branch.endswith("_full160_pred"):
            prefix = branch[: -len("_full160_pred")]
            break
    if not prefix:
        prefix = "v58c"
    for key in (f"{prefix}_pool_delta_l2_mean", f"{prefix}_dynamic_slot_offset_l2_mean", f"{prefix}_energy_risk_mean"):
        if key in metrics:
            print(f"{key}: {_fmt(metrics[key])}")


def evaluate(args: argparse.Namespace) -> None:
    _require_eval_args(args)
    protocol_settings = _resolve_protocol_settings(args)
    _validate_protocol_assumptions(args, protocol_settings)
    _set_seed(args.seed)

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
            cfg_path=str(args.slow_cfg_path),
            checkpoint_path=str(args.slow_checkpoint),
        )
    )
    refiner_variant = _checkpoint_variant(str(args.refiner_checkpoint))
    refiner = load_social_cvae_teacher_refiner(str(args.refiner_checkpoint), map_location=device).to(device)
    refiner.eval()
    selector = None
    selector_variant = None
    include_selector = bool(args.include_selector or args.selector_checkpoint)
    if include_selector:
        selector_variant = _checkpoint_variant(str(args.selector_checkpoint))
        selector = load_social_cvae_group_selector(str(args.selector_checkpoint), map_location=device).to(device)
        selector.eval()
    if not bool(getattr(refiner.config, "use_set_generator", False)):
        raise SystemExit("--refiner-checkpoint must be trained with use_set_generator=True")
    max_slots = int(getattr(refiner.config, "max_residual_slots", 1))
    if int(args.residual_slots) > max_slots:
        raise SystemExit(f"--residual-slots {args.residual_slots} exceeds checkpoint max_residual_slots={max_slots}")

    normalization_stats, normalization_meta = _resolve_normalization_stats(
        data_norm=args.data_norm,
        normalization_source=protocol_settings.normalization_source,
        predictors=[slow_predictor],
        samples=selected_samples,
        stats_owner=slow_predictor,
        data_root=data_root,
        subset=args.subset,
        protocol_settings=protocol_settings,
    )
    slow_predictor._set_normalization_stats(normalization_stats)

    prefix = str(args.diagnostic_prefix)
    branches = [
        "slow_pred",
        *_branches(
            prefix,
            int(args.residual_slots),
            int(args.keep_k),
            front_slot_start=int(args.front_slot_start),
            front_slots=int(args.front_slots),
            include_selector=include_selector,
            selector_branch_name=args.selector_branch_name,
        ),
    ]
    deterministic_branches = [branch for branch in branches if branch != "slow_pred"]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[eval_v58c_fair20_residual_slots] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} slots={args.residual_slots} keep_k={args.keep_k} "
        f"oracle_metric={args.oracle_select_metric} selector_variant={selector_variant}"
    )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_v58c_fair20_residual_slots] warning: selected_samples normalization is diagnostic only")

    aux_weight = 0
    pool_delta_l2_sum = 0.0
    dynamic_slot_offset_sum = 0.0
    dynamic_slot_offset_seen = False
    energy_risk_sum = 0.0
    front_random_generator = torch.Generator()
    front_random_generator.manual_seed(int(args.seed) + 1_000_003)

    selected_sample_pairs = list(enumerate(selected_samples))
    chunks = list(_iter_chunks(selected_sample_pairs, args.batch_scenes))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk = [sample for _scene_index, sample in chunk_pairs]
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
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
                num_samples=int(args.residual_slots),
                z_mode="slots",
            ),
            runs=int(args.latency_runs),
            device=device,
        )
        refined = refiner_outputs["refined"]
        flat = _flatten_refined(refined)
        base_flat = _base_for_flat(refined, slow_output.slow_pred)
        ground_truth = batch["fut_traj_original_scale"].to(device=device)
        batch_size, num_candidates, num_agents = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
        num_base_modes = int(slow_output.slow_pred.shape[1])
        if num_base_modes != int(args.keep_k):
            raise SystemExit(
                f"V58-C fair20 expects slow base modes == keep_k, got base_modes={num_base_modes} "
                f"keep_k={args.keep_k}"
            )

        for slot_index in range(int(args.residual_slots)):
            field_name = f"{prefix}_slot{slot_index}_20_pred"
            slot_indices = _fixed_slot_indices(
                batch_size=batch_size,
                slot_index=slot_index,
                num_base_modes=num_base_modes,
                num_agents=num_agents,
                device=flat.device,
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=refined[:, slot_index],
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=slot_indices,
                num_base_modes=num_base_modes,
            )
            valid_count = int(batch["agent_mask"].bool().sum().item())
            _merge_aux_values(
                aux_accumulators[field_name],
                {
                    "selected_slot_mean": float(slot_index),
                    "selected_slot0_ratio": 1.0 if int(slot_index) == 0 else 0.0,
                },
                weight=valid_count,
            )

        oracle_slots = _per_base_oracle_slots(refined, ground_truth, metric=str(args.oracle_select_metric))
        oracle_per_base_indices = _slot_flat_indices(oracle_slots, num_base_modes=num_base_modes)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=f"{prefix}_per_base_oracle20_pred",
            prediction=_gather_candidates(flat, oracle_per_base_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=oracle_per_base_indices,
            num_base_modes=num_base_modes,
        )
        valid_base = _valid_base_mask(batch["agent_mask"].to(device=device), num_base_modes)
        valid_count = int(batch["agent_mask"].bool().sum().item())
        _merge_aux_values(
            aux_accumulators[f"{prefix}_per_base_oracle20_pred"],
            {
                "selected_slot_mean": _mean_on_valid(oracle_slots, valid_base),
                "selected_slot0_ratio": _mean_on_valid((oracle_slots == 0).to(dtype=torch.float32), valid_base),
            },
            weight=valid_count,
        )

        global_oracle_indices = _oracle_indices(
            flat,
            ground_truth,
            keep_k=int(args.keep_k),
            metric=str(args.oracle_select_metric),
        )
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=f"{prefix}_global_oracle20_pred",
            prediction=_gather_candidates(flat, global_oracle_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=_gather_candidates(base_flat, global_oracle_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=global_oracle_indices,
            num_base_modes=num_base_modes,
        )
        global_slots = torch.div(global_oracle_indices, num_base_modes, rounding_mode="floor")
        valid_global = batch["agent_mask"].bool().to(device=device)[:, None, :].expand_as(global_slots)
        _merge_aux_values(
            aux_accumulators[f"{prefix}_global_oracle20_pred"],
            {
                "selected_slot_mean": _mean_on_valid(global_slots, valid_global),
                "selected_slot0_ratio": _mean_on_valid((global_slots == 0).to(dtype=torch.float32), valid_global),
            },
            weight=valid_count,
        )

        if int(args.front_slots) > 0:
            front_start = int(args.front_slot_start)
            front_end = front_start + int(args.front_slots)
            front_candidates = refined[:, front_start:front_end]

            random_local_slots = torch.randint(
                low=0,
                high=int(args.front_slots),
                size=(batch_size, num_base_modes, num_agents),
                generator=front_random_generator,
                device=torch.device("cpu"),
            ).to(device=flat.device)
            random_actual_slots = random_local_slots + int(front_start)
            random_indices = _slot_flat_indices(random_actual_slots, num_base_modes=num_base_modes)
            random_branch = _front_random_branch_name(prefix, int(args.front_slot_start), int(args.front_slots))
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=random_branch,
                prediction=_gather_candidates(flat, random_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=random_indices,
                num_base_modes=num_base_modes,
            )
            _merge_aux_values(
                aux_accumulators[random_branch],
                {
                    "selected_slot_mean": _mean_on_valid(random_actual_slots, valid_base),
                    "selected_slot0_ratio": _mean_on_valid(
                        (random_actual_slots == 0).to(dtype=torch.float32),
                        valid_base,
                    ),
                },
                weight=valid_count,
            )

            roundrobin_slots = (
                torch.arange(num_base_modes, device=flat.device, dtype=torch.long)[None, :, None]
                .expand(batch_size, num_base_modes, num_agents)
                .remainder(int(args.front_slots))
                + int(front_start)
            )
            roundrobin_indices = _slot_flat_indices(roundrobin_slots, num_base_modes=num_base_modes)
            roundrobin_branch = _front_roundrobin_branch_name(
                prefix,
                int(args.front_slot_start),
                int(args.front_slots),
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=roundrobin_branch,
                prediction=_gather_candidates(flat, roundrobin_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=roundrobin_indices,
                num_base_modes=num_base_modes,
            )
            _merge_aux_values(
                aux_accumulators[roundrobin_branch],
                {
                    "selected_slot_mean": _mean_on_valid(roundrobin_slots, valid_base),
                    "selected_slot0_ratio": _mean_on_valid(
                        (roundrobin_slots == 0).to(dtype=torch.float32),
                        valid_base,
                    ),
                },
                weight=valid_count,
            )

            front_oracle_slots = _per_base_oracle_slots(
                front_candidates,
                ground_truth,
                metric=str(args.oracle_select_metric),
            )
            front_actual_slots = front_oracle_slots + int(front_start)
            front_oracle_indices = _slot_flat_indices(front_actual_slots, num_base_modes=num_base_modes)

            if selector is not None:
                selector_latencies, selector_outputs = _measure_predict_latency_ms(
                    lambda: selector.select(
                        front_candidates,
                        base_trajectory=slow_output.slow_pred,
                        past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                        temporal_energy_features=temporal_energy.to(device=device),
                    ),
                    runs=int(args.latency_runs),
                    device=device,
                )
                selector_local_slots, selector_raw_local_slots, selector_fallback_mask = (
                    _selector_indices_with_optional_slot0_fallback(selector_outputs["logits"], args=args)
                )
                selector_actual_slots = selector_local_slots + int(front_start)
                selector_raw_actual_slots = selector_raw_local_slots + int(front_start)
                selector_indices = _slot_flat_indices(selector_actual_slots, num_base_modes=num_base_modes)
                selector_branch = args.selector_branch_name or _front_selector_branch_name(
                    prefix,
                    int(args.front_slot_start),
                    int(args.front_slots),
                )
                selector_branch_latencies = [
                    float(refiner_ms) + float(selector_ms)
                    for refiner_ms, selector_ms in zip(refiner_latencies, selector_latencies)
                ]
                _add_branch(
                    accumulators,
                    aux_accumulators,
                    field_name=selector_branch,
                    prediction=_gather_candidates(flat, selector_indices),
                    batch=batch,
                    miss_threshold=float(args.miss_threshold),
                    latencies_ms=selector_branch_latencies,
                    base_for_delta=slow_output.slow_pred,
                    spread_base=slow_output.slow_pred,
                    selected_flat_indices=selector_indices,
                    num_base_modes=num_base_modes,
                )
                selector_aux_values: Dict[str, float] = {
                    "selected_slot_mean": _mean_on_valid(selector_actual_slots, valid_base),
                    "selected_slot0_ratio": _mean_on_valid(
                        (selector_actual_slots == 0).to(dtype=torch.float32),
                        valid_base,
                    ),
                    "raw_selected_slot_mean": _mean_on_valid(selector_raw_actual_slots, valid_base),
                    "raw_selected_slot0_ratio": _mean_on_valid(
                        (selector_raw_actual_slots == 0).to(dtype=torch.float32),
                        valid_base,
                    ),
                    "selector_fallback_to_slot0_ratio": _mean_on_valid(
                        selector_fallback_mask.to(dtype=torch.float32),
                        valid_base,
                    ),
                    "front_oracle_slot_accuracy": _mean_on_valid(
                        (selector_local_slots == front_oracle_slots).to(dtype=torch.float32),
                        valid_base,
                    ),
                }
                for actual_slot in range(1, 4):
                    if front_start <= actual_slot < front_end:
                        selector_aux_values[f"selected_slot{actual_slot}_ratio"] = _mean_on_valid(
                            (selector_actual_slots == actual_slot).to(dtype=torch.float32),
                            valid_base,
                        )
                        selector_aux_values[f"raw_selected_slot{actual_slot}_ratio"] = _mean_on_valid(
                            (selector_raw_actual_slots == actual_slot).to(dtype=torch.float32),
                            valid_base,
                        )

                if front_start == 0:
                    probs = torch.softmax(selector_outputs["logits"], dim=1)
                    selector_selected_prob = _gather_slot_values(probs, selector_local_slots)
                    selector_raw_prob = _gather_slot_values(probs, selector_raw_local_slots)
                    selector_slot0_prob = probs[:, 0]
                    selector_raw_prob_margin = selector_raw_prob - selector_slot0_prob

                    front_ade, front_fde = _candidate_ade_fde_slots(front_candidates, ground_truth)
                    base_ade, base_fde = _base_ade_fde(slow_output.slow_pred, ground_truth)
                    front_score = _score_from_ade_fde(front_ade, front_fde, metric=str(args.oracle_select_metric))
                    base_score = _score_from_ade_fde(base_ade, base_fde, metric=str(args.oracle_select_metric))

                    slot0_ade = front_ade[:, 0]
                    slot0_fde = front_fde[:, 0]
                    slot0_score = front_score[:, 0]
                    selected_ade = _gather_slot_values(front_ade, selector_local_slots)
                    selected_fde = _gather_slot_values(front_fde, selector_local_slots)
                    selected_score = _gather_slot_values(front_score, selector_local_slots)
                    raw_ade = _gather_slot_values(front_ade, selector_raw_local_slots)
                    raw_fde = _gather_slot_values(front_fde, selector_raw_local_slots)
                    raw_score = _gather_slot_values(front_score, selector_raw_local_slots)

                    selected_nonzero = selector_local_slots != 0
                    raw_nonzero = selector_raw_local_slots != 0
                    oracle_nonzero = front_oracle_slots != 0
                    oracle_slot0 = front_oracle_slots == 0
                    front_best_score = front_score.min(dim=1).values
                    front_all_bad_vs_slow = front_best_score >= (base_score - EPS)
                    slot0_good_vs_slow = slot0_score < (base_score - EPS)

                    selector_aux_values.update(
                        {
                            "selected_nonzero_ratio": _mean_on_valid(
                                selected_nonzero.to(dtype=torch.float32),
                                valid_base,
                            ),
                            "raw_selected_nonzero_ratio": _mean_on_valid(
                                raw_nonzero.to(dtype=torch.float32),
                                valid_base,
                            ),
                            "front_oracle_nonzero_ratio": _mean_on_valid(
                                oracle_nonzero.to(dtype=torch.float32),
                                valid_base,
                            ),
                            "front_slot0_good_vs_slow_ratio": _mean_on_valid(
                                slot0_good_vs_slow.to(dtype=torch.float32),
                                valid_base,
                            ),
                            "front_all_bad_vs_slow_ratio": _mean_on_valid(
                                front_all_bad_vs_slow.to(dtype=torch.float32),
                                valid_base,
                            ),
                            "selector_mean_dade_vs_slot0": _mean_on_valid(selected_ade - slot0_ade, valid_base),
                            "selector_mean_dfde_vs_slot0": _mean_on_valid(selected_fde - slot0_fde, valid_base),
                            "selector_mean_dscore_vs_slot0": _mean_on_valid(
                                selected_score - slot0_score,
                                valid_base,
                            ),
                            "selector_mean_dscore_vs_slow": _mean_on_valid(
                                selected_score - base_score,
                                valid_base,
                            ),
                            "selector_raw_mean_dade_vs_slot0": _mean_on_valid(raw_ade - slot0_ade, valid_base),
                            "selector_raw_mean_dfde_vs_slot0": _mean_on_valid(raw_fde - slot0_fde, valid_base),
                            "selector_raw_mean_dscore_vs_slot0": _mean_on_valid(
                                raw_score - slot0_score,
                                valid_base,
                            ),
                            "selector_selected_prob_mean": _mean_on_valid(selector_selected_prob, valid_base),
                            "selector_raw_prob_mean": _mean_on_valid(selector_raw_prob, valid_base),
                            "selector_slot0_prob_mean": _mean_on_valid(selector_slot0_prob, valid_base),
                            "selector_raw_prob_margin_mean": _mean_on_valid(
                                selector_raw_prob_margin,
                                valid_base,
                            ),
                        }
                    )

                _merge_aux_values(
                    aux_accumulators[selector_branch],
                    selector_aux_values,
                    weight=valid_count,
                )
                if front_start == 0:
                    selector_aux = aux_accumulators[selector_branch]
                    accepted_nonzero = selector_local_slots != 0
                    fallback_nonzero = selector_fallback_mask.to(dtype=torch.bool)
                    selected_prob_margin = selector_selected_prob - selector_slot0_prob

                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_better_slot0_ade_ratio",
                        selected_ade < (slot0_ade - EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_better_slot0_fde_ratio",
                        selected_fde < (slot0_fde - EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_hurt_slot0_ade_ratio",
                        selected_ade > (slot0_ade + EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_hurt_slot0_fde_ratio",
                        selected_fde > (slot0_fde + EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_improves_slow_score_ratio",
                        selected_score < (base_score - EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "accepted_nonzero_hurts_slow_score_ratio",
                        selected_score > (base_score + EPS),
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "accepted_nonzero_mean_dade_vs_slot0",
                        selected_ade - slot0_ade,
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "accepted_nonzero_mean_dfde_vs_slot0",
                        selected_fde - slot0_fde,
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "accepted_nonzero_mean_dscore_vs_slot0",
                        selected_score - slot0_score,
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "accepted_nonzero_prob_mean",
                        selector_selected_prob,
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "accepted_nonzero_prob_margin_mean",
                        selected_prob_margin,
                        accepted_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "raw_nonzero_hurt_slot0_ade_ratio",
                        raw_ade > (slot0_ade + EPS),
                        raw_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "raw_nonzero_hurt_slot0_fde_ratio",
                        raw_fde > (slot0_fde + EPS),
                        raw_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "fallback_raw_hurt_slot0_ade_ratio",
                        raw_ade > (slot0_ade + EPS),
                        fallback_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "fallback_raw_hurt_slot0_fde_ratio",
                        raw_fde > (slot0_fde + EPS),
                        fallback_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "fallback_raw_prob_mean",
                        selector_raw_prob,
                        fallback_nonzero,
                        valid_base,
                    )
                    _merge_conditional_mean(
                        selector_aux,
                        "fallback_raw_prob_margin_mean",
                        selector_raw_prob_margin,
                        fallback_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "missed_oracle_nonzero_ratio",
                        selector_local_slots == 0,
                        oracle_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "oracle_nonzero_recall_ratio",
                        selector_local_slots == front_oracle_slots,
                        oracle_nonzero,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "oracle_slot0_recall_ratio",
                        selector_local_slots == 0,
                        oracle_slot0,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "all_bad_fallback_to_slot0_ratio",
                        selector_local_slots == 0,
                        front_all_bad_vs_slow,
                        valid_base,
                    )
                    _merge_conditional_ratio(
                        selector_aux,
                        "all_bad_nonzero_accept_ratio",
                        selector_local_slots != 0,
                        front_all_bad_vs_slow,
                        valid_base,
                    )

            front_oracle_branch = _front_oracle_branch_name(prefix, int(args.front_slot_start), int(args.front_slots))
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=front_oracle_branch,
                prediction=_gather_candidates(flat, front_oracle_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=front_oracle_indices,
                num_base_modes=num_base_modes,
            )
            _merge_aux_values(
                aux_accumulators[front_oracle_branch],
                {
                    "selected_slot_mean": _mean_on_valid(front_actual_slots, valid_base),
                    "selected_slot0_ratio": _mean_on_valid(
                        (front_actual_slots == 0).to(dtype=torch.float32),
                        valid_base,
                    ),
                },
                weight=valid_count,
            )

            front_flat_indices = []
            for slot_index in range(front_start, front_end):
                front_flat_indices.append(
                    _fixed_slot_indices(
                        batch_size=batch_size,
                        slot_index=slot_index,
                        num_base_modes=num_base_modes,
                        num_agents=num_agents,
                        device=flat.device,
                    )
                )
            front_pool_indices = torch.cat(front_flat_indices, dim=1)
            front_pool = _gather_candidates(flat, front_pool_indices)
            front_global_local = _oracle_indices(
                front_pool,
                ground_truth,
                keep_k=int(args.keep_k),
                metric=str(args.oracle_select_metric),
            )
            front_global_indices = torch.gather(front_pool_indices, dim=1, index=front_global_local)
            front_global_branch = _front_global_oracle_branch_name(
                prefix,
                int(args.front_slot_start),
                int(args.front_slots),
            )
            _add_branch(
                accumulators,
                aux_accumulators,
                field_name=front_global_branch,
                prediction=_gather_candidates(flat, front_global_indices),
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                base_for_delta=_gather_candidates(base_flat, front_global_indices),
                spread_base=slow_output.slow_pred,
                selected_flat_indices=front_global_indices,
                num_base_modes=num_base_modes,
            )
            front_global_slots = torch.div(front_global_indices, num_base_modes, rounding_mode="floor")
            valid_front_global = batch["agent_mask"].bool().to(device=device)[:, None, :].expand_as(front_global_slots)
            _merge_aux_values(
                aux_accumulators[front_global_branch],
                {
                    "selected_slot_mean": _mean_on_valid(front_global_slots, valid_front_global),
                    "selected_slot0_ratio": _mean_on_valid(
                        (front_global_slots == 0).to(dtype=torch.float32),
                        valid_front_global,
                    ),
                },
                weight=valid_count,
            )

        full_branch = _full_branch_name(prefix, int(args.residual_slots), int(args.keep_k))
        full_indices = _all_indices(batch_size, num_candidates, num_agents, device=flat.device)
        _add_branch(
            accumulators,
            aux_accumulators,
            field_name=full_branch,
            prediction=flat,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            base_for_delta=base_flat,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=full_indices,
            num_base_modes=num_base_modes,
        )

        if valid_count > 0:
            pool_delta_l2 = torch.linalg.norm(refiner_outputs["delta"], dim=-1).mean(dim=-1).mean().detach().cpu()
            pool_delta_l2_sum += float(pool_delta_l2) * valid_count
            dynamic_slot_offset = refiner_outputs.get("dynamic_slot_offset")
            if torch.is_tensor(dynamic_slot_offset):
                dynamic_slot_offset_seen = True
                offset_l2 = torch.linalg.norm(dynamic_slot_offset, dim=-1).mean().detach().cpu()
                dynamic_slot_offset_sum += float(offset_l2) * valid_count
            risk_mean, _risk_count = _energy_risk_mean(temporal_energy, batch["agent_mask"].bool())
            energy_risk_sum += float(risk_mean) * valid_count
            aux_weight += valid_count

        should_log = chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0
        if should_log:
            print(
                "[eval_v58c_fair20_residual_slots] "
                f"processed_chunks={chunk_index}/{len(chunks)} "
                f"raw_scenes={min(chunk_index * args.batch_scenes, len(selected_samples))}/{len(selected_samples)}"
            )

    metrics: Dict[str, float] = {}
    for field_name, accumulator in accumulators.items():
        metrics.update(accumulator.finalize())
        metrics.update(aux_accumulators[field_name].finalize(field_name))
    if aux_weight > 0:
        metrics[f"{prefix}_pool_delta_l2_mean"] = float(pool_delta_l2_sum / aux_weight)
        if dynamic_slot_offset_seen:
            metrics[f"{prefix}_dynamic_slot_offset_l2_mean"] = float(dynamic_slot_offset_sum / aux_weight)
        metrics[f"{prefix}_energy_risk_mean"] = float(energy_risk_sum / aux_weight)

    benchmark_comparable = _is_benchmark_comparable_run(
        protocol_settings=protocol_settings,
        sample_mode=args.sample_mode,
        agents=agents,
    )
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_v58c_fair20_residual_slots",
            "variant": "v58c_fair20_residual_slot_diagnostics",
            "diagnostic_prefix": prefix,
            "refiner_variant": refiner_variant,
            "selector_variant": selector_variant,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "front_slot_start": int(args.front_slot_start),
            "front_slots": int(args.front_slots),
            "include_selector": bool(include_selector),
            "selector_branch_name": args.selector_branch_name,
            "selector_confidence_fallback_to_slot0": bool(args.selector_confidence_fallback_to_slot0),
            "selector_fallback_prob_margin": float(args.selector_fallback_prob_margin),
            "selector_fallback_min_selected_prob": float(args.selector_fallback_min_selected_prob),
            "full_pool_candidates": int(args.residual_slots) * int(args.keep_k),
            "oracle_select_metric": args.oracle_select_metric,
            "benchmark_comparable": benchmark_comparable,
            "diagnostic_normalization": _is_diagnostic_normalization_source(protocol_settings.normalization_source),
        },
        "args": _coerce_jsonable(vars(args)),
        "branches": list(branches),
        "deterministic_branches": list(deterministic_branches),
        "dataset": {
            **_coerce_jsonable(dataset.summary()),
            "data_root": data_root.as_posix(),
            "num_selected_scenes": len(selected_samples),
            "num_selected_eval_items": int(selected_eval_items),
        },
        "normalization_stats": _coerce_jsonable(normalization_stats),
        "normalization_meta": _coerce_jsonable(normalization_meta),
        "slow_checkpoint": Path(str(args.slow_checkpoint)).expanduser().resolve().as_posix(),
        "refiner_checkpoint": Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix(),
        "selector_checkpoint": (
            Path(str(args.selector_checkpoint)).expanduser().resolve().as_posix()
            if args.selector_checkpoint
            else None
        ),
        "metrics": _coerce_jsonable(metrics),
    }
    _print_eval_summary(metrics, branches=deterministic_branches)
    output_path = Path(str(args.output_json)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


def _run_id(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{seed}"


def _seed_row(
    args: argparse.Namespace,
    project_root: Path,
    seed: int,
    splits: Sequence[str],
    branches: Sequence[str],
) -> Dict[str, Any]:
    if not args.run_prefix:
        raise SystemExit("--run-prefix is required with --summarize-only")
    run_id = _run_id(str(args.run_prefix), int(seed))
    row: Dict[str, Any] = {"run_id": run_id, "seed": int(seed)}
    missing: List[str] = []
    for split in splits:
        eval_path = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "eval_results"
            / run_id
            / f"{args.eval_file_prefix}_{split}.json"
        )
        payload = _load_json(eval_path)
        if payload is None:
            missing.append(eval_path.as_posix())
            metrics: Mapping[str, Any] = {}
        else:
            raw_metrics = payload.get("metrics", {})
            metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        split_row: Dict[str, Any] = {}
        for branch in branches:
            branch_row: Dict[str, Any] = {}
            for metric in METRICS:
                value = _metric(metrics, branch, metric)
                slow = _metric(metrics, "slow_pred", metric)
                branch_row[metric] = value
                branch_row[f"d{metric}"] = None if value is None or slow is None else value - slow
            for aux_name in AUX_METRICS:
                branch_row[aux_name] = _num(metrics.get(f"{branch}_{aux_name}"))
            split_row[branch] = branch_row
        split_row["slow"] = {metric: _metric(metrics, "slow_pred", metric) for metric in METRICS}
        split_row["slow"]["latency_avg_ms"] = _num(metrics.get("slow_pred_latency_avg_ms"))
        prefix = str(args.diagnostic_prefix)
        split_row["pool_delta_l2_mean"] = _num(metrics.get(f"{prefix}_pool_delta_l2_mean"))
        split_row["dynamic_slot_offset_l2_mean"] = _num(metrics.get(f"{prefix}_dynamic_slot_offset_l2_mean"))
        split_row["energy_risk_mean"] = _num(metrics.get(f"{prefix}_energy_risk_mean"))
        row[f"official_{split}"] = split_row
    row["missing_files"] = missing
    return row


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    branches: Sequence[str],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        split_key = f"official_{split}"
        split_agg: Dict[str, Any] = {}
        for branch in branches:
            branch_agg: Dict[str, Any] = {
                "available_official_seeds": sum(
                    1 for row in rows if _num(row.get(split_key, {}).get(branch, {}).get("dFDE_min")) is not None
                )
            }
            for metric in METRICS:
                branch_agg[f"mean_d{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(f"d{metric}") for row in rows
                )
                branch_agg[f"mean_{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(metric) for row in rows
                )
            for aux_name in AUX_METRICS:
                branch_agg[f"mean_{aux_name}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(aux_name) for row in rows
                )
            split_agg[branch] = branch_agg
        split_agg["global_aux"] = {
            "mean_pool_delta_l2_mean": _mean(row.get(split_key, {}).get("pool_delta_l2_mean") for row in rows),
            "mean_dynamic_slot_offset_l2_mean": _mean(
                row.get(split_key, {}).get("dynamic_slot_offset_l2_mean") for row in rows
            ),
            "mean_energy_risk_mean": _mean(row.get(split_key, {}).get("energy_risk_mean") for row in rows),
        }
        aggregate[split] = split_agg
    return aggregate


def _best_fixed_slot(
    aggregate_split: Mapping[str, Any],
    residual_slots: int,
    metric_key: str,
    *,
    prefix: str,
) -> Optional[str]:
    candidates: List[tuple[float, str]] = []
    for slot_index in range(int(residual_slots)):
        branch = f"{prefix}_slot{slot_index}_20_pred"
        value = _num(aggregate_split.get(branch, {}).get(metric_key))
        if value is not None:
            candidates.append((float(value), branch))
    if not candidates:
        return None
    value, branch = min(candidates, key=lambda item: item[0])
    return f"{branch} {metric_key}={_fmt(value, signed=metric_key.startswith('mean_d'))}"


def _render(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    splits: Sequence[str],
    branches: Sequence[str],
    residual_slots: int,
    prefix: str,
) -> str:
    lines: List[str] = []
    for row in rows:
        lines.append(f"===== {row['run_id']} =====")
        if row.get("missing_files"):
            lines.append("missing summary inputs:")
            for path in row["missing_files"]:
                lines.append(f"  {path}")
        for split in splits:
            official = row.get(f"official_{split}", {})
            lines.append("")
            lines.append(f"-- official {split} {prefix.upper()} fair20 diagnostics - Slow --")
            for branch in branches:
                branch_row = official.get(branch, {})
                lines.append(f"{branch}:")
                for metric in METRICS:
                    lines.append(
                        f"  d{metric}: {_fmt(branch_row.get(f'd{metric}'), signed=True)}  "
                        f"value={_fmt(branch_row.get(metric))}  "
                        f"slow={_fmt(official.get('slow', {}).get(metric))}"
                    )
                for aux_name in AUX_METRICS:
                    value = branch_row.get(aux_name)
                    if value is not None:
                        lines.append(f"  {aux_name}: {_fmt(value)}")
            lines.append(f"pool_delta_l2_mean: {_fmt(official.get('pool_delta_l2_mean'))}")
            lines.append(f"dynamic_slot_offset_l2_mean: {_fmt(official.get('dynamic_slot_offset_l2_mean'))}")
            lines.append(f"energy_risk_mean: {_fmt(official.get('energy_risk_mean'))}")

    lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        split_agg = aggregate.get(split, {})
        lines.append("")
        lines.append(f"-- {split} --")
        best_min = _best_fixed_slot(split_agg, residual_slots, "mean_dFDE_min", prefix=prefix)
        best_avg = _best_fixed_slot(split_agg, residual_slots, "mean_dFDE_avg", prefix=prefix)
        if best_min is not None:
            lines.append(f"best_fixed_slot_by_dFDE_min: {best_min}")
        if best_avg is not None:
            lines.append(f"best_fixed_slot_by_dFDE_avg: {best_avg}")
        for branch in branches:
            mean = split_agg.get(branch, {})
            lines.append(f"{branch}: available={int(mean.get('available_official_seeds') or 0)}/{len(rows)}")
            for metric in METRICS:
                lines.append(
                    f"  mean d{metric}: {_fmt(mean.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(mean.get(f'mean_{metric}'))}"
                )
            for aux_name in AUX_METRICS:
                value = mean.get(f"mean_{aux_name}")
                if value is not None:
                    lines.append(f"  mean {aux_name}: {_fmt(value)}")
        global_aux = split_agg.get("global_aux", {})
        lines.append(f"mean pool_delta_l2_mean: {_fmt(global_aux.get('mean_pool_delta_l2_mean'))}")
        lines.append(
            "mean dynamic_slot_offset_l2_mean: "
            f"{_fmt(global_aux.get('mean_dynamic_slot_offset_l2_mean'))}"
        )
        lines.append(f"mean energy_risk_mean: {_fmt(global_aux.get('mean_energy_risk_mean'))}")
    return "\n".join(lines).rstrip() + "\n"


def summarize(args: argparse.Namespace) -> None:
    if not args.run_prefix:
        raise SystemExit("--run-prefix is required with --summarize-only")
    include_selector = bool(args.include_selector or args.selector_checkpoint)
    if include_selector and int(args.front_slots) <= 0:
        raise SystemExit("--include-selector/--selector-checkpoint requires --front-slots > 0")
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    prefix = str(args.diagnostic_prefix)
    branches = _branches(
        prefix,
        int(args.residual_slots),
        int(args.keep_k),
        front_slot_start=int(args.front_slot_start),
        front_slots=int(args.front_slots),
        include_selector=include_selector,
        selector_branch_name=args.selector_branch_name,
    )
    rows = [_seed_row(args, project_root, seed, splits, branches) for seed in seeds]
    aggregate = _aggregate(rows, splits, branches)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.eval_v58c_fair20_residual_slots",
            "summary_mode": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": args.run_prefix,
            "eval_file_prefix": args.eval_file_prefix,
            "seeds": seeds,
            "splits": splits,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "diagnostic_prefix": prefix,
            "front_slot_start": int(args.front_slot_start),
            "front_slots": int(args.front_slots),
            "include_selector": bool(include_selector),
            "selector_branch_name": args.selector_branch_name,
            "selector_confidence_fallback_to_slot0": bool(args.selector_confidence_fallback_to_slot0),
            "selector_fallback_prob_margin": float(args.selector_fallback_prob_margin),
            "selector_fallback_min_selected_prob": float(args.selector_fallback_min_selected_prob),
            "branches": list(branches),
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    default_root = project_root / "trustmoe_traj" / "analysis" / "experiment_runs" / str(args.run_prefix)
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else default_root / f"{args.eval_file_prefix}_summary.json"
    )
    output_txt = (
        Path(args.output_txt).expanduser().resolve()
        if args.output_txt
        else default_root / f"{args.eval_file_prefix}_summary.txt"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered = _render(rows, aggregate, splits, branches, int(args.residual_slots), prefix)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.summarize_only):
        summarize(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
