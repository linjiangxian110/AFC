"""Evaluate a V58-K residual slot quality scorer on official fair-K=20 branches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.models import MoFlowSlowPredictor, load_social_cvae_teacher_refiner, load_v58_slot_quality_scorer
from trustmoe_traj.scripts.analogical_future_coverage import (
    AnalogicalFutureBank,
    build_eth_analogical_future_bank,
    split_float_list,
)
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
from trustmoe_traj.scripts.eval_v58c_fair20_residual_slots import (
    EPS,
    METRICS,
    _base_ade_fde,
    _candidate_ade_fde_slots,
    _candidate_score_slots,
    _count_on_valid,
    _fixed_slot_indices,
    _full_branch_name,
    _gather_slot_values,
    _mean_on_mask,
    _mean_on_valid,
    _merge_aux_metric,
    _merge_aux_values,
    _merge_conditional_mean,
    _merge_conditional_ratio,
    _per_base_oracle_slots,
    _print_eval_summary,
    _ratio_on_mask,
    _score_from_ade_fde,
    _slot_flat_indices,
    _valid_base_mask,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V58-K residual slot quality scorer.")
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

    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--quality-checkpoint", type=str, required=True)
    parser.add_argument("--residual-slots", type=int, default=8)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--candidate-slots", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--diagnostic-prefix", type=str, default="v58k")
    parser.add_argument("--branch-name", type=str, default=None)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="auto",
        choices=["auto", "quality_threshold", "two_stage_replacement"],
        help="auto uses the checkpoint training_mode when available.",
    )
    parser.add_argument("--accept-prob-threshold", type=float, default=0.5)
    parser.add_argument("--oracle-select-metric", type=str, default="fde", choices=["fde", "ade_fde"])
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--enable-afc", action="store_true", help="Compute non-learning Analogical Future Coverage metrics.")
    parser.add_argument("--afc-train-split", type=str, default="train")
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.5,1.0")
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--enable-anchor-qd", action="store_true", help="Add anchor-preserving AFC-aware conservative branch.")
    parser.add_argument(
        "--anchor-qd-selection-mode",
        type=str,
        default="per_base",
        choices=["per_base", "set_coverage", "set_coverage_floor", "role_transport"],
        help=(
            "per_base keeps the V59A local accept rule; set_coverage adds greedy AFC-mode coverage for diversity modes; "
            "set_coverage_floor also hard-preserves anchor modes and rejects corrections that shrink spread below the teacher floor; "
            "role_transport preserves one output per base mode and assigns non-anchor modes to AFC future roles."
        ),
    )
    parser.add_argument("--anchor-qd-alpha", type=float, default=1.0, help="Weight for learned quality/acceptance proxy.")
    parser.add_argument("--anchor-qd-beta", type=float, default=0.5, help="Weight for analogical-future support.")
    parser.add_argument("--anchor-qd-coverage-weight", type=float, default=0.8, help="Extra novelty weight for set_coverage selection.")
    parser.add_argument("--anchor-qd-coverage-clusters", type=int, default=6, help="Number of greedy AFC proxy centers for set_coverage selection.")
    parser.add_argument("--anchor-qd-residual-penalty", type=float, default=0.05, help="Penalty for residual magnitude.")
    parser.add_argument("--anchor-qd-margin", type=float, default=0.0, help="Required combined-score gain over base.")
    parser.add_argument("--anchor-qd-tau", type=float, default=1.0, help="Temperature for AFC support exp(-ADE/tau).")
    parser.add_argument("--anchor-qd-anchor-k", type=int, default=4, help="First K base modes use stricter quality acceptance.")
    parser.add_argument("--anchor-qd-anchor-min-prob", type=float, default=None, help="Min quality probability for anchor modes; default uses accept threshold.")
    parser.add_argument("--anchor-qd-diversity-min-prob", type=float, default=0.35, help="Min quality probability for non-anchor modes.")
    parser.add_argument("--anchor-qd-base-quality", type=float, default=0.5, help="Base trajectory quality proxy in combined score.")
    parser.add_argument("--anchor-qd-max-residual-l2", type=float, default=0.0, help="Reject corrections above this mean residual L2; <=0 disables.")
    parser.add_argument("--anchor-qd-spread-floor-endpoint-ratio", type=float, default=0.85)
    parser.add_argument("--anchor-qd-spread-floor-trajectory-ratio", type=float, default=0.85)

    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item]


def _candidate_branch_name(prefix: str, candidate_slots: Sequence[int]) -> str:
    slots = [int(item) for item in candidate_slots]
    if not slots:
        raise ValueError("candidate_slots must not be empty")
    if slots == list(range(min(slots), max(slots) + 1)):
        return f"{prefix}_slots{min(slots)}to{max(slots)}_quality20_pred"
    return f"{prefix}_slots{'x'.join(str(item) for item in slots)}_quality20_pred"


def _candidate_oracle_branch_name(prefix: str, candidate_slots: Sequence[int]) -> str:
    slots = [int(item) for item in candidate_slots]
    if slots == list(range(min(slots), max(slots) + 1)):
        return f"{prefix}_slots{min(slots)}to{max(slots)}_oracle20_pred"
    return f"{prefix}_slots{'x'.join(str(item) for item in slots)}_oracle20_pred"


def _candidate_global_oracle_branch_name(prefix: str, candidate_slots: Sequence[int]) -> str:
    slots = [int(item) for item in candidate_slots]
    if slots == list(range(min(slots), max(slots) + 1)):
        return f"{prefix}_slots{min(slots)}to{max(slots)}_global_oracle20_pred"
    return f"{prefix}_slots{'x'.join(str(item) for item in slots)}_global_oracle20_pred"


def _raw_quality_branch_name(quality_branch: str) -> str:
    if quality_branch.endswith("_20_pred"):
        return f"{quality_branch[:-len('_20_pred')]}_raw_quality20_pred"
    return f"{quality_branch}_raw_quality20_pred"


def _raw_quality_global_branch_name(quality_branch: str) -> str:
    if quality_branch.endswith("_20_pred"):
        return f"{quality_branch[:-len('_20_pred')]}_raw_quality_global20_pred"
    return f"{quality_branch}_raw_quality_global20_pred"


def _anchor_qd_branch_name(quality_branch: str) -> str:
    if quality_branch.endswith("_20_pred"):
        return f"{quality_branch[:-len('_20_pred')]}_anchor_qd20_pred"
    return f"{quality_branch}_anchor_qd20_pred"


def _quality_global_indices(
    scores: torch.Tensor,
    *,
    candidate_slot_ids: torch.Tensor,
    num_base_modes: int,
    keep_k: int,
) -> torch.Tensor:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores [B,S,K,A], got {tuple(scores.shape)}")
    batch_size, num_slots, num_modes, num_agents = [int(item) for item in scores.shape]
    if int(num_modes) != int(num_base_modes):
        raise ValueError(f"scores num_modes={num_modes} does not match base modes={num_base_modes}")
    keep = min(int(keep_k), int(num_slots) * int(num_modes))
    slot_ids = candidate_slot_ids.to(device=scores.device, dtype=torch.long)
    if int(slot_ids.numel()) != int(num_slots):
        raise ValueError(f"candidate_slot_ids has {int(slot_ids.numel())} entries, expected {num_slots}")
    modes = torch.arange(num_modes, device=scores.device, dtype=torch.long)
    pool_indices = (slot_ids[:, None] * int(num_base_modes) + modes[None, :]).reshape(num_slots * num_modes)
    pool_indices = pool_indices[None, :, None].expand(batch_size, num_slots * num_modes, num_agents)
    flat_scores = scores.reshape(batch_size, num_slots * num_modes, num_agents)
    local = torch.topk(flat_scores, k=keep, dim=1, largest=True).indices
    return torch.gather(pool_indices, dim=1, index=local)


def _anchor_qd_select(
    *,
    slow_pred: torch.Tensor,
    slot_candidates: torch.Tensor,
    candidate_slot_ids: torch.Tensor,
    quality_prob: torch.Tensor,
    candidate_afc_support: Optional[torch.Tensor],
    base_afc_support: Optional[torch.Tensor],
    alpha: float,
    beta: float,
    residual_penalty: float,
    margin: float,
    anchor_k: int,
    anchor_min_prob: float,
    diversity_min_prob: float,
    base_quality: float,
    max_residual_l2: float,
) -> Dict[str, torch.Tensor]:
    if slow_pred.ndim != 5:
        raise ValueError(f"slow_pred must have shape [B,K,A,T,2], got {tuple(slow_pred.shape)}")
    if slot_candidates.ndim != 6:
        raise ValueError(f"slot_candidates must have shape [B,S,K,A,T,2], got {tuple(slot_candidates.shape)}")
    if quality_prob.shape != slot_candidates.shape[:4]:
        raise ValueError(f"quality_prob shape {tuple(quality_prob.shape)} does not match candidates {tuple(slot_candidates.shape[:4])}")
    batch_size, num_slots, num_modes, num_agents = [int(item) for item in slot_candidates.shape[:4]]
    slot_ids = candidate_slot_ids.to(device=slot_candidates.device, dtype=torch.long)
    if int(slot_ids.numel()) != num_slots:
        raise ValueError(f"candidate_slot_ids has {int(slot_ids.numel())} entries, expected {num_slots}")

    corrected_positions = (slot_ids != 0).nonzero(as_tuple=False).reshape(-1)
    if int(corrected_positions.numel()) <= 0:
        selected_slots = torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.long)
        return {
            "prediction": slow_pred,
            "selected_slots": selected_slots,
            "accept": torch.zeros_like(selected_slots, dtype=torch.bool),
            "quality_prob": torch.zeros_like(selected_slots, dtype=torch.float32),
            "candidate_afc": torch.zeros_like(selected_slots, dtype=torch.float32),
            "base_afc": torch.zeros_like(selected_slots, dtype=torch.float32),
            "combined_margin": torch.zeros_like(selected_slots, dtype=torch.float32),
            "residual_l2": torch.zeros_like(selected_slots, dtype=torch.float32),
        }

    corrected_candidates = slot_candidates.index_select(dim=1, index=corrected_positions)
    corrected_prob = quality_prob.index_select(dim=1, index=corrected_positions)
    if candidate_afc_support is None:
        corrected_afc = torch.zeros_like(corrected_prob)
    else:
        corrected_afc = candidate_afc_support.to(device=slot_candidates.device, dtype=torch.float32).index_select(
            dim=1,
            index=corrected_positions,
        )
    if base_afc_support is None:
        base_afc = torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.float32)
    else:
        base_afc = base_afc_support.to(device=slot_candidates.device, dtype=torch.float32)

    residual_l2_all = torch.linalg.norm(corrected_candidates - slow_pred[:, None, ...], dim=-1).mean(dim=-1)
    combined = (
        float(alpha) * corrected_prob.to(dtype=torch.float32)
        + float(beta) * corrected_afc
        - float(residual_penalty) * residual_l2_all
    )
    best_local = combined.argmax(dim=1)
    best_combined = _gather_slot_values(combined, best_local)
    best_prob = _gather_slot_values(corrected_prob, best_local)
    best_afc = _gather_slot_values(corrected_afc, best_local)
    best_residual_l2 = _gather_slot_values(residual_l2_all, best_local)

    base_combined = float(alpha) * float(base_quality) + float(beta) * base_afc
    mode_ids = torch.arange(num_modes, device=slot_candidates.device, dtype=torch.long)[None, :, None].expand(batch_size, num_modes, num_agents)
    min_prob = torch.where(
        mode_ids < int(anchor_k),
        torch.full_like(best_prob, fill_value=float(anchor_min_prob)),
        torch.full_like(best_prob, fill_value=float(diversity_min_prob)),
    )
    accept = (best_prob >= min_prob) & (best_combined >= (base_combined + float(margin)))
    if float(max_residual_l2) > 0.0:
        accept = accept & (best_residual_l2 <= float(max_residual_l2))

    gather_index = best_local[:, None, :, :, None, None].expand(batch_size, 1, num_modes, num_agents, slow_pred.shape[-2], slow_pred.shape[-1])
    best_prediction = torch.gather(corrected_candidates, dim=1, index=gather_index).squeeze(1)
    selected_prediction = torch.where(accept[..., None, None], best_prediction, slow_pred)
    best_actual_slots = slot_ids[corrected_positions][best_local]
    selected_slots = torch.where(accept, best_actual_slots, torch.zeros_like(best_actual_slots))
    return {
        "prediction": selected_prediction,
        "selected_slots": selected_slots,
        "accept": accept,
        "quality_prob": best_prob,
        "candidate_afc": best_afc,
        "base_afc": base_afc,
        "combined_margin": best_combined - base_combined,
        "residual_l2": best_residual_l2,
    }


def _afc_proxy_cluster_centers(proxies: torch.Tensor, *, max_clusters: int) -> torch.Tensor:
    query_count, proxy_count = [int(item) for item in proxies.shape[:2]]
    keep = max(1, min(int(max_clusters), proxy_count))
    if proxy_count <= keep:
        return proxies[:, :keep]
    endpoints = proxies[:, :, -1, :]
    selected_rows: List[torch.Tensor] = []
    for query_index in range(query_count):
        chosen = [0]
        min_dist = torch.linalg.norm(endpoints[query_index] - endpoints[query_index, 0:1], dim=-1)
        for _ in range(1, keep):
            next_index = int(min_dist.argmax().item())
            chosen.append(next_index)
            dist = torch.linalg.norm(endpoints[query_index] - endpoints[query_index, next_index : next_index + 1], dim=-1)
            min_dist = torch.minimum(min_dist, dist)
        selected_rows.append(proxies[query_index, torch.as_tensor(chosen, dtype=torch.long)])
    return torch.stack(selected_rows, dim=0)


def _afc_center_support_for_prediction(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    afc_bank: Optional[AnalogicalFutureBank],
    *,
    clusters: int,
    tau: float,
) -> Optional[torch.Tensor]:
    if afc_bank is None:
        return None
    if prediction.ndim not in {5, 6}:
        raise ValueError(f"prediction must have shape [B,K,A,T,2] or [B,S,K,A,T,2], got {tuple(prediction.shape)}")
    _features, valid, top_indices = afc_bank._query(batch)
    query_count = int(top_indices.shape[0])
    if query_count <= 0:
        return None
    tau_value = max(float(tau), 1e-6)
    proxies = afc_bank.futures[top_indices].to(dtype=torch.float32)
    centers = _afc_proxy_cluster_centers(proxies, max_clusters=int(clusters))
    center_count = int(centers.shape[1])
    pred = prediction.detach().to(device="cpu", dtype=torch.float32)

    if prediction.ndim == 5:
        batch_size, num_modes, num_agents = [int(item) for item in pred.shape[:3]]
        pred_by_agent = pred.permute(0, 2, 1, 3, 4)
        if tuple(pred_by_agent.shape[:2]) != tuple(valid.shape):
            raise ValueError(f"prediction batch/agent shape {tuple(pred_by_agent.shape[:2])} does not match mask {tuple(valid.shape)}")
        pred_valid = pred_by_agent[valid]
        ade = torch.linalg.norm(pred_valid[:, :, None, :, :] - centers[:, None, :, :, :], dim=-1).mean(dim=-1)
        support_valid = torch.exp(-ade / tau_value)
        support_by_agent = torch.zeros((batch_size, num_agents, num_modes, center_count), dtype=torch.float32)
        support_by_agent[valid] = support_valid
        return support_by_agent.permute(0, 2, 1, 3)

    batch_size, num_slots, num_modes, num_agents = [int(item) for item in pred.shape[:4]]
    pred_by_agent = pred.permute(0, 3, 1, 2, 4, 5)
    if tuple(pred_by_agent.shape[:2]) != tuple(valid.shape):
        raise ValueError(f"prediction batch/agent shape {tuple(pred_by_agent.shape[:2])} does not match mask {tuple(valid.shape)}")
    pred_valid = pred_by_agent[valid]
    ade = torch.linalg.norm(pred_valid[:, :, :, None, :, :] - centers[:, None, None, :, :, :], dim=-1).mean(dim=-1)
    support_valid = torch.exp(-ade / tau_value)
    support_by_agent = torch.zeros((batch_size, num_agents, num_slots, num_modes, center_count), dtype=torch.float32)
    support_by_agent[valid] = support_valid
    return support_by_agent.permute(0, 2, 3, 1, 4)


def _offdiag_mean_2d(pairwise: torch.Tensor) -> torch.Tensor:
    num_modes = int(pairwise.shape[0])
    if num_modes <= 1:
        return pairwise.new_tensor(0.0)
    keep = ~torch.eye(num_modes, dtype=torch.bool, device=pairwise.device)
    return pairwise[keep].mean()


def _endpoint_spread_modes(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3:
        raise ValueError(f"Expected prediction [K,T,2], got {tuple(prediction.shape)}")
    endpoints = prediction[:, -1, :]
    return _offdiag_mean_2d(torch.cdist(endpoints, endpoints, p=2))


def _trajectory_spread_modes(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3:
        raise ValueError(f"Expected prediction [K,T,2], got {tuple(prediction.shape)}")
    pairwise = torch.linalg.norm(prediction[:, None, :, :] - prediction[None, :, :, :], dim=-1).mean(dim=-1)
    return _offdiag_mean_2d(pairwise)


def _passes_spread_floor(
    candidate_set: torch.Tensor,
    base_set: torch.Tensor,
    *,
    endpoint_ratio: float,
    trajectory_ratio: float,
) -> bool:
    if float(endpoint_ratio) > 0.0:
        endpoint_floor = float(endpoint_ratio) * _endpoint_spread_modes(base_set).detach()
        if bool((_endpoint_spread_modes(candidate_set) + EPS < endpoint_floor).detach().cpu().item()):
            return False
    if float(trajectory_ratio) > 0.0:
        trajectory_floor = float(trajectory_ratio) * _trajectory_spread_modes(base_set).detach()
        if bool((_trajectory_spread_modes(candidate_set) + EPS < trajectory_floor).detach().cpu().item()):
            return False
    return True


def _anchor_qd_set_coverage_select(
    *,
    slow_pred: torch.Tensor,
    slot_candidates: torch.Tensor,
    candidate_slot_ids: torch.Tensor,
    quality_prob: torch.Tensor,
    candidate_afc_support: Optional[torch.Tensor],
    base_afc_support: Optional[torch.Tensor],
    candidate_center_support: Optional[torch.Tensor],
    base_center_support: Optional[torch.Tensor],
    alpha: float,
    beta: float,
    coverage_weight: float,
    residual_penalty: float,
    margin: float,
    anchor_k: int,
    anchor_min_prob: float,
    diversity_min_prob: float,
    base_quality: float,
    max_residual_l2: float,
    hard_preserve_anchor: bool = False,
    spread_floor_endpoint_ratio: float = 0.0,
    spread_floor_trajectory_ratio: float = 0.0,
) -> Dict[str, torch.Tensor]:
    per_base = _anchor_qd_select(
        slow_pred=slow_pred,
        slot_candidates=slot_candidates,
        candidate_slot_ids=candidate_slot_ids,
        quality_prob=quality_prob,
        candidate_afc_support=candidate_afc_support,
        base_afc_support=base_afc_support,
        alpha=alpha,
        beta=beta,
        residual_penalty=residual_penalty,
        margin=margin,
        anchor_k=anchor_k,
        anchor_min_prob=anchor_min_prob,
        diversity_min_prob=diversity_min_prob,
        base_quality=base_quality,
        max_residual_l2=max_residual_l2,
    )
    if bool(hard_preserve_anchor):
        anchor_count = max(0, min(int(anchor_k), int(slow_pred.shape[1])))
        if anchor_count > 0:
            per_base = dict(per_base)
            per_base["prediction"] = per_base["prediction"].clone()
            per_base["selected_slots"] = per_base["selected_slots"].clone()
            per_base["accept"] = per_base["accept"].clone()
            per_base["prediction"][:, :anchor_count] = slow_pred[:, :anchor_count]
            per_base["selected_slots"][:, :anchor_count] = 0
            per_base["accept"][:, :anchor_count] = False
    if candidate_center_support is None or base_center_support is None:
        return per_base

    batch_size, num_slots, num_modes, num_agents = [int(item) for item in slot_candidates.shape[:4]]
    slot_ids = candidate_slot_ids.to(device=slot_candidates.device, dtype=torch.long)
    corrected_positions = (slot_ids != 0).nonzero(as_tuple=False).reshape(-1)
    if int(corrected_positions.numel()) <= 0:
        return per_base

    corrected_candidates = slot_candidates.index_select(dim=1, index=corrected_positions)
    corrected_prob = quality_prob.index_select(dim=1, index=corrected_positions).to(dtype=torch.float32)
    corrected_centers = candidate_center_support.to(device=slot_candidates.device, dtype=torch.float32).index_select(
        dim=1,
        index=corrected_positions,
    )
    if candidate_afc_support is None:
        corrected_afc = torch.zeros_like(corrected_prob)
    else:
        corrected_afc = candidate_afc_support.to(device=slot_candidates.device, dtype=torch.float32).index_select(
            dim=1,
            index=corrected_positions,
        )
    base_afc = (
        torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.float32)
        if base_afc_support is None
        else base_afc_support.to(device=slot_candidates.device, dtype=torch.float32)
    )
    base_centers = base_center_support.to(device=slot_candidates.device, dtype=torch.float32)
    residual_l2_all = torch.linalg.norm(corrected_candidates - slow_pred[:, None, ...], dim=-1).mean(dim=-1)
    base_combined = float(alpha) * float(base_quality) + float(beta) * base_afc
    combined = (
        float(alpha) * corrected_prob
        + float(beta) * corrected_afc
        - float(residual_penalty) * residual_l2_all
    )

    selected_prediction = slow_pred.clone()
    selected_slots = torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.long)
    accept = torch.zeros_like(selected_slots, dtype=torch.bool)
    best_prob = torch.zeros_like(base_afc)
    best_afc = torch.zeros_like(base_afc)
    best_residual_l2 = torch.zeros_like(base_afc)
    best_margin = torch.zeros_like(base_afc)
    spread_floor_reject = torch.zeros_like(selected_slots, dtype=torch.bool)

    actual_slots = slot_ids[corrected_positions]
    center_count = int(corrected_centers.shape[-1])
    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            covered = base_centers[batch_index, :, agent_index, :].amax(dim=0).clone()
            for mode_index in range(num_modes):
                min_prob = float(anchor_min_prob) if mode_index < int(anchor_k) else float(diversity_min_prob)
                mode_combined = combined[batch_index, :, mode_index, agent_index]
                center_support = corrected_centers[batch_index, :, mode_index, agent_index]
                novelty = (center_support * (1.0 - covered).clamp_min(0.0)[None, :]).amax(dim=1)
                if mode_index < int(anchor_k):
                    score = mode_combined
                else:
                    score = mode_combined + float(coverage_weight) * novelty
                local_index = int(score.argmax().detach().cpu().item())
                candidate_margin = mode_combined[local_index] - base_combined[batch_index, mode_index, agent_index]
                candidate_prob = corrected_prob[batch_index, local_index, mode_index, agent_index]
                candidate_residual = residual_l2_all[batch_index, local_index, mode_index, agent_index]
                keep_candidate = (
                    float(candidate_prob.detach().cpu()) >= min_prob
                    and float(candidate_margin.detach().cpu()) >= float(margin)
                )
                if bool(hard_preserve_anchor) and mode_index < int(anchor_k):
                    keep_candidate = False
                if float(max_residual_l2) > 0.0 and float(candidate_residual.detach().cpu()) > float(max_residual_l2):
                    keep_candidate = False
                if keep_candidate and (
                    float(spread_floor_endpoint_ratio) > 0.0 or float(spread_floor_trajectory_ratio) > 0.0
                ):
                    tentative = selected_prediction[batch_index, :, agent_index].clone()
                    tentative[mode_index] = corrected_candidates[
                        batch_index,
                        local_index,
                        mode_index,
                        agent_index,
                    ]
                    if not _passes_spread_floor(
                        tentative,
                        slow_pred[batch_index, :, agent_index],
                        endpoint_ratio=float(spread_floor_endpoint_ratio),
                        trajectory_ratio=float(spread_floor_trajectory_ratio),
                    ):
                        keep_candidate = False
                        spread_floor_reject[batch_index, mode_index, agent_index] = True
                if keep_candidate:
                    selected_prediction[batch_index, mode_index, agent_index] = corrected_candidates[
                        batch_index,
                        local_index,
                        mode_index,
                        agent_index,
                    ]
                    selected_slots[batch_index, mode_index, agent_index] = actual_slots[local_index]
                    accept[batch_index, mode_index, agent_index] = True
                    covered = torch.maximum(covered, center_support[local_index])
                else:
                    covered = torch.maximum(covered, base_centers[batch_index, mode_index, agent_index])
                best_prob[batch_index, mode_index, agent_index] = candidate_prob
                best_afc[batch_index, mode_index, agent_index] = corrected_afc[batch_index, local_index, mode_index, agent_index]
                best_residual_l2[batch_index, mode_index, agent_index] = candidate_residual
                best_margin[batch_index, mode_index, agent_index] = candidate_margin

    return {
        "prediction": selected_prediction,
        "selected_slots": selected_slots,
        "accept": accept,
        "quality_prob": best_prob,
        "candidate_afc": best_afc,
        "base_afc": base_afc,
        "combined_margin": best_margin,
        "residual_l2": best_residual_l2,
        "spread_floor_reject": spread_floor_reject,
    }


def _anchor_qd_role_transport_select(
    *,
    slow_pred: torch.Tensor,
    slot_candidates: torch.Tensor,
    candidate_slot_ids: torch.Tensor,
    quality_prob: torch.Tensor,
    candidate_afc_support: Optional[torch.Tensor],
    base_afc_support: Optional[torch.Tensor],
    candidate_center_support: Optional[torch.Tensor],
    base_center_support: Optional[torch.Tensor],
    alpha: float,
    beta: float,
    coverage_weight: float,
    residual_penalty: float,
    margin: float,
    anchor_k: int,
    diversity_min_prob: float,
    base_quality: float,
    max_residual_l2: float,
    spread_floor_endpoint_ratio: float,
    spread_floor_trajectory_ratio: float,
) -> Dict[str, torch.Tensor]:
    if candidate_center_support is None or base_center_support is None:
        return _anchor_qd_set_coverage_select(
            slow_pred=slow_pred,
            slot_candidates=slot_candidates,
            candidate_slot_ids=candidate_slot_ids,
            quality_prob=quality_prob,
            candidate_afc_support=candidate_afc_support,
            base_afc_support=base_afc_support,
            candidate_center_support=candidate_center_support,
            base_center_support=base_center_support,
            alpha=alpha,
            beta=beta,
            coverage_weight=coverage_weight,
            residual_penalty=residual_penalty,
            margin=margin,
            anchor_k=anchor_k,
            anchor_min_prob=1.0,
            diversity_min_prob=diversity_min_prob,
            base_quality=base_quality,
            max_residual_l2=max_residual_l2,
            hard_preserve_anchor=True,
            spread_floor_endpoint_ratio=spread_floor_endpoint_ratio,
            spread_floor_trajectory_ratio=spread_floor_trajectory_ratio,
        )

    batch_size, num_slots, num_modes, num_agents = [int(item) for item in slot_candidates.shape[:4]]
    slot_ids = candidate_slot_ids.to(device=slot_candidates.device, dtype=torch.long)
    corrected_positions = (slot_ids != 0).nonzero(as_tuple=False).reshape(-1)
    if int(corrected_positions.numel()) <= 0:
        selected_slots = torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.long)
        return {
            "prediction": slow_pred,
            "selected_slots": selected_slots,
            "accept": torch.zeros_like(selected_slots, dtype=torch.bool),
            "quality_prob": torch.zeros_like(selected_slots, dtype=torch.float32),
            "candidate_afc": torch.zeros_like(selected_slots, dtype=torch.float32),
            "base_afc": torch.zeros_like(selected_slots, dtype=torch.float32),
            "combined_margin": torch.zeros_like(selected_slots, dtype=torch.float32),
            "residual_l2": torch.zeros_like(selected_slots, dtype=torch.float32),
            "spread_floor_reject": torch.zeros_like(selected_slots, dtype=torch.bool),
            "role_support": torch.zeros_like(selected_slots, dtype=torch.float32),
        }

    corrected_candidates = slot_candidates.index_select(dim=1, index=corrected_positions)
    corrected_prob = quality_prob.index_select(dim=1, index=corrected_positions).to(dtype=torch.float32)
    corrected_centers = candidate_center_support.to(device=slot_candidates.device, dtype=torch.float32).index_select(
        dim=1,
        index=corrected_positions,
    )
    base_centers = base_center_support.to(device=slot_candidates.device, dtype=torch.float32)
    if candidate_afc_support is None:
        corrected_afc = torch.zeros_like(corrected_prob)
    else:
        corrected_afc = candidate_afc_support.to(device=slot_candidates.device, dtype=torch.float32).index_select(
            dim=1,
            index=corrected_positions,
        )
    base_afc = (
        torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.float32)
        if base_afc_support is None
        else base_afc_support.to(device=slot_candidates.device, dtype=torch.float32)
    )
    residual_l2_all = torch.linalg.norm(corrected_candidates - slow_pred[:, None, ...], dim=-1).mean(dim=-1)
    base_combined = float(alpha) * float(base_quality) + float(beta) * base_afc
    combined = (
        float(alpha) * corrected_prob
        + float(beta) * corrected_afc
        - float(residual_penalty) * residual_l2_all
    )

    selected_prediction = slow_pred.clone()
    selected_slots = torch.zeros((batch_size, num_modes, num_agents), device=slot_candidates.device, dtype=torch.long)
    accept = torch.zeros_like(selected_slots, dtype=torch.bool)
    best_prob = torch.zeros_like(base_afc)
    best_afc = torch.zeros_like(base_afc)
    best_residual_l2 = torch.zeros_like(base_afc)
    best_margin = torch.zeros_like(base_afc)
    best_role_support = torch.zeros_like(base_afc)
    spread_floor_reject = torch.zeros_like(selected_slots, dtype=torch.bool)

    actual_slots = slot_ids[corrected_positions]
    center_count = int(corrected_centers.shape[-1])
    for batch_index in range(batch_size):
        for agent_index in range(num_agents):
            covered = base_centers[batch_index, :, agent_index, :].amax(dim=0).clone()
            for mode_index in range(num_modes):
                if mode_index < int(anchor_k):
                    covered = torch.maximum(covered, base_centers[batch_index, mode_index, agent_index])
                    best_prob[batch_index, mode_index, agent_index] = float(base_quality)
                    best_afc[batch_index, mode_index, agent_index] = base_afc[batch_index, mode_index, agent_index]
                    best_role_support[batch_index, mode_index, agent_index] = base_centers[
                        batch_index,
                        mode_index,
                        agent_index,
                    ].amax()
                    continue
                role_index = int(mode_index % max(center_count, 1))
                mode_combined = combined[batch_index, :, mode_index, agent_index]
                role_support = corrected_centers[batch_index, :, mode_index, agent_index, role_index]
                novelty = role_support * (1.0 - covered[role_index]).clamp_min(0.0)
                score = mode_combined + float(coverage_weight) * (0.75 * role_support + 0.25 * novelty)
                local_index = int(score.argmax().detach().cpu().item())
                candidate_margin = mode_combined[local_index] - base_combined[batch_index, mode_index, agent_index]
                candidate_prob = corrected_prob[batch_index, local_index, mode_index, agent_index]
                candidate_residual = residual_l2_all[batch_index, local_index, mode_index, agent_index]
                keep_candidate = (
                    float(candidate_prob.detach().cpu()) >= float(diversity_min_prob)
                    and float(candidate_margin.detach().cpu()) >= float(margin)
                )
                if float(max_residual_l2) > 0.0 and float(candidate_residual.detach().cpu()) > float(max_residual_l2):
                    keep_candidate = False
                if keep_candidate and (
                    float(spread_floor_endpoint_ratio) > 0.0 or float(spread_floor_trajectory_ratio) > 0.0
                ):
                    tentative = selected_prediction[batch_index, :, agent_index].clone()
                    tentative[mode_index] = corrected_candidates[
                        batch_index,
                        local_index,
                        mode_index,
                        agent_index,
                    ]
                    if not _passes_spread_floor(
                        tentative,
                        slow_pred[batch_index, :, agent_index],
                        endpoint_ratio=float(spread_floor_endpoint_ratio),
                        trajectory_ratio=float(spread_floor_trajectory_ratio),
                    ):
                        keep_candidate = False
                        spread_floor_reject[batch_index, mode_index, agent_index] = True
                if keep_candidate:
                    selected_prediction[batch_index, mode_index, agent_index] = corrected_candidates[
                        batch_index,
                        local_index,
                        mode_index,
                        agent_index,
                    ]
                    selected_slots[batch_index, mode_index, agent_index] = actual_slots[local_index]
                    accept[batch_index, mode_index, agent_index] = True
                    covered = torch.maximum(covered, corrected_centers[batch_index, local_index, mode_index, agent_index])
                else:
                    covered = torch.maximum(covered, base_centers[batch_index, mode_index, agent_index])
                best_prob[batch_index, mode_index, agent_index] = candidate_prob
                best_afc[batch_index, mode_index, agent_index] = corrected_afc[batch_index, local_index, mode_index, agent_index]
                best_residual_l2[batch_index, mode_index, agent_index] = candidate_residual
                best_margin[batch_index, mode_index, agent_index] = candidate_margin
                best_role_support[batch_index, mode_index, agent_index] = role_support[local_index]

    return {
        "prediction": selected_prediction,
        "selected_slots": selected_slots,
        "accept": accept,
        "quality_prob": best_prob,
        "candidate_afc": best_afc,
        "base_afc": base_afc,
        "combined_margin": best_margin,
        "residual_l2": best_residual_l2,
        "spread_floor_reject": spread_floor_reject,
        "role_support": best_role_support,
    }


def _add_afc_aux(
    aux_accumulators: Mapping[str, AuxAccumulator],
    *,
    field_name: str,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    afc_bank: Optional[AnalogicalFutureBank],
) -> None:
    if afc_bank is None:
        return
    valid_count = int(batch["agent_mask"].bool().sum().item())
    if valid_count <= 0:
        return
    aux_accumulators[field_name].add(
        afc_bank.metrics_for_prediction(prediction, batch),
        weight=valid_count,
    )


def _add_v58_branch(
    accumulators: Mapping[str, Any],
    aux_accumulators: Mapping[str, AuxAccumulator],
    *,
    field_name: str,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    miss_threshold: float,
    latencies_ms: Sequence[float],
    afc_bank: Optional[AnalogicalFutureBank],
    base_for_delta: Optional[torch.Tensor] = None,
    spread_base: Optional[torch.Tensor] = None,
    selected_flat_indices: Optional[torch.Tensor] = None,
    num_base_modes: Optional[int] = None,
) -> None:
    _add_branch(
        accumulators,
        aux_accumulators,
        field_name=field_name,
        prediction=prediction,
        batch=batch,
        miss_threshold=float(miss_threshold),
        latencies_ms=latencies_ms,
        base_for_delta=base_for_delta,
        spread_base=spread_base,
        selected_flat_indices=selected_flat_indices,
        num_base_modes=num_base_modes,
    )
    _add_afc_aux(
        aux_accumulators,
        field_name=field_name,
        prediction=prediction,
        batch=batch,
        afc_bank=afc_bank,
    )


def _quality_select_indices(
    logits: torch.Tensor,
    *,
    candidate_slot_ids: torch.Tensor,
    accept_prob_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,S,K,A], got {tuple(logits.shape)}")
    slot_ids = candidate_slot_ids.to(device=logits.device, dtype=torch.long)
    slot0_positions = (slot_ids == 0).nonzero(as_tuple=False).reshape(-1)
    if int(slot0_positions.numel()) <= 0:
        raise ValueError("candidate_slot_ids must include slot0")
    slot0_pos = int(slot0_positions[0].item())
    probs = torch.sigmoid(logits)
    nonzero_mask = slot_ids != 0
    masked_logits = logits.clone()
    masked_logits[:, ~nonzero_mask, :, :] = -float("inf")
    if bool(nonzero_mask.any().item()):
        raw_local = masked_logits.argmax(dim=1).to(dtype=torch.long)
    else:
        raw_local = torch.full_like(logits[:, 0, :, :], fill_value=slot0_pos, dtype=torch.long)
    raw_prob = torch.gather(probs, dim=1, index=raw_local[:, None, :, :]).squeeze(1)
    accept = raw_prob >= float(accept_prob_threshold)
    slot0_local = torch.full_like(raw_local, fill_value=slot0_pos, dtype=torch.long)
    selected_local = torch.where(accept, raw_local, slot0_local)
    selected_prob = torch.gather(probs, dim=1, index=selected_local[:, None, :, :]).squeeze(1)
    fallback = (raw_local != slot0_pos) & (selected_local == slot0_pos)
    return selected_local, raw_local, fallback, selected_prob, raw_prob


def _two_stage_select_indices(
    rank_logits: torch.Tensor,
    accept_logits: torch.Tensor,
    *,
    candidate_slot_ids: torch.Tensor,
    accept_prob_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rank_logits.ndim != 4 or accept_logits.ndim != 4:
        raise ValueError(
            f"Expected rank/accept logits [B,S,K,A], got {tuple(rank_logits.shape)} and {tuple(accept_logits.shape)}"
        )
    slot_ids = candidate_slot_ids.to(device=rank_logits.device, dtype=torch.long)
    slot0_positions = (slot_ids == 0).nonzero(as_tuple=False).reshape(-1)
    if int(slot0_positions.numel()) <= 0:
        raise ValueError("candidate_slot_ids must include slot0")
    slot0_pos = int(slot0_positions[0].item())
    nonzero_mask = slot_ids != 0
    masked_rank = rank_logits.clone()
    masked_rank[:, ~nonzero_mask, :, :] = -float("inf")
    if bool(nonzero_mask.any().item()):
        raw_local = masked_rank.argmax(dim=1).to(dtype=torch.long)
    else:
        raw_local = torch.full_like(rank_logits[:, 0, :, :], fill_value=slot0_pos, dtype=torch.long)
    raw_accept_logit = torch.gather(accept_logits, dim=1, index=raw_local[:, None, :, :]).squeeze(1)
    raw_accept_prob = torch.sigmoid(raw_accept_logit)
    accept = raw_accept_prob >= float(accept_prob_threshold)
    slot0_local = torch.full_like(raw_local, fill_value=slot0_pos, dtype=torch.long)
    selected_local = torch.where(accept, raw_local, slot0_local)
    selected_prob = torch.where(accept, raw_accept_prob, 1.0 - raw_accept_prob)
    fallback = (raw_local != slot0_pos) & (selected_local == slot0_pos)
    raw_rank_score = torch.gather(rank_logits, dim=1, index=raw_local[:, None, :, :]).squeeze(1)
    return selected_local, raw_local, fallback, selected_prob, raw_accept_prob, raw_rank_score


def _actual_slots(local_slots: torch.Tensor, candidate_slot_ids: torch.Tensor) -> torch.Tensor:
    flat = candidate_slot_ids.to(device=local_slots.device, dtype=torch.long)
    return flat[local_slots.to(dtype=torch.long)]


def evaluate(args: argparse.Namespace) -> None:
    candidate_slots = _split_ints(str(args.candidate_slots))
    if 0 not in candidate_slots:
        raise SystemExit("--candidate-slots must include 0 for slot0 fallback")
    if any(slot < 0 or slot >= int(args.residual_slots) for slot in candidate_slots):
        raise SystemExit("--candidate-slots entries must be within [0, residual_slots)")
    _set_seed(int(args.seed))
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
            cfg_path=str(args.slow_cfg_path),
            checkpoint_path=str(args.slow_checkpoint),
        )
    )
    refiner_variant = _checkpoint_variant(str(args.refiner_checkpoint))
    refiner = load_social_cvae_teacher_refiner(str(args.refiner_checkpoint), map_location=device).to(device)
    refiner.eval()
    scorer, scorer_checkpoint = load_v58_slot_quality_scorer(str(args.quality_checkpoint), map_location=device)
    scorer = scorer.to(device)
    scorer.eval()
    include_index_features = bool(scorer_checkpoint.get("meta", {}).get("include_index_features", False))
    checkpoint_mode = str(scorer_checkpoint.get("meta", {}).get("training_mode", "binary_good"))
    selection_mode = str(args.selection_mode)
    if selection_mode == "auto":
        selection_mode = "two_stage_replacement" if checkpoint_mode == "two_stage_replacement" else "quality_threshold"
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
    afc_bank: Optional[AnalogicalFutureBank] = None
    if bool(args.enable_afc):
        afc_bank = build_eth_analogical_future_bank(
            data_root=data_root,
            subset=args.subset,
            train_split=str(args.afc_train_split),
            sample_mode=args.sample_mode,
            data_norm=args.data_norm,
            rotate=bool(args.rotate),
            rotate_time_frame=int(args.rotate_time_frame),
            normalization_stats=normalization_stats,
            min_agents=protocol_settings.min_agents,
            prefer_cache=protocol_settings.prefer_cache,
            max_train_scenes=args.afc_max_train_scenes,
            batch_scenes=int(args.afc_batch_scenes),
            top_m=int(args.afc_top_m),
            eps_values=split_float_list(str(args.afc_eps)),
        )
        print(
            "[eval_v58_slot_quality_scorer] "
            f"AFC enabled train_split={args.afc_train_split} bank_size={afc_bank.bank_size} "
            f"top_m={int(args.afc_top_m)} eps={str(args.afc_eps)}"
        )

    prefix = str(args.diagnostic_prefix)
    quality_branch = str(args.branch_name) if args.branch_name else _candidate_branch_name(prefix, candidate_slots)
    raw_quality_branch = _raw_quality_branch_name(quality_branch)
    raw_quality_global_branch = _raw_quality_global_branch_name(quality_branch)
    anchor_qd_branch = _anchor_qd_branch_name(quality_branch)
    candidate_oracle_branch = _candidate_oracle_branch_name(prefix, candidate_slots)
    candidate_global_branch = _candidate_global_oracle_branch_name(prefix, candidate_slots)
    full_branch = _full_branch_name(prefix, int(args.residual_slots), int(args.keep_k))
    branches = [
        "slow_pred",
        *[f"{prefix}_slot{slot}_20_pred" for slot in range(int(args.residual_slots))],
        quality_branch,
        raw_quality_branch,
        raw_quality_global_branch,
        *([anchor_qd_branch] if bool(args.enable_anchor_qd) else []),
        candidate_oracle_branch,
        candidate_global_branch,
        f"{prefix}_per_base_oracle20_pred",
        f"{prefix}_global_oracle20_pred",
        full_branch,
    ]
    deterministic_branches = [branch for branch in branches if branch != "slow_pred"]
    accumulators = {field_name: BranchAccumulator(field_name, args.miss_threshold) for field_name in branches}
    aux_accumulators = {field_name: AuxAccumulator() for field_name in branches}

    print(
        "[eval_v58_slot_quality_scorer] "
        f"split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} refiner={Path(str(args.refiner_checkpoint)).expanduser().resolve().as_posix()} "
        f"variant={refiner_variant} quality={Path(str(args.quality_checkpoint)).expanduser().resolve().as_posix()} "
        f"candidate_slots={candidate_slots} selection_mode={selection_mode} threshold={float(args.accept_prob_threshold):.3f}"
    )
    if bool(args.enable_anchor_qd):
        print(
            "[eval_v58_slot_quality_scorer] "
            f"anchor_qd enabled branch={anchor_qd_branch} alpha={float(args.anchor_qd_alpha):.3f} "
            f"beta={float(args.anchor_qd_beta):.3f} residual_penalty={float(args.anchor_qd_residual_penalty):.3f} "
            f"margin={float(args.anchor_qd_margin):.3f} anchor_k={int(args.anchor_qd_anchor_k)} "
            f"diversity_min_prob={float(args.anchor_qd_diversity_min_prob):.3f} "
            f"selection_mode={args.anchor_qd_selection_mode} "
            f"coverage_weight={float(args.anchor_qd_coverage_weight):.3f} "
            f"coverage_clusters={int(args.anchor_qd_coverage_clusters)} "
            f"spread_floor_endpoint={float(args.anchor_qd_spread_floor_endpoint_ratio):.3f} "
            f"spread_floor_trajectory={float(args.anchor_qd_spread_floor_trajectory_ratio):.3f}"
        )
    if _is_diagnostic_normalization_source(protocol_settings.normalization_source):
        print("[eval_v58_slot_quality_scorer] warning: selected_samples normalization is diagnostic only")

    candidate_slot_tensor_cpu = torch.tensor(candidate_slots, dtype=torch.long)
    aux_weight = 0
    pool_delta_l2_sum = 0.0
    dynamic_slot_offset_sum = 0.0
    dynamic_slot_offset_seen = False
    energy_risk_sum = 0.0
    chunks = list(_iter_chunks(list(enumerate(selected_samples)), args.batch_scenes))
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
        _add_afc_aux(
            aux_accumulators,
            field_name="slow_pred",
            prediction=slow_output.slow_pred,
            batch=batch,
            afc_bank=afc_bank,
        )

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
            raise SystemExit(f"Expected base modes == keep_k, got {num_base_modes} vs {args.keep_k}")

        valid_count = int(batch["agent_mask"].bool().sum().item())
        valid_base = _valid_base_mask(batch["agent_mask"].to(device=device), num_base_modes)

        for slot_index in range(int(args.residual_slots)):
            field_name = f"{prefix}_slot{slot_index}_20_pred"
            slot_indices = _fixed_slot_indices(
                batch_size=batch_size,
                slot_index=slot_index,
                num_base_modes=num_base_modes,
                num_agents=num_agents,
                device=flat.device,
            )
            _add_v58_branch(
                accumulators,
                aux_accumulators,
                field_name=field_name,
                prediction=refined[:, slot_index],
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=refiner_latencies,
                afc_bank=afc_bank,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=slot_indices,
                num_base_modes=num_base_modes,
            )
            _merge_aux_values(
                aux_accumulators[field_name],
                {
                    "selected_slot_mean": float(slot_index),
                    "selected_slot0_ratio": 1.0 if int(slot_index) == 0 else 0.0,
                },
                weight=valid_count,
            )

        candidate_slot_tensor = candidate_slot_tensor_cpu.to(device=refined.device)
        slot_candidates = refined.index_select(dim=1, index=candidate_slot_tensor)
        if selection_mode == "two_stage_replacement":
            quality_latencies, quality_outputs = _measure_predict_latency_ms(
                lambda: scorer.score_candidate_outputs(
                    slot_candidates,
                    base_trajectory=slow_output.slow_pred,
                    past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                    temporal_energy_features=temporal_energy.to(device=device),
                    candidate_slot_ids=candidate_slot_tensor,
                    max_slot_id=int(args.residual_slots) - 1,
                    include_index_features=include_index_features,
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            rank_logits = quality_outputs["rank_logits"]
            accept_logits = quality_outputs["accept_logits"]
            selected_local, raw_local, fallback_mask, selected_prob, raw_prob, raw_rank_score = _two_stage_select_indices(
                rank_logits,
                accept_logits,
                candidate_slot_ids=candidate_slot_tensor,
                accept_prob_threshold=float(args.accept_prob_threshold),
            )
            raw_quality_scores = rank_logits
        else:
            quality_latencies, quality_logits = _measure_predict_latency_ms(
                lambda: scorer.score_candidates(
                    slot_candidates,
                    base_trajectory=slow_output.slow_pred,
                    past_traj_original_scale=batch["past_traj_original_scale"].to(device=device),
                    temporal_energy_features=temporal_energy.to(device=device),
                    candidate_slot_ids=candidate_slot_tensor,
                    max_slot_id=int(args.residual_slots) - 1,
                    include_index_features=include_index_features,
                ),
                runs=int(args.latency_runs),
                device=device,
            )
            selected_local, raw_local, fallback_mask, selected_prob, raw_prob = _quality_select_indices(
                quality_logits,
                candidate_slot_ids=candidate_slot_tensor,
                accept_prob_threshold=float(args.accept_prob_threshold),
            )
            raw_rank_score = torch.gather(quality_logits, dim=1, index=raw_local[:, None, :, :]).squeeze(1)
            raw_quality_scores = quality_logits
        selected_actual_slots = _actual_slots(selected_local, candidate_slot_tensor)
        raw_actual_slots = _actual_slots(raw_local, candidate_slot_tensor)
        selector_indices = _slot_flat_indices(selected_actual_slots, num_base_modes=num_base_modes)
        raw_selector_indices = _slot_flat_indices(raw_actual_slots, num_base_modes=num_base_modes)
        raw_quality_global_indices = _quality_global_indices(
            raw_quality_scores,
            candidate_slot_ids=candidate_slot_tensor,
            num_base_modes=num_base_modes,
            keep_k=int(args.keep_k),
        )
        selector_latencies = [float(r_ms) + float(q_ms) for r_ms, q_ms in zip(refiner_latencies, quality_latencies)]
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=quality_branch,
            prediction=_gather_candidates(flat, selector_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=selector_latencies,
            afc_bank=afc_bank,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=selector_indices,
            num_base_modes=num_base_modes,
        )
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=raw_quality_branch,
            prediction=_gather_candidates(flat, raw_selector_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=selector_latencies,
            afc_bank=afc_bank,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=raw_selector_indices,
            num_base_modes=num_base_modes,
        )
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=raw_quality_global_branch,
            prediction=_gather_candidates(flat, raw_quality_global_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=selector_latencies,
            afc_bank=afc_bank,
            base_for_delta=_gather_candidates(base_flat, raw_quality_global_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=raw_quality_global_indices,
            num_base_modes=num_base_modes,
        )

        anchor_qd_outputs: Optional[Dict[str, torch.Tensor]] = None
        if bool(args.enable_anchor_qd):
            candidate_center_support: Optional[torch.Tensor] = None
            base_center_support: Optional[torch.Tensor] = None
            if afc_bank is not None:
                candidate_afc_support = afc_bank.support_for_prediction(
                    slot_candidates,
                    batch,
                    tau=float(args.anchor_qd_tau),
                ).to(device=refined.device)
                base_afc_support = afc_bank.support_for_prediction(
                    slow_output.slow_pred,
                    batch,
                    tau=float(args.anchor_qd_tau),
                ).to(device=refined.device)
                if str(args.anchor_qd_selection_mode) in {"set_coverage", "set_coverage_floor", "role_transport"}:
                    candidate_center_support = _afc_center_support_for_prediction(
                        slot_candidates,
                        batch,
                        afc_bank,
                        clusters=int(args.anchor_qd_coverage_clusters),
                        tau=float(args.anchor_qd_tau),
                    )
                    base_center_support = _afc_center_support_for_prediction(
                        slow_output.slow_pred,
                        batch,
                        afc_bank,
                        clusters=int(args.anchor_qd_coverage_clusters),
                        tau=float(args.anchor_qd_tau),
                    )
                    if candidate_center_support is not None:
                        candidate_center_support = candidate_center_support.to(device=refined.device)
                    if base_center_support is not None:
                        base_center_support = base_center_support.to(device=refined.device)
            else:
                candidate_afc_support = None
                base_afc_support = None
            if selection_mode == "two_stage_replacement":
                all_quality_prob = torch.sigmoid(accept_logits)
            else:
                all_quality_prob = torch.sigmoid(quality_logits)
            anchor_min_prob = (
                float(args.accept_prob_threshold)
                if args.anchor_qd_anchor_min_prob is None
                else float(args.anchor_qd_anchor_min_prob)
            )
            if str(args.anchor_qd_selection_mode) == "role_transport":
                anchor_qd_outputs = _anchor_qd_role_transport_select(
                    slow_pred=slow_output.slow_pred,
                    slot_candidates=slot_candidates,
                    candidate_slot_ids=candidate_slot_tensor,
                    quality_prob=all_quality_prob,
                    candidate_afc_support=candidate_afc_support,
                    base_afc_support=base_afc_support,
                    candidate_center_support=candidate_center_support,
                    base_center_support=base_center_support,
                    alpha=float(args.anchor_qd_alpha),
                    beta=float(args.anchor_qd_beta),
                    coverage_weight=float(args.anchor_qd_coverage_weight),
                    residual_penalty=float(args.anchor_qd_residual_penalty),
                    margin=float(args.anchor_qd_margin),
                    anchor_k=int(args.anchor_qd_anchor_k),
                    diversity_min_prob=float(args.anchor_qd_diversity_min_prob),
                    base_quality=float(args.anchor_qd_base_quality),
                    max_residual_l2=float(args.anchor_qd_max_residual_l2),
                    spread_floor_endpoint_ratio=float(args.anchor_qd_spread_floor_endpoint_ratio),
                    spread_floor_trajectory_ratio=float(args.anchor_qd_spread_floor_trajectory_ratio),
                )
            elif str(args.anchor_qd_selection_mode) in {"set_coverage", "set_coverage_floor"}:
                anchor_qd_outputs = _anchor_qd_set_coverage_select(
                    slow_pred=slow_output.slow_pred,
                    slot_candidates=slot_candidates,
                    candidate_slot_ids=candidate_slot_tensor,
                    quality_prob=all_quality_prob,
                    candidate_afc_support=candidate_afc_support,
                    base_afc_support=base_afc_support,
                    candidate_center_support=candidate_center_support,
                    base_center_support=base_center_support,
                    alpha=float(args.anchor_qd_alpha),
                    beta=float(args.anchor_qd_beta),
                    coverage_weight=float(args.anchor_qd_coverage_weight),
                    residual_penalty=float(args.anchor_qd_residual_penalty),
                    margin=float(args.anchor_qd_margin),
                    anchor_k=int(args.anchor_qd_anchor_k),
                    anchor_min_prob=anchor_min_prob,
                    diversity_min_prob=float(args.anchor_qd_diversity_min_prob),
                    base_quality=float(args.anchor_qd_base_quality),
                    max_residual_l2=float(args.anchor_qd_max_residual_l2),
                    hard_preserve_anchor=str(args.anchor_qd_selection_mode) == "set_coverage_floor",
                    spread_floor_endpoint_ratio=(
                        float(args.anchor_qd_spread_floor_endpoint_ratio)
                        if str(args.anchor_qd_selection_mode) == "set_coverage_floor"
                        else 0.0
                    ),
                    spread_floor_trajectory_ratio=(
                        float(args.anchor_qd_spread_floor_trajectory_ratio)
                        if str(args.anchor_qd_selection_mode) == "set_coverage_floor"
                        else 0.0
                    ),
                )
            else:
                anchor_qd_outputs = _anchor_qd_select(
                    slow_pred=slow_output.slow_pred,
                    slot_candidates=slot_candidates,
                    candidate_slot_ids=candidate_slot_tensor,
                    quality_prob=all_quality_prob,
                    candidate_afc_support=candidate_afc_support,
                    base_afc_support=base_afc_support,
                    alpha=float(args.anchor_qd_alpha),
                    beta=float(args.anchor_qd_beta),
                    residual_penalty=float(args.anchor_qd_residual_penalty),
                    margin=float(args.anchor_qd_margin),
                    anchor_k=int(args.anchor_qd_anchor_k),
                    anchor_min_prob=anchor_min_prob,
                    diversity_min_prob=float(args.anchor_qd_diversity_min_prob),
                    base_quality=float(args.anchor_qd_base_quality),
                    max_residual_l2=float(args.anchor_qd_max_residual_l2),
                )
            anchor_qd_indices = _slot_flat_indices(
                anchor_qd_outputs["selected_slots"],
                num_base_modes=num_base_modes,
            )
            _add_v58_branch(
                accumulators,
                aux_accumulators,
                field_name=anchor_qd_branch,
                prediction=anchor_qd_outputs["prediction"],
                batch=batch,
                miss_threshold=float(args.miss_threshold),
                latencies_ms=selector_latencies,
                afc_bank=afc_bank,
                base_for_delta=slow_output.slow_pred,
                spread_base=slow_output.slow_pred,
                selected_flat_indices=anchor_qd_indices,
                num_base_modes=num_base_modes,
            )

        candidate_oracle_local = _per_base_oracle_slots(slot_candidates, ground_truth, metric=str(args.oracle_select_metric))
        candidate_oracle_actual = _actual_slots(candidate_oracle_local, candidate_slot_tensor)
        candidate_oracle_indices = _slot_flat_indices(candidate_oracle_actual, num_base_modes=num_base_modes)
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=candidate_oracle_branch,
            prediction=_gather_candidates(flat, candidate_oracle_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            afc_bank=afc_bank,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=candidate_oracle_indices,
            num_base_modes=num_base_modes,
        )
        candidate_pool_indices = torch.cat(
            [
                _fixed_slot_indices(
                    batch_size=batch_size,
                    slot_index=int(slot),
                    num_base_modes=num_base_modes,
                    num_agents=num_agents,
                    device=flat.device,
                )
                for slot in candidate_slots
            ],
            dim=1,
        )
        candidate_pool = _gather_candidates(flat, candidate_pool_indices)
        candidate_global_local = _oracle_indices(
            candidate_pool,
            ground_truth,
            keep_k=int(args.keep_k),
            metric=str(args.oracle_select_metric),
        )
        candidate_global_indices = torch.gather(candidate_pool_indices, dim=1, index=candidate_global_local)
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=candidate_global_branch,
            prediction=_gather_candidates(flat, candidate_global_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            afc_bank=afc_bank,
            base_for_delta=_gather_candidates(base_flat, candidate_global_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=candidate_global_indices,
            num_base_modes=num_base_modes,
        )

        oracle_slots = _per_base_oracle_slots(refined, ground_truth, metric=str(args.oracle_select_metric))
        oracle_per_base_indices = _slot_flat_indices(oracle_slots, num_base_modes=num_base_modes)
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=f"{prefix}_per_base_oracle20_pred",
            prediction=_gather_candidates(flat, oracle_per_base_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            afc_bank=afc_bank,
            base_for_delta=slow_output.slow_pred,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=oracle_per_base_indices,
            num_base_modes=num_base_modes,
        )
        global_oracle_indices = _oracle_indices(flat, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=f"{prefix}_global_oracle20_pred",
            prediction=_gather_candidates(flat, global_oracle_indices),
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            afc_bank=afc_bank,
            base_for_delta=_gather_candidates(base_flat, global_oracle_indices),
            spread_base=slow_output.slow_pred,
            selected_flat_indices=global_oracle_indices,
            num_base_modes=num_base_modes,
        )
        full_indices = _all_indices(batch_size, num_candidates, num_agents, device=flat.device)
        _add_v58_branch(
            accumulators,
            aux_accumulators,
            field_name=full_branch,
            prediction=flat,
            batch=batch,
            miss_threshold=float(args.miss_threshold),
            latencies_ms=refiner_latencies,
            afc_bank=afc_bank,
            base_for_delta=base_flat,
            spread_base=slow_output.slow_pred,
            selected_flat_indices=full_indices,
            num_base_modes=num_base_modes,
        )

        slot_ade, slot_fde = _candidate_ade_fde_slots(slot_candidates, ground_truth)
        base_ade, base_fde = _base_ade_fde(slow_output.slow_pred, ground_truth)
        slot_score = _score_from_ade_fde(slot_ade, slot_fde, metric=str(args.oracle_select_metric))
        base_score = _score_from_ade_fde(base_ade, base_fde, metric=str(args.oracle_select_metric))
        slot0_pos = int((candidate_slot_tensor == 0).nonzero(as_tuple=False).reshape(-1)[0].item())
        slot0_ade = slot_ade[:, slot0_pos]
        slot0_fde = slot_fde[:, slot0_pos]
        slot0_score = slot_score[:, slot0_pos]
        selected_ade = _gather_slot_values(slot_ade, selected_local)
        selected_fde = _gather_slot_values(slot_fde, selected_local)
        selected_score = _gather_slot_values(slot_score, selected_local)
        raw_ade = _gather_slot_values(slot_ade, raw_local)
        raw_fde = _gather_slot_values(slot_fde, raw_local)
        raw_score = _gather_slot_values(slot_score, raw_local)
        selected_nonzero = selected_actual_slots != 0
        raw_nonzero = raw_actual_slots != 0
        oracle_nonzero = candidate_oracle_actual != 0
        oracle_slot0 = candidate_oracle_actual == 0
        candidate_best_score = slot_score.min(dim=1).values
        candidate_all_bad_vs_slow = candidate_best_score >= (base_score - EPS)
        slot0_good_vs_slow = slot0_score < (base_score - EPS)
        selected_prob_margin = raw_prob - float(args.accept_prob_threshold)

        selector_aux_values: Dict[str, float] = {
            "selected_slot_mean": _mean_on_valid(selected_actual_slots, valid_base),
            "selected_slot0_ratio": _mean_on_valid((selected_actual_slots == 0).to(dtype=torch.float32), valid_base),
            "raw_selected_slot_mean": _mean_on_valid(raw_actual_slots, valid_base),
            "raw_selected_slot0_ratio": _mean_on_valid((raw_actual_slots == 0).to(dtype=torch.float32), valid_base),
            "selector_fallback_to_slot0_ratio": _mean_on_valid(fallback_mask.to(dtype=torch.float32), valid_base),
            "front_oracle_slot_accuracy": _mean_on_valid((selected_actual_slots == candidate_oracle_actual).to(dtype=torch.float32), valid_base),
            "selected_nonzero_ratio": _mean_on_valid(selected_nonzero.to(dtype=torch.float32), valid_base),
            "raw_selected_nonzero_ratio": _mean_on_valid(raw_nonzero.to(dtype=torch.float32), valid_base),
            "front_oracle_nonzero_ratio": _mean_on_valid(oracle_nonzero.to(dtype=torch.float32), valid_base),
            "front_slot0_good_vs_slow_ratio": _mean_on_valid(slot0_good_vs_slow.to(dtype=torch.float32), valid_base),
            "front_all_bad_vs_slow_ratio": _mean_on_valid(candidate_all_bad_vs_slow.to(dtype=torch.float32), valid_base),
            "selector_mean_dade_vs_slot0": _mean_on_valid(selected_ade - slot0_ade, valid_base),
            "selector_mean_dfde_vs_slot0": _mean_on_valid(selected_fde - slot0_fde, valid_base),
            "selector_mean_dscore_vs_slot0": _mean_on_valid(selected_score - slot0_score, valid_base),
            "selector_mean_dscore_vs_slow": _mean_on_valid(selected_score - base_score, valid_base),
            "selector_raw_mean_dade_vs_slot0": _mean_on_valid(raw_ade - slot0_ade, valid_base),
            "selector_raw_mean_dfde_vs_slot0": _mean_on_valid(raw_fde - slot0_fde, valid_base),
            "selector_raw_mean_dscore_vs_slot0": _mean_on_valid(raw_score - slot0_score, valid_base),
            "selector_selected_prob_mean": _mean_on_valid(selected_prob, valid_base),
            "selector_raw_prob_mean": _mean_on_valid(raw_prob, valid_base),
            "selector_raw_prob_margin_mean": _mean_on_valid(selected_prob_margin, valid_base),
            "selector_raw_rank_score_mean": _mean_on_valid(raw_rank_score, valid_base),
            "selector_accept_prob_threshold": float(args.accept_prob_threshold),
        }
        for slot in candidate_slots:
            selector_aux_values[f"selected_slot{slot}_ratio"] = _mean_on_valid(
                (selected_actual_slots == int(slot)).to(dtype=torch.float32),
                valid_base,
            )
            selector_aux_values[f"raw_selected_slot{slot}_ratio"] = _mean_on_valid(
                (raw_actual_slots == int(slot)).to(dtype=torch.float32),
                valid_base,
            )
        _merge_aux_values(aux_accumulators[quality_branch], selector_aux_values, weight=valid_count)
        _merge_aux_values(
            aux_accumulators[raw_quality_branch],
            {
                "selected_slot_mean": _mean_on_valid(raw_actual_slots, valid_base),
                "selected_slot0_ratio": _mean_on_valid((raw_actual_slots == 0).to(dtype=torch.float32), valid_base),
                "selected_nonzero_ratio": _mean_on_valid(raw_nonzero.to(dtype=torch.float32), valid_base),
                "selector_raw_prob_mean": _mean_on_valid(raw_prob, valid_base),
                "selector_raw_rank_score_mean": _mean_on_valid(raw_rank_score, valid_base),
            },
            weight=valid_count,
        )
        raw_quality_global_slots = torch.div(raw_quality_global_indices, num_base_modes, rounding_mode="floor")
        valid_raw_quality_global = batch["agent_mask"].bool().to(device=device)[:, None, :].expand_as(raw_quality_global_slots)
        _merge_aux_values(
            aux_accumulators[raw_quality_global_branch],
            {
                "selected_slot_mean": _mean_on_valid(raw_quality_global_slots, valid_raw_quality_global),
                "selected_slot0_ratio": _mean_on_valid(
                    (raw_quality_global_slots == 0).to(dtype=torch.float32),
                    valid_raw_quality_global,
                ),
                "selected_nonzero_ratio": _mean_on_valid(
                    (raw_quality_global_slots != 0).to(dtype=torch.float32),
                    valid_raw_quality_global,
                ),
            },
            weight=valid_count,
        )
        if anchor_qd_outputs is not None:
            anchor_qd_slots = anchor_qd_outputs["selected_slots"]
            anchor_qd_accept = anchor_qd_outputs["accept"]
            anchor_qd_spread_floor_reject = anchor_qd_outputs.get(
                "spread_floor_reject",
                torch.zeros_like(anchor_qd_accept, dtype=torch.bool),
            )
            anchor_qd_role_support = anchor_qd_outputs.get(
                "role_support",
                torch.zeros_like(anchor_qd_outputs["candidate_afc"], dtype=torch.float32),
            )
            anchor_qd_ade, anchor_qd_fde = _base_ade_fde(anchor_qd_outputs["prediction"], ground_truth)
            anchor_qd_score = _score_from_ade_fde(anchor_qd_ade, anchor_qd_fde, metric=str(args.oracle_select_metric))
            mode_ids = torch.arange(num_base_modes, device=device, dtype=torch.long)[None, :, None].expand_as(anchor_qd_slots)
            anchor_mask = mode_ids < int(args.anchor_qd_anchor_k)
            diversity_mask = ~anchor_mask
            _merge_aux_values(
                aux_accumulators[anchor_qd_branch],
                {
                    "selected_slot_mean": _mean_on_valid(anchor_qd_slots, valid_base),
                    "selected_slot0_ratio": _mean_on_valid((anchor_qd_slots == 0).to(dtype=torch.float32), valid_base),
                    "selected_nonzero_ratio": _mean_on_valid((anchor_qd_slots != 0).to(dtype=torch.float32), valid_base),
                    "anchor_qd_corrected_ratio": _mean_on_valid(anchor_qd_accept.to(dtype=torch.float32), valid_base),
                    "anchor_qd_base_fallback_ratio": _mean_on_valid((~anchor_qd_accept).to(dtype=torch.float32), valid_base),
                    "anchor_qd_quality_prob_mean": _mean_on_valid(anchor_qd_outputs["quality_prob"], valid_base),
                    "anchor_qd_candidate_afc_support_mean": _mean_on_valid(anchor_qd_outputs["candidate_afc"], valid_base),
                    "anchor_qd_base_afc_support_mean": _mean_on_valid(anchor_qd_outputs["base_afc"], valid_base),
                    "anchor_qd_combined_margin_mean": _mean_on_valid(anchor_qd_outputs["combined_margin"], valid_base),
                    "anchor_qd_residual_l2_mean": _mean_on_valid(anchor_qd_outputs["residual_l2"], valid_base),
                    "anchor_qd_role_support_mean": _mean_on_valid(anchor_qd_role_support, valid_base),
                    "anchor_qd_spread_floor_reject_ratio": _mean_on_valid(
                        anchor_qd_spread_floor_reject.to(dtype=torch.float32),
                        valid_base,
                    ),
                    "anchor_qd_alpha": float(args.anchor_qd_alpha),
                    "anchor_qd_beta": float(args.anchor_qd_beta),
                    "anchor_qd_residual_penalty": float(args.anchor_qd_residual_penalty),
                    "anchor_qd_margin": float(args.anchor_qd_margin),
                    "anchor_qd_tau": float(args.anchor_qd_tau),
                    "anchor_qd_anchor_k": float(args.anchor_qd_anchor_k),
                    "anchor_qd_selection_mode_set_coverage": 1.0 if str(args.anchor_qd_selection_mode) == "set_coverage" else 0.0,
                    "anchor_qd_selection_mode_set_coverage_floor": (
                        1.0 if str(args.anchor_qd_selection_mode) == "set_coverage_floor" else 0.0
                    ),
                    "anchor_qd_selection_mode_role_transport": (
                        1.0 if str(args.anchor_qd_selection_mode) == "role_transport" else 0.0
                    ),
                    "anchor_qd_coverage_weight": float(args.anchor_qd_coverage_weight),
                    "anchor_qd_coverage_clusters": float(args.anchor_qd_coverage_clusters),
                    "anchor_qd_spread_floor_endpoint_ratio": float(args.anchor_qd_spread_floor_endpoint_ratio),
                    "anchor_qd_spread_floor_trajectory_ratio": float(args.anchor_qd_spread_floor_trajectory_ratio),
                    "anchor_qd_anchor_min_prob": (
                        float(args.accept_prob_threshold)
                        if args.anchor_qd_anchor_min_prob is None
                        else float(args.anchor_qd_anchor_min_prob)
                    ),
                    "anchor_qd_diversity_min_prob": float(args.anchor_qd_diversity_min_prob),
                    "anchor_qd_base_quality": float(args.anchor_qd_base_quality),
                    "anchor_qd_max_residual_l2": float(args.anchor_qd_max_residual_l2),
                    "anchor_qd_mean_dade_vs_slow": _mean_on_valid(anchor_qd_ade - base_ade, valid_base),
                    "anchor_qd_mean_dfde_vs_slow": _mean_on_valid(anchor_qd_fde - base_fde, valid_base),
                    "anchor_qd_mean_dscore_vs_slow": _mean_on_valid(anchor_qd_score - base_score, valid_base),
                },
                weight=valid_count,
            )
            _merge_conditional_ratio(
                aux_accumulators[anchor_qd_branch],
                "anchor_qd_anchor_corrected_ratio",
                anchor_qd_accept,
                anchor_mask,
                valid_base,
            )
            _merge_conditional_ratio(
                aux_accumulators[anchor_qd_branch],
                "anchor_qd_diversity_corrected_ratio",
                anchor_qd_accept,
                diversity_mask,
                valid_base,
            )
            _merge_conditional_ratio(
                aux_accumulators[anchor_qd_branch],
                "anchor_qd_corrected_improves_slow_score_ratio",
                anchor_qd_score < (base_score - EPS),
                anchor_qd_accept,
                valid_base,
            )
            _merge_conditional_ratio(
                aux_accumulators[anchor_qd_branch],
                "anchor_qd_corrected_hurts_slow_score_ratio",
                anchor_qd_score > (base_score + EPS),
                anchor_qd_accept,
                valid_base,
            )
        selector_aux = aux_accumulators[quality_branch]
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_better_slot0_ade_ratio", selected_ade < (slot0_ade - EPS), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_better_slot0_fde_ratio", selected_fde < (slot0_fde - EPS), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_hurt_slot0_ade_ratio", selected_ade > (slot0_ade + EPS), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_hurt_slot0_fde_ratio", selected_fde > (slot0_fde + EPS), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_improves_slow_score_ratio", selected_score < (base_score - EPS), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "accepted_nonzero_hurts_slow_score_ratio", selected_score > (base_score + EPS), selected_nonzero, valid_base)
        _merge_conditional_mean(selector_aux, "accepted_nonzero_mean_dade_vs_slot0", selected_ade - slot0_ade, selected_nonzero, valid_base)
        _merge_conditional_mean(selector_aux, "accepted_nonzero_mean_dfde_vs_slot0", selected_fde - slot0_fde, selected_nonzero, valid_base)
        _merge_conditional_mean(selector_aux, "accepted_nonzero_mean_dscore_vs_slot0", selected_score - slot0_score, selected_nonzero, valid_base)
        _merge_conditional_mean(selector_aux, "accepted_nonzero_prob_mean", selected_prob, selected_nonzero, valid_base)
        _merge_conditional_mean(selector_aux, "accepted_nonzero_prob_margin_mean", selected_prob - float(args.accept_prob_threshold), selected_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "raw_nonzero_hurt_slot0_ade_ratio", raw_ade > (slot0_ade + EPS), raw_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "raw_nonzero_hurt_slot0_fde_ratio", raw_fde > (slot0_fde + EPS), raw_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "fallback_raw_hurt_slot0_ade_ratio", raw_ade > (slot0_ade + EPS), fallback_mask, valid_base)
        _merge_conditional_ratio(selector_aux, "fallback_raw_hurt_slot0_fde_ratio", raw_fde > (slot0_fde + EPS), fallback_mask, valid_base)
        _merge_conditional_mean(selector_aux, "fallback_raw_prob_mean", raw_prob, fallback_mask, valid_base)
        _merge_conditional_mean(selector_aux, "fallback_raw_prob_margin_mean", raw_prob - float(args.accept_prob_threshold), fallback_mask, valid_base)
        _merge_conditional_ratio(selector_aux, "missed_oracle_nonzero_ratio", selected_actual_slots == 0, oracle_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "oracle_nonzero_recall_ratio", selected_actual_slots == candidate_oracle_actual, oracle_nonzero, valid_base)
        _merge_conditional_ratio(selector_aux, "oracle_slot0_recall_ratio", selected_actual_slots == 0, oracle_slot0, valid_base)
        _merge_conditional_ratio(selector_aux, "all_bad_fallback_to_slot0_ratio", selected_actual_slots == 0, candidate_all_bad_vs_slow, valid_base)
        _merge_conditional_ratio(selector_aux, "all_bad_nonzero_accept_ratio", selected_actual_slots != 0, candidate_all_bad_vs_slow, valid_base)

        _merge_aux_values(
            aux_accumulators[candidate_oracle_branch],
            {
                "selected_slot_mean": _mean_on_valid(candidate_oracle_actual, valid_base),
                "selected_slot0_ratio": _mean_on_valid((candidate_oracle_actual == 0).to(dtype=torch.float32), valid_base),
            },
            weight=valid_count,
        )
        global_slots = torch.div(candidate_global_indices, num_base_modes, rounding_mode="floor")
        valid_global = batch["agent_mask"].bool().to(device=device)[:, None, :].expand_as(global_slots)
        _merge_aux_values(
            aux_accumulators[candidate_global_branch],
            {
                "selected_slot_mean": _mean_on_valid(global_slots, valid_global),
                "selected_slot0_ratio": _mean_on_valid((global_slots == 0).to(dtype=torch.float32), valid_global),
            },
            weight=valid_count,
        )
        _merge_aux_values(
            aux_accumulators[f"{prefix}_per_base_oracle20_pred"],
            {
                "selected_slot_mean": _mean_on_valid(oracle_slots, valid_base),
                "selected_slot0_ratio": _mean_on_valid((oracle_slots == 0).to(dtype=torch.float32), valid_base),
            },
            weight=valid_count,
        )
        global_slots_full = torch.div(global_oracle_indices, num_base_modes, rounding_mode="floor")
        valid_global_full = batch["agent_mask"].bool().to(device=device)[:, None, :].expand_as(global_slots_full)
        _merge_aux_values(
            aux_accumulators[f"{prefix}_global_oracle20_pred"],
            {
                "selected_slot_mean": _mean_on_valid(global_slots_full, valid_global_full),
                "selected_slot0_ratio": _mean_on_valid((global_slots_full == 0).to(dtype=torch.float32), valid_global_full),
            },
            weight=valid_count,
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

        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(int(args.log_every), 1) == 0:
            print(
                "[eval_v58_slot_quality_scorer] "
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
            "script": "trustmoe_traj.scripts.eval_v58_slot_quality_scorer",
            "variant": "v58k_slot_quality_scorer_eval",
            "diagnostic_prefix": prefix,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_settings.protocol,
            "split": args.split,
            "residual_slots": int(args.residual_slots),
            "keep_k": int(args.keep_k),
            "candidate_slots": list(candidate_slots),
            "quality_branch": quality_branch,
            "selection_mode": selection_mode,
            "checkpoint_training_mode": checkpoint_mode,
            "accept_prob_threshold": float(args.accept_prob_threshold),
            "oracle_select_metric": args.oracle_select_metric,
            "afc_enabled": bool(args.enable_afc),
            "afc_train_split": str(args.afc_train_split),
            "afc_top_m": int(args.afc_top_m),
            "afc_eps": str(args.afc_eps),
            "afc_bank_size": None if afc_bank is None else int(afc_bank.bank_size),
            "anchor_qd_enabled": bool(args.enable_anchor_qd),
            "anchor_qd_branch": anchor_qd_branch if bool(args.enable_anchor_qd) else None,
            "anchor_qd_selection_mode": str(args.anchor_qd_selection_mode),
            "anchor_qd_alpha": float(args.anchor_qd_alpha),
            "anchor_qd_beta": float(args.anchor_qd_beta),
            "anchor_qd_coverage_weight": float(args.anchor_qd_coverage_weight),
            "anchor_qd_coverage_clusters": int(args.anchor_qd_coverage_clusters),
            "anchor_qd_spread_floor_endpoint_ratio": float(args.anchor_qd_spread_floor_endpoint_ratio),
            "anchor_qd_spread_floor_trajectory_ratio": float(args.anchor_qd_spread_floor_trajectory_ratio),
            "anchor_qd_residual_penalty": float(args.anchor_qd_residual_penalty),
            "anchor_qd_margin": float(args.anchor_qd_margin),
            "anchor_qd_anchor_k": int(args.anchor_qd_anchor_k),
            "refiner_variant": refiner_variant,
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
        "quality_checkpoint": Path(str(args.quality_checkpoint)).expanduser().resolve().as_posix(),
        "metrics": _coerce_jsonable(metrics),
    }
    _print_eval_summary(metrics, branches=deterministic_branches)
    output_path = Path(str(args.output_json)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
