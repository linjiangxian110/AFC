"""Export AFC Experiment 5 retrieval-visualization cases.

This is a second-stage diagnostic. It does not train a model. It reconstructs
the AFC retrieval bank for a small set of query cases, exports per-case
retrieval/prediction metadata, and draws qualitative trajectory panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.models import MoFlowSlowPredictor
from trustmoe_traj.scripts.analogical_future_coverage import (
    AnalogicalFutureBank,
    _distance_weights,
    _weighted_entropy_effective_count,
    _weighted_modes_one,
    build_eth_analogical_future_bank,
    split_float_list,
)
from trustmoe_traj.scripts.diagnose_headroom_analysis import (
    _afc_greedy_indices,
    _predict_slow_repeated_pool,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    _gather_candidates,
    _oracle_indices,
    _structured_fps_indices,
)
from trustmoe_traj.scripts.run_eval import (
    DEFAULT_DATA_ROOT,
    EVAL_PROTOCOLS,
    NORMALIZATION_SOURCES,
    _build_base_per_sample_records,
    _count_selected_eval_items,
    _infer_agents,
    _iter_chunks,
    _resolve_device,
    _resolve_normalization_stats,
    _resolve_protocol_settings,
    _select_samples,
    _validate_protocol_assumptions,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import _predictor_cfg, _set_seed


BRANCHES: Sequence[Tuple[str, str]] = (
    ("slow20_pred", "slow20"),
    ("gt_oracle20_pred", "GT oracle"),
    ("endpoint_fps20_pred", "Endpoint FPS"),
    ("afc_greedy20_pred", "AFC greedy"),
)


@dataclass
class CasePayload:
    row: Dict[str, Any]
    plot_payload: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export AFC Experiment 5 retrieval visualization cases.")
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
    parser.add_argument("--max-scenes", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--num-to-gen", type=int, default=1)
    parser.add_argument("--slow-cfg-path", type=str, required=True)
    parser.add_argument("--slow-checkpoint", type=str, required=True)
    parser.add_argument("--keep-k", type=int, default=20)
    parser.add_argument("--slow-pool-k", type=int, default=200)
    parser.add_argument("--oracle-select-metric", type=str, default="ade_fde", choices=["fde", "ade_fde"])
    parser.add_argument("--afc-selection-tau", type=float, default=1.0)
    parser.add_argument("--afc-train-split", type=str, default="train")
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=float, default=0.5)
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--max-cases-per-type", type=int, default=2)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="afc_exp5_retrieval_cases")
    parser.set_defaults(prefer_cache=None)
    parser.add_argument("--prefer-cache", dest="prefer_cache", action="store_true")
    parser.add_argument("--no-prefer-cache", dest="prefer_cache", action="store_false")
    return parser


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _scene_meta(sample: Mapping[str, Any]) -> Dict[str, Any]:
    meta = sample.get("scene_meta", {})
    if hasattr(meta, "to_dict"):
        return dict(meta.to_dict())
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


def _agent_id(sample: Mapping[str, Any], agent_index: int) -> str:
    extras = sample.get("extras", {}) or {}
    agent_ids = extras.get("agent_ids")
    try:
        return str(int(agent_ids[int(agent_index)]))
    except Exception:
        return str(int(agent_index))


def _to_list(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().to(torch.float32).tolist()


def _accuracy(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    dist = torch.linalg.norm(pred - gt[None, :, :], dim=-1)
    ade = dist.mean(dim=-1)
    fde = dist[:, -1]
    return {
        "ADE_avg": float(ade.mean().item()),
        "FDE_avg": float(fde.mean().item()),
        "ADE_min": float(ade.min().item()),
        "FDE_min": float(fde.min().item()),
    }


def _endpoint_spread(pred: torch.Tensor) -> float:
    endpoints = pred[:, -1, :]
    if int(endpoints.shape[0]) <= 1:
        return 0.0
    pairwise = torch.cdist(endpoints, endpoints, p=2)
    keep = ~torch.eye(int(endpoints.shape[0]), dtype=torch.bool, device=endpoints.device)
    return float(pairwise[keep].mean().item())


def _branch_query_metrics(
    pred: torch.Tensor,
    proxies: torch.Tensor,
    distances: torch.Tensor,
    gt: torch.Tensor,
    *,
    eps: float,
    scale: float,
) -> Dict[str, Any]:
    ade_pairwise = torch.linalg.norm(pred[:, None, :, :] - proxies[None, :, :, :], dim=-1).mean(dim=-1)
    proxy_to_pred = ade_pairwise.min(dim=0).values
    pred_to_proxy = ade_pairwise.min(dim=1).values
    proxy_pairwise = torch.linalg.norm(proxies[:, None, :, :] - proxies[None, :, :, :], dim=-1).mean(dim=-1)
    centers, mode_weights = _weighted_modes_one(proxies, distances, proxy_pairwise, float(eps), scale=scale)

    if int(centers.shape[0]) <= 0:
        mode_to_pred = torch.empty((0,), dtype=torch.float32)
        pred_to_mode = torch.full((int(pred.shape[0]),), float("inf"), dtype=torch.float32)
        mode_covered = torch.empty((0,), dtype=torch.bool)
        pred_matched = torch.zeros((int(pred.shape[0]),), dtype=torch.bool)
        weighted_mode_recall = 0.0
        mode_precision = 0.0
        mode_chamfer = float("inf")
    else:
        mode_pairwise = torch.linalg.norm(pred[:, None, :, :] - centers[None, :, :, :], dim=-1).mean(dim=-1)
        mode_to_pred = mode_pairwise.min(dim=0).values
        pred_to_mode = mode_pairwise.min(dim=1).values
        mode_covered = mode_to_pred <= float(eps)
        pred_matched = pred_to_mode <= float(eps)
        weighted_mode_recall = float((mode_weights * mode_covered.to(dtype=torch.float32)).sum().item())
        mode_precision = float(pred_matched.to(dtype=torch.float32).mean().item())
        mode_chamfer = float(0.5 * ((mode_weights * mode_to_pred).sum().item() + pred_to_mode.mean().item()))

    gt_dist = torch.linalg.norm(pred - gt[None, :, :], dim=-1).mean(dim=-1)
    unsupported = (~pred_matched) & (gt_dist > float(eps))
    accuracy = _accuracy(pred, gt)
    return {
        **accuracy,
        "AFC_WMR": weighted_mode_recall,
        "AFC_precision": mode_precision,
        "Unsupported": float(unsupported.to(dtype=torch.float32).mean().item()),
        "AFC_chamfer": mode_chamfer,
        "mode_count": int(centers.shape[0]),
        "endpoint_spread": _endpoint_spread(pred),
        "supported_mask": pred_matched.detach().cpu().bool().tolist(),
        "unsupported_mask": unsupported.detach().cpu().bool().tolist(),
        "mode_centers": centers.detach().cpu().to(torch.float32),
        "mode_weights": mode_weights.detach().cpu().to(torch.float32),
        "proxy_to_pred_mean": float(proxy_to_pred.mean().item()),
        "pred_to_proxy_mean": float(pred_to_proxy.mean().item()),
    }


def _case_type_for_row(row: Mapping[str, Any]) -> str:
    if str(row.get("case_type")):
        return str(row["case_type"])
    return "case"


def _choose_cases(cases: Sequence[CasePayload], max_per_type: int) -> List[CasePayload]:
    by_key = {row_type: [] for row_type in ["high_confidence_success", "gt_oracle_afc_drop", "endpoint_fps_unsupported", "low_confidence_caution"]}
    for case in cases:
        row = case.row
        confidence = float(row["retrieval_confidence"])
        slow_wmr = float(row["slow20_AFC_WMR"])
        gt_drop = float(row["gt_oracle_AFC_WMR"]) - slow_wmr
        fps_unsupported = float(row["endpoint_fps_Unsupported"])
        by_key["high_confidence_success"].append(((-confidence, -slow_wmr), case))
        by_key["gt_oracle_afc_drop"].append(((gt_drop, float(row["gt_oracle_FDE_avg"]) - float(row["slow20_FDE_avg"])), case))
        by_key["endpoint_fps_unsupported"].append(((-fps_unsupported, -float(row["endpoint_fps_endpoint_spread"])), case))
        by_key["low_confidence_caution"].append(((confidence, -slow_wmr), case))

    selected: List[CasePayload] = []
    used: set[Tuple[str, str, str]] = set()
    for case_type in by_key:
        for _score, case in sorted(by_key[case_type], key=lambda item: item[0]):
            identity = (str(case.row["dataset"]), str(case.row["sample_id"]), str(case.row["agent_index"]))
            if identity in used:
                continue
            new_row = dict(case.row)
            new_row["case_type"] = case_type
            selected.append(CasePayload(row=new_row, plot_payload=case.plot_payload))
            used.add(identity)
            if sum(1 for item in selected if item.row.get("case_type") == case_type) >= int(max_per_type):
                break
    return selected


def _confidence_bins(cases: Sequence[CasePayload]) -> List[Dict[str, Any]]:
    values = sorted(float(case.row["retrieval_confidence"]) for case in cases)
    if not values:
        return []
    low_cut = values[max(0, int(len(values) / 3) - 1)]
    high_cut = values[max(0, int(2 * len(values) / 3) - 1)]
    bins = {"High": [], "Medium": [], "Low": []}
    for case in cases:
        confidence = float(case.row["retrieval_confidence"])
        if confidence >= high_cut:
            bins["High"].append(case)
        elif confidence <= low_cut:
            bins["Low"].append(case)
        else:
            bins["Medium"].append(case)
    rows: List[Dict[str, Any]] = []
    for label in ["High", "Medium", "Low"]:
        items = bins[label]
        if not items:
            rows.append({"confidence_bin": label, "num_samples": 0})
            continue
        rows.append(
            {
                "confidence_bin": label,
                "num_samples": len(items),
                "avg_retrieval_confidence": sum(float(item.row["retrieval_confidence"]) for item in items) / len(items),
                "avg_top1_distance": sum(float(item.row["retrieval_top1_distance"]) for item in items) / len(items),
                "avg_effective_m": sum(float(item.row["retrieval_effective_m"]) for item in items) / len(items),
                "avg_mode_count": sum(float(item.row["retrieved_mode_count"]) for item in items) / len(items),
                "avg_slow20_AFC_WMR": sum(float(item.row["slow20_AFC_WMR"]) for item in items) / len(items),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case_type",
        "dataset",
        "split",
        "sample_id",
        "seq_id",
        "frame_id",
        "agent_id",
        "agent_index",
        "retrieval_confidence",
        "retrieval_top1_distance",
        "retrieval_effective_m",
        "retrieved_mode_count",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _plot_case(case: CasePayload, output_base: Path, *, eps: float) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"plot_skipped={output_base.as_posix()} reason={exc}")
        return False

    payload = case.plot_payload
    row = case.row
    past = torch.as_tensor(payload["past"], dtype=torch.float32)
    gt = torch.as_tensor(payload["gt"], dtype=torch.float32)
    proxies = torch.as_tensor(payload["proxies"], dtype=torch.float32)
    modes = torch.as_tensor(payload["mode_centers"], dtype=torch.float32)
    branch_predictions = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in payload["predictions"].items()}
    branch_support = payload["supported_masks"]

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.6), constrained_layout=True)
    title = (
        f"{row['case_type']} | {row['dataset']} {row['seq_id']} frame={row['frame_id']} "
        f"agent={row['agent_id']} conf={float(row['retrieval_confidence']):.3f}"
    )
    fig.suptitle(title, fontsize=10)

    def draw_context(ax: Any) -> None:
        ax.plot(past[:, 0], past[:, 1], color="#111111", linewidth=2.2, label="observed past")
        ax.plot(gt[:, 0], gt[:, 1], color="#000000", linewidth=1.8, linestyle="--", label="GT")
        ax.scatter([0.0], [0.0], color="#111111", s=12, zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#DDDDDD", linewidth=0.6)

    ax = axes[0]
    draw_context(ax)
    for proxy in proxies:
        ax.plot(proxy[:, 0], proxy[:, 1], color="#8EC7E8", linewidth=0.8, alpha=0.30)
    for mode in modes:
        ax.plot(mode[:, 0], mode[:, 1], color="#1F77B4", linewidth=1.8, alpha=0.85)
    ax.set_title("Retrieved analogical futures")

    branch_styles = {
        "slow20_pred": ("slow20", "#333333"),
        "gt_oracle20_pred": ("GT oracle", "#D62728"),
        "endpoint_fps20_pred": ("Endpoint FPS", "#F58518"),
        "afc_greedy20_pred": ("AFC greedy", "#1F77B4"),
    }
    for ax, (branch, (label, color)) in zip(axes[1:], branch_styles.items()):
        draw_context(ax)
        pred = branch_predictions[branch]
        supported = [bool(item) for item in branch_support[branch]]
        for index, traj in enumerate(pred):
            line_color = color if supported[index] else "#B8B8B8"
            line_style = "-" if supported[index] else "--"
            alpha = 0.70 if supported[index] else 0.45
            ax.plot(traj[:, 0], traj[:, 1], color=line_color, linewidth=1.1, linestyle=line_style, alpha=alpha)
        ax.set_title(
            f"{label}\nWMR={float(row[f'{label_key(branch)}_AFC_WMR']):.3f}, "
            f"Unsup={float(row[f'{label_key(branch)}_Unsupported']):.3f}",
            fontsize=9,
        )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)
    return True


def label_key(branch: str) -> str:
    if branch == "slow20_pred":
        return "slow20"
    if branch == "gt_oracle20_pred":
        return "gt_oracle"
    if branch == "endpoint_fps20_pred":
        return "endpoint_fps"
    if branch == "afc_greedy20_pred":
        return "afc_greedy"
    raise KeyError(branch)


def _render_summary(*, args: argparse.Namespace, selected: Sequence[CasePayload], confidence_rows: Sequence[Mapping[str, Any]], plots: Sequence[Path]) -> str:
    lines = [
        "# AFC Experiment 5 Retrieval Case Export Summary",
        "",
        f"- run_id: `{args.run_id}`",
        f"- dataset: `{args.subset}`",
        f"- split: `{args.split}`",
        f"- AFC Top-M: `{args.afc_top_m}`",
        f"- AFC eps: `{args.afc_eps}`",
        f"- selected cases: `{len(selected)}`",
        "",
        "## Selected Cases",
        "",
        "| case_type | sample_id | agent_id | retrieval_conf | mode_count | slow20 WMR | GT oracle WMR | endpoint unsupported |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in selected:
        row = case.row
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case_type", "")),
                    str(row.get("sample_id", "")),
                    str(row.get("agent_id", "")),
                    _fmt(_num(row.get("retrieval_confidence"))),
                    _fmt(_num(row.get("retrieved_mode_count"))),
                    _fmt(_num(row.get("slow20_AFC_WMR"))),
                    _fmt(_num(row.get("gt_oracle_AFC_WMR"))),
                    _fmt(_num(row.get("endpoint_fps_Unsupported"))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Confidence Bins", "", "| bin | #samples | avg conf | avg top1 dist | avg effective M | avg mode count | avg slow20 WMR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in confidence_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("confidence_bin", "")),
                    str(row.get("num_samples", "")),
                    _fmt(_num(row.get("avg_retrieval_confidence"))),
                    _fmt(_num(row.get("avg_top1_distance"))),
                    _fmt(_num(row.get("avg_effective_m"))),
                    _fmt(_num(row.get("avg_mode_count"))),
                    _fmt(_num(row.get("avg_slow20_AFC_WMR"))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot.as_posix()}`")
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Use high-confidence cases to show that retrieved futures are visually plausible analogical futures.",
            "- Use GT-oracle-drop cases to explain why GT-centric selection can reduce plausible-mode coverage.",
            "- Use endpoint-FPS cases to show that geometric spread can produce unsupported predictions.",
            "- Keep low-confidence cases as cautionary examples rather than metric failures.",
        ]
    )
    return "\n".join(lines) + "\n"


def _collect_cases(args: argparse.Namespace) -> Tuple[List[CasePayload], Dict[str, Any]]:
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
            cfg_path=args.slow_cfg_path,
            checkpoint_path=args.slow_checkpoint,
        )
    )
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
    afc_bank = build_eth_analogical_future_bank(
        data_root=data_root,
        subset=args.subset,
        train_split=str(args.afc_train_split),
        sample_mode=str(args.sample_mode),
        data_norm=str(args.data_norm),
        rotate=bool(args.rotate),
        rotate_time_frame=int(args.rotate_time_frame),
        normalization_stats=normalization_stats,
        min_agents=int(protocol_settings.min_agents),
        prefer_cache=bool(protocol_settings.prefer_cache),
        max_train_scenes=args.afc_max_train_scenes,
        batch_scenes=int(args.afc_batch_scenes),
        top_m=int(args.afc_top_m),
        eps_values=(float(args.afc_eps),),
    )
    print(
        "[export_afc_exp5_retrieval_cases] "
        f"subset={args.subset} split={args.split} scenes={len(selected_samples)} eval_items={selected_eval_items} "
        f"device={device} afc_bank={afc_bank.bank_size} top_m={args.afc_top_m} eps={args.afc_eps}"
    )

    cases: List[CasePayload] = []
    chunks = list(_iter_chunks(list(enumerate(selected_samples)), int(args.batch_scenes)))
    for chunk_index, chunk_pairs in enumerate(chunks, start=1):
        chunk_scene_indices = [int(scene_index) for scene_index, _sample in chunk_pairs]
        chunk = [sample for _scene_index, sample in chunk_pairs]
        chunk_samples_by_scene_index = {
            int(scene_index): sample
            for scene_index, sample in chunk_pairs
        }
        record_map, _next_eval_item_index = _build_base_per_sample_records(
            samples=chunk,
            global_scene_indices=chunk_scene_indices,
            sample_mode=str(args.sample_mode),
            eval_item_offset=0,
        )
        batch = slow_predictor.build_moflow_batch(chunk, normalization_stats=normalization_stats, as_torch=True)
        slow20 = slow_predictor.predict(batch, return_all_states=False).slow_pred
        pool = _predict_slow_repeated_pool(slow_predictor, batch, pool_k=int(args.slow_pool_k), first_prediction=slow20)
        ground_truth = batch["fut_traj_original_scale"].to(device=device, dtype=torch.float32)
        gt_indices = _oracle_indices(pool, ground_truth, keep_k=int(args.keep_k), metric=str(args.oracle_select_metric))
        fps_indices = _structured_fps_indices(pool[..., -1, :], keep_k=int(args.keep_k))
        afc_indices = _afc_greedy_indices(pool, batch, afc_bank, keep_k=int(args.keep_k), tau=float(args.afc_selection_tau))
        branches = {
            "slow20_pred": slow20,
            "gt_oracle20_pred": _gather_candidates(pool, gt_indices),
            "endpoint_fps20_pred": _gather_candidates(pool, fps_indices),
            "afc_greedy20_pred": _gather_candidates(pool, afc_indices),
        }
        _features, valid, top_indices, top_distances, _finite_counts = afc_bank._query_with_distances(batch)
        proxies_all = afc_bank.futures[top_indices]
        scale = afc_bank.retrieval_scale
        valid_positions = valid.nonzero(as_tuple=False)
        past_rel = batch["past_traj_original_scale"][..., 2:4].detach().cpu().to(torch.float32)
        gt_rel = batch["fut_traj_original_scale"].detach().cpu().to(torch.float32)
        for query_index, position in enumerate(valid_positions):
            batch_index = int(position[0].item())
            agent_axis_index = int(position[1].item())
            record = record_map.get((batch_index, agent_axis_index))
            if record is None:
                raise RuntimeError(
                    "Unable to map AFC query back to source sample: "
                    f"batch_index={batch_index} agent_axis_index={agent_axis_index} "
                    f"chunk_scenes={len(chunk)} sample_mode={args.sample_mode!r}"
                )
            selected_scene_index = int(record.get("selected_scene_index", -1))
            sample = chunk_samples_by_scene_index.get(selected_scene_index)
            if sample is None:
                raise RuntimeError(
                    "Unable to locate source sample for mapped AFC query: "
                    f"selected_scene_index={selected_scene_index} chunk_scene_indices={chunk_scene_indices}"
                )
            source_agent_index = int(record.get("source_agent_index", agent_axis_index))
            meta = dict(record.get("scene_meta") or _scene_meta(sample))
            sample_id = record.get("sample_id") or meta.get("sample_id") or f"{args.subset}_{selected_scene_index}"
            distances = top_distances[query_index].detach().cpu().to(torch.float32)
            proxies = proxies_all[query_index].detach().cpu().to(torch.float32)
            retrieval_weights = _distance_weights(distances, scale=scale)
            confidence = float(torch.exp(-distances[0].clamp_min(0.0) / scale).item()) if int(distances.numel()) > 0 else 0.0
            effective_m = _weighted_entropy_effective_count(retrieval_weights)
            proxy_pairwise = torch.linalg.norm(proxies[:, None, :, :] - proxies[None, :, :, :], dim=-1).mean(dim=-1)
            mode_centers, mode_weights = _weighted_modes_one(proxies, distances, proxy_pairwise, float(args.afc_eps), scale=scale)
            row: Dict[str, Any] = {
                "run_id": args.run_id,
                "dataset": args.subset,
                "split": args.split,
                "sample_index": selected_scene_index,
                "sample_id": sample_id,
                "seq_id": record.get("seq_id") or meta.get("seq_id", ""),
                "frame_id": record.get("frame_id") or meta.get("frame_id", ""),
                "source_file": record.get("source_file") or meta.get("source_file", ""),
                "batch_index": batch_index,
                "agent_axis_index": agent_axis_index,
                "agent_index": source_agent_index,
                "agent_id": _agent_id(sample, source_agent_index),
                "retrieval_confidence": confidence,
                "retrieval_top1_distance": float(distances[0].item()) if int(distances.numel()) > 0 else float("nan"),
                "retrieval_top_m_distance": float(distances.mean().item()) if int(distances.numel()) > 0 else float("nan"),
                "retrieval_effective_m": effective_m,
                "retrieved_mode_count": int(mode_centers.shape[0]),
            }
            plot_predictions: Dict[str, Any] = {}
            supported_masks: Dict[str, Any] = {}
            branch_mode_centers: Dict[str, Any] = {}
            for branch, label in BRANCHES:
                pred = branches[branch].detach().cpu().to(torch.float32)[batch_index, :, agent_axis_index]
                gt_query = gt_rel[batch_index, agent_axis_index]
                metrics = _branch_query_metrics(pred, proxies, distances, gt_query, eps=float(args.afc_eps), scale=scale)
                key = label_key(branch)
                for metric_name in ("ADE_avg", "FDE_avg", "ADE_min", "FDE_min", "AFC_WMR", "AFC_precision", "Unsupported", "AFC_chamfer", "mode_count", "endpoint_spread"):
                    row[f"{key}_{metric_name}"] = metrics[metric_name]
                plot_predictions[branch] = _to_list(pred)
                supported_masks[branch] = metrics["supported_mask"]
                branch_mode_centers[branch] = _to_list(metrics["mode_centers"])
            plot_payload = {
                "past": _to_list(past_rel[batch_index, agent_axis_index]),
                "gt": _to_list(gt_rel[batch_index, agent_axis_index]),
                "proxies": _to_list(proxies),
                "mode_centers": _to_list(mode_centers),
                "mode_weights": _to_list(mode_weights),
                "predictions": plot_predictions,
                "supported_masks": supported_masks,
                "branch_mode_centers": branch_mode_centers,
            }
            cases.append(CasePayload(row=row, plot_payload=plot_payload))
        if chunk_index % 10 == 0 or chunk_index == len(chunks):
            print(f"[export_afc_exp5_retrieval_cases] processed_chunks={chunk_index}/{len(chunks)} cases={len(cases)}")

    meta = {
        "dataset_summary": dataset.summary(),
        "normalization_meta": normalization_meta,
        "afc_bank_size": afc_bank.bank_size,
        "afc_feature_dim": afc_bank.feature_dim,
        "collected_cases": len(cases),
    }
    return cases, meta


def main() -> None:
    args = build_parser().parse_args()
    if int(args.keep_k) <= 0:
        raise SystemExit("--keep-k must be positive")
    if int(args.slow_pool_k) < int(args.keep_k):
        raise SystemExit("--slow-pool-k must be >= --keep-k")
    if int(args.max_cases_per_type) <= 0:
        raise SystemExit("--max-cases-per-type must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases, meta = _collect_cases(args)
    selected = _choose_cases(all_cases, int(args.max_cases_per_type))
    confidence_rows = _confidence_bins(all_cases)
    plots: List[Path] = []
    for index, case in enumerate(selected, start=1):
        safe_type = str(case.row.get("case_type", "case")).replace("/", "_")
        sample_id = str(case.row.get("sample_id", "sample")).replace("/", "_")
        output_base = output_dir / f"case_{index:02d}_{safe_type}_{sample_id}_agent{case.row.get('agent_id', case.row.get('agent_index'))}"
        if _plot_case(case, output_base, eps=float(args.afc_eps)):
            plots.extend([output_base.with_suffix(".png"), output_base.with_suffix(".pdf")])

    _write_csv(output_dir / "afc_exp5_all_cases.csv", [case.row for case in all_cases])
    _write_csv(output_dir / "afc_exp5_selected_cases.csv", [case.row for case in selected])
    _write_csv(output_dir / "afc_exp5_confidence_bins.csv", confidence_rows)
    (output_dir / "afc_exp5_selected_cases.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "args": vars(args),
                "selected_cases": [
                    {"row": case.row, "plot_payload": case.plot_payload}
                    for case in selected
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = _render_summary(args=args, selected=selected, confidence_rows=confidence_rows, plots=plots)
    (output_dir / "afc_exp5_retrieval_case_summary.md").write_text(summary, encoding="utf-8")
    print(f"all_cases_csv={(output_dir / 'afc_exp5_all_cases.csv').as_posix()}")
    print(f"selected_cases_csv={(output_dir / 'afc_exp5_selected_cases.csv').as_posix()}")
    print(f"confidence_bins_csv={(output_dir / 'afc_exp5_confidence_bins.csv').as_posix()}")
    print(f"selected_cases_json={(output_dir / 'afc_exp5_selected_cases.json').as_posix()}")
    print(f"summary_md={(output_dir / 'afc_exp5_retrieval_case_summary.md').as_posix()}")
    for plot in plots:
        print(f"plot={plot.as_posix()}")


if __name__ == "__main__":
    main()
