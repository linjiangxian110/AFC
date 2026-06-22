"""Summarize V58-K slot quality scorer official eval JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
BASE_AUX_METRICS: Sequence[str] = (
    "latency_avg_ms",
    "delta_l2_mean",
    "endpoint_ratio",
    "trajectory_ratio",
    "unique_base_mode_ratio",
    "endpoint_cluster_count_eps05",
    "endpoint_cluster_count_ratio_eps05",
    "endpoint_cluster_entropy_eps05",
    "endpoint_cluster_entropy_ratio_eps05",
    "endpoint_cluster_count_eps10",
    "endpoint_cluster_count_ratio_eps10",
    "endpoint_cluster_entropy_eps10",
    "endpoint_cluster_entropy_ratio_eps10",
    "trajectory_cluster_count_eps05",
    "trajectory_cluster_count_ratio_eps05",
    "trajectory_cluster_entropy_eps05",
    "trajectory_cluster_entropy_ratio_eps05",
    "trajectory_cluster_count_eps10",
    "trajectory_cluster_count_ratio_eps10",
    "trajectory_cluster_entropy_eps10",
    "trajectory_cluster_entropy_ratio_eps10",
    "afc_bank_size",
    "afc_top_m",
    "afc_valid_query_count",
    "afc_proxy_to_pred_ade",
    "afc_pred_to_proxy_ade",
    "afc_chamfer",
    "afc_recall_eps05",
    "afc_precision_eps05",
    "afc_mode_count_eps05",
    "afc_mode_coverage_eps05",
    "afc_recall_eps10",
    "afc_precision_eps10",
    "afc_mode_count_eps10",
    "afc_mode_coverage_eps10",
    "selected_slot_mean",
    "selected_slot0_ratio",
    "raw_selected_slot_mean",
    "raw_selected_slot0_ratio",
    "selector_fallback_to_slot0_ratio",
    "front_oracle_slot_accuracy",
    "selected_nonzero_ratio",
    "raw_selected_nonzero_ratio",
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
    "selector_raw_prob_margin_mean",
    "selector_raw_rank_score_mean",
    "selector_accept_prob_threshold",
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
    "anchor_qd_corrected_ratio",
    "anchor_qd_base_fallback_ratio",
    "anchor_qd_quality_prob_mean",
    "anchor_qd_candidate_afc_support_mean",
    "anchor_qd_base_afc_support_mean",
    "anchor_qd_combined_margin_mean",
    "anchor_qd_residual_l2_mean",
    "anchor_qd_role_support_mean",
    "anchor_qd_spread_floor_reject_ratio",
    "anchor_qd_alpha",
    "anchor_qd_beta",
    "anchor_qd_residual_penalty",
    "anchor_qd_margin",
    "anchor_qd_tau",
    "anchor_qd_anchor_k",
    "anchor_qd_anchor_min_prob",
    "anchor_qd_diversity_min_prob",
    "anchor_qd_base_quality",
    "anchor_qd_max_residual_l2",
    "anchor_qd_selection_mode_role_transport",
    "anchor_qd_mean_dade_vs_slow",
    "anchor_qd_mean_dfde_vs_slow",
    "anchor_qd_mean_dscore_vs_slow",
    "anchor_qd_anchor_corrected_ratio",
    "anchor_qd_diversity_corrected_ratio",
    "anchor_qd_corrected_improves_slow_score_ratio",
    "anchor_qd_corrected_hurts_slow_score_ratio",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V58-K slot quality scorer eval outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--input-root", type=str, default=None)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--eval-file-prefix", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--branches", type=str, default="")
    parser.add_argument("--diagnostic-prefix", type=str, default="v58k")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item]


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


def _metric(metrics: Mapping[str, Any], branch: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{branch}_{name}"))


def _run_id(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{int(seed)}"


def _discover_aux(metrics: Mapping[str, Any], branches: Sequence[str]) -> List[str]:
    aux = set(BASE_AUX_METRICS)
    metric_suffixes = set(METRICS)
    for branch in branches:
        prefix = f"{branch}_"
        for key in metrics.keys():
            if not str(key).startswith(prefix):
                continue
            suffix = str(key)[len(prefix) :]
            if suffix not in metric_suffixes:
                aux.add(suffix)
    return sorted(aux)


def _seed_row(
    *,
    input_root: Path,
    run_prefix: str,
    eval_file_prefix: str,
    seed: int,
    splits: Sequence[str],
    requested_branches: Sequence[str],
) -> Dict[str, Any]:
    run_id = _run_id(run_prefix, int(seed))
    row: Dict[str, Any] = {"run_id": run_id, "seed": int(seed), "missing_files": []}
    discovered: List[str] = []
    aux_names = set(BASE_AUX_METRICS)
    for split in splits:
        path = input_root / run_id / f"{eval_file_prefix}_{split}.json"
        payload = _load_json(path)
        if payload is None:
            row["missing_files"].append(path.as_posix())
            row[f"official_{split}"] = {"slow": {}}
            continue
        raw_metrics = payload.get("metrics", {})
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        branches = list(requested_branches) if requested_branches else list(payload.get("deterministic_branches", []))
        if not discovered:
            discovered = branches
        aux_names.update(_discover_aux(metrics, branches))
        split_row: Dict[str, Any] = {}
        for branch in branches:
            branch_row: Dict[str, Any] = {}
            for metric in METRICS:
                value = _metric(metrics, branch, metric)
                slow = _metric(metrics, "slow_pred", metric)
                branch_row[metric] = value
                branch_row[f"d{metric}"] = None if value is None or slow is None else value - slow
            for aux in sorted(aux_names):
                value = _num(metrics.get(f"{branch}_{aux}"))
                if value is not None:
                    branch_row[aux] = value
            split_row[branch] = branch_row
        split_row["slow"] = {metric: _metric(metrics, "slow_pred", metric) for metric in METRICS}
        split_row["slow"]["latency_avg_ms"] = _num(metrics.get("slow_pred_latency_avg_ms"))
        row[f"official_{split}"] = split_row
    row["branches"] = list(discovered or requested_branches)
    row["aux_names"] = sorted(aux_names)
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str], branches: Sequence[str], aux_names: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in splits:
        split_key = f"official_{split}"
        split_result: Dict[str, Any] = {}
        for branch in branches:
            branch_result: Dict[str, Any] = {
                "available_official_seeds": sum(
                    1 for row in rows if _num(row.get(split_key, {}).get(branch, {}).get("dFDE_min")) is not None
                )
            }
            for metric in METRICS:
                branch_result[f"mean_d{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(f"d{metric}") for row in rows
                )
                branch_result[f"mean_{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(metric) for row in rows
                )
            for aux in aux_names:
                branch_result[f"mean_{aux}"] = _mean(row.get(split_key, {}).get(branch, {}).get(aux) for row in rows)
            split_result[branch] = branch_result
        result[split] = split_result
    return result


def _render(
    *,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    splits: Sequence[str],
    branches: Sequence[str],
    aux_names: Sequence[str],
) -> str:
    lines: List[str] = []
    focus_aux = [
        "latency_avg_ms",
        "endpoint_ratio",
        "trajectory_ratio",
        "unique_base_mode_ratio",
        "endpoint_cluster_count_eps05",
        "endpoint_cluster_count_ratio_eps05",
        "endpoint_cluster_entropy_eps05",
        "endpoint_cluster_entropy_ratio_eps05",
        "endpoint_cluster_count_eps10",
        "endpoint_cluster_count_ratio_eps10",
        "endpoint_cluster_entropy_eps10",
        "endpoint_cluster_entropy_ratio_eps10",
        "trajectory_cluster_count_eps05",
        "trajectory_cluster_count_ratio_eps05",
        "trajectory_cluster_entropy_eps05",
        "trajectory_cluster_entropy_ratio_eps05",
        "trajectory_cluster_count_eps10",
        "trajectory_cluster_count_ratio_eps10",
        "trajectory_cluster_entropy_eps10",
        "trajectory_cluster_entropy_ratio_eps10",
        "afc_bank_size",
        "afc_top_m",
        "afc_valid_query_count",
        "afc_proxy_to_pred_ade",
        "afc_pred_to_proxy_ade",
        "afc_chamfer",
        "afc_recall_eps05",
        "afc_precision_eps05",
        "afc_mode_count_eps05",
        "afc_mode_coverage_eps05",
        "afc_recall_eps10",
        "afc_precision_eps10",
        "afc_mode_count_eps10",
        "afc_mode_coverage_eps10",
        "selected_slot_mean",
        "selected_slot0_ratio",
        "selected_nonzero_ratio",
        "raw_selected_nonzero_ratio",
        "selector_fallback_to_slot0_ratio",
        "selector_raw_prob_mean",
        "selector_raw_prob_margin_mean",
        "selector_raw_rank_score_mean",
        "accepted_nonzero_hurt_slot0_ade_ratio",
        "accepted_nonzero_hurt_slot0_fde_ratio",
        "accepted_nonzero_better_slot0_ade_ratio",
        "accepted_nonzero_better_slot0_fde_ratio",
        "missed_oracle_nonzero_ratio",
        "oracle_nonzero_recall_ratio",
        "all_bad_fallback_to_slot0_ratio",
        "all_bad_nonzero_accept_ratio",
        "anchor_qd_corrected_ratio",
        "anchor_qd_base_fallback_ratio",
        "anchor_qd_quality_prob_mean",
        "anchor_qd_candidate_afc_support_mean",
        "anchor_qd_base_afc_support_mean",
        "anchor_qd_combined_margin_mean",
        "anchor_qd_residual_l2_mean",
        "anchor_qd_role_support_mean",
        "anchor_qd_spread_floor_reject_ratio",
        "anchor_qd_anchor_corrected_ratio",
        "anchor_qd_diversity_corrected_ratio",
        "anchor_qd_corrected_improves_slow_score_ratio",
        "anchor_qd_corrected_hurts_slow_score_ratio",
        "anchor_qd_mean_dade_vs_slow",
        "anchor_qd_mean_dfde_vs_slow",
        "anchor_qd_mean_dscore_vs_slow",
    ]
    for row in rows:
        lines.append(f"===== {row['run_id']} =====")
        if row.get("missing_files"):
            lines.append("missing summary inputs:")
            for path in row["missing_files"]:
                lines.append(f"  {path}")
        for split in splits:
            official = row.get(f"official_{split}", {})
            lines.append("")
            lines.append(f"-- official {split} V58K quality - Slow --")
            for branch in branches:
                branch_row = official.get(branch, {})
                lines.append(f"{branch}:")
                for metric in METRICS:
                    lines.append(
                        f"  d{metric}: {_fmt(branch_row.get(f'd{metric}'), signed=True)}  "
                        f"value={_fmt(branch_row.get(metric))}  slow={_fmt(official.get('slow', {}).get(metric))}"
                    )
                for aux in focus_aux:
                    if branch_row.get(aux) is not None:
                        lines.append(f"  {aux}: {_fmt(branch_row.get(aux))}")

    lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        lines.append("")
        lines.append(f"-- {split} --")
        split_agg = aggregate.get(split, {})
        for branch in branches:
            mean = split_agg.get(branch, {})
            lines.append(f"{branch}: available={int(mean.get('available_official_seeds') or 0)}/{len(rows)}")
            for metric in METRICS:
                lines.append(
                    f"  mean d{metric}: {_fmt(mean.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(mean.get(f'mean_{metric}'))}"
                )
            for aux in focus_aux:
                value = mean.get(f"mean_{aux}")
                if value is not None:
                    lines.append(f"  mean {aux}: {_fmt(value)}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_root = (
        Path(args.input_root).expanduser().resolve()
        if args.input_root
        else project_root / "trustmoe_traj" / "analysis" / "eval_results"
    )
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    requested_branches = _split_items(args.branches)
    rows = [
        _seed_row(
            input_root=input_root,
            run_prefix=str(args.run_prefix),
            eval_file_prefix=str(args.eval_file_prefix),
            seed=seed,
            splits=splits,
            requested_branches=requested_branches,
        )
        for seed in seeds
    ]
    branches: List[str] = list(requested_branches)
    if not branches:
        seen = set()
        for row in rows:
            for branch in row.get("branches", []):
                if branch not in seen:
                    seen.add(branch)
                    branches.append(branch)
    aux_names = sorted({aux for row in rows for aux in row.get("aux_names", [])})
    aggregate = _aggregate(rows, splits, branches, aux_names)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_v58_slot_quality_scorer",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": str(args.run_prefix),
            "eval_file_prefix": str(args.eval_file_prefix),
            "seeds": seeds,
            "splits": splits,
            "branches": branches,
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
    rendered = _render(rows=rows, aggregate=aggregate, splits=splits, branches=branches, aux_names=aux_names)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
