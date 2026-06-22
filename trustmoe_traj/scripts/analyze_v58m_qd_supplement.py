"""Analyze V58M quality-diversity supplement summaries.

The script reads summary JSONs produced by `summarize_v58_slot_quality_scorer`
and builds a compact table comparing:

* V58M conservative selector;
* learned quality-only raw selector without accept-threshold fallback;
* learned quality-only global top-20 selector without base-mode preservation;
* GT-oriented per-base oracle;
* GT-oriented global oracle, a deliberately collapsed/reference upper bound for
  observed-future precision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


METRICS: Sequence[str] = (
    "mean_dADE_min",
    "mean_dFDE_min",
    "mean_dADE_avg",
    "mean_dFDE_avg",
    "mean_dMissRate",
    "mean_endpoint_ratio",
    "mean_trajectory_ratio",
    "mean_unique_base_mode_ratio",
    "mean_endpoint_cluster_count_ratio_eps05",
    "mean_endpoint_cluster_entropy_ratio_eps05",
    "mean_endpoint_cluster_count_ratio_eps10",
    "mean_endpoint_cluster_entropy_ratio_eps10",
    "mean_trajectory_cluster_count_ratio_eps05",
    "mean_trajectory_cluster_entropy_ratio_eps05",
    "mean_trajectory_cluster_count_ratio_eps10",
    "mean_trajectory_cluster_entropy_ratio_eps10",
    "mean_selected_slot0_ratio",
    "mean_selected_nonzero_ratio",
)


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return ""
    prefix = "+" if signed and numeric >= 0.0 else ""
    return f"{prefix}{numeric:.6f}"


def _render_template(template: str, *, dataset: str, tag: str) -> str:
    return template.replace("{dataset}", dataset).replace("{ds}", dataset).replace("{tag}", tag)


def _find_summary(
    input_root: Path,
    *,
    dataset: str,
    tag: str,
    run_prefix_template: str,
    eval_prefix_template: str,
) -> Optional[Path]:
    run_prefix = _render_template(run_prefix_template, dataset=dataset, tag=tag)
    eval_prefix = _render_template(eval_prefix_template, dataset=dataset, tag=tag)
    exact = input_root / f"{run_prefix}_{eval_prefix}_summary.json"
    if exact.exists():
        return exact
    matches = sorted(input_root.rglob(f"*{dataset}*{tag}*summary.json"))
    if matches:
        return matches[-1]
    matches = sorted(input_root.rglob(f"*{tag}*summary.json"))
    for item in matches:
        if eval_prefix in item.name:
            return item
    return None


def _branch_names(eval_prefix: str) -> Dict[str, str]:
    return {
        "v58m_conservative": f"{eval_prefix}_20_pred",
        "anchor_qd": f"{eval_prefix}_anchor_qd20_pred",
        "quality_only_raw": f"{eval_prefix}_raw_quality20_pred",
        "quality_only_global_raw": f"{eval_prefix}_raw_quality_global20_pred",
        "per_base_oracle_gt": f"{eval_prefix}_per_base_oracle20_pred",
        "global_oracle_gt": f"{eval_prefix}_global_oracle20_pred",
        "front3_per_base_oracle_gt": f"{eval_prefix}_slots0to2_oracle20_pred",
        "front3_global_oracle_gt": f"{eval_prefix}_slots0to2_global_oracle20_pred",
        "full_pool_reference": f"{eval_prefix}_full160_pred",
    }


def _collect_rows(
    *,
    input_root: Path,
    datasets: Sequence[str],
    tags: Sequence[str],
    split: str,
    run_prefix_template: str,
    eval_prefix_template: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        for tag in tags:
            eval_prefix = _render_template(eval_prefix_template, dataset=dataset, tag=tag)
            summary_path = _find_summary(
                input_root,
                dataset=dataset,
                tag=tag,
                run_prefix_template=run_prefix_template,
                eval_prefix_template=eval_prefix_template,
            )
            if summary_path is None:
                missing.append(f"{dataset}/{tag}: missing summary")
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            aggregate = payload.get("aggregate", {})
            split_rows = aggregate.get(split, {})
            if not isinstance(split_rows, Mapping):
                missing.append(f"{dataset}/{tag}: missing split={split} in {summary_path.as_posix()}")
                continue
            for label, branch in _branch_names(eval_prefix).items():
                branch_row = split_rows.get(branch)
                if not isinstance(branch_row, Mapping):
                    if label in {"v58m_conservative", "quality_only_raw", "quality_only_global_raw", "global_oracle_gt"}:
                        missing.append(f"{dataset}/{tag}/{label}: missing branch={branch}")
                    continue
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "tag": tag,
                    "split": split,
                    "variant": label,
                    "branch": branch,
                    "available": branch_row.get("available_official_seeds"),
                    "summary_json": summary_path.as_posix(),
                }
                for metric in METRICS:
                    row[metric] = branch_row.get(metric)
                rows.append(row)
    return rows, missing


def _variant_order(variant: Any) -> int:
    order = {
        "v58m_conservative": 0,
        "anchor_qd": 1,
        "quality_only_raw": 2,
        "quality_only_global_raw": 3,
        "front3_global_oracle_gt": 4,
        "global_oracle_gt": 5,
        "front3_per_base_oracle_gt": 6,
        "per_base_oracle_gt": 7,
        "full_pool_reference": 8,
    }
    return order.get(str(variant), 99)


def _row_index(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row.get("dataset", "")), str(row.get("tag", "")), str(row.get("variant", ""))): row
        for row in rows
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any], metric: str) -> Optional[float]:
    left_value = _num(left.get(metric))
    right_value = _num(right.get(metric))
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _diversity_guard_status(row: Mapping[str, Any]) -> str:
    avg_ok = (_num(row.get("mean_dADE_avg")) or 0.0) < 0.0 and (_num(row.get("mean_dFDE_avg")) or 0.0) < 0.0
    base_ok = (_num(row.get("mean_unique_base_mode_ratio")) or 0.0) >= 0.99
    ep_cluster = _num(row.get("mean_endpoint_cluster_count_ratio_eps05"))
    traj_cluster = _num(row.get("mean_trajectory_cluster_count_ratio_eps10"))
    cluster_values = [value for value in (ep_cluster, traj_cluster) if value is not None]
    cluster_ok = bool(cluster_values) and min(cluster_values) >= 0.90
    spread_values = [
        value
        for value in (
            _num(row.get("mean_endpoint_ratio")),
            _num(row.get("mean_trajectory_ratio")),
        )
        if value is not None
    ]
    spread_shrink = bool(spread_values) and min(spread_values) < 0.95
    if avg_ok and base_ok and cluster_ok and spread_shrink:
        return "avg gain, base/cluster preserved, spread shrinks"
    if avg_ok and base_ok and cluster_ok:
        return "avg gain, base/cluster preserved"
    if avg_ok and base_ok:
        return "avg gain, base modes preserved"
    if avg_ok:
        return "avg gain only"
    return "not quality-positive"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "tag",
        "split",
        "variant",
        "available",
        *METRICS,
        "branch",
        "summary_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# V58M Quality-Diversity Supplement")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item}`")
        lines.append("")
    lines.append("## Quality-Diversity Table")
    lines.append("")
    lines.append(
        "| dataset | threshold | variant | dADE_avg | dFDE_avg | dADE_min | dFDE_min | endpoint_ratio | trajectory_ratio | base_mode_ratio | ep_cluster_ratio@0.5 | ep_entropy_ratio@0.5 | traj_cluster_ratio@1.0 | traj_entropy_ratio@1.0 | slot0_ratio | nonzero_ratio | status |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (str(item.get("dataset")), str(item.get("tag")), _variant_order(item.get("variant")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("tag", "")),
                    str(row.get("variant", "")),
                    _fmt(row.get("mean_dADE_avg"), signed=True),
                    _fmt(row.get("mean_dFDE_avg"), signed=True),
                    _fmt(row.get("mean_dADE_min"), signed=True),
                    _fmt(row.get("mean_dFDE_min"), signed=True),
                    _fmt(row.get("mean_endpoint_ratio")),
                    _fmt(row.get("mean_trajectory_ratio")),
                    _fmt(row.get("mean_unique_base_mode_ratio")),
                    _fmt(row.get("mean_endpoint_cluster_count_ratio_eps05")),
                    _fmt(row.get("mean_endpoint_cluster_entropy_ratio_eps05")),
                    _fmt(row.get("mean_trajectory_cluster_count_ratio_eps10")),
                    _fmt(row.get("mean_trajectory_cluster_entropy_ratio_eps10")),
                    _fmt(row.get("mean_selected_slot0_ratio")),
                    _fmt(row.get("mean_selected_nonzero_ratio")),
                    _diversity_guard_status(row),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Collapsed Baseline Check")
    lines.append("")
    lines.append(
        "| dataset | threshold | collapsed baseline | baseline dADE_avg | baseline dFDE_avg | avg better than V58M | base-mode drop vs V58M | ep-cluster drop@0.5 | traj-cluster drop@1.0 | endpoint spread drop | trajectory spread drop |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    indexed = _row_index(rows)
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    tags = sorted({str(row.get("tag", "")) for row in rows if row.get("tag")})
    collapsed_variants = ("quality_only_global_raw", "front3_global_oracle_gt", "global_oracle_gt")
    for dataset in datasets:
        for tag in tags:
            v58m = indexed.get((dataset, tag, "v58m_conservative"))
            if v58m is None:
                continue
            for variant in collapsed_variants:
                baseline = indexed.get((dataset, tag, variant))
                if baseline is None:
                    continue
                avg_better_values = [
                    _metric_delta(v58m, baseline, "mean_dADE_avg"),
                    _metric_delta(v58m, baseline, "mean_dFDE_avg"),
                ]
                avg_better = [value for value in avg_better_values if value is not None]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dataset,
                            tag,
                            variant,
                            _fmt(baseline.get("mean_dADE_avg"), signed=True),
                            _fmt(baseline.get("mean_dFDE_avg"), signed=True),
                            _fmt(sum(avg_better) / len(avg_better) if avg_better else None, signed=True),
                            _fmt(_metric_delta(v58m, baseline, "mean_unique_base_mode_ratio"), signed=True),
                            _fmt(_metric_delta(v58m, baseline, "mean_endpoint_cluster_count_ratio_eps05"), signed=True),
                            _fmt(_metric_delta(v58m, baseline, "mean_trajectory_cluster_count_ratio_eps10"), signed=True),
                            _fmt(_metric_delta(v58m, baseline, "mean_endpoint_ratio"), signed=True),
                            _fmt(_metric_delta(v58m, baseline, "mean_trajectory_ratio"), signed=True),
                        ]
                    )
                    + " |"
                )
    lines.append("")
    lines.append("Positive values in the diversity-drop columns mean V58M preserves more diversity than the collapsed baseline.")
    lines.append("")
    lines.append("## V58M Verdict")
    lines.append("")
    lines.append("| dataset | threshold | dADE_avg | dFDE_avg | base_mode_ratio | ep_cluster_ratio@0.5 | traj_cluster_ratio@1.0 | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (str(item.get("dataset")), str(item.get("tag")))):
        if str(row.get("variant")) != "v58m_conservative":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("tag", "")),
                    _fmt(row.get("mean_dADE_avg"), signed=True),
                    _fmt(row.get("mean_dFDE_avg"), signed=True),
                    _fmt(row.get("mean_unique_base_mode_ratio")),
                    _fmt(row.get("mean_endpoint_cluster_count_ratio_eps05")),
                    _fmt(row.get("mean_trajectory_cluster_count_ratio_eps10")),
                    _diversity_guard_status(row),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Reading Rule")
    lines.append("")
    lines.append("- `quality_only_raw` is the learned scorer without conservative accept-threshold fallback.")
    lines.append("- `quality_only_global_raw` is the learned scorer's global top-20 over the front candidate pool; it intentionally removes base-mode preservation.")
    lines.append("- `global_oracle_gt` and `front3_global_oracle_gt` are GT-oriented collapsed references; they are not deployable methods.")
    lines.append("- A useful V58M story needs better avgADE/avgFDE than slow while keeping base-mode and cluster ratios much closer to 1 than GT-oriented global oracle.")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze V58M quality-diversity supplement summaries.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="eth,hotel,zara1")
    parser.add_argument("--threshold-tags", type=str, default="p95,p97,p99")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--run-prefix-template",
        type=str,
        default="20260601_ciisr_{dataset}_v58m_front3_pareto_adefde_rot6",
    )
    parser.add_argument(
        "--eval-prefix-template",
        type=str,
        default="v58m_{dataset}_front3_pareto_rot6_{tag}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "analysis"
    rows, missing = _collect_rows(
        input_root=input_root,
        datasets=_split_items(args.datasets),
        tags=_split_items(args.threshold_tags),
        split=str(args.split),
        run_prefix_template=str(args.run_prefix_template),
        eval_prefix_template=str(args.eval_prefix_template),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v58m_qd_supplement_summary.csv"
    md_path = output_dir / "v58m_qd_supplement_summary.md"
    json_path = output_dir / "v58m_qd_supplement_summary.json"
    _write_csv(csv_path, rows)
    md_path.write_text(_render_markdown(rows, missing), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": rows, "missing": missing}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"analysis_csv={csv_path.as_posix()}")
    print(f"analysis_md={md_path.as_posix()}")
    print(f"analysis_json={json_path.as_posix()}")
    if missing:
        print("missing_inputs=" + ",".join(missing))


if __name__ == "__main__":
    main()
