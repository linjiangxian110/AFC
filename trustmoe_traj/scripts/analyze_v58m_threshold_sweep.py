"""Analyze V58M accept-threshold sweep summaries.

This script reads JSON files produced by
`summarize_v58_slot_quality_scorer.py` and writes a compact CSV/Markdown
report across datasets and thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


METRIC_COLUMNS: Sequence[str] = (
    "mean_dADE_min",
    "mean_dFDE_min",
    "mean_dADE_avg",
    "mean_dFDE_avg",
    "mean_dMissRate",
    "mean_ADE_min",
    "mean_FDE_min",
    "mean_ADE_avg",
    "mean_FDE_avg",
    "mean_MissRate",
    "mean_latency_avg_ms",
    "mean_endpoint_ratio",
    "mean_trajectory_ratio",
    "mean_unique_base_mode_ratio",
    "mean_selected_slot0_ratio",
    "mean_selected_nonzero_ratio",
    "mean_selector_fallback_to_slot0_ratio",
    "mean_accepted_nonzero_hurt_slot0_ade_ratio",
    "mean_accepted_nonzero_hurt_slot0_fde_ratio",
    "mean_missed_oracle_nonzero_ratio",
    "mean_oracle_nonzero_recall_ratio",
    "mean_all_bad_fallback_to_slot0_ratio",
    "mean_all_bad_nonzero_accept_ratio",
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
    matches = sorted(input_root.rglob(f"*{dataset}*summary.json"))
    for item in matches:
        if tag in item.name:
            return item
    return None


def _branch_from_payload(payload: Mapping[str, Any], eval_prefix: str) -> Optional[str]:
    branches = payload.get("meta", {}).get("branches", [])
    if isinstance(branches, list):
        expected = f"{eval_prefix}_20_pred"
        if expected in branches:
            return expected
        for branch in branches:
            if isinstance(branch, str) and branch.startswith(eval_prefix):
                return branch
        for branch in branches:
            if isinstance(branch, str):
                return branch
    return None


def _row_status(row: Mapping[str, Any]) -> str:
    dade_min = _num(row.get("mean_dADE_min"))
    dfde_min = _num(row.get("mean_dFDE_min"))
    dade_avg = _num(row.get("mean_dADE_avg"))
    dfde_avg = _num(row.get("mean_dFDE_avg"))
    min_ok = dade_min is not None and dfde_min is not None and dade_min <= 0.0 and dfde_min <= 0.0
    avg_ok = dade_avg is not None and dfde_avg is not None and dade_avg < 0.0 and dfde_avg < 0.0
    if min_ok and avg_ok:
        return "min+avg gain"
    if min_ok:
        return "min safe"
    if avg_ok:
        return "avg gain only"
    return "not useful"


def _min_penalty(row: Mapping[str, Any]) -> float:
    dade_min = _num(row.get("mean_dADE_min"))
    dfde_min = _num(row.get("mean_dFDE_min"))
    penalty = 0.0
    if dade_min is None:
        penalty += 999.0
    else:
        penalty += max(0.0, dade_min)
    if dfde_min is None:
        penalty += 999.0
    else:
        penalty += max(0.0, dfde_min)
    return float(penalty)


def _avg_gain(row: Mapping[str, Any]) -> float:
    dade_avg = _num(row.get("mean_dADE_avg"))
    dfde_avg = _num(row.get("mean_dFDE_avg"))
    gain = 0.0
    if dade_avg is not None:
        gain += -dade_avg
    if dfde_avg is not None:
        gain += -dfde_avg
    return float(gain)


def _best_test_row(rows: Sequence[Mapping[str, Any]], dataset: str) -> Optional[Mapping[str, Any]]:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("split") == "test"]
    if not candidates:
        return None
    safe = [
        row
        for row in candidates
        if (_num(row.get("mean_dADE_min")) is not None and _num(row.get("mean_dADE_min")) <= 0.0)
        and (_num(row.get("mean_dFDE_min")) is not None and _num(row.get("mean_dFDE_min")) <= 0.0)
    ]
    if safe:
        return sorted(safe, key=lambda row: (-_avg_gain(row), str(row.get("tag"))))[0]
    return sorted(candidates, key=lambda row: (_min_penalty(row), -_avg_gain(row), str(row.get("tag"))))[0]


def _collect_rows(
    *,
    input_root: Path,
    datasets: Sequence[str],
    tags: Sequence[str],
    splits: Sequence[str],
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
                missing.append(f"{dataset}/{tag}")
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            branch = _branch_from_payload(payload, eval_prefix)
            if branch is None:
                missing.append(f"{dataset}/{tag}: no branch in {summary_path.as_posix()}")
                continue
            aggregate = payload.get("aggregate", {})
            for split in splits:
                branch_row = aggregate.get(split, {}).get(branch, {})
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "tag": tag,
                    "threshold": tag.replace("p", "0.", 1) if tag.startswith("p") else tag,
                    "split": split,
                    "branch": branch,
                    "summary_json": summary_path.as_posix(),
                    "available": branch_row.get("available_official_seeds"),
                }
                for column in METRIC_COLUMNS:
                    row[column] = branch_row.get(column)
                row["status"] = _row_status(row)
                row["min_penalty"] = _min_penalty(row)
                row["avg_gain_sum"] = _avg_gain(row)
                rows.append(row)
    return rows, missing


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "tag",
        "threshold",
        "split",
        "available",
        *METRIC_COLUMNS,
        "status",
        "min_penalty",
        "avg_gain_sum",
        "branch",
        "summary_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str], datasets: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# V58M Threshold Sweep Analysis")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item}`")
        lines.append("")
    lines.append("## Test Split Summary")
    lines.append("")
    lines.append(
        "| dataset | threshold | dADE_min | dFDE_min | dADE_avg | dFDE_avg | dMissRate | slot0_ratio | nonzero_ratio | fallback | status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row.get("split") != "test":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("tag", "")),
                    _fmt(row.get("mean_dADE_min"), signed=True),
                    _fmt(row.get("mean_dFDE_min"), signed=True),
                    _fmt(row.get("mean_dADE_avg"), signed=True),
                    _fmt(row.get("mean_dFDE_avg"), signed=True),
                    _fmt(row.get("mean_dMissRate"), signed=True),
                    _fmt(row.get("mean_selected_slot0_ratio")),
                    _fmt(row.get("mean_selected_nonzero_ratio")),
                    _fmt(row.get("mean_selector_fallback_to_slot0_ratio")),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Test Split Quality-Diversity Summary")
    lines.append("")
    lines.append(
        "| dataset | threshold | dADE_avg | dFDE_avg | endpoint_ratio | trajectory_ratio | unique_base_mode_ratio | slot0_ratio | status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row.get("split") != "test":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("tag", "")),
                    _fmt(row.get("mean_dADE_avg"), signed=True),
                    _fmt(row.get("mean_dFDE_avg"), signed=True),
                    _fmt(row.get("mean_endpoint_ratio")),
                    _fmt(row.get("mean_trajectory_ratio")),
                    _fmt(row.get("mean_unique_base_mode_ratio")),
                    _fmt(row.get("mean_selected_slot0_ratio")),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Diversity Interpretation")
    lines.append("")
    lines.append("- `endpoint_ratio` and `trajectory_ratio` are measured relative to `slow_pred`; values near 1 mean diversity is preserved.")
    lines.append("- Values far below 1 suggest mode collapse or diversity shrinkage; values above 1 mean the selected set is more spread than slow.")
    lines.append("- `unique_base_mode_ratio` checks whether final K predictions still cover distinct original base modes.")
    lines.append("")
    lines.append("## Recommended Test Threshold")
    lines.append("")
    lines.append("| dataset | picked | reason | dADE_min | dFDE_min | dADE_avg | dFDE_avg |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for dataset in datasets:
        best = _best_test_row(rows, dataset)
        if best is None:
            lines.append(f"| {dataset} |  | missing |  |  |  |  |")
            continue
        safe = _min_penalty(best) == 0.0
        reason = "min safe, maximize avg gain" if safe else "lowest min penalty, then avg gain"
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    str(best.get("tag", "")),
                    reason,
                    _fmt(best.get("mean_dADE_min"), signed=True),
                    _fmt(best.get("mean_dFDE_min"), signed=True),
                    _fmt(best.get("mean_dADE_avg"), signed=True),
                    _fmt(best.get("mean_dFDE_avg"), signed=True),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append("- `min+avg gain`: both `dADE_min` and `dFDE_min` are non-positive, and both avg deltas improve.")
    lines.append("- `avg gain only`: avg improves but at least one min metric is hurt.")
    lines.append("- If no threshold is min-safe, use the threshold with the smallest positive min penalty as the least harmful setting.")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze V58M threshold sweep summaries.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="eth,hotel,univ,zara1,zara2")
    parser.add_argument("--threshold-tags", type=str, default="p95,p97,p99")
    parser.add_argument("--splits", type=str, default="val,test")
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
    datasets = _split_items(args.datasets)
    tags = _split_items(args.threshold_tags)
    splits = _split_items(args.splits)
    rows, missing = _collect_rows(
        input_root=input_root,
        datasets=datasets,
        tags=tags,
        splits=splits,
        run_prefix_template=str(args.run_prefix_template),
        eval_prefix_template=str(args.eval_prefix_template),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v58m_threshold_sweep_summary.csv"
    md_path = output_dir / "v58m_threshold_sweep_summary.md"
    json_path = output_dir / "v58m_threshold_sweep_summary.json"
    _write_csv(csv_path, rows)
    md_path.write_text(_render_markdown(rows, missing, datasets), encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": rows, "missing": list(missing)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"analysis_csv={csv_path.as_posix()}")
    print(f"analysis_md={md_path.as_posix()}")
    print(f"analysis_json={json_path.as_posix()}")
    if missing:
        print("missing_inputs=" + ",".join(missing))


if __name__ == "__main__":
    main()
