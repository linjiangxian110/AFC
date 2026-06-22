"""Summarize AFC experiment-1 multi-seed and epsilon stability.

This script consumes ``diagnose_headroom_analysis`` JSON files and produces a
compact claim-facing summary for the AFC metric-distinguishability experiment.
It keeps all deltas relative to the original ``slow20_pred`` teacher output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_BRANCHES: Sequence[str] = (
    "slow20_pred",
    "cv_linear20_pred",
    "random_spread_s1p5_pred",
    "random_spread_s3_pred",
    "random_spread_s5_pred",
    "random_spread_s8_pred",
    "slow200_full_pred",
    "slow200_random20_mean10_pred",
    "slow200_endpoint_fps20_pred",
    "slow200_gt_oracle20_pred",
    "slow200_afc_greedy20_pred",
)
EPS_LABELS: Sequence[Tuple[str, str]] = (("0.3", "eps03"), ("0.5", "eps05"), ("1.0", "eps10"))
DELTA_METRICS: Sequence[str] = (
    "ADE_avg",
    "FDE_avg",
    "ADE_min",
    "FDE_min",
    "afc_chamfer",
    "afc_weighted_mode_recall_eps03",
    "afc_mode_precision_eps03",
    "afc_unsupported_ratio_eps03",
    "afc_weighted_mode_recall_eps05",
    "afc_mode_precision_eps05",
    "afc_unsupported_ratio_eps05",
    "afc_weighted_mode_recall_eps10",
    "afc_mode_precision_eps10",
    "afc_unsupported_ratio_eps10",
)
ABS_METRICS: Sequence[str] = ("endpoint_ratio", "trajectory_ratio")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AFC experiment-1 multi-seed stability.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="eth,hotel,zara1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--branches", type=str, default=",".join(DEFAULT_BRANCHES))
    parser.add_argument(
        "--file-template",
        type=str,
        default="{input_root}/{run_id}_{dataset}_seed{seed}/{dataset}_{split}_headroom.json",
    )
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    if len(kept) <= 1:
        return 0.0 if kept else None
    mean = sum(kept) / len(kept)
    return math.sqrt(sum((item - mean) ** 2 for item in kept) / (len(kept) - 1))


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _fmt_pm(mean: Optional[float], std: Optional[float], *, signed: bool = True) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return _fmt(mean, signed=signed)
    return f"{_fmt(mean, signed=signed)}+-{std:.6f}"


def _branch_metric(metrics: Mapping[str, Any], branch: str, metric: str) -> Optional[float]:
    return _num(metrics.get(f"{branch}_{metric}"))


def _load_records(
    *,
    input_root: Path,
    run_id: str,
    datasets: Sequence[str],
    seeds: Sequence[str],
    splits: Sequence[str],
    branches: Sequence[str],
    file_template: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        for seed in seeds:
            for split in splits:
                raw_path = file_template.format(
                    input_root=input_root.as_posix(),
                    run_id=run_id,
                    dataset=dataset,
                    seed=seed,
                    split=split,
                )
                path = Path(raw_path)
                if not path.exists():
                    missing.append(path.as_posix())
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                available = set(payload.get("branches", []))
                for branch in branches:
                    if branch not in available:
                        continue
                    row: Dict[str, Any] = {
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "branch": branch,
                        "source_json": path.as_posix(),
                    }
                    for metric in DELTA_METRICS:
                        value = _branch_metric(metrics, branch, metric)
                        slow = _branch_metric(metrics, "slow20_pred", metric)
                        row[f"d{metric}"] = None if value is None or slow is None else value - slow
                    for metric in ABS_METRICS:
                        row[metric] = _branch_metric(metrics, branch, metric)
                    records.append(row)
    return records, missing


def _aggregate(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["dataset"]), str(row["split"]), str(row["branch"]))].append(row)
    rows: List[Dict[str, Any]] = []
    for (dataset, split, branch), items in sorted(groups.items()):
        out: Dict[str, Any] = {"dataset": dataset, "split": split, "branch": branch, "n": len(items)}
        for metric in DELTA_METRICS:
            key = f"d{metric}"
            values = [item.get(key) for item in items]
            out[f"mean_{key}"] = _mean(values)
            out[f"std_{key}"] = _std(values)
            out[f"neg_{key}"] = sum(1 for value in values if value is not None and float(value) < 0)
            out[f"pos_{key}"] = sum(1 for value in values if value is not None and float(value) > 0)
        for metric in ABS_METRICS:
            values = [item.get(metric) for item in items]
            out[f"mean_{metric}"] = _mean(values)
            out[f"std_{metric}"] = _std(values)
            out[f"gt1_{metric}"] = sum(1 for value in values if value is not None and float(value) > 1.0)
            out[f"lt1_{metric}"] = sum(1 for value in values if value is not None and float(value) < 1.0)
        rows.append(out)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["dataset", "split", "branch", "n"]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _find(rows: Sequence[Mapping[str, Any]], dataset: str, branch: str, split: str = "test") -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("dataset")) == dataset and str(row.get("split")) == split and str(row.get("branch")) == branch:
            return row
    return None


def _render_eps_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("## Eps Stability Compact Table")
    lines.append("")
    lines.append(
        "| dataset | branch | n | dWMR@0.3 | dWMR@0.5 | dWMR@1.0 | "
        "dPrec@0.3 | dPrec@0.5 | dPrec@1.0 | dUnsup@0.3 | dUnsup@0.5 | dUnsup@1.0 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("branch", "")),
                    str(row.get("n", "")),
                    *(
                        _fmt_pm(row.get(f"mean_dafc_weighted_mode_recall_{label}"), row.get(f"std_dafc_weighted_mode_recall_{label}"))
                        for _eps, label in EPS_LABELS
                    ),
                    *(
                        _fmt_pm(row.get(f"mean_dafc_mode_precision_{label}"), row.get(f"std_dafc_mode_precision_{label}"))
                        for _eps, label in EPS_LABELS
                    ),
                    *(
                        _fmt_pm(row.get(f"mean_dafc_unsupported_ratio_{label}"), row.get(f"std_dafc_unsupported_ratio_{label}"))
                        for _eps, label in EPS_LABELS
                    ),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_eps05_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("## Three-Seed Summary at eps=0.5")
    lines.append("")
    lines.append(
        "| dataset | branch | n | dADE_avg | dFDE_avg | dWMR@0.5 | dPrec@0.5 | dUnsup@0.5 | "
        "dChamfer | endpoint_ratio | trajectory_ratio |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("branch", "")),
                    str(row.get("n", "")),
                    _fmt_pm(row.get("mean_dADE_avg"), row.get("std_dADE_avg")),
                    _fmt_pm(row.get("mean_dFDE_avg"), row.get("std_dFDE_avg")),
                    _fmt_pm(row.get("mean_dafc_weighted_mode_recall_eps05"), row.get("std_dafc_weighted_mode_recall_eps05")),
                    _fmt_pm(row.get("mean_dafc_mode_precision_eps05"), row.get("std_dafc_mode_precision_eps05")),
                    _fmt_pm(row.get("mean_dafc_unsupported_ratio_eps05"), row.get("std_dafc_unsupported_ratio_eps05")),
                    _fmt_pm(row.get("mean_dafc_chamfer"), row.get("std_dafc_chamfer")),
                    _fmt_pm(row.get("mean_endpoint_ratio"), row.get("std_endpoint_ratio"), signed=False),
                    _fmt_pm(row.get("mean_trajectory_ratio"), row.get("std_trajectory_ratio"), signed=False),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_claim_checks(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> List[str]:
    lines: List[str] = []
    lines.append("## Claim Checks")
    lines.append("")
    checks = [
        ("GT oracle hurts WMR while improving ADE", "slow200_gt_oracle20_pred"),
        ("Fake geometric diversity hurts AFC support", "random_spread_s3_pred"),
        ("More samples have little WMR gain", "slow200_full_pred"),
        ("Random subsampling does not improve WMR", "slow200_random20_mean10_pred"),
        ("Endpoint FPS increases spread but hurts precision", "slow200_endpoint_fps20_pred"),
    ]
    for title, branch in checks:
        lines.append(f"### {title}")
        for dataset in datasets:
            row = _find(rows, dataset, branch)
            if row is None:
                lines.append(f"- `{dataset}`: missing `{branch}`.")
                continue
            n = int(row.get("n", 0) or 0)
            wmr_parts = []
            for eps, label in EPS_LABELS:
                neg = int(row.get(f"neg_dafc_weighted_mode_recall_{label}", 0) or 0)
                pos = int(row.get(f"pos_dafc_weighted_mode_recall_{label}", 0) or 0)
                wmr_parts.append(f"WMR@{eps}: {neg}/{n} negative, {pos}/{n} positive")
            precision_neg = int(row.get("neg_dafc_mode_precision_eps05", 0) or 0)
            unsupported_pos = int(row.get("pos_dafc_unsupported_ratio_eps05", 0) or 0)
            endpoint_gt1 = int(row.get("gt1_endpoint_ratio", 0) or 0)
            lines.append(
                f"- `{dataset}` `{branch}`: "
                + "; ".join(wmr_parts)
                + f"; Prec@0.5 negative {precision_neg}/{n}; Unsupported@0.5 positive {unsupported_pos}/{n}; "
                + f"endpoint_ratio>1 {endpoint_gt1}/{n}."
            )
        lines.append("")
    return lines


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str], datasets: Sequence[str]) -> str:
    lines: List[str] = ["# AFC Experiment 1 Stability Summary", ""]
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    lines.extend(_render_eps05_table(rows))
    lines.extend(_render_eps_table(rows))
    lines.extend(_render_claim_checks(rows, datasets))
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append("- Values are mean+-std over available seeds, always relative to `slow20_pred` for delta metrics.")
    lines.append("- `endpoint_ratio` and `trajectory_ratio` are already relative to `slow20_pred`; values above 1 mean larger geometric spread.")
    lines.append("- Stability is strongest when the sign of WMR / precision / unsupported is consistent across seeds and across eps=0.3/0.5/1.0.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "analysis"
    datasets = _split_items(args.datasets)
    records, missing = _load_records(
        input_root=input_root,
        run_id=str(args.run_id),
        datasets=datasets,
        seeds=_split_items(args.seeds),
        splits=_split_items(args.splits),
        branches=_split_items(args.branches),
        file_template=str(args.file_template),
    )
    rows = _aggregate(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "afc_exp1_stability_summary.csv"
    md_path = output_dir / "afc_exp1_stability_summary.md"
    json_path = output_dir / "afc_exp1_stability_summary.json"
    _write_csv(csv_path, rows)
    md_path.write_text(_render_markdown(rows, missing, datasets), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": rows, "missing": missing}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"stability_csv={csv_path.as_posix()}")
    print(f"stability_md={md_path.as_posix()}")
    print(f"stability_json={json_path.as_posix()}")
    if missing:
        print("missing_inputs=" + ",".join(missing))


if __name__ == "__main__":
    main()
