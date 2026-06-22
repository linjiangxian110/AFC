"""Analyze AFC Experiment 3 sampling headroom from existing summary CSVs.

This is a second-stage analysis script. It does not train or evaluate a model.
It consumes headroom/stability summary CSVs and extracts the same-model
sampling branches:

- slow20_pred
- slow50_full_pred
- slow100_full_pred
- slow200_full_pred

The goal is to test whether increasing K naturally improves analogical future
coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BRANCH_TO_K: Mapping[str, int] = {
    "slow20_pred": 20,
    "slow50_full_pred": 50,
    "slow100_full_pred": 100,
    "slow200_full_pred": 200,
}

COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "ADE_min": ("mean_ADE_min", "ADE_min"),
    "FDE_min": ("mean_FDE_min", "FDE_min"),
    "ADE_avg": ("mean_ADE_avg", "ADE_avg"),
    "FDE_avg": ("mean_FDE_avg", "FDE_avg"),
    "AFC_WMR05": (
        "mean_afc_weighted_mode_recall_eps05",
        "afc_weighted_mode_recall_eps05",
        "mean_afc_mode_coverage_eps05",
        "afc_mode_coverage_eps05",
    ),
    "AFC_precision05": (
        "mean_afc_mode_precision_eps05",
        "afc_mode_precision_eps05",
        "mean_afc_precision_eps05",
        "afc_precision_eps05",
    ),
    "Unsupported05": (
        "mean_afc_unsupported_ratio_eps05",
        "afc_unsupported_ratio_eps05",
    ),
    "AFC_chamfer": (
        "mean_afc_mode_chamfer_eps05",
        "afc_mode_chamfer_eps05",
        "mean_afc_chamfer",
        "afc_chamfer",
    ),
    "endpoint_spread": ("mean_endpoint_spread", "endpoint_spread"),
    "trajectory_spread": ("mean_trajectory_spread", "trajectory_spread"),
    "endpoint_ratio": ("mean_endpoint_ratio", "endpoint_ratio"),
    "trajectory_ratio": ("mean_trajectory_ratio", "trajectory_ratio"),
    "dADE_min": ("mean_dADE_min", "dADE_min"),
    "dFDE_min": ("mean_dFDE_min", "dFDE_min"),
    "dADE_avg": ("mean_dADE_avg", "dADE_avg"),
    "dFDE_avg": ("mean_dFDE_avg", "dFDE_avg"),
    "dAFC_WMR05": (
        "mean_dafc_weighted_mode_recall_eps05",
        "dafc_weighted_mode_recall_eps05",
        "mean_dafc_mode_coverage_eps05",
        "dafc_mode_coverage_eps05",
    ),
    "dAFC_precision05": (
        "mean_dafc_mode_precision_eps05",
        "dafc_mode_precision_eps05",
        "mean_dafc_precision_eps05",
        "dafc_precision_eps05",
    ),
    "dUnsupported05": (
        "mean_dafc_unsupported_ratio_eps05",
        "dafc_unsupported_ratio_eps05",
    ),
    "dAFC_chamfer": (
        "mean_dafc_mode_chamfer_eps05",
        "dafc_mode_chamfer_eps05",
        "mean_dafc_chamfer",
        "dafc_chamfer",
    ),
}

ABS_PLOT_METRICS: Sequence[Tuple[str, str]] = (
    ("FDE_min", "minFDE"),
    ("ADE_avg", "ADE_avg"),
    ("FDE_avg", "FDE_avg"),
    ("endpoint_spread_proxy", "Endpoint spread"),
    ("AFC_WMR05", "AFC-WMR@0.5"),
    ("Unsupported05", "Unsupported@0.5"),
)

DELTA_PLOT_METRICS: Sequence[Tuple[str, str]] = (
    ("dFDE_min", "Delta minFDE"),
    ("dADE_avg", "Delta ADE_avg"),
    ("dFDE_avg", "Delta FDE_avg"),
    ("endpoint_ratio", "Endpoint spread ratio"),
    ("dAFC_WMR05", "Delta AFC-WMR@0.5"),
    ("dUnsupported05", "Delta Unsupported@0.5"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze AFC Experiment 3 sampling headroom.")
    parser.add_argument("--input-csv", nargs="+", required=True, help="One or more headroom/stability summary CSVs.")
    parser.add_argument("--output-dir", required=True, help="Directory for Experiment 3 tables and plots.")
    parser.add_argument("--run-id", default="afc_exp3_sampling_headroom", help="Run label written to outputs.")
    parser.add_argument("--expected-datasets", default="eth hotel zara1", help="Datasets expected in this run.")
    parser.add_argument("--expected-ks", default="20 50 100 200", help="K values expected in this run.")
    parser.add_argument("--split", default="test", help="Split label to keep; empty keeps all splits.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/JSON/Markdown.")
    parser.add_argument("--plot-name-suffix", default="", help="Suffix appended to plot filenames.")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in _split_items(raw)]


def _safe_suffix(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")
    return f"_{safe}" if safe else ""


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _first_num(row: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        value = _num(row.get(name))
        if value is not None:
            return value
    return None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _standardize_row(row: Mapping[str, Any], source_csv: Path, run_id: str) -> Optional[Dict[str, Any]]:
    branch = str(row.get("branch", row.get("method", ""))).strip()
    if branch not in BRANCH_TO_K:
        return None
    out: Dict[str, Any] = {
        "run_id": run_id,
        "source_csv": source_csv.as_posix(),
        "dataset": str(row.get("dataset", "")).strip(),
        "split": str(row.get("split", "")).strip(),
        "branch": branch,
        "k": BRANCH_TO_K[branch],
        "n": row.get("n", ""),
    }
    for metric, aliases in COLUMN_ALIASES.items():
        out[metric] = _first_num(row, aliases)
    out["endpoint_spread_proxy"] = out.get("endpoint_spread")
    out["endpoint_spread_proxy_kind"] = "endpoint_spread"
    if out["endpoint_spread_proxy"] is None:
        out["endpoint_spread_proxy"] = out.get("endpoint_ratio")
        out["endpoint_spread_proxy_kind"] = "endpoint_ratio"
    out["trajectory_spread_proxy"] = out.get("trajectory_spread")
    out["trajectory_spread_proxy_kind"] = "trajectory_spread"
    if out["trajectory_spread_proxy"] is None:
        out["trajectory_spread_proxy"] = out.get("trajectory_ratio")
        out["trajectory_spread_proxy_kind"] = "trajectory_ratio"
    if not out["dataset"]:
        return None
    return out


def _load_rows(input_csvs: Sequence[str], run_id: str, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in input_csvs:
        path = Path(raw).expanduser().resolve()
        for row in _read_csv(path):
            normalized = _standardize_row(row, path, run_id)
            if normalized is None:
                continue
            if split and str(normalized.get("split", "")) != split:
                continue
            rows.append(normalized)
    return rows


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _aggregate_for_plot(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Dict[int, float]]:
    grouped: Dict[Tuple[str, int], List[Optional[float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["k"]))].append(_num(row.get(metric)))
    result: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (dataset, k), values in grouped.items():
        value = _mean(values)
        if value is not None:
            result[dataset][k] = float(value)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _missing_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_datasets: Sequence[str],
    expected_ks: Sequence[int],
) -> List[Dict[str, Any]]:
    by_key = {(str(row["dataset"]), int(row["k"])): row for row in rows}
    checks = [
        "ADE_avg",
        "FDE_avg",
        "ADE_min",
        "FDE_min",
        "AFC_WMR05",
        "AFC_precision05",
        "Unsupported05",
        "dAFC_WMR05",
        "dUnsupported05",
    ]
    report: List[Dict[str, Any]] = []
    for dataset in expected_datasets:
        for k in expected_ks:
            row = by_key.get((dataset, int(k)))
            if row is None:
                report.append({"dataset": dataset, "k": int(k), "status": "missing_row", "missing_fields": "all"})
                continue
            missing = [field for field in checks if row.get(field) is None]
            if missing:
                report.append(
                    {
                        "dataset": dataset,
                        "k": int(k),
                        "status": "missing_fields",
                        "missing_fields": ",".join(missing),
                        "source_csv": row.get("source_csv", ""),
                    }
                )
    return report


def _plot_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[Tuple[str, str]],
    expected_ks: Sequence[int],
    output_base: Path,
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot_skipped={output_base.as_posix()} reason={exc}")
        return False

    datasets = sorted({str(row["dataset"]) for row in rows})
    if not datasets:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes_flat = list(axes.ravel())
    plotted_any = False
    for ax, (metric, label) in zip(axes_flat, metrics):
        grouped = _aggregate_for_plot(rows, metric)
        metric_plotted = False
        for dataset in datasets:
            points = [(k, grouped.get(dataset, {}).get(k)) for k in expected_ks]
            xs = [k for k, value in points if value is not None]
            ys = [float(value) for _k, value in points if value is not None]
            if len(xs) < 1:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=dataset)
            metric_plotted = True
            plotted_any = True
        ax.set_title(label)
        ax.set_xlabel("K")
        ax.set_xticks(list(expected_ks))
        ax.grid(True, alpha=0.3)
        if not metric_plotted:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(title, y=1.04)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return plotted_any


def _render_summary(
    *,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    expected_ks: Sequence[int],
) -> str:
    lines: List[str] = [
        "# AFC Experiment 3 Sampling Headroom Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- rows: `{len(rows)}`",
        f"- expected K: `{', '.join(str(k) for k in expected_ks)}`",
        "",
        "## K Headroom Table",
        "",
        "| dataset | K | branch | n | ADE_avg | FDE_avg | minADE | minFDE | AFC-WMR@0.5 | AFC-Precision@0.5 | Unsupported@0.5 | dAFC-WMR@0.5 | dUnsupported@0.5 | endpoint_ratio | source |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), int(item["k"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("k", "")),
                    str(row.get("branch", "")),
                    str(row.get("n", "")),
                    _fmt(_num(row.get("ADE_avg"))),
                    _fmt(_num(row.get("FDE_avg"))),
                    _fmt(_num(row.get("ADE_min"))),
                    _fmt(_num(row.get("FDE_min"))),
                    _fmt(_num(row.get("AFC_WMR05"))),
                    _fmt(_num(row.get("AFC_precision05"))),
                    _fmt(_num(row.get("Unsupported05"))),
                    _fmt(_num(row.get("dAFC_WMR05")), signed=True),
                    _fmt(_num(row.get("dUnsupported05")), signed=True),
                    _fmt(_num(row.get("endpoint_ratio"))),
                    str(row.get("source_csv", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Missing Report", ""])
    if not missing:
        lines.append("- No missing expected K rows or checked fields.")
    else:
        lines.append("| dataset | K | status | missing fields | source |")
        lines.append("|---|---:|---|---|---|")
        for item in missing:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("dataset", "")),
                        str(item.get("k", "")),
                        str(item.get("status", "")),
                        str(item.get("missing_fields", "")),
                        str(item.get("source_csv", "")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If `dAFC-WMR@0.5` remains near zero from K=20 to K=200, larger sampling mainly densifies existing modes.",
            "- If minADE/minFDE improve while AFC-WMR stays flat, best-of-K gains do not imply broader plausible-future coverage.",
            "- If Unsupported@0.5 does not decrease, extra samples are not automatically better supported by analogical futures.",
            "- Treat this as a same-model sampling diagnostic, not a new method or an external-baseline comparison.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_datasets = _split_items(args.expected_datasets)
    expected_ks = _split_ints(args.expected_ks)
    rows = _load_rows(args.input_csv, str(args.run_id), str(args.split))
    rows = [row for row in rows if str(row["dataset"]) in set(expected_datasets)]
    missing = _missing_report(rows, expected_datasets=expected_datasets, expected_ks=expected_ks)

    row_fields = [
        "run_id",
        "dataset",
        "split",
        "branch",
        "k",
        "n",
        "ADE_min",
        "FDE_min",
        "ADE_avg",
        "FDE_avg",
        "AFC_WMR05",
        "AFC_precision05",
        "Unsupported05",
        "AFC_chamfer",
        "endpoint_spread",
        "trajectory_spread",
        "endpoint_ratio",
        "trajectory_ratio",
        "dADE_min",
        "dFDE_min",
        "dADE_avg",
        "dFDE_avg",
        "dAFC_WMR05",
        "dAFC_precision05",
        "dUnsupported05",
        "dAFC_chamfer",
        "source_csv",
    ]
    delta_fields = [
        "run_id",
        "dataset",
        "split",
        "branch",
        "k",
        "n",
        "dADE_min",
        "dFDE_min",
        "dADE_avg",
        "dFDE_avg",
        "dAFC_WMR05",
        "dAFC_precision05",
        "dUnsupported05",
        "dAFC_chamfer",
        "endpoint_ratio",
        "trajectory_ratio",
        "source_csv",
    ]
    _write_csv(output_dir / "afc_exp3_headroom_rows.csv", rows, row_fields)
    _write_csv(output_dir / "afc_exp3_headroom_delta.csv", rows, delta_fields)
    _write_csv(
        output_dir / "afc_exp3_missing_report.csv",
        missing,
        ["dataset", "k", "status", "missing_fields", "source_csv"],
    )
    (output_dir / "afc_exp3_sampling_headroom_summary.md").write_text(
        _render_summary(run_id=str(args.run_id), rows=rows, missing=missing, expected_ks=expected_ks),
        encoding="utf-8",
    )
    (output_dir / "afc_exp3_headroom_rows.json").write_text(
        json.dumps({"rows": rows, "missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    suffix = _safe_suffix(args.plot_name_suffix)
    if not args.no_plots:
        _plot_lines(
            rows,
            metrics=ABS_PLOT_METRICS,
            expected_ks=expected_ks,
            output_base=output_dir / f"afc_exp3_k_curve_absolute{suffix}",
            title=f"{args.run_id}: absolute sampling headroom",
        )
        _plot_lines(
            rows,
            metrics=DELTA_PLOT_METRICS,
            expected_ks=expected_ks,
            output_base=output_dir / f"afc_exp3_k_curve_delta{suffix}",
            title=f"{args.run_id}: delta from slow20",
        )
    print(f"rows_csv={(output_dir / 'afc_exp3_headroom_rows.csv').as_posix()}")
    print(f"delta_csv={(output_dir / 'afc_exp3_headroom_delta.csv').as_posix()}")
    print(f"missing_csv={(output_dir / 'afc_exp3_missing_report.csv').as_posix()}")
    print(f"summary_md={(output_dir / 'afc_exp3_sampling_headroom_summary.md').as_posix()}")
    if missing:
        print(f"missing_count={len(missing)}")


if __name__ == "__main__":
    main()
