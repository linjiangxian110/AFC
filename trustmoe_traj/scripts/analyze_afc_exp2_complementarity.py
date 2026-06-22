"""Analyze AFC Experiment 2 complementarity from existing summary CSVs.

This is a second-stage analysis script: it does not train or evaluate a model.
It consumes AFC/headroom summary CSVs, standardizes the metric names, and
reports whether AFC metrics duplicate or complement ADE/FDE and geometric
spread metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MetricPair = Tuple[str, str, str]

DATASET_COLORS: Mapping[str, Tuple[str, str]] = {
    "eth": ("blue", "#4c78a8"),
    "hotel": ("orange", "#f58518"),
    "univ": ("green", "#54a24b"),
    "zara1": ("purple", "#b279a2"),
    "zara2": ("red", "#e45756"),
    "sdd": ("teal", "#72b7b2"),
}

ABS_PAIRS: Sequence[MetricPair] = (
    ("ADE_avg", "AFC_WMR05", "Accuracy vs AFC coverage"),
    ("FDE_avg", "AFC_WMR05", "Endpoint accuracy vs AFC coverage"),
    ("EnergyScore", "AFC_WMR05", "Energy Score vs AFC coverage"),
    ("EnergyScore", "Unsupported05", "Energy Score vs unsupported predictions"),
    ("EnergyScore", "AFC_chamfer", "Energy Score vs AFC distance"),
    ("endpoint_spread_proxy", "AFC_precision05", "Geometric endpoint spread vs analogical support"),
    ("trajectory_spread_proxy", "Unsupported05", "Geometric trajectory spread vs unsupported predictions"),
    ("AFC_WMR05", "AFC_chamfer", "AFC discrete coverage vs continuous distance"),
)
DELTA_PAIRS: Sequence[MetricPair] = (
    ("dADE_avg", "dAFC_WMR05", "Accuracy gain/loss vs AFC coverage change"),
    ("dFDE_avg", "dAFC_WMR05", "Endpoint accuracy gain/loss vs AFC coverage change"),
    ("dEnergyScore", "dAFC_WMR05", "Energy Score change vs AFC coverage change"),
    ("dEnergyScore", "dUnsupported05", "Energy Score change vs unsupported change"),
    ("dEnergyScore", "dAFC_chamfer", "Energy Score change vs AFC distance change"),
    ("endpoint_ratio", "dAFC_precision05", "Relative endpoint spread vs AFC support change"),
    ("trajectory_ratio", "dUnsupported05", "Relative trajectory spread vs unsupported change"),
    ("dAFC_WMR05", "dAFC_chamfer", "AFC coverage change vs AFC distance change"),
)
SYMMETRIC_DELTA_PAIRS: Sequence[MetricPair] = (
    ("dEndpoint_spread_proxy", "dAFC_precision05", "Endpoint spread change vs AFC support change"),
    ("dTrajectory_spread_proxy", "dUnsupported05", "Trajectory spread change vs unsupported change"),
)
RATIO_PAIRS: Sequence[MetricPair] = (
    ("endpoint_ratio", "AFC_precision_ratio05", "Relative endpoint spread vs relative AFC support"),
    ("trajectory_ratio", "Unsupported_ratio05", "Relative trajectory spread vs relative unsupported ratio"),
)

COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "ADE_avg": ("mean_ADE_avg", "ADE_avg"),
    "FDE_avg": ("mean_FDE_avg", "FDE_avg"),
    "ADE_min": ("mean_ADE_min", "ADE_min"),
    "FDE_min": ("mean_FDE_min", "FDE_min"),
    "EnergyScore": ("mean_energy_score", "energy_score"),
    "EnergyGTTerm": ("mean_energy_gt_term", "energy_gt_term"),
    "EnergyPairTerm": ("mean_energy_pair_term", "energy_pair_term"),
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
    "dADE_avg": ("mean_dADE_avg", "dADE_avg"),
    "dFDE_avg": ("mean_dFDE_avg", "dFDE_avg"),
    "dADE_min": ("mean_dADE_min", "dADE_min"),
    "dFDE_min": ("mean_dFDE_min", "dFDE_min"),
    "dEnergyScore": ("mean_denergy_score", "denergy_score"),
    "dEnergyGTTerm": ("mean_denergy_gt_term", "denergy_gt_term"),
    "dEnergyPairTerm": ("mean_denergy_pair_term", "denergy_pair_term"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze AFC Experiment 2 complementarity.")
    parser.add_argument("--input-csv", nargs="+", required=True, help="One or more AFC summary CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for Experiment 2 tables and plots.")
    parser.add_argument("--run-id", default="afc_exp2_complementarity", help="Run label written to outputs.")
    parser.add_argument(
        "--exclude-branches",
        default="",
        help="Exact branch names to exclude, separated by commas or spaces.",
    )
    parser.add_argument(
        "--exclude-branch-contains",
        default="",
        help="Branch-name substrings to exclude, separated by commas or spaces.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV/JSON/Markdown; skip matplotlib scatter plots.",
    )
    parser.add_argument(
        "--plot-name-suffix",
        default="",
        help="Optional suffix appended to plot filenames, for example 'seed0_fullmetrics_legend'.",
    )
    parser.add_argument("--min-points", type=int, default=3, help="Minimum rows needed for a correlation.")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _safe_filename_suffix(raw: str) -> str:
    suffix = str(raw).strip()
    if not suffix:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in suffix)
    safe = safe.strip("_")
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
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _first_num(row: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        value = _num(row.get(name))
        if value is not None:
            return value
    return None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _standardize_row(row: Mapping[str, Any], source_csv: Path, run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_id": run_id,
        "source_csv": source_csv.as_posix(),
        "dataset": str(row.get("dataset", "")).strip(),
        "split": str(row.get("split", "")).strip(),
        "branch": str(row.get("branch", row.get("method", ""))).strip(),
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
    return out


def _load_rows(input_csvs: Sequence[str], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in input_csvs:
        path = Path(raw).expanduser().resolve()
        for row in _read_csv(path):
            normalized = _standardize_row(row, path, run_id)
            if normalized["dataset"] and normalized["branch"]:
                rows.append(normalized)
    _add_baseline_relative_metrics(rows)
    return rows


def _safe_ratio(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None or abs(float(baseline)) <= 1e-12:
        return None
    return float(value) / float(baseline)


def _add_baseline_relative_metrics(rows: Sequence[Dict[str, Any]]) -> None:
    baselines: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if _is_baseline(row):
            baselines[(str(row.get("dataset", "")), str(row.get("split", "")))] = row

    for row in rows:
        baseline = baselines.get((str(row.get("dataset", "")), str(row.get("split", ""))))
        endpoint_ratio = _num(row.get("endpoint_ratio"))
        trajectory_ratio = _num(row.get("trajectory_ratio"))
        endpoint_spread = _num(row.get("endpoint_spread"))
        trajectory_spread = _num(row.get("trajectory_spread"))
        afc_precision = _num(row.get("AFC_precision05"))
        unsupported = _num(row.get("Unsupported05"))

        baseline_endpoint_spread = _num(baseline.get("endpoint_spread")) if baseline else None
        baseline_trajectory_spread = _num(baseline.get("trajectory_spread")) if baseline else None
        baseline_afc_precision = _num(baseline.get("AFC_precision05")) if baseline else None
        baseline_unsupported = _num(baseline.get("Unsupported05")) if baseline else None

        if endpoint_ratio is not None:
            row["dEndpoint_spread_proxy"] = endpoint_ratio - 1.0
            row["endpoint_spread_proxy_change_kind"] = "endpoint_ratio_minus_1"
        elif endpoint_spread is not None and baseline_endpoint_spread is not None:
            row["dEndpoint_spread_proxy"] = endpoint_spread - baseline_endpoint_spread
            row["endpoint_spread_proxy_change_kind"] = "endpoint_spread_delta"
        else:
            row["dEndpoint_spread_proxy"] = None
            row["endpoint_spread_proxy_change_kind"] = ""

        if trajectory_ratio is not None:
            row["dTrajectory_spread_proxy"] = trajectory_ratio - 1.0
            row["trajectory_spread_proxy_change_kind"] = "trajectory_ratio_minus_1"
        elif trajectory_spread is not None and baseline_trajectory_spread is not None:
            row["dTrajectory_spread_proxy"] = trajectory_spread - baseline_trajectory_spread
            row["trajectory_spread_proxy_change_kind"] = "trajectory_spread_delta"
        else:
            row["dTrajectory_spread_proxy"] = None
            row["trajectory_spread_proxy_change_kind"] = ""

        row["AFC_precision_ratio05"] = _safe_ratio(afc_precision, baseline_afc_precision)
        row["Unsupported_ratio05"] = _safe_ratio(unsupported, baseline_unsupported)


def _filter_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    exclude_branches: Sequence[str],
    exclude_contains: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exact = set(exclude_branches)
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for row in rows:
        branch = str(row.get("branch", ""))
        if branch in exact or any(token in branch for token in exclude_contains):
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row.keys()})
    preferred = [
        "run_id",
        "dataset",
        "split",
        "branch",
        "n",
        "source_csv",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _is_baseline(row: Mapping[str, Any]) -> bool:
    return str(row.get("branch", "")) == "slow20_pred"


def _paired_values(
    rows: Sequence[Mapping[str, Any]],
    x_metric: str,
    y_metric: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = _num(row.get(x_metric))
        y = _num(row.get(y_metric))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def _correlation_rows(rows: Sequence[Mapping[str, Any]], min_points: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    scopes: List[Tuple[str, str, Sequence[Mapping[str, Any]], Sequence[MetricPair]]] = [
        ("global_absolute", "all", rows, ABS_PAIRS),
        ("global_delta_nonbaseline", "all", [row for row in rows if not _is_baseline(row)], DELTA_PAIRS),
        (
            "global_symmetric_delta_nonbaseline",
            "all",
            [row for row in rows if not _is_baseline(row)],
            SYMMETRIC_DELTA_PAIRS,
        ),
        ("global_ratio_nonbaseline", "all", [row for row in rows if not _is_baseline(row)], RATIO_PAIRS),
    ]
    for dataset in datasets:
        subset = [row for row in rows if str(row.get("dataset")) == dataset]
        subset_nonbaseline = [row for row in subset if not _is_baseline(row)]
        scopes.append(("dataset_absolute", dataset, subset, ABS_PAIRS))
        scopes.append(("dataset_delta_nonbaseline", dataset, subset_nonbaseline, DELTA_PAIRS))
        scopes.append(("dataset_symmetric_delta_nonbaseline", dataset, subset_nonbaseline, SYMMETRIC_DELTA_PAIRS))
        scopes.append(("dataset_ratio_nonbaseline", dataset, subset_nonbaseline, RATIO_PAIRS))

    for scope, dataset, subset, pairs in scopes:
        for x_metric, y_metric, interpretation in pairs:
            xs, ys = _paired_values(subset, x_metric, y_metric)
            pearson = _pearson(xs, ys) if len(xs) >= min_points else None
            spearman = _spearman(xs, ys) if len(xs) >= min_points else None
            out.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "pair": f"{x_metric} vs {y_metric}",
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": len(xs),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "interpretation": interpretation,
                }
            )
    return out


def _render_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    excluded_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plots: Sequence[Path],
) -> str:
    lines: List[str] = []
    lines.append("# AFC Experiment 2 Complementarity Summary")
    lines.append("")
    lines.append("This analysis tests whether AFC duplicates ADE/FDE or geometric spread metrics.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Standardized rows: `{(output_dir / 'afc_exp2_rows.csv').as_posix()}`")
    lines.append(f"- Correlation table: `{(output_dir / 'afc_exp2_correlations.csv').as_posix()}`")
    sources = sorted({str(row.get("source_csv", "")) for row in rows if row.get("source_csv")})
    for source in sources:
        lines.append(f"- Source CSV: `{source}`")
    lines.append(f"- Rows used: `{len(rows)}`")
    if excluded_rows:
        lines.append(f"- Rows excluded by branch filter: `{len(excluded_rows)}`")
        excluded_branches = sorted({str(row.get("branch", "")) for row in excluded_rows})
        for branch in excluded_branches:
            count = sum(1 for row in excluded_rows if str(row.get("branch", "")) == branch)
            lines.append(f"  - `{branch}`: {count}")
    if plots:
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        for plot in plots:
            lines.append(f"- `{plot.as_posix()}`")
        lines.append("")
        lines.append("## Plot Encoding")
        lines.append("")
        lines.append("Color encodes dataset:")
        lines.append("")
        lines.append("| dataset | color |")
        lines.append("|---|---|")
        datasets_in_rows = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
        for dataset in datasets_in_rows:
            color_name, color_hex = DATASET_COLORS.get(dataset, ("fallback gray", "#666666"))
            lines.append(f"| `{dataset}` | {color_name} `{color_hex}` |")
        lines.append("")
        lines.append("Marker encodes branch group:")
        lines.append("")
        lines.append("| marker | branch group | rule |")
        lines.append("|---|---|---|")
        lines.append("| circle | baseline | `branch == slow20_pred` |")
        lines.append("| square | GT oracle | branch contains `gt_oracle` |")
        lines.append("| triangle | Endpoint FPS | branch contains `endpoint_fps` |")
        lines.append("| diamond | CV weak model | branch contains `cv` |")
        lines.append("| X | fake spread | branch contains `random_spread` |")
        lines.append("| pentagon | AFC greedy | branch contains `afc_greedy` |")
        lines.append("| down triangle | larger/full sampling | branch contains `full` |")
        lines.append("| left triangle | random subset | branch contains `random` but not `random_spread` |")
        lines.append("| small circle | other | fallback branch group |")
    lines.append("")
    lines.append("## Global Correlations")
    lines.append("")
    lines.append("| scope | pair | n | Pearson r | Spearman rho | reading |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in correlations:
        if not str(row.get("scope", "")).startswith("global"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scope", "")),
                    str(row.get("pair", "")),
                    str(row.get("n", "")),
                    _fmt(_num(row.get("pearson_r")), signed=True),
                    _fmt(_num(row.get("spearman_rho")), signed=True),
                    str(row.get("interpretation", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append("- Strong ADE/FDE improvement with negative `dAFC_WMR05` supports the GT-centric collapse claim.")
    lines.append("- Energy Score rows test whether a proper set-level score explains AFC behavior; weak or inconsistent Energy-vs-AFC correlations support reporting AFC as a separate empirical-support profile.")
    lines.append("- Endpoint/trajectory spread ratios above 1 with lower precision or higher unsupported ratio support the fake-diversity claim.")
    lines.append("- Weak or moderate ADE/FDE-vs-AFC correlations support the claim that best-of-K accuracy and AFC coverage are not the same quantity.")
    lines.append("- Strong spread-vs-support correlations should be reported directly: they indicate that geometric dispersion can drive support loss, while AFC still supplies mode-level recall, precision, unsupported ratio, and Chamfer decomposition.")
    lines.append("- Symmetric delta and ratio rows are included to check that spread/AFC conclusions are not artifacts of comparing a ratio against a difference.")
    lines.append("- Correlation alone is not sufficient; connect this table back to Experiment 1 diagnostic branches and later retrieval visualization.")
    return "\n".join(lines).rstrip() + "\n"


def _branch_marker_and_label(branch: str) -> Tuple[str, str]:
    if branch == "slow20_pred":
        return "o", "baseline: slow20_pred"
    if "gt_oracle" in branch:
        return "s", "GT oracle"
    if "endpoint_fps" in branch:
        return "^", "Endpoint FPS"
    if "cv" in branch:
        return "D", "CV weak"
    if "random_spread" in branch:
        return "X", "fake spread"
    if "afc_greedy" in branch:
        return "P", "AFC greedy"
    if "full" in branch:
        return "v", "larger/full sampling"
    if "random" in branch:
        return "<", "random subset"
    return "o", "other diagnostic branch"


def _plot_scatter_grid(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[MetricPair],
    output_stem: Path,
    title: str,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return []

    filtered_pairs: List[MetricPair] = []
    for x_metric, y_metric, interpretation in pairs:
        xs, ys = _paired_values(rows, x_metric, y_metric)
        if len(xs) >= 2:
            filtered_pairs.append((x_metric, y_metric, interpretation))
    if not filtered_pairs:
        return []

    colors = {dataset: color_hex for dataset, (_color_name, color_hex) in DATASET_COLORS.items()}
    fig, axes = plt.subplots(1, len(filtered_pairs), figsize=(3.2 * len(filtered_pairs), 3.7), squeeze=False)
    axes_flat = axes[0]
    datasets_in_plot = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    marker_labels: Dict[str, str] = {}
    for ax, (x_metric, y_metric, interpretation) in zip(axes_flat, filtered_pairs):
        for row in rows:
            x = _num(row.get(x_metric))
            y = _num(row.get(y_metric))
            if x is None or y is None:
                continue
            dataset = str(row.get("dataset", ""))
            branch = str(row.get("branch", ""))
            marker, marker_label = _branch_marker_and_label(branch)
            marker_labels.setdefault(marker_label, marker)
            ax.scatter(
                [x],
                [y],
                c=colors.get(dataset, "#666666"),
                marker=marker,
                s=32,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(interpretation, fontsize=8)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.8)
    fig.suptitle(title, fontsize=10)
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5,
            markerfacecolor=colors.get(dataset, "#666666"),
            markeredgecolor="white",
            label=f"dataset: {dataset}",
        )
        for dataset in datasets_in_plot
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markersize=5,
            markerfacecolor="#555555",
            markeredgecolor="white",
            label=f"branch: {label}",
        )
        for label, marker in marker_labels.items()
    ]
    legend_handles = color_handles + marker_handles
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(legend_handles)),
            fontsize=6.5,
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 0.93))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in outputs:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_rows = _load_rows(args.input_csv, str(args.run_id))
    rows, excluded_rows = _filter_rows(
        raw_rows,
        exclude_branches=_split_items(args.exclude_branches),
        exclude_contains=_split_items(args.exclude_branch_contains),
    )
    correlations = _correlation_rows(rows, int(args.min_points))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "afc_exp2_rows.csv", rows)
    if excluded_rows:
        _write_csv(output_dir / "afc_exp2_excluded_rows.csv", excluded_rows)
    _write_csv(output_dir / "afc_exp2_correlations.csv", correlations)
    (output_dir / "afc_exp2_rows.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "rows": rows,
                "excluded_rows": excluded_rows,
                "correlations": correlations,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plots: List[Path] = []
    if not args.no_plots:
        plot_suffix = _safe_filename_suffix(args.plot_name_suffix)
        plots.extend(
            _plot_scatter_grid(
                rows,
                ABS_PAIRS,
                output_dir / f"afc_exp2_scatter_absolute{plot_suffix}",
                "AFC Experiment 2: absolute metrics",
            )
        )
        delta_rows = [row for row in rows if not _is_baseline(row)]
        plots.extend(
            _plot_scatter_grid(
                delta_rows,
                DELTA_PAIRS,
                output_dir / f"afc_exp2_scatter_delta{plot_suffix}",
                "AFC Experiment 2: deltas vs slow20",
            )
        )

    summary = _render_markdown(rows=rows, excluded_rows=excluded_rows, correlations=correlations, output_dir=output_dir, plots=plots)
    (output_dir / "afc_exp2_complementarity_summary.md").write_text(summary, encoding="utf-8")

    print(f"rows_csv={(output_dir / 'afc_exp2_rows.csv').as_posix()}")
    if excluded_rows:
        print(f"excluded_rows_csv={(output_dir / 'afc_exp2_excluded_rows.csv').as_posix()}")
    print(f"correlations_csv={(output_dir / 'afc_exp2_correlations.csv').as_posix()}")
    print(f"summary_md={(output_dir / 'afc_exp2_complementarity_summary.md').as_posix()}")
    for plot in plots:
        print(f"plot={plot.as_posix()}")


if __name__ == "__main__":
    main()
