"""Analyze AFC Experiment 4 Top-M / eps stability from summary CSVs.

This script is a second-stage analyzer. It does not train or evaluate a model.
It consumes one CSV per Top-M setting, where each CSV is a headroom summary
produced by ``summarize_headroom_analysis.py`` or a compatible stability
summary. Each input should be passed as ``TOPM=PATH``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EPS_LABELS: Sequence[Tuple[str, str]] = (("0.3", "eps03"), ("0.5", "eps05"), ("1.0", "eps10"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze AFC Top-M / eps stability.")
    parser.add_argument(
        "--input-csv",
        nargs="+",
        required=True,
        help="Inputs as TOPM=CSV, e.g. 5=/path/headroom_summary.csv.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="afc_exp4_topm_eps_stability")
    parser.add_argument("--expected-datasets", default="eth hotel zara1")
    parser.add_argument("--expected-top-ms", default="5 10 20 50")
    parser.add_argument("--expected-eps", default="0.3 0.5 1.0")
    parser.add_argument("--full-branch", default="slow200_full_pred")
    parser.add_argument("--gt-oracle-branch", default="slow200_gt_oracle20_pred")
    parser.add_argument("--endpoint-fps-branch", default="slow200_endpoint_fps20_pred")
    parser.add_argument("--limited-gain-threshold", type=float, default=0.005)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in _split_items(raw)]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _parse_topm_path(raw: str) -> Tuple[int, Path]:
    if "=" in raw:
        left, right = raw.split("=", 1)
        return int(left.strip()), Path(right.strip()).expanduser().resolve()
    match = re.search(r"(?:topm|top_m|m)(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), Path(raw).expanduser().resolve()
    raise SystemExit(f"Input must be TOPM=CSV when Top-M cannot be inferred: {raw}")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        value = _num(row.get(name))
        if value is not None:
            return value
    return None


def _eps_metric(row: Mapping[str, Any], base: str, eps_label: str) -> Optional[float]:
    aliases = {
        "AFC_WMR": (
            f"mean_afc_weighted_mode_recall_{eps_label}",
            f"afc_weighted_mode_recall_{eps_label}",
            f"mean_afc_mode_coverage_{eps_label}",
            f"afc_mode_coverage_{eps_label}",
        ),
        "AFC_precision": (
            f"mean_afc_mode_precision_{eps_label}",
            f"afc_mode_precision_{eps_label}",
            f"mean_afc_precision_{eps_label}",
            f"afc_precision_{eps_label}",
        ),
        "Unsupported": (
            f"mean_afc_unsupported_ratio_{eps_label}",
            f"afc_unsupported_ratio_{eps_label}",
        ),
        "dAFC_WMR": (
            f"mean_dafc_weighted_mode_recall_{eps_label}",
            f"dafc_weighted_mode_recall_{eps_label}",
            f"mean_dafc_mode_coverage_{eps_label}",
            f"dafc_mode_coverage_{eps_label}",
        ),
        "dAFC_precision": (
            f"mean_dafc_mode_precision_{eps_label}",
            f"dafc_mode_precision_{eps_label}",
            f"mean_dafc_precision_{eps_label}",
            f"dafc_precision_{eps_label}",
        ),
        "dUnsupported": (
            f"mean_dafc_unsupported_ratio_{eps_label}",
            f"dafc_unsupported_ratio_{eps_label}",
        ),
        "dAFC_chamfer": (
            f"mean_dafc_mode_chamfer_{eps_label}",
            f"dafc_mode_chamfer_{eps_label}",
            "mean_dafc_chamfer",
            "dafc_chamfer",
        ),
    }
    return _first(row, aliases[base])


def _load_rows(input_csvs: Sequence[str], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in input_csvs:
        top_m, path = _parse_topm_path(raw)
        for row in _read_csv(path):
            dataset = str(row.get("dataset", "")).strip()
            split = str(row.get("split", "")).strip()
            branch = str(row.get("branch", row.get("method", ""))).strip()
            if not dataset or not branch:
                continue
            for eps, label in EPS_LABELS:
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "split": split,
                        "branch": branch,
                        "top_m": int(top_m),
                        "eps": float(eps),
                        "n": row.get("n", ""),
                        "dADE_avg": _first(row, ("mean_dADE_avg", "dADE_avg")),
                        "dFDE_avg": _first(row, ("mean_dFDE_avg", "dFDE_avg")),
                        "endpoint_ratio": _first(row, ("mean_endpoint_ratio", "endpoint_ratio")),
                        "trajectory_ratio": _first(row, ("mean_trajectory_ratio", "trajectory_ratio")),
                        "AFC_WMR": _eps_metric(row, "AFC_WMR", label),
                        "AFC_precision": _eps_metric(row, "AFC_precision", label),
                        "Unsupported": _eps_metric(row, "Unsupported", label),
                        "dAFC_WMR": _eps_metric(row, "dAFC_WMR", label),
                        "dAFC_precision": _eps_metric(row, "dAFC_precision", label),
                        "dUnsupported": _eps_metric(row, "dUnsupported", label),
                        "dAFC_chamfer": _eps_metric(row, "dAFC_chamfer", label),
                        "source_csv": path.as_posix(),
                    }
                )
    return rows


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    return None if not kept else sum(kept) / len(kept)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        preferred = ["run_id", "dataset", "split", "branch", "top_m", "eps", "n"]
        fields = sorted({field for row in rows for field in row.keys()})
        fieldnames = preferred + [field for field in fields if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _condition_for_claim(row: Mapping[str, Any], claim: str, threshold: float) -> Optional[bool]:
    if claim == "gt_oracle_lowers_wmr":
        value = _num(row.get("dAFC_WMR"))
        return None if value is None else value < 0.0
    if claim == "endpoint_fps_hurts_support":
        precision = _num(row.get("dAFC_precision"))
        unsupported = _num(row.get("dUnsupported"))
        if precision is None and unsupported is None:
            return None
        return (precision is not None and precision <= 0.0) or (unsupported is not None and unsupported >= 0.0)
    if claim == "full_sampling_limited_wmr_gain":
        value = _num(row.get("dAFC_WMR"))
        return None if value is None else abs(value) <= float(threshold)
    return None


def _claim_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    top_ms: Sequence[int],
    eps_values: Sequence[float],
    gt_branch: str,
    fps_branch: str,
    full_branch: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    claim_specs = [
        ("gt_oracle_lowers_wmr", gt_branch, "GT oracle lowers AFC-WMR"),
        ("endpoint_fps_hurts_support", fps_branch, "Endpoint FPS hurts AFC support"),
        ("full_sampling_limited_wmr_gain", full_branch, "Full sampling has limited AFC-WMR gain"),
    ]
    result: List[Dict[str, Any]] = []
    by_key = {
        (str(row.get("dataset")), str(row.get("branch")), int(row.get("top_m")), float(row.get("eps"))): row
        for row in rows
    }
    for claim_key, branch, claim_name in claim_specs:
        stable_total = 0
        observed_total = 0
        for dataset in datasets:
            dataset_stable = 0
            dataset_observed = 0
            missing = 0
            for top_m in top_ms:
                for eps in eps_values:
                    row = by_key.get((dataset, branch, int(top_m), float(eps)))
                    if row is None:
                        missing += 1
                        continue
                    verdict = _condition_for_claim(row, claim_key, threshold)
                    if verdict is None:
                        missing += 1
                        continue
                    dataset_observed += 1
                    observed_total += 1
                    if verdict:
                        dataset_stable += 1
                        stable_total += 1
            result.append(
                {
                    "scope": "dataset",
                    "dataset": dataset,
                    "claim_key": claim_key,
                    "claim": claim_name,
                    "branch": branch,
                    "stable_configs": dataset_stable,
                    "observed_configs": dataset_observed,
                    "expected_configs": len(top_ms) * len(eps_values),
                    "missing_configs": missing,
                    "stability_ratio": None if dataset_observed == 0 else dataset_stable / dataset_observed,
                }
            )
        result.append(
            {
                "scope": "global",
                "dataset": "ALL",
                "claim_key": claim_key,
                "claim": claim_name,
                "branch": branch,
                "stable_configs": stable_total,
                "observed_configs": observed_total,
                "expected_configs": len(datasets) * len(top_ms) * len(eps_values),
                "missing_configs": len(datasets) * len(top_ms) * len(eps_values) - observed_total,
                "stability_ratio": None if observed_total == 0 else stable_total / observed_total,
            }
        )
    return result


def _metric_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    branch: str,
    metric: str,
    top_ms: Sequence[int],
    eps_values: Sequence[float],
) -> List[List[Optional[float]]]:
    grouped: Dict[Tuple[int, float], List[Optional[float]]] = defaultdict(list)
    for row in rows:
        if str(row.get("branch")) != branch:
            continue
        grouped[(int(row.get("top_m")), float(row.get("eps")))].append(_num(row.get(metric)))
    grid: List[List[Optional[float]]] = []
    for eps in eps_values:
        grid.append([_mean(grouped.get((int(top_m), float(eps)), [])) for top_m in top_ms])
    return grid


def _plot_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    branch: str,
    metric: str,
    title: str,
    output_base: Path,
    top_ms: Sequence[int],
    eps_values: Sequence[float],
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"plot_skipped={output_base.as_posix()} reason={exc}")
        return False
    grid = _metric_grid(rows, branch=branch, metric=metric, top_ms=top_ms, eps_values=eps_values)
    values = np.array([[np.nan if value is None else float(value) for value in line] for line in grid], dtype=float)
    if np.isnan(values).all():
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    vmax = np.nanmax(np.abs(values))
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else float(vmax)
    image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(top_ms)), [str(item) for item in top_ms])
    ax.set_yticks(range(len(eps_values)), [str(item) for item in eps_values])
    ax.set_xlabel("Top-M")
    ax.set_ylabel("eps")
    ax.set_title(title)
    for i, eps in enumerate(eps_values):
        for j, top_m in enumerate(top_ms):
            value = values[i, j]
            text = "NA" if np.isnan(value) else f"{value:+.4f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def _render_summary(
    *,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    top_ms: Sequence[int],
    eps_values: Sequence[float],
) -> str:
    lines: List[str] = [
        "# AFC Experiment 4 Top-M / eps Stability Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- normalized rows: `{len(rows)}`",
        f"- Top-M grid: `{', '.join(str(item) for item in top_ms)}`",
        f"- eps grid: `{', '.join(str(item) for item in eps_values)}`",
        "",
        "## Claim Stability Table",
        "",
        "| scope | dataset | claim | branch | stable / observed | expected | missing | ratio |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in claim_rows:
        ratio = _num(row.get("stability_ratio"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scope", "")),
                    str(row.get("dataset", "")),
                    str(row.get("claim", "")),
                    str(row.get("branch", "")),
                    f"{row.get('stable_configs', 0)} / {row.get('observed_configs', 0)}",
                    str(row.get("expected_configs", "")),
                    str(row.get("missing_configs", "")),
                    _fmt(ratio),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Strong stability: ratio >= 0.83, roughly 10/12 configs per dataset.",
            "- Moderate stability: ratio >= 0.67.",
            "- Missing configs should be treated as missing evidence, not negative evidence.",
            "- This experiment tests relative trends, not equality of absolute AFC values across hyperparameters.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _split_items(args.expected_datasets)
    top_ms = _split_ints(args.expected_top_ms)
    eps_values = [float(item) for item in _split_items(args.expected_eps)]
    rows = _load_rows(args.input_csv, str(args.run_id))
    rows = [row for row in rows if str(row.get("dataset")) in set(datasets)]
    claim_rows = _claim_rows(
        rows,
        datasets=datasets,
        top_ms=top_ms,
        eps_values=eps_values,
        gt_branch=str(args.gt_oracle_branch),
        fps_branch=str(args.endpoint_fps_branch),
        full_branch=str(args.full_branch),
        threshold=float(args.limited_gain_threshold),
    )
    _write_csv(output_dir / "afc_exp4_rows.csv", rows)
    _write_csv(output_dir / "afc_exp4_claim_stability.csv", claim_rows)
    (output_dir / "afc_exp4_topm_eps_summary.md").write_text(
        _render_summary(run_id=str(args.run_id), rows=rows, claim_rows=claim_rows, top_ms=top_ms, eps_values=eps_values),
        encoding="utf-8",
    )
    (output_dir / "afc_exp4_rows.json").write_text(
        json.dumps({"rows": rows, "claim_rows": claim_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.no_plots:
        _plot_heatmap(
            rows,
            branch=str(args.gt_oracle_branch),
            metric="dAFC_WMR",
            title="GT oracle: mean Delta AFC-WMR",
            output_base=output_dir / "afc_exp4_heatmap_gt_oracle_delta_wmr",
            top_ms=top_ms,
            eps_values=eps_values,
        )
        _plot_heatmap(
            rows,
            branch=str(args.endpoint_fps_branch),
            metric="dAFC_precision",
            title="Endpoint FPS: mean Delta AFC-Precision",
            output_base=output_dir / "afc_exp4_heatmap_endpoint_fps_delta_precision",
            top_ms=top_ms,
            eps_values=eps_values,
        )
        _plot_heatmap(
            rows,
            branch=str(args.full_branch),
            metric="dAFC_WMR",
            title=f"{args.full_branch}: mean Delta AFC-WMR",
            output_base=output_dir / "afc_exp4_heatmap_full_sampling_delta_wmr",
            top_ms=top_ms,
            eps_values=eps_values,
        )
    print(f"rows_csv={(output_dir / 'afc_exp4_rows.csv').as_posix()}")
    print(f"claims_csv={(output_dir / 'afc_exp4_claim_stability.csv').as_posix()}")
    print(f"summary_md={(output_dir / 'afc_exp4_topm_eps_summary.md').as_posix()}")


if __name__ == "__main__":
    main()
