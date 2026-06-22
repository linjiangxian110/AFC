"""Summarize AFC Experiment 6 retrieval-feature ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trustmoe_traj.scripts.analogical_future_coverage import AFC_FEATURE_VARIANTS


FEATURE_LABELS: Mapping[str, str] = {
    "past_shape": "Past shape only",
    "past_velocity": "Past + velocity",
    "past_velocity_accel": "Past + velocity + acceleration",
    "past_velocity_social": "Past + velocity + social risk",
    "full_past_social": "Full past-social feature",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AFC Experiment 6 feature-ablation JSON files.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="zara1")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--feature-variants", type=str, default=",".join(AFC_FEATURE_VARIANTS))
    parser.add_argument(
        "--file-template",
        type=str,
        default="{input_root}/{run_id}_{feature_variant}_{dataset}_seed{seed}/{dataset}_{split}_headroom.json",
    )
    parser.add_argument("--eps-label", type=str, default="eps05")
    parser.add_argument("--slow-branch", type=str, default="slow20_pred")
    parser.add_argument("--cv-branch", type=str, default="cv_linear20_pred")
    parser.add_argument("--gt-oracle-branch", type=str, default="slow100_gt_oracle20_pred")
    parser.add_argument("--fps-branch", type=str, default="slow100_endpoint_fps20_pred")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _metric(metrics: Mapping[str, Any], branch: str, metric: str) -> Optional[float]:
    return _num(metrics.get(f"{branch}_{metric}"))


def _choose_branch(branches: Sequence[str], requested: str, contains: str) -> Optional[str]:
    if requested in branches:
        return requested
    candidates = [branch for branch in branches if contains in branch]
    if not candidates:
        return None

    def pool_size(branch: str) -> int:
        digits = ""
        for char in branch:
            if char.isdigit():
                digits += char
            elif digits:
                break
        return int(digits or 0)

    return sorted(candidates, key=lambda item: (pool_size(item), item))[0]


def _load_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    input_root = Path(args.input_root).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for feature_variant in _split_items(args.feature_variants):
        if feature_variant not in AFC_FEATURE_VARIANTS:
            raise SystemExit(f"Unsupported feature variant: {feature_variant}")
        for dataset in _split_items(args.datasets):
            for seed in _split_items(args.seeds):
                for split in _split_items(args.splits):
                    path = Path(
                        str(args.file_template).format(
                            input_root=input_root.as_posix(),
                            run_id=str(args.run_id),
                            feature_variant=feature_variant,
                            dataset=dataset,
                            seed=seed,
                            split=split,
                        )
                    )
                    if not path.exists():
                        missing.append(path.as_posix())
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    metrics = payload.get("metrics", {})
                    branches = [str(item) for item in payload.get("branches", [])]
                    eps = str(args.eps_label)
                    slow = str(args.slow_branch)
                    cv = _choose_branch(branches, str(args.cv_branch), "cv_linear")
                    gt = _choose_branch(branches, str(args.gt_oracle_branch), "_gt_oracle20_pred")
                    fps = _choose_branch(branches, str(args.fps_branch), "_endpoint_fps20_pred")
                    slow_wmr = _metric(metrics, slow, f"afc_weighted_mode_recall_{eps}")
                    cv_wmr = _metric(metrics, cv or "", f"afc_weighted_mode_recall_{eps}") if cv else None
                    gt_wmr = _metric(metrics, gt or "", f"afc_weighted_mode_recall_{eps}") if gt else None
                    slow_unsupported = _metric(metrics, slow, f"afc_unsupported_ratio_{eps}")
                    fps_unsupported = _metric(metrics, fps or "", f"afc_unsupported_ratio_{eps}") if fps else None
                    row: Dict[str, Any] = {
                        "feature_variant": feature_variant,
                        "feature_label": FEATURE_LABELS.get(feature_variant, feature_variant),
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "source_json": path.as_posix(),
                        "slow_branch": slow,
                        "cv_branch": cv,
                        "gt_oracle_branch": gt,
                        "fps_branch": fps,
                        "retrieval_confidence": _metric(metrics, slow, "afc_retrieval_confidence"),
                        "retrieval_top1_distance": _metric(metrics, slow, "afc_retrieval_top1_distance"),
                        "retrieval_effective_m": _metric(metrics, slow, "afc_retrieval_effective_m"),
                        "future_mode_count": _metric(metrics, slow, f"afc_mode_count_{eps}"),
                        "intra_cluster_distance": _metric(metrics, slow, f"afc_mode_intra_distance_{eps}"),
                        "mode_entropy": _metric(metrics, slow, f"afc_mode_entropy_{eps}"),
                        "slow20_wmr": slow_wmr,
                        "cv_wmr": cv_wmr,
                        "gt_oracle_wmr": gt_wmr,
                        "slow20_unsupported": slow_unsupported,
                        "fps_unsupported": fps_unsupported,
                        "slow20_minus_cv_wmr": None if slow_wmr is None or cv_wmr is None else slow_wmr - cv_wmr,
                        "gt_oracle_afc_drop": None if slow_wmr is None or gt_wmr is None else slow_wmr - gt_wmr,
                        "fps_unsupported_gap": None
                        if fps_unsupported is None or slow_unsupported is None
                        else fps_unsupported - slow_unsupported,
                    }
                    row["discriminability_score"] = _mean(
                        [
                            row["slow20_minus_cv_wmr"],
                            row["gt_oracle_afc_drop"],
                            row["fps_unsupported_gap"],
                        ]
                    )
                    rows.append(row)
    return rows, missing


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "retrieval_confidence",
        "retrieval_top1_distance",
        "retrieval_effective_m",
        "future_mode_count",
        "intra_cluster_distance",
        "mode_entropy",
        "slow20_minus_cv_wmr",
        "gt_oracle_afc_drop",
        "fps_unsupported_gap",
        "discriminability_score",
    ]
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["feature_variant"]), str(row["dataset"]), str(row["split"]))].append(row)
        groups[(str(row["feature_variant"]), "ALL", str(row["split"]))].append(row)
    out_rows: List[Dict[str, Any]] = []
    order = {variant: index for index, variant in enumerate(AFC_FEATURE_VARIANTS)}
    for (feature_variant, dataset, split), items in sorted(
        groups.items(),
        key=lambda item: (item[0][1] != "ALL", item[0][1], order.get(item[0][0], 999), item[0][2]),
    ):
        out: Dict[str, Any] = {
            "feature_variant": feature_variant,
            "feature_label": FEATURE_LABELS.get(feature_variant, feature_variant),
            "dataset": dataset,
            "split": split,
            "n": len(items),
        }
        for metric in metrics:
            out[metric] = _mean(item.get(metric) for item in items)
        out_rows.append(out)
    return out_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "feature_variant",
        "feature_label",
        "dataset",
        "split",
        "n",
        "retrieval_confidence",
        "retrieval_top1_distance",
        "retrieval_effective_m",
        "future_mode_count",
        "intra_cluster_distance",
        "mode_entropy",
        "slow20_minus_cv_wmr",
        "gt_oracle_afc_drop",
        "fps_unsupported_gap",
        "discriminability_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# AFC Experiment 6 Feature Ablation Summary")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    lines.append("## Ablation Table")
    lines.append("")
    lines.append(
        "| dataset | feature | n | retrieval conf ↑ | top1 dist ↓ | effective M ↑ | "
        "mode count | intra dist ↓ | entropy ↑/→ | slow-CV WMR gap ↑ | GT-oracle drop ↑ | FPS unsupported gap ↑ | score ↑ |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("feature_label", "")),
                    str(row.get("n", "")),
                    _fmt(_num(row.get("retrieval_confidence"))),
                    _fmt(_num(row.get("retrieval_top1_distance"))),
                    _fmt(_num(row.get("retrieval_effective_m"))),
                    _fmt(_num(row.get("future_mode_count"))),
                    _fmt(_num(row.get("intra_cluster_distance"))),
                    _fmt(_num(row.get("mode_entropy"))),
                    _fmt(_num(row.get("slow20_minus_cv_wmr")), signed=True),
                    _fmt(_num(row.get("gt_oracle_afc_drop")), signed=True),
                    _fmt(_num(row.get("fps_unsupported_gap")), signed=True),
                    _fmt(_num(row.get("discriminability_score")), signed=True),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `slow-CV WMR gap` tests whether the retrieval feature separates base MoFlow from a weak constant-velocity predictor.")
    lines.append("- `GT-oracle drop` tests whether the feature still exposes GT-centric coverage collapse.")
    lines.append("- `FPS unsupported gap` tests whether the feature still flags geometric fake diversity.")
    lines.append("- The full feature is not required to win every column, but it should keep or improve retrieval confidence/mode quality while preserving discriminability.")
    return "\n".join(lines).rstrip() + "\n"


def _write_plot(path_base: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"plot_skipped={exc}")
        return

    plot_rows = [row for row in rows if str(row.get("dataset")) == "ALL"]
    if not plot_rows:
        plot_rows = list(rows)
    plot_rows = [row for row in plot_rows if _num(row.get("discriminability_score")) is not None]
    if not plot_rows:
        return

    labels = [str(row.get("feature_label", row.get("feature_variant", ""))) for row in plot_rows]
    metrics = [
        ("slow20_minus_cv_wmr", "slow-CV WMR gap"),
        ("gt_oracle_afc_drop", "GT-oracle drop"),
        ("fps_unsupported_gap", "FPS unsupported gap"),
    ]
    x = list(range(len(plot_rows)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(plot_rows)), 4.2))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for metric_index, (key, label) in enumerate(metrics):
        values = [_num(row.get(key)) or 0.0 for row in plot_rows]
        offsets = [item + (metric_index - 1) * width for item in x]
        ax.bar(offsets, values, width=width, label=label, color=colors[metric_index], alpha=0.9)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("diagnostic gap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=1)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"plot={path_base.with_suffix('.png').as_posix()}")
    print(f"plot={path_base.with_suffix('.pdf').as_posix()}")


def main() -> None:
    args = build_parser().parse_args()
    rows, missing = _load_rows(args)
    aggregate = _aggregate(rows)
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "afc_exp6_feature_ablation_summary.csv"
    md_path = output_dir / "afc_exp6_feature_ablation_summary.md"
    json_path = output_dir / "afc_exp6_feature_ablation_summary.json"
    _write_csv(csv_path, aggregate)
    md_path.write_text(_render_markdown(aggregate, missing), encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": aggregate, "raw_rows": rows, "missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not bool(args.no_plots):
        _write_plot(output_dir / "afc_exp6_discriminability_bars", aggregate)
    print(f"summary_csv={csv_path.as_posix()}")
    print(f"summary_md={md_path.as_posix()}")
    print(f"summary_json={json_path.as_posix()}")
    if missing:
        print("missing_inputs=" + ",".join(missing))


if __name__ == "__main__":
    main()
