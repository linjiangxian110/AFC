"""Summarize AFC Experiment 7 leakage and robustness audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SETTINGS: Sequence[str] = (
    "default_train_bank",
    "same_source_filtered",
    "temporal_gap_filtered",
    "same_source_temporal_filtered",
    "scene_exclusion",
    "randomized_bank",
)

SETTING_LABELS: Mapping[str, str] = {
    "default_train_bank": "Default train-bank AFC",
    "same_source_filtered": "Same-source filtered",
    "temporal_gap_filtered": "Temporal-gap filtered",
    "same_source_temporal_filtered": "Same-source + temporal filtered",
    "scene_exclusion": "Scene exclusion",
    "randomized_bank": "Randomized bank sanity check",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AFC Experiment 7 leakage/robustness JSON files.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--settings", type=str, default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--datasets", type=str, default="zara1")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument(
        "--file-template",
        type=str,
        default="{input_root}/{run_id}_{setting}_{dataset}_seed{seed}/{dataset}_{split}_headroom.json",
    )
    parser.add_argument("--eps-label", type=str, default="eps05")
    parser.add_argument("--slow-branch", type=str, default="slow20_pred")
    parser.add_argument("--cv-branch", type=str, default="cv_linear20_pred")
    parser.add_argument("--gt-oracle-branch", type=str, default="slow100_gt_oracle20_pred")
    parser.add_argument("--fps-branch", type=str, default="slow100_endpoint_fps20_pred")
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


def _trend_label(row: Mapping[str, Any]) -> str:
    if str(row.get("setting")) == "randomized_bank":
        return "sanity"
    checks = [
        _num(row.get("slow20_minus_cv_wmr")),
        _num(row.get("gt_oracle_afc_drop")),
    ]
    kept = [item for item in checks if item is not None]
    if not kept:
        return "unknown"
    return "yes" if all(item > 0 for item in kept) else "mixed"


def _load_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    input_root = Path(args.input_root).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    eps = str(args.eps_label)
    for setting in _split_items(args.settings):
        for dataset in _split_items(args.datasets):
            for seed in _split_items(args.seeds):
                for split in _split_items(args.splits):
                    path = Path(
                        str(args.file_template).format(
                            input_root=input_root.as_posix(),
                            run_id=str(args.run_id),
                            setting=setting,
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
                    meta = payload.get("meta", {})
                    branches = [str(item) for item in payload.get("branches", [])]
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
                        "setting": setting,
                        "setting_label": SETTING_LABELS.get(setting, setting),
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "source_json": path.as_posix(),
                        "afc_train_split": meta.get("afc_train_split", payload.get("args", {}).get("afc_train_split", "train")),
                        "afc_source_id_field": meta.get("afc_source_id_field", payload.get("args", {}).get("afc_source_id_field")),
                        "afc_temporal_gap_frames": meta.get("afc_temporal_gap_frames", payload.get("args", {}).get("afc_temporal_gap_frames")),
                        "slow20_wmr": slow_wmr,
                        "slow20_precision": _metric(metrics, slow, f"afc_mode_precision_{eps}"),
                        "slow20_unsupported": slow_unsupported,
                        "slow20_chamfer": _metric(metrics, slow, "afc_chamfer"),
                        "bank_size": _metric(metrics, slow, "afc_bank_size"),
                        "raw_query_count": _metric(metrics, slow, "afc_raw_query_count"),
                        "valid_query_count": _metric(metrics, slow, "afc_valid_query_count"),
                        "invalid_query_count": _metric(metrics, slow, "afc_retrieval_invalid_query_count"),
                        "candidate_count": _metric(metrics, slow, "afc_retrieval_candidate_count"),
                        "min_candidate_count": _metric(metrics, slow, "afc_retrieval_min_candidate_count"),
                        "finite_fraction": _metric(metrics, slow, "afc_retrieval_finite_fraction"),
                        "retrieval_confidence": _metric(metrics, slow, "afc_retrieval_confidence"),
                        "slow20_minus_cv_wmr": None if slow_wmr is None or cv_wmr is None else slow_wmr - cv_wmr,
                        "gt_oracle_afc_drop": None if slow_wmr is None or gt_wmr is None else slow_wmr - gt_wmr,
                        "fps_unsupported_gap": None
                        if fps_unsupported is None or slow_unsupported is None
                        else fps_unsupported - slow_unsupported,
                    }
                    row["main_trend_preserved"] = _trend_label(row)
                    rows.append(row)
    return rows, missing


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "slow20_wmr",
        "slow20_precision",
        "slow20_unsupported",
        "slow20_chamfer",
        "bank_size",
        "raw_query_count",
        "valid_query_count",
        "invalid_query_count",
        "candidate_count",
        "min_candidate_count",
        "finite_fraction",
        "retrieval_confidence",
        "slow20_minus_cv_wmr",
        "gt_oracle_afc_drop",
        "fps_unsupported_gap",
    ]
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["setting"]), str(row["dataset"]), str(row["split"]))].append(row)
        groups[(str(row["setting"]), "ALL", str(row["split"]))].append(row)
    setting_order = {setting: index for index, setting in enumerate(DEFAULT_SETTINGS)}
    out_rows: List[Dict[str, Any]] = []
    for (setting, dataset, split), items in sorted(
        groups.items(),
        key=lambda item: (item[0][1] != "ALL", item[0][1], setting_order.get(item[0][0], 999), item[0][2]),
    ):
        out: Dict[str, Any] = {
            "setting": setting,
            "setting_label": SETTING_LABELS.get(setting, setting),
            "dataset": dataset,
            "split": split,
            "n": len(items),
        }
        for metric in metrics:
            out[metric] = _mean(item.get(metric) for item in items)
        trend_labels = [str(item.get("main_trend_preserved", "unknown")) for item in items]
        if trend_labels and all(item == trend_labels[0] for item in trend_labels):
            out["main_trend_preserved"] = trend_labels[0]
        elif trend_labels:
            out["main_trend_preserved"] = "mixed"
        else:
            out["main_trend_preserved"] = "unknown"
        out_rows.append(out)
    return out_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "setting",
        "setting_label",
        "dataset",
        "split",
        "n",
        "slow20_wmr",
        "slow20_precision",
        "slow20_unsupported",
        "slow20_chamfer",
        "bank_size",
        "raw_query_count",
        "valid_query_count",
        "invalid_query_count",
        "candidate_count",
        "min_candidate_count",
        "finite_fraction",
        "retrieval_confidence",
        "slow20_minus_cv_wmr",
        "gt_oracle_afc_drop",
        "fps_unsupported_gap",
        "main_trend_preserved",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# AFC Experiment 7 Leakage / Robustness Summary")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    lines.append("## Robustness Table")
    lines.append("")
    lines.append(
        "| dataset | setting | n | WMR@0.5 | Precision@0.5 | Unsupported@0.5 | Chamfer | "
        "bank | valid/raw queries | candidate mean/min | finite | slow-CV WMR gap | GT-oracle drop | FPS unsupported gap | trend |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        valid_raw = f"{_fmt(_num(row.get('valid_query_count')))}/{_fmt(_num(row.get('raw_query_count')))}"
        cand = f"{_fmt(_num(row.get('candidate_count')))}/{_fmt(_num(row.get('min_candidate_count')))}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("setting_label", "")),
                    str(row.get("n", "")),
                    _fmt(_num(row.get("slow20_wmr"))),
                    _fmt(_num(row.get("slow20_precision"))),
                    _fmt(_num(row.get("slow20_unsupported"))),
                    _fmt(_num(row.get("slow20_chamfer"))),
                    _fmt(_num(row.get("bank_size"))),
                    valid_raw,
                    cand,
                    _fmt(_num(row.get("finite_fraction"))),
                    _fmt(_num(row.get("slow20_minus_cv_wmr")), signed=True),
                    _fmt(_num(row.get("gt_oracle_afc_drop")), signed=True),
                    _fmt(_num(row.get("fps_unsupported_gap")), signed=True),
                    str(row.get("main_trend_preserved", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Non-randomized filtering rows should preserve the main relative trends, not identical absolute AFC values.")
    lines.append("- Randomized-bank rows are a sanity check: weaker or unstable AFC evidence is expected when feature/future correspondence is broken.")
    lines.append("- Candidate mean/min and invalid-query counts are audit fields; low candidate counts should be reported before interpreting metric changes.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    rows, missing = _load_rows(args)
    aggregated = _aggregate(rows)
    output_dir = Path(args.output_dir or Path(args.input_root) / "analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "afc_exp7_leakage_robustness_summary.csv"
    md_path = output_dir / "afc_exp7_leakage_robustness_summary.md"
    json_path = output_dir / "afc_exp7_leakage_robustness_summary.json"
    _write_csv(csv_path, aggregated)
    md_path.write_text(_render_markdown(aggregated, missing), encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": rows, "aggregated": aggregated, "missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary_csv={csv_path.as_posix()}")
    print(f"summary_md={md_path.as_posix()}")
    print(f"summary_json={json_path.as_posix()}")


if __name__ == "__main__":
    main()
