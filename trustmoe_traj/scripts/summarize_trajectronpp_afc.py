"""Summarize Trajectron++ AFC evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_METRICS: Sequence[str] = (
    "ADE_min",
    "FDE_min",
    "ADE_avg",
    "FDE_avg",
    "MissRate",
    "afc_weighted_mode_recall_eps05",
    "afc_mode_precision_eps05",
    "afc_unsupported_ratio_eps05",
    "afc_chamfer",
    "endpoint_spread",
    "trajectory_spread",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Trajectron++ AFC results.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="eth,hotel,univ,zara1,zara2")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--branch-name", type=str, default="trajectronpp20_pred")
    parser.add_argument(
        "--file-template",
        type=str,
        default="{input_root}/{run_id}_{dataset}_seed{seed}/{dataset}_{split}_trajectronpp_afc.json",
    )
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def _num(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    return None if not kept else sum(kept) / len(kept)


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    if len(kept) <= 1:
        return 0.0 if kept else None
    mean = sum(kept) / len(kept)
    return math.sqrt(sum((item - mean) ** 2 for item in kept) / (len(kept) - 1))


def _fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.6f}"


def _fmt_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "NA"
    if std is None or abs(float(std)) < 1e-12:
        return _fmt(mean)
    return f"{_fmt(mean)}+-{float(std):.6f}"


def _metric(metrics: Mapping[str, Any], branch: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{branch}_{name}"))


def _load_records(
    *,
    input_root: Path,
    run_id: str,
    datasets: Sequence[str],
    seeds: Sequence[str],
    splits: Sequence[str],
    branch: str,
    file_template: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        for seed in seeds:
            for split in splits:
                path = Path(file_template.format(input_root=input_root.as_posix(), run_id=run_id, dataset=dataset, seed=seed, split=split))
                if not path.exists():
                    missing.append(path.as_posix())
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                row: Dict[str, Any] = {"dataset": dataset, "seed": seed, "split": split, "branch": branch, "source_json": path.as_posix()}
                for name in DEFAULT_METRICS:
                    row[name] = _metric(metrics, branch, name)
                rows.append(row)
    return rows, missing


def _aggregate(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["dataset"]), str(row["split"]), str(row["branch"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, split, branch), items in sorted(groups.items()):
        row: Dict[str, Any] = {"dataset": dataset, "split": split, "branch": branch, "n": len(items)}
        for name in DEFAULT_METRICS:
            values = [item.get(name) for item in items]
            row[f"mean_{name}"] = _mean(values)
            row[f"std_{name}"] = _std(values)
        out.append(row)
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["dataset", "split", "branch", "n"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + [field for field in fields if field not in preferred])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in writer.fieldnames})


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> None:
    lines: List[str] = ["# Trajectron++ AFC Summary", ""]
    lines.append("| dataset | split | n | ADE_min | FDE_min | ADE_avg | FDE_avg | AFC-WMR@0.5 | AFC-Precision@0.5 | Unsupported@0.5 | AFC-Chamfer | Endpoint spread | Trajectory spread |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("split", "")),
                    str(row.get("n", "")),
                    _fmt_pm(row.get("mean_ADE_min"), row.get("std_ADE_min")),
                    _fmt_pm(row.get("mean_FDE_min"), row.get("std_FDE_min")),
                    _fmt_pm(row.get("mean_ADE_avg"), row.get("std_ADE_avg")),
                    _fmt_pm(row.get("mean_FDE_avg"), row.get("std_FDE_avg")),
                    _fmt_pm(row.get("mean_afc_weighted_mode_recall_eps05"), row.get("std_afc_weighted_mode_recall_eps05")),
                    _fmt_pm(row.get("mean_afc_mode_precision_eps05"), row.get("std_afc_mode_precision_eps05")),
                    _fmt_pm(row.get("mean_afc_unsupported_ratio_eps05"), row.get("std_afc_unsupported_ratio_eps05")),
                    _fmt_pm(row.get("mean_afc_chamfer"), row.get("std_afc_chamfer")),
                    _fmt_pm(row.get("mean_endpoint_spread"), row.get("std_endpoint_spread")),
                    _fmt_pm(row.get("mean_trajectory_spread"), row.get("std_trajectory_spread")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", "", "- This table evaluates Trajectron++ K=20 sampled prediction sets under the standard AFC protocol."])
    if missing:
        lines.extend(["", "## Missing Files", ""])
        lines.extend(f"- `{item}`" for item in missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "analysis"
    records, missing = _load_records(
        input_root=input_root,
        run_id=str(args.run_id),
        datasets=_split_items(args.datasets),
        seeds=_split_items(args.seeds),
        splits=_split_items(args.splits),
        branch=str(args.branch_name),
        file_template=str(args.file_template),
    )
    rows = _aggregate(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trajectronpp_afc_summary.csv", rows)
    _write_md(output_dir / "trajectronpp_afc_summary.md", rows, missing)
    (output_dir / "trajectronpp_afc_summary.json").write_text(
        json.dumps({"rows": rows, "records": records, "missing": missing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary_md={(output_dir / 'trajectronpp_afc_summary.md').as_posix()}")
    print(f"summary_csv={(output_dir / 'trajectronpp_afc_summary.csv').as_posix()}")
    print(f"summary_json={(output_dir / 'trajectronpp_afc_summary.json').as_posix()}")


if __name__ == "__main__":
    main()
