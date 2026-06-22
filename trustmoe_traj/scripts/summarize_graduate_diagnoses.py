"""Summarize Residual Graduate diagnose JSON files across runs.

This helper reads outputs from ``diagnose_residual_graduate.py`` and prints a
compact Markdown table focused on best-of-K regressions, student-best hurt, and
mode diversity shrinkage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_METRICS: Sequence[str] = (
    "fde_min_delta_mean",
    "fde_min_worse_rate",
    "fde_avg_delta_mean",
    "student_best_mode_fde_delta_mean",
    "student_best_mode_worse_rate",
    "student_best_hurt_and_fde_min_worse_rate",
    "best_mode_switch_rate",
    "endpoint_spread_ratio_mean",
    "trajectory_diversity_ratio_mean",
    "gate_mean",
    "delta_l2_mean",
    "pearson_student_best_fde_delta_vs_fde_min_delta",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Residual Graduate diagnose JSON files.")
    parser.add_argument(
        "--diagnose-json",
        action="append",
        default=[],
        help=(
            "Diagnose JSON path. Can be repeated. Optionally prefix with LABEL=, "
            "for example v7_val=path/to/diagnose_val.json."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "Run eval directory containing diagnose_val.json / diagnose_test.json. "
            "Can be repeated. Optionally prefix with LABEL=."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated split names to load from each --run-dir. Default: val,test.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric key to include. Can be repeated. Defaults to the core diagnose metrics.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path for machine-readable rows.")
    return parser


def _split_label_path(spec: str) -> tuple[Optional[str], Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in spec: {spec!r}")
        return label, Path(raw_path).expanduser()
    return None, Path(spec).expanduser()


def _load_summary(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"Diagnose JSON missing summary object: {path}")
    return summary


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6f}"


def _default_label(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if parent:
        return f"{parent}/{stem}"
    return stem


def _collect_json_specs(args: argparse.Namespace) -> List[tuple[str, Path]]:
    specs: List[tuple[str, Path]] = []

    for item in args.diagnose_json:
        label, path = _split_label_path(item)
        specs.append((label or _default_label(path), path))

    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    for item in args.run_dir:
        label, run_dir = _split_label_path(item)
        for split in splits:
            path = run_dir / f"diagnose_{split}.json"
            row_label = f"{label or run_dir.name}_{split}"
            specs.append((row_label, path))

    if not specs:
        raise SystemExit("Provide at least one --diagnose-json or --run-dir")
    return specs


def _build_rows(specs: Iterable[tuple[str, Path]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"Diagnose JSON not found: {path}")
        summary = _load_summary(path)
        row: Dict[str, Any] = {
            "label": label,
            "path": path.as_posix(),
        }
        for metric in metrics:
            row[metric] = summary.get(metric)
        rows.append(row)
    return rows


def _print_markdown(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> None:
    headers = ["label", *metrics]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---", *[":---:" for _ in metrics]]) + " |")
    for row in rows:
        values = [str(row.get("label", ""))]
        values.extend(_format_float(row.get(metric)) for metric in metrics)
        print("| " + " | ".join(values) + " |")


def main() -> None:
    args = build_parser().parse_args()
    metrics = tuple(args.metric) if args.metric else tuple(DEFAULT_METRICS)
    specs = _collect_json_specs(args)
    rows = _build_rows(specs, metrics)
    _print_markdown(rows, metrics)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\noutput_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
