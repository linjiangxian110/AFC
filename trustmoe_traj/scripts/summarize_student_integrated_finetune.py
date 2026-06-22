"""Aggregate V18-B fine-tuned fast student experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V18-B fine-tuned student outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="student_finetune")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _split_ints(raw: str) -> List[int]:
    return [int(item) for item in _split_items(raw)]


def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [item for item in (_num(value) for value in values) if item is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _fmt(value: Any, *, signed: bool = False, digits: int = 6) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0 else ""
    return f"{prefix}{numeric:.{digits}f}"


def _metric(metrics: Mapping[str, Any], field: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{field}_{name}"))


def _run_id(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{seed}"


def _seed_row(args: argparse.Namespace, project_root: Path, seed: int, splits: Sequence[str]) -> Dict[str, Any]:
    run_id = _run_id(args.run_prefix, seed)
    train_summary_path = (
        project_root
        / "trustmoe_traj"
        / "analysis"
        / "student_integrated_models"
        / run_id
        / f"{args.run_name}_summary.json"
    )
    train_summary = _load_json(train_summary_path) or {}
    row: Dict[str, Any] = {
        "run_id": run_id,
        "seed": int(seed),
        "best_epoch": train_summary.get("meta", {}).get("best_epoch"),
        "best_checkpoint": train_summary.get("meta", {}).get("best_checkpoint"),
        "selection_metric": train_summary.get("meta", {}).get("selection_metric"),
        "best_selection_score": train_summary.get("meta", {}).get("best_selection_score"),
        "train_summary_missing": not train_summary_path.exists(),
    }
    best_val = train_summary.get("best_val_metrics", {})
    if isinstance(best_val, Mapping):
        row["cache_val_finetuned_FDE_min"] = best_val.get("finetuned_FDE_min")
        row["cache_val_dFDE_min"] = best_val.get("dFDE_min")
        row["cache_val_dMissRate"] = best_val.get("dMissRate")
        row["cache_val_student_best_hurt_mean"] = best_val.get("student_best_hurt_mean")
        row["cache_val_student_best_worse_rate"] = best_val.get("student_best_worse_rate")
        row["cache_val_endpoint_ratio"] = best_val.get("endpoint_ratio")
        row["cache_val_trajectory_ratio"] = best_val.get("trajectory_ratio")

    missing: List[str] = []
    for split in splits:
        eval_path = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "eval_results"
            / run_id
            / f"v18b_official_{split}.json"
        )
        payload = _load_json(eval_path)
        if payload is None:
            missing.append(eval_path.as_posix())
            metrics = {}
        else:
            metrics = payload.get("metrics", {})
        split_row: Dict[str, Any] = {}
        if isinstance(metrics, Mapping):
            for metric in METRICS:
                tuned = _metric(metrics, "finetuned_fast_pred", metric)
                fast = _metric(metrics, "fast_pred", metric)
                split_row[f"finetuned_fast_{metric}"] = tuned
                split_row[f"fast_{metric}"] = fast
                split_row[f"d{metric}"] = None if tuned is None or fast is None else tuned - fast
            split_row["slow_FDE_min"] = _metric(metrics, "slow_pred", "FDE_min")
            split_row["finetuned_latency_avg_ms"] = _num(metrics.get("finetuned_fast_pred_latency_avg_ms"))
            split_row["fast_latency_avg_ms"] = _num(metrics.get("fast_pred_latency_avg_ms"))
        row[f"official_{split}"] = split_row
    row["missing_files"] = missing
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        key = f"official_{split}"
        split_row: Dict[str, Any] = {
            "available_official_seeds": sum(1 for row in rows if _num(row.get(key, {}).get("dFDE_min")) is not None)
        }
        for metric in METRICS:
            split_row[f"mean_d{metric}"] = _mean(row.get(key, {}).get(f"d{metric}") for row in rows)
        split_row["mean_finetuned_latency_avg_ms"] = _mean(
            row.get(key, {}).get("finetuned_latency_avg_ms") for row in rows
        )
        split_row["mean_fast_latency_avg_ms"] = _mean(row.get(key, {}).get("fast_latency_avg_ms") for row in rows)
        aggregate[split] = split_row
    return aggregate


def _render(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], splits: Sequence[str]) -> str:
    lines: List[str] = []
    for row in rows:
        lines.append(f"===== {row['run_id']} =====")
        if row.get("missing_files"):
            lines.append("missing summary inputs:")
            for path in row["missing_files"]:
                lines.append(f"  {path}")
        lines.append(f"best_epoch: {row.get('best_epoch')}")
        lines.append(f"best checkpoint: {row.get('best_checkpoint')}")
        lines.append(f"selection_metric: {row.get('selection_metric')}")
        lines.append(f"best_selection_score: {row.get('best_selection_score')}")
        lines.append(f"cache-val finetuned_FDE_min: {row.get('cache_val_finetuned_FDE_min')}")
        lines.append(f"cache-val dFDE_min: {_fmt(row.get('cache_val_dFDE_min'), signed=True)}")
        lines.append(f"cache-val dMissRate: {_fmt(row.get('cache_val_dMissRate'), signed=True)}")
        lines.append(f"cache-val student_best_hurt_mean: {row.get('cache_val_student_best_hurt_mean')}")
        lines.append(f"cache-val student_best_worse_rate: {row.get('cache_val_student_best_worse_rate')}")
        lines.append(f"cache-val endpoint_ratio: {row.get('cache_val_endpoint_ratio')}")
        lines.append(f"cache-val trajectory_ratio: {row.get('cache_val_trajectory_ratio')}")
        lines.append("")
        for split in splits:
            official = row.get(f"official_{split}", {})
            lines.append(f"-- official {split} FinetunedFast - Fast --")
            for metric in METRICS:
                lines.append(
                    f"d{metric}: {_fmt(official.get(f'd{metric}'), signed=True)}  "
                    f"finetuned_fast={_fmt(official.get(f'finetuned_fast_{metric}'))}  "
                    f"fast={_fmt(official.get(f'fast_{metric}'))}"
                )
            lines.append(f"slow_FDE_min: {official.get('slow_FDE_min')}")
            lines.append(f"finetuned_latency_avg_ms: {official.get('finetuned_latency_avg_ms')}")
            lines.append(f"fast_latency_avg_ms: {official.get('fast_latency_avg_ms')}")
            lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        mean = aggregate.get(split, {})
        lines.append("")
        lines.append(f"-- {split} --")
        lines.append(f"available official seeds: {int(mean.get('available_official_seeds') or 0)}/{len(rows)}")
        for metric in METRICS:
            lines.append(f"mean d{metric}: {_fmt(mean.get(f'mean_d{metric}'), signed=True)}")
        lines.append(f"mean finetuned_latency_avg_ms: {_fmt(mean.get('mean_finetuned_latency_avg_ms'))}")
        lines.append(f"mean fast_latency_avg_ms: {_fmt(mean.get('mean_fast_latency_avg_ms'))}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    rows = [_seed_row(args, project_root, seed, splits) for seed in seeds]
    aggregate = _aggregate(rows, splits)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_student_integrated_finetune",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": args.run_prefix,
            "run_name": args.run_name,
            "seeds": seeds,
            "splits": splits,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    default_root = project_root / "trustmoe_traj" / "analysis" / "experiment_runs" / args.run_prefix
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_root / "aggregate_summary.json"
    output_txt = Path(args.output_txt).expanduser().resolve() if args.output_txt else default_root / "aggregate_summary.txt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered = _render(rows, aggregate, splits)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
