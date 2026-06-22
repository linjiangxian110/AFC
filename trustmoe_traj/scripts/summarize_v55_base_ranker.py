"""Aggregate V55-D base-ranker train/eval outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
BRANCHES: Sequence[str] = (
    "v55d_ranker_top5x4_pred",
    "v55d_ranker_top4x4_next4slot0_pred",
    "v55d_ranker_top3x4_next8slot0_pred",
    "v55d_ranker_top10x2_slot01_pred",
    "v55c_oracle_base_top5x4_pred",
    "v55c_teacher_order_top5x4_pred",
    "v38_slot0_20_pred",
    "v38_oracle20_from80_pred",
    "v38_full80_pred",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V55-D high-potential base ranker outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="v55_base_ranker")
    parser.add_argument("--eval-file-prefix", type=str, default="v55d_official")
    parser.add_argument("--seeds", type=str, default="0,1,2")
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


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0 else ""
    return f"{prefix}{numeric:.6f}"


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
        / "v55_base_ranker_models"
        / run_id
        / f"{args.run_name}_summary.json"
    )
    train_summary = _load_json(train_summary_path) or {}
    row: Dict[str, Any] = {
        "run_id": run_id,
        "seed": int(seed),
        "best_epoch": train_summary.get("meta", {}).get("best_epoch"),
        "best_checkpoint": train_summary.get("meta", {}).get("best_checkpoint"),
        "refiner_checkpoint": train_summary.get("meta", {}).get("refiner_checkpoint"),
        "best_selection_score": train_summary.get("meta", {}).get("best_selection_score"),
        "train_summary_missing": not train_summary_path.exists(),
    }
    best_val = train_summary.get("best_val_metrics", {})
    if isinstance(best_val, Mapping):
        for key in (
            "top1_acc",
            "topk_best_hit",
            "topk_recall",
            "top5x4_dFDE_min",
            "top5x4_dMissRate",
            "top10x2_slot01_dFDE_min",
        ):
            row[f"cache_val_{key}"] = best_val.get(key)

    missing: List[str] = []
    for split in splits:
        eval_path = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "eval_results"
            / run_id
            / f"{args.eval_file_prefix}_{split}.json"
        )
        payload = _load_json(eval_path)
        if payload is None:
            missing.append(eval_path.as_posix())
            metrics: Mapping[str, Any] = {}
        else:
            raw_metrics = payload.get("metrics", {})
            metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        split_row: Dict[str, Any] = {"slow": {metric: _metric(metrics, "slow_pred", metric) for metric in METRICS}}
        split_row["slow"]["latency_avg_ms"] = _num(metrics.get("slow_pred_latency_avg_ms"))
        for branch in BRANCHES:
            branch_row: Dict[str, Any] = {}
            for metric in METRICS:
                value = _metric(metrics, branch, metric)
                slow = _metric(metrics, "slow_pred", metric)
                branch_row[metric] = value
                branch_row[f"d{metric}"] = None if value is None or slow is None else value - slow
            for aux in (
                "latency_avg_ms",
                "delta_l2_mean",
                "endpoint_ratio",
                "trajectory_ratio",
                "unique_base_mode_ratio",
                "ranker_top1_acc",
                "ranker_topk_best_hit",
                "ranker_topk_recall",
            ):
                branch_row[aux] = _num(metrics.get(f"{branch}_{aux}"))
            split_row[branch] = branch_row
        row[f"official_{split}"] = split_row
    row["missing_files"] = missing
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        split_key = f"official_{split}"
        split_agg: Dict[str, Any] = {}
        for branch in BRANCHES:
            branch_agg: Dict[str, Any] = {
                "available_official_seeds": sum(
                    1 for row in rows if _num(row.get(split_key, {}).get(branch, {}).get("dFDE_min")) is not None
                )
            }
            for metric in METRICS:
                branch_agg[f"mean_{metric}"] = _mean(row.get(split_key, {}).get(branch, {}).get(metric) for row in rows)
                branch_agg[f"mean_d{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(f"d{metric}") for row in rows
                )
            for aux in (
                "latency_avg_ms",
                "delta_l2_mean",
                "endpoint_ratio",
                "trajectory_ratio",
                "unique_base_mode_ratio",
                "ranker_top1_acc",
                "ranker_topk_best_hit",
                "ranker_topk_recall",
            ):
                branch_agg[f"mean_{aux}"] = _mean(row.get(split_key, {}).get(branch, {}).get(aux) for row in rows)
            split_agg[branch] = branch_agg
        aggregate[split] = split_agg
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
        lines.append(f"refiner checkpoint: {row.get('refiner_checkpoint')}")
        lines.append(f"best_selection_score: {row.get('best_selection_score')}")
        lines.append(f"cache-val top1_acc: {_fmt(row.get('cache_val_top1_acc'))}")
        lines.append(f"cache-val top5_best_hit: {_fmt(row.get('cache_val_topk_best_hit'))}")
        lines.append(f"cache-val top5_recall: {_fmt(row.get('cache_val_topk_recall'))}")
        lines.append(f"cache-val top5x4 dFDE_min: {_fmt(row.get('cache_val_top5x4_dFDE_min'), signed=True)}")
        lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        lines.append("")
        lines.append(f"-- {split} --")
        split_agg = aggregate.get(split, {})
        for branch in BRANCHES:
            mean = split_agg.get(branch, {})
            lines.append(f"{branch}: available={int(mean.get('available_official_seeds') or 0)}/{len(rows)}")
            for metric in METRICS:
                lines.append(
                    f"  mean d{metric}: {_fmt(mean.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(mean.get(f'mean_{metric}'))}"
                )
            if branch.startswith("v55d_ranker_"):
                lines.append(f"  mean ranker_top1_acc: {_fmt(mean.get('mean_ranker_top1_acc'))}")
                lines.append(f"  mean ranker_topk_best_hit: {_fmt(mean.get('mean_ranker_topk_best_hit'))}")
                lines.append(f"  mean ranker_topk_recall: {_fmt(mean.get('mean_ranker_topk_recall'))}")
            if mean.get("mean_unique_base_mode_ratio") is not None:
                lines.append(f"  mean unique_base_mode_ratio: {_fmt(mean.get('mean_unique_base_mode_ratio'))}")
            lines.append(f"  mean latency_avg_ms: {_fmt(mean.get('mean_latency_avg_ms'))}")
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
            "script": "trustmoe_traj.scripts.summarize_v55_base_ranker",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": args.run_prefix,
            "run_name": args.run_name,
            "eval_file_prefix": args.eval_file_prefix,
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
