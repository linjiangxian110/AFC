"""Aggregate per-base residual slot selector train/eval outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
SELECTOR_BRANCH_BY_FAMILY: Mapping[str, str] = {
    "v55a": "v55_slot_selector20_pred",
    "v57b": "v57b_semantic_slot_selector20_pred",
    "v57c": "v57c_conservative_semantic_slot_selector20_pred",
}


def _branches(branch_family: str) -> Sequence[str]:
    if branch_family == "v57c":
        return (
            "v57c_conservative_semantic_slot_selector20_pred",
            "v57c_per_base_oracle20_pred",
            "v57a_slot0_20_pred",
            "v57a_oracle20_from_semantic_pool_pred",
            "v57a_full_semantic_pool_pred",
        )
    if branch_family == "v57b":
        return (
            "v57b_semantic_slot_selector20_pred",
            "v57b_per_base_oracle20_pred",
            "v57a_slot0_20_pred",
            "v57a_oracle20_from_semantic_pool_pred",
            "v57a_full_semantic_pool_pred",
        )
    return (
        "v55_slot_selector20_pred",
        "v55_per_base_oracle20_pred",
        "v38_slot0_20_pred",
        "v38_oracle20_from80_pred",
        "v38_full80_pred",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize per-base residual slot selector outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="social_cvae_selector")
    parser.add_argument("--eval-file-prefix", type=str, default="v55a_official")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--branch-family", type=str, default="v55a", choices=["v55a", "v57b", "v57c"])
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


def _seed_row(
    args: argparse.Namespace,
    project_root: Path,
    seed: int,
    splits: Sequence[str],
    branches: Sequence[str],
) -> Dict[str, Any]:
    run_id = _run_id(args.run_prefix, seed)
    train_summary_path = (
        project_root
        / "trustmoe_traj"
        / "analysis"
        / "social_cvae_selector_models"
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
        row["cache_val_dFDE_min"] = best_val.get("dFDE_min")
        row["cache_val_dMissRate"] = best_val.get("dMissRate")
        row["cache_val_target_accuracy"] = best_val.get("target_accuracy")
        row["cache_val_target_mean_ratio"] = best_val.get("target_mean_ratio")
        row["cache_val_selected_mean_ratio"] = best_val.get("selected_mean_ratio")
        row["cache_val_target_utility_gain"] = best_val.get("target_utility_gain")

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
        split_row: Dict[str, Any] = {}
        for branch in branches:
            branch_row: Dict[str, Any] = {}
            for metric in METRICS:
                value = _metric(metrics, branch, metric)
                slow = _metric(metrics, "slow_pred", metric)
                branch_row[metric] = value
                branch_row[f"d{metric}"] = None if value is None or slow is None else value - slow
            branch_row["latency_avg_ms"] = _num(metrics.get(f"{branch}_latency_avg_ms"))
            branch_row["delta_l2_mean"] = _num(metrics.get(f"{branch}_delta_l2_mean"))
            branch_row["endpoint_ratio"] = _num(metrics.get(f"{branch}_endpoint_ratio"))
            branch_row["trajectory_ratio"] = _num(metrics.get(f"{branch}_trajectory_ratio"))
            branch_row["unique_base_mode_ratio"] = _num(metrics.get(f"{branch}_unique_base_mode_ratio"))
            branch_row["selected_slot_mean"] = _num(metrics.get(f"{branch}_selected_slot_mean"))
            branch_row["selected_slot0_ratio"] = _num(metrics.get(f"{branch}_selected_slot0_ratio"))
            branch_row["per_base_oracle_slot_accuracy"] = _num(
                metrics.get(f"{branch}_per_base_oracle_slot_accuracy")
            )
            split_row[branch] = branch_row
        split_row["slow"] = {
            metric: _metric(metrics, "slow_pred", metric)
            for metric in METRICS
        }
        split_row["slow"]["latency_avg_ms"] = _num(metrics.get("slow_pred_latency_avg_ms"))
        row[f"official_{split}"] = split_row
    row["missing_files"] = missing
    return row


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    branches: Sequence[str],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        split_key = f"official_{split}"
        split_agg: Dict[str, Any] = {}
        for branch in branches:
            branch_agg: Dict[str, Any] = {
                "available_official_seeds": sum(
                    1 for row in rows if _num(row.get(split_key, {}).get(branch, {}).get("dFDE_min")) is not None
                )
            }
            for metric in METRICS:
                branch_agg[f"mean_d{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(f"d{metric}") for row in rows
                )
                branch_agg[f"mean_{metric}"] = _mean(
                    row.get(split_key, {}).get(branch, {}).get(metric) for row in rows
                )
            for aux in (
                "latency_avg_ms",
                "delta_l2_mean",
                "endpoint_ratio",
                "trajectory_ratio",
                "unique_base_mode_ratio",
                "selected_slot_mean",
                "selected_slot0_ratio",
                "per_base_oracle_slot_accuracy",
            ):
                branch_agg[f"mean_{aux}"] = _mean(row.get(split_key, {}).get(branch, {}).get(aux) for row in rows)
            split_agg[branch] = branch_agg
        aggregate[split] = split_agg
    return aggregate


def _render(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    splits: Sequence[str],
    branches: Sequence[str],
    selector_branch: str,
) -> str:
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
        lines.append(f"cache-val dFDE_min: {_fmt(row.get('cache_val_dFDE_min'), signed=True)}")
        lines.append(f"cache-val target_accuracy: {_fmt(row.get('cache_val_target_accuracy'))}")
        lines.append(f"cache-val target_slot0_ratio: {_fmt(row.get('cache_val_target_mean_ratio'))}")
        lines.append(f"cache-val selected_slot0_ratio: {_fmt(row.get('cache_val_selected_mean_ratio'))}")
        lines.append(f"cache-val target_utility_gain: {_fmt(row.get('cache_val_target_utility_gain'))}")
        lines.append("")
        for split in splits:
            official = row.get(f"official_{split}", {})
            for branch in branches:
                branch_row = official.get(branch, {})
                lines.append(f"-- official {split} {branch} - Slow --")
                for metric in METRICS:
                    lines.append(
                        f"d{metric}: {_fmt(branch_row.get(f'd{metric}'), signed=True)}  "
                        f"branch={_fmt(branch_row.get(metric))}  "
                        f"slow={_fmt(official.get('slow', {}).get(metric))}"
                    )
                if branch == selector_branch:
                    lines.append(f"selected_slot_mean: {branch_row.get('selected_slot_mean')}")
                    lines.append(f"selected_slot0_ratio: {branch_row.get('selected_slot0_ratio')}")
                    lines.append(f"per_base_oracle_slot_accuracy: {branch_row.get('per_base_oracle_slot_accuracy')}")
                lines.append(f"latency_avg_ms: {branch_row.get('latency_avg_ms')}")
                lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        lines.append("")
        lines.append(f"-- {split} --")
        split_agg = aggregate.get(split, {})
        for branch in branches:
            mean = split_agg.get(branch, {})
            lines.append(f"{branch}: available={int(mean.get('available_official_seeds') or 0)}/{len(rows)}")
            for metric in METRICS:
                lines.append(
                    f"  mean d{metric}: {_fmt(mean.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(mean.get(f'mean_{metric}'))}"
                )
            if branch == selector_branch:
                lines.append(f"  mean selected_slot_mean: {_fmt(mean.get('mean_selected_slot_mean'))}")
                lines.append(f"  mean selected_slot0_ratio: {_fmt(mean.get('mean_selected_slot0_ratio'))}")
                lines.append(
                    "  mean per_base_oracle_slot_accuracy: "
                    f"{_fmt(mean.get('mean_per_base_oracle_slot_accuracy'))}"
                )
            lines.append(f"  mean latency_avg_ms: {_fmt(mean.get('mean_latency_avg_ms'))}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    branches = _branches(str(args.branch_family))
    selector_branch = SELECTOR_BRANCH_BY_FAMILY[str(args.branch_family)]
    rows = [_seed_row(args, project_root, seed, splits, branches) for seed in seeds]
    aggregate = _aggregate(rows, splits, branches)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_v55_slot_selector",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": args.run_prefix,
            "run_name": args.run_name,
            "eval_file_prefix": args.eval_file_prefix,
            "branch_family": args.branch_family,
            "seeds": seeds,
            "splits": splits,
            "branches": list(branches),
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
    rendered = _render(rows, aggregate, splits, branches, selector_branch)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
