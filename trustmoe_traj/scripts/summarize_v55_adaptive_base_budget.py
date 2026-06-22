"""Aggregate V55-C adaptive base-budget diagnosis outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V55-C adaptive base-budget diagnosis outputs.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
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
    prefix = "+" if signed and numeric >= 0.0 else ""
    return f"{prefix}{numeric:.6f}"


def _metric(metrics: Mapping[str, Any], field: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{field}_{name}"))


def _group_metric(metrics: Mapping[str, Any], group: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{group}_mean_{name}"))


def _load_rows(input_root: Path, seeds: Sequence[int], splits: Sequence[str]) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    branch_set: List[str] = []
    group_set: List[str] = []
    for split in splits:
        for seed in seeds:
            path = input_root / f"seed{seed}_{split}.json"
            payload = _load_json(path)
            if payload is None:
                rows.append({"seed": seed, "split": split, "path": path.as_posix(), "missing": True})
                continue
            deterministic = [item for item in payload.get("deterministic_branches", []) if item != "slow_pred"]
            random_groups = list((payload.get("random_groups", {}) or {}).keys())
            for branch in deterministic:
                if branch not in branch_set:
                    branch_set.append(branch)
            for group in random_groups:
                if group not in group_set:
                    group_set.append(group)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "path": path.as_posix(),
                    "missing": False,
                    "metrics": payload.get("metrics", {}),
                    "deterministic_branches": deterministic,
                    "random_groups": random_groups,
                }
            )
    return rows, branch_set, group_set


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    branches: Sequence[str],
    random_groups: Sequence[str],
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        split_rows = [row for row in rows if row.get("split") == split and not row.get("missing")]
        split_agg: Dict[str, Any] = {}
        for branch in branches:
            branch_agg: Dict[str, Any] = {
                "available": sum(1 for row in split_rows if _metric(row.get("metrics", {}), branch, "FDE_min") is not None)
            }
            for metric in METRICS:
                branch_agg[f"mean_{metric}"] = _mean(_metric(row.get("metrics", {}), branch, metric) for row in split_rows)
                branch_agg[f"mean_d{metric}"] = _mean(
                    (
                        None
                        if _metric(row.get("metrics", {}), branch, metric) is None
                        or _metric(row.get("metrics", {}), "slow_pred", metric) is None
                        else _metric(row.get("metrics", {}), branch, metric)
                        - _metric(row.get("metrics", {}), "slow_pred", metric)
                    )
                    for row in split_rows
                )
            for aux in ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio"):
                branch_agg[f"mean_{aux}"] = _mean(
                    row.get("metrics", {}).get(f"{branch}_{aux}") for row in split_rows
                )
            split_agg[branch] = branch_agg
        for group in random_groups:
            group_agg: Dict[str, Any] = {
                "available": sum(1 for row in split_rows if _group_metric(row.get("metrics", {}), group, "FDE_min") is not None)
            }
            for metric in METRICS:
                group_agg[f"mean_{metric}"] = _mean(_group_metric(row.get("metrics", {}), group, metric) for row in split_rows)
                group_agg[f"mean_d{metric}"] = _mean(
                    (
                        None
                        if _group_metric(row.get("metrics", {}), group, metric) is None
                        or _metric(row.get("metrics", {}), "slow_pred", metric) is None
                        else _group_metric(row.get("metrics", {}), group, metric)
                        - _metric(row.get("metrics", {}), "slow_pred", metric)
                    )
                    for row in split_rows
                )
            split_agg[group] = group_agg
        aggregate[split] = split_agg
    return aggregate


def _render(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    splits: Sequence[str],
    branches: Sequence[str],
    random_groups: Sequence[str],
    *,
    requested: int,
) -> str:
    lines: List[str] = []
    missing = [row for row in rows if row.get("missing")]
    if missing:
        lines.append("===== MISSING FILES =====")
        for row in missing:
            lines.append(str(row.get("path")))
        lines.append("")
    for split in splits:
        lines.append(f"===== {split.upper()} MEAN DELTAS =====")
        split_agg = aggregate.get(split, {})
        for branch in branches:
            row = split_agg.get(branch, {})
            lines.append(f"-- {branch} available={int(row.get('available') or 0)}/{requested} --")
            for metric in METRICS:
                lines.append(
                    f"mean d{metric}: {_fmt(row.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(row.get(f'mean_{metric}'))}"
                )
            if row.get("mean_unique_base_mode_ratio") is not None:
                lines.append(f"unique_base_mode_ratio: {_fmt(row.get('mean_unique_base_mode_ratio'))}")
            lines.append("")
        for group in random_groups:
            row = split_agg.get(group, {})
            lines.append(f"-- {group} random mean available={int(row.get('available') or 0)}/{requested} --")
            for metric in METRICS:
                lines.append(
                    f"mean d{metric}: {_fmt(row.get(f'mean_d{metric}'), signed=True)}  "
                    f"value={_fmt(row.get(f'mean_{metric}'))}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    rows, branches, random_groups = _load_rows(input_root, seeds, splits)
    aggregate = _aggregate(rows, splits, branches, random_groups)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_v55_adaptive_base_budget",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": input_root.as_posix(),
            "seeds": seeds,
            "splits": splits,
            "branches": branches,
            "random_groups": random_groups,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    default_root = project_root / "trustmoe_traj" / "analysis" / "experiment_runs" / input_root.name
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_root / "aggregate_summary.json"
    output_txt = Path(args.output_txt).expanduser().resolve() if args.output_txt else default_root / "aggregate_summary.txt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered = _render(rows, aggregate, splits, branches, random_groups, requested=len(seeds))
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
