"""Aggregate V58-B0 elite residual space diagnosis outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUPS: Sequence[str] = (
    "global_top20",
    "per_base_top1",
    "per_base_top2",
    "gain_gt_0",
    "gain_gt_0p05",
    "gain_gt_0p1",
    "gain_gt_0p15",
)
MEAN_KEYS: Sequence[str] = (
    "mean_gain",
    "mean_positive_gain",
    "mean_candidate_score",
    "mean_base_score",
    "mean_endpoint_norm",
    "mean_trajectory_norm",
    "mean_forward",
    "mean_lateral",
    "mean_abs_lateral",
    "mean_base_rank",
    "mean_slot_id",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V58-B0 elite residual space diagnosis outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_GROUPS))
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


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0 else ""
    return f"{prefix}{numeric:.6f}"


def _sum_list(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    for value in values:
        if not isinstance(value, list):
            continue
        if not result:
            result = [0 for _ in value]
        for index, item in enumerate(value):
            result[index] += int(item or 0)
    return result


def _ratios(counts: Sequence[int]) -> List[float]:
    total = max(sum(int(item) for item in counts), 1)
    return [float(int(item) / total) for item in counts]


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    numer = 0.0
    denom = 0.0
    for row in rows:
        value = _num(row.get(key))
        count = _num(row.get("count"))
        if value is None or count is None or count <= 0:
            continue
        numer += value * count
        denom += count
    if denom <= 0:
        return None
    return float(numer / denom)


def _aggregate_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = [_num(row.get("count")) or 0.0 for row in rows]
    valid_agents = [_num(row.get("valid_agent_count")) or 0.0 for row in rows]
    slot_counts = _sum_list(row.get("slot_counts") for row in rows)
    rank_counts = _sum_list(row.get("base_rank_counts") for row in rows)
    gain_hist = _sum_list(row.get("gain_hist") for row in rows)
    endpoint_hist = _sum_list(row.get("endpoint_norm_hist") for row in rows)
    result: Dict[str, Any] = {
        "count": int(sum(counts)),
        "valid_agent_count": int(sum(valid_agents)),
        "selected_per_valid_agent": float(sum(counts) / max(sum(valid_agents), 1.0)),
        "slot_counts": slot_counts,
        "slot_ratios": _ratios(slot_counts),
        "base_rank_counts": rank_counts,
        "base_rank_ratios": _ratios(rank_counts),
        "gain_hist": gain_hist,
        "endpoint_norm_hist": endpoint_hist,
    }
    for key in MEAN_KEYS:
        result[key] = _weighted_mean(rows, key)
    return result


def _seed_row(input_root: Path, seed: int, splits: Sequence[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"seed": int(seed), "splits": {}, "missing_files": []}
    for split in splits:
        path = input_root / f"seed{seed}_{split}.json"
        payload = _load_json(path)
        if payload is None:
            row["missing_files"].append(path.as_posix())
            row["splits"][split] = {}
            continue
        row["splits"][split] = {
            "meta": payload.get("meta", {}),
            "groups": payload.get("groups", {}),
            "residual_clusters": payload.get("residual_clusters", {}),
        }
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str], groups: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in splits:
        split_result: Dict[str, Any] = {}
        for group in groups:
            group_rows: List[Mapping[str, Any]] = []
            for row in rows:
                group_row = row.get("splits", {}).get(split, {}).get("groups", {}).get(group)
                if isinstance(group_row, Mapping):
                    group_rows.append(group_row)
            split_result[group] = {
                "available_seeds": len(group_rows),
                **_aggregate_group(group_rows),
            }
        result[split] = split_result
    return result


def _render(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], splits: Sequence[str], groups: Sequence[str]) -> str:
    lines: List[str] = []
    for row in rows:
        lines.append(f"===== seed{row['seed']} =====")
        if row.get("missing_files"):
            lines.append("missing diagnosis inputs:")
            for path in row["missing_files"]:
                lines.append(f"  {path}")
        for split in splits:
            lines.append(f"-- {split} --")
            split_row = row.get("splits", {}).get(split, {})
            group_map = split_row.get("groups", {}) if isinstance(split_row, Mapping) else {}
            for group in groups:
                stats = group_map.get(group, {}) if isinstance(group_map, Mapping) else {}
                lines.append(
                    f"{group}: count={int(_num(stats.get('count')) or 0)} "
                    f"per_agent={_fmt(stats.get('selected_per_valid_agent'))} "
                    f"gain={_fmt(stats.get('mean_gain'), signed=True)} "
                    f"pos={_fmt(stats.get('mean_positive_gain'))} "
                    f"norm={_fmt(stats.get('mean_endpoint_norm'))} "
                    f"rank={_fmt(stats.get('mean_base_rank'))} "
                    f"slot={_fmt(stats.get('mean_slot_id'))}"
                )
                if isinstance(stats.get("slot_ratios"), list):
                    lines.append(f"  slot_ratios={ [round(float(item), 4) for item in stats['slot_ratios']] }")
                if isinstance(stats.get("base_rank_ratios"), list):
                    lines.append(f"  base_rank_top8={ [round(float(item), 4) for item in stats['base_rank_ratios'][:8]] }")
        lines.append("")
    lines.append("===== AGGREGATE =====")
    for split in splits:
        lines.append(f"-- {split} --")
        split_agg = aggregate.get(split, {})
        for group in groups:
            stats = split_agg.get(group, {})
            lines.append(
                f"{group}: seeds={int(stats.get('available_seeds') or 0)}/{len(rows)} "
                f"count={int(_num(stats.get('count')) or 0)} "
                f"per_agent={_fmt(stats.get('selected_per_valid_agent'))} "
                f"gain={_fmt(stats.get('mean_gain'), signed=True)} "
                f"pos={_fmt(stats.get('mean_positive_gain'))} "
                f"norm={_fmt(stats.get('mean_endpoint_norm'))} "
                f"rank={_fmt(stats.get('mean_base_rank'))} "
                f"slot={_fmt(stats.get('mean_slot_id'))}"
            )
            if isinstance(stats.get("slot_ratios"), list):
                lines.append(f"  slot_ratios={ [round(float(item), 4) for item in stats['slot_ratios']] }")
            if isinstance(stats.get("base_rank_ratios"), list):
                lines.append(f"  base_rank_top8={ [round(float(item), 4) for item in stats['base_rank_ratios'][:8]] }")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser()
    if not input_root.is_absolute():
        input_root = project_root / input_root
    input_root = input_root.resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    groups = _split_items(args.groups)
    rows = [_seed_row(input_root, seed, splits) for seed in seeds]
    aggregate = _aggregate(rows, splits, groups)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_v58_elite_residual_space",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": input_root.as_posix(),
            "seeds": seeds,
            "splits": splits,
            "groups": groups,
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
    rendered = _render(rows, aggregate, splits, groups)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
