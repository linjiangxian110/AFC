"""Aggregate V58 slot residual distribution diagnosis outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DEFAULTS: Sequence[str] = ("train", "val", "test")
GROUP_DEFAULTS: Sequence[str] = ("good_min_safe", "bad_min_hurt", "neutral", "strong_good")
RATIO_KEYS: Sequence[str] = (
    "good_min_safe_ratio",
    "bad_min_hurt_ratio",
    "neutral_ratio",
    "strong_good_ratio",
)
GROUP_MEAN_KEYS: Sequence[str] = (
    "mean_dADE_vs_slot0",
    "mean_dFDE_vs_slot0",
    "mean_endpoint_norm",
    "mean_trajectory_norm",
    "mean_base_rank",
)
SEPARATION_KEYS: Sequence[str] = ("low_centroid_l2", "low_mean_abs_z_gap", "low_max_abs_z_gap")
PROBE_KEYS: Sequence[str] = ("auc", "ap", "accuracy_at_0p5", "positive_rate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V58 slot residual distribution diagnosis outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="train,val,test")
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
    prefix = "+" if signed and numeric >= 0.0 else ""
    return f"{prefix}{numeric:.6f}"


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [item for item in (_num(value) for value in values) if item is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _rows(input_root: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        path = input_root / f"seed{seed}.json"
        payload = _load_json(path)
        rows.append(
            {
                "seed": int(seed),
                "path": path.as_posix(),
                "payload": payload,
                "missing": payload is None,
            }
        )
    return rows


def _aggregate_split(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    split_rows = [
        row.get("payload", {}).get("splits", {}).get(split, {})
        for row in rows
        if isinstance(row.get("payload"), Mapping)
    ]
    result: Dict[str, Any] = {"available_seeds": len(split_rows)}
    for key in RATIO_KEYS:
        result[key] = _mean(row.get("groups", {}).get("label_ratios", {}).get(key) for row in split_rows)
    for key in SEPARATION_KEYS:
        result[key] = _mean(row.get("groups", {}).get("separation", {}).get(key) for row in split_rows)
    for group in GROUP_DEFAULTS:
        group_result: Dict[str, Any] = {}
        for key in ("count", *GROUP_MEAN_KEYS):
            group_result[key] = _mean(row.get("groups", {}).get(group, {}).get(key) for row in split_rows)
        result[group] = group_result
    return result


def _aggregate_probes(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for kind in ("low", "residual"):
        kind_result: Dict[str, Any] = {}
        for split in splits:
            split_result: Dict[str, Any] = {}
            for key in PROBE_KEYS:
                split_result[key] = _mean(
                    row.get("payload", {}).get("probes", {}).get(kind, {}).get(split, {}).get(key)
                    for row in rows
                    if isinstance(row.get("payload"), Mapping)
                )
            split_result["samples"] = _mean(
                row.get("payload", {}).get("probes", {}).get(kind, {}).get(split, {}).get("samples")
                for row in rows
                if isinstance(row.get("payload"), Mapping)
            )
            kind_result[split] = split_result
        result[kind] = kind_result
    return result


def _aggregate_top_features(rows: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    feature_scores: Dict[str, List[float]] = {}
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        features = (
            payload.get("splits", {})
            .get(split, {})
            .get("groups", {})
            .get("separation", {})
            .get("low_top_z_features", [])
        )
        if not isinstance(features, list):
            continue
        for item in features:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name"))
            score = _num(item.get("z_gap"))
            if score is None:
                continue
            feature_scores.setdefault(name, []).append(abs(float(score)))
    ranked = sorted(
        ((name, _mean(values), len(values)) for name, values in feature_scores.items()),
        key=lambda item: float(item[1] or 0.0),
        reverse=True,
    )
    return [
        {"name": name, "mean_abs_z_gap": score, "available_seeds": count}
        for name, score, count in ranked[:10]
    ]


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> Dict[str, Any]:
    return {
        "splits": {
            split: {
                **_aggregate_split(rows, split),
                "top_low_features": _aggregate_top_features(rows, split),
            }
            for split in splits
        },
        "probes": _aggregate_probes(rows, splits),
    }


def _render(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], splits: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("===== INPUTS =====")
    for row in rows:
        status = "missing" if row.get("missing") else "ok"
        lines.append(f"seed{row['seed']}: {status} {row['path']}")
    lines.append("")
    lines.append("===== AGGREGATE SPLITS =====")
    for split in splits:
        row = aggregate.get("splits", {}).get(split, {})
        lines.append(f"-- {split} --")
        lines.append(
            "labels: "
            f"good={_fmt(row.get('good_min_safe_ratio'))} "
            f"bad={_fmt(row.get('bad_min_hurt_ratio'))} "
            f"neutral={_fmt(row.get('neutral_ratio'))} "
            f"strong_good={_fmt(row.get('strong_good_ratio'))}"
        )
        lines.append(
            "separation: "
            f"low_l2={_fmt(row.get('low_centroid_l2'))} "
            f"mean_abs_z={_fmt(row.get('low_mean_abs_z_gap'))} "
            f"max_abs_z={_fmt(row.get('low_max_abs_z_gap'))}"
        )
        for group in ("good_min_safe", "bad_min_hurt"):
            stats = row.get(group, {})
            lines.append(
                f"{group}: count={_fmt(stats.get('count'))} "
                f"dADE={_fmt(stats.get('mean_dADE_vs_slot0'), signed=True)} "
                f"dFDE={_fmt(stats.get('mean_dFDE_vs_slot0'), signed=True)} "
                f"endpoint_norm={_fmt(stats.get('mean_endpoint_norm'))} "
                f"traj_norm={_fmt(stats.get('mean_trajectory_norm'))} "
                f"base_rank={_fmt(stats.get('mean_base_rank'))}"
            )
        top = row.get("top_low_features", [])
        if isinstance(top, list):
            lines.append(
                "top_low_features="
                + str([(item.get("name"), round(float(item.get("mean_abs_z_gap") or 0.0), 3)) for item in top[:6]])
            )
    lines.append("")
    lines.append("===== PROBES =====")
    for kind in ("low", "residual"):
        lines.append(f"-- {kind} --")
        for split in splits:
            stats = aggregate.get("probes", {}).get(kind, {}).get(split, {})
            lines.append(
                f"{split}: auc={_fmt(stats.get('auc'))} ap={_fmt(stats.get('ap'))} "
                f"acc={_fmt(stats.get('accuracy_at_0p5'))} "
                f"pos={_fmt(stats.get('positive_rate'))} samples={_fmt(stats.get('samples'))}"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    rows = _rows(input_root, seeds)
    aggregate = _aggregate(rows, splits)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_v58_slot_residual_distribution",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": input_root.as_posix(),
            "seeds": list(seeds),
            "splits": list(splits),
        },
        "rows": [
            {"seed": row["seed"], "path": row["path"], "missing": bool(row.get("missing"))}
            for row in rows
        ],
        "aggregate": aggregate,
    }
    default_root = Path(args.project_root).expanduser().resolve() / "trustmoe_traj" / "analysis" / "experiment_runs"
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else default_root / "v58_slot_residual_distribution_summary.json"
    )
    output_txt = (
        Path(args.output_txt).expanduser().resolve()
        if args.output_txt
        else default_root / "v58_slot_residual_distribution_summary.txt"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render(rows, aggregate, splits)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
