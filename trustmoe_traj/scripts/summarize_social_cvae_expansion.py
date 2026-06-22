"""Aggregate V25 SocialCVAE residual expansion diagnosis outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
AUX_METRICS: Sequence[str] = ("delta_l2_mean", "endpoint_ratio", "trajectory_ratio", "unique_base_mode_ratio")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V25 SocialCVAE expansion diagnosis outputs.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--eval-file-prefix", type=str, default="v25_expansion")
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


def _fmt(value: Any, *, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0 else ""
    return f"{prefix}{numeric:.6f}"


def _run_id(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{seed}"


def _metric(metrics: Mapping[str, Any], field: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{field}_{name}"))


def _branch_row(metrics: Mapping[str, Any], branch: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for metric in METRICS:
        value = _metric(metrics, branch, metric)
        slow = _metric(metrics, "slow_pred", metric)
        row[metric] = value
        row[f"d{metric}"] = None if value is None or slow is None else value - slow
    for aux_name in AUX_METRICS:
        row[aux_name] = _num(metrics.get(f"{branch}_{aux_name}"))
    return row


def _seed_row(args: argparse.Namespace, project_root: Path, seed: int, splits: Sequence[str]) -> Dict[str, Any]:
    run_id = _run_id(args.run_prefix, seed)
    row: Dict[str, Any] = {"run_id": run_id, "seed": int(seed), "missing_files": []}
    for split in splits:
        path = (
            project_root
            / "trustmoe_traj"
            / "analysis"
            / "social_cvae_expansion_diagnosis"
            / run_id
            / f"{args.eval_file_prefix}_{split}.json"
        )
        payload = _load_json(path)
        if payload is None:
            row["missing_files"].append(path.as_posix())
            row[f"official_{split}"] = {"branches": [], "metrics": {}}
            continue
        metrics = payload.get("metrics", {})
        branches = payload.get("branches", [])
        if not isinstance(metrics, Mapping):
            metrics = {}
        if not isinstance(branches, list):
            branches = []
        branch_metrics = {str(branch): _branch_row(metrics, str(branch)) for branch in branches}
        row[f"official_{split}"] = {
            "branches": [str(branch) for branch in branches],
            "metrics": branch_metrics,
            "refiner_checkpoint": payload.get("refiner_checkpoint"),
            "refiner_variant": payload.get("meta", {}).get("refiner_variant"),
            "residual_sample_counts": payload.get("meta", {}).get("residual_sample_counts"),
            "oracle_select_metric": payload.get("meta", {}).get("oracle_select_metric"),
        }
    return row


def _all_branches(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for row in rows:
        for split in splits:
            for branch in row.get(f"official_{split}", {}).get("branches", []):
                if branch not in ordered:
                    ordered.append(str(branch))
    return ordered


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str], branches: Sequence[str]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        split_key = f"official_{split}"
        split_out: Dict[str, Any] = {}
        for branch in branches:
            available = [
                row.get(split_key, {}).get("metrics", {}).get(branch, {})
                for row in rows
                if _num(row.get(split_key, {}).get("metrics", {}).get(branch, {}).get("dFDE_min")) is not None
            ]
            branch_out: Dict[str, Any] = {"available_official_seeds": len(available)}
            for metric in METRICS:
                branch_out[f"mean_d{metric}"] = _mean(item.get(f"d{metric}") for item in available)
                branch_out[f"mean_{metric}"] = _mean(item.get(metric) for item in available)
            for aux_name in AUX_METRICS:
                branch_out[f"mean_{aux_name}"] = _mean(item.get(aux_name) for item in available)
            split_out[branch] = branch_out
        aggregate[split] = split_out
    return aggregate


def _render(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], splits: Sequence[str], branches: Sequence[str]) -> str:
    lines: List[str] = []
    for row in rows:
        lines.append(f"===== {row['run_id']} =====")
        if row.get("missing_files"):
            lines.append("missing diagnosis inputs:")
            for path in row["missing_files"]:
                lines.append(f"  {path}")
        for split in splits:
            split_row = row.get(f"official_{split}", {})
            lines.append("")
            lines.append(f"-- official {split} --")
            lines.append(f"refiner_variant: {split_row.get('refiner_variant')}")
            lines.append(f"residual_sample_counts: {split_row.get('residual_sample_counts')}")
            for branch in split_row.get("branches", []):
                if branch == "slow_pred":
                    continue
                metrics = split_row.get("metrics", {}).get(branch, {})
                lines.append(f"{branch}: dFDE_min={_fmt(metrics.get('dFDE_min'), signed=True)} "
                             f"dFDE_avg={_fmt(metrics.get('dFDE_avg'), signed=True)} "
                             f"dMissRate={_fmt(metrics.get('dMissRate'), signed=True)} "
                             f"endpoint_ratio={_fmt(metrics.get('endpoint_ratio'))} "
                             f"trajectory_ratio={_fmt(metrics.get('trajectory_ratio'))} "
                             f"unique_base_mode_ratio={_fmt(metrics.get('unique_base_mode_ratio'))}")
        lines.append("")
    lines.append(f"===== MEAN DELTAS (requested={len(rows)}) =====")
    for split in splits:
        lines.append("")
        lines.append(f"-- {split} --")
        split_agg = aggregate.get(split, {})
        for branch in branches:
            if branch == "slow_pred":
                continue
            item = split_agg.get(branch, {})
            lines.append(
                f"{branch}: available={int(item.get('available_official_seeds') or 0)}/{len(rows)} "
                f"mean_dADE_min={_fmt(item.get('mean_dADE_min'), signed=True)} "
                f"mean_dFDE_min={_fmt(item.get('mean_dFDE_min'), signed=True)} "
                f"mean_dADE_avg={_fmt(item.get('mean_dADE_avg'), signed=True)} "
                f"mean_dFDE_avg={_fmt(item.get('mean_dFDE_avg'), signed=True)} "
                f"mean_dMissRate={_fmt(item.get('mean_dMissRate'), signed=True)} "
                f"endpoint_ratio={_fmt(item.get('mean_endpoint_ratio'))} "
                f"trajectory_ratio={_fmt(item.get('mean_trajectory_ratio'))} "
                f"unique_base_mode_ratio={_fmt(item.get('mean_unique_base_mode_ratio'))}"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    seeds = _split_ints(args.seeds)
    splits = _split_items(args.splits)
    rows = [_seed_row(args, project_root, seed, splits) for seed in seeds]
    branches = _all_branches(rows, splits)
    aggregate = _aggregate(rows, splits, branches)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.summarize_social_cvae_expansion",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_prefix": args.run_prefix,
            "eval_file_prefix": args.eval_file_prefix,
            "seeds": seeds,
            "splits": splits,
        },
        "branches": branches,
        "rows": rows,
        "aggregate": aggregate,
    }
    default_root = project_root / "trustmoe_traj" / "analysis" / "experiment_runs" / args.run_prefix / "v25_expansion"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_root / "aggregate_summary.json"
    output_txt = Path(args.output_txt).expanduser().resolve() if args.output_txt else default_root / "aggregate_summary.txt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered = _render(rows, aggregate, splits, branches)
    output_txt.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
