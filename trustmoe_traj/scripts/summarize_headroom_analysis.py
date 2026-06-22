"""Summarize headroom analysis JSON files across datasets/seeds/splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METRICS: Sequence[str] = (
    "ADE_min",
    "FDE_min",
    "ADE_avg",
    "FDE_avg",
    "MissRate",
    "energy_score",
    "energy_gt_term",
    "energy_pair_term",
    "afc_recall_eps03",
    "afc_precision_eps03",
    "afc_mode_coverage_eps03",
    "afc_mode_recall_eps03",
    "afc_weighted_mode_recall_eps03",
    "afc_mode_precision_eps03",
    "afc_mode_chamfer_eps03",
    "afc_mode_intra_distance_eps03",
    "afc_mode_entropy_eps03",
    "afc_unsupported_ratio_eps03",
    "afc_recall_eps05",
    "afc_precision_eps05",
    "afc_mode_coverage_eps05",
    "afc_mode_recall_eps05",
    "afc_weighted_mode_recall_eps05",
    "afc_mode_precision_eps05",
    "afc_mode_chamfer_eps05",
    "afc_mode_intra_distance_eps05",
    "afc_mode_entropy_eps05",
    "afc_unsupported_ratio_eps05",
    "afc_recall_eps10",
    "afc_precision_eps10",
    "afc_mode_coverage_eps10",
    "afc_mode_recall_eps10",
    "afc_weighted_mode_recall_eps10",
    "afc_mode_precision_eps10",
    "afc_mode_chamfer_eps10",
    "afc_mode_intra_distance_eps10",
    "afc_mode_entropy_eps10",
    "afc_unsupported_ratio_eps10",
    "afc_chamfer",
    "afc_retrieval_confidence",
    "afc_retrieval_effective_m",
    "afc_retrieval_top1_distance",
    "afc_retrieval_top_m_distance",
    "endpoint_ratio",
    "trajectory_ratio",
    "unique_base_mode_ratio",
)
DELTA_METRICS: Sequence[str] = (
    "ADE_min",
    "FDE_min",
    "ADE_avg",
    "FDE_avg",
    "energy_score",
    "energy_gt_term",
    "energy_pair_term",
    "afc_recall_eps03",
    "afc_mode_coverage_eps03",
    "afc_weighted_mode_recall_eps03",
    "afc_mode_precision_eps03",
    "afc_mode_chamfer_eps03",
    "afc_mode_intra_distance_eps03",
    "afc_mode_entropy_eps03",
    "afc_unsupported_ratio_eps03",
    "afc_recall_eps05",
    "afc_mode_coverage_eps05",
    "afc_weighted_mode_recall_eps05",
    "afc_mode_precision_eps05",
    "afc_mode_chamfer_eps05",
    "afc_mode_intra_distance_eps05",
    "afc_mode_entropy_eps05",
    "afc_unsupported_ratio_eps05",
    "afc_recall_eps10",
    "afc_mode_coverage_eps10",
    "afc_weighted_mode_recall_eps10",
    "afc_mode_precision_eps10",
    "afc_mode_chamfer_eps10",
    "afc_mode_intra_distance_eps10",
    "afc_mode_entropy_eps10",
    "afc_unsupported_ratio_eps10",
    "afc_chamfer",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize headroom analysis outputs.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="eth,hotel,zara1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument(
        "--file-template",
        type=str,
        default="{input_root}/{run_id}_{dataset}_seed{seed}/{dataset}_{split}_headroom.json",
    )
    return parser


def _split_items(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


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


def _branch_metric(metrics: Mapping[str, Any], branch: str, metric: str) -> Optional[float]:
    return _num(metrics.get(f"{branch}_{metric}"))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(item) for item in values if item is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _sort_branch(branch: str) -> Tuple[int, str]:
    if branch == "slow20_pred":
        return (0, branch)
    if branch == "cv_linear20_pred":
        return (1, branch)
    if branch.startswith("random_spread"):
        return (2, branch)
    if branch.startswith("slow") and "_full" in branch:
        return (3, branch)
    if branch.startswith("slow") and "_afc_greedy20" in branch:
        return (4, branch)
    if branch.startswith("slow") and "_endpoint_fps20" in branch:
        return (5, branch)
    if branch.startswith("slow") and "_random20_pred" in branch:
        return (6, branch)
    if branch.startswith("slow") and "_random20_mean" in branch:
        return (6, branch)
    if branch.startswith("slow") and "_random20_trial" in branch:
        return (7, branch)
    if branch.startswith("slow") and "_gt_oracle20" in branch:
        return (8, branch)
    if branch.startswith("residual") and "afc_greedy20" in branch:
        return (9, branch)
    if branch.startswith("residual") and "endpoint_fps20" in branch:
        return (10, branch)
    if branch.startswith("residual") and "gt_oracle20" in branch:
        return (11, branch)
    if branch.startswith("residual") and "full" in branch:
        return (12, branch)
    return (99, branch)


def _load_rows(
    *,
    input_root: Path,
    run_id: str,
    datasets: Sequence[str],
    seeds: Sequence[str],
    splits: Sequence[str],
    file_template: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        for seed in seeds:
            for split in splits:
                raw_path = file_template.format(
                    input_root=input_root.as_posix(),
                    run_id=run_id,
                    dataset=dataset,
                    seed=seed,
                    split=split,
                )
                path = Path(raw_path)
                if not path.exists():
                    missing.append(path.as_posix())
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                branches = payload.get("branches", [])
                for branch in branches:
                    row: Dict[str, Any] = {
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "branch": branch,
                        "source_json": path.as_posix(),
                    }
                    for metric in METRICS:
                        row[metric] = _branch_metric(metrics, branch, metric)
                    for metric in DELTA_METRICS:
                        value = _branch_metric(metrics, branch, metric)
                        slow = _branch_metric(metrics, "slow20_pred", metric)
                        row[f"d{metric}"] = None if value is None or slow is None else value - slow
                    rows.append(row)
    return rows, missing


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["split"]), str(row["branch"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (dataset, split, branch), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], _sort_branch(item[0][2]))):
        out: Dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "branch": branch,
            "n": len(items),
        }
        for metric in METRICS:
            out[f"mean_{metric}"] = _mean(item.get(metric) for item in items)
        for metric in DELTA_METRICS:
            out[f"mean_d{metric}"] = _mean(item.get(f"d{metric}") for item in items)
        result.append(out)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "branch",
        "n",
        *(f"mean_{metric}" for metric in METRICS),
        *(f"mean_d{metric}" for metric in DELTA_METRICS),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _best_branch(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    split: str,
    contains: str,
    *,
    prefix: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("dataset")) == dataset
        and str(row.get("split")) == split
        and contains in str(row.get("branch"))
        and (prefix is None or str(row.get("branch", "")).startswith(prefix))
    ]
    if not candidates:
        return None

    def score(row: Mapping[str, Any]) -> float:
        weighted = _num(row.get("mean_dafc_weighted_mode_recall_eps05"))
        if weighted is not None:
            return float(weighted)
        legacy = _num(row.get("mean_dafc_mode_coverage_eps05"))
        return -1e9 if legacy is None else float(legacy)

    return max(candidates, key=score)


def _render_markdown(rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# Headroom Analysis Summary")
    lines.append("")
    if missing:
        lines.append("## Missing Inputs")
        lines.append("")
        for path in missing:
            lines.append(f"- `{path}`")
        lines.append("")
    lines.append("## Headroom Table")
    lines.append("")
    lines.append(
        "| dataset | split | branch | n | dADE_avg | dFDE_avg | dADE_min | dFDE_min | "
        "dEnergy | Energy | "
        "dAFC recall@0.5 | dAFC mode@0.5 | dAFC wMode@0.5 | dAFC precision@0.5 | dUnsupported@0.5 | "
        "dAFC recall@1.0 | dAFC wMode@1.0 | dUnsupported@1.0 | dChamfer | retrieval_conf | "
        "endpoint_ratio | trajectory_ratio | base_mode_ratio |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("split", "")),
                    str(row.get("branch", "")),
                    str(row.get("n", "")),
                    _fmt(_num(row.get("mean_dADE_avg")), signed=True),
                    _fmt(_num(row.get("mean_dFDE_avg")), signed=True),
                    _fmt(_num(row.get("mean_dADE_min")), signed=True),
                    _fmt(_num(row.get("mean_dFDE_min")), signed=True),
                    _fmt(_num(row.get("mean_denergy_score")), signed=True),
                    _fmt(_num(row.get("mean_energy_score"))),
                    _fmt(_num(row.get("mean_dafc_recall_eps05")), signed=True),
                    _fmt(_num(row.get("mean_dafc_mode_coverage_eps05")), signed=True),
                    _fmt(_num(row.get("mean_dafc_weighted_mode_recall_eps05")), signed=True),
                    _fmt(_num(row.get("mean_dafc_mode_precision_eps05")), signed=True),
                    _fmt(_num(row.get("mean_dafc_unsupported_ratio_eps05")), signed=True),
                    _fmt(_num(row.get("mean_dafc_recall_eps10")), signed=True),
                    _fmt(_num(row.get("mean_dafc_weighted_mode_recall_eps10")), signed=True),
                    _fmt(_num(row.get("mean_dafc_unsupported_ratio_eps10")), signed=True),
                    _fmt(_num(row.get("mean_dafc_chamfer")), signed=True),
                    _fmt(_num(row.get("mean_afc_retrieval_confidence"))),
                    _fmt(_num(row.get("mean_endpoint_ratio"))),
                    _fmt(_num(row.get("mean_trajectory_ratio"))),
                    _fmt(_num(row.get("mean_unique_base_mode_ratio"))),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Diagnosis Hints")
    lines.append("")
    datasets = sorted({str(row.get("dataset")) for row in rows})
    splits = sorted({str(row.get("split")) for row in rows})
    for dataset in datasets:
        for split in splits:
            slow_pool = _best_branch(rows, dataset, split, "_afc_greedy20_pred", prefix="slow")
            residual = _best_branch(rows, dataset, split, "afc_greedy20", prefix="residual")
            if slow_pool is not None:
                lines.append(
                    f"- `{dataset}/{split}` best slow-pool AFC greedy `{slow_pool['branch']}`: "
                    f"dAFC weighted-mode@0.5={_fmt(_num(slow_pool.get('mean_dafc_weighted_mode_recall_eps05')), signed=True)}, "
                    f"dAFC recall@0.5={_fmt(_num(slow_pool.get('mean_dafc_recall_eps05')), signed=True)}, "
                    f"dUnsupported@0.5={_fmt(_num(slow_pool.get('mean_dafc_unsupported_ratio_eps05')), signed=True)}, "
                    f"dADE_avg={_fmt(_num(slow_pool.get('mean_dADE_avg')), signed=True)}."
                )
            if residual is not None:
                lines.append(
                    f"- `{dataset}/{split}` residual AFC greedy `{residual['branch']}`: "
                    f"dAFC weighted-mode@0.5={_fmt(_num(residual.get('mean_dafc_weighted_mode_recall_eps05')), signed=True)}, "
                    f"dAFC recall@0.5={_fmt(_num(residual.get('mean_dafc_recall_eps05')), signed=True)}, "
                    f"dUnsupported@0.5={_fmt(_num(residual.get('mean_dafc_unsupported_ratio_eps05')), signed=True)}, "
                    f"dADE_avg={_fmt(_num(residual.get('mean_dADE_avg')), signed=True)}."
                )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- If slow-pool AFC greedy improves weighted-mode recall over slow20, the base sampler has latent plausible-mode coverage headroom.")
    lines.append("- If unsupported ratio rises, the extra diversity is less supported by current GT or analogical future modes.")
    lines.append("- If residual AFC greedy does not improve over slow20 or slow-pool greedy, the residual generator is not creating useful new plausible modes.")
    lines.append("- If a GT oracle improves avg/min but AFC drops, the improvement is GT-centric rather than set-coverage positive.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "analysis"
    rows, missing = _load_rows(
        input_root=input_root,
        run_id=str(args.run_id),
        datasets=_split_items(args.datasets),
        seeds=_split_items(args.seeds),
        splits=_split_items(args.splits),
        file_template=str(args.file_template),
    )
    aggregate = _aggregate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "headroom_summary.csv"
    md_path = output_dir / "headroom_summary.md"
    json_path = output_dir / "headroom_summary.json"
    _write_csv(csv_path, aggregate)
    md_path.write_text(_render_markdown(aggregate, missing), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": aggregate, "missing": missing}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"analysis_csv={csv_path.as_posix()}")
    print(f"analysis_md={md_path.as_posix()}")
    print(f"analysis_json={json_path.as_posix()}")
    if missing:
        print("missing_inputs=" + ",".join(missing))


if __name__ == "__main__":
    main()
