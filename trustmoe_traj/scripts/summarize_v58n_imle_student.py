"""Summarize V58N IMLE student run_eval outputs across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence


METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate", "latency_avg_ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V58N IMLE student evaluation JSONs.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument(
        "--file-template",
        type=str,
        default="{run_prefix}_seed{seed}_{split}.json",
        help="Template relative to input-root.",
    )
    parser.add_argument("--student-branch", type=str, default="fast_pred")
    parser.add_argument("--baseline-branch", type=str, default="slow_pred")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    return parser


def _split(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def _metric_key(branch: str, metric: str) -> str:
    if metric == "latency_avg_ms":
        return f"{branch}_latency_avg_ms"
    return f"{branch}_{metric}"


def _load(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_metrics(payload: Mapping[str, Any], *, student_branch: str, baseline_branch: str) -> Dict[str, float]:
    metrics = payload.get("metrics", {})
    row: Dict[str, float] = {}
    for metric in METRICS:
        student_key = _metric_key(student_branch, metric)
        baseline_key = _metric_key(baseline_branch, metric)
        if student_key not in metrics:
            continue
        row[f"{student_branch}_{metric}"] = float(metrics[student_key])
        if baseline_key in metrics:
            row[f"{baseline_branch}_{metric}"] = float(metrics[baseline_key])
            if metric != "latency_avg_ms":
                row[f"d{metric}"] = float(metrics[student_key]) - float(metrics[baseline_key])
    return row


def _mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean(float(row[key]) for row in rows if key in row) for key in keys}


def _build_lines(payload: Mapping[str, Any]) -> List[str]:
    args = payload["args"]
    lines = [
        "===== V58N IMLE STUDENT SUMMARY =====",
        f"run_prefix={args['run_prefix']} requested_seeds={args['seeds']}",
        f"student_branch={args['student_branch']} baseline_branch={args['baseline_branch']}",
        "",
        "===== INPUTS =====",
    ]
    for item in payload["inputs"]:
        status = "ok" if item["available"] else "missing"
        lines.append(f"{item['split']} seed{item['seed']}: {status} {item['path']}")
    lines.append("")
    lines.append("===== MEAN DELTAS =====")
    for split, split_payload in payload["splits"].items():
        lines.append(f"-- {split} --")
        lines.append(f"available={split_payload['available']}/{split_payload['requested']}")
        if not split_payload["mean"]:
            continue
        mean_payload = split_payload["mean"]
        for metric in METRICS:
            dkey = f"d{metric}"
            skey = f"{args['student_branch']}_{metric}"
            bkey = f"{args['baseline_branch']}_{metric}"
            if metric == "latency_avg_ms":
                if skey in mean_payload:
                    lines.append(f"  mean {args['student_branch']} latency_avg_ms: {mean_payload[skey]:.6f}")
                if bkey in mean_payload:
                    lines.append(f"  mean {args['baseline_branch']} latency_avg_ms: {mean_payload[bkey]:.6f}")
                continue
            if dkey in mean_payload:
                lines.append(
                    f"  mean d{metric}: {mean_payload[dkey]:+.6f} "
                    f"{args['student_branch']}={mean_payload[skey]:.6f} "
                    f"{args['baseline_branch']}={mean_payload[bkey]:.6f}"
                )
    return lines


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    seeds = [int(item) for item in _split(args.seeds)]
    splits = _split(args.splits)

    inputs: List[Dict[str, Any]] = []
    split_rows: Dict[str, List[Dict[str, float]]] = {split: [] for split in splits}
    for split in splits:
        for seed in seeds:
            rel = args.file_template.format(run_prefix=args.run_prefix, seed=seed, split=split)
            path = input_root / rel
            payload = _load(path)
            item = {"split": split, "seed": seed, "path": path.as_posix(), "available": payload is not None}
            inputs.append(item)
            if payload is None:
                continue
            split_rows[split].append(
                _row_metrics(payload, student_branch=str(args.student_branch), baseline_branch=str(args.baseline_branch))
            )

    split_payload: Dict[str, Any] = {}
    for split, rows in split_rows.items():
        split_payload[split] = {
            "requested": len(seeds),
            "available": len(rows),
            "rows": rows,
            "mean": _mean_rows(rows) if rows else {},
        }

    result = {
        "args": vars(args),
        "inputs": inputs,
        "splits": split_payload,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_txt = Path(args.output_txt).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = _build_lines(result)
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"summary_json={output_json.as_posix()}")
    print(f"summary_txt={output_txt.as_posix()}")


if __name__ == "__main__":
    main()
