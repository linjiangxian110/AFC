"""Precheck whether observable signals can predict harmful residual updates.

This script joins:

1. ``diagnose_residual_graduate.py --save-records`` outputs, which contain
   fast/student vs graduate per-item deltas.
2. ``run_eval.py --output-per-sample-json`` outputs, which contain scene,
   motion, prediction-proxy, and optional fast/slow teacher metrics.

It answers the V13 precheck question: can proxy / teacher-advantage / residual
runtime signals predict when residual correction hurts ``FDE_min``?
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trustmoe_traj.scripts.analyze_per_sample_differences import (
    DEFAULT_PROXY_FEATURES,
    _auc,
    _average_ranks,
    _mean,
    _numeric_value,
    _pearson,
    _spearman,
    _summarize_values,
)


DEFAULT_METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "Miss")
PREDICTION_PROXY_MARKERS: Sequence[str] = (
    "num_modes",
    "trajectory_spread",
    "endpoint_variance",
    "endpoint_pairwise",
    "collision",
)
RESIDUAL_RUNTIME_FEATURES: Sequence[str] = (
    "gate_mean",
    "delta_l2_mean",
    "endpoint_spread_student",
    "endpoint_spread_graduate",
    "endpoint_spread_delta",
    "endpoint_spread_ratio",
    "trajectory_diversity_student",
    "trajectory_diversity_graduate",
    "trajectory_diversity_delta",
    "trajectory_diversity_ratio",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze whether proxy features predict harmful Residual Graduate corrections."
    )
    parser.add_argument(
        "--diagnose-json",
        required=True,
        help="Path from diagnose_residual_graduate.py run with --save-records.",
    )
    parser.add_argument(
        "--per-sample-json",
        required=True,
        help="Path from run_eval.py --output-per-sample-json, preferably with --baseline both.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--fast-branch", default="fast_pred")
    parser.add_argument("--slow-branch", default="slow_pred")
    parser.add_argument(
        "--harm-margin",
        type=float,
        default=0.0,
        help="Residual is harmful when graduate_FDE_min - student_FDE_min exceeds this margin.",
    )
    parser.add_argument(
        "--student-best-harm-margin",
        type=float,
        default=0.0,
        help="Student-best mode is hurt when its FDE delta exceeds this margin.",
    )
    parser.add_argument(
        "--proxy-features",
        default="auto",
        help="Comma-separated feature keys to analyze, or 'auto'.",
    )
    parser.add_argument("--proxy-bins", type=int, default=5)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"records": payload, "meta": {}}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload type for {path}: {type(payload)!r}")
    return payload


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _is_prediction_proxy_key(key: str, *, fast_branch: str) -> bool:
    if not key.startswith(f"{fast_branch}_"):
        return False
    if key.endswith("_threshold"):
        return False
    return any(marker in key for marker in PREDICTION_PROXY_MARKERS)


def _feature_category(feature: str) -> str:
    if feature.startswith("teacher_advantage_") or feature.startswith("teacher_better_"):
        return "teacher_signal"
    if feature in RESIDUAL_RUNTIME_FEATURES:
        return "residual_runtime"
    if feature.startswith("proxy_"):
        return "observable_scene_motion"
    if any(marker in feature for marker in PREDICTION_PROXY_MARKERS):
        return "prediction_proxy"
    return "other"


def _join_records(
    diagnose_records: Sequence[Mapping[str, Any]],
    per_sample_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_sample_by_eval_item: Dict[int, Mapping[str, Any]] = {}
    for record in per_sample_records:
        value = record.get("eval_item_index")
        if value is None:
            continue
        per_sample_by_eval_item[int(value)] = record

    joined: List[Dict[str, Any]] = []
    missing = 0
    duplicate_keys = len(per_sample_records) - len(per_sample_by_eval_item)
    for diagnose_record in diagnose_records:
        value = diagnose_record.get("eval_item_index")
        if value is None:
            missing += 1
            continue
        per_sample_record = per_sample_by_eval_item.get(int(value))
        if per_sample_record is None:
            missing += 1
            continue

        merged = dict(per_sample_record)
        for key, item in diagnose_record.items():
            if key in merged and key != "eval_item_index":
                merged[f"diagnose_{key}"] = item
            else:
                merged[key] = item
        joined.append(merged)

    return joined, {
        "diagnose_records": len(diagnose_records),
        "per_sample_records": len(per_sample_records),
        "joined_records": len(joined),
        "missing_diagnose_records": missing,
        "duplicate_per_sample_eval_item_keys": duplicate_keys,
        "join_key": "eval_item_index",
    }


def _add_teacher_advantage_features(
    records: Sequence[Dict[str, Any]],
    *,
    fast_branch: str,
    slow_branch: str,
) -> List[str]:
    added: List[str] = []
    for metric in DEFAULT_METRICS:
        fast_key = f"{fast_branch}_{metric}"
        slow_key = f"{slow_branch}_{metric}"
        advantage_key = f"teacher_advantage_{metric}"
        better_key = f"teacher_better_{metric}"
        has_any = False
        for record in records:
            fast_value = _numeric_value(record, fast_key)
            slow_value = _numeric_value(record, slow_key)
            if fast_value is None or slow_value is None:
                continue
            has_any = True
            advantage = fast_value - slow_value
            record[advantage_key] = advantage
            record[better_key] = bool(advantage > 0.0)
        if has_any:
            added.extend([advantage_key, better_key])
    return added


def _resolve_features(
    records: Sequence[Mapping[str, Any]],
    *,
    requested: str,
    fast_branch: str,
    added_teacher_features: Sequence[str],
) -> List[str]:
    present = {str(key) for record in records for key in record.keys()}
    if requested.strip().lower() != "auto":
        return [key.strip() for key in requested.split(",") if key.strip() in present]

    features: List[str] = []
    for key in DEFAULT_PROXY_FEATURES:
        if key in present:
            features.append(key)
    for key in RESIDUAL_RUNTIME_FEATURES:
        if key in present and key not in features:
            features.append(key)
    for key in added_teacher_features:
        if key in present and key not in features:
            features.append(key)

    extras = sorted(
        key
        for key in present
        if key not in features and _is_prediction_proxy_key(key, fast_branch=fast_branch)
    )
    return features + extras


def _build_bins(
    items: Sequence[Tuple[float, float, bool]],
    *,
    num_bins: int,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[0])
    bin_count = max(1, min(int(num_bins), len(ordered)))
    bins: List[Dict[str, Any]] = []
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        chunk = ordered[start:end]
        if not chunk:
            continue
        feature_values = [item[0] for item in chunk]
        deltas = [item[1] for item in chunk]
        labels = [item[2] for item in chunk]
        bins.append(
            {
                "bin_index": int(bin_index),
                "count": len(chunk),
                "feature_min": min(feature_values),
                "feature_max": max(feature_values),
                "feature_mean": _mean(feature_values),
                "fde_min_delta_mean": _mean(deltas),
                "harm_rate": float(sum(labels) / len(labels)),
            }
        )
    return bins


def _threshold_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    feature: str,
    direction: str,
) -> Optional[Dict[str, Any]]:
    items: List[Tuple[float, float, float]] = []
    for record in records:
        feature_value = _numeric_value(record, feature)
        student_fde = _numeric_value(record, "student_FDE_min")
        graduate_fde = _numeric_value(record, "graduate_FDE_min")
        if feature_value is None or student_fde is None or graduate_fde is None:
            continue
        items.append((feature_value, student_fde, graduate_fde))
    if len(items) < 2:
        return None

    graduate_mean = _mean([item[2] for item in items])
    student_mean = _mean([item[1] for item in items])
    oracle_mean = _mean([min(item[1], item[2]) for item in items])
    ordered = sorted(items, key=lambda item: item[0], reverse=direction == "high_feature_values")

    best: Optional[Dict[str, Any]] = None
    total = len(ordered)
    for count in range(1, total):
        abstain_ids = set(range(count))
        values = [
            item[1] if index in abstain_ids else item[2]
            for index, item in enumerate(ordered)
        ]
        mean_value = _mean(values)
        threshold = ordered[count - 1][0]
        candidate = {
            "direction": direction,
            "threshold": threshold,
            "abstain_to_student_count": count,
            "abstain_to_student_ratio": float(count / total),
            "final_FDE_min_mean": mean_value,
            "graduate_FDE_min_mean": graduate_mean,
            "student_FDE_min_mean": student_mean,
            "oracle_fast_graduate_FDE_min_mean": oracle_mean,
            "improvement_vs_graduate": graduate_mean - mean_value,
            "gap_to_oracle": mean_value - oracle_mean,
        }
        if best is None or float(candidate["final_FDE_min_mean"]) < float(best["final_FDE_min_mean"]):
            best = candidate
    return best


def _rank_features(features: Mapping[str, Mapping[str, Any]]) -> List[str]:
    def score(name: str) -> Tuple[float, float, float, int]:
        item = features[name]
        best_auc = item.get("best_auc")
        sweep = item.get("best_abstention")
        auc_score = -1.0 if best_auc is None else float(best_auc)
        corr_score = max(
            abs(float(item.get("spearman_fde_min_delta") or 0.0)),
            abs(float(item.get("pearson_fde_min_delta") or 0.0)),
        )
        sweep_score = 0.0
        if isinstance(sweep, Mapping):
            sweep_score = float(sweep.get("improvement_vs_graduate") or 0.0)
        return auc_score, corr_score, sweep_score, int(item.get("num_records", 0))

    return sorted(features.keys(), key=score, reverse=True)


def _summarize_target(records: Sequence[Mapping[str, Any]], *, harm_margin: float, student_best_harm_margin: float) -> Dict[str, Any]:
    deltas = [_numeric_value(record, "fde_min_delta") for record in records]
    deltas = [float(item) for item in deltas if item is not None]
    ade_deltas = [_numeric_value(record, "ade_min_delta") for record in records]
    ade_deltas = [float(item) for item in ade_deltas if item is not None]
    sb_deltas = [_numeric_value(record, "student_best_mode_fde_delta") for record in records]
    sb_deltas = [float(item) for item in sb_deltas if item is not None]

    graduate_values = [_numeric_value(record, "graduate_FDE_min") for record in records]
    student_values = [_numeric_value(record, "student_FDE_min") for record in records]
    paired = [
        (float(student), float(graduate))
        for student, graduate in zip(student_values, graduate_values)
        if student is not None and graduate is not None
    ]

    return {
        "num_records": len(records),
        "harm_margin": float(harm_margin),
        "student_best_harm_margin": float(student_best_harm_margin),
        "fde_min_delta": _summarize_values(deltas),
        "ade_min_delta": _summarize_values(ade_deltas),
        "student_best_mode_fde_delta": _summarize_values(sb_deltas),
        "fde_min_harm_count": sum(1 for item in deltas if item > float(harm_margin)),
        "fde_min_harm_rate": float(sum(1 for item in deltas if item > float(harm_margin)) / len(deltas)) if deltas else None,
        "student_best_hurt_count": sum(1 for item in sb_deltas if item > float(student_best_harm_margin)),
        "student_best_hurt_rate": (
            float(sum(1 for item in sb_deltas if item > float(student_best_harm_margin)) / len(sb_deltas))
            if sb_deltas
            else None
        ),
        "student_FDE_min_mean": _mean([item[0] for item in paired]) if paired else None,
        "graduate_FDE_min_mean": _mean([item[1] for item in paired]) if paired else None,
        "oracle_fast_graduate_FDE_min_mean": _mean([min(item[0], item[1]) for item in paired]) if paired else None,
    }


def analyze(
    *,
    diagnose_payload: Mapping[str, Any],
    per_sample_payload: Mapping[str, Any],
    fast_branch: str,
    slow_branch: str,
    harm_margin: float,
    student_best_harm_margin: float,
    proxy_features: str,
    proxy_bins: int,
    top_features: int,
    top_k: int,
) -> Dict[str, Any]:
    diagnose_records = diagnose_payload.get("records")
    if not isinstance(diagnose_records, list):
        raise ValueError(
            "Diagnose payload does not contain `records`. "
            "Rerun diagnose_residual_graduate.py with --save-records."
        )
    per_sample_records = per_sample_payload.get("records")
    if not isinstance(per_sample_records, list):
        raise ValueError("Per-sample payload does not contain `records`.")

    records, join_summary = _join_records(diagnose_records, per_sample_records)
    if not records:
        raise ValueError("No records could be joined. Check split/protocol/sample-mode alignment.")

    added_teacher_features = _add_teacher_advantage_features(records, fast_branch=fast_branch, slow_branch=slow_branch)
    features = _resolve_features(
        records,
        requested=proxy_features,
        fast_branch=fast_branch,
        added_teacher_features=added_teacher_features,
    )

    labels = [
        bool((_numeric_value(record, "fde_min_delta") or 0.0) > float(harm_margin))
        for record in records
    ]
    sb_labels = [
        bool((_numeric_value(record, "student_best_mode_fde_delta") or 0.0) > float(student_best_harm_margin))
        for record in records
    ]

    feature_analysis: Dict[str, Dict[str, Any]] = {}
    for feature in features:
        items: List[Tuple[float, float, bool]] = []
        sb_items: List[Tuple[float, bool]] = []
        for index, record in enumerate(records):
            feature_value = _numeric_value(record, feature)
            target_value = _numeric_value(record, "fde_min_delta")
            if feature_value is None or target_value is None:
                continue
            items.append((feature_value, target_value, labels[index]))
            sb_items.append((feature_value, sb_labels[index]))

        if len(items) < 2:
            continue

        feature_values = [item[0] for item in items]
        fde_deltas = [item[1] for item in items]
        harm_labels = [item[2] for item in items]
        auc_high = _auc(feature_values, harm_labels)
        best_auc = None if auc_high is None else max(float(auc_high), 1.0 - float(auc_high))
        best_direction = None
        if auc_high is not None:
            best_direction = "high_feature_values" if float(auc_high) >= 0.5 else "low_feature_values"

        sb_auc_high = _auc([item[0] for item in sb_items], [item[1] for item in sb_items])
        best_sb_auc = None if sb_auc_high is None else max(float(sb_auc_high), 1.0 - float(sb_auc_high))

        best_sweeps = []
        for direction in ("high_feature_values", "low_feature_values"):
            sweep = _threshold_sweep(records, feature=feature, direction=direction)
            if sweep is not None:
                best_sweeps.append(sweep)
        best_abstention = None
        if best_sweeps:
            best_abstention = min(best_sweeps, key=lambda item: float(item["final_FDE_min_mean"]))

        feature_analysis[feature] = {
            "category": _feature_category(feature),
            "num_records": len(items),
            "coverage_ratio": float(len(items) / len(records)),
            "feature": _summarize_values(feature_values),
            "fde_min_delta": _summarize_values(fde_deltas),
            "pearson_fde_min_delta": _pearson(feature_values, fde_deltas),
            "spearman_fde_min_delta": _spearman(feature_values, fde_deltas),
            "auc_harm_high_feature": auc_high,
            "best_auc": best_auc,
            "best_direction": best_direction,
            "auc_student_best_hurt_high_feature": sb_auc_high,
            "best_student_best_hurt_auc": best_sb_auc,
            "harm_rate": float(sum(harm_labels) / len(harm_labels)),
            "bins": _build_bins(items, num_bins=proxy_bins),
            "best_abstention": best_abstention,
        }

    ranked_features = _rank_features(feature_analysis)
    top_bad = sorted(
        records,
        key=lambda record: float(_numeric_value(record, "fde_min_delta") or 0.0),
        reverse=True,
    )[: max(int(top_k), 0)]
    top_good = sorted(
        records,
        key=lambda record: float(_numeric_value(record, "fde_min_delta") or 0.0),
    )[: max(int(top_k), 0)]

    return {
        "meta": {
            "script": "trustmoe_traj.scripts.analyze_residual_precheck",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fast_branch": fast_branch,
            "slow_branch": slow_branch,
            "proxy_features": proxy_features,
            "proxy_bins": int(proxy_bins),
            "top_features": int(top_features),
            "diagnose_meta": diagnose_payload.get("meta", {}),
            "per_sample_meta": per_sample_payload.get("meta", {}),
        },
        "join": join_summary,
        "target": _summarize_target(
            records,
            harm_margin=harm_margin,
            student_best_harm_margin=student_best_harm_margin,
        ),
        "features": feature_analysis,
        "ranked_features": ranked_features,
        "top_features": ranked_features[: max(int(top_features), 0)],
        "top_harm_records": top_bad,
        "top_improvement_records": top_good,
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    join = analysis["join"]
    target = analysis["target"]
    features = analysis["features"]
    ranked = list(analysis.get("ranked_features", []))
    top_features = list(analysis.get("top_features", []))

    lines: List[str] = []
    lines.append("# Residual Harm Precheck")
    lines.append("")
    lines.append("## Join")
    lines.append("")
    lines.append(f"- diagnose records: `{join['diagnose_records']}`")
    lines.append(f"- per-sample records: `{join['per_sample_records']}`")
    lines.append(f"- joined records: `{join['joined_records']}`")
    lines.append(f"- missing diagnose records: `{join['missing_diagnose_records']}`")
    lines.append("")

    lines.append("## Target")
    lines.append("")
    lines.append(f"- harm label: `fde_min_delta > {target['harm_margin']}`")
    lines.append(f"- FDE_min harm rate: `{_fmt(target.get('fde_min_harm_rate'), 4)}`")
    lines.append(f"- student-best hurt rate: `{_fmt(target.get('student_best_hurt_rate'), 4)}`")
    lines.append(f"- student FDE_min mean: `{_fmt(target.get('student_FDE_min_mean'))}`")
    lines.append(f"- graduate FDE_min mean: `{_fmt(target.get('graduate_FDE_min_mean'))}`")
    lines.append(f"- oracle fast/graduate FDE_min mean: `{_fmt(target.get('oracle_fast_graduate_FDE_min_mean'))}`")
    lines.append("")

    if ranked:
        lines.append("## Feature Predictiveness")
        lines.append("")
        lines.append(
            "| Feature | Category | N | Pearson | Spearman | Harm AUC | Direction | "
            "SB-hurt AUC | Best abstain FDE | Improve vs grad | Abstain ratio |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for feature in ranked[: len(top_features)]:
            item = features[feature]
            abstain = item.get("best_abstention") or {}
            lines.append(
                "| "
                f"`{feature}` | "
                f"{item.get('category')} | "
                f"{item.get('num_records')} | "
                f"{_fmt(item.get('pearson_fde_min_delta'), 4)} | "
                f"{_fmt(item.get('spearman_fde_min_delta'), 4)} | "
                f"{_fmt(item.get('best_auc'), 4)} | "
                f"{item.get('best_direction') or 'NA'} | "
                f"{_fmt(item.get('best_student_best_hurt_auc'), 4)} | "
                f"{_fmt(abstain.get('final_FDE_min_mean'))} | "
                f"{_fmt(abstain.get('improvement_vs_graduate'))} | "
                f"{_fmt(abstain.get('abstain_to_student_ratio'), 4)} |"
            )
        lines.append("")

    if top_features:
        lines.append("## Top Feature Bins")
        lines.append("")
        for feature in top_features[: min(5, len(top_features))]:
            item = features[feature]
            lines.append(f"### `{feature}`")
            lines.append("")
            lines.append("| Bin | Count | Feature min | Feature max | FDE delta mean | Harm rate |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for row in item.get("bins", []):
                lines.append(
                    "| "
                    f"{row.get('bin_index')} | "
                    f"{row.get('count')} | "
                    f"{_fmt(row.get('feature_min'))} | "
                    f"{_fmt(row.get('feature_max'))} | "
                    f"{_fmt(row.get('fde_min_delta_mean'))} | "
                    f"{_fmt(row.get('harm_rate'), 4)} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    diagnose_path = Path(args.diagnose_json).expanduser().resolve()
    per_sample_path = Path(args.per_sample_json).expanduser().resolve()
    diagnose_payload = _load_payload(diagnose_path)
    per_sample_payload = _load_payload(per_sample_path)

    analysis = analyze(
        diagnose_payload=diagnose_payload,
        per_sample_payload=per_sample_payload,
        fast_branch=str(args.fast_branch),
        slow_branch=str(args.slow_branch),
        harm_margin=float(args.harm_margin),
        student_best_harm_margin=float(args.student_best_harm_margin),
        proxy_features=str(args.proxy_features),
        proxy_bins=int(args.proxy_bins),
        top_features=int(args.top_features),
        top_k=int(args.top_k),
    )

    print("[analyze_residual_precheck] completed")
    print(f"joined_records={analysis['join']['joined_records']}")
    print(f"fde_min_harm_rate={analysis['target'].get('fde_min_harm_rate')}")
    print(f"student_best_hurt_rate={analysis['target'].get('student_best_hurt_rate')}")
    print(f"graduate_FDE_min_mean={analysis['target'].get('graduate_FDE_min_mean')}")
    print(f"oracle_fast_graduate_FDE_min_mean={analysis['target'].get('oracle_fast_graduate_FDE_min_mean')}")
    for feature in analysis.get("top_features", [])[: min(10, int(args.top_features))]:
        item = analysis["features"][feature]
        abstain = item.get("best_abstention") or {}
        print(
            "top_feature="
            f"{feature} category={item.get('category')} "
            f"best_auc={item.get('best_auc')} "
            f"spearman={item.get('spearman_fde_min_delta')} "
            f"abstain_improve={abstain.get('improvement_vs_graduate')}"
        )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")

    if args.output_md:
        output_path = Path(args.output_md).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(analysis), encoding="utf-8")
        print(f"output_md={output_path.as_posix()}")


if __name__ == "__main__":
    main()
