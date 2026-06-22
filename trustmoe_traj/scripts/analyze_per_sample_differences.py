"""Analyze per-sample fast/slow differences from run_eval outputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_METRICS: Sequence[str] = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg")
DEFAULT_PROXY_FEATURES: Sequence[str] = (
    "proxy_scene_num_agents",
    "proxy_scene_bbox_area_last",
    "proxy_scene_bbox_diag_last",
    "proxy_scene_density_last",
    "proxy_scene_neighbor_min_dist_last",
    "proxy_scene_neighbor_mean_dist_last",
    "proxy_target_neighbor_min_dist_last",
    "proxy_target_neighbor_mean_dist_last",
    "proxy_history_speed_mean",
    "proxy_history_speed_max",
    "proxy_history_speed_std",
    "proxy_history_accel_mean",
    "proxy_history_accel_max",
    "proxy_history_heading_change_mean",
    "proxy_history_heading_change_max",
    "proxy_history_path_length",
    "proxy_history_displacement",
    "proxy_history_straightness",
    "fast_pred_num_modes",
    "fast_pred_trajectory_spread_mean",
    "fast_pred_trajectory_spread_max",
    "fast_pred_endpoint_variance",
    "fast_pred_endpoint_pairwise_dist_mean",
    "fast_pred_endpoint_pairwise_dist_max",
    "fast_pred_collision_min_dist",
    "fast_pred_collision_risk",
)
PREDICTION_PROXY_MARKERS: Sequence[str] = (
    "num_modes",
    "trajectory_spread",
    "endpoint_variance",
    "endpoint_pairwise",
    "collision",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize fast-vs-slow per-sample differences and oracle routing upper bound."
    )
    parser.add_argument("--per-sample-json", required=True, help="Path produced by run_eval --output-per-sample-json")
    parser.add_argument("--output-json", default=None, help="Optional machine-readable analysis JSON")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path")
    parser.add_argument("--fast-branch", default="fast_pred")
    parser.add_argument("--slow-branch", default="slow_pred")
    parser.add_argument("--route-metric", default="FDE_min", choices=DEFAULT_METRICS)
    parser.add_argument(
        "--route-delta",
        type=float,
        default=0.0,
        help="Call slow when fast_metric - slow_metric is larger than this margin.",
    )
    parser.add_argument(
        "--proxy-features",
        default="auto",
        help="Comma-separated proxy feature keys to analyze, or 'auto' for known scene/fast proxy keys.",
    )
    parser.add_argument("--proxy-bins", type=int, default=5, help="Equal-frequency bins per proxy feature")
    parser.add_argument("--top-proxies", type=int, default=12, help="How many ranked proxy features to show")
    parser.add_argument("--top-k", type=int, default=20, help="Top fast-failure samples to include in the report")
    return parser


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"records": payload, "meta": {}}
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Invalid per-sample payload: {path}")
    return payload


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_values(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": _mean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else float("nan"),
    }


def _numeric_value(record: Mapping[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][1] == order[index][1]:
            end += 1
        average_rank = float((index + 1 + end) / 2.0)
        for position in range(index, end):
            ranks[order[position][0]] = average_rank
        index = end
    return ranks


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_var = sum(value * value for value in x_centered)
    y_var = sum(value * value for value in y_centered)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    numerator = sum(x_val * y_val for x_val, y_val in zip(x_centered, y_centered))
    return float(numerator / math.sqrt(x_var * y_var))


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return _pearson(_average_ranks(x_values), _average_ranks(y_values))


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    positives = int(sum(1 for label in labels if label))
    negatives = int(len(labels) - positives)
    if positives <= 0 or negatives <= 0:
        return None

    ranks = _average_ranks(scores)
    rank_sum_positive = sum(rank for rank, label in zip(ranks, labels) if label)
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _looks_like_auto_proxy_key(key: str, *, fast_branch: str) -> bool:
    if key.startswith("proxy_"):
        return True
    if not key.startswith(f"{fast_branch}_"):
        return False
    if key.endswith("_threshold"):
        return False
    return any(marker in key for marker in PREDICTION_PROXY_MARKERS)


def _resolve_proxy_features(
    records: Sequence[Mapping[str, Any]],
    *,
    requested: str,
    fast_branch: str,
) -> List[str]:
    present_keys = {str(key) for record in records for key in record.keys()}
    if requested.strip().lower() != "auto":
        return [feature.strip() for feature in requested.split(",") if feature.strip() in present_keys]

    features: List[str] = [feature for feature in DEFAULT_PROXY_FEATURES if feature in present_keys]
    extras = sorted(
        key
        for key in present_keys
        if key not in features and _looks_like_auto_proxy_key(key, fast_branch=fast_branch)
    )
    return features + extras


def _build_feature_bins(
    items: Sequence[Tuple[float, float, bool, float, float]],
    *,
    num_bins: int,
) -> List[Dict[str, float]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[0])
    bins: List[Dict[str, float]] = []
    bin_count = max(1, min(int(num_bins), len(ordered)))
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        chunk = ordered[start:end]
        if not chunk:
            continue
        feature_values = [item[0] for item in chunk]
        deltas = [item[1] for item in chunk]
        labels = [item[2] for item in chunk]
        fast_values = [item[3] for item in chunk]
        slow_values = [item[4] for item in chunk]
        bins.append(
            {
                "bin_index": float(bin_index),
                "count": float(len(chunk)),
                "feature_min": float(min(feature_values)),
                "feature_max": float(max(feature_values)),
                "feature_mean": _mean(feature_values),
                "fast_minus_slow_mean": _mean(deltas),
                "slow_needed_ratio": float(sum(labels) / len(labels)),
                "fast_route_metric_mean": _mean(fast_values),
                "slow_route_metric_mean": _mean(slow_values),
            }
        )
    return bins


def _rank_proxy_features(features: Mapping[str, Mapping[str, Any]]) -> List[str]:
    def score(feature_name: str) -> Tuple[float, float, int]:
        item = features[feature_name]
        best_auc = item.get("best_auc")
        spearman = item.get("spearman_fast_minus_slow")
        auc_score = -1.0 if best_auc is None else float(best_auc)
        corr_score = -1.0 if spearman is None else abs(float(spearman))
        return auc_score, corr_score, int(item.get("num_records", 0))

    return sorted(features.keys(), key=score, reverse=True)


def _metric_key(branch: str, metric: str) -> str:
    return f"{branch}_{metric}"


def _is_complete_record(record: Mapping[str, Any], fast_branch: str, slow_branch: str) -> bool:
    return all(_metric_key(branch, metric) in record for branch in (fast_branch, slow_branch) for metric in DEFAULT_METRICS)


def _coerce_float(record: Mapping[str, Any], key: str) -> float:
    return float(record[key])


def _short_source_file(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return Path(str(value)).name


def _analyze_proxy_features(
    records: Sequence[Mapping[str, Any]],
    *,
    proxy_features: Sequence[str],
    fast_branch: str,
    slow_branch: str,
    route_metric: str,
    route_delta: float,
    proxy_bins: int,
) -> Dict[str, Any]:
    route_key_fast = _metric_key(fast_branch, route_metric)
    route_key_slow = _metric_key(slow_branch, route_metric)
    feature_summary: Dict[str, Dict[str, Any]] = {}

    for feature in proxy_features:
        items: List[Tuple[float, float, bool, float, float]] = []
        for record in records:
            feature_value = _numeric_value(record, feature)
            if feature_value is None:
                continue
            fast_value = _numeric_value(record, route_key_fast)
            slow_value = _numeric_value(record, route_key_slow)
            if fast_value is None or slow_value is None:
                continue
            delta = fast_value - slow_value
            items.append((feature_value, delta, delta > float(route_delta), fast_value, slow_value))

        if not items:
            continue

        feature_values = [item[0] for item in items]
        deltas = [item[1] for item in items]
        labels = [item[2] for item in items]
        auc_high = _auc(feature_values, labels)
        best_auc = None
        direction = None
        if auc_high is not None:
            best_auc = max(auc_high, 1.0 - auc_high)
            direction = "high_feature_values" if auc_high >= 0.5 else "low_feature_values"

        feature_summary[feature] = {
            "num_records": len(items),
            "coverage_ratio": float(len(items) / len(records)) if records else 0.0,
            "feature": _summarize_values(feature_values),
            "fast_minus_slow": _summarize_values(deltas),
            "pearson_fast_minus_slow": _pearson(feature_values, deltas),
            "spearman_fast_minus_slow": _spearman(feature_values, deltas),
            "auc_slow_needed_high_feature": auc_high,
            "best_auc": best_auc,
            "best_direction": direction,
            "slow_needed_ratio": float(sum(labels) / len(labels)),
            "bins": _build_feature_bins(items, num_bins=proxy_bins),
        }

    ranked = _rank_proxy_features(feature_summary)
    return {
        "target": {
            "route_metric": route_metric,
            "route_delta": float(route_delta),
            "continuous_target": f"{fast_branch}_{route_metric} - {slow_branch}_{route_metric}",
            "positive_label": f"{fast_branch}_{route_metric} - {slow_branch}_{route_metric} > {route_delta}",
        },
        "requested_features": list(proxy_features),
        "features": feature_summary,
        "ranked_features": ranked,
    }


def analyze_payload(
    payload: Mapping[str, Any],
    *,
    fast_branch: str,
    slow_branch: str,
    route_metric: str,
    route_delta: float,
    proxy_features: str,
    proxy_bins: int,
    top_proxies: int,
    top_k: int,
) -> Dict[str, Any]:
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("per-sample payload `records` must be a list")

    records = [
        record
        for record in raw_records
        if isinstance(record, Mapping) and _is_complete_record(record, fast_branch, slow_branch)
    ]
    if not records:
        raise ValueError("No records contain both fast and slow metrics")

    metric_summary: Dict[str, Dict[str, Any]] = {}
    for metric in DEFAULT_METRICS:
        fast_values = [_coerce_float(record, _metric_key(fast_branch, metric)) for record in records]
        slow_values = [_coerce_float(record, _metric_key(slow_branch, metric)) for record in records]
        deltas = [fast - slow for fast, slow in zip(fast_values, slow_values)]
        oracle_values = [min(fast, slow) for fast, slow in zip(fast_values, slow_values)]

        metric_summary[metric] = {
            "fast": _summarize_values(fast_values),
            "slow": _summarize_values(slow_values),
            "oracle_best_of_fast_slow": _summarize_values(oracle_values),
            "fast_minus_slow": _summarize_values(deltas),
            "fast_better_count": sum(1 for delta in deltas if delta < -route_delta),
            "slow_better_count": sum(1 for delta in deltas if delta > route_delta),
            "tie_count": sum(1 for delta in deltas if abs(delta) <= route_delta),
        }

    route_key_fast = _metric_key(fast_branch, route_metric)
    route_key_slow = _metric_key(slow_branch, route_metric)
    route_deltas = [_coerce_float(record, route_key_fast) - _coerce_float(record, route_key_slow) for record in records]
    slow_needed = [delta > float(route_delta) for delta in route_deltas]
    resolved_proxy_features = _resolve_proxy_features(
        records,
        requested=proxy_features,
        fast_branch=fast_branch,
    )
    proxy_analysis = _analyze_proxy_features(
        records,
        proxy_features=resolved_proxy_features,
        fast_branch=fast_branch,
        slow_branch=slow_branch,
        route_metric=route_metric,
        route_delta=route_delta,
        proxy_bins=proxy_bins,
    )
    ranked_proxy_features = proxy_analysis["ranked_features"]
    top_proxy_features = ranked_proxy_features[: max(int(top_proxies), 0)]

    top_failures = sorted(
        records,
        key=lambda record: _coerce_float(record, route_key_fast) - _coerce_float(record, route_key_slow),
        reverse=True,
    )[: max(int(top_k), 0)]

    top_failure_rows: List[Dict[str, Any]] = []
    for record in top_failures:
        fast_value = _coerce_float(record, route_key_fast)
        slow_value = _coerce_float(record, route_key_slow)
        top_failure_rows.append(
            {
                "eval_item_index": record.get("eval_item_index"),
                "selected_scene_index": record.get("selected_scene_index"),
                "source_agent_index": record.get("source_agent_index"),
                "source_file": record.get("source_file"),
                "seq_id": record.get("seq_id"),
                "frame_id": record.get("frame_id"),
                f"{fast_branch}_{route_metric}": fast_value,
                f"{slow_branch}_{route_metric}": slow_value,
                "fast_minus_slow": fast_value - slow_value,
                f"{fast_branch}_ADE_min": record.get(f"{fast_branch}_ADE_min"),
                f"{slow_branch}_ADE_min": record.get(f"{slow_branch}_ADE_min"),
                "proxy_features": {
                    feature: _numeric_value(record, feature)
                    for feature in top_proxy_features
                    if _numeric_value(record, feature) is not None
                },
            }
        )

    proxy_analysis["top_features"] = top_proxy_features

    return {
        "meta": {
            "script": "trustmoe_traj.scripts.analyze_per_sample_differences",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_meta": payload.get("meta", {}),
            "route_metric": route_metric,
            "route_delta": float(route_delta),
            "fast_branch": fast_branch,
            "slow_branch": slow_branch,
            "proxy_features": proxy_features,
            "proxy_bins": int(proxy_bins),
            "top_proxies": int(top_proxies),
        },
        "counts": {
            "raw_records": len(raw_records),
            "paired_records": len(records),
            "slow_needed_count": int(sum(slow_needed)),
            "slow_needed_ratio": float(sum(slow_needed) / len(records)),
            "fast_accepted_count": int(len(records) - sum(slow_needed)),
            "fast_accepted_ratio": float((len(records) - sum(slow_needed)) / len(records)),
        },
        "metrics": metric_summary,
        "proxy_analysis": proxy_analysis,
        "top_fast_failures": top_failure_rows,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(analysis: Mapping[str, Any]) -> str:
    meta = analysis["meta"]
    counts = analysis["counts"]
    metrics = analysis["metrics"]
    source_meta = meta.get("source_meta", {})

    lines: List[str] = []
    lines.append("# Fast/Slow Per-Sample Difference Analysis")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- subset: `{source_meta.get('subset', '')}`")
    lines.append(f"- split: `{source_meta.get('split', '')}`")
    lines.append(f"- baseline: `{source_meta.get('baseline', '')}`")
    lines.append(f"- protocol: `{source_meta.get('protocol', '')}`")
    lines.append(f"- route metric: `{meta['route_metric']}`")
    lines.append(f"- route delta: `{meta['route_delta']}`")
    lines.append(f"- paired records: `{counts['paired_records']}` / raw `{counts['raw_records']}`")
    lines.append(f"- slow-needed ratio: `{_fmt(counts['slow_needed_ratio'], 4)}`")
    lines.append("")

    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Fast mean | Slow mean | Oracle mean | Fast-Slow mean | Fast-Slow p95 | Slow better | Fast better | Tie |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for metric in DEFAULT_METRICS:
        item = metrics[metric]
        lines.append(
            "| "
            f"{metric} | "
            f"{_fmt(item['fast']['mean'])} | "
            f"{_fmt(item['slow']['mean'])} | "
            f"{_fmt(item['oracle_best_of_fast_slow']['mean'])} | "
            f"{_fmt(item['fast_minus_slow']['mean'])} | "
            f"{_fmt(item['fast_minus_slow']['p95'])} | "
            f"{item['slow_better_count']} | "
            f"{item['fast_better_count']} | "
            f"{item['tie_count']} |"
        )
    lines.append("")

    lines.append("## Routing Upper Bound")
    lines.append("")
    route_metric = meta["route_metric"]
    route_item = metrics[route_metric]
    lines.append(
        f"- If routing had oracle access to `{route_metric}`, the mean `{route_metric}` would move "
        f"from fast `{_fmt(route_item['fast']['mean'])}` / slow `{_fmt(route_item['slow']['mean'])}` "
        f"to oracle `{_fmt(route_item['oracle_best_of_fast_slow']['mean'])}`."
    )
    lines.append(
        f"- With delta `{meta['route_delta']}`, slow would be called for "
        f"`{counts['slow_needed_count']}` samples (`{_fmt(counts['slow_needed_ratio'], 4)}`)."
    )
    lines.append("")

    proxy_analysis = analysis.get("proxy_analysis", {})
    proxy_features = proxy_analysis.get("features", {})
    ranked_proxy_features = list(proxy_analysis.get("ranked_features", []))
    top_proxy_features = list(proxy_analysis.get("top_features", []))
    if ranked_proxy_features:
        lines.append("## Proxy Predictiveness")
        lines.append("")
        lines.append(
            "| Feature | N | Coverage | Pearson delta | Spearman delta | AUC high | Best AUC | Direction | Slow-needed |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
        for feature in ranked_proxy_features[: max(int(meta.get("top_proxies", 12)), 0)]:
            item = proxy_features[feature]
            lines.append(
                "| "
                f"`{feature}` | "
                f"{item.get('num_records')} | "
                f"{_fmt(item.get('coverage_ratio'), 3)} | "
                f"{_fmt(item.get('pearson_fast_minus_slow'), 4)} | "
                f"{_fmt(item.get('spearman_fast_minus_slow'), 4)} | "
                f"{_fmt(item.get('auc_slow_needed_high_feature'), 4)} | "
                f"{_fmt(item.get('best_auc'), 4)} | "
                f"{item.get('best_direction') or 'NA'} | "
                f"{_fmt(item.get('slow_needed_ratio'), 4)} |"
            )
        lines.append("")

        lines.append("## Proxy Binning")
        lines.append("")
        for feature in ranked_proxy_features[: min(5, max(int(meta.get("top_proxies", 12)), 0))]:
            item = proxy_features[feature]
            lines.append(f"### `{feature}`")
            lines.append("")
            lines.append(
                "| Bin | Count | Feature min | Feature max | Feature mean | Fast-Slow mean | Slow-needed | Fast mean | Slow mean |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for bin_item in item.get("bins", []):
                lines.append(
                    "| "
                    f"{int(bin_item.get('bin_index', 0))} | "
                    f"{int(bin_item.get('count', 0))} | "
                    f"{_fmt(bin_item.get('feature_min'))} | "
                    f"{_fmt(bin_item.get('feature_max'))} | "
                    f"{_fmt(bin_item.get('feature_mean'))} | "
                    f"{_fmt(bin_item.get('fast_minus_slow_mean'))} | "
                    f"{_fmt(bin_item.get('slow_needed_ratio'), 4)} | "
                    f"{_fmt(bin_item.get('fast_route_metric_mean'))} | "
                    f"{_fmt(bin_item.get('slow_route_metric_mean'))} |"
                )
            lines.append("")

    lines.append("## Top Fast Failures")
    lines.append("")
    lines.append("| Rank | Eval Item | Scene | Agent | Source | Frame | Fast route metric | Slow route metric | Fast-Slow | Fast ADE_min | Slow ADE_min |")
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(analysis["top_fast_failures"], start=1):
        fast_key = f"{meta['fast_branch']}_{route_metric}"
        slow_key = f"{meta['slow_branch']}_{route_metric}"
        fast_ade_key = f"{meta['fast_branch']}_ADE_min"
        slow_ade_key = f"{meta['slow_branch']}_ADE_min"
        lines.append(
            "| "
            f"{rank} | "
            f"{row.get('eval_item_index')} | "
            f"{row.get('selected_scene_index')} | "
            f"{row.get('source_agent_index')} | "
            f"{_short_source_file(row.get('source_file'))} | "
            f"{row.get('frame_id')} | "
            f"{_fmt(row.get(fast_key))} | "
            f"{_fmt(row.get(slow_key))} | "
            f"{_fmt(row.get('fast_minus_slow'))} | "
            f"{_fmt(row.get(fast_ade_key))} | "
            f"{_fmt(row.get(slow_ade_key))} |"
        )
    lines.append("")

    snapshot_features = top_proxy_features[:5]
    if snapshot_features:
        lines.append("## Top Failure Proxy Snapshot")
        lines.append("")
        header = "| Rank | Eval Item | " + " | ".join(f"`{feature}`" for feature in snapshot_features) + " |"
        separator = "|---:|---:|" + "|".join("---:" for _feature in snapshot_features) + "|"
        lines.append(header)
        lines.append(separator)
        for rank, row in enumerate(analysis["top_fast_failures"], start=1):
            proxy_values = row.get("proxy_features", {})
            cells = " | ".join(_fmt(proxy_values.get(feature), 4) for feature in snapshot_features)
            lines.append(f"| {rank} | {row.get('eval_item_index')} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    per_sample_path = Path(args.per_sample_json).expanduser().resolve()
    payload = _load_payload(per_sample_path)
    analysis = analyze_payload(
        payload,
        fast_branch=args.fast_branch,
        slow_branch=args.slow_branch,
        route_metric=args.route_metric,
        route_delta=args.route_delta,
        proxy_features=args.proxy_features,
        proxy_bins=args.proxy_bins,
        top_proxies=args.top_proxies,
        top_k=args.top_k,
    )
    markdown = render_markdown(analysis)

    print(markdown)

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_json.as_posix()}")

    if args.output_md:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"output_md={output_md.as_posix()}")


if __name__ == "__main__":
    main()
