"""Diagnose whether current errors are student-limited or teacher-limited.

The script reads a teacher/student prediction cache produced by
``export_teacher_student_predictions.py`` and partitions eval items into:

- both_good
- teacher_good_student_bad
- student_good_teacher_bad
- both_bad

It can optionally join a per-sample JSON produced by
``eval_student_hidden_adapter.py --output-per-sample-json`` to check where a
hidden/query adapter improves or hurts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trustmoe_traj.data.transforms import PAST_SOCIAL_RISK_FEATURE_NAMES


CATEGORIES: Sequence[str] = (
    "both_good",
    "teacher_good_student_bad",
    "student_good_teacher_bad",
    "both_bad",
)
RELATIONS: Sequence[str] = ("teacher_better", "student_better", "similar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute fast-student failures to student compression vs teacher/model bottlenecks."
    )
    parser.add_argument("--cache-path", required=True, help="Teacher/student .pt cache path.")
    parser.add_argument(
        "--adapter-per-sample-json",
        default=None,
        help="Optional per-sample JSON from eval_student_hidden_adapter.py.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--bad-fde-threshold",
        type=float,
        default=None,
        help="Absolute FDE_min threshold for good/bad. If omitted, uses --bad-fde-quantile on student FDE.",
    )
    parser.add_argument(
        "--bad-fde-quantile",
        type=float,
        default=0.80,
        help="Student FDE_min quantile used as bad threshold when --bad-fde-threshold is omitted.",
    )
    parser.add_argument(
        "--teacher-student-margin",
        type=float,
        default=0.0,
        help="Teacher/student must beat the other by this FDE margin to count as better.",
    )
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--adapter-branch", default="hidden_adapter_pred")
    parser.add_argument("--fast-branch", default="fast_pred")
    parser.add_argument("--slow-branch", default="slow_pred")
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"records": payload, "meta": {}}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload type: {type(payload)!r}")
    return payload


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    return sum(clean) / len(clean) if clean else None


def _rate(values: Iterable[bool]) -> Optional[float]:
    clean = [bool(item) for item in values]
    return sum(1 for item in clean if item) / len(clean) if clean else None


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty values")
    ordered = sorted(float(item) for item in values)
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _to_float(tensor: torch.Tensor, row: int, agent: int) -> float:
    return float(tensor[row, agent].item())


def _to_bool(tensor: torch.Tensor, row: int, agent: int) -> bool:
    return bool(tensor[row, agent].item())


def _min_errors_from_prediction(prediction: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = prediction.detach().cpu().to(torch.float32)
    gt = ground_truth.detach().cpu().to(torch.float32)
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)
    if pred.ndim != 5:
        raise ValueError(f"prediction must have shape [N,K,A,T,2], got {tuple(pred.shape)}")
    if gt.ndim != 4:
        raise ValueError(f"ground_truth must have shape [N,A,T,2], got {tuple(gt.shape)}")
    distances = torch.linalg.norm(pred - gt[:, None, ...], dim=-1)
    ade = distances.mean(dim=-1)
    fde = distances[..., -1]
    return {
        "ADE_min": ade.min(dim=1).values.contiguous(),
        "FDE_min": fde.min(dim=1).values.contiguous(),
    }


def _resolve_min_error_tensors(tensors: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if "agent_mask" not in tensors:
        raise ValueError("Cache is missing required tensor: agent_mask")
    out: Dict[str, torch.Tensor] = {
        "agent_mask": tensors["agent_mask"].detach().cpu().bool(),
    }

    if "student_FDE_min_per_agent" in tensors and "student_ADE_min_per_agent" in tensors:
        out["student_FDE_min"] = tensors["student_FDE_min_per_agent"].detach().cpu().to(torch.float32)
        out["student_ADE_min"] = tensors["student_ADE_min_per_agent"].detach().cpu().to(torch.float32)
    else:
        for name in ("student_pred", "ground_truth"):
            if name not in tensors:
                raise ValueError(f"Cache missing {name}; cannot compute student min errors")
        student_errors = _min_errors_from_prediction(tensors["student_pred"], tensors["ground_truth"])
        out["student_FDE_min"] = student_errors["FDE_min"]
        out["student_ADE_min"] = student_errors["ADE_min"]

    if "teacher_FDE_min_per_agent" in tensors and "teacher_ADE_min_per_agent" in tensors:
        out["teacher_FDE_min"] = tensors["teacher_FDE_min_per_agent"].detach().cpu().to(torch.float32)
        out["teacher_ADE_min"] = tensors["teacher_ADE_min_per_agent"].detach().cpu().to(torch.float32)
    else:
        for name in ("teacher_pred", "ground_truth"):
            if name not in tensors:
                raise ValueError(f"Cache missing {name}; cannot compute teacher min errors")
        teacher_errors = _min_errors_from_prediction(tensors["teacher_pred"], tensors["ground_truth"])
        out["teacher_FDE_min"] = teacher_errors["FDE_min"]
        out["teacher_ADE_min"] = teacher_errors["ADE_min"]
    return out


def _get_record(records: Sequence[Mapping[str, Any]], row: int, agent: int) -> Dict[str, Any]:
    if row < len(records):
        record = dict(records[row])
    else:
        record = {}
    record.setdefault("cache_row_index", int(row))
    record.setdefault("agent_axis_index", int(agent))
    record.setdefault("eval_item_index", int(row))
    return record


def _categorize(*, student_fde: float, teacher_fde: float, bad_threshold: float) -> str:
    student_bad = student_fde > bad_threshold
    teacher_bad = teacher_fde > bad_threshold
    if not student_bad and not teacher_bad:
        return "both_good"
    if student_bad and not teacher_bad:
        return "teacher_good_student_bad"
    if not student_bad and teacher_bad:
        return "student_good_teacher_bad"
    return "both_bad"


def _relation(*, student_fde: float, teacher_fde: float, margin: float) -> str:
    margin = float(margin)
    if teacher_fde + margin < student_fde:
        return "teacher_better"
    if student_fde + margin < teacher_fde:
        return "student_better"
    return "similar"


def _flatten_cache(payload: Mapping[str, Any], *, bad_threshold: float, margin: float, miss_threshold: float) -> List[Dict[str, Any]]:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("Cache payload is missing `tensors`")
    min_errors = _resolve_min_error_tensors(tensors)

    records = payload.get("records", [])
    if not isinstance(records, list):
        records = []

    student_fde = min_errors["student_FDE_min"]
    teacher_fde = min_errors["teacher_FDE_min"]
    student_ade = min_errors["student_ADE_min"]
    teacher_ade = min_errors["teacher_ADE_min"]
    mask = min_errors["agent_mask"]
    social_risk = tensors.get("past_social_risk_features")
    if isinstance(social_risk, torch.Tensor):
        social_risk = social_risk.detach().cpu().to(torch.float32)
    else:
        social_risk = None

    rows: List[Dict[str, Any]] = []
    for row_index in range(int(student_fde.shape[0])):
        for agent_index in range(int(student_fde.shape[1])):
            if not _to_bool(mask, row_index, agent_index):
                continue
            s_fde = _to_float(student_fde, row_index, agent_index)
            t_fde = _to_float(teacher_fde, row_index, agent_index)
            s_ade = _to_float(student_ade, row_index, agent_index)
            t_ade = _to_float(teacher_ade, row_index, agent_index)
            record = _get_record(records, row_index, agent_index)
            item: Dict[str, Any] = {
                "eval_item_index": int(record.get("eval_item_index", row_index)),
                "cache_row_index": int(record.get("cache_row_index", row_index)),
                "agent_axis_index": int(agent_index),
                "selected_scene_index": record.get("selected_scene_index"),
                "source_agent_index": record.get("source_agent_index"),
                "dataset": record.get("dataset"),
                "subset": record.get("subset"),
                "split": record.get("split"),
                "scene_meta": record.get("scene_meta"),
                "student_ADE_min": s_ade,
                "student_FDE_min": s_fde,
                "teacher_ADE_min": t_ade,
                "teacher_FDE_min": t_fde,
                "student_Miss": bool(s_fde > float(miss_threshold)),
                "teacher_Miss": bool(t_fde > float(miss_threshold)),
                "teacher_advantage_FDE_min": s_fde - t_fde,
                "teacher_advantage_ADE_min": s_ade - t_ade,
                "oracle_student_teacher_FDE_min": min(s_fde, t_fde),
                "category": _categorize(student_fde=s_fde, teacher_fde=t_fde, bad_threshold=bad_threshold),
                "relation": _relation(student_fde=s_fde, teacher_fde=t_fde, margin=margin),
            }
            if social_risk is not None:
                values = social_risk[row_index, agent_index]
                for feature_index, name in enumerate(PAST_SOCIAL_RISK_FEATURE_NAMES):
                    if feature_index < int(values.shape[0]):
                        item[f"past_social_risk_{name}"] = float(values[feature_index].item())
            rows.append(item)
    return rows


def _resolve_bad_threshold(payload: Mapping[str, Any], *, explicit: Optional[float], quantile: float) -> Dict[str, Any]:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("Cache payload is missing `tensors`")
    min_errors = _resolve_min_error_tensors(tensors)
    student_fde = min_errors["student_FDE_min"]
    mask = min_errors["agent_mask"]
    values = [float(item) for item in student_fde[mask].tolist()]
    if explicit is not None:
        return {
            "bad_fde_threshold": float(explicit),
            "threshold_source": "explicit",
            "bad_fde_quantile": None,
            "student_FDE_min_mean": _mean(values),
        }
    threshold = _quantile(values, float(quantile))
    return {
        "bad_fde_threshold": float(threshold),
        "threshold_source": "student_fde_quantile",
        "bad_fde_quantile": float(quantile),
        "student_FDE_min_mean": _mean(values),
    }


def _index_adapter_records(path: Path, *, adapter_branch: str, fast_branch: str, slow_branch: str) -> Dict[int, Dict[str, Any]]:
    payload = _load_json(path)
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("Adapter per-sample payload `records` must be a list")
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping) or record.get("eval_item_index") is None:
            continue
        eval_item_index = int(record["eval_item_index"])
        adapter_fde = record.get(f"{adapter_branch}_FDE_min")
        fast_fde = record.get(f"{fast_branch}_FDE_min")
        slow_fde = record.get(f"{slow_branch}_FDE_min")
        item = {
            "adapter_FDE_min": None if adapter_fde is None else float(adapter_fde),
            "adapter_ADE_min": None if record.get(f"{adapter_branch}_ADE_min") is None else float(record[f"{adapter_branch}_ADE_min"]),
            "eval_fast_FDE_min": None if fast_fde is None else float(fast_fde),
            "eval_slow_FDE_min": None if slow_fde is None else float(slow_fde),
            "adapter_Miss": record.get(f"{adapter_branch}_Miss"),
            "eval_fast_Miss": record.get(f"{fast_branch}_Miss"),
            "eval_slow_Miss": record.get(f"{slow_branch}_Miss"),
        }
        if item["adapter_FDE_min"] is not None and item["eval_fast_FDE_min"] is not None:
            item["adapter_delta_vs_eval_fast_FDE_min"] = item["adapter_FDE_min"] - item["eval_fast_FDE_min"]
            item["adapter_improved_eval_fast"] = item["adapter_delta_vs_eval_fast_FDE_min"] < 0.0
            item["adapter_hurt_eval_fast"] = item["adapter_delta_vs_eval_fast_FDE_min"] > 0.0
        indexed[eval_item_index] = item
    return indexed


def _attach_adapter(rows: List[Dict[str, Any]], adapter_records: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    matched = 0
    for row in rows:
        adapter = adapter_records.get(int(row["eval_item_index"]))
        if not adapter:
            continue
        matched += 1
        row.update(adapter)
    return {
        "adapter_records": len(adapter_records),
        "matched_records": matched,
        "match_ratio": matched / len(rows) if rows else None,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "student_FDE_min_mean": _mean(row.get("student_FDE_min") for row in rows),
        "teacher_FDE_min_mean": _mean(row.get("teacher_FDE_min") for row in rows),
        "teacher_advantage_FDE_min_mean": _mean(row.get("teacher_advantage_FDE_min") for row in rows),
        "oracle_student_teacher_FDE_min_mean": _mean(row.get("oracle_student_teacher_FDE_min") for row in rows),
        "student_Miss_rate": _rate(bool(row.get("student_Miss")) for row in rows),
        "teacher_Miss_rate": _rate(bool(row.get("teacher_Miss")) for row in rows),
        "teacher_better_rate": _rate(row.get("relation") == "teacher_better" for row in rows),
        "student_better_rate": _rate(row.get("relation") == "student_better" for row in rows),
        "similar_rate": _rate(row.get("relation") == "similar" for row in rows),
        "adapter_FDE_min_mean": _mean(row.get("adapter_FDE_min") for row in rows),
        "adapter_delta_vs_eval_fast_FDE_min_mean": _mean(row.get("adapter_delta_vs_eval_fast_FDE_min") for row in rows),
        "adapter_improved_eval_fast_rate": _rate(bool(row.get("adapter_improved_eval_fast")) for row in rows if row.get("adapter_improved_eval_fast") is not None),
        "adapter_hurt_eval_fast_rate": _rate(bool(row.get("adapter_hurt_eval_fast")) for row in rows if row.get("adapter_hurt_eval_fast") is not None),
        "past_social_risk_nearest_distance_mean": _mean(row.get("past_social_risk_nearest_distance") for row in rows),
        "past_social_risk_close_count_1p0_mean": _mean(row.get("past_social_risk_close_count_1p0") for row in rows),
        "past_social_risk_max_approaching_speed_mean": _mean(row.get("past_social_risk_max_approaching_speed") for row in rows),
    }


def _summaries_by_key(rows: Sequence[Mapping[str, Any]], key: str, values: Sequence[str]) -> Dict[str, Any]:
    return {
        value: _summarize_rows([row for row in rows if row.get(key) == value])
        for value in values
    }


def _decision_hints(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    total = len(rows)
    if total <= 0:
        return ["No valid rows were available."]
    by_category = {category: [row for row in rows if row.get("category") == category] for category in CATEGORIES}
    student_bad = by_category["teacher_good_student_bad"] + by_category["both_bad"]
    hints: List[str] = []
    if student_bad:
        teacher_rescue_ratio = len(by_category["teacher_good_student_bad"]) / len(student_bad)
        both_bad_ratio = len(by_category["both_bad"]) / len(student_bad)
        if teacher_rescue_ratio >= 0.4:
            hints.append("Many student-bad cases are teacher-good: student-side internal compensation likely has room.")
        if both_bad_ratio >= 0.5:
            hints.append("Most student-bad cases are also teacher-bad: teacher/model/interaction capacity may be the bottleneck.")
    student_good_teacher_bad_ratio = len(by_category["student_good_teacher_bad"]) / total
    if student_good_teacher_bad_ratio >= 0.1:
        hints.append("Student-good/teacher-bad cases are non-trivial: teacher should not be used as a hard ceiling.")
    adapter_values = [row for row in rows if row.get("adapter_delta_vs_eval_fast_FDE_min") is not None]
    if adapter_values:
        both_good_rows = [row for row in adapter_values if row.get("category") == "both_good"]
        both_good_hurt = _rate(bool(row.get("adapter_hurt_eval_fast")) for row in both_good_rows)
        if both_good_hurt is not None and both_good_hurt > 0.5:
            hints.append("Adapter often hurts both-good cases: correction gate should learn to stay low on already-good samples.")
    if not hints:
        hints.append("No single bottleneck dominates under the current thresholds; inspect category tables and top examples.")
    return hints


def _top_rows(rows: Sequence[Mapping[str, Any]], *, key: str, top_k: int, reverse: bool = True) -> List[Dict[str, Any]]:
    valid = [row for row in rows if row.get(key) is not None]
    ordered = sorted(valid, key=lambda row: float(row[key]), reverse=reverse)
    keep_keys = (
        "eval_item_index",
        "cache_row_index",
        "selected_scene_index",
        "source_agent_index",
        "category",
        "relation",
        "student_FDE_min",
        "teacher_FDE_min",
        "teacher_advantage_FDE_min",
        "adapter_FDE_min",
        "adapter_delta_vs_eval_fast_FDE_min",
    )
    return [{name: row.get(name) for name in keep_keys} for row in ordered[: max(int(top_k), 0)]]


def analyze(
    *,
    cache_path: Path,
    adapter_per_sample_json: Optional[Path],
    bad_fde_threshold: Optional[float],
    bad_fde_quantile: float,
    margin: float,
    miss_threshold: float,
    adapter_branch: str,
    fast_branch: str,
    slow_branch: str,
    top_k: int,
) -> Dict[str, Any]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid cache payload type: {type(payload)!r}")
    threshold_meta = _resolve_bad_threshold(payload, explicit=bad_fde_threshold, quantile=bad_fde_quantile)
    rows = _flatten_cache(
        payload,
        bad_threshold=float(threshold_meta["bad_fde_threshold"]),
        margin=float(margin),
        miss_threshold=float(miss_threshold),
    )

    adapter_join = None
    if adapter_per_sample_json is not None:
        adapter_records = _index_adapter_records(
            adapter_per_sample_json,
            adapter_branch=adapter_branch,
            fast_branch=fast_branch,
            slow_branch=slow_branch,
        )
        adapter_join = _attach_adapter(rows, adapter_records)

    by_category = _summaries_by_key(rows, "category", CATEGORIES)
    by_relation = _summaries_by_key(rows, "relation", RELATIONS)
    student_bad_rows = [row for row in rows if row.get("category") in {"teacher_good_student_bad", "both_bad"}]

    return {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_teacher_student_bottleneck",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cache_path": cache_path.as_posix(),
            "adapter_per_sample_json": None if adapter_per_sample_json is None else adapter_per_sample_json.as_posix(),
            "cache_meta": _jsonable(payload.get("meta", {})),
        },
        "threshold": threshold_meta,
        "settings": {
            "teacher_student_margin": float(margin),
            "miss_threshold": float(miss_threshold),
            "adapter_branch": adapter_branch,
            "fast_branch": fast_branch,
            "slow_branch": slow_branch,
        },
        "adapter_join": adapter_join,
        "overall": _summarize_rows(rows),
        "student_bad_subset": _summarize_rows(student_bad_rows),
        "by_category": by_category,
        "by_relation": by_relation,
        "decision_hints": _decision_hints(rows),
        "top_teacher_advantage": _top_rows(rows, key="teacher_advantage_FDE_min", top_k=top_k, reverse=True),
        "top_student_advantage": _top_rows(rows, key="teacher_advantage_FDE_min", top_k=top_k, reverse=False),
        "top_adapter_harm": _top_rows(rows, key="adapter_delta_vs_eval_fast_FDE_min", top_k=top_k, reverse=True),
        "top_adapter_improve": _top_rows(rows, key="adapter_delta_vs_eval_fast_FDE_min", top_k=top_k, reverse=False),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def render_markdown(analysis: Mapping[str, Any]) -> str:
    threshold = analysis["threshold"]
    lines: List[str] = []
    lines.append("# Teacher-Student Bottleneck Diagnosis")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- bad FDE threshold: `{_fmt(threshold.get('bad_fde_threshold'))}`")
    lines.append(f"- threshold source: `{threshold.get('threshold_source')}`")
    if threshold.get("bad_fde_quantile") is not None:
        lines.append(f"- bad FDE quantile: `{_fmt(threshold.get('bad_fde_quantile'), 3)}`")
    lines.append(f"- teacher/student margin: `{_fmt(analysis['settings'].get('teacher_student_margin'))}`")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    overall = analysis["overall"]
    lines.append(f"- count: `{overall.get('count')}`")
    lines.append(f"- student FDE_min mean: `{_fmt(overall.get('student_FDE_min_mean'))}`")
    lines.append(f"- teacher FDE_min mean: `{_fmt(overall.get('teacher_FDE_min_mean'))}`")
    lines.append(f"- teacher-student oracle FDE_min mean: `{_fmt(overall.get('oracle_student_teacher_FDE_min_mean'))}`")
    lines.append(f"- teacher better rate: `{_fmt(overall.get('teacher_better_rate'), 4)}`")
    lines.append(f"- student better rate: `{_fmt(overall.get('student_better_rate'), 4)}`")
    lines.append("")

    lines.append("## Category Summary")
    lines.append("")
    lines.append(
        "| Category | Count | Student FDE | Teacher FDE | Oracle FDE | Teacher better | Adapter dFDE vs eval fast | Adapter improve |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for category in CATEGORIES:
        item = analysis["by_category"][category]
        lines.append(
            "| "
            f"`{category}` | "
            f"{item.get('count')} | "
            f"{_fmt(item.get('student_FDE_min_mean'))} | "
            f"{_fmt(item.get('teacher_FDE_min_mean'))} | "
            f"{_fmt(item.get('oracle_student_teacher_FDE_min_mean'))} | "
            f"{_fmt(item.get('teacher_better_rate'), 4)} | "
            f"{_fmt(item.get('adapter_delta_vs_eval_fast_FDE_min_mean'))} | "
            f"{_fmt(item.get('adapter_improved_eval_fast_rate'), 4)} |"
        )
    lines.append("")

    lines.append("## Decision Hints")
    lines.append("")
    for hint in analysis.get("decision_hints", []):
        lines.append(f"- {hint}")
    lines.append("")

    if analysis.get("adapter_join") is not None:
        join = analysis["adapter_join"]
        lines.append("## Adapter Join")
        lines.append("")
        lines.append(f"- adapter records: `{join.get('adapter_records')}`")
        lines.append(f"- matched records: `{join.get('matched_records')}`")
        lines.append(f"- match ratio: `{_fmt(join.get('match_ratio'), 4)}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    cache_path = Path(args.cache_path).expanduser().resolve()
    adapter_path = Path(args.adapter_per_sample_json).expanduser().resolve() if args.adapter_per_sample_json else None
    analysis = analyze(
        cache_path=cache_path,
        adapter_per_sample_json=adapter_path,
        bad_fde_threshold=args.bad_fde_threshold,
        bad_fde_quantile=float(args.bad_fde_quantile),
        margin=float(args.teacher_student_margin),
        miss_threshold=float(args.miss_threshold),
        adapter_branch=str(args.adapter_branch),
        fast_branch=str(args.fast_branch),
        slow_branch=str(args.slow_branch),
        top_k=int(args.top_k),
    )

    print("[diagnose_teacher_student_bottleneck] completed")
    print(f"bad_fde_threshold={analysis['threshold']['bad_fde_threshold']}")
    print(f"overall_count={analysis['overall']['count']}")
    print(f"teacher_better_rate={analysis['overall']['teacher_better_rate']}")
    print(f"student_better_rate={analysis['overall']['student_better_rate']}")
    for category in CATEGORIES:
        item = analysis["by_category"][category]
        print(
            f"{category}: count={item['count']} "
            f"student_FDE={item.get('student_FDE_min_mean')} "
            f"teacher_FDE={item.get('teacher_FDE_min_mean')} "
            f"adapter_dFDE={item.get('adapter_delta_vs_eval_fast_FDE_min_mean')}"
        )
    for hint in analysis.get("decision_hints", []):
        print(f"hint={hint}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_jsonable(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"output_json={output_path.as_posix()}")

    if args.output_md:
        output_path = Path(args.output_md).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(analysis), encoding="utf-8")
        print(f"output_md={output_path.as_posix()}")


if __name__ == "__main__":
    main()
