"""Audit SDD prediction bundles against the MoFlow SDD coordinate contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from trustmoe_traj.data.adapters.sdd import DEFAULT_SDD_DATA_ROOT, SDDAdapterConfig, SDDTrajectoryDataset
from trustmoe_traj.scripts.analogical_future_coverage import build_sdd_analogical_future_bank, split_float_list
from trustmoe_traj.scripts.run_eval import _coerce_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an exported SDD prediction bundle for compatibility with the "
            "MoFlow SDD evaluation convention: future_rel = future_abs - last_obs_abs."
        )
    )
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--baseline-name", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_SDD_DATA_ROOT))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--afc-max-train-scenes", type=int, default=1000)
    parser.add_argument("--afc-batch-scenes", type=int, default=256)
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.5")
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _finite_stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().cpu().to(dtype=torch.float32).reshape(-1)
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return {"count": float(values.numel()), "finite_count": 0.0}
    return {
        "count": float(values.numel()),
        "finite_count": float(finite.numel()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "abs_max": float(finite.abs().max().item()),
    }


def _append_stats(acc: Dict[str, List[torch.Tensor]], key: str, tensor: torch.Tensor) -> None:
    acc.setdefault(key, []).append(tensor.detach().cpu().to(dtype=torch.float32).reshape(-1))


def _record_to_batch(record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    agent_mask = _tensor(record["agent_mask"]).to(dtype=torch.bool)
    return {
        "past_traj_original_scale": _tensor(record["past_traj_original_scale"]).unsqueeze(0),
        "past_social_risk_features": _tensor(record["past_social_risk_features"]).unsqueeze(0),
        "fut_traj_original_scale": _tensor(record["fut_traj_original_scale"]).unsqueeze(0),
        "fut_traj_vel": _tensor(record["fut_traj_vel"]).unsqueeze(0),
        "agent_mask": agent_mask.unsqueeze(0),
    }


def _prediction(record: Mapping[str, Any]) -> torch.Tensor:
    return _tensor(record["prediction_rel"]).unsqueeze(0)


def _manual_ade_fde(record: Mapping[str, Any]) -> Dict[str, float]:
    pred = _tensor(record["prediction_rel"])  # [K,A,T,2]
    gt = _tensor(record["fut_traj_original_scale"])  # [A,T,2]
    if pred.ndim != 4 or gt.ndim != 3:
        return {}
    dist = torch.linalg.norm(pred - gt.unsqueeze(0), dim=-1)
    ade = dist.mean(dim=-1)  # [K,A]
    fde = dist[..., -1]  # [K,A]
    return {
        "ADE_min_manual": float(ade.min(dim=0).values.mean().item()),
        "FDE_min_manual": float(fde.min(dim=0).values.mean().item()),
        "ADE_avg_manual": float(ade.mean().item()),
        "FDE_avg_manual": float(fde.mean().item()),
    }


def _mean_metrics(items: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(str(key), []).append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in sorted(buckets.items()) if values}


def _max_abs_or_none(tensor: torch.Tensor) -> Optional[float]:
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if int(tensor.numel()) <= 0:
        return None
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) <= 0:
        return None
    return float(finite.abs().max().item())


def _as_agent_time_xy(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim == 2 and int(tensor.shape[-1]) == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[-1]) == 2:
        return tensor
    raise ValueError(f"{name} must have shape [T,2] or [A,T,2], got {tuple(tensor.shape)}")


def _record_consistency(record: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    obs = _as_agent_time_xy(_tensor(record["obs_abs"]), name="obs_abs")
    future = _as_agent_time_xy(_tensor(record["future_abs"]), name="future_abs")
    pred_abs = _tensor(record["prediction_abs"])
    pred_rel = _tensor(record["prediction_rel"])
    gt_rel = _tensor(record["fut_traj_original_scale"])
    last_obs = obs[:, -1:, :]
    return {
        "prediction_rel_max_abs_error": _max_abs_or_none((pred_abs - last_obs.unsqueeze(0)) - pred_rel),
        "gt_rel_max_abs_error": _max_abs_or_none((future - last_obs) - gt_rel),
    }


def _max_consistency(items: Iterable[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[float]] = {}
    for item in items:
        for key, value in item.items():
            if value is not None:
                buckets.setdefault(key, []).append(float(value))
    return {key: (max(values) if values else None) for key, values in sorted(buckets.items())}


def _sdd_dataset_scale(data_root: Path, split: str, max_records: Optional[int]) -> Dict[str, Any]:
    dataset = SDDTrajectoryDataset(SDDAdapterConfig(data_root=data_root, split=split, max_samples=max_records))
    acc: Dict[str, List[torch.Tensor]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        past = _tensor(sample["past_traj"][0])
        future = _tensor(sample["future_traj"][0])
        rel_future = future - past[-1:].expand_as(future)
        _append_stats(acc, "sdd_obs_abs", past)
        _append_stats(acc, "sdd_future_abs", future)
        _append_stats(acc, "sdd_future_rel", rel_future)
        _append_stats(acc, "sdd_future_endpoint_rel_norm", torch.linalg.norm(rel_future[-1], dim=-1).reshape(1))
    return {
        "summary": dataset.summary(),
        "stats": {key: _finite_stats(torch.cat(values)) for key, values in sorted(acc.items()) if values},
    }


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = Path(args.bundle).expanduser().resolve()
    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(bundle_path, map_location="cpu")
    records = list(payload.get("records", []))
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    if not records:
        raise SystemExit(f"No records found in bundle: {bundle_path}")

    bundle_meta = payload.get("meta", {})
    baseline_name = str(args.baseline_name or bundle_meta.get("baseline") or bundle_meta.get("model_name") or "unknown")

    acc: Dict[str, List[torch.Tensor]] = {}
    manual_metrics: List[Dict[str, float]] = []
    consistency: List[Dict[str, Optional[float]]] = []
    for record in records:
        obs = _tensor(record["obs_abs"])
        future = _tensor(record["future_abs"])
        pred_abs = _tensor(record["prediction_abs"])
        pred_rel = _tensor(record["prediction_rel"])
        gt_rel = _tensor(record["fut_traj_original_scale"])
        _append_stats(acc, "bundle_obs_abs", obs)
        _append_stats(acc, "bundle_future_abs", future)
        _append_stats(acc, "bundle_prediction_abs", pred_abs)
        _append_stats(acc, "bundle_prediction_rel", pred_rel)
        _append_stats(acc, "bundle_gt_future_rel", gt_rel)
        _append_stats(acc, "bundle_prediction_endpoint_rel_norm", torch.linalg.norm(pred_rel[..., -1, :], dim=-1))
        _append_stats(acc, "bundle_gt_endpoint_rel_norm", torch.linalg.norm(gt_rel[..., -1, :], dim=-1))
        manual_metrics.append(_manual_ade_fde(record))
        consistency.append(_record_consistency(record))

    data_root = Path(args.data_root).expanduser().resolve()
    eps_values = split_float_list(str(args.afc_eps))
    afc_metrics: List[Dict[str, float]] = []
    afc_error: Optional[str] = None
    try:
        bank = build_sdd_analogical_future_bank(
            data_root=data_root,
            train_split="train",
            sample_mode="per_scene",
            data_norm="original",
            rotate=False,
            rotate_time_frame=0,
            normalization_stats=None,
            max_train_scenes=args.afc_max_train_scenes,
            batch_scenes=int(args.afc_batch_scenes),
            top_m=int(args.afc_top_m),
            eps_values=eps_values,
        )
        for record in records:
            afc_metrics.append(bank.metrics_for_prediction(_prediction(record), _record_to_batch(record)))
    except Exception as exc:  # keep usable on machines without SDD pkl
        bank = None
        afc_error = repr(exc)

    dataset_scales: Dict[str, Any] = {}
    dataset_error: Optional[str] = None
    try:
        dataset_scales["train"] = _sdd_dataset_scale(data_root, "train", args.afc_max_train_scenes)
        dataset_scales[str(args.split)] = _sdd_dataset_scale(data_root, str(args.split), args.max_records)
    except Exception as exc:
        dataset_error = repr(exc)

    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.audit_sdd_prediction_bundle",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_name": baseline_name,
            "bundle": bundle_path.as_posix(),
            "data_root": data_root.as_posix(),
            "num_records": int(len(records)),
            "coordinate_contract": (
                "MoFlow SDD evaluates future trajectories as relative displacements in the SDD original "
                "coordinate scale: fut_traj_original_scale = future_abs - last_obs_abs; "
                "prediction_rel must use the same origin and scale."
            ),
            "unit_note": "SDD ADE/FDE are pixel-like SDD coordinates after each baseline adapter scale.",
            "afc_bank_size": None if bank is None else int(bank.bank_size),
            "afc_top_m": int(args.afc_top_m),
            "afc_eps": [float(item) for item in eps_values],
        },
        "bundle_meta": _coerce_jsonable(bundle_meta),
        "bundle_stats": {key: _finite_stats(torch.cat(values)) for key, values in sorted(acc.items()) if values},
        "coordinate_consistency_max": _max_consistency(consistency),
        "manual_metrics_mean": _mean_metrics(manual_metrics),
        "afc_metrics_mean": _mean_metrics(afc_metrics),
        "afc_error": afc_error,
        "sdd_dataset_scales": _coerce_jsonable(dataset_scales),
        "sdd_dataset_error": dataset_error,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_coerce_jsonable(output), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"output_json={output_path.as_posix()}")
    print(f"baseline_name={baseline_name}")
    print(f"records={len(records)}")
    for key, value in output["coordinate_consistency_max"].items():
        print(f"{key}={value}")
    for key in ("ADE_min_manual", "FDE_min_manual", "ADE_avg_manual", "FDE_avg_manual"):
        print(f"{key}={output['manual_metrics_mean'].get(key)}")
    for key in ("afc_retrieval_top1_distance", "afc_chamfer", "afc_weighted_mode_recall_eps05", "afc_unsupported_ratio_eps05"):
        print(f"{key}={output['afc_metrics_mean'].get(key)}")
    if afc_error:
        print(f"afc_error={afc_error}")
    if dataset_error:
        print(f"sdd_dataset_error={dataset_error}")


if __name__ == "__main__":
    main()
