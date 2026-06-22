"""Diagnose EigenTrajectory SDD bundle scale without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch

from trustmoe_traj.scripts.run_eval import _coerce_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether an EigenTrajectory SDD bundle has a simple scale error.")
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--scale-candidates", type=str, default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10")
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().cpu().to(dtype=torch.float32).reshape(-1)
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) == 0:
        return {"count": float(values.numel()), "finite_count": 0.0}
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=torch.float32)
    vals = torch.quantile(finite, qs)
    return {
        "count": float(values.numel()),
        "finite_count": float(finite.numel()),
        "min": float(vals[0].item()),
        "p01": float(vals[1].item()),
        "p05": float(vals[2].item()),
        "p25": float(vals[3].item()),
        "median": float(vals[4].item()),
        "p75": float(vals[5].item()),
        "p95": float(vals[6].item()),
        "p99": float(vals[7].item()),
        "max": float(vals[8].item()),
        "mean": float(finite.mean().item()),
        "abs_max": float(finite.abs().max().item()),
    }


def _parse_scales(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).replace(" ", ",").split(",") if item.strip()]


def _record_metrics(pred_rel: torch.Tensor, gt_rel: torch.Tensor) -> Dict[str, float]:
    dist = torch.linalg.norm(pred_rel - gt_rel.unsqueeze(0), dim=-1)
    ade = dist.mean(dim=-1)
    fde = dist[..., -1]
    return {
        "ADE_min": float(ade.min(dim=0).values.mean().item()),
        "FDE_min": float(fde.min(dim=0).values.mean().item()),
        "ADE_avg": float(ade.mean().item()),
        "FDE_avg": float(fde.mean().item()),
    }


def _mean_dicts(items: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    buckets: Dict[str, list[float]] = {}
    for item in items:
        for key, value in item.items():
            buckets.setdefault(str(key), []).append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in sorted(buckets.items()) if values}


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
        raise SystemExit(f"No records in bundle: {bundle_path}")

    pred_values = []
    gt_values = []
    pred_endpoint_norms = []
    gt_endpoint_norms = []
    original_metrics = []
    scale_metrics: Dict[float, list[Dict[str, float]]] = {scale: [] for scale in _parse_scales(args.scale_candidates)}

    consistency_errors = []
    for record in records:
        obs = _tensor(record["obs_abs"])
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        last_obs = obs[:, -1:, :]
        pred_abs = _tensor(record["prediction_abs"])
        pred_rel = _tensor(record["prediction_rel"])
        gt_rel = _tensor(record["fut_traj_original_scale"])
        future_abs = _tensor(record["future_abs"])
        if future_abs.ndim == 2:
            future_abs = future_abs.unsqueeze(0)

        pred_values.append(pred_rel.reshape(-1))
        gt_values.append(gt_rel.reshape(-1))
        pred_endpoint_norms.append(torch.linalg.norm(pred_rel[..., -1, :], dim=-1).reshape(-1))
        gt_endpoint_norms.append(torch.linalg.norm(gt_rel[..., -1, :], dim=-1).reshape(-1))
        consistency_errors.append(float(((pred_abs - last_obs.unsqueeze(0)) - pred_rel).abs().max().item()))
        consistency_errors.append(float(((future_abs - last_obs) - gt_rel).abs().max().item()))

        original_metrics.append(_record_metrics(pred_rel, gt_rel))
        for scale in scale_metrics:
            scale_metrics[scale].append(_record_metrics(pred_rel * float(scale), gt_rel))

    scale_rows = []
    for scale, items in sorted(scale_metrics.items(), key=lambda kv: kv[0]):
        row = {"scale": float(scale)}
        row.update(_mean_dicts(items))
        scale_rows.append(row)
    best_by_ade_avg = min(scale_rows, key=lambda row: row["ADE_avg"])
    best_by_ade_min = min(scale_rows, key=lambda row: row["ADE_min"])

    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.diagnose_eigentrajectory_sdd_scale",
            "bundle": bundle_path.as_posix(),
            "records": int(len(records)),
            "note": "Scales are applied to prediction_rel only, without retraining, to test for a simple multiplicative scale error.",
        },
        "bundle_meta": _coerce_jsonable(payload.get("meta", {})),
        "coordinate_consistency_abs_max": max(consistency_errors) if consistency_errors else None,
        "stats": {
            "prediction_rel": _quantiles(torch.cat(pred_values)),
            "gt_future_rel": _quantiles(torch.cat(gt_values)),
            "prediction_endpoint_rel_norm": _quantiles(torch.cat(pred_endpoint_norms)),
            "gt_endpoint_rel_norm": _quantiles(torch.cat(gt_endpoint_norms)),
        },
        "original_metrics": _mean_dicts(original_metrics),
        "scale_sweep": scale_rows,
        "best_by_ADE_avg": best_by_ade_avg,
        "best_by_ADE_min": best_by_ade_min,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_coerce_jsonable(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"output_json={output_path.as_posix()}")
    print(f"records={len(records)}")
    print(f"coordinate_consistency_abs_max={output['coordinate_consistency_abs_max']}")
    print("prediction_endpoint_rel_norm=", output["stats"]["prediction_endpoint_rel_norm"])
    print("gt_endpoint_rel_norm=", output["stats"]["gt_endpoint_rel_norm"])
    print("original_metrics=", output["original_metrics"])
    print("best_by_ADE_avg=", best_by_ade_avg)
    print("best_by_ADE_min=", best_by_ade_min)


if __name__ == "__main__":
    main()
