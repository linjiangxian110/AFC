"""Evaluate exported PECNet prediction bundles with AFC metrics."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)
from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.scripts.analogical_future_coverage import (
    AnalogicalFutureBank,
    _agent_features_from_batch,
    split_float_list,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import (
    AuxAccumulator,
    _cluster_count_entropy_values,
    _endpoint_pairwise,
    _endpoint_spread,
    _mean_float,
    _trajectory_pairwise,
    _trajectory_spread,
)
from trustmoe_traj.scripts.run_eval import BranchAccumulator, _coerce_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PECNet exported bundle with AFC metrics.")
    parser.add_argument("--pecnet-root", type=str, required=True)
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--dataset-label", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--branch-name", type=str, default="pecnet20_pred")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--afc-train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--afc-max-train-batches", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _tensor(value: Any, *, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _record_to_batch(record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    agent_mask = _tensor(record["agent_mask"], dtype=None).to(dtype=torch.bool)
    return {
        "past_traj_original_scale": _tensor(record["past_traj_original_scale"]).unsqueeze(0),
        "past_social_risk_features": _tensor(record["past_social_risk_features"]).unsqueeze(0),
        "fut_traj_original_scale": _tensor(record["fut_traj_original_scale"]).unsqueeze(0),
        "fut_traj_vel": _tensor(record["fut_traj_vel"]).unsqueeze(0),
        "agent_mask": agent_mask.unsqueeze(0),
    }


def _prediction_from_record(record: Mapping[str, Any]) -> torch.Tensor:
    return _tensor(record["prediction_rel"]).unsqueeze(0)


def _normalise_traj_for_features(traj: np.ndarray, *, data_scale: float) -> np.ndarray:
    item = np.asarray(traj, dtype=np.float32).copy()
    item -= item[:, :1, :]
    item *= float(data_scale)
    item /= float(data_scale)
    return item


def _pecnet_item_to_batch(traj: np.ndarray, *, data_scale: float) -> Dict[str, torch.Tensor]:
    item = _normalise_traj_for_features(traj, data_scale=float(data_scale))
    past_abs = item[:, :8, :]
    future_abs = item[:, 8:, :]
    agent_mask = np.ones((past_abs.shape[0],), dtype=np.int64)
    features = build_moflow_eth_feature_arrays(past_abs, future_abs, rotate=False)
    social = compute_past_social_risk_features(past_abs, agent_mask)
    return {
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]).unsqueeze(0),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)).unsqueeze(0),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]).unsqueeze(0),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]).unsqueeze(0),
        "agent_mask": torch.from_numpy(agent_mask.astype(np.bool_)).unsqueeze(0),
    }


def _build_pecnet_afc_bank(
    *,
    pecnet_root: Path,
    hyper_params: Mapping[str, Any],
    train_split: str,
    top_m: int,
    eps_values: List[float],
    max_train_batches: Optional[int],
) -> AnalogicalFutureBank:
    batch_size = int(hyper_params["train_b_size"] if train_split == "train" else hyper_params["test_b_size"])
    pool_path = (
        pecnet_root
        / "social_pool_data"
        / f"{train_split}_all_{batch_size}_{int(hyper_params['time_thresh'])}_{int(hyper_params['dist_thresh'])}.pickle"
    )
    if not pool_path.exists():
        raise SystemExit(f"Missing PECNet AFC pool data: {pool_path}")
    with pool_path.open("rb") as handle:
        trajectories, _masks = pickle.load(handle)
    limit = len(trajectories) if max_train_batches is None else min(
        int(max_train_batches),
        len(trajectories),
    )
    feature_chunks: List[torch.Tensor] = []
    future_chunks: List[torch.Tensor] = []
    for index in range(limit):
        raw_traj = np.asarray(trajectories[index], dtype=np.float32)
        if raw_traj.ndim != 3 or int(raw_traj.shape[-1]) < 4:
            raise ValueError(f"Expected PECNet raw trajectory [A,T,>=4], got {raw_traj.shape}")
        batch = _pecnet_item_to_batch(raw_traj[:, :, 2:], data_scale=float(hyper_params["data_scale"]))
        valid = batch["agent_mask"].bool()
        feature_chunks.append(_agent_features_from_batch(batch)[valid])
        future_chunks.append(batch["fut_traj_original_scale"][valid])
        if (index + 1) % 50 == 0:
            print(f"[evaluate_pecnet_afc] bank_batches={index + 1}/{limit}")
    if not feature_chunks:
        raise SystemExit("PECNet AFC bank is empty")
    return AnalogicalFutureBank.from_tensors(
        torch.cat(feature_chunks, dim=0),
        torch.cat(future_chunks, dim=0),
        top_m=int(top_m),
        eps_values=eps_values,
    )


def _spread_aux(prediction: torch.Tensor, agent_mask: torch.Tensor) -> Dict[str, float]:
    valid = agent_mask.to(dtype=torch.bool)
    if int(valid.sum().item()) <= 0:
        return {}
    result: Dict[str, float] = {}
    endpoint = _endpoint_spread(prediction)
    trajectory = _trajectory_spread(prediction)
    result["endpoint_spread"] = float(endpoint[valid].mean().detach().cpu())
    result["trajectory_spread"] = float(trajectory[valid].mean().detach().cpu())
    endpoint_pairwise = _endpoint_pairwise(prediction)
    trajectory_pairwise = _trajectory_pairwise(prediction)
    for eps, label in ((0.5, "eps05"), (1.0, "eps10")):
        endpoint_counts, endpoint_entropies = _cluster_count_entropy_values(endpoint_pairwise, mask=valid, eps=eps)
        trajectory_counts, trajectory_entropies = _cluster_count_entropy_values(trajectory_pairwise, mask=valid, eps=eps)
        endpoint_count = _mean_float(endpoint_counts)
        endpoint_entropy = _mean_float(endpoint_entropies)
        trajectory_count = _mean_float(trajectory_counts)
        trajectory_entropy = _mean_float(trajectory_entropies)
        if endpoint_count is not None:
            result[f"endpoint_cluster_count_{label}"] = float(endpoint_count)
        if endpoint_entropy is not None:
            result[f"endpoint_cluster_entropy_{label}"] = float(endpoint_entropy)
        if trajectory_count is not None:
            result[f"trajectory_cluster_count_{label}"] = float(trajectory_count)
        if trajectory_entropy is not None:
            result[f"trajectory_cluster_entropy_{label}"] = float(trajectory_entropy)
    return result


def main() -> None:
    args = build_parser().parse_args()
    pecnet_root = Path(args.pecnet_root).expanduser().resolve()
    bundle_path = Path(args.bundle).expanduser().resolve()
    payload = torch.load(bundle_path, map_location="cpu")
    bundle_meta = dict(payload.get("meta", {}))
    hyper_params = dict(payload.get("hyper_params", {}))
    dataset_label = str(args.dataset_label or bundle_meta.get("dataset_label") or "all")
    split = str(args.split or bundle_meta.get("split") or "test")
    records = list(payload.get("records", []))
    if args.max_batches is not None:
        records = records[: int(args.max_batches)]
    if not records:
        raise SystemExit(f"No records found in bundle: {bundle_path}")

    eps_values = split_float_list(str(args.afc_eps))
    afc_bank = _build_pecnet_afc_bank(
        pecnet_root=pecnet_root,
        hyper_params=hyper_params,
        train_split=str(args.afc_train_split),
        top_m=int(args.afc_top_m),
        eps_values=eps_values,
        max_train_batches=args.afc_max_train_batches,
    )

    branch = str(args.branch_name)
    accumulator = BranchAccumulator(branch, float(args.miss_threshold))
    aux_accumulator = AuxAccumulator()
    print(
        "[evaluate_pecnet_afc] "
        f"dataset_label={dataset_label} split={split} records={len(records)} branch={branch} "
        f"afc_bank={afc_bank.bank_size} top_m={afc_bank.top_m} eps={eps_values}"
    )
    for index, record in enumerate(records):
        batch = _record_to_batch(record)
        prediction = _prediction_from_record(record)
        summary = evaluate_model_output(
            {branch: prediction},
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=(branch,),
        )
        accumulator.add_chunk(summary.metrics, [0.0])
        valid_count = int(batch["agent_mask"].bool().sum().item())
        aux_accumulator.add(_spread_aux(prediction, batch["agent_mask"].bool()), weight=valid_count)
        for key, value in afc_bank.metrics_for_prediction(prediction, batch).items():
            aux_accumulator.add_metric(key, value, weight=valid_count)
        if (index + 1) % 20 == 0:
            print(f"[evaluate_pecnet_afc] evaluated_records={index + 1}/{len(records)}")

    metrics: Dict[str, float] = {}
    metrics.update(accumulator.finalize())
    metrics.update(aux_accumulator.finalize(branch))
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.evaluate_pecnet_afc",
            "baseline": "PECNet",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_label": dataset_label,
            "split": split,
            "branch": branch,
            "num_records": int(len(records)),
            "num_valid_agents": int(metrics.get(f"{branch}_num_valid_agents", 0.0)),
            "afc_bank_size": int(afc_bank.bank_size),
            "afc_top_m": int(afc_bank.top_m),
            "afc_eps": [float(item) for item in eps_values],
            "bundle": bundle_path.as_posix(),
            "pecnet_root": pecnet_root.as_posix(),
        },
        "args": _coerce_jsonable(vars(args)),
        "bundle_meta": _coerce_jsonable(bundle_meta),
        "branches": [branch],
        "metrics": _coerce_jsonable(metrics),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"output_json={output_path.as_posix()}")


if __name__ == "__main__":
    main()
