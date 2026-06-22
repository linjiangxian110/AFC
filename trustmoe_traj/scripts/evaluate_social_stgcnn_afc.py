"""Evaluate exported Social-STGCNN prediction bundles with AFC metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
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
    build_sdd_analogical_future_bank,
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
    parser = argparse.ArgumentParser(description="Evaluate Social-STGCNN exported bundle with AFC metrics.")
    parser.add_argument("--social-root", type=str, required=True)
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--branch-name", type=str, default="social_stgcnn20_pred")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--afc-train-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _load_social_trajectory_dataset(social_root: Path) -> Any:
    root = social_root.resolve()
    sys.path.insert(0, str(root))
    try:
        utils = importlib.import_module("utils")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return utils.TrajectoryDataset


def _tensor(value: Any, *, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        result = value.detach().cpu()
    else:
        result = torch.as_tensor(value)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


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


def _social_dataset_item_to_batch(item: Any) -> Dict[str, torch.Tensor]:
    obs_traj = _tensor(item[0]).numpy()  # [A,2,P]
    pred_traj = _tensor(item[1]).numpy()  # [A,2,T]
    obs_abs = np.asarray(obs_traj, dtype=np.float32).transpose(0, 2, 1)
    future_abs = np.asarray(pred_traj, dtype=np.float32).transpose(0, 2, 1)
    agent_mask = np.ones((obs_abs.shape[0],), dtype=np.int64)
    features = build_moflow_eth_feature_arrays(obs_abs, future_abs, rotate=False)
    social = compute_past_social_risk_features(obs_abs, agent_mask)
    return {
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]).unsqueeze(0),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)).unsqueeze(0),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]).unsqueeze(0),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]).unsqueeze(0),
        "agent_mask": torch.from_numpy(agent_mask.astype(np.bool_)).unsqueeze(0),
    }


def _build_social_afc_bank(
    *,
    social_root: Path,
    dataset: str,
    train_split: str,
    top_m: int,
    eps_values: List[float],
    max_train_scenes: Optional[int],
) -> AnalogicalFutureBank:
    trajectory_dataset_cls = _load_social_trajectory_dataset(social_root)
    args_path = social_root / "checkpoint" / f"social-stgcnn-{dataset}" / "args.pkl"
    if not args_path.exists():
        raise SystemExit(f"Missing Social-STGCNN args.pkl for AFC bank: {args_path}")

    import pickle

    with args_path.open("rb") as handle:
        model_args = pickle.load(handle)
    data_dir = social_root / "datasets" / str(dataset) / str(train_split)
    if not data_dir.exists():
        raise SystemExit(f"Missing Social-STGCNN AFC data split: {data_dir}")

    train_dataset = trajectory_dataset_cls(
        str(data_dir) + "/",
        obs_len=int(model_args.obs_seq_len),
        pred_len=int(model_args.pred_seq_len),
        skip=1,
        norm_lap_matr=True,
    )
    limit = len(train_dataset) if max_train_scenes is None else min(int(max_train_scenes), len(train_dataset))
    if limit <= 0:
        raise SystemExit(f"AFC train split is empty: dataset={dataset} split={train_split}")

    feature_chunks: List[torch.Tensor] = []
    future_chunks: List[torch.Tensor] = []
    for index in range(limit):
        batch = _social_dataset_item_to_batch(train_dataset[index])
        features = _agent_features_from_batch(batch)
        valid = batch["agent_mask"].bool()
        feature_chunks.append(features[valid])
        future_chunks.append(batch["fut_traj_original_scale"][valid])
        if (index + 1) % 500 == 0:
            print(f"[evaluate_social_stgcnn_afc] bank_scenes={index + 1}/{limit}")

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
    social_root = Path(args.social_root).expanduser().resolve()
    bundle_path = Path(args.bundle).expanduser().resolve()
    payload = torch.load(bundle_path, map_location="cpu")
    bundle_meta = dict(payload.get("meta", {}))
    dataset = str(args.dataset or bundle_meta.get("dataset") or "")
    split = str(args.split or bundle_meta.get("split") or "")
    if not dataset:
        raise SystemExit("--dataset is required when the bundle has no dataset meta")
    if not split:
        raise SystemExit("--split is required when the bundle has no split meta")

    records = list(payload.get("records", []))
    if args.max_scenes is not None:
        records = records[: int(args.max_scenes)]
    if not records:
        raise SystemExit(f"No records found in bundle: {bundle_path}")

    eps_values = split_float_list(str(args.afc_eps))
    if dataset == "sdd":
        if args.data_root is None:
            raise SystemExit("--data-root is required for dataset=sdd")
        afc_bank = build_sdd_analogical_future_bank(
            data_root=Path(args.data_root).expanduser().resolve(),
            train_split=str(args.afc_train_split),
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
    else:
        afc_bank = _build_social_afc_bank(
            social_root=social_root,
            dataset=dataset,
            train_split=str(args.afc_train_split),
            top_m=int(args.afc_top_m),
            eps_values=eps_values,
            max_train_scenes=args.afc_max_train_scenes,
        )

    branch = str(args.branch_name)
    accumulator = BranchAccumulator(branch, float(args.miss_threshold))
    aux_accumulator = AuxAccumulator()

    print(
        "[evaluate_social_stgcnn_afc] "
        f"dataset={dataset} split={split} records={len(records)} branch={branch} "
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
        # Inference timing is handled by the upstream Social-STGCNN test smoke.
        # Keep a zero placeholder so the shared accumulator can finalize.
        accumulator.add_chunk(summary.metrics, [0.0])
        valid_count = int(batch["agent_mask"].bool().sum().item())
        aux_accumulator.add(_spread_aux(prediction, batch["agent_mask"].bool()), weight=valid_count)
        for key, value in afc_bank.metrics_for_prediction(prediction, batch).items():
            aux_accumulator.add_metric(key, value, weight=valid_count)
        if (index + 1) % 100 == 0:
            print(f"[evaluate_social_stgcnn_afc] evaluated_records={index + 1}/{len(records)}")

    metrics: Dict[str, float] = {}
    metrics.update(accumulator.finalize())
    metrics.update(aux_accumulator.finalize(branch))

    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.evaluate_social_stgcnn_afc",
            "baseline": "Social-STGCNN",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "split": split,
            "branch": branch,
            "num_records": int(len(records)),
            "num_valid_agents": int(metrics.get(f"{branch}_num_valid_agents", 0.0)),
            "afc_bank_size": int(afc_bank.bank_size),
            "afc_top_m": int(afc_bank.top_m),
            "afc_eps": [float(item) for item in eps_values],
            "bundle": bundle_path.as_posix(),
            "social_root": social_root.as_posix(),
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
