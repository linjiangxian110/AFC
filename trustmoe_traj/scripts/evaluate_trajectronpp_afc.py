"""Evaluate Trajectron++ prediction bundles with the standard ETH-UCY AFC bank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.scripts.analogical_future_coverage import (
    build_eth_analogical_future_bank,
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
from trustmoe_traj.scripts.run_eval import BranchAccumulator, DEFAULT_DATA_ROOT, _coerce_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Trajectron++ exported bundle with AFC metrics.")
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--branch-name", type=str, default="trajectronpp20_pred")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--afc-train-split", type=str, default="train")
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
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
    bundle_path = Path(args.bundle).expanduser().resolve()
    payload = torch.load(bundle_path, map_location="cpu")
    bundle_meta = dict(payload.get("meta", {}))
    subset = str(args.subset or bundle_meta.get("subset") or "")
    split = str(args.split or bundle_meta.get("split") or "")
    if not subset:
        raise SystemExit("--subset is required when bundle has no subset meta")
    if not split:
        raise SystemExit("--split is required when bundle has no split meta")
    records = list(payload.get("records", []))
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    if not records:
        raise SystemExit(f"No records found in bundle: {bundle_path}")

    data_root = Path(args.data_root).expanduser().resolve()
    eps_values = split_float_list(str(args.afc_eps))
    afc_bank = build_eth_analogical_future_bank(
        data_root=data_root,
        subset=subset,
        train_split=str(args.afc_train_split),
        sample_mode="per_scene",
        data_norm="original",
        rotate=False,
        rotate_time_frame=0,
        normalization_stats=None,
        min_agents=1,
        prefer_cache=False,
        max_train_scenes=args.afc_max_train_scenes,
        batch_scenes=int(args.afc_batch_scenes),
        top_m=int(args.afc_top_m),
        eps_values=eps_values,
    )

    branch = str(args.branch_name)
    accumulator = BranchAccumulator(branch, float(args.miss_threshold))
    aux_accumulator = AuxAccumulator()
    print(
        "[evaluate_trajectronpp_afc] "
        f"subset={subset} split={split} records={len(records)} branch={branch} "
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
        if (index + 1) % 50 == 0:
            print(f"[evaluate_trajectronpp_afc] evaluated_records={index + 1}/{len(records)}")

    metrics: Dict[str, float] = {}
    metrics.update(accumulator.finalize())
    metrics.update(aux_accumulator.finalize(branch))
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.evaluate_trajectronpp_afc",
            "baseline": "Trajectron++",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subset": subset,
            "split": split,
            "branch": branch,
            "num_records": int(len(records)),
            "num_valid_agents": int(metrics.get(f"{branch}_num_valid_agents", 0.0)),
            "afc_bank_size": int(afc_bank.bank_size),
            "afc_top_m": int(afc_bank.top_m),
            "afc_eps": [float(item) for item in eps_values],
            "bundle": bundle_path.as_posix(),
            "data_root": data_root.as_posix(),
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
