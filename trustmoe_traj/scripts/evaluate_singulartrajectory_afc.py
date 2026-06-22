"""Evaluate SingularTrajectory prediction bundles with the standard ETH-UCY AFC bank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch

from trustmoe_traj.evaluation import evaluate_model_output
from trustmoe_traj.scripts.analogical_future_coverage import (
    build_eth_analogical_future_bank,
    split_float_list,
)
from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import AuxAccumulator
from trustmoe_traj.scripts.evaluate_mid_afc import (
    _prediction_from_record,
    _record_to_batch,
    _spread_aux,
)
from trustmoe_traj.scripts.run_eval import BranchAccumulator, DEFAULT_DATA_ROOT, _coerce_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SingularTrajectory exported bundle with AFC metrics.")
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--branch-name", type=str, default="singulartrajectory20_pred")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--afc-train-split", type=str, default="train")
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--afc-max-train-scenes", type=int, default=None)
    parser.add_argument("--afc-batch-scenes", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = Path(args.bundle).expanduser().resolve()
    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(bundle_path, map_location="cpu")
    bundle_meta = dict(payload.get("meta", {}))
    dataset = str(args.dataset or bundle_meta.get("dataset") or "")
    split = str(args.split or bundle_meta.get("split") or "")
    if not dataset:
        raise SystemExit("--dataset is required when bundle has no dataset meta")
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
        subset=dataset,
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
        "[evaluate_singulartrajectory_afc] "
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
        accumulator.add_chunk(summary.metrics, [0.0])
        valid_count = int(batch["agent_mask"].bool().sum().item())
        aux_accumulator.add(_spread_aux(prediction, batch["agent_mask"].bool()), weight=valid_count)
        for key, value in afc_bank.metrics_for_prediction(prediction, batch).items():
            aux_accumulator.add_metric(key, value, weight=valid_count)
        if (index + 1) % 100 == 0:
            print(f"[evaluate_singulartrajectory_afc] evaluated_records={index + 1}/{len(records)}")

    metrics: Dict[str, float] = {}
    metrics.update(accumulator.finalize())
    metrics.update(aux_accumulator.finalize(branch))
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.evaluate_singulartrajectory_afc",
            "baseline": "SingularTrajectory",
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
