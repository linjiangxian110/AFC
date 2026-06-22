"""P0 checks for SDD AFC protocol adaptation.

This script does not train a model.  It verifies that SDD data and MoFlow SDD
handles are present, then runs a tiny AFC smoke test on deterministic branches
when ``sdd_train.pkl`` and ``sdd_test.pkl`` are available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from trustmoe_traj.data.adapters.sdd import SDDAdapterConfig, SDDTrajectoryDataset, resolve_sdd_pickle_path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDD_ROOT = DEFAULT_REPO_ROOT / "MoFlow" / "data" / "sdd"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPO_ROOT / "trustmoe_traj" / "analysis" / "sdd_p0"
DEFAULT_MOFLOW_FILES = (
    "MoFlow/data/dataloader_sdd.py",
    "MoFlow/fm_sdd.py",
    "MoFlow/eval_sdd.py",
    "MoFlow/cfg/sdd/cor_fm.yml",
    "MoFlow/cfg/sdd/imle.yml",
)
DEFAULT_CHECKPOINT_ROOTS = (
    DEFAULT_REPO_ROOT / "MoFlow" / "results_sdd" / "cor_fm",
    Path("/mnt/data/lck/code/moflow/MoFlow/results_sdd/cor_fm"),
    Path("/mnt/data/lck/code/TrustMoE-Traj/MoFlow/results_sdd/cor_fm"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check and smoke-test SDD AFC P0 infrastructure.")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_SDD_ROOT))
    parser.add_argument("--run-id", type=str, default="sdd_afc_p0_check")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--max-train-scenes", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--batch-scenes", type=int, default=64)
    parser.add_argument("--afc-batch-scenes", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--afc-top-m", type=int, default=20)
    parser.add_argument("--afc-eps", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--require-smoke", action="store_true", help="Exit with an error if the AFC smoke cannot run.")
    return parser


def _path_status(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": path.as_posix(),
        "exists": bool(exists),
        "is_file": bool(path.is_file()) if exists else False,
        "is_dir": bool(path.is_dir()) if exists else False,
        "size_bytes": int(path.stat().st_size) if exists and path.is_file() else None,
    }


def _find_sdd_checkpoints() -> List[str]:
    matches: List[str] = []
    for root in DEFAULT_CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*sdd*rot_6*")):
            if path.is_dir():
                matches.append(path.as_posix())
    return matches


def _iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if getattr(tensor, "ndim", 1) == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(item) for item in value]
    return value


def _constant_velocity_prediction(batch: Mapping[str, Any], *, keep_k: int) -> Any:
    import torch

    past = torch.as_tensor(batch["past_traj_original_scale"], dtype=torch.float32)
    future = torch.as_tensor(batch["fut_traj_original_scale"], dtype=torch.float32)
    past_rel = past[..., 2:4]
    if int(past_rel.shape[-2]) >= 2:
        velocity = past_rel[..., -1, :] - past_rel[..., -2, :]
    else:
        velocity = torch.zeros_like(past_rel[..., -1, :])
    steps = torch.arange(1, int(future.shape[-2]) + 1, dtype=past.dtype)
    prediction = velocity[:, :, None, :] * steps[None, None, :, None]
    return prediction[:, None, ...].expand(
        int(prediction.shape[0]),
        int(keep_k),
        int(prediction.shape[1]),
        int(prediction.shape[2]),
        int(prediction.shape[3]),
    ).contiguous()


def _gt_repeat_prediction(batch: Mapping[str, Any], *, keep_k: int) -> Any:
    import torch

    future = torch.as_tensor(batch["fut_traj_original_scale"], dtype=torch.float32)
    return future[:, None, ...].expand(
        int(future.shape[0]),
        int(keep_k),
        int(future.shape[1]),
        int(future.shape[2]),
        int(future.shape[3]),
    ).contiguous()


def _run_smoke(args: argparse.Namespace, data_root: Path) -> Dict[str, Any]:
    import torch

    from trustmoe_traj.data.transforms import build_moflow_eth_batch
    from trustmoe_traj.evaluation import evaluate_model_output
    from trustmoe_traj.scripts.analogical_future_coverage import (
        build_sdd_analogical_future_bank,
        split_float_list,
    )
    from trustmoe_traj.scripts.diagnose_v38_candidate_distribution import AuxAccumulator
    from trustmoe_traj.scripts.run_eval import BranchAccumulator

    eps_values = split_float_list(str(args.afc_eps))
    train_dataset = SDDTrajectoryDataset(SDDAdapterConfig(data_root=data_root, split="train", max_samples=args.max_train_scenes))
    test_dataset = SDDTrajectoryDataset(SDDAdapterConfig(data_root=data_root, split="test", max_samples=args.max_records))
    if len(train_dataset) <= 0 or len(test_dataset) <= 0:
        raise RuntimeError("SDD train/test dataset is empty")

    afc_bank = build_sdd_analogical_future_bank(
        data_root=data_root,
        train_split="train",
        sample_mode="per_scene",
        data_norm="original",
        rotate=False,
        max_train_scenes=int(args.max_train_scenes),
        batch_scenes=int(args.afc_batch_scenes),
        top_m=int(args.afc_top_m),
        eps_values=eps_values,
    )
    branches = ("cv_linear20_pred", "gt_repeat20_pred")
    branch_accumulators = {
        branch: BranchAccumulator(branch, float(args.miss_threshold))
        for branch in branches
    }
    aux_accumulators = {branch: AuxAccumulator() for branch in branches}

    samples = [test_dataset[index] for index in range(len(test_dataset))]
    for chunk in _iter_chunks(samples, int(args.batch_scenes)):
        batch = build_moflow_eth_batch(
            chunk,
            data_norm="original",
            sample_mode="per_scene",
            rotate=False,
            fixed_num_agents=1,
            as_torch=True,
        )
        predictions = {
            "cv_linear20_pred": _constant_velocity_prediction(batch, keep_k=int(args.k)),
            "gt_repeat20_pred": _gt_repeat_prediction(batch, keep_k=int(args.k)),
        }
        summary = evaluate_model_output(
            predictions,
            batch,
            miss_threshold=float(args.miss_threshold),
            prediction_fields=branches,
        )
        valid_count = int(torch.as_tensor(batch["agent_mask"]).bool().sum().item())
        for branch in branches:
            branch_accumulators[branch].add_chunk(summary.metrics, [0.0])
            for key, value in afc_bank.metrics_for_prediction(predictions[branch], batch).items():
                aux_accumulators[branch].add_metric(key, value, weight=valid_count)

    metrics: Dict[str, float] = {}
    for branch in branches:
        metrics.update(branch_accumulators[branch].finalize())
        metrics.update(aux_accumulators[branch].finalize(branch))
    return {
        "status": "ok",
        "train_summary": train_dataset.summary(),
        "test_summary": test_dataset.summary(),
        "branches": list(branches),
        "metrics": metrics,
        "afc_bank": {
            "bank_size": int(afc_bank.bank_size),
            "feature_dim": int(afc_bank.feature_dim),
            "top_m": int(afc_bank.top_m),
            "eps": [float(item) for item in eps_values],
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    train_pkl = resolve_sdd_pickle_path(data_root, "train")
    test_pkl = resolve_sdd_pickle_path(data_root, "test")
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else DEFAULT_OUTPUT_ROOT / str(args.run_id) / "sdd_afc_p0_check.json"
    )

    checks = {
        "data_root": _path_status(data_root),
        "sdd_train_pkl": _path_status(train_pkl),
        "sdd_test_pkl": _path_status(test_pkl),
        "moflow_files": {
            rel_path: _path_status(DEFAULT_REPO_ROOT / rel_path)
            for rel_path in DEFAULT_MOFLOW_FILES
        },
        "slow_checkpoint_candidates": _find_sdd_checkpoints(),
    }

    smoke: Dict[str, Any]
    if train_pkl.exists() and test_pkl.exists():
        try:
            smoke = _run_smoke(args, data_root)
        except Exception as exc:
            smoke = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if bool(args.require_smoke):
                raise
    else:
        smoke = {
            "status": "skipped_missing_data",
            "reason": "sdd_train.pkl or sdd_test.pkl is missing",
        }
        if bool(args.require_smoke):
            raise SystemExit("SDD smoke required but sdd_train.pkl or sdd_test.pkl is missing")

    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.check_sdd_afc_p0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(args.run_id),
            "data_root": data_root.as_posix(),
        },
        "args": _coerce_jsonable(vars(args)),
        "checks": _coerce_jsonable(checks),
        "smoke": _coerce_jsonable(smoke),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"output_json={output_path.as_posix()}")
    print(f"data_train_exists={checks['sdd_train_pkl']['exists']}")
    print(f"data_test_exists={checks['sdd_test_pkl']['exists']}")
    print(f"checkpoint_candidates={len(checks['slow_checkpoint_candidates'])}")
    print(f"smoke_status={smoke.get('status')}")


if __name__ == "__main__":
    main()
