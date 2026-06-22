"""Prepare MoFlow-style SDD pickle data for ETH-UCY-style external baselines.

Social-STGCNN, GraphTERN, and EigenTrajectory all reuse the common text format

    frame_id<TAB>ped_id<TAB>x<TAB>y

under ``datasets/<name>/{train,val,test}``.  The MoFlow SDD pickle used in this
project stores single-agent 8+12 trajectory samples, so each exported text file
contains one continuous 20-frame single-agent sequence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from trustmoe_traj.data.adapters.sdd import build_sdd_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SDD pkl files to external-baseline text datasets.")
    parser.add_argument("--sdd-data-root", type=str, required=True)
    parser.add_argument("--output-dataset-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="sdd")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", type=str, default=None)
    return parser


def _split_train_val(num_items: int, val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    if num_items <= 1:
        return list(range(num_items)), []
    rng = np.random.default_rng(int(seed))
    indices = np.arange(num_items)
    rng.shuffle(indices)
    val_count = max(1, int(round(float(val_fraction) * float(num_items))))
    val_count = min(val_count, num_items - 1)
    val = sorted(int(item) for item in indices[:val_count])
    train = sorted(int(item) for item in indices[val_count:])
    return train, val


def _write_records(samples: Sequence[Dict[str, Any]], indices: Iterable[int], output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for out_index, sample_index in enumerate(indices):
        sample = samples[int(sample_index)]
        past = np.asarray(sample["past_traj"], dtype=np.float32)[0, :, :2]
        future = np.asarray(sample["future_traj"], dtype=np.float32)[0, :, :2]
        trajectory = np.concatenate([past, future], axis=0)
        path = output_dir / f"{prefix}_{out_index:06d}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for frame_id, xy in enumerate(trajectory):
                handle.write(f"{frame_id}\t0\t{float(xy[0]):.6f}\t{float(xy[1]):.6f}\n")
        count += 1
    return count


def _clear_split_dirs(dataset_dir: Path, split_names: Sequence[str]) -> None:
    for split in split_names:
        split_dir = dataset_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 <= float(args.val_fraction) < 1.0):
        raise SystemExit("--val-fraction must be in [0, 1)")

    output_root = Path(args.output_dataset_root).expanduser().resolve()
    dataset_dir = output_root / str(args.dataset_name)
    if dataset_dir.exists() and not args.force:
        raise SystemExit(f"Output dataset already exists, pass --force to overwrite: {dataset_dir}")
    _clear_split_dirs(dataset_dir, ("train", "val", "test"))

    train_samples = build_sdd_samples(
        args.sdd_data_root,
        "train",
        max_samples=args.max_train_records,
    )
    test_samples = build_sdd_samples(
        args.sdd_data_root,
        "test",
        max_samples=args.max_test_records,
    )
    train_indices, val_indices = _split_train_val(len(train_samples), float(args.val_fraction), int(args.seed))
    train_count = _write_records(train_samples, train_indices, dataset_dir / "train", "sdd_train")
    val_count = _write_records(train_samples, val_indices, dataset_dir / "val", "sdd_val")
    test_count = _write_records(test_samples, range(len(test_samples)), dataset_dir / "test", "sdd_test")

    summary = {
        "script": "trustmoe_traj.scripts.prepare_sdd_external_text_dataset",
        "sdd_data_root": Path(args.sdd_data_root).expanduser().resolve().as_posix(),
        "output_dataset_root": output_root.as_posix(),
        "dataset_dir": dataset_dir.as_posix(),
        "dataset_name": str(args.dataset_name),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "num_train_source": int(len(train_samples)),
        "num_test_source": int(len(test_samples)),
        "num_train_files": int(train_count),
        "num_val_files": int(val_count),
        "num_test_files": int(test_count),
    }
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else dataset_dir / "sdd_prepare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"dataset_dir={dataset_dir.as_posix()}")
    print(f"summary_json={summary_path.as_posix()}")
    print(f"train_files={train_count} val_files={val_count} test_files={test_count}")


if __name__ == "__main__":
    main()
