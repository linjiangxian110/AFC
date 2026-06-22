"""ETH 主缓存生成脚本。

用法示例：
    python -m trustmoe_traj.scripts.prepare_eth_cache --subset all --split all
    python -m trustmoe_traj.scripts.prepare_eth_cache --subset eth --split train --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from trustmoe_traj.data import (
    DEFAULT_DATASET,
    DEFAULT_DELIM,
    DEFAULT_MIN_AGENTS,
    DEFAULT_OBS_LEN,
    DEFAULT_PRED_LEN,
    DEFAULT_PROCESSED_DIRNAME,
    DEFAULT_SKIP,
    ETH_SUBSETS,
    prepare_eth_split_cache,
    resolve_eth_cache_path,
)

DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "ETH"


def _normalize_subset_arg(subset: str) -> List[str]:
    return list(ETH_SUBSETS) if subset == "all" else [subset]


def _normalize_split_arg(split: str) -> List[str]:
    return list(DEFAULT_SPLITS) if split == "all" else [split]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 ETH 五个子集生成 TrustMoE-Traj 主缓存 pickle。")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="ETH 原始数据根目录")
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", *ETH_SUBSETS],
        help="处理哪个 subset；all 表示五个子集全部处理",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", *DEFAULT_SPLITS],
        help="处理哪个 split；all 表示 train/val/test 全部处理",
    )
    parser.add_argument("--obs-len", type=int, default=DEFAULT_OBS_LEN, help="观测长度")
    parser.add_argument("--pred-len", type=int, default=DEFAULT_PRED_LEN, help="预测长度")
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP, help="滑窗步长")
    parser.add_argument("--min-agents", type=int, default=DEFAULT_MIN_AGENTS, help="最少 agent 数")
    parser.add_argument("--delim", type=str, default=DEFAULT_DELIM, help="原始 txt 分隔符，默认 tab")
    parser.add_argument(
        "--processed-dirname",
        type=str,
        default=DEFAULT_PROCESSED_DIRNAME,
        help="缓存输出目录名，默认 processed",
    )
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET, help="写入 cache_meta 的数据集名")
    parser.add_argument("--overwrite", action="store_true", help="若缓存已存在，是否覆盖重建")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    subsets = _normalize_subset_arg(args.subset)
    splits = _normalize_split_arg(args.split)

    print(f"[prepare_eth_cache] data_root={data_root.as_posix()}")
    print(f"[prepare_eth_cache] subsets={subsets}")
    print(f"[prepare_eth_cache] splits={splits}")

    generated_paths: List[Path] = []
    skipped_paths: List[Path] = []

    for subset in subsets:
        for split in splits:
            cache_path = resolve_eth_cache_path(
                data_root,
                subset,
                split,
                processed_dirname=args.processed_dirname,
            )
            if cache_path.exists() and not args.overwrite:
                skipped_paths.append(cache_path)
                print(f"[skip] {subset}/{split} -> {cache_path.as_posix()} (already exists)")
                continue

            saved_path = prepare_eth_split_cache(
                data_root,
                subset,
                split,
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                skip=args.skip,
                min_agents=args.min_agents,
                delim=args.delim,
                dataset_name=args.dataset_name,
                processed_dirname=args.processed_dirname,
                overwrite=args.overwrite,
            )
            generated_paths.append(saved_path)
            print(f"[ok] {subset}/{split} -> {saved_path.as_posix()}")

    print("[prepare_eth_cache] done")
    print(f"[prepare_eth_cache] generated={len(generated_paths)} skipped={len(skipped_paths)}")


if __name__ == "__main__":
    main()