"""ETH 数据集到 TrustMoE-Traj 统一 schema 的适配实现。"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..schema import DEFAULT_DATASET, ETH_SUBSETS, SceneMeta

try:  # pragma: no cover - torch 可能尚未安装
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore[override]
        """最小 Dataset 占位，避免强依赖 torch。"""

        pass


DEFAULT_OBS_LEN = 8
DEFAULT_PRED_LEN = 12
DEFAULT_SKIP = 1
DEFAULT_MIN_AGENTS = 1
DEFAULT_DELIM = "\t"
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "ETH"
DEFAULT_PROCESSED_DIRNAME = "processed"
ETH_MAIN_CACHE_SCHEMA_VERSION = "trustmoe_eth_main_cache_v1"


@dataclass(frozen=True)
class ETHAdapterConfig:
    """ETH adapter 的最小配置。"""

    data_root: Union[str, Path] = DEFAULT_DATA_ROOT
    subset: str = "eth"
    split: str = "train"
    obs_len: int = DEFAULT_OBS_LEN
    pred_len: int = DEFAULT_PRED_LEN
    skip: int = DEFAULT_SKIP
    min_agents: int = DEFAULT_MIN_AGENTS
    delim: Optional[str] = DEFAULT_DELIM
    dataset_name: str = DEFAULT_DATASET
    processed_dirname: str = DEFAULT_PROCESSED_DIRNAME
    prefer_cache: bool = False
    cache_path: Optional[Union[str, Path]] = None

    @property
    def seq_len(self) -> int:
        return self.obs_len + self.pred_len

    def resolved_data_root(self) -> Path:
        return Path(self.data_root).resolve()

    def resolved_cache_dir(self) -> Path:
        return resolve_eth_processed_dir(
            self.resolved_data_root(),
            processed_dirname=self.processed_dirname,
        )

    def resolved_cache_path(self) -> Path:
        if self.cache_path is not None:
            return Path(self.cache_path).resolve()
        return resolve_eth_cache_path(
            self.resolved_data_root(),
            self.subset,
            self.split,
            processed_dirname=self.processed_dirname,
        )


def _validate_subset(subset: str) -> None:
    if subset not in ETH_SUBSETS:
        raise ValueError(f"Unsupported ETH subset: {subset!r}. Expected one of {ETH_SUBSETS}")


def _validate_split(split: str) -> None:
    if split not in DEFAULT_SPLITS:
        raise ValueError(f"Unsupported ETH split: {split!r}. Expected one of {DEFAULT_SPLITS}")


def _normalize_delim(delim: Optional[str]) -> Optional[str]:
    if delim in (None, "", "whitespace"):
        return None
    if delim == "tab":
        return "\t"
    return delim


def _cache_meta_matches_config(cache_meta: Dict[str, Any], config: ETHAdapterConfig) -> Tuple[bool, List[str]]:
    expected_values = {
        "dataset": config.dataset_name,
        "subset": config.subset,
        "split": config.split,
        "obs_len": int(config.obs_len),
        "pred_len": int(config.pred_len),
        "seq_len": int(config.seq_len),
        "skip": int(config.skip),
        "min_agents": int(config.min_agents),
        "delim": _normalize_delim(config.delim),
    }
    mismatch_fields: List[str] = []
    for key, expected in expected_values.items():
        actual = cache_meta.get(key)
        if actual != expected:
            mismatch_fields.append(key)
    return len(mismatch_fields) == 0, mismatch_fields


def _read_line_to_values(line: str, delim: Optional[str]) -> List[float]:
    stripped = line.strip()
    if not stripped:
        return []

    if delim is None:
        parts = stripped.split()
    else:
        parts = stripped.split(delim)
        if len(parts) == 1:
            parts = stripped.split()
    return [float(item) for item in parts]


def read_eth_txt(path: Union[str, Path], delim: Optional[str] = DEFAULT_DELIM) -> np.ndarray:
    """读取单个 ETH/UCY 文本文件，预期每行格式为 `frame_id ped_id x y`。"""

    file_path = Path(path)
    normalized_delim = _normalize_delim(delim)
    rows: List[List[float]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = _read_line_to_values(line, normalized_delim)
            if not values:
                continue
            if len(values) < 4:
                raise ValueError(f"Invalid ETH row in {file_path}: {line!r}")
            rows.append(values[:4])

    if not rows:
        return np.empty((0, 4), dtype=np.float32)

    data = np.asarray(rows, dtype=np.float32)
    order = np.lexsort((data[:, 1], data[:, 0]))
    return data[order]


def discover_eth_split_files(
    data_root: Union[str, Path],
    subset: str,
    split: str,
) -> List[Path]:
    """发现某个 subset/split 下的所有 txt 文件。"""

    _validate_subset(subset)
    _validate_split(split)

    split_dir = Path(data_root).resolve() / subset / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ETH split directory not found: {split_dir}")
    return sorted(path for path in split_dir.glob("*.txt") if path.is_file())


def resolve_eth_processed_dir(
    data_root: Union[str, Path],
    processed_dirname: str = DEFAULT_PROCESSED_DIRNAME,
) -> Path:
    """返回 ETH 主缓存目录。"""

    return Path(data_root).resolve() / processed_dirname


def build_eth_cache_filename(subset: str, split: str) -> str:
    """生成默认缓存文件名。"""

    _validate_subset(subset)
    _validate_split(split)
    return f"{subset}_{split}.pkl"


def resolve_eth_cache_path(
    data_root: Union[str, Path],
    subset: str,
    split: str,
    *,
    processed_dirname: str = DEFAULT_PROCESSED_DIRNAME,
) -> Path:
    """返回某个 subset/split 的主缓存路径。"""

    return resolve_eth_processed_dir(data_root, processed_dirname) / build_eth_cache_filename(subset, split)


def _normalize_source_file(file_path: Path, data_root: Union[str, Path]) -> str:
    try:
        return file_path.resolve().relative_to(Path(data_root).resolve()).as_posix()
    except Exception:
        return file_path.as_posix()


def _scene_meta_to_dict(scene_meta: Any) -> Dict[str, Any]:
    if isinstance(scene_meta, SceneMeta):
        return scene_meta.to_dict()
    if isinstance(scene_meta, dict):
        return dict(scene_meta)
    raise TypeError(f"Unsupported scene_meta type for cache serialization: {type(scene_meta)!r}")


def _sample_to_cache_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    extras = sample.get("extras") or {}
    past_traj = np.asarray(sample["past_traj"], dtype=np.float32)
    future_traj = np.asarray(sample["future_traj"], dtype=np.float32)
    agent_mask = np.asarray(sample.get("agent_mask", np.ones((past_traj.shape[0],), dtype=np.int64)), dtype=np.int64)

    num_agents = extras.get("num_agents")
    if num_agents is None:
        num_agents = int(past_traj.shape[0])

    return {
        "past_traj": past_traj,
        "future_traj": future_traj,
        "agent_mask": agent_mask,
        "scene_meta": _scene_meta_to_dict(sample["scene_meta"]),
        "extras": {
            "agent_ids": np.asarray(extras.get("agent_ids", np.arange(past_traj.shape[0])), dtype=np.int64),
            "frame_ids": np.asarray(extras.get("frame_ids", []), dtype=np.int64),
            "num_agents": int(num_agents),
        },
    }


def _summarize_cache_stats(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "agent_count_min": 0,
            "agent_count_max": 0,
            "agent_count_mean": 0.0,
            "samples_per_source_file": {},
        }

    agent_counts = [int(sample["extras"]["num_agents"]) for sample in samples]
    source_counter = Counter(sample["scene_meta"].get("source_file", "") for sample in samples)
    return {
        "agent_count_min": int(min(agent_counts)),
        "agent_count_max": int(max(agent_counts)),
        "agent_count_mean": float(np.mean(agent_counts)),
        "samples_per_source_file": dict(sorted(source_counter.items())),
    }


def build_samples_from_eth_file(
    file_path: Union[str, Path],
    *,
    subset: str,
    split: str,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    skip: int = DEFAULT_SKIP,
    min_agents: int = DEFAULT_MIN_AGENTS,
    delim: Optional[str] = DEFAULT_DELIM,
    dataset_name: str = DEFAULT_DATASET,
) -> List[Dict[str, Any]]:
    """把单个 ETH 原始文件切成统一多 agent 样本。"""

    _validate_subset(subset)
    _validate_split(split)

    path = Path(file_path)
    seq_len = obs_len + pred_len
    data = read_eth_txt(path, delim=delim)
    if data.size == 0:
        return []

    frames = np.unique(data[:, 0]).tolist()
    if len(frames) < seq_len:
        return []

    frame_data = [data[data[:, 0] == frame] for frame in frames]
    source_file = _normalize_source_file(path, data_root)
    samples: List[Dict[str, Any]] = []

    for sample_index, frame_start_idx in enumerate(range(0, len(frames) - seq_len + 1, skip)):
        current_frames = np.asarray(frames[frame_start_idx : frame_start_idx + seq_len], dtype=np.float32)
        current_frame_blocks = frame_data[frame_start_idx : frame_start_idx + seq_len]
        current_sequence = np.concatenate(current_frame_blocks, axis=0)
        ped_ids = np.unique(current_sequence[:, 1])

        trajectories: List[np.ndarray] = []
        kept_agent_ids: List[int] = []

        for ped_id in ped_ids:
            ped_sequence = current_sequence[current_sequence[:, 1] == ped_id]
            ped_sequence = ped_sequence[np.argsort(ped_sequence[:, 0])]
            if ped_sequence.shape[0] != seq_len:
                continue
            if not np.array_equal(ped_sequence[:, 0], current_frames):
                continue

            trajectories.append(ped_sequence[:, 2:4].astype(np.float32))
            kept_agent_ids.append(int(ped_id))

        if len(trajectories) < min_agents:
            continue

        trajectory_array = np.stack(trajectories, axis=0)
        sample_id = f"{subset}_{split}_{path.stem}_{sample_index:05d}"
        scene_meta = SceneMeta(
            dataset=dataset_name,
            subset=subset,
            sample_id=sample_id,
            seq_id=path.stem,
            frame_id=int(current_frames[0]),
            split=split,
            source_file=source_file,
        )

        samples.append(
            {
                "past_traj": trajectory_array[:, :obs_len, :],
                "future_traj": trajectory_array[:, obs_len:, :],
                "agent_mask": np.ones((len(trajectories),), dtype=np.int64),
                "scene_meta": scene_meta,
                "extras": {
                    "agent_ids": np.asarray(kept_agent_ids, dtype=np.int64),
                    "frame_ids": current_frames.astype(np.int64),
                    "num_agents": len(trajectories),
                },
            }
        )

    return samples


def load_eth_split_samples(
    data_root: Union[str, Path],
    subset: Union[str, Sequence[str]],
    split: str,
    *,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    skip: int = DEFAULT_SKIP,
    min_agents: int = DEFAULT_MIN_AGENTS,
    delim: Optional[str] = DEFAULT_DELIM,
    dataset_name: str = DEFAULT_DATASET,
) -> List[Dict[str, Any]]:
    """加载一个或多个 subset 的指定 split 样本。"""

    subsets = [subset] if isinstance(subset, str) else list(subset)
    all_samples: List[Dict[str, Any]] = []

    for current_subset in subsets:
        files = discover_eth_split_files(data_root, current_subset, split)
        for file_path in files:
            all_samples.extend(
                build_samples_from_eth_file(
                    file_path,
                    subset=current_subset,
                    split=split,
                    data_root=data_root,
                    obs_len=obs_len,
                    pred_len=pred_len,
                    skip=skip,
                    min_agents=min_agents,
                    delim=delim,
                    dataset_name=dataset_name,
                )
            )
    return all_samples


def build_eth_cache_payload(
    data_root: Union[str, Path],
    subset: str,
    split: str,
    *,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    skip: int = DEFAULT_SKIP,
    min_agents: int = DEFAULT_MIN_AGENTS,
    delim: Optional[str] = DEFAULT_DELIM,
    dataset_name: str = DEFAULT_DATASET,
) -> Dict[str, Any]:
    """构建某个 subset/split 的主缓存 payload。"""

    resolved_root = Path(data_root).resolve()
    source_files = discover_eth_split_files(resolved_root, subset, split)
    raw_samples = load_eth_split_samples(
        resolved_root,
        subset,
        split,
        obs_len=obs_len,
        pred_len=pred_len,
        skip=skip,
        min_agents=min_agents,
        delim=delim,
        dataset_name=dataset_name,
    )
    samples = [_sample_to_cache_record(sample) for sample in raw_samples]
    cache_stats = _summarize_cache_stats(samples)

    cache_meta = {
        "schema_version": ETH_MAIN_CACHE_SCHEMA_VERSION,
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seq_len": int(obs_len + pred_len),
        "skip": int(skip),
        "min_agents": int(min_agents),
        "delim": _normalize_delim(delim) if delim not in ("tab",) else "\t",
        "num_samples": len(samples),
        "source_files": [_normalize_source_file(path, resolved_root) for path in source_files],
        "data_representation": "absolute_xy_from_source_txt",
        "sample_format": "per-scene variable-agent samples before batch padding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "trustmoe_traj.data.adapters.eth.build_eth_cache_payload",
        "sample_fields": ["past_traj", "future_traj", "agent_mask", "scene_meta", "extras"],
        "extras_fields": ["agent_ids", "frame_ids", "num_agents"],
    }
    return {
        "cache_meta": cache_meta,
        "cache_stats": cache_stats,
        "samples": samples,
    }


def save_eth_split_cache(payload: Dict[str, Any], cache_path: Union[str, Path]) -> Path:
    """把主缓存 payload 落盘为 pickle 文件。"""

    path = Path(cache_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_eth_split_cache(cache_path: Union[str, Path]) -> Dict[str, Any]:
    """读取单个 ETH 主缓存文件。"""

    path = Path(cache_path).resolve()
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid ETH cache payload type: {type(payload)!r}")
    for key in ("cache_meta", "samples"):
        if key not in payload:
            raise ValueError(f"Invalid ETH cache payload, missing key: {key}")
    return payload


def load_cached_eth_split_samples(cache_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """从主缓存读取样本列表。"""

    payload = load_eth_split_cache(cache_path)
    return list(payload["samples"])


def prepare_eth_split_cache(
    data_root: Union[str, Path],
    subset: str,
    split: str,
    *,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    skip: int = DEFAULT_SKIP,
    min_agents: int = DEFAULT_MIN_AGENTS,
    delim: Optional[str] = DEFAULT_DELIM,
    dataset_name: str = DEFAULT_DATASET,
    processed_dirname: str = DEFAULT_PROCESSED_DIRNAME,
    overwrite: bool = False,
) -> Path:
    """生成并保存某个 subset/split 的主缓存。"""

    cache_path = resolve_eth_cache_path(
        data_root,
        subset,
        split,
        processed_dirname=processed_dirname,
    )
    if cache_path.exists() and not overwrite:
        return cache_path

    payload = build_eth_cache_payload(
        data_root,
        subset,
        split,
        obs_len=obs_len,
        pred_len=pred_len,
        skip=skip,
        min_agents=min_agents,
        delim=delim,
        dataset_name=dataset_name,
    )
    return save_eth_split_cache(payload, cache_path)


def prepare_eth_all_caches(
    data_root: Union[str, Path],
    *,
    subsets: Sequence[str] = ETH_SUBSETS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    obs_len: int = DEFAULT_OBS_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
    skip: int = DEFAULT_SKIP,
    min_agents: int = DEFAULT_MIN_AGENTS,
    delim: Optional[str] = DEFAULT_DELIM,
    dataset_name: str = DEFAULT_DATASET,
    processed_dirname: str = DEFAULT_PROCESSED_DIRNAME,
    overwrite: bool = False,
) -> List[Path]:
    """批量生成多个 subset/split 的主缓存。"""

    cache_paths: List[Path] = []
    for subset in subsets:
        for split in splits:
            cache_paths.append(
                prepare_eth_split_cache(
                    data_root,
                    subset,
                    split,
                    obs_len=obs_len,
                    pred_len=pred_len,
                    skip=skip,
                    min_agents=min_agents,
                    delim=delim,
                    dataset_name=dataset_name,
                    processed_dirname=processed_dirname,
                    overwrite=overwrite,
                )
            )
    return cache_paths


class ETHTrajectoryDataset(Dataset):
    """ETH 多 agent 样本数据集。"""

    def __init__(self, config: Optional[ETHAdapterConfig] = None, **overrides: Any) -> None:
        if config is None:
            config = ETHAdapterConfig(**overrides)
        elif overrides:
            config = ETHAdapterConfig(**{**config.__dict__, **overrides})

        _validate_subset(config.subset)
        _validate_split(config.split)

        self.config = config
        self.data_root = config.resolved_data_root()
        self.cache_path = config.resolved_cache_path()
        self.cache_meta: Dict[str, Any] = {}
        self.cache_stats: Dict[str, Any] = {}
        self.loaded_from_cache = False
        self.cache_available = self.cache_path.exists()
        self.cache_compatible: Optional[bool] = None
        self.cache_mismatch_fields: List[str] = []

        if config.prefer_cache and self.cache_available:
            payload = load_eth_split_cache(self.cache_path)
            self.cache_meta = dict(payload.get("cache_meta", {}))
            self.cache_stats = dict(payload.get("cache_stats", {}))
            self.cache_compatible, self.cache_mismatch_fields = _cache_meta_matches_config(self.cache_meta, config)

            if self.cache_compatible:
                self.samples = list(payload["samples"])
                source_files = self.cache_meta.get("source_files", [])
                self.files = [self.data_root / Path(source_file) for source_file in source_files]
                self.loaded_from_cache = True
            else:
                self.files = discover_eth_split_files(self.data_root, config.subset, config.split)
                self.samples = load_eth_split_samples(
                    self.data_root,
                    config.subset,
                    config.split,
                    obs_len=config.obs_len,
                    pred_len=config.pred_len,
                    skip=config.skip,
                    min_agents=config.min_agents,
                    delim=config.delim,
                    dataset_name=config.dataset_name,
                )
        else:
            self.files = discover_eth_split_files(self.data_root, config.subset, config.split)
            self.samples = load_eth_split_samples(
                self.data_root,
                config.subset,
                config.split,
                obs_len=config.obs_len,
                pred_len=config.pred_len,
                skip=config.skip,
                min_agents=config.min_agents,
                delim=config.delim,
                dataset_name=config.dataset_name,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def summary(self) -> Dict[str, Any]:
        return {
            "data_root": self.data_root.as_posix(),
            "subset": self.config.subset,
            "split": self.config.split,
            "num_files": len(self.files),
            "num_samples": len(self.samples),
            "obs_len": self.config.obs_len,
            "pred_len": self.config.pred_len,
            "skip": self.config.skip,
            "min_agents": self.config.min_agents,
            "cache_path": self.cache_path.as_posix(),
            "cache_available": self.cache_available,
            "cache_compatible": self.cache_compatible,
            "cache_mismatch_fields": list(self.cache_mismatch_fields),
            "loaded_from_cache": self.loaded_from_cache,
        }


__all__ = [
    "DEFAULT_OBS_LEN",
    "DEFAULT_PRED_LEN",
    "DEFAULT_SKIP",
    "DEFAULT_MIN_AGENTS",
    "DEFAULT_DELIM",
    "DEFAULT_PROCESSED_DIRNAME",
    "ETH_MAIN_CACHE_SCHEMA_VERSION",
    "ETHAdapterConfig",
    "ETHTrajectoryDataset",
    "read_eth_txt",
    "discover_eth_split_files",
    "resolve_eth_processed_dir",
    "build_eth_cache_filename",
    "resolve_eth_cache_path",
    "build_samples_from_eth_file",
    "load_eth_split_samples",
    "build_eth_cache_payload",
    "save_eth_split_cache",
    "load_eth_split_cache",
    "load_cached_eth_split_samples",
    "prepare_eth_split_cache",
    "prepare_eth_all_caches",
]
