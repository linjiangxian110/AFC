"""SDD pickle adapter for AFC protocol experiments."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..schema import SceneMeta

try:  # pragma: no cover
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore[override]
        pass


DEFAULT_SDD_OBS_LEN = 8
DEFAULT_SDD_PRED_LEN = 12
DEFAULT_SDD_SPLITS = ("train", "test")
DEFAULT_SDD_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "MoFlow" / "data" / "sdd"


@dataclass(frozen=True)
class SDDAdapterConfig:
    """Configuration for loading MoFlow-style SDD pickle files."""

    data_root: Union[str, Path] = DEFAULT_SDD_DATA_ROOT
    split: str = "train"
    obs_len: int = DEFAULT_SDD_OBS_LEN
    pred_len: int = DEFAULT_SDD_PRED_LEN
    dataset_name: str = "SDD"
    subset: str = "sdd"
    max_samples: Optional[int] = None

    def resolved_data_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()

    def resolved_pickle_path(self) -> Path:
        return resolve_sdd_pickle_path(self.resolved_data_root(), self.split)


def _validate_split(split: str) -> None:
    if split not in DEFAULT_SDD_SPLITS:
        raise ValueError(f"Unsupported SDD split: {split!r}. Expected one of {DEFAULT_SDD_SPLITS}")


def resolve_sdd_pickle_path(data_root: Union[str, Path], split: str) -> Path:
    _validate_split(split)
    return Path(data_root).expanduser().resolve() / "original" / f"sdd_{split}.pkl"


def load_sdd_pickle(path: Union[str, Path]) -> Sequence[Any]:
    file_path = Path(path).expanduser().resolve()
    with file_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not hasattr(payload, "__len__") or not hasattr(payload, "__getitem__"):
        raise ValueError(f"Invalid SDD pickle payload type: {type(payload)!r}")
    return payload


def _scene_to_past_future(scene: Any, *, obs_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(scene, dict):
        past = scene.get("past_traj", scene.get("past", scene.get("obs_traj")))
        future = scene.get("future_traj", scene.get("future", scene.get("fut_traj")))
    else:
        past = scene[0] if len(scene) > 0 else None
        future = scene[1] if len(scene) > 1 else None
    if past is None or future is None:
        raise ValueError("SDD scene does not contain past/future trajectory arrays")
    past_array = np.asarray(past, dtype=np.float32)
    future_array = np.asarray(future, dtype=np.float32)
    if past_array.ndim == 3 and int(past_array.shape[0]) == 1:
        past_array = past_array[0]
    if future_array.ndim == 3 and int(future_array.shape[0]) == 1:
        future_array = future_array[0]
    if past_array.ndim != 2 or int(past_array.shape[-1]) < 2:
        raise ValueError(f"SDD past trajectory must have shape [T,>=2], got {past_array.shape}")
    if future_array.ndim != 2 or int(future_array.shape[-1]) < 2:
        raise ValueError(f"SDD future trajectory must have shape [T,>=2], got {future_array.shape}")
    past_xy = past_array[: int(obs_len), :2].astype(np.float32, copy=False)
    future_xy = future_array[: int(pred_len), :2].astype(np.float32, copy=False)
    if int(past_xy.shape[0]) != int(obs_len) or int(future_xy.shape[0]) != int(pred_len):
        raise ValueError(
            f"SDD scene has invalid horizon: past={past_xy.shape[0]} future={future_xy.shape[0]} "
            f"expected {obs_len}/{pred_len}"
        )
    return past_xy, future_xy


def _scene_extra_meta(scene: Any) -> Dict[str, Any]:
    if not isinstance(scene, dict):
        return {}
    extras: Dict[str, Any] = {}
    for key, value in scene.items():
        if key in {"past_traj", "past", "obs_traj", "future_traj", "future", "fut_traj"}:
            continue
        try:
            if np.isscalar(value) or isinstance(value, (str, int, float, bool)):
                extras[key] = value
        except Exception:
            continue
    return extras


def build_sdd_samples(
    data_root: Union[str, Path],
    split: str,
    *,
    obs_len: int = DEFAULT_SDD_OBS_LEN,
    pred_len: int = DEFAULT_SDD_PRED_LEN,
    dataset_name: str = "SDD",
    subset: str = "sdd",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load SDD pickle scenes as TrustMoE standard single-agent samples."""

    pkl_path = resolve_sdd_pickle_path(data_root, split)
    scenes = list(load_sdd_pickle(pkl_path))
    if max_samples is not None:
        scenes = scenes[: int(max_samples)]
    samples: List[Dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        past_xy, future_xy = _scene_to_past_future(scene, obs_len=obs_len, pred_len=pred_len)
        sample_id = f"sdd_{split}_{index:06d}"
        scene_meta = SceneMeta(
            dataset=dataset_name,
            subset=subset,
            sample_id=sample_id,
            seq_id="sdd",
            frame_id=index,
            split=split,
            source_file=pkl_path.name,
            extras=_scene_extra_meta(scene),
        )
        frame_ids = np.arange(obs_len + pred_len, dtype=np.int64) + int(index) * int(obs_len + pred_len)
        samples.append(
            {
                "past_traj": past_xy[None, :, :],
                "future_traj": future_xy[None, :, :],
                "agent_mask": np.ones((1,), dtype=np.int64),
                "scene_meta": scene_meta,
                "extras": {
                    "agent_ids": np.asarray([0], dtype=np.int64),
                    "frame_ids": frame_ids,
                    "num_agents": 1,
                    "sdd_scene_index": int(index),
                },
            }
        )
    return samples


class SDDTrajectoryDataset(Dataset):
    """MoFlow SDD pickle dataset exposed through the TrustMoE standard schema."""

    def __init__(self, config: Optional[SDDAdapterConfig] = None, **overrides: Any) -> None:
        if config is None:
            config = SDDAdapterConfig(**overrides)
        elif overrides:
            config = SDDAdapterConfig(**{**config.__dict__, **overrides})
        _validate_split(config.split)
        self.config = config
        self.data_root = config.resolved_data_root()
        self.pickle_path = config.resolved_pickle_path()
        self.samples = build_sdd_samples(
            self.data_root,
            config.split,
            obs_len=int(config.obs_len),
            pred_len=int(config.pred_len),
            dataset_name=str(config.dataset_name),
            subset=str(config.subset),
            max_samples=config.max_samples,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def summary(self) -> Dict[str, Any]:
        agent_counts = [int(sample["extras"]["num_agents"]) for sample in self.samples]
        source_counts = Counter(str(sample["scene_meta"].source_file) for sample in self.samples)
        return {
            "data_root": self.data_root.as_posix(),
            "pickle_path": self.pickle_path.as_posix(),
            "split": self.config.split,
            "num_samples": len(self.samples),
            "obs_len": int(self.config.obs_len),
            "pred_len": int(self.config.pred_len),
            "agent_count_min": int(min(agent_counts)) if agent_counts else 0,
            "agent_count_max": int(max(agent_counts)) if agent_counts else 0,
            "source_files": dict(source_counts),
        }


__all__ = [
    "DEFAULT_SDD_DATA_ROOT",
    "DEFAULT_SDD_OBS_LEN",
    "DEFAULT_SDD_PRED_LEN",
    "DEFAULT_SDD_SPLITS",
    "SDDAdapterConfig",
    "SDDTrajectoryDataset",
    "build_sdd_samples",
    "load_sdd_pickle",
    "resolve_sdd_pickle_path",
]
