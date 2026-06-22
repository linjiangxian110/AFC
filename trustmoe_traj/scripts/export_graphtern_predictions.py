"""Export GraphTERN prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GraphTERN K=20 prediction bundles.")
    parser.add_argument("--graphtern-root", type=str, required=True)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pruning", type=int, default=4)
    parser.add_argument("--clustering", action="store_true", default=True)
    parser.add_argument("--no-clustering", dest="clustering", action="store_false")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _graphtern_import_context(graphtern_root: Path) -> Iterator[None]:
    root = graphtern_root.resolve()
    old_cwd = Path.cwd()
    inserted: List[str] = []
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted.append(text)
    for module_name in list(sys.modules):
        if module_name == "graphtern" or module_name.startswith("graphtern."):
            sys.modules.pop(module_name, None)
        if module_name == "utils" or module_name.startswith("utils."):
            sys.modules.pop(module_name, None)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for item in inserted:
            try:
                sys.path.remove(item)
            except ValueError:
                pass


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_paths(graphtern_root: Path, checkpoint_root: Optional[str], tag: str, dataset: str) -> Dict[str, Path]:
    root = Path(checkpoint_root).expanduser() if checkpoint_root else graphtern_root / "checkpoint"
    exp_dir = root / str(tag)
    return {
        "experiment_dir": exp_dir,
        "args": exp_dir / "args.pkl",
        "model": exp_dir / f"{dataset}_best.pth",
    }


def _state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("GraphTERN checkpoint is not a state dict")
    if any(str(key).startswith("module.") for key in payload.keys()):
        return {str(key).removeprefix("module."): value for key, value in payload.items()}
    return payload


def _record_from_scene(
    *,
    dataset: str,
    split: str,
    scene_index: int,
    obs_abs: np.ndarray,
    future_abs: np.ndarray,
    prediction_abs: np.ndarray,
) -> Dict[str, Any]:
    agent_mask = np.ones((obs_abs.shape[0],), dtype=np.int64)
    prediction_rel = prediction_abs - obs_abs[None, :, -1:, :]
    features = build_moflow_eth_feature_arrays(obs_abs, future_abs, rotate=False)
    social = compute_past_social_risk_features(obs_abs, agent_mask)
    return {
        "dataset": str(dataset),
        "split": str(split),
        "scene_index": int(scene_index),
        "obs_abs": torch.from_numpy(obs_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(prediction_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(prediction_rel.astype(np.float32, copy=False)),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
    }


def _build_model(*, args: Any, k: int, device: torch.device) -> torch.nn.Module:
    from graphtern.model import graph_tern

    model = graph_tern(
        n_epgcn=int(getattr(args, "n_epgcn", 1)),
        n_epcnn=int(getattr(args, "n_epcnn", 6)),
        n_trgcn=int(getattr(args, "n_trgcn", 1)),
        n_trcnn=int(getattr(args, "n_trcnn", 3)),
        seq_len=int(getattr(args, "obs_seq_len", 8)),
        pred_seq_len=int(getattr(args, "pred_seq_len", 12)),
        n_ways=int(getattr(args, "n_ways", 3)),
        n_smpl=int(k),
    )
    return model.to(device)


@torch.no_grad()
def _predict_records(
    *,
    graphtern_root: Path,
    args: Any,
    model: torch.nn.Module,
    dataset: str,
    split: str,
    k: int,
    pruning: int,
    clustering: bool,
    num_workers: int,
    max_scenes: Optional[int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    from utils.dataloader import TrajectoryDataset

    dataset_dir = graphtern_root / "datasets" / dataset / split
    data = TrajectoryDataset(
        dataset_dir.as_posix() + "/",
        obs_len=int(getattr(args, "obs_seq_len", 8)),
        pred_len=int(getattr(args, "pred_seq_len", 12)),
        skip=1,
    )
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=int(num_workers), pin_memory=True)
    records: List[Dict[str, Any]] = []
    model.eval()
    model.n_smpl = int(k)

    for scene_index, batch in enumerate(loader):
        if max_scenes is not None and scene_index >= int(max_scenes):
            break
        s_obs, _s_trgt = [tensor.to(device=device, non_blocking=True) for tensor in batch[-2:]]
        _, _, v_refi, _ = model(s_obs, pruning=int(pruning), clustering=bool(clustering))

        obs_abs = s_obs[:, 0].squeeze(0).permute(1, 0, 2).detach().cpu().numpy()
        future_abs = _s_trgt[:, 0].squeeze(0).permute(1, 0, 2).detach().cpu().numpy()
        prediction_abs = v_refi.permute(0, 2, 1, 3).detach().cpu().numpy()
        records.append(
            _record_from_scene(
                dataset=dataset,
                split=split,
                scene_index=scene_index,
                obs_abs=obs_abs,
                future_abs=future_abs,
                prediction_abs=prediction_abs,
            )
        )
        if (scene_index + 1) % 100 == 0:
            print(f"[export_graphtern_predictions] scenes={scene_index + 1}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    graphtern_root = Path(args.graphtern_root).expanduser().resolve()
    output_path = Path(args.output_bundle).expanduser().resolve()
    if not graphtern_root.exists():
        raise SystemExit(f"Missing GraphTERN root: {graphtern_root.as_posix()}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GraphTERN export currently requires CUDA")
    device = torch.device(str(args.device))
    paths = _checkpoint_paths(graphtern_root, args.checkpoint_root, str(args.tag), str(args.dataset))
    if not paths["args"].exists():
        raise SystemExit(f"Missing GraphTERN args.pkl: {paths['args'].as_posix()}")
    if not paths["model"].exists():
        raise SystemExit(f"Missing GraphTERN checkpoint: {paths['model'].as_posix()}")

    with _graphtern_import_context(graphtern_root):
        train_args = _load_pickle(paths["args"])
        model = _build_model(args=train_args, k=int(args.k), device=device)
        model.load_state_dict(_state_dict(_torch_load(paths["model"], device)), strict=False)
        records = _predict_records(
            graphtern_root=graphtern_root,
            args=train_args,
            model=model,
            dataset=str(args.dataset),
            split=str(args.split),
            k=int(args.k),
            pruning=int(args.pruning),
            clustering=bool(args.clustering),
            num_workers=int(args.num_workers),
            max_scenes=args.max_scenes,
            device=device,
        )

    if not records:
        raise SystemExit("No GraphTERN records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_graphtern_predictions",
            "baseline": "GraphTERN",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "graphtern_root": graphtern_root.as_posix(),
            "checkpoint_root": paths["experiment_dir"].parent.as_posix(),
            "tag": str(args.tag),
            "checkpoint": paths["model"].as_posix(),
            "dataset": str(args.dataset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "pruning": int(args.pruning),
            "clustering": bool(args.clustering),
            "num_records": int(len(records)),
            "num_valid_agents": int(valid_agents),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
