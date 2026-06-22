"""Export MID prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

try:
    import dill
except Exception:
    import pickle as dill

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd")


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MID K-sample predictions.")
    parser.add_argument("--mid-root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--subset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=5, help="Number of DDIM/DDPM steps; converted to MID stride.")
    parser.add_argument(
        "--sdd-scale-factor",
        type=float,
        default=1.0,
        help=(
            "Scale MID SDD coordinates only when intentionally leaving the MoFlow "
            "SDD AFC coordinate convention. The default is 1.0 because MoFlow "
            "AFC builds its SDD bank from sdd_*.pkl in the same scaled coordinate "
            "space used by the MID SDD preprocessing output."
        ),
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _mid_import_context(mid_root: Path) -> Iterator[None]:
    root = mid_root.resolve()
    old_cwd = Path.cwd()
    inserted: List[str] = []
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted.append(text)
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


def _load_config(
    *,
    config_path: Path,
    model_dir: Path,
    processed_dir: Path,
    subset: str,
    checkpoint_epoch: int,
    seed: int,
) -> AttrDict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AttrDict(payload)
    config.config = config_path.as_posix()
    config.exp_name = model_dir.name
    config.dataset = str(subset)
    config.data_dir = processed_dir.as_posix()
    config.eval_mode = True
    config.eval_at = int(checkpoint_epoch)
    config.seed = int(seed)
    config.preprocess_workers = int(config.get("preprocess_workers", 0))
    return config


def _state_position() -> Dict[str, List[str]]:
    return {"position": ["x", "y"]}


def _record_from_group(
    *,
    subset: str,
    scene_index: int,
    scene_name: str,
    timestep: int,
    node_items: Sequence[Tuple[Any, np.ndarray]],
    max_hl: int,
    ph: int,
    sdd_scale_factor: float,
) -> Dict[str, Any]:
    position_state = _state_position()
    past_items: List[np.ndarray] = []
    future_items: List[np.ndarray] = []
    pred_items: List[np.ndarray] = []
    node_ids: List[str] = []
    for node, pred in node_items:
        past = node.get(np.asarray([int(timestep) - int(max_hl), int(timestep)]), position_state)
        future = node.get(np.asarray([int(timestep) + 1, int(timestep) + int(ph)]), position_state)
        if np.isnan(past).any() or np.isnan(future).any():
            continue
        if past.shape[0] != int(max_hl) + 1 or future.shape[0] != int(ph):
            continue
        if pred.shape[0] <= 0 or pred.shape[1] != int(ph):
            continue
        past_items.append(np.asarray(past[:, :2], dtype=np.float32))
        future_items.append(np.asarray(future[:, :2], dtype=np.float32))
        pred_items.append(np.asarray(pred[:, :, :2], dtype=np.float32))
        node_ids.append(str(node.id))

    if not past_items:
        raise ValueError("No valid MID nodes in prediction group")
    past_abs = np.stack(past_items, axis=0)
    future_abs = np.stack(future_items, axis=0)
    pred_abs = np.stack(pred_items, axis=1)  # [K,A,T,2]
    coordinate_scale_factor = float(sdd_scale_factor) if str(subset) == "sdd" else 1.0
    if coordinate_scale_factor != 1.0:
        past_abs = past_abs * coordinate_scale_factor
        future_abs = future_abs * coordinate_scale_factor
        pred_abs = pred_abs * coordinate_scale_factor
    last_obs = past_abs[:, -1:, :]
    pred_rel = (pred_abs - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    features = build_moflow_eth_feature_arrays(past_abs, future_abs, rotate=False)
    agent_mask = np.ones((past_abs.shape[0],), dtype=np.int64)
    social = compute_past_social_risk_features(past_abs, agent_mask)
    return {
        "scene_index": int(scene_index),
        "scene_name": str(scene_name),
        "timestep": int(timestep),
        "node_ids": node_ids,
        "obs_abs": torch.from_numpy(past_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(pred_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(pred_rel),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
        "coordinate_scale_factor": float(coordinate_scale_factor),
    }


@torch.no_grad()
def _predict_records(
    *,
    agent: Any,
    k: int,
    sampling: str,
    sampling_steps: int,
    sdd_scale_factor: float,
    max_scenes: Optional[int],
    max_records: Optional[int],
) -> List[Dict[str, Any]]:
    from dataset import get_timesteps_data

    records: List[Dict[str, Any]] = []
    node_type = "PEDESTRIAN"
    ph = int(agent.hyperparams["prediction_horizon"])
    max_hl = int(agent.hyperparams["maximum_history_length"])
    stride = max(1, int(100 // max(1, int(sampling_steps))))
    for scene_index, scene in enumerate(agent.eval_scenes):
        if max_scenes is not None and scene_index >= int(max_scenes):
            break
        for t in range(0, int(scene.timesteps), 10):
            timesteps = np.arange(t, t + 10)
            batch = get_timesteps_data(
                env=agent.eval_env,
                scene=scene,
                t=timesteps,
                node_type=node_type,
                state=agent.hyperparams["state"],
                pred_state=agent.hyperparams["pred_state"],
                edge_types=agent.eval_env.get_edge_types(),
                min_ht=max_hl,
                max_ht=max_hl,
                min_ft=ph,
                max_ft=ph,
                hyperparams=agent.hyperparams,
            )
            if batch is None:
                continue
            test_batch, nodes, timesteps_o = batch
            prediction = agent.model.generate(
                test_batch,
                node_type,
                num_points=ph,
                sample=int(k),
                bestof=True,
                sampling=str(sampling),
                step=stride,
            )
            groups: Dict[int, List[Tuple[Any, np.ndarray]]] = {}
            for index, ts in enumerate(timesteps_o):
                groups.setdefault(int(ts), []).append((nodes[index], np.asarray(prediction[:, index], dtype=np.float32)))
            for ts, items in sorted(groups.items()):
                try:
                    record = _record_from_group(
                        subset=str(agent.config.dataset),
                        scene_index=scene_index,
                        scene_name=str(scene.name),
                        timestep=int(ts),
                        node_items=items,
                        max_hl=max_hl,
                        ph=ph,
                        sdd_scale_factor=float(sdd_scale_factor),
                    )
                except ValueError:
                    continue
                records.append(record)
                if max_records is not None and len(records) >= int(max_records):
                    return records
        print(f"[export_mid_predictions] scene={scene_index + 1}/{len(agent.eval_scenes)} records={len(records)}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    mid_root = Path(args.mid_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve() if args.processed_dir else mid_root / "processed_data_noise"
    if not mid_root.exists():
        raise SystemExit(f"Missing MID root: {mid_root.as_posix()}")
    if not config_path.exists():
        raise SystemExit(f"Missing MID config: {config_path.as_posix()}")
    if not model_dir.exists():
        raise SystemExit(f"Missing MID model dir: {model_dir.as_posix()}")
    if not (processed_dir / f"{args.subset}_{args.split}.pkl").exists():
        raise SystemExit(f"Missing MID processed data: {(processed_dir / f'{args.subset}_{args.split}.pkl').as_posix()}")

    with _mid_import_context(mid_root):
        from mid import MID

        config = _load_config(
            config_path=config_path,
            model_dir=model_dir,
            processed_dir=processed_dir,
            subset=str(args.subset),
            checkpoint_epoch=int(args.checkpoint_epoch),
            seed=int(args.seed),
        )
        agent = MID(config)
        agent.model.eval()
        records = _predict_records(
            agent=agent,
            k=int(args.k),
            sampling=str(args.sampling),
            sampling_steps=int(args.sampling_steps),
            sdd_scale_factor=float(args.sdd_scale_factor),
            max_scenes=args.max_scenes,
            max_records=args.max_records,
        )

    if not records:
        raise SystemExit("MID export produced no valid records")
    bundle = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_mid_predictions",
            "baseline": "MID",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mid_root": mid_root.as_posix(),
            "config": config_path.as_posix(),
            "model_dir": model_dir.as_posix(),
            "checkpoint_epoch": int(args.checkpoint_epoch),
            "processed_dir": processed_dir.as_posix(),
            "subset": str(args.subset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "sampling": str(args.sampling),
            "sampling_steps": int(args.sampling_steps),
            "coordinate_contract": "MoFlow SDD relative future convention: prediction_rel and fut_traj_original_scale are relative to the last observed position.",
            "sdd_scale_factor": float(args.sdd_scale_factor),
            "num_records": int(len(records)),
        },
        "records": records,
    }
    output_path = Path(args.output_bundle).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(f"output_bundle={output_path.as_posix()}")
    print(f"records={len(records)}")


if __name__ == "__main__":
    main()
