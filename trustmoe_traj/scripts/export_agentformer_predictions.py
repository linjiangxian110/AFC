"""Export AgentFormer prediction bundles for standard AFC evaluation.

This adapter runs the upstream AgentFormer/DLow inference code, keeps K sampled
future trajectories per scene, and stores the same compact record structure used
by the other external-baseline AFC adapters.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export AgentFormer K-sample predictions on ETH-UCY subsets.")
    parser.add_argument("--agentformer-root", type=str, required=True)
    parser.add_argument("--cfg-id", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _agentformer_import_context(agentformer_root: Path) -> Iterator[None]:
    root = agentformer_root.resolve()
    old_cwd = Path.cwd()
    inserted = False
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
        os.chdir(old_cwd)


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _import_agentformer_modules() -> Dict[str, Any]:
    from data.dataloader import data_generator
    from model.model_lib import model_dict
    from utils.config import Config
    from utils.utils import print_log

    return {
        "Config": Config,
        "data_generator": data_generator,
        "model_dict": model_dict,
        "print_log": print_log,
    }


def _safe_log_handle(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+", encoding="utf-8")


def _load_cfg_and_model(
    *,
    modules: Mapping[str, Any],
    cfg_id: str,
    epoch: Optional[int],
    device: torch.device,
) -> tuple[Any, int, torch.nn.Module, Path]:
    cfg = modules["Config"](cfg_id)
    if str(cfg.dataset) not in DATASETS:
        raise SystemExit(f"AgentFormer cfg {cfg_id!r} is not an ETH-UCY subset cfg: dataset={cfg.dataset!r}")
    resolved_epoch = cfg.get_last_epoch() if epoch is None else int(epoch)
    if resolved_epoch is None:
        raise SystemExit(
            f"No AgentFormer checkpoint found for cfg={cfg_id!r}. "
            f"Expected files under {Path(cfg.model_dir).resolve().as_posix()}/model_*.p"
        )
    checkpoint_path = Path(cfg.model_path % int(resolved_epoch)).resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing AgentFormer checkpoint: {checkpoint_path.as_posix()}")
    if str(cfg.get("model_id", "agentformer")) == "dlow":
        pred_cfg = modules["Config"](cfg.pred_cfg)
        pred_checkpoint = Path(pred_cfg.model_path % int(cfg.pred_epoch)).resolve()
        if not pred_checkpoint.exists():
            raise SystemExit(
                f"Missing AgentFormer predictor checkpoint required by DLow cfg={cfg_id!r}: "
                f"{pred_checkpoint.as_posix()}"
            )

    model_id = cfg.get("model_id", "agentformer")
    model = modules["model_dict"][model_id](cfg)
    model.set_device(device)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_dict"], strict=False)
    return cfg, int(resolved_epoch), model, checkpoint_path


def _record_from_agentformer_data(
    *,
    scene_index: int,
    data: Mapping[str, Any],
    prediction_scaled: torch.Tensor,
    traj_scale: float,
) -> Dict[str, Any]:
    past_abs = (
        torch.stack(list(data["pre_motion_3D"]), dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
        * float(traj_scale)
    )
    future_abs = (
        torch.stack(list(data["fut_motion_3D"]), dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
        * float(traj_scale)
    )
    pred_abs = prediction_scaled.detach().cpu().numpy().astype(np.float32, copy=False)
    if pred_abs.ndim != 4:
        raise ValueError(f"AgentFormer prediction must have shape [K,A,T,2], got {pred_abs.shape}")
    if int(pred_abs.shape[1]) != int(past_abs.shape[0]):
        raise ValueError(f"Agent count mismatch: pred={pred_abs.shape}, past={past_abs.shape}")

    last_obs = past_abs[:, -1:, :]
    pred_rel = (pred_abs - last_obs[None, :, :, :]).astype(np.float32, copy=False)
    features = build_moflow_eth_feature_arrays(past_abs, future_abs, rotate=False)
    agent_mask = np.ones((past_abs.shape[0],), dtype=np.int64)
    social = compute_past_social_risk_features(past_abs, agent_mask)

    return {
        "scene_index": int(scene_index),
        "seq": str(data.get("seq", "")),
        "frame": int(data.get("frame", -1)),
        "obs_abs": torch.from_numpy(past_abs.astype(np.float32, copy=False)),
        "future_abs": torch.from_numpy(future_abs.astype(np.float32, copy=False)),
        "prediction_abs": torch.from_numpy(pred_abs.astype(np.float32, copy=False)),
        "prediction_rel": torch.from_numpy(pred_rel),
        "past_traj_original_scale": torch.from_numpy(features["past_traj_original_scale"]),
        "past_social_risk_features": torch.from_numpy(social.astype(np.float32, copy=False)),
        "fut_traj_original_scale": torch.from_numpy(features["fut_traj_original_scale"]),
        "fut_traj_vel": torch.from_numpy(features["fut_traj_vel"]),
        "agent_mask": torch.from_numpy(agent_mask),
    }


@torch.no_grad()
def _predict_records(
    *,
    model: torch.nn.Module,
    generator: Any,
    cfg: Any,
    k: int,
    max_scenes: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = 0
    while not generator.is_epoch_end():
        data = generator()
        if data is None:
            continue
        if max_scenes is not None and seen >= int(max_scenes):
            break
        model.set_data(data)
        sample_motion, _data = model.inference(mode="infer", sample_num=int(k), need_weights=False)
        sample_motion = sample_motion.transpose(0, 1).contiguous() * float(cfg.traj_scale)
        records.append(
            _record_from_agentformer_data(
                scene_index=seen,
                data=data,
                prediction_scaled=sample_motion,
                traj_scale=float(cfg.traj_scale),
            )
        )
        seen += 1
        if seen % 50 == 0:
            print(f"[export_agentformer_predictions] processed_scenes={seen}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    if int(args.k) <= 0:
        raise SystemExit("--k must be positive")
    _set_seed(int(args.seed))
    agentformer_root = Path(args.agentformer_root).expanduser().resolve()
    if not agentformer_root.exists():
        raise SystemExit(f"Missing AgentFormer root: {agentformer_root.as_posix()}")

    device = _resolve_device(str(args.device))
    with _agentformer_import_context(agentformer_root):
        modules = _import_agentformer_modules()
        cfg, epoch, model, checkpoint_path = _load_cfg_and_model(
            modules=modules,
            cfg_id=str(args.cfg_id),
            epoch=args.epoch,
            device=device,
        )
        if str(cfg.dataset) != str(args.subset):
            raise SystemExit(f"cfg dataset={cfg.dataset!r} does not match --subset={args.subset!r}")

        log = _safe_log_handle(Path(cfg.log_dir) / "log_export_afc.txt")
        try:
            generator = modules["data_generator"](cfg, log, split=str(args.split), phase="testing")
            print(
                "[export_agentformer_predictions] "
                f"subset={args.subset} split={args.split} cfg={args.cfg_id} epoch={epoch} "
                f"k={args.k} device={device} checkpoint={checkpoint_path.as_posix()}"
            )
            records = _predict_records(
                model=model,
                generator=generator,
                cfg=cfg,
                k=int(args.k),
                max_scenes=args.max_scenes,
            )
        finally:
            log.close()

    if not records:
        raise SystemExit("No AgentFormer records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_agentformer_predictions",
            "baseline": "AgentFormer",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agentformer_root": agentformer_root.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "cfg_id": str(args.cfg_id),
            "epoch": int(epoch),
            "subset": str(args.subset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
            "num_scenes": int(len(records)),
            "num_valid_agents": int(valid_agents),
            "max_scenes": None if args.max_scenes is None else int(args.max_scenes),
        },
        "records": records,
    }
    output_path = Path(args.output_bundle).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
