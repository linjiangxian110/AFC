"""Export EigenTrajectory prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export EigenTrajectory K=20 prediction bundles.")
    parser.add_argument("--eigen-root", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["test"])
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _eigen_import_context(eigen_root: Path) -> Iterator[None]:
    root = eigen_root.resolve()
    old_cwd = Path.cwd()
    inserted: List[str] = []
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted.append(text)
    for module_name in ("baseline", "EigenTrajectory", "utils"):
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


def _build_trainer(*, eigen_root: Path, cfg: Path, tag: str, gpu_id: str, seed: int) -> Any:
    with _eigen_import_context(eigen_root):
        baseline_pkg = importlib.import_module("baseline")
        eigen_pkg = importlib.import_module("EigenTrajectory")
        utils_pkg = importlib.import_module("utils")
        trainer_module = importlib.import_module("utils.trainer")

        hyper_params = utils_pkg.get_exp_config(cfg.as_posix())
        baseline_name = str(hyper_params.baseline)
        baseline_module = getattr(baseline_pkg, baseline_name)
        predictor_model = baseline_module.TrajectoryPredictor
        hook_func = utils_pkg.DotDict(
            {
                "model_forward_pre_hook": baseline_module.model_forward_pre_hook,
                "model_forward": baseline_module.model_forward,
                "model_forward_post_hook": baseline_module.model_forward_post_hook,
            }
        )
        candidates = [
            name
            for name in trainer_module.__dict__.keys()
            if name.startswith("ET") and name.endswith("Trainer") and baseline_name.lower() in name.lower()
        ]
        if not candidates:
            raise SystemExit(f"No EigenTrajectory trainer found for baseline={baseline_name}")
        model_trainer_cls = getattr(trainer_module, sorted(candidates, key=len)[-1])
        args = SimpleNamespace(cfg=cfg.as_posix(), tag=str(tag), gpu_id=str(gpu_id), test=True, seed=int(seed), epochs=None)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        trainer = model_trainer_cls(
            base_model=predictor_model,
            model=eigen_pkg.EigenTrajectory,
            hook_func=hook_func,
            args=args,
            hyper_params=hyper_params,
        )
        trainer.load_model()
        trainer.model.eval()
        return trainer


def _additional_information(baseline: str, batch: Any, num_samples: int) -> Optional[Dict[str, Any]]:
    baseline = str(baseline).lower()
    if baseline in {"lbebm", "implicit"}:
        return {"num_samples": int(num_samples)}
    if baseline in {"pecnet"}:
        scene_mask = batch[-2].cuda(non_blocking=True)
        return {"scene_mask": scene_mask, "num_samples": int(num_samples)}
    if baseline in {"agentformer"}:
        scene_mask = batch[-2].cuda(non_blocking=True)
        seq_start_end = batch[-1].cuda(non_blocking=True)
        return {"scene_mask": scene_mask, "seq_start_end": seq_start_end, "num_samples": int(num_samples)}
    return None


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


@torch.no_grad()
def _predict_records(*, trainer: Any, dataset: str, split: str, max_scenes: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    hyper_params = trainer.hyper_params
    for scene_index, batch in enumerate(trainer.loader_test):
        if max_scenes is not None and scene_index >= int(max_scenes):
            break
        obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]
        addl_info = _additional_information(str(hyper_params.baseline), batch, int(hyper_params.num_samples))
        output = trainer.model(obs_traj, addl_info=addl_info)
        prediction = output["recon_traj"]
        if int(prediction.shape[0]) != int(hyper_params.num_samples):
            raise RuntimeError(f"Unexpected EigenTrajectory sample count: {tuple(prediction.shape)}")
        obs_np = obs_traj.detach().cpu().numpy().astype(np.float32, copy=False)
        future_np = pred_traj.detach().cpu().numpy().astype(np.float32, copy=False)
        prediction_np = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
        records.append(
            _record_from_scene(
                dataset=dataset,
                split=split,
                scene_index=scene_index,
                obs_abs=obs_np,
                future_abs=future_np,
                prediction_abs=prediction_np,
            )
        )
        if (scene_index + 1) % 100 == 0:
            print(f"[export_eigentrajectory_predictions] scenes={scene_index + 1}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    eigen_root = Path(args.eigen_root).expanduser().resolve()
    cfg = Path(args.cfg).expanduser().resolve()
    output_path = Path(args.output_bundle).expanduser().resolve()
    if not eigen_root.exists():
        raise SystemExit(f"Missing EigenTrajectory root: {eigen_root.as_posix()}")
    if not cfg.exists():
        raise SystemExit(f"Missing EigenTrajectory config: {cfg.as_posix()}")
    if str(args.split) != "test":
        raise SystemExit("Only split=test is supported for EigenTrajectory export")
    if not torch.cuda.is_available():
        raise SystemExit("EigenTrajectory export currently requires CUDA")

    trainer = _build_trainer(eigen_root=eigen_root, cfg=cfg, tag=str(args.tag), gpu_id=str(args.gpu_id), seed=int(args.seed))
    baseline = str(args.baseline or trainer.hyper_params.baseline)
    records = _predict_records(trainer=trainer, dataset=str(args.dataset), split=str(args.split), max_scenes=args.max_scenes)
    if not records:
        raise SystemExit("No EigenTrajectory records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_eigentrajectory_predictions",
            "baseline": f"EigenTrajectory-{baseline}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eigen_root": eigen_root.as_posix(),
            "cfg": cfg.as_posix(),
            "tag": str(args.tag),
            "dataset": str(args.dataset),
            "split": str(args.split),
            "k": int(args.k),
            "seed": int(args.seed),
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
