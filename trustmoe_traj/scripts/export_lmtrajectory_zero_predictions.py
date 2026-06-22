"""Export LMTrajectory zero-shot prediction dumps for AFC evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")
DEFAULT_MODEL = "gpt-3.5-turbo-0301"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LMTrajectory zero-shot K=20 prediction bundles.")
    parser.add_argument("--lm-root", type=str, required=True)
    parser.add_argument("--dump-json", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["test"])
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


def _dump_path(lm_root: Path, dataset: str, model_name: str, dump_json: Optional[str]) -> Path:
    if dump_json:
        return Path(dump_json).expanduser().resolve()
    return (lm_root / "zero-shot" / "output_dump" / model_name / f"{dataset}_chatgpt_api_dump.json").resolve()


def _record_from_scene(
    *,
    dataset: str,
    split: str,
    scene_index: int,
    scene_id: str,
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
        "source_scene_id": str(scene_id),
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


def _extract_records(payload: Dict[str, Any], *, dataset: str, split: str, k: int, max_scenes: Optional[int]) -> List[Dict[str, Any]]:
    if str(payload.get("dataset")) != dataset:
        raise SystemExit(f"Dump dataset mismatch: expected={dataset} actual={payload.get('dataset')}")
    pred_len = int(payload.get("pred_len", 12))
    records: List[Dict[str, Any]] = []
    scenes = payload.get("data", {})
    if not isinstance(scenes, dict):
        raise SystemExit("Unexpected LMTrajectory dump format: data must be a dict")

    for scene_index, (scene_id, scene) in enumerate(scenes.items()):
        if max_scenes is not None and len(records) >= int(max_scenes):
            break
        obs_abs = np.asarray(scene.get("obs_traj"), dtype=np.float32)
        future_abs = np.asarray(scene.get("pred_traj"), dtype=np.float32)
        llm_processed = np.asarray(scene.get("llm_processed"), dtype=np.float32)
        if obs_abs.ndim != 3 or obs_abs.shape[-1] != 2:
            raise RuntimeError(f"Invalid obs_traj shape for scene={scene_id}: {obs_abs.shape}")
        if future_abs.ndim != 3 or future_abs.shape[-1] != 2:
            raise RuntimeError(f"Invalid pred_traj shape for scene={scene_id}: {future_abs.shape}")
        if llm_processed.ndim != 4 or llm_processed.shape[-2:] != (pred_len, 2):
            raise RuntimeError(f"Invalid llm_processed shape for scene={scene_id}: {llm_processed.shape}")
        if llm_processed.shape[0] != obs_abs.shape[0]:
            raise RuntimeError(f"Agent count mismatch for scene={scene_id}: obs={obs_abs.shape} pred={llm_processed.shape}")
        if llm_processed.shape[1] < int(k):
            raise RuntimeError(f"Not enough LMTrajectory samples for scene={scene_id}: {llm_processed.shape[1]} < {k}")
        prediction_abs = np.transpose(llm_processed[:, : int(k), :, :], (1, 0, 2, 3))
        records.append(
            _record_from_scene(
                dataset=dataset,
                split=split,
                scene_index=scene_index,
                scene_id=scene_id,
                obs_abs=obs_abs,
                future_abs=future_abs,
                prediction_abs=prediction_abs,
            )
        )
        if len(records) % 100 == 0:
            print(f"[export_lmtrajectory_zero_predictions] scenes={len(records)}")
    return records


def main() -> None:
    args = build_parser().parse_args()
    lm_root = Path(args.lm_root).expanduser().resolve()
    dump_path = _dump_path(lm_root, str(args.dataset), str(args.model_name), args.dump_json)
    output_path = Path(args.output_bundle).expanduser().resolve()
    if not lm_root.exists():
        raise SystemExit(f"Missing LMTrajectory root: {lm_root.as_posix()}")
    if not dump_path.exists():
        raise SystemExit(
            f"Missing LMTrajectory zero-shot dump: {dump_path.as_posix()}\n"
            "Hint: download and extract LMTraj-ZERO_output_trajectory.zip into the LMTrajectory root."
        )
    if str(args.split) != "test":
        raise SystemExit("Only split=test is supported for LMTrajectory zero-shot export")

    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    records = _extract_records(payload, dataset=str(args.dataset), split=str(args.split), k=int(args.k), max_scenes=args.max_scenes)
    if not records:
        raise SystemExit("No LMTrajectory records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    output = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_lmtrajectory_zero_predictions",
            "baseline": f"LMTrajectory-ZERO-{args.model_name}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lm_root": lm_root.as_posix(),
            "dump_json": dump_path.as_posix(),
            "model_name": str(args.model_name),
            "dataset": str(args.dataset),
            "split": str(args.split),
            "k": int(args.k),
            "num_records": int(len(records)),
            "num_valid_agents": int(valid_agents),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"output_bundle={output_path.as_posix()}")


if __name__ == "__main__":
    main()
