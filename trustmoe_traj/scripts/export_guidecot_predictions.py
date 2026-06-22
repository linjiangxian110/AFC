"""Export GUIDE-CoT prediction bundles for AFC evaluation."""

from __future__ import annotations

import argparse
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from trustmoe_traj.data.transforms import (
    build_moflow_eth_feature_arrays,
    compute_past_social_risk_features,
)


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GUIDE-CoT K=20 prediction bundles.")
    parser.add_argument("--guide-root", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="test", choices=["test"])
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--preprocessing-num-workers", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--output-bundle", type=str, required=True)
    return parser


@contextmanager
def _guide_llm_context(guide_root: Path) -> Iterator[None]:
    llm_root = guide_root.resolve() / "llm_module"
    old_cwd = Path.cwd()
    inserted: List[str] = []
    for path in (llm_root,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(text)
    for module_name in ("model", "utils"):
        sys.modules.pop(module_name, None)
    os.chdir(llm_root)
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


def _scene_records(
    *,
    dataset: str,
    split: str,
    obs_traj: np.ndarray,
    pred_traj: np.ndarray,
    all_preds: np.ndarray,
    seq_start_end: Sequence[Sequence[int]],
    scene_id: Sequence[str],
    max_scenes: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    spans = list(seq_start_end)
    if max_scenes is not None:
        spans = spans[: int(max_scenes)]
    for scene_index, (start_raw, end_raw) in enumerate(spans):
        start = int(start_raw)
        end = int(end_raw)
        if end <= start:
            continue
        prediction_abs = np.transpose(all_preds[start:end], (1, 0, 2, 3))
        records.append(
            _record_from_scene(
                dataset=dataset,
                split=split,
                scene_index=scene_index,
                scene_id=str(scene_id[start]),
                obs_abs=obs_traj[start:end],
                future_abs=pred_traj[start:end],
                prediction_abs=prediction_abs,
            )
        )
    return records


def _select_rows_for_scenes(
    *,
    max_scenes: Optional[int],
    seq_start_end: Sequence[Sequence[int]],
    arrays: Dict[str, np.ndarray],
) -> Optional[int]:
    if max_scenes is None:
        return None
    keep_scenes = max(int(max_scenes), 0)
    if keep_scenes <= 0:
        return 0
    spans = list(seq_start_end)[:keep_scenes]
    if not spans:
        return 0
    row_limit = int(spans[-1][1])
    for key, value in list(arrays.items()):
        arrays[key] = value[:row_limit]
    return row_limit


@torch.no_grad()
def _export_with_official_components(
    *,
    guide_root: Path,
    cfg_path: Path,
    dataset: str,
    split: str,
    checkpoint_name: str,
    k: int,
    seed: int,
    max_scenes: Optional[int],
    preprocessing_num_workers: Optional[int],
    inference_batch_size: Optional[int],
) -> List[Dict[str, Any]]:
    with _guide_llm_context(guide_root):
        from accelerate import Accelerator
        from datasets import load_dataset
        from model.nltoolkit import init_nltk
        from torch.utils.data import DataLoader
        from transformers import (
            AutoConfig,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            set_seed,
        )
        from utils.config import get_exp_config
        from utils.converter import batch_text2traj
        from utils.dataloader import get_dataloader
        from utils.homography import generate_homography, image2world
        from utils.postprocessor import postprocess_trajectory_simple

        cfg = get_exp_config(str(cfg_path))
        cfg.dataset_name = dataset
        cfg.checkpoint_name = checkpoint_name
        cfg.num_samples = int(k)
        cfg.best_of_n = int(k)
        if preprocessing_num_workers is not None:
            cfg.preprocessing_num_workers = max(int(preprocessing_num_workers), 1)
        if inference_batch_size is not None:
            cfg.per_device_inference_batch_size = int(inference_batch_size)
        if cfg.seed is None:
            cfg.seed = int(seed)
        set_seed(int(seed))
        init_nltk()

        goals_path = Path("..") / "goal_module" / "output" / dataset / "VisSem" / f"{dataset}-test-goals-20.npy"
        if not goals_path.exists():
            raise SystemExit(
                f"Missing GUIDE-CoT goal predictions: {goals_path.as_posix()}\n"
                "Hint: run goal_module test first to generate VisSem goals."
            )
        goals = np.load(goals_path).transpose(1, 0, 2).astype(np.float32, copy=False)
        if goals.shape[1] < int(k):
            raise SystemExit(f"GUIDE-CoT goals only contain {goals.shape[1]} samples, requested k={k}")
        goals = goals[:, : int(k), :]
        print(f"Goals shape: {goals.shape}")

        checkpoint_path = Path(cfg.checkpoint_path) / checkpoint_name
        if not checkpoint_path.exists():
            raise SystemExit(f"Missing GUIDE-CoT LLM checkpoint: {checkpoint_path.as_posix()}")
        accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)

        dataloader = get_dataloader(os.path.join(cfg.dataset_path, dataset), split, cfg.obs_len, cfg.pred_len, batch_size=1e8)
        arrays = {
            "obs_traj": dataloader.dataset.obs_traj.numpy().astype(np.float32, copy=False),
            "pred_traj": dataloader.dataset.pred_traj.numpy().astype(np.float32, copy=False),
            "non_linear_ped": dataloader.dataset.non_linear_ped.numpy(),
        }
        homography = dataloader.dataset.homography
        scene_id = dataloader.dataset.scene_id
        seq_start_end = dataloader.dataset.seq_start_end
        row_limit = _select_rows_for_scenes(max_scenes=max_scenes, seq_start_end=seq_start_end, arrays=arrays)
        active_seq_start_end = list(seq_start_end)
        obs_traj = arrays["obs_traj"]
        pred_traj = arrays["pred_traj"]
        if row_limit is not None:
            active_seq_start_end = active_seq_start_end[: int(max_scenes or 0)]
            goals = goals[:row_limit]
            scene_id = scene_id[:row_limit]

        batch_size_per_gpu = obs_traj.shape[0] // accelerator.state.num_processes + 1
        if batch_size_per_gpu < cfg.per_device_inference_batch_size:
            print(
                "per_device_inference_batch_size is automatically reduced "
                f"from {cfg.per_device_inference_batch_size} to {batch_size_per_gpu}."
            )
            cfg.per_device_inference_batch_size = batch_size_per_gpu

        for key, value in homography.items():
            homography[key] = value.copy() @ generate_homography(scale=0.25)

        suffix = f"{dataset}-{split}-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-{cfg.valid_dataset_type}.json"
        preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")
        data_file = os.path.join(preprocessed_dataset_path, suffix)
        if not os.path.exists(data_file):
            raise SystemExit(f"Missing GUIDE-CoT preprocessed test file: {data_file}")
        raw_datasets = load_dataset(data_file.split(".")[-1], data_files={split: data_file}, cache_dir=cfg.cache_dir)
        if row_limit is not None:
            raw_datasets[split] = raw_datasets[split].select(range(row_limit))

        model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=False, cache_dir=cfg.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=False,
            cache_dir=cfg.cache_dir,
            use_fast=not cfg.use_slow_tokenizer,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            config=model_config,
            trust_remote_code=False,
            cache_dir=cfg.cache_dir,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if accelerator.is_local_main_process:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {trainable}")

        column_names = raw_datasets[split].column_names
        history_column = cfg.history_column
        future_column = cfg.future_column
        if history_column not in column_names or future_column not in column_names:
            raise SystemExit(f"Unexpected GUIDE-CoT columns: {column_names}")

        num_goals = goals.shape[1]
        samples_per_goal = max(int(cfg.num_samples) // int(num_goals), 1)
        all_obs = np.array(raw_datasets[split]["obs_traj"]).astype(np.float32)
        all_preds: List[np.ndarray] = []
        error_ids: List[int] = []
        padding = "max_length" if cfg.pad_to_max_length else False

        for goal_idx in range(num_goals):
            print(f"Goal index: {goal_idx}")

            def preprocess_function(examples: Dict[str, Any], indices: Sequence[int]) -> Dict[str, Any]:
                inputs = list(examples[history_column])
                for item_index, row_index in enumerate(indices):
                    goal = goals[int(row_index), goal_idx]
                    goal_text = "(" + ", ".join(f"{int(goal[j]):d}" for j in range(2)) + ")"
                    goal_context = f"Pedestrian 0 will arrive at coordinate {goal_text} after the next 12 frames."
                    target_idx = inputs[item_index].find("context: ") + 9
                    inputs[item_index] = inputs[item_index][:target_idx] + goal_context + " " + inputs[item_index][target_idx:]
                targets = examples[future_column]
                model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)
                labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)
                if padding == "max_length":
                    labels["input_ids"] = [
                        [(token if token != tokenizer.pad_token_id else -100) for token in label]
                        for label in labels["input_ids"]
                    ]
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            test_dataset = raw_datasets[split].map(
                preprocess_function,
                batched=True,
                with_indices=True,
                num_proc=cfg.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not cfg.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
            collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
            eval_dataloader = DataLoader(test_dataset, collate_fn=collator, batch_size=cfg.per_device_inference_batch_size)
            model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

            tmp_preds: List[np.ndarray] = []
            for step, batch in enumerate(eval_dataloader):
                generated_trials: List[np.ndarray] = []
                for _ in range(1 if cfg.deterministic else samples_per_goal):
                    generate_kwargs: Dict[str, Any] = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                        "max_length": cfg.max_target_length,
                    }
                    if cfg.deterministic:
                        generate_kwargs["num_beams"] = cfg.num_beams
                    else:
                        generate_kwargs.update(
                            {"do_sample": True, "top_k": cfg.top_k, "temperature": cfg.temperature}
                        )
                    generated_tokens = accelerator.unwrap_model(model).generate(**generate_kwargs)
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    generated_tokens = accelerator.gather_for_metrics((generated_tokens))
                    generated_tokens = generated_tokens.cpu().numpy()
                    generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens
                    if not cfg.use_slow_tokenizer:
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    else:
                        filtered = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
                        decoded_preds = tokenizer.sp_model.decode(filtered.tolist())
                    decoded_preds = [pred.strip() for pred in decoded_preds]
                    traj_data = batch_text2traj(decoded_preds, frame=cfg.pred_len, dim=2)
                    for pid in range(len(traj_data)):
                        if traj_data[pid] is None:
                            ped_id = cfg.per_device_inference_batch_size * accelerator.state.num_processes * step + pid
                            error_ids.append(ped_id)
                            traj_data[pid] = np.tile(all_obs[ped_id, -1], (cfg.pred_len, 1))
                    generated_trials.append(np.array(traj_data))
                tmp_preds.append(np.stack(generated_trials, axis=1))
            all_preds.append(np.concatenate(tmp_preds, axis=0).astype(np.float32))

        if not accelerator.is_local_main_process:
            return []

        cfg.best_of_n = 1
        postprocessed: List[np.ndarray] = []
        for goal_idx in range(num_goals):
            postprocessed.append(
                postprocess_trajectory_simple(all_preds[goal_idx], obs_traj, active_seq_start_end, scene_id, homography, cfg)
            )
        pred_world = np.concatenate(postprocessed, axis=1).astype(np.float32)
        pred_world[:, :, -1] = goals
        for ped_id in range(pred_world.shape[0]):
            if cfg.metric == "pixel":
                h_mat = homography[scene_id[ped_id]]
                pred_world[ped_id] = image2world(pred_world[ped_id], h_mat)

        print(f"Test dataset: {dataset}")
        print(f"Total pedestrian number: {pred_world.shape[0]}")
        if error_ids:
            print(f"decode_error_count={len(error_ids)}")
        return _scene_records(
            dataset=dataset,
            split=split,
            obs_traj=obs_traj,
            pred_traj=pred_traj,
            all_preds=pred_world,
            seq_start_end=active_seq_start_end,
            scene_id=scene_id,
            max_scenes=max_scenes,
        )


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    guide_root = Path(args.guide_root).expanduser().resolve()
    cfg = Path(args.cfg).expanduser().resolve()
    output_path = Path(args.output_bundle).expanduser().resolve()
    if not guide_root.exists():
        raise SystemExit(f"Missing GUIDE-CoT root: {guide_root.as_posix()}")
    if not cfg.exists():
        raise SystemExit(f"Missing GUIDE-CoT config: {cfg.as_posix()}")
    if str(args.split) != "test":
        raise SystemExit("Only split=test is supported for GUIDE-CoT export")

    records = _export_with_official_components(
        guide_root=guide_root,
        cfg_path=cfg,
        dataset=str(args.dataset),
        split=str(args.split),
        checkpoint_name=str(args.checkpoint_name),
        k=int(args.k),
        seed=int(args.seed),
        max_scenes=args.max_scenes,
        preprocessing_num_workers=args.preprocessing_num_workers,
        inference_batch_size=args.inference_batch_size,
    )
    if not records:
        raise SystemExit("No GUIDE-CoT records were exported")
    valid_agents = sum(int(record["agent_mask"].bool().sum().item()) for record in records)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.export_guidecot_predictions",
            "baseline": "GUIDE-CoT",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "guide_root": guide_root.as_posix(),
            "cfg": cfg.as_posix(),
            "checkpoint_name": str(args.checkpoint_name),
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
