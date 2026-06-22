"""Prepare GUIDE-CoT LLM config files for controlled external-baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch GUIDE-CoT llm_module JSON config.")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser


def _set_if_not_none(config: Dict[str, Any], key: str, value: Optional[Any]) -> None:
    if value is not None:
        config[key] = value


def main() -> None:
    args = build_parser().parse_args()
    template = Path(args.template).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not template.exists():
        raise SystemExit(f"Missing GUIDE-CoT config template: {template.as_posix()}")

    config: Dict[str, Any] = json.loads(template.read_text(encoding="utf-8"))
    config["dataset_name"] = str(args.dataset)
    config["num_train_epochs"] = int(args.epochs)
    config["max_train_steps"] = None
    config["seed"] = int(args.seed)
    config["checkpoint_path"] = str(Path(args.checkpoint_path).expanduser().resolve().as_posix()) + "/"
    config["cache_dir"] = str(Path(args.cache_dir).expanduser().resolve().as_posix()) + "/"
    config["preprocessing_num_workers"] = max(int(args.preprocessing_num_workers), 1)
    config["num_samples"] = int(args.num_samples)
    config["best_of_n"] = int(args.num_samples)
    config["save_every"] = int(args.save_every)
    config["use_logger"] = False
    config["logger_type"] = ""
    config["overwrite_cache"] = bool(args.overwrite_cache)
    _set_if_not_none(config, "per_device_train_batch_size", args.train_batch_size)
    _set_if_not_none(config, "per_device_eval_batch_size", args.eval_batch_size)
    _set_if_not_none(config, "per_device_inference_batch_size", args.inference_batch_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"config={output.as_posix()}")


if __name__ == "__main__":
    main()
