"""Prepare SingularTrajectory config files for TrustMoE AFC runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a concrete SingularTrajectory config.")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--task", type=str, default="stochastic")
    parser.add_argument("--baseline", type=str, default="transformerdiffusion")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    template = Path(args.template).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not template.exists():
        raise SystemExit(f"Missing SingularTrajectory config template: {template.as_posix()}")
    config = _read_json(template)
    config["dataset"] = str(args.dataset)
    config["task"] = str(args.task)
    config["baseline"] = str(args.baseline)
    config["dataset_dir"] = Path(args.dataset_dir).expanduser().resolve().as_posix()
    config["checkpoint_dir"] = Path(args.checkpoint_dir).expanduser().resolve().as_posix()
    if args.num_epochs is not None:
        config["num_epochs"] = int(args.num_epochs)
    if args.num_samples is not None:
        config["num_samples"] = int(args.num_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"config={output.as_posix()}")


if __name__ == "__main__":
    main()
