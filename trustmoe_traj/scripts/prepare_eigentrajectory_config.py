"""Generate concrete EigenTrajectory config files from repository templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an EigenTrajectory config for one dataset/baseline.")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    template = Path(args.template).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload: Dict[str, Any] = json.loads(template.read_text(encoding="utf-8"))
    payload["dataset"] = str(args.dataset)
    payload["baseline"] = str(args.baseline)
    payload["checkpoint_dir"] = str(Path(args.checkpoint_dir).expanduser().resolve().as_posix())
    if args.dataset_dir:
        payload["dataset_dir"] = str(Path(args.dataset_dir).expanduser().resolve().as_posix()).rstrip("/") + "/"
    if args.epochs is not None:
        payload["num_epochs"] = int(args.epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"config={output.as_posix()}")


if __name__ == "__main__":
    main()
