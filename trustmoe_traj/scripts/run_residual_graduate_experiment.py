"""Run a full Residual Graduate experiment pipeline.

The pipeline is intentionally thin: it calls the existing export, train, eval,
diagnose, and summary readers in order. For a new architecture variant, add a
small preset here or pass extra CLI fragments without rewriting the whole set
of terminal commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAST_RUN = (
    PROJECT_ROOT
    / "MoFlow"
    / "results_eth_ucy"
    / "imle"
    / "trustmoe_fast_eth_v1_IMLE_gen_set_M_20_load_enc_GT_0.00_Chamfer_1.00_REG_S_eth_rot_6_min_max_LR0.0001_WD0.01_BS24_EP50"
)
DEFAULT_SLOW_RUN = (
    PROJECT_ROOT
    / "MoFlow"
    / "results_eth_ucy"
    / "cor_fm"
    / "trustmoe_slow_eth_v1_retry_FM_S10_log_m-0.5_s1.5_dire_drop_emb_m0.5_k20.0_IS_TN_NN_A_REG_S_eth_rot_6_min_max_LR0.0001_WD0.01_CLS_1.0_BS32_EP150"
)

VALID_STAGES = ("export", "train", "eval", "diagnose", "summary")
EVAL_METRICS = ("ADE_min", "FDE_min", "ADE_avg", "FDE_avg", "MissRate")
DIAGNOSE_METRICS = (
    "endpoint_spread_ratio_mean",
    "trajectory_diversity_ratio_mean",
    "student_best_mode_fde_delta_mean",
    "student_best_mode_worse_rate",
    "best_selector_prob_mean",
    "best_selector_student_best_prob_mean",
    "best_refine_delta_l2_mean",
    "temporal_gate_mean",
    "temporal_gate_early_mean",
    "temporal_gate_mid_mean",
    "temporal_gate_late_mean",
    "temporal_refine_delta_l2_mean",
)


def _add_bool_argument(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    default: bool,
    help: Optional[str] = None,
) -> None:
    """Add a Python 3.8-compatible --flag / --no-flag pair."""

    if not option.startswith("--"):
        raise ValueError(f"Boolean option must start with '--', got {option!r}")
    dest = option[2:].replace("-", "_")
    negative = "--no-" + option[2:]
    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=dest, action="store_true", help=help)
    group.add_argument(negative, dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: bool(default)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run export-cache -> train seeds -> official eval -> diagnose -> "
            "aggregate summary for Residual Graduate variants."
        )
    )
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument(
        "--preset",
        type=str,
        default="v15-light",
        choices=[
            "v17-temporal-refiner-b2",
            "v17-temporal-refiner",
            "v17-best-refiner",
            "v16-mode-context",
            "v15-light",
            "v14-energy",
            "none",
        ],
        help="Built-in argument preset. New architectures can add a preset in this runner.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="v15_light_energyhead_timegate_s2_rankgt_nohurt10_divboth02_ep100",
        help="Used for default run/cache names when --run-prefix or --cache-path is omitted.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Per-seed run IDs are built as RUN_PREFIX_seedN. Default: YYYYMMDD_auto_EXPERIMENT.",
    )
    parser.add_argument("--run-name", type=str, default="graduate")
    parser.add_argument("--seeds", type=str, default="0", help="Comma/space-separated seeds, for example 0,1,2.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Default direct torch device, for example cuda:0.")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional comma/space-separated devices assigned round-robin by seed, for example cuda:0,cuda:1.",
    )
    parser.add_argument("--eval-device", type=str, default=None, help="Override device for eval/diagnose.")
    parser.add_argument("--export-device", type=str, default=None, help="Override device for cache export.")
    parser.add_argument(
        "--parallel-seeds",
        action="store_true",
        help="Train seed runs concurrently. Use --devices cuda:0,cuda:1 to spread work across GPUs.",
    )
    parser.add_argument(
        "--max-parallel-seeds",
        type=int,
        default=None,
        help="Optional maximum concurrent seed trainings. Default: launch all requested seeds.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma/space-separated stages: export,train,eval,diagnose,summary or all.",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="auto",
        choices=["auto", "refresh", "skip"],
        help="auto exports only if the cache is missing; refresh always exports; skip never exports.",
    )
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep later seeds running after a seed failure.")
    _add_bool_argument(
        parser,
        "--skip-existing",
        default=True,
        help=(
            "Skip train/eval/diagnose steps whose expected output files already exist. "
            "Use --no-skip-existing to overwrite existing run outputs."
        ),
    )
    _add_bool_argument(
        parser,
        "--cuda-launch-blocking",
        default=True,
        help="Set CUDA_LAUNCH_BLOCKING=1 for export/eval/diagnose subprocesses.",
    )

    parser.add_argument("--fast-run", type=str, default=str(DEFAULT_FAST_RUN))
    parser.add_argument("--slow-run", type=str, default=str(DEFAULT_SLOW_RUN))
    parser.add_argument("--fast-cfg-path", type=str, default=None)
    parser.add_argument("--fast-checkpoint", type=str, default=None)
    parser.add_argument("--slow-cfg-path", type=str, default=None)
    parser.add_argument("--slow-checkpoint", type=str, default=None)

    parser.add_argument("--protocol", type=str, default="official_align")
    parser.add_argument("--subset", type=str, default="eth")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-splits", type=str, default="val,test")
    parser.add_argument("--diagnose-splits", type=str, default="val,test")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sample-mode", type=str, default="per_agent")
    parser.add_argument("--data-norm", type=str, default="min_max")
    parser.add_argument("--normalization-source", type=str, default="train_split")
    parser.add_argument("--batch-scenes", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    _add_bool_argument(parser, "--rotate", default=True)
    parser.add_argument("--rotate-time-frame", type=int, default=6)
    parser.add_argument("--latency-runs", type=int, default=1)
    _add_bool_argument(parser, "--include-slow", default=True)
    _add_bool_argument(parser, "--diagnose-save-records", default=True)

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-residual-blocks", type=int, default=4)
    parser.add_argument("--block-expansion", type=float, default=2.0)
    parser.add_argument("--collision-sigma", type=float, default=0.5)
    parser.add_argument("--collision-radius", type=float, default=0.2)
    parser.add_argument("--interaction-energy-temporal-stride", type=int, default=2)
    parser.add_argument("--energy-condition-dim", type=int, default=32)
    parser.add_argument("--mode-context-num-layers", type=int, default=1)
    parser.add_argument("--mode-context-num-heads", type=int, default=4)
    parser.add_argument("--mode-context-dropout", type=float, default=0.0)
    parser.add_argument("--best-refine-scale", type=float, default=1.0)
    parser.add_argument("--max-best-refine", type=float, default=None)
    parser.add_argument("--temporal-refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-refine-scale", type=float, default=1.0)
    parser.add_argument("--max-temporal-refine", type=float, default=None)
    parser.add_argument("--temporal-gate-init-bias", type=float, default=None)
    parser.add_argument("--lambda-gt-min", type=float, default=1.0)
    parser.add_argument("--lambda-rank-gt", type=float, default=0.1)
    parser.add_argument("--rank-gt-good-frac", type=float, default=0.25)
    parser.add_argument("--rank-gt-mid-frac", type=float, default=0.50)
    parser.add_argument("--lambda-good-nohurt", type=float, default=1.0)
    parser.add_argument("--good-nohurt-frac", type=float, default=0.25)
    parser.add_argument("--good-nohurt-margin", type=float, default=0.0)
    parser.add_argument("--lambda-diversity-preserve", type=float, default=0.2)
    parser.add_argument("--diversity-preserve-kind", type=str, default="both")
    parser.add_argument("--diversity-preserve-target-ratio", type=float, default=0.98)
    parser.add_argument("--lambda-teacher", type=float, default=0.0)
    parser.add_argument("--lambda-keep", type=float, default=0.0)
    parser.add_argument("--lambda-residual", type=float, default=0.001)
    parser.add_argument("--lambda-gate", type=float, default=0.001)
    parser.add_argument("--lambda-best-selector", type=float, default=None)
    parser.add_argument("--best-selector-top-k", type=int, default=1)
    parser.add_argument("--best-selector-positive-weight", type=float, default=4.0)
    parser.add_argument("--lambda-best-refine", type=float, default=None)
    parser.add_argument("--best-refine-top-k", type=int, default=1)
    parser.add_argument("--best-refine-ade-weight", type=float, default=0.25)
    parser.add_argument("--best-refine-fde-weight", type=float, default=1.0)
    parser.add_argument("--lambda-temporal-gate", type=float, default=None)
    parser.add_argument("--lambda-temporal-smoothness", type=float, default=None)
    parser.add_argument("--lambda-temporal-energy-gate", type=float, default=None)
    parser.add_argument("--lambda-temporal-energy-gt", type=float, default=None)
    parser.add_argument("--temporal-energy-gt-top-k", type=int, default=2)
    parser.add_argument("--temporal-energy-gt-risk-floor", type=float, default=0.05)

    parser.add_argument(
        "--export-extra",
        action="append",
        default=[],
        help="Extra raw args for export script, shell-split. Can be repeated.",
    )
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        help="Extra raw args for train script, shell-split. Can be repeated.",
    )
    parser.add_argument(
        "--eval-extra",
        action="append",
        default=[],
        help="Extra raw args for eval script, shell-split. Can be repeated.",
    )
    parser.add_argument(
        "--diagnose-extra",
        action="append",
        default=[],
        help="Extra raw args for diagnose script, shell-split. Can be repeated.",
    )
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--summary-txt", type=str, default=None)
    return parser


def _split_items(raw: str) -> List[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _split_ints(raw: str) -> List[int]:
    values = _split_items(raw)
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in values]


def _parse_stages(raw: str) -> List[str]:
    items = _split_items(raw)
    if not items or "all" in items:
        return list(VALID_STAGES)
    invalid = [item for item in items if item not in VALID_STAGES]
    if invalid:
        raise ValueError(f"Unknown stage(s): {invalid}. Valid stages: {VALID_STAGES}")
    return items


def _extra_args(chunks: Sequence[str]) -> List[str]:
    args: List[str] = []
    for chunk in chunks:
        args.extend(shlex.split(chunk))
    return args


def _path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _default_run_prefix(args: argparse.Namespace) -> str:
    date = datetime.now().strftime("%Y%m%d")
    return f"{date}_auto_{args.experiment_name}"


def _default_cache_path(project_root: Path, args: argparse.Namespace) -> Path:
    return (
        project_root
        / "trustmoe_traj"
        / "analysis"
        / "teacher_student_cache"
        / f"official_align_{args.subset}_{args.train_split}_teacher_student_predictions_{args.experiment_name}.pt"
    )


def _resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    project_root = _path(args.project_root)
    fast_run = _path(args.fast_run)
    slow_run = _path(args.slow_run)
    cache_path = _path(args.cache_path) if args.cache_path else _default_cache_path(project_root, args)
    run_prefix = args.run_prefix or _default_run_prefix(args)
    run_root = project_root / "trustmoe_traj" / "analysis" / "experiment_runs" / run_prefix
    data_root = _path(args.data_root) if args.data_root else project_root / "trustmoe_traj" / "data" / "ETH"
    summary_json = _path(args.summary_json) if args.summary_json else run_root / "aggregate_summary.json"
    summary_txt = _path(args.summary_txt) if args.summary_txt else run_root / "aggregate_summary.txt"
    return {
        "project_root": project_root,
        "fast_run": fast_run,
        "slow_run": slow_run,
        "fast_cfg": _path(args.fast_cfg_path) if args.fast_cfg_path else fast_run / "imle_updated.yml",
        "fast_ckpt": _path(args.fast_checkpoint) if args.fast_checkpoint else fast_run / "models" / "checkpoint_best.pt",
        "slow_cfg": _path(args.slow_cfg_path) if args.slow_cfg_path else slow_run / "cor_fm_updated.yml",
        "slow_ckpt": _path(args.slow_checkpoint) if args.slow_checkpoint else slow_run / "models" / "checkpoint_best.pt",
        "cache": cache_path,
        "run_root": run_root,
        "data_root": data_root,
        "summary_json": summary_json,
        "summary_txt": summary_txt,
    }


def _device_list(args: argparse.Namespace) -> List[str]:
    return _split_items(args.devices) if args.devices else [str(args.device)]


def _device_for_index(args: argparse.Namespace, index: int) -> str:
    devices = _device_list(args)
    return devices[index % len(devices)]


def _env_for_step(args: argparse.Namespace, *, cuda_launch_blocking: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if cuda_launch_blocking and args.cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    return env


def _module_cmd(module_name: str) -> List[str]:
    return [sys.executable, "-u", "-m", module_name]


def _display_cmd(cmd: Sequence[str]) -> str:
    return shlex.join(str(item) for item in cmd)


def _run_command(
    *,
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str],
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display = _display_cmd(cmd)
    print(f"\n[run_residual_graduate_experiment] {name}")
    print(display)
    if dry_run:
        log_path.write_text(display + "\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# {name}\n# cwd={cwd.as_posix()}\n# command={display}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = int(process.wait())
        log_file.write(f"\n# return_code={return_code}\n")
    if return_code != 0:
        print(f"[run_residual_graduate_experiment] FAILED {name}: return_code={return_code}")
    return return_code


def _stream_process_output(
    *,
    seed: int,
    process: subprocess.Popen[str],
    log_file: Any,
    print_lock: threading.Lock,
) -> None:
    if process.stdout is None:
        return
    prefix = f"[seed{seed}] "
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        with print_lock:
            print(prefix + line, end="", flush=True)


def _shared_dataset_args(args: argparse.Namespace, paths: Mapping[str, Path], split: str) -> List[str]:
    cmd = [
        "--protocol",
        args.protocol,
        "--subset",
        args.subset,
        "--split",
        split,
        "--data-root",
        paths["data_root"].as_posix(),
        "--sample-mode",
        args.sample_mode,
        "--data-norm",
        args.data_norm,
        "--normalization-source",
        args.normalization_source,
        "--batch-scenes",
        str(args.batch_scenes),
    ]
    if args.max_scenes is not None:
        cmd.extend(["--max-scenes", str(args.max_scenes)])
    cmd.append("--rotate" if args.rotate else "--no-rotate")
    cmd.extend(["--rotate-time-frame", str(args.rotate_time_frame)])
    return cmd


def _preset_export_args(args: argparse.Namespace) -> List[str]:
    if args.preset == "none":
        return []
    energy_presets = {"v15-light", "v16-mode-context", "v17-best-refiner", "v17-temporal-refiner", "v17-temporal-refiner-b2"}
    energy_stride = (
        args.interaction_energy_temporal_stride
        if args.preset in energy_presets
        else 1
    )
    cmd = [
        "--include-interaction-energy-features",
        "--interaction-energy-temporal-stride",
        str(energy_stride),
        "--collision-sigma",
        str(args.collision_sigma),
        "--collision-radius",
        str(args.collision_radius),
    ]
    if args.preset in {"v17-temporal-refiner", "v17-temporal-refiner-b2"}:
        cmd.append("--include-temporal-interaction-energy-features")
    return cmd


def _preset_train_args(args: argparse.Namespace) -> List[str]:
    if args.preset == "none":
        return []
    energy_presets = {"v15-light", "v16-mode-context", "v17-best-refiner", "v17-temporal-refiner", "v17-temporal-refiner-b2"}
    energy_stride = (
        args.interaction_energy_temporal_stride
        if args.preset in energy_presets
        else 1
    )
    cmd = [
        "--use-interaction-energy",
        "--interaction-energy-temporal-stride",
        str(energy_stride),
        "--collision-sigma",
        str(args.collision_sigma),
        "--collision-radius",
        str(args.collision_radius),
    ]
    if args.preset in {"v15-light", "v16-mode-context", "v17-best-refiner", "v17-temporal-refiner", "v17-temporal-refiner-b2"}:
        cmd.extend(
            [
                "--use-energy-conditioned-heads",
                "--energy-condition-dim",
                str(args.energy_condition_dim),
                "--use-time-aware-gate",
            ]
        )
    if args.preset in {"v16-mode-context", "v17-best-refiner", "v17-temporal-refiner", "v17-temporal-refiner-b2"}:
        cmd.extend(
            [
                "--use-mode-set-context",
                "--mode-context-num-layers",
                str(args.mode_context_num_layers),
                "--mode-context-num-heads",
                str(args.mode_context_num_heads),
                "--mode-context-dropout",
                str(args.mode_context_dropout),
            ]
        )
    if args.preset == "v17-best-refiner":
        cmd.extend(
            [
                "--use-best-mode-refiner",
                "--best-refine-scale",
                str(args.best_refine_scale),
            ]
        )
        if args.max_best_refine is not None:
            cmd.extend(["--max-best-refine", str(args.max_best_refine)])
    if args.preset in {"v17-temporal-refiner", "v17-temporal-refiner-b2"}:
        cmd.extend(
            [
                "--use-temporal-energy-refiner",
                "--temporal-refiner-hidden-dim",
                str(args.temporal_refiner_hidden_dim),
                "--temporal-refine-scale",
                str(args.temporal_refine_scale),
            ]
        )
        if args.max_temporal_refine is not None:
            cmd.extend(["--max-temporal-refine", str(args.max_temporal_refine)])
        temporal_gate_init_bias = args.temporal_gate_init_bias
        if temporal_gate_init_bias is None and args.preset == "v17-temporal-refiner-b2":
            temporal_gate_init_bias = 1.0
        if temporal_gate_init_bias is not None:
            cmd.extend(["--temporal-gate-init-bias", str(temporal_gate_init_bias)])
    return cmd


def _build_export_cmd(args: argparse.Namespace, paths: Mapping[str, Path], export_device: str) -> List[str]:
    cmd = [
        *_module_cmd("trustmoe_traj.scripts.export_teacher_student_predictions"),
        *_shared_dataset_args(args, paths, args.train_split),
        "--device",
        export_device,
        "--fast-cfg-path",
        paths["fast_cfg"].as_posix(),
        "--fast-checkpoint",
        paths["fast_ckpt"].as_posix(),
        "--slow-cfg-path",
        paths["slow_cfg"].as_posix(),
        "--slow-checkpoint",
        paths["slow_ckpt"].as_posix(),
        *_preset_export_args(args),
        "--output-cache",
        paths["cache"].as_posix(),
        "--output-summary-json",
        (paths["run_root"] / "cache_export_summary.json").as_posix(),
    ]
    cmd.extend(_extra_args(args.export_extra))
    return cmd


def _run_id(args: argparse.Namespace, seed: int) -> str:
    return f"{args.run_prefix or _default_run_prefix(args)}_seed{seed}"


def _graduate_dir(paths: Mapping[str, Path], args: argparse.Namespace, seed: int) -> Path:
    return paths["project_root"] / "trustmoe_traj" / "analysis" / "graduate_models" / _run_id(args, seed)


def _eval_dir(paths: Mapping[str, Path], args: argparse.Namespace, seed: int) -> Path:
    return paths["project_root"] / "trustmoe_traj" / "analysis" / "eval_results" / _run_id(args, seed)


def _train_outputs_exist(paths: Mapping[str, Path], args: argparse.Namespace, seed: int) -> bool:
    run_dir = _graduate_dir(paths, args, seed)
    return (run_dir / f"{args.run_name}_summary.json").exists() and (run_dir / f"{args.run_name}_best.pt").exists()


def _eval_output_path(paths: Mapping[str, Path], args: argparse.Namespace, seed: int, split: str) -> Path:
    return _eval_dir(paths, args, seed) / f"official_{split}.json"


def _diagnose_output_path(paths: Mapping[str, Path], args: argparse.Namespace, seed: int, split: str) -> Path:
    return _eval_dir(paths, args, seed) / f"diagnose_{split}.json"


def _effective_v17_optional(args: argparse.Namespace, name: str, default: float) -> Optional[float]:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    if args.preset == "v17-best-refiner":
        return float(default)
    return None


def _effective_temporal_optional(args: argparse.Namespace, name: str, default: float) -> Optional[float]:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    if args.preset == "v17-temporal-refiner":
        return float(default)
    if args.preset == "v17-temporal-refiner-b2":
        defaults = {
            "lambda_temporal_gate": 0.0,
            "lambda_temporal_smoothness": 0.005,
            "lambda_temporal_energy_gate": 0.0,
            "lambda_temporal_energy_gt": 0.2,
        }
        return float(defaults.get(name, default))
    return None


def _build_train_cmd(args: argparse.Namespace, paths: Mapping[str, Path], seed: int, device: str) -> List[str]:
    lambda_best_selector = _effective_v17_optional(args, "lambda_best_selector", 0.1)
    lambda_best_refine = _effective_v17_optional(args, "lambda_best_refine", 0.2)
    lambda_temporal_gate = _effective_temporal_optional(args, "lambda_temporal_gate", 0.001)
    lambda_temporal_smoothness = _effective_temporal_optional(args, "lambda_temporal_smoothness", 0.01)
    lambda_temporal_energy_gate = _effective_temporal_optional(args, "lambda_temporal_energy_gate", 0.01)
    lambda_temporal_energy_gt = _effective_temporal_optional(args, "lambda_temporal_energy_gt", 0.0)
    cmd = [
        *_module_cmd("trustmoe_traj.scripts.train_residual_graduate"),
        "--cache-path",
        paths["cache"].as_posix(),
        "--output-dir",
        _graduate_dir(paths, args, seed).as_posix(),
        "--run-name",
        args.run_name,
        "--device",
        device,
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-residual-blocks",
        str(args.num_residual_blocks),
        "--block-expansion",
        str(args.block_expansion),
        *_preset_train_args(args),
        "--lambda-gt-min",
        str(args.lambda_gt_min),
        "--lambda-rank-gt",
        str(args.lambda_rank_gt),
        "--rank-gt-good-frac",
        str(args.rank_gt_good_frac),
        "--rank-gt-mid-frac",
        str(args.rank_gt_mid_frac),
        "--lambda-good-nohurt",
        str(args.lambda_good_nohurt),
        "--good-nohurt-frac",
        str(args.good_nohurt_frac),
        "--good-nohurt-margin",
        str(args.good_nohurt_margin),
        "--lambda-diversity-preserve",
        str(args.lambda_diversity_preserve),
        "--diversity-preserve-kind",
        args.diversity_preserve_kind,
        "--diversity-preserve-target-ratio",
        str(args.diversity_preserve_target_ratio),
        "--lambda-teacher",
        str(args.lambda_teacher),
        "--lambda-keep",
        str(args.lambda_keep),
        "--lambda-residual",
        str(args.lambda_residual),
        "--lambda-gate",
        str(args.lambda_gate),
    ]
    if lambda_best_selector is not None:
        cmd.extend(
            [
                "--lambda-best-selector",
                str(lambda_best_selector),
                "--best-selector-top-k",
                str(args.best_selector_top_k),
                "--best-selector-positive-weight",
                str(args.best_selector_positive_weight),
            ]
        )
    if lambda_best_refine is not None:
        cmd.extend(
            [
                "--lambda-best-refine",
                str(lambda_best_refine),
                "--best-refine-top-k",
                str(args.best_refine_top_k),
                "--best-refine-ade-weight",
                str(args.best_refine_ade_weight),
                "--best-refine-fde-weight",
                str(args.best_refine_fde_weight),
            ]
        )
    if lambda_temporal_gate is not None:
        cmd.extend(["--lambda-temporal-gate", str(lambda_temporal_gate)])
    if lambda_temporal_smoothness is not None:
        cmd.extend(["--lambda-temporal-smoothness", str(lambda_temporal_smoothness)])
    if lambda_temporal_energy_gate is not None:
        cmd.extend(["--lambda-temporal-energy-gate", str(lambda_temporal_energy_gate)])
    if lambda_temporal_energy_gt is not None:
        cmd.extend(
            [
                "--lambda-temporal-energy-gt",
                str(lambda_temporal_energy_gt),
                "--temporal-energy-gt-top-k",
                str(args.temporal_energy_gt_top_k),
                "--temporal-energy-gt-risk-floor",
                str(args.temporal_energy_gt_risk_floor),
            ]
        )
    cmd.extend(_extra_args(args.train_extra))
    return cmd


def _build_eval_cmd(args: argparse.Namespace, paths: Mapping[str, Path], seed: int, split: str, device: str) -> List[str]:
    cmd = [
        *_module_cmd("trustmoe_traj.scripts.eval_residual_graduate"),
        *_shared_dataset_args(args, paths, split),
        "--device",
        device,
        "--latency-runs",
        str(args.latency_runs),
        "--graduate-checkpoint",
        (_graduate_dir(paths, args, seed) / f"{args.run_name}_best.pt").as_posix(),
        "--fast-cfg-path",
        paths["fast_cfg"].as_posix(),
        "--fast-checkpoint",
        paths["fast_ckpt"].as_posix(),
        "--output-json",
        _eval_output_path(paths, args, seed, split).as_posix(),
    ]
    if args.include_slow:
        cmd.extend(
            [
                "--include-slow",
                "--slow-cfg-path",
                paths["slow_cfg"].as_posix(),
                "--slow-checkpoint",
                paths["slow_ckpt"].as_posix(),
            ]
        )
    cmd.extend(_extra_args(args.eval_extra))
    return cmd


def _build_diagnose_cmd(args: argparse.Namespace, paths: Mapping[str, Path], seed: int, split: str, device: str) -> List[str]:
    cmd = [
        *_module_cmd("trustmoe_traj.scripts.diagnose_residual_graduate"),
        *_shared_dataset_args(args, paths, split),
        "--device",
        device,
        "--graduate-checkpoint",
        (_graduate_dir(paths, args, seed) / f"{args.run_name}_best.pt").as_posix(),
        "--fast-cfg-path",
        paths["fast_cfg"].as_posix(),
        "--fast-checkpoint",
        paths["fast_ckpt"].as_posix(),
        "--output-json",
        _diagnose_output_path(paths, args, seed, split).as_posix(),
    ]
    if args.diagnose_save_records:
        cmd.append("--save-records")
    cmd.extend(_extra_args(args.diagnose_extra))
    return cmd


def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 6, signed: bool = False) -> str:
    numeric = _num(value)
    if numeric is None:
        return "None"
    prefix = "+" if signed and numeric >= 0 else ""
    return f"{prefix}{numeric:.{digits}f}"


def _metric(metrics: Mapping[str, Any], prefix: str, name: str) -> Optional[float]:
    return _num(metrics.get(f"{prefix}_{name}"))


def _seed_summary(args: argparse.Namespace, paths: Mapping[str, Path], seed: int) -> Dict[str, Any]:
    run_id = _run_id(args, seed)
    missing_files: List[str] = []
    train_summary_path = _graduate_dir(paths, args, seed) / f"{args.run_name}_summary.json"
    train_summary = _load_json(train_summary_path)
    if train_summary is None:
        missing_files.append(train_summary_path.as_posix())
        train_summary = {}
    row: Dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "train_summary_path": train_summary_path.as_posix(),
        "eval_dir": _eval_dir(paths, args, seed).as_posix(),
        "best_epoch": train_summary.get("meta", {}).get("best_epoch"),
        "best_checkpoint": train_summary.get("meta", {}).get("best_checkpoint"),
    }
    best_train = train_summary.get("best_train_metrics", {})
    best_val = train_summary.get("best_val_metrics", {})
    if isinstance(best_train, Mapping):
        row["train_cache_graduate_FDE_min"] = best_train.get("graduate_FDE_min")
        row["train_cache_gate_mean"] = best_train.get("graduate_gate_mean")
        row["train_cache_energy_mean"] = best_train.get("interaction_energy_feature_mean")
        row["train_cache_energy_delta_l2_mean"] = best_train.get("energy_delta_l2_mean")
        row["train_cache_best_selector_prob_mean"] = best_train.get("best_selector_prob_mean")
        row["train_cache_best_refine_delta_l2_mean"] = best_train.get("best_refine_delta_l2_mean")
        row["train_cache_temporal_gate_mean"] = best_train.get("temporal_gate_mean")
        row["train_cache_temporal_refine_delta_l2_mean"] = best_train.get("temporal_refine_delta_l2_mean")
    if isinstance(best_val, Mapping):
        row["cache_val_graduate_FDE_min"] = best_val.get("graduate_FDE_min")

    splits = _split_items(args.eval_splits)
    for split in splits:
        eval_json_path = _eval_output_path(paths, args, seed, split)
        eval_payload = _load_json(eval_json_path)
        if eval_payload is None:
            missing_files.append(eval_json_path.as_posix())
            eval_payload = {}
        metrics = eval_payload.get("metrics", {})
        split_row: Dict[str, Any] = {}
        if isinstance(metrics, Mapping):
            for name in EVAL_METRICS:
                graduate = _metric(metrics, "graduate_pred", name)
                fast = _metric(metrics, "fast_pred", name)
                split_row[f"graduate_{name}"] = graduate
                split_row[f"fast_{name}"] = fast
                split_row[f"d{name}"] = None if graduate is None or fast is None else graduate - fast
            split_row["slow_FDE_min"] = _metric(metrics, "slow_pred", "FDE_min")
            split_row["head_latency_avg_ms"] = _num(metrics.get("graduate_head_latency_avg_ms"))
            split_row["graduate_gate_mean"] = _num(metrics.get("graduate_gate_mean"))
            split_row["graduate_delta_l2_mean"] = _num(metrics.get("graduate_delta_l2_mean"))
            split_row["graduate_best_selector_prob_mean"] = _num(metrics.get("graduate_best_selector_prob_mean"))
            split_row["graduate_best_refine_delta_l2_mean"] = _num(metrics.get("graduate_best_refine_delta_l2_mean"))
            split_row["graduate_temporal_gate_mean"] = _num(metrics.get("graduate_temporal_gate_mean"))
            split_row["graduate_temporal_refine_delta_l2_mean"] = _num(
                metrics.get("graduate_temporal_refine_delta_l2_mean")
            )
        row[f"official_{split}"] = split_row

        diagnose_json_path = _diagnose_output_path(paths, args, seed, split)
        diagnose_payload = _load_json(diagnose_json_path)
        if diagnose_payload is None:
            missing_files.append(diagnose_json_path.as_posix())
            diagnose_payload = {}
        diagnose_summary = diagnose_payload.get("summary", {})
        diagnose_row: Dict[str, Any] = {}
        if isinstance(diagnose_summary, Mapping):
            for name in DIAGNOSE_METRICS:
                diagnose_row[name] = diagnose_summary.get(name)
            diagnose_row["fde_min_delta_mean"] = diagnose_summary.get("fde_min_delta_mean")
            diagnose_row["fde_avg_delta_mean"] = diagnose_summary.get("fde_avg_delta_mean")
            diagnose_row["gate_mean"] = diagnose_summary.get("gate_mean")
            diagnose_row["delta_l2_mean"] = diagnose_summary.get("delta_l2_mean")
            diagnose_row["best_selector_prob_mean"] = diagnose_summary.get("best_selector_prob_mean")
            diagnose_row["best_selector_student_best_prob_mean"] = diagnose_summary.get(
                "best_selector_student_best_prob_mean"
            )
            diagnose_row["best_refine_delta_l2_mean"] = diagnose_summary.get("best_refine_delta_l2_mean")
            diagnose_row["temporal_gate_mean"] = diagnose_summary.get("temporal_gate_mean")
            diagnose_row["temporal_gate_early_mean"] = diagnose_summary.get("temporal_gate_early_mean")
            diagnose_row["temporal_gate_mid_mean"] = diagnose_summary.get("temporal_gate_mid_mean")
            diagnose_row["temporal_gate_late_mean"] = diagnose_summary.get("temporal_gate_late_mean")
            diagnose_row["temporal_refine_delta_l2_mean"] = diagnose_summary.get("temporal_refine_delta_l2_mean")
        row[f"diagnose_{split}"] = diagnose_row
    row["missing_files"] = missing_files
    return row


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [item for item in (_num(value) for value in values) if item is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _aggregate(rows: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for split in splits:
        official_key = f"official_{split}"
        diagnose_key = f"diagnose_{split}"
        split_row: Dict[str, Any] = {}
        split_row["available_official_seeds"] = sum(
            1 for row in rows if _num(row.get(official_key, {}).get("dFDE_min")) is not None
        )
        split_row["available_diagnose_seeds"] = sum(
            1 for row in rows if _num(row.get(diagnose_key, {}).get("student_best_mode_fde_delta_mean")) is not None
        )
        for metric in EVAL_METRICS:
            split_row[f"mean_d{metric}"] = _mean(
                row.get(official_key, {}).get(f"d{metric}") for row in rows
            )
        for metric in DIAGNOSE_METRICS:
            split_row[f"mean_{metric}"] = _mean(
                row.get(diagnose_key, {}).get(metric) for row in rows
            )
        aggregate[split] = split_row
    return aggregate


def _render_summary(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], splits: Sequence[str]) -> str:
    lines: List[str] = []
    for row in rows:
        lines.append(f"===== {row['run_id']} =====")
        missing_files = row.get("missing_files") or []
        if missing_files:
            lines.append("missing summary inputs:")
            for path in missing_files:
                lines.append(f"  {path}")
        lines.append(f"best_epoch: {row.get('best_epoch')}")
        lines.append(f"best checkpoint: {row.get('best_checkpoint')}")
        lines.append(f"train-cache best graduate_FDE_min: {row.get('train_cache_graduate_FDE_min')}")
        lines.append(f"train-cache gate: {row.get('train_cache_gate_mean')}")
        lines.append(f"train-cache energy_mean: {row.get('train_cache_energy_mean')}")
        if row.get("train_cache_energy_delta_l2_mean") is not None:
            lines.append(f"train-cache energy_delta_l2: {row.get('train_cache_energy_delta_l2_mean')}")
        if row.get("train_cache_best_selector_prob_mean") is not None:
            lines.append(f"train-cache best_selector_prob: {row.get('train_cache_best_selector_prob_mean')}")
        if row.get("train_cache_best_refine_delta_l2_mean") is not None:
            lines.append(f"train-cache best_refine_delta_l2: {row.get('train_cache_best_refine_delta_l2_mean')}")
        if row.get("train_cache_temporal_gate_mean") is not None:
            lines.append(f"train-cache temporal_gate: {row.get('train_cache_temporal_gate_mean')}")
        if row.get("train_cache_temporal_refine_delta_l2_mean") is not None:
            lines.append(
                "train-cache temporal_refine_delta_l2: "
                f"{row.get('train_cache_temporal_refine_delta_l2_mean')}"
            )
        lines.append("")
        for split in splits:
            official = row.get(f"official_{split}", {})
            lines.append(f"-- official {split} Graduate - Fast --")
            for metric in EVAL_METRICS:
                lines.append(
                    f"d{metric}: {_fmt(official.get(f'd{metric}'), signed=True)}  "
                    f"graduate={_fmt(official.get(f'graduate_{metric}'))}  "
                    f"fast={_fmt(official.get(f'fast_{metric}'))}"
                )
            lines.append(f"slow_FDE_min: {official.get('slow_FDE_min')}")
            lines.append(f"head_latency_avg_ms: {official.get('head_latency_avg_ms')}")
            lines.append(f"graduate_gate_mean: {official.get('graduate_gate_mean')}")
            lines.append(f"graduate_delta_l2_mean: {official.get('graduate_delta_l2_mean')}")
            if official.get("graduate_best_selector_prob_mean") is not None:
                lines.append(f"graduate_best_selector_prob_mean: {official.get('graduate_best_selector_prob_mean')}")
            if official.get("graduate_best_refine_delta_l2_mean") is not None:
                lines.append(
                    f"graduate_best_refine_delta_l2_mean: {official.get('graduate_best_refine_delta_l2_mean')}"
                )
            if official.get("graduate_temporal_gate_mean") is not None:
                lines.append(f"graduate_temporal_gate_mean: {official.get('graduate_temporal_gate_mean')}")
            if official.get("graduate_temporal_refine_delta_l2_mean") is not None:
                lines.append(
                    "graduate_temporal_refine_delta_l2_mean: "
                    f"{official.get('graduate_temporal_refine_delta_l2_mean')}"
                )
            lines.append("")
            diagnose = row.get(f"diagnose_{split}", {})
            lines.append(f"-- diagnose {split} --")
            lines.append(f"endpoint_ratio: {diagnose.get('endpoint_spread_ratio_mean')}")
            lines.append(f"traj_ratio: {diagnose.get('trajectory_diversity_ratio_mean')}")
            lines.append(f"student_best_fde_delta: {diagnose.get('student_best_mode_fde_delta_mean')}")
            lines.append(f"student_best_worse_rate: {diagnose.get('student_best_mode_worse_rate')}")
            if diagnose.get("best_selector_prob_mean") is not None:
                lines.append(f"best_selector_prob_mean: {diagnose.get('best_selector_prob_mean')}")
            if diagnose.get("best_selector_student_best_prob_mean") is not None:
                lines.append(
                    "best_selector_student_best_prob_mean: "
                    f"{diagnose.get('best_selector_student_best_prob_mean')}"
                )
            if diagnose.get("best_refine_delta_l2_mean") is not None:
                lines.append(f"best_refine_delta_l2_mean: {diagnose.get('best_refine_delta_l2_mean')}")
            if diagnose.get("temporal_gate_mean") is not None:
                lines.append(f"temporal_gate_mean: {diagnose.get('temporal_gate_mean')}")
                lines.append(f"temporal_gate_early_mean: {diagnose.get('temporal_gate_early_mean')}")
                lines.append(f"temporal_gate_mid_mean: {diagnose.get('temporal_gate_mid_mean')}")
                lines.append(f"temporal_gate_late_mean: {diagnose.get('temporal_gate_late_mean')}")
            if diagnose.get("temporal_refine_delta_l2_mean") is not None:
                lines.append(f"temporal_refine_delta_l2_mean: {diagnose.get('temporal_refine_delta_l2_mean')}")
            lines.append("")
    lines.append(f"===== MEAN DELTAS (available seeds only; requested={len(rows)}) =====")
    for split in splits:
        split_mean = aggregate.get(split, {})
        lines.append("")
        lines.append(f"-- {split} --")
        lines.append(
            "available seeds: "
            f"official={int(split_mean.get('available_official_seeds') or 0)}/{len(rows)} "
            f"diagnose={int(split_mean.get('available_diagnose_seeds') or 0)}/{len(rows)}"
        )
        for metric in EVAL_METRICS:
            lines.append(f"mean d{metric}: {_fmt(split_mean.get(f'mean_d{metric}'), signed=True)}")
        lines.append(f"mean endpoint_ratio: {_fmt(split_mean.get('mean_endpoint_spread_ratio_mean'))}")
        lines.append(f"mean traj_ratio: {_fmt(split_mean.get('mean_trajectory_diversity_ratio_mean'))}")
        lines.append(f"mean student_best_fde_delta: {_fmt(split_mean.get('mean_student_best_mode_fde_delta_mean'))}")
        lines.append(f"mean student_best_worse_rate: {_fmt(split_mean.get('mean_student_best_mode_worse_rate'))}")
        lines.append(f"mean best_selector_prob: {_fmt(split_mean.get('mean_best_selector_prob_mean'))}")
        lines.append(
            "mean best_selector_student_best_prob: "
            f"{_fmt(split_mean.get('mean_best_selector_student_best_prob_mean'))}"
        )
        lines.append(f"mean best_refine_delta_l2: {_fmt(split_mean.get('mean_best_refine_delta_l2_mean'))}")
        lines.append(f"mean temporal_gate: {_fmt(split_mean.get('mean_temporal_gate_mean'))}")
        lines.append(f"mean temporal_gate_early: {_fmt(split_mean.get('mean_temporal_gate_early_mean'))}")
        lines.append(f"mean temporal_gate_mid: {_fmt(split_mean.get('mean_temporal_gate_mid_mean'))}")
        lines.append(f"mean temporal_gate_late: {_fmt(split_mean.get('mean_temporal_gate_late_mean'))}")
        lines.append(f"mean temporal_refine_delta_l2: {_fmt(split_mean.get('mean_temporal_refine_delta_l2_mean'))}")
    return "\n".join(lines).rstrip() + "\n"


def _write_summary(args: argparse.Namespace, paths: Mapping[str, Path], seeds: Sequence[int]) -> None:
    splits = _split_items(args.eval_splits)
    rows = [_seed_summary(args, paths, seed) for seed in seeds]
    aggregate = _aggregate(rows, splits)
    payload = {
        "meta": {
            "script": "trustmoe_traj.scripts.run_residual_graduate_experiment",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "preset": args.preset,
            "experiment_name": args.experiment_name,
            "run_prefix": args.run_prefix or _default_run_prefix(args),
            "cache_path": paths["cache"].as_posix(),
        },
        "args": vars(args),
        "rows": rows,
        "aggregate": aggregate,
    }
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_text = _render_summary(rows, aggregate, splits)
    paths["summary_txt"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_txt"].write_text(summary_text, encoding="utf-8")
    print("\n[run_residual_graduate_experiment] aggregate summary")
    print(summary_text)
    print(f"summary_json={paths['summary_json'].as_posix()}")
    print(f"summary_txt={paths['summary_txt'].as_posix()}")


def _run_train_parallel(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    seeds: Sequence[int],
    logs_dir: Path,
) -> List[int]:
    max_parallel = len(seeds) if args.max_parallel_seeds is None else int(args.max_parallel_seeds)
    if max_parallel < 1:
        raise ValueError("--max-parallel-seeds must be >= 1")
    if args.dry_run:
        return_codes_by_seed: Dict[int, int] = {}
        for batch_start in range(0, len(seeds), max_parallel):
            batch = list(enumerate(seeds))[batch_start : batch_start + max_parallel]
            print(
                "[run_residual_graduate_experiment] dry-run parallel train batch "
                f"{batch_start // max_parallel + 1}: seeds={[seed for _, seed in batch]}"
            )
            for index, seed in batch:
                if args.skip_existing and _train_outputs_exist(paths, args, seed):
                    print(f"[run_residual_graduate_experiment] skip existing train seed={seed}")
                    return_codes_by_seed[int(seed)] = 0
                    continue
                return_codes_by_seed[int(seed)] = _run_command(
                    name=f"train seed={seed}",
                    cmd=_build_train_cmd(args, paths, seed, _device_for_index(args, index)),
                    cwd=paths["project_root"],
                    log_path=logs_dir / f"train_seed{seed}.log",
                    env=_env_for_step(args, cuda_launch_blocking=False),
                    dry_run=True,
                )
        return [return_codes_by_seed.get(int(seed), 1) for seed in seeds]

    return_codes_by_seed: Dict[int, int] = {}
    indexed_seeds = list(enumerate(seeds))
    print_lock = threading.Lock()
    for batch_start in range(0, len(indexed_seeds), max_parallel):
        batch = indexed_seeds[batch_start : batch_start + max_parallel]
        processes: List[tuple[int, subprocess.Popen[str], Path, Any, threading.Thread]] = []
        print(
            "\n[run_residual_graduate_experiment] parallel train batch "
            f"{batch_start // max_parallel + 1}: seeds={[seed for _, seed in batch]}"
        )
        for index, seed in batch:
            if args.skip_existing and _train_outputs_exist(paths, args, seed):
                print(f"[run_residual_graduate_experiment] skip existing train seed={seed}")
                return_codes_by_seed[int(seed)] = 0
                continue
            device = _device_for_index(args, index)
            cmd = _build_train_cmd(args, paths, seed, device)
            log_path = logs_dir / f"train_seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            display = _display_cmd(cmd)
            print(f"[run_residual_graduate_experiment] launch train seed={seed} device={device}")
            print(display)
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"# train seed={seed}\n# command={display}\n\n")
            log_file.flush()
            process = subprocess.Popen(
                cmd,
                cwd=str(paths["project_root"]),
                env=_env_for_step(args, cuda_launch_blocking=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stream_thread = threading.Thread(
                target=_stream_process_output,
                kwargs={
                    "seed": seed,
                    "process": process,
                    "log_file": log_file,
                    "print_lock": print_lock,
                },
                daemon=True,
            )
            stream_thread.start()
            processes.append((seed, process, log_path, log_file, stream_thread))

        for seed, process, log_path, log_file, stream_thread in processes:
            return_code = int(process.wait())
            stream_thread.join()
            log_file.write(f"\n# return_code={return_code}\n")
            log_file.close()
            return_codes_by_seed[int(seed)] = return_code
            status = "completed" if return_code == 0 else "FAILED"
            print(f"[run_residual_graduate_experiment] {status} train seed={seed} log={log_path.as_posix()}")
    return [return_codes_by_seed.get(int(seed), 1) for seed in seeds]


def main() -> None:
    args = build_parser().parse_args()
    seeds = _split_ints(args.seeds)
    stages = _parse_stages(args.stages)
    if args.run_prefix is None:
        args.run_prefix = _default_run_prefix(args)
    paths = _resolve_paths(args)
    logs_dir = paths["run_root"] / "logs"
    paths["run_root"].mkdir(parents=True, exist_ok=True)
    manifest_path = paths["run_root"] / "manifest.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "trustmoe_traj.scripts.run_residual_graduate_experiment",
        "args": vars(args),
        "seeds": list(seeds),
        "stages": stages,
        "paths": {key: value.as_posix() for key, value in paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run_residual_graduate_experiment] manifest={manifest_path.as_posix()}")

    failures: List[str] = []
    export_device = args.export_device or args.device
    eval_device_default = args.eval_device

    if "export" in stages:
        should_export = args.cache_mode == "refresh" or (args.cache_mode == "auto" and not paths["cache"].exists())
        if args.cache_mode == "skip":
            should_export = False
        if should_export:
            code = _run_command(
                name="export cache",
                cmd=_build_export_cmd(args, paths, export_device),
                cwd=paths["project_root"],
                log_path=logs_dir / "export_cache.log",
                env=_env_for_step(args, cuda_launch_blocking=True),
                dry_run=args.dry_run,
            )
            if code != 0:
                raise SystemExit(code)
        else:
            print(f"[run_residual_graduate_experiment] skip export cache_path={paths['cache'].as_posix()}")

    if "train" in stages and not paths["cache"].exists() and not args.dry_run:
        raise FileNotFoundError(
            f"Cache not found before training: {paths['cache']}. "
            "Run with --stages export,train,... or provide an existing --cache-path."
        )

    if "train" in stages:
        if args.parallel_seeds:
            codes = _run_train_parallel(args, paths, seeds, logs_dir)
            for seed, code in zip(seeds, codes):
                if code != 0:
                    failures.append(f"train seed={seed} return_code={code}")
            if failures and not args.continue_on_error:
                raise SystemExit("\n".join(failures))
        else:
            for index, seed in enumerate(seeds):
                if args.skip_existing and _train_outputs_exist(paths, args, seed):
                    print(f"[run_residual_graduate_experiment] skip existing train seed={seed}")
                    continue
                code = _run_command(
                    name=f"train seed={seed}",
                    cmd=_build_train_cmd(args, paths, seed, _device_for_index(args, index)),
                    cwd=paths["project_root"],
                    log_path=logs_dir / f"train_seed{seed}.log",
                    env=_env_for_step(args, cuda_launch_blocking=False),
                    dry_run=args.dry_run,
                )
                if code != 0:
                    failures.append(f"train seed={seed} return_code={code}")
                    if not args.continue_on_error:
                        raise SystemExit(code)

    for index, seed in enumerate(seeds):
        seed_device = eval_device_default or _device_for_index(args, index)
        if "eval" in stages:
            for split in _split_items(args.eval_splits):
                eval_output = _eval_output_path(paths, args, seed, split)
                if args.skip_existing and eval_output.exists():
                    print(
                        "[run_residual_graduate_experiment] skip existing "
                        f"eval seed={seed} split={split} output={eval_output.as_posix()}"
                    )
                    continue
                code = _run_command(
                    name=f"eval seed={seed} split={split}",
                    cmd=_build_eval_cmd(args, paths, seed, split, seed_device),
                    cwd=paths["project_root"],
                    log_path=logs_dir / f"eval_seed{seed}_{split}.log",
                    env=_env_for_step(args, cuda_launch_blocking=True),
                    dry_run=args.dry_run,
                )
                if code != 0:
                    failures.append(f"eval seed={seed} split={split} return_code={code}")
                    if not args.continue_on_error:
                        raise SystemExit(code)
        if "diagnose" in stages:
            for split in _split_items(args.diagnose_splits):
                diagnose_output = _diagnose_output_path(paths, args, seed, split)
                if args.skip_existing and diagnose_output.exists():
                    print(
                        "[run_residual_graduate_experiment] skip existing "
                        f"diagnose seed={seed} split={split} output={diagnose_output.as_posix()}"
                    )
                    continue
                code = _run_command(
                    name=f"diagnose seed={seed} split={split}",
                    cmd=_build_diagnose_cmd(args, paths, seed, split, seed_device),
                    cwd=paths["project_root"],
                    log_path=logs_dir / f"diagnose_seed{seed}_{split}.log",
                    env=_env_for_step(args, cuda_launch_blocking=True),
                    dry_run=args.dry_run,
                )
                if code != 0:
                    failures.append(f"diagnose seed={seed} split={split} return_code={code}")
                    if not args.continue_on_error:
                        raise SystemExit(code)

    if "summary" in stages and not args.dry_run:
        _write_summary(args, paths, seeds)
    elif "summary" in stages:
        print("[run_residual_graduate_experiment] dry-run skips reading summary inputs")

    if failures:
        raise SystemExit("Completed with failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
