"""Train PECNet checkpoints for individual ETH-UCY subsets.

The upstream PECNet training script reads pooled `social_pool_data` pickles.
This adapter keeps the PECNet architecture and loss from the official code, but
feeds TrustMoE ETH-UCY train/val splits so each subset can have a dedicated
checkpoint before AFC evaluation.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from trustmoe_traj.data.adapters.eth import ETHAdapterConfig, ETHTrajectoryDataset
from trustmoe_traj.scripts.run_eval import DEFAULT_DATA_ROOT


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PECNet on one ETH-UCY subset split.")
    parser.add_argument("--pecnet-root", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--subset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--config-file", type=str, default="optimal.yaml")
    parser.add_argument("--save-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--best-of-n", type=int, default=None)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-eval-scenes", type=int, default=None)
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(raw: str, gpu_index: int = 0) -> torch.device:
    if raw == "auto":
        raw = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but CUDA is unavailable")
    return device


def _load_pecnet_modules(pecnet_root: Path) -> Dict[str, Any]:
    utils_root = pecnet_root.resolve() / "utils"
    sys.path.insert(0, str(utils_root))
    try:
        models = importlib.import_module("models")
    finally:
        try:
            sys.path.remove(str(utils_root))
        except ValueError:
            pass
    return {"PECNet": models.PECNet}


def _load_hyper_params(pecnet_root: Path, config_file: str) -> Dict[str, Any]:
    config_path = pecnet_root / "config" / config_file
    if not config_path.exists():
        raise SystemExit(f"Missing PECNet config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    return dict(payload)


def _load_samples(
    *,
    data_root: Path,
    subset: str,
    split: str,
    min_agents: int,
    max_scenes: int | None,
) -> List[Mapping[str, Any]]:
    dataset = ETHTrajectoryDataset(
        ETHAdapterConfig(
            data_root=data_root,
            subset=subset,
            split=split,
            min_agents=int(min_agents),
            prefer_cache=False,
        )
    )
    limit = len(dataset) if max_scenes is None else min(int(max_scenes), len(dataset))
    return [dataset[index] for index in range(limit)]


def _sample_tensors(
    sample: Mapping[str, Any],
    *,
    hyper_params: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    past_length = int(hyper_params["past_length"])
    future_length = int(hyper_params["future_length"])
    data_scale = float(hyper_params["data_scale"])

    past_abs = np.asarray(sample["past_traj"], dtype=np.float32)
    future_abs = np.asarray(sample["future_traj"], dtype=np.float32)
    agent_mask = np.asarray(sample.get("agent_mask", np.ones((past_abs.shape[0],), dtype=np.int64)), dtype=np.int64)
    active = agent_mask.astype(bool)
    past_abs = past_abs[active]
    future_abs = future_abs[active]
    if int(past_abs.shape[0]) <= 0:
        raise ValueError("sample contains no active agents")
    if int(past_abs.shape[1]) != past_length or int(future_abs.shape[1]) != future_length:
        raise ValueError(f"Unexpected trajectory length: past={past_abs.shape}, future={future_abs.shape}")

    origin = past_abs[:, :1, :]
    traj_shifted = np.concatenate([past_abs - origin, future_abs - origin], axis=1).astype(np.float32, copy=False)
    traj_scaled = torch.as_tensor(traj_shifted * data_scale, dtype=torch.float64, device=device)
    initial_pos = torch.as_tensor(past_abs[:, past_length - 1, :] / 1000.0, dtype=torch.float64, device=device)
    mask = torch.ones((past_abs.shape[0], past_abs.shape[0]), dtype=torch.float64, device=device)
    x = traj_scaled[:, :past_length, :]
    y = traj_scaled[:, past_length:, :]
    x_flat = x.contiguous().view(-1, x.shape[1] * x.shape[2])
    return x_flat, y, mask, initial_pos, traj_scaled


def _calculate_loss(
    *,
    dest: torch.Tensor,
    dest_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    future: torch.Tensor,
    interpolated_future: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rcl = criterion(dest, dest_recon)
    adl = criterion(future, interpolated_future)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rcl, kld, adl


def _iter_samples(samples: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    items = list(samples)
    random.shuffle(items)
    return items


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: List[Mapping[str, Any]],
    hyper_params: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_rcl = 0.0
    total_kld = 0.0
    total_adl = 0.0
    count = 0
    for sample in _iter_samples(samples):
        x_flat, y, mask, initial_pos, _traj = _sample_tensors(sample, hyper_params=hyper_params, device=device)
        dest = y[:, -1, :]
        future = y[:, :-1, :].contiguous().view(y.size(0), -1)
        dest_recon, mu, logvar, interpolated_future = model.forward(
            x_flat,
            initial_pos,
            dest=dest,
            mask=mask,
            device=device,
        )
        optimizer.zero_grad()
        rcl, kld, adl = _calculate_loss(
            dest=dest,
            dest_recon=dest_recon,
            mu=mu,
            logvar=logvar,
            future=future,
            interpolated_future=interpolated_future,
            criterion=criterion,
        )
        loss = rcl + kld * float(hyper_params["kld_reg"]) + adl * float(hyper_params["adl_reg"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_rcl += float(rcl.detach().cpu())
        total_kld += float(kld.detach().cpu())
        total_adl += float(adl.detach().cpu())
        count += 1
    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "rcl": total_rcl / denom,
        "kld": total_kld / denom,
        "adl": total_adl / denom,
        "count": float(count),
    }


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    samples: List[Mapping[str, Any]],
    hyper_params: Mapping[str, Any],
    device: torch.device,
    best_of_n: int,
) -> Dict[str, float]:
    model.eval()
    data_scale = float(hyper_params["data_scale"])
    future_length = int(hyper_params["future_length"])
    ade_values: List[float] = []
    fde_values: List[float] = []
    avg_fde_values: List[float] = []
    agent_counts: List[int] = []
    for sample in samples:
        x_flat, y, mask, initial_pos, _traj = _sample_tensors(sample, hyper_params=hyper_params, device=device)
        y_np = y.detach().cpu().numpy()
        dest = y_np[:, -1, :]
        all_l2_errors_dest: List[np.ndarray] = []
        all_guesses: List[np.ndarray] = []
        for _ in range(int(best_of_n)):
            dest_recon = model.forward(x_flat, initial_pos, device=device)
            dest_recon_np = dest_recon.detach().cpu().numpy()
            all_guesses.append(dest_recon_np)
            all_l2_errors_dest.append(np.linalg.norm(dest_recon_np - dest, axis=1))
        all_l2 = np.asarray(all_l2_errors_dest)
        all_guesses_np = np.asarray(all_guesses)
        indices = np.argmin(all_l2, axis=0)
        best_dest = all_guesses_np[indices, np.arange(x_flat.shape[0]), :]
        best_dest_t = torch.as_tensor(best_dest, dtype=torch.float64, device=device)
        interpolated = model.predict(x_flat, best_dest_t, mask, initial_pos)
        predicted = torch.cat((interpolated, best_dest_t), dim=1).reshape(-1, future_length, 2)
        pred_np = predicted.detach().cpu().numpy()
        ade = np.linalg.norm(y_np - pred_np, axis=2).mean(axis=1) / data_scale
        fde = np.linalg.norm(y_np[:, -1, :] - pred_np[:, -1, :], axis=1) / data_scale
        avg_fde = all_l2.mean(axis=0) / data_scale
        ade_values.extend(ade.tolist())
        fde_values.extend(fde.tolist())
        avg_fde_values.extend(avg_fde.tolist())
        agent_counts.append(int(x_flat.shape[0]))
    return {
        "ADE_min": float(np.mean(ade_values)) if ade_values else float("nan"),
        "FDE_min": float(np.mean(fde_values)) if fde_values else float("nan"),
        "FDE_avg_dest": float(np.mean(avg_fde_values)) if avg_fde_values else float("nan"),
        "num_scenes": float(len(samples)),
        "num_agents": float(sum(agent_counts)),
    }


def _build_model(pecnet_cls: Any, hyper_params: Mapping[str, Any], verbose: bool) -> torch.nn.Module:
    return pecnet_cls(
        hyper_params["enc_past_size"],
        hyper_params["enc_dest_size"],
        hyper_params["enc_latent_size"],
        hyper_params["dec_size"],
        hyper_params["predictor_hidden_size"],
        hyper_params["non_local_theta_size"],
        hyper_params["non_local_phi_size"],
        hyper_params["non_local_g_size"],
        hyper_params["fdim"],
        hyper_params["zdim"],
        hyper_params["nonlocal_pools"],
        hyper_params["non_local_dim"],
        hyper_params["sigma"],
        hyper_params["past_length"],
        hyper_params["future_length"],
        verbose,
    ).double()


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    pecnet_root = Path(args.pecnet_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    hyper_params = _load_hyper_params(pecnet_root, str(args.config_file))
    if args.epochs is not None:
        hyper_params["num_epochs"] = int(args.epochs)
    if args.best_of_n is not None:
        hyper_params["n_values"] = int(args.best_of_n)
    hyper_params["gpu_index"] = int(hyper_params.get("gpu_index", 0))
    hyper_params["trained_subset"] = str(args.subset)
    hyper_params["train_data_root"] = data_root.as_posix()
    hyper_params["train_adapter"] = "trustmoe_traj.scripts.train_pecnet_eth_subset"

    device = _resolve_device(str(args.device), int(hyper_params.get("gpu_index", 0)))
    pecnet_cls = _load_pecnet_modules(pecnet_root)["PECNet"]
    model = _build_model(pecnet_cls, hyper_params, bool(args.verbose)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyper_params["learning_rate"]))

    train_samples = _load_samples(
        data_root=data_root,
        subset=str(args.subset),
        split="train",
        min_agents=int(args.min_agents),
        max_scenes=args.max_train_scenes,
    )
    eval_samples = _load_samples(
        data_root=data_root,
        subset=str(args.subset),
        split=str(args.eval_split),
        min_agents=int(args.min_agents),
        max_scenes=args.max_eval_scenes,
    )
    if not train_samples:
        raise SystemExit(f"No train samples found: subset={args.subset}")
    if not eval_samples:
        raise SystemExit(f"No eval samples found: subset={args.subset} split={args.eval_split}")

    save_file = args.save_file or f"PECNET_{args.subset}_officialcfg_seed{args.seed}.pt"
    save_path = pecnet_root / "saved_models" / save_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[train_pecnet_eth_subset] "
        f"subset={args.subset} train_scenes={len(train_samples)} eval_scenes={len(eval_samples)} "
        f"eval_split={args.eval_split} epochs={hyper_params['num_epochs']} best_of_n={hyper_params['n_values']} "
        f"device={device} save={save_path.as_posix()}"
    )

    best_eval = float("inf")
    best_payload: Dict[str, Any] | None = None
    eval_every = max(int(args.eval_every), 1)
    for epoch in range(1, int(hyper_params["num_epochs"]) + 1):
        train_metrics = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            samples=train_samples,
            hyper_params=hyper_params,
            device=device,
        )
        should_eval = epoch == 1 or epoch == int(hyper_params["num_epochs"]) or epoch % eval_every == 0
        if should_eval:
            eval_metrics = _evaluate(
                model=model,
                samples=eval_samples,
                hyper_params=hyper_params,
                device=device,
                best_of_n=int(hyper_params["n_values"]),
            )
            current = float(eval_metrics["ADE_min"])
            improved = current < best_eval
            if improved:
                best_eval = current
                best_payload = {
                    "hyper_params": dict(hyper_params),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "meta": {
                        "script": "trustmoe_traj.scripts.train_pecnet_eth_subset",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "subset": str(args.subset),
                        "seed": int(args.seed),
                        "epoch": int(epoch),
                        "eval_split": str(args.eval_split),
                        "eval_metric": "ADE_min",
                        "eval_metrics": eval_metrics,
                        "train_metrics": train_metrics,
                    },
                }
                torch.save(best_payload, save_path)
            print(
                "[train_pecnet_eth_subset] "
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6f} "
                f"eval_ADE_min={eval_metrics['ADE_min']:.6f} eval_FDE_min={eval_metrics['FDE_min']:.6f} "
                f"best_ADE_min={best_eval:.6f} saved={improved}"
            )
        else:
            print(
                "[train_pecnet_eth_subset] "
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6f} "
                f"rcl={train_metrics['rcl']:.6f} kld={train_metrics['kld']:.6f} adl={train_metrics['adl']:.6f}"
            )

    if best_payload is None:
        torch.save(
            {
                "hyper_params": dict(hyper_params),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "meta": {
                    "script": "trustmoe_traj.scripts.train_pecnet_eth_subset",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "subset": str(args.subset),
                    "seed": int(args.seed),
                },
            },
            save_path,
        )
    print(f"checkpoint={save_path.as_posix()}")


if __name__ == "__main__":
    main()
