"""Build V57-A semantic residual prototypes from an elite residual pool."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from torch.utils.data import DataLoader

from trustmoe_traj.models import load_social_cvae_teacher_refiner
from trustmoe_traj.scripts.train_social_cvae_refiner import (
    DEFAULT_CACHE_PATH,
    CacheDataset,
    _load_cache,
    _move_batch,
    _prepare_refiner_tensors,
    _resolve_device,
    _select_indices,
    _set_seed,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "v57_semantic_residual_prototypes"
    / "semantic_residual_prototypes.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build semantic residual prototypes for V57-A.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--refiner-checkpoint", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--cache-split", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-residual-slots", type=int, default=4)
    parser.add_argument("--num-prototypes", type=int, default=7)
    parser.add_argument("--elite-topk", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--fde-weight", type=float, default=1.0)
    parser.add_argument("--max-residuals", type=int, default=100000)
    parser.add_argument("--min-endpoint-norm", type=float, default=0.0)
    parser.add_argument("--max-endpoint-norm", type=float, default=1.5)
    parser.add_argument("--kmeans-iters", type=int, default=40)
    parser.add_argument("--kmeans-chunk-size", type=int, default=32768)
    parser.add_argument("--allow-energy-fallback", action="store_true")
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _score(refined: torch.Tensor, ground_truth: torch.Tensor, *, fde_weight: float) -> torch.Tensor:
    dist = torch.linalg.norm(refined - ground_truth[:, None, None, ...], dim=-1)
    return dist.mean(dim=-1) + float(fde_weight) * dist[..., -1]


@torch.no_grad()
def _collect_elite_residuals(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    source_residual_slots: int,
    elite_topk: int,
    min_gain: float,
    fde_weight: float,
    min_endpoint_norm: float,
    max_endpoint_norm: float,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model.refine(
            batch["teacher_pred"],
            past_traj_original_scale=batch["past_traj_original_scale"],
            temporal_energy_features=batch["teacher_temporal_interaction_energy_features"],
            num_samples=int(source_residual_slots),
            z_mode="slots",
        )
        refined = outputs["refined"]
        delta = outputs["delta"]
        scores = _score(refined, batch["ground_truth"], fde_weight=fde_weight)
        base_dist = torch.linalg.norm(batch["teacher_pred"] - batch["ground_truth"][:, None, ...], dim=-1)
        base_scores = base_dist.mean(dim=-1) + float(fde_weight) * base_dist[..., -1]
        gain = base_scores[:, None, :, :] - scores
        batch_size, num_slots, num_modes, num_agents = scores.shape
        flat_scores = scores.reshape(batch_size, num_slots * num_modes, num_agents)
        flat_gain = gain.reshape(batch_size, num_slots * num_modes, num_agents)
        flat_delta = delta.reshape(batch_size, num_slots * num_modes, num_agents, delta.shape[-2], delta.shape[-1])
        keep_k = max(1, min(int(elite_topk), int(flat_scores.shape[1])))
        top_indices = flat_scores.topk(k=keep_k, dim=1, largest=False).indices
        top_gain = torch.gather(flat_gain, dim=1, index=top_indices)
        mask = batch["agent_mask"].bool()
        for batch_index in range(batch_size):
            for agent_index in range(num_agents):
                if not bool(mask[batch_index, agent_index].item()):
                    continue
                selected = flat_delta[batch_index, top_indices[batch_index, :, agent_index], agent_index]
                endpoint_norm = torch.linalg.norm(selected[:, -1, :], dim=-1)
                keep = top_gain[batch_index, :, agent_index] >= float(min_gain)
                keep = keep & (endpoint_norm >= float(min_endpoint_norm))
                if float(max_endpoint_norm) > 0.0:
                    keep = keep & (endpoint_norm <= float(max_endpoint_norm))
                if bool(keep.any().item()):
                    chunks.append(selected[keep].detach().cpu())
    if not chunks:
        raise RuntimeError("No elite residuals were collected; relax norm filters or check the checkpoint/cache.")
    return torch.cat(chunks, dim=0)


def _kmeans(data: torch.Tensor, *, num_clusters: int, iters: int, chunk_size: int, seed: int) -> torch.Tensor:
    if int(data.shape[0]) < int(num_clusters):
        raise ValueError(f"Need at least {num_clusters} residuals, got {int(data.shape[0])}")
    device = data.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    init_index = torch.randperm(int(data.shape[0]), generator=generator, device=device)[: int(num_clusters)]
    centers = data[init_index].clone()
    for _ in range(int(iters)):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(int(num_clusters), device=device, dtype=data.dtype)
        for chunk in data.split(max(int(chunk_size), 1), dim=0):
            labels = torch.cdist(chunk, centers, p=2).argmin(dim=1)
            sums.index_add_(0, labels, chunk)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=data.dtype))
        empty = counts <= 0
        if bool(empty.any().item()):
            replacement = torch.randperm(int(data.shape[0]), generator=generator, device=device)[: int(empty.sum().item())]
            sums[empty] = data[replacement]
            counts[empty] = 1.0
        centers = sums / counts[:, None].clamp_min(1.0)
    return centers


def _stats(residuals: torch.Tensor, prototypes: torch.Tensor) -> Dict[str, Any]:
    endpoint_norm = torch.linalg.norm(residuals[:, -1, :], dim=-1)
    prototype_endpoint_norm = torch.linalg.norm(prototypes[:, -1, :], dim=-1)
    prototype_traj_norm = torch.linalg.norm(prototypes, dim=-1).mean(dim=-1)
    return {
        "num_residuals": int(residuals.shape[0]),
        "future_steps": int(residuals.shape[1]),
        "endpoint_norm_mean": float(endpoint_norm.mean().item()),
        "endpoint_norm_p90": float(torch.quantile(endpoint_norm, 0.90).item()),
        "prototype_endpoint_norm": [float(item) for item in prototype_endpoint_norm.cpu().tolist()],
        "prototype_trajectory_norm": [float(item) for item in prototype_traj_norm.cpu().tolist()],
    }


def main() -> None:
    args = build_parser().parse_args()
    if int(args.source_residual_slots) <= 1:
        raise SystemExit("--source-residual-slots must be > 1")
    if int(args.num_prototypes) <= 0:
        raise SystemExit("--num-prototypes must be positive")
    if int(args.elite_topk) <= 0:
        raise SystemExit("--elite-topk must be positive")
    if float(args.min_gain) < 0.0:
        raise SystemExit("--min-gain must be non-negative")
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    payload = _load_cache(Path(args.cache_path).expanduser().resolve())
    tensors = _prepare_refiner_tensors(payload, args=args)
    num_items = int(tensors["ground_truth"].shape[0])
    train_indices, _val_indices = _select_indices(
        num_items,
        seed=int(args.seed),
        max_items=args.max_items,
        val_fraction=float(args.val_fraction),
    )
    indices = list(range(num_items)) if str(args.cache_split) == "all" else train_indices
    loader = DataLoader(
        CacheDataset(tensors, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    model = load_social_cvae_teacher_refiner(args.refiner_checkpoint, map_location=device).to(device)
    model.eval()
    residuals = _collect_elite_residuals(
        model,
        loader,
        device=device,
        source_residual_slots=int(args.source_residual_slots),
        elite_topk=int(args.elite_topk),
        min_gain=float(args.min_gain),
        fde_weight=float(args.fde_weight),
        min_endpoint_norm=float(args.min_endpoint_norm),
        max_endpoint_norm=float(args.max_endpoint_norm),
    )
    if int(args.max_residuals) > 0 and int(residuals.shape[0]) > int(args.max_residuals):
        generator = torch.Generator()
        generator.manual_seed(int(args.seed))
        keep = torch.randperm(int(residuals.shape[0]), generator=generator)[: int(args.max_residuals)]
        residuals = residuals[keep]
    data = residuals.reshape(residuals.shape[0], -1).to(device=device, dtype=torch.float32)
    centers = _kmeans(
        data,
        num_clusters=int(args.num_prototypes),
        iters=int(args.kmeans_iters),
        chunk_size=int(args.kmeans_chunk_size),
        seed=int(args.seed),
    )
    prototypes = centers.reshape(int(args.num_prototypes), residuals.shape[1], residuals.shape[2]).detach().cpu()
    order = torch.linalg.norm(prototypes, dim=-1).mean(dim=-1).argsort()
    prototypes = prototypes[order].contiguous()
    summary = _stats(residuals, prototypes)
    meta = {
        "script": "trustmoe_traj.scripts.build_v57_semantic_residual_prototypes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": Path(args.cache_path).expanduser().resolve().as_posix(),
        "refiner_checkpoint": Path(args.refiner_checkpoint).expanduser().resolve().as_posix(),
        "args": _jsonable(vars(args)),
        "summary": summary,
    }
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes, "meta": meta}, output_path)
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else output_path.with_suffix(".json")
    )
    output_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"num_residuals={summary['num_residuals']}")
    print(f"prototype_endpoint_norm={summary['prototype_endpoint_norm']}")
    print(f"output_path={output_path.as_posix()}")
    print(f"output_json={output_json.as_posix()}")


if __name__ == "__main__":
    main()
