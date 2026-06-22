"""Small einops fallback for the limited MoFlow patterns used in smoke tests.

This fallback is intentionally narrow. It only supports the concrete patterns
used by the current MoFlow ETH baseline path that we import for wrapper
construction / forward / loss smoke tests.
"""

from __future__ import annotations

from typing import Any

import torch


def _norm(pattern: str) -> str:
    return " ".join(pattern.strip().split())


def _require_size(name: str, sizes: dict[str, Any]) -> int:
    if name not in sizes:
        raise ValueError(f"Missing einops size argument: {name}")
    return int(sizes[name])


def rearrange(tensor: torch.Tensor, pattern: str, **sizes: Any) -> torch.Tensor:
    pattern = _norm(pattern)

    if pattern == "b a p d -> b a (p d)":
        b, a, p, d = tensor.shape
        return tensor.reshape(b, a, p * d)

    if pattern == "b k a d -> (b a) k d":
        b, k, a, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b * a, k, d)

    if pattern == "(b a) k d -> (b k) a d":
        total, k, d = tensor.shape
        b = _require_size("b", sizes)
        a = int(sizes.get("a", total // b))
        return tensor.reshape(b, a, k, d).permute(0, 2, 1, 3).reshape(b * k, a, d)

    if pattern == "(b a) k d -> b k a d":
        total, k, d = tensor.shape
        b = _require_size("b", sizes)
        a = int(sizes.get("a", total // b))
        return tensor.reshape(b, a, k, d).permute(0, 2, 1, 3)

    if pattern == "b k a d -> (b k) a d":
        b, k, a, d = tensor.shape
        return tensor.reshape(b * k, a, d)

    if pattern == "(b k) a d -> b k a d":
        total, a, d = tensor.shape
        b = _require_size("b", sizes)
        k = int(sizes.get("k", total // b))
        return tensor.reshape(b, k, a, d)

    if pattern == "b m k a d -> (b m) k a d":
        b, m, k, a, d = tensor.shape
        return tensor.reshape(b * m, k, a, d)

    if pattern == "(b m) k a d -> b m k a d":
        total, k, a, d = tensor.shape
        m = _require_size("m", sizes)
        b = int(sizes.get("b", total // m))
        return tensor.reshape(b, m, k, a, d)

    if pattern == "b a f d -> b 1 a (f d)":
        b, a, f, d = tensor.shape
        return tensor.reshape(b, 1, a, f * d)

    if pattern == "b k a (f d) -> (b a) k f d":
        b, k, a, fd = tensor.shape
        f = _require_size("f", sizes)
        d = int(sizes.get("d", fd // f))
        return tensor.reshape(b, k, a, f, d).permute(0, 2, 1, 3, 4).reshape(b * a, k, f, d)

    if pattern == "b k a (f d) -> b k a f d":
        b, k, a, fd = tensor.shape
        f = _require_size("f", sizes)
        d = int(sizes.get("d", fd // f))
        return tensor.reshape(b, k, a, f, d)

    if pattern == "b k a -> (b a) k":
        b, k, a = tensor.shape
        return tensor.permute(0, 2, 1).reshape(b * a, k)

    if pattern == "b s k a (f d) -> b s k a f d":
        b, s, k, a, fd = tensor.shape
        f = _require_size("f", sizes)
        d = int(sizes.get("d", fd // f))
        return tensor.reshape(b, s, k, a, f, d)

    if pattern == "b t k a (f d) -> (b a) t k f d":
        b, t, k, a, fd = tensor.shape
        f = _require_size("f", sizes)
        d = int(sizes.get("d", fd // f))
        return tensor.reshape(b, t, k, a, f, d).permute(0, 3, 1, 2, 4, 5).reshape(b * a, t, k, f, d)

    if pattern == "(b a) k f d -> b k a f d":
        total, k, f, d = tensor.shape
        b = _require_size("b", sizes)
        a = int(sizes.get("a", total // b))
        return tensor.reshape(b, a, k, f, d).permute(0, 2, 1, 3, 4)

    if pattern == "B M K A T D -> B A M K 1 T D":
        return tensor.permute(0, 3, 1, 2, 4, 5).unsqueeze(4)

    if pattern == "B K A T D -> B A 1 1 K T D":
        return tensor.permute(0, 2, 1, 3, 4).unsqueeze(2).unsqueeze(2)

    if pattern == "B A T D -> B 1 1 A T D":
        return tensor.unsqueeze(1).unsqueeze(1)

    raise NotImplementedError(f"Unsupported fallback rearrange pattern: {pattern}")


def repeat(tensor: torch.Tensor, pattern: str, **sizes: Any) -> torch.Tensor:
    pattern = _norm(pattern)

    if pattern == "b a d -> b k a d":
        b, a, d = tensor.shape
        k = _require_size("k", sizes)
        return tensor.unsqueeze(1).expand(b, k, a, d)

    if pattern == "b d -> b k a d":
        b, d = tensor.shape
        k = _require_size("k", sizes)
        a = _require_size("a", sizes)
        return tensor[:, None, None, :].expand(b, k, a, d)

    if pattern == "k d -> b k a d":
        k, d = tensor.shape
        b = _require_size("b", sizes)
        a = _require_size("a", sizes)
        return tensor[None, :, None, :].expand(b, k, a, d)

    if pattern == "a d -> b k a d":
        a, d = tensor.shape
        b = _require_size("b", sizes)
        k = _require_size("k", sizes)
        return tensor[None, None, :, :].expand(b, k, a, d)

    if pattern == "b a d -> b m k a d":
        b, a, d = tensor.shape
        m = _require_size("m", sizes)
        k = _require_size("k", sizes)
        return tensor[:, None, None, :, :].expand(b, m, k, a, d)

    if pattern == "k d -> b m k a d":
        k, d = tensor.shape
        b = _require_size("b", sizes)
        m = _require_size("m", sizes)
        a = _require_size("a", sizes)
        return tensor[None, None, :, None, :].expand(b, m, k, a, d)

    if pattern == "a d -> b m k a d":
        a, d = tensor.shape
        b = _require_size("b", sizes)
        m = _require_size("m", sizes)
        k = _require_size("k", sizes)
        return tensor[None, None, None, :, :].expand(b, m, k, a, d)

    if pattern == "b m d -> b m k a d":
        b, m, d = tensor.shape
        k = _require_size("k", sizes)
        a = _require_size("a", sizes)
        return tensor[:, :, None, None, :].expand(b, m, k, a, d)

    if pattern == "b a f d -> b k a (f d)":
        b, a, f, d = tensor.shape
        k = _require_size("k", sizes)
        return tensor.reshape(b, a, f * d).unsqueeze(1).expand(b, k, a, f * d)

    raise NotImplementedError(f"Unsupported fallback repeat pattern: {pattern}")


def reduce(_tensor: torch.Tensor, _pattern: str, _reduction: str, **_sizes: Any) -> torch.Tensor:
    raise NotImplementedError("The local einops fallback does not implement reduce()")


__all__ = [
    "rearrange",
    "repeat",
    "reduce",
]
