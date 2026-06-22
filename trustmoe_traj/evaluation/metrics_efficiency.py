"""Efficiency metrics for TrustMoE-Traj evaluators."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def summarize_latency_ms(latencies_ms: float | Iterable[float]) -> Dict[str, float]:
    """Summarize latency values in milliseconds."""

    if isinstance(latencies_ms, (int, float)):
        values = np.asarray([float(latencies_ms)], dtype=np.float64)
    else:
        values = np.asarray([float(item) for item in latencies_ms], dtype=np.float64)

    if values.size == 0:
        raise ValueError("Latency summary requires at least one value")

    return {
        "latency_count": float(values.size),
        "latency_avg_ms": float(values.mean()),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_min_ms": float(values.min()),
        "latency_max_ms": float(values.max()),
    }


def summarize_route_usage(route_decision: np.ndarray | Iterable[float | int]) -> Dict[str, float]:
    """Summarize slow-branch usage ratio from route decisions."""

    values = np.asarray(list(route_decision), dtype=np.float64)
    if values.size == 0:
        raise ValueError("route_decision summary requires at least one value")
    return {
        "route_count": float(values.size),
        "slow_usage_ratio": float(values.mean()),
    }


__all__ = [
    "summarize_latency_ms",
    "summarize_route_usage",
]
