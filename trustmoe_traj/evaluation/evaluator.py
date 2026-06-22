"""Unified evaluator for TrustMoE-Traj baseline outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from trustmoe_traj.data.schema import ModelOutput

from .metrics_accuracy import infer_ground_truth_from_batch, summarize_accuracy_metrics
from .metrics_efficiency import summarize_latency_ms, summarize_route_usage


PREDICTION_FIELDS: Sequence[str] = ("fast_pred", "slow_pred", "final_pred")


@dataclass
class EvaluationSummary:
    """Container for evaluator outputs."""

    metrics: Dict[str, float] = field(default_factory=dict)
    available_predictions: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_model_output(output: ModelOutput | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(output, ModelOutput):
        return output.to_dict()
    return output


def evaluate_model_output(
    output: ModelOutput | Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    miss_threshold: float = 2.0,
    prediction_fields: Sequence[str] = PREDICTION_FIELDS,
    latency_ms: Optional[Mapping[str, float | Iterable[float]]] = None,
    route_decision: Optional[Iterable[float | int]] = None,
) -> EvaluationSummary:
    """Evaluate available prediction branches in a ModelOutput."""

    payload = _coerce_model_output(output)
    gt_payload = infer_ground_truth_from_batch(batch)
    ground_truth = gt_payload["ground_truth"]
    agent_mask = gt_payload["agent_mask"]

    metrics: Dict[str, float] = {}
    available_predictions = []

    for field_name in prediction_fields:
        prediction = payload.get(field_name)
        if prediction is None:
            continue
        branch_metrics = summarize_accuracy_metrics(
            prediction,
            ground_truth,
            agent_mask=agent_mask,
            miss_threshold=miss_threshold,
        )
        metrics.update({f"{field_name}_{key}": value for key, value in branch_metrics.items()})
        available_predictions.append(field_name)

    if latency_ms:
        for branch_name, branch_latency in latency_ms.items():
            branch_summary = summarize_latency_ms(branch_latency)
            metrics.update({f"{branch_name}_{key}": value for key, value in branch_summary.items()})

    if route_decision is not None:
        metrics.update(summarize_route_usage(route_decision))

    return EvaluationSummary(
        metrics=metrics,
        available_predictions=tuple(available_predictions),
    )


__all__ = [
    "PREDICTION_FIELDS",
    "EvaluationSummary",
    "evaluate_model_output",
]
