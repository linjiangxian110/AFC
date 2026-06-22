"""TrustMoE-Traj 数据协议定义。

当前 V1 决策：
1. 首发数据集：ETH
2. 数据组织：统一多 Agent 格式
3. 第一版暂不引入地图特征，但保留可选字段接口
4. 第一版固定最小字段：past_traj / future_traj / agent_mask / scene_meta
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union


TensorLike = Any

DEFAULT_DATASET = "ETH"
ETH_SUBSETS: Tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")

MIN_REQUIRED_BATCH_FIELDS: Tuple[str, ...] = (
    "past_traj",
    "future_traj",
    "agent_mask",
    "scene_meta",
)

OPTIONAL_BATCH_FIELDS: Tuple[str, ...] = (
    "scene_mask",
    "map_feat",
    "neighbor_attr",
    "extras",
)

STANDARD_OUTPUT_FIELDS: Tuple[str, ...] = (
    "fast_pred",
    "slow_pred",
    "final_pred",
    "route_score",
    "route_decision",
    "expert_weights",
    "expert_states",
    "uncertainty",
    "extras",
)


@dataclass
class SceneMeta:
    """场景元信息。

    第一版建议最少保留 dataset / subset / sample_id / seq_id / frame_id / split / source_file。
    其它信息统一放入 extras，避免前期字段膨胀。
    """

    dataset: str = DEFAULT_DATASET
    subset: str = ""
    sample_id: Optional[str] = None
    seq_id: Optional[str] = None
    frame_id: Optional[int] = None
    split: Optional[str] = None
    source_file: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryBatch:
    """统一 batch 协议。

    约定形状：
    - past_traj:  [B, A, T_obs, 2]
    - future_traj:[B, A, T_pred, 2]
    - agent_mask: [B, A]

    第一版：
    - 使用多 Agent 格式，即使 ETH 的某些样本 agent 数较少，也统一 pad 到该格式。
    - map_feat 暂不使用，但保留字段，避免后期大改接口。
    """

    past_traj: TensorLike
    future_traj: TensorLike
    agent_mask: TensorLike
    scene_meta: Union[
        SceneMeta,
        Dict[str, Any],
        Sequence[SceneMeta],
        Sequence[Dict[str, Any]],
    ]
    scene_mask: Optional[TensorLike] = None
    map_feat: Optional[TensorLike] = None
    neighbor_attr: Optional[TensorLike] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.scene_meta, SceneMeta):
            payload["scene_meta"] = self.scene_meta.to_dict()
        return payload


@dataclass
class ModelOutput:
    """统一模型输出协议。

    说明：
    - fast_pred / slow_pred / final_pred 建议均保持可直接送入 evaluator 的轨迹格式
    - 第一版中某些字段可为空，例如 slow_pred 在未触发时可为 None
    """

    fast_pred: Optional[TensorLike] = None
    slow_pred: Optional[TensorLike] = None
    final_pred: Optional[TensorLike] = None
    route_score: Optional[TensorLike] = None
    route_decision: Optional[TensorLike] = None
    expert_weights: Optional[TensorLike] = None
    expert_states: Optional[TensorLike] = None
    uncertainty: Optional[TensorLike] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def describe_batch_schema() -> Dict[str, Any]:
    """返回当前 batch 协议说明，便于调试或日志记录。"""

    return {
        "default_dataset": DEFAULT_DATASET,
        "eth_subsets": ETH_SUBSETS,
        "multi_agent_format": True,
        "min_required_fields": MIN_REQUIRED_BATCH_FIELDS,
        "optional_fields": OPTIONAL_BATCH_FIELDS,
        "notes": {
            "past_traj": "[B, A, T_obs, 2]",
            "future_traj": "[B, A, T_pred, 2]",
            "agent_mask": "[B, A]，1 表示有效 agent，0 表示 padding agent",
            "scene_meta": "场景元信息，第一版至少记录 dataset/subset/sample_id/seq_id/frame_id/split/source_file",
            "map_feat": "第一版保留接口但默认不启用",
        },
    }


def describe_output_schema() -> Dict[str, Any]:
    """返回当前模型输出协议说明。"""

    return {
        "standard_output_fields": STANDARD_OUTPUT_FIELDS,
        "notes": {
            "fast_pred": "Fast predictor 输出的候选轨迹",
            "slow_pred": "Slow predictor 输出的候选轨迹或精化轨迹",
            "final_pred": "路由之后的最终输出",
            "route_score": "Decision Router 的置信分数",
            "route_decision": "是否升级 slow 的决策结果",
            "expert_weights": "Fusion Router 的专家权重",
            "expert_states": "专家表示层输出",
            "uncertainty": "spread / endpoint variance / collision proxy 等统计量",
        },
    }