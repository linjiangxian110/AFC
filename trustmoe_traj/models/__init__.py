"""Model wrappers exposed by the TrustMoE-Traj main project."""

from .fast_predictor import MoFlowFastPredictor, MoFlowPredictorConfig
from .interaction_energy import (
    INTERACTION_ENERGY_FEATURE_NAMES,
    TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM,
    TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES,
    InteractionEnergyConfig,
    InteractionEnergyFeatureBuilder,
    TemporalInteractionEnergyFeatureBuilder,
    compute_interaction_energy_features,
    compute_temporal_interaction_energy_features,
    compute_trajectory_aware_interaction_summary_features,
)
from .residual_graduate import (
    FiLMResidualMLPBlock,
    ModeSetContextEncoder,
    ResidualGraduateConfig,
    ResidualGraduateModel,
    SocialContextEncoder,
    build_residual_graduate_from_cache_shapes,
)
from .slow_predictor import MoFlowSDDSlowPredictor, MoFlowSlowPredictor
from .social_cvae_refiner import (
    SocialCVAETeacherRefiner,
    SocialCVAETeacherRefinerConfig,
    load_social_cvae_teacher_refiner,
)
from .social_cvae_selector import (
    SocialCVAEGroupSelector,
    SocialCVAEGroupSelectorConfig,
    load_social_cvae_group_selector,
)
from .student_integrated_adapter import (
    StudentIntegratedAdapterConfig,
    StudentIntegratedEnergyAdapter,
    build_student_integrated_adapter_from_cache_shapes,
    load_student_integrated_adapter,
)
from .student_hidden_adapter import (
    StudentHiddenAdapterConfig,
    StudentReadoutHiddenAdapter,
    build_student_hidden_adapter_for_model,
    load_student_hidden_adapter,
)
from .teacher_flow_adapter import (
    TeacherFlowAdapter,
    TeacherFlowAdapterConfig,
    build_teacher_flow_adapter_for_engine,
    load_teacher_flow_adapter,
)
from .v55_base_ranker import V55BaseRanker, V55BaseRankerConfig, load_v55_base_ranker
from .v58_slot_quality_scorer import (
    V58SlotQualityScorer,
    V58SlotQualityScorerConfig,
    build_v58_slot_quality_features,
    load_v58_slot_quality_scorer,
    v58_slot_quality_feature_names,
)

__all__ = [
    "MoFlowPredictorConfig",
    "MoFlowFastPredictor",
    "MoFlowSlowPredictor",
    "MoFlowSDDSlowPredictor",
    "SocialCVAETeacherRefiner",
    "SocialCVAETeacherRefinerConfig",
    "load_social_cvae_teacher_refiner",
    "SocialCVAEGroupSelector",
    "SocialCVAEGroupSelectorConfig",
    "load_social_cvae_group_selector",
    "INTERACTION_ENERGY_FEATURE_NAMES",
    "TRAJECTORY_AWARE_INTERACTION_FEATURE_DIM",
    "TRAJECTORY_AWARE_INTERACTION_FEATURE_NAMES",
    "InteractionEnergyConfig",
    "InteractionEnergyFeatureBuilder",
    "TemporalInteractionEnergyFeatureBuilder",
    "compute_interaction_energy_features",
    "compute_temporal_interaction_energy_features",
    "compute_trajectory_aware_interaction_summary_features",
    "FiLMResidualMLPBlock",
    "ModeSetContextEncoder",
    "ResidualGraduateConfig",
    "ResidualGraduateModel",
    "SocialContextEncoder",
    "build_residual_graduate_from_cache_shapes",
    "StudentIntegratedAdapterConfig",
    "StudentIntegratedEnergyAdapter",
    "build_student_integrated_adapter_from_cache_shapes",
    "load_student_integrated_adapter",
    "StudentHiddenAdapterConfig",
    "StudentReadoutHiddenAdapter",
    "build_student_hidden_adapter_for_model",
    "load_student_hidden_adapter",
    "TeacherFlowAdapter",
    "TeacherFlowAdapterConfig",
    "build_teacher_flow_adapter_for_engine",
    "load_teacher_flow_adapter",
    "V55BaseRanker",
    "V55BaseRankerConfig",
    "load_v55_base_ranker",
    "V58SlotQualityScorer",
    "V58SlotQualityScorerConfig",
    "build_v58_slot_quality_features",
    "load_v58_slot_quality_scorer",
    "v58_slot_quality_feature_names",
]
