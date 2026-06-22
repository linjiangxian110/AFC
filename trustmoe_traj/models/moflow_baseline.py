"""MoFlow baseline predictor wrappers for TrustMoE-Traj.

This module provides the minimum glue code needed to:
1. Build ETH fast/slow baselines inside the new TrustMoE-Traj project.
2. Reuse the already-implemented TrustMoE ETH transform layer.
3. Run one-batch forward / sampling / loss smoke tests without relying on the
   original MoFlow training scripts as the main entrypoint.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn as nn
import yaml

from trustmoe_traj.data.transforms import (
    DEFAULT_MOFLOW_SAMPLE_MODE,
    SUPPORTED_MOFLOW_SAMPLE_MODES,
    build_moflow_eth_batch,
    compute_moflow_eth_norm_stats,
)
from trustmoe_traj.data.schema import ModelOutput
from trustmoe_traj.models._einops_fallback import rearrange as fallback_rearrange
from trustmoe_traj.models._einops_fallback import reduce as fallback_reduce
from trustmoe_traj.models._einops_fallback import repeat as fallback_repeat


class AttrDict(dict):
    """Minimal attribute-access dict compatible with MoFlow config usage."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self) -> "AttrDict":  # pragma: no cover - trivial wrapper
        return AttrDict(super().copy())


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _moflow_root() -> Path:
    return _repo_root() / "MoFlow"


def _ensure_moflow_on_path() -> Path:
    root = _moflow_root().resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _install_optional_dependency_stubs() -> None:
    """Install lightweight stubs for MoFlow's import-time optional deps.

    The baseline smoke test only needs model construction / forward / loss.
    Several MoFlow modules import plotting or analysis libraries at module load
    time even though they are not used in the code path we execute here.
    """

    try:
        importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        matplotlib_mod = types.ModuleType("matplotlib")
        pyplot_mod = types.ModuleType("matplotlib.pyplot")
        pyplot_mod.figure = lambda *args, **kwargs: None
        pyplot_mod.plot = lambda *args, **kwargs: None
        pyplot_mod.close = lambda *args, **kwargs: None
        pyplot_mod.subplots = lambda *args, **kwargs: (None, None)
        matplotlib_mod.pyplot = pyplot_mod
        sys.modules.setdefault("matplotlib", matplotlib_mod)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_mod)

    try:
        importlib.import_module("git")
    except ModuleNotFoundError:
        git_mod = types.ModuleType("git")

        class _RepoStub:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ModuleNotFoundError("gitpython is not installed in the current environment")

        git_mod.Repo = _RepoStub
        sys.modules.setdefault("git", git_mod)

    try:
        importlib.import_module("scipy.stats")
    except ModuleNotFoundError:
        scipy_mod = types.ModuleType("scipy")
        stats_mod = types.ModuleType("scipy.stats")

        def _gaussian_kde_stub(*_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError("scipy is not installed in the current environment")

        stats_mod.gaussian_kde = _gaussian_kde_stub
        scipy_mod.stats = stats_mod
        sys.modules.setdefault("scipy", scipy_mod)
        sys.modules.setdefault("scipy.stats", stats_mod)

    try:
        importlib.import_module("einops")
    except ModuleNotFoundError:
        einops_mod = types.ModuleType("einops")
        einops_mod.rearrange = fallback_rearrange
        einops_mod.repeat = fallback_repeat
        einops_mod.reduce = fallback_reduce
        sys.modules.setdefault("einops", einops_mod)

    try:
        importlib.import_module("easydict")
    except ModuleNotFoundError:
        easydict_mod = types.ModuleType("easydict")

        class _EasyDictStub(dict):
            def __getattr__(self, name: str) -> Any:
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value: Any) -> None:
                self[name] = value

        easydict_mod.EasyDict = _EasyDictStub
        sys.modules.setdefault("easydict", easydict_mod)


def _import_moflow_modules() -> Dict[str, Any]:
    _ensure_moflow_on_path()
    _install_optional_dependency_stubs()
    return {
        "flow_matching": importlib.import_module("models.flow_matching"),
        "imle": importlib.import_module("models.imle"),
        "backbone": importlib.import_module("models.backbone_eth_ucy"),
    }


def _load_yaml_config(cfg_path: str | Path) -> AttrDict:
    with Path(cfg_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return _to_attr_dict(yaml.safe_load(handle))


def _infer_cfg_path_from_checkpoint(checkpoint_path: str | Path) -> Optional[Path]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    result_dir = checkpoint.parent.parent
    if not result_dir.exists():
        return None

    updated_ymls = sorted(result_dir.glob("*_updated.yml"))
    if updated_ymls:
        return updated_ymls[0]

    ymls = sorted(result_dir.glob("*.yml"))
    if ymls:
        return ymls[0]
    return None


def _resolve_cfg_path(
    explicit_cfg_path: Optional[str | Path],
    checkpoint_path: Optional[str | Path],
    default_cfg_path: Path,
) -> Path:
    if explicit_cfg_path is not None:
        return Path(explicit_cfg_path).expanduser().resolve()
    if checkpoint_path is not None:
        inferred = _infer_cfg_path_from_checkpoint(checkpoint_path)
        if inferred is not None:
            return inferred
    return default_cfg_path.resolve()


def _build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _to_torch_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _normalize_device_str(device: str | torch.device) -> str:
    return str(_to_torch_device(device))


def _disable_unstable_mha_fastpath() -> None:
    """Avoid CUDA fused Transformer fastpath crashes on some workstation stacks."""

    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath_enabled = getattr(mha_backend, "set_fastpath_enabled", None)
    if callable(set_fastpath_enabled):
        set_fastpath_enabled(False)


def _validate_sample_mode(sample_mode: str) -> None:
    if sample_mode not in SUPPORTED_MOFLOW_SAMPLE_MODES:
        raise ValueError(
            f"Unsupported sample_mode={sample_mode!r}. Expected one of {SUPPORTED_MOFLOW_SAMPLE_MODES}"
        )


@dataclass
class MoFlowPredictorConfig:
    """Configuration shared by MoFlow fast / slow baseline wrappers."""

    subset: str = "eth"
    sample_mode: str = DEFAULT_MOFLOW_SAMPLE_MODE
    agents: int = 1
    data_norm: str = "min_max"
    rotate: bool = False
    rotate_time_frame: int = 0
    device: str = "cpu"
    cfg_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    num_to_gen: int = 1
    sampling_steps: Optional[int] = None
    solver: Optional[str] = None
    lin_poly_p: Optional[int] = None
    lin_poly_long_step: Optional[int] = None
    log_level: int = logging.INFO
    dataset: str = "eth_ucy"


class _MoFlowPredictorBase(nn.Module):
    """Common utilities for MoFlow baseline wrappers."""

    predictor_name = "moflow"
    default_cfg_relpath = ""

    def __init__(self, config: Optional[MoFlowPredictorConfig] = None) -> None:
        super().__init__()
        _disable_unstable_mha_fastpath()
        self.runtime_config = config or MoFlowPredictorConfig()
        _validate_sample_mode(self.runtime_config.sample_mode)

        default_cfg_path = _moflow_root() / self.default_cfg_relpath
        cfg_path = _resolve_cfg_path(
            self.runtime_config.cfg_path,
            self.runtime_config.checkpoint_path,
            default_cfg_path,
        )
        self.cfg = _load_yaml_config(cfg_path)
        self.cfg.cfg_path = cfg_path.as_posix()

        self.device = _to_torch_device(self.runtime_config.device)
        self.logger = _build_logger(
            f"trustmoe_traj.{self.predictor_name}",
            level=self.runtime_config.log_level,
        )
        self.normalization_stats: Dict[str, float] = {}

        self._apply_common_overrides()
        self._apply_kind_specific_overrides()

        modules = _import_moflow_modules()
        self._build_engine(modules)
        self.to(self.device)

        if self.runtime_config.checkpoint_path:
            self.load_checkpoint(self.runtime_config.checkpoint_path)

    def _apply_common_overrides(self) -> None:
        cfg = self.cfg
        cfg.dataset = str(self.runtime_config.dataset)
        cfg.subset = self.runtime_config.subset
        cfg.rotate = self.runtime_config.rotate
        cfg.rotate_time_frame = self.runtime_config.rotate_time_frame
        cfg.rotate_aug = False
        cfg.data_norm = self.runtime_config.data_norm
        cfg.device = _normalize_device_str(self.runtime_config.device)
        cfg.agents = int(self.runtime_config.agents)
        cfg.max_num_ckpts = 1

        if "MODEL" not in cfg or "CONTEXT_ENCODER" not in cfg.MODEL:
            raise ValueError("Invalid MoFlow config: missing MODEL.CONTEXT_ENCODER")

        cfg.MODEL.USE_PRE_NORM = False
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = int(self.runtime_config.agents)

        if self.runtime_config.sampling_steps is not None:
            cfg.sampling_steps = int(self.runtime_config.sampling_steps)
        if self.runtime_config.solver is not None:
            cfg.solver = str(self.runtime_config.solver)
        if self.runtime_config.lin_poly_p is not None:
            cfg.lin_poly_p = int(self.runtime_config.lin_poly_p)
        if self.runtime_config.lin_poly_long_step is not None:
            cfg.lin_poly_long_step = int(self.runtime_config.lin_poly_long_step)

    def _apply_kind_specific_overrides(self) -> None:
        raise NotImplementedError

    def _build_engine(self, modules: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        state = torch.load(path, map_location=self.device)
        state_dict = state.get("model") if isinstance(state, Mapping) and "model" in state else state
        self.engine.load_state_dict(state_dict)
        self.logger.info("[%s] Loaded checkpoint: %s", self.predictor_name, path.as_posix())

    def _set_normalization_stats(self, stats: Optional[Mapping[str, float]]) -> None:
        self.normalization_stats = {} if stats is None else {key: float(val) for key, val in stats.items()}
        if self.runtime_config.data_norm == "min_max" and self.normalization_stats:
            self.cfg.past_traj_min = self.normalization_stats["past_traj_min"]
            self.cfg.past_traj_max = self.normalization_stats["past_traj_max"]
            self.cfg.fut_traj_min = self.normalization_stats["fut_traj_min"]
            self.cfg.fut_traj_max = self.normalization_stats["fut_traj_max"]

    def infer_normalization_stats(self, samples: Sequence[Any]) -> Dict[str, float]:
        if self.runtime_config.data_norm != "min_max":
            return {}
        fixed_num_agents = 1 if self.runtime_config.sample_mode == "per_agent" else self.runtime_config.agents
        stats = compute_moflow_eth_norm_stats(
            samples,
            sample_mode=self.runtime_config.sample_mode,
            rotate=self.runtime_config.rotate,
            rotate_time_frame=self.runtime_config.rotate_time_frame,
            fixed_num_agents=fixed_num_agents,
        )
        return {
            "past_traj_min": float(stats["past_traj_min"]),
            "past_traj_max": float(stats["past_traj_max"]),
            "fut_traj_min": float(stats["fut_traj_min"]),
            "fut_traj_max": float(stats["fut_traj_max"]),
        }

    def get_cfg_normalization_stats(self) -> Dict[str, float]:
        if self.runtime_config.data_norm != "min_max":
            return {}
        keys = ("past_traj_min", "past_traj_max", "fut_traj_min", "fut_traj_max")
        if any(self.cfg.get(key) is None for key in keys):
            return {}
        return {key: float(self.cfg[key]) for key in keys}

    def build_moflow_batch(
        self,
        samples: Sequence[Any],
        *,
        normalization_stats: Optional[Mapping[str, float]] = None,
        as_torch: bool = True,
    ) -> Dict[str, Any]:
        if not samples:
            raise ValueError("build_moflow_batch received an empty sample sequence")

        stats = dict(normalization_stats or self.infer_normalization_stats(samples))
        self._set_normalization_stats(stats or None)

        fixed_num_agents = 1 if self.runtime_config.sample_mode == "per_agent" else self.runtime_config.agents
        return build_moflow_eth_batch(
            samples,
            data_norm=self.runtime_config.data_norm,
            sample_mode=self.runtime_config.sample_mode,
            rotate=self.runtime_config.rotate,
            rotate_time_frame=self.runtime_config.rotate_time_frame,
            fixed_num_agents=fixed_num_agents,
            normalization_stats=stats or None,
            as_torch=as_torch,
        )

    def _prepare_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def _reshape_future(self, tensor: torch.Tensor) -> torch.Tensor:
        future_frames = int(self.cfg.future_frames)
        if tensor.ndim == 4:
            return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], future_frames, 2)
        if tensor.ndim == 5:
            return tensor.reshape(
                tensor.shape[0],
                tensor.shape[1],
                tensor.shape[2],
                tensor.shape[3],
                future_frames,
                2,
            )
        raise ValueError(f"Unsupported future tensor shape: {tuple(tensor.shape)}")

    def _to_metric_scale(self, future_tensor: torch.Tensor) -> torch.Tensor:
        if self.runtime_config.data_norm == "original":
            return future_tensor
        if self.runtime_config.data_norm != "min_max":
            raise NotImplementedError(
                f"Unsupported MoFlow normalization for wrapper output: {self.runtime_config.data_norm!r}"
            )
        if not self.normalization_stats:
            raise ValueError("Missing normalization stats for min_max unnormalization")

        min_val = float(self.normalization_stats["fut_traj_min"])
        max_val = float(self.normalization_stats["fut_traj_max"])
        return ((future_tensor + 1.0) * (max_val - min_val) / 2.0 + min_val).to(torch.float32)

    def forward(self, batch: Mapping[str, Any], *args: Any, **kwargs: Any) -> ModelOutput:
        return self.predict(batch, *args, **kwargs)


class MoFlowSlowPredictor(_MoFlowPredictorBase):
    """Wrapper around MoFlow teacher / slow baseline for ETH."""

    predictor_name = "moflow_slow"
    default_cfg_relpath = "cfg/eth_ucy/cor_fm.yml"

    def _apply_kind_specific_overrides(self) -> None:
        cfg = self.cfg
        cfg.denoising_method = "fm"
        cfg.objective = "pred_data"
        # Preserve training-time sampling behavior when an updated MoFlow config
        # is provided from the original results directory.
        cfg.tied_noise = cfg.get("tied_noise", False)
        cfg.t_schedule = cfg.get("t_schedule", "logit_normal")
        cfg.logit_norm_mean = cfg.get("logit_norm_mean", -0.5)
        cfg.logit_norm_std = cfg.get("logit_norm_std", 1.5)
        cfg.fm_wrapper = cfg.get("fm_wrapper", "direct")
        cfg.fm_rew_sqrt = cfg.get("fm_rew_sqrt", False)
        cfg.fm_in_scaling = cfg.get("fm_in_scaling", False)
        cfg.LOSS_NN_MODE = cfg.get("LOSS_NN_MODE", "agent")
        cfg.LOSS_REG_REDUCTION = cfg.get("LOSS_REG_REDUCTION", "sum")
        cfg.LOSS_REG_SQUARED = cfg.get("LOSS_REG_SQUARED", False)
        cfg.LOSS_VELOCITY = cfg.get("LOSS_VELOCITY", False)
        cfg.drop_method = cfg.get("drop_method", "emb")
        cfg.drop_logi_k = cfg.get("drop_logi_k", 20.0)
        cfg.drop_logi_m = cfg.get("drop_logi_m", 0.5)

    def _build_engine(self, modules: Mapping[str, Any]) -> None:
        backbone = modules["backbone"].ETHMotionTransformer(
            model_config=self.cfg.MODEL,
            logger=self.logger,
            config=self.cfg,
        )
        self.engine = modules["flow_matching"].FlowMatcher(self.cfg, backbone, logger=self.logger)

    def attach_teacher_flow_adapter(self, adapter: nn.Module) -> None:
        """Attach a frozen-teacher flow adapter inside the slow teacher process."""

        self.engine.teacher_flow_adapter = adapter.to(self.device)

    def clear_teacher_flow_adapter(self) -> None:
        """Remove a previously attached teacher flow adapter."""

        if hasattr(self.engine, "teacher_flow_adapter"):
            self.engine.teacher_flow_adapter = None

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        *,
        log_dict: Optional[MutableMapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_batch(batch)
        log_payload = {"cur_epoch": 0} if log_dict is None else dict(log_dict)
        was_training = self.engine.training
        self.engine.train()
        loss, loss_reg, loss_cls, loss_vel = self.engine(prepared, log_payload)
        if not was_training:
            self.engine.eval()
        return {
            "loss": loss,
            "loss_reg": loss_reg,
            "loss_cls": loss_cls,
            "loss_vel": loss_vel,
        }

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, Any],
        *,
        return_all_states: bool = False,
    ) -> ModelOutput:
        prepared = self._prepare_batch(batch)
        was_training = self.engine.training
        self.engine.eval()
        slow_raw, pred_at_t_raw, t_seq, latent_states, pred_score = self.engine.sample(
            prepared,
            num_trajs=self.cfg.denoising_head_preds,
            return_all_states=return_all_states,
        )
        if was_training:
            self.engine.train()

        slow_pred_normalized = self._reshape_future(slow_raw)
        slow_pred_metric = self._to_metric_scale(slow_pred_normalized)

        teacher_latent = slow_pred_normalized
        teacher_latent_sequence = None
        if torch.is_tensor(latent_states):
            teacher_latent_sequence = latent_states.reshape(
                latent_states.shape[0],
                latent_states.shape[1],
                latent_states.shape[2],
                latent_states.shape[3],
                int(self.cfg.future_frames),
                2,
            )
            teacher_latent = teacher_latent_sequence[:, -1]

        pred_at_t_metric = self._to_metric_scale(
            pred_at_t_raw.reshape(
                pred_at_t_raw.shape[0],
                pred_at_t_raw.shape[1],
                pred_at_t_raw.shape[2],
                pred_at_t_raw.shape[3],
                int(self.cfg.future_frames),
                2,
            )
        )

        return ModelOutput(
            slow_pred=slow_pred_metric,
            final_pred=slow_pred_metric,
            extras={
                "slow_pred_normalized": slow_pred_normalized,
                "teacher_latent": teacher_latent,
                "teacher_latent_sequence": teacher_latent_sequence,
                "pred_at_t_metric": pred_at_t_metric,
                "pred_score": pred_score,
                "t_seq": t_seq,
                "sample_mode": self.runtime_config.sample_mode,
            },
        )


class MoFlowSDDSlowPredictor(MoFlowSlowPredictor):
    """Wrapper around MoFlow teacher / slow baseline for SDD."""

    predictor_name = "moflow_sdd_slow"
    default_cfg_relpath = "cfg/sdd/cor_fm.yml"


class MoFlowFastPredictor(_MoFlowPredictorBase):
    """Wrapper around MoFlow student / fast baseline for ETH."""

    predictor_name = "moflow_fast"
    default_cfg_relpath = "cfg/eth_ucy/imle.yml"

    def _apply_kind_specific_overrides(self) -> None:
        cfg = self.cfg
        cfg.objective = cfg.get("objective", "set")
        cfg.latent_tau = cfg.get("latent_tau", 0)
        cfg.num_to_gen = int(self.runtime_config.num_to_gen)
        cfg.load_pretrained = False
        cfg.loss_reg_gt_weight = cfg.get("loss_reg_gt_weight", 0.0)
        cfg.loss_reg_chamfer_weight = cfg.get("loss_reg_chamfer_weight", 1.0)
        cfg.loss_reg_reduction = cfg.get("loss_reg_reduction", "sum")
        cfg.loss_reg_squared = cfg.get("loss_reg_squared", False)

    def _build_engine(self, modules: Mapping[str, Any]) -> None:
        backbone = modules["backbone"].ETHIMLETransformer(
            model_config=self.cfg.MODEL,
            logger=self.logger,
            config=self.cfg,
        )
        self.engine = modules["imle"].IMLE(self.cfg, backbone, logger=self.logger)

    def attach_student_integrated_adapter(self, adapter: nn.Module) -> None:
        """Attach a V18-style adapter inside the fast student generator."""

        self.engine.model.student_integrated_adapter = adapter.to(self.device)

    def clear_student_integrated_adapter(self) -> None:
        """Remove a previously attached student-integrated adapter."""

        if hasattr(self.engine.model, "student_integrated_adapter"):
            self.engine.model.student_integrated_adapter = None

    def attach_student_hidden_adapter(self, adapter: nn.Module) -> None:
        """Attach a V19-style readout-token adapter inside the fast student."""

        self.engine.model.student_hidden_adapter = adapter.to(self.device)

    def clear_student_hidden_adapter(self) -> None:
        """Remove a previously attached readout-token adapter."""

        if hasattr(self.engine.model, "student_hidden_adapter"):
            self.engine.model.student_hidden_adapter = None

    def attach_student_query_adapter(self, adapter: nn.Module) -> None:
        """Attach a V20-style query-token adapter before the motion decoder."""

        self.engine.model.student_query_adapter = adapter.to(self.device)

    def clear_student_query_adapter(self) -> None:
        """Remove a previously attached query-token adapter."""

        if hasattr(self.engine.model, "student_query_adapter"):
            self.engine.model.student_query_adapter = None

    def _resolve_teacher_latent(self, teacher_latent: Any) -> torch.Tensor:
        if isinstance(teacher_latent, ModelOutput):
            teacher_latent = teacher_latent.extras.get("teacher_latent")
        elif isinstance(teacher_latent, Mapping) and "teacher_latent" in teacher_latent:
            teacher_latent = teacher_latent["teacher_latent"]

        if teacher_latent is None:
            raise ValueError("teacher_latent is required to compute IMLE distillation loss")
        if not torch.is_tensor(teacher_latent):
            raise TypeError(f"teacher_latent must be a tensor, got {type(teacher_latent)!r}")

        teacher_latent = teacher_latent.to(self.device, dtype=torch.float32)
        if teacher_latent.ndim == 5:
            return teacher_latent
        if teacher_latent.ndim == 4:
            return teacher_latent.reshape(
                teacher_latent.shape[0],
                teacher_latent.shape[1],
                teacher_latent.shape[2],
                int(self.cfg.future_frames),
                2,
            )
        raise ValueError(
            f"teacher_latent must have shape [B, K, A, F, 2] or [B, K, A, F*2], got {tuple(teacher_latent.shape)}"
        )

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        *,
        teacher_latent: Any,
    ) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_batch(batch)
        prepared = dict(prepared)
        prepared["y_t"] = self._resolve_teacher_latent(teacher_latent)

        was_training = self.engine.training
        self.engine.train()
        loss, loss_chamfer, loss_gt = self.engine(prepared)
        if not was_training:
            self.engine.eval()
        return {
            "loss": loss,
            "loss_chamfer": loss_chamfer,
            "loss_gt": loss_gt,
        }

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, Any],
        *,
        num_to_gen: Optional[int] = None,
    ) -> ModelOutput:
        prepared = self._prepare_batch(batch)
        was_training = self.engine.training
        self.engine.eval()
        generated = self.engine(prepared, num_to_gen=num_to_gen or self.runtime_config.num_to_gen)
        if was_training:
            self.engine.train()

        reshaped = self._reshape_future(generated)
        metric = self._to_metric_scale(reshaped)

        if metric.ndim == 6:
            fast_pred_metric = metric[:, 0]
            fast_pred_normalized = reshaped[:, 0]
        else:
            fast_pred_metric = metric
            fast_pred_normalized = reshaped

        return ModelOutput(
            fast_pred=fast_pred_metric,
            final_pred=fast_pred_metric,
            extras={
                "fast_pred_normalized": fast_pred_normalized,
                "all_generated_metric": metric if metric.ndim == 6 else None,
                "all_generated_normalized": reshaped if reshaped.ndim == 6 else None,
                "sample_mode": self.runtime_config.sample_mode,
            },
        )


__all__ = [
    "MoFlowPredictorConfig",
    "MoFlowFastPredictor",
    "MoFlowSlowPredictor",
    "MoFlowSDDSlowPredictor",
]
