from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.optim.optimizers import AdamWConfig
except Exception:  # pragma: no cover - exercised when LeRobot is unavailable locally.

    class PreTrainedConfig:
        @classmethod
        def register_subclass(cls, _name: str):
            def decorator(subclass):
                return subclass

            return decorator

        def __post_init__(self) -> None:
            return None

    @dataclass
    class AdamWConfig:
        lr: float = 1e-4
        weight_decay: float = 1e-4


DecoderArch = Literal["mlp_film", "temporal_transformer"]


@PreTrainedConfig.register_subclass("pi0_lite_flow")
@dataclass
class PI0LiteFlowConfig(PreTrainedConfig):
    """Configuration for a pi0-lite conditional flow matching action decoder."""

    horizon: int = 16
    action_dim: int = 7
    cond_dim: int = 256
    n_action_steps: int = 1

    decoder_arch: DecoderArch = "mlp_film"
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.0

    transformer_nhead: int = 4
    transformer_num_layers: int = 3
    transformer_dim_feedforward: int = 1024

    inference_steps: int = 8
    action_eps: float = 1e-6

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        if self.n_action_steps <= 0 or self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must be in [1, horizon]")
        if self.decoder_arch not in {"mlp_film", "temporal_transformer"}:
            raise ValueError(f"Unsupported decoder_arch: {self.decoder_arch}")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        """Validate LeRobot features when the base config provides them.

        This policy consumes a precomputed conditioning vector rather than raw image
        features, so local unit tests and early experiments can run without a full
        LeRobot dataset feature map.
        """

        action_feature = getattr(self, "action_feature", None)
        output_features = getattr(self, "output_features", None)
        if output_features is not None and action_feature is None:
            raise ValueError("PI0LiteFlowPolicy requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None
