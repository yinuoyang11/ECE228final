from __future__ import annotations

from dataclasses import dataclass

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


@PreTrainedConfig.register_subclass("bc_regression")
@dataclass
class BCRegressionConfig(PreTrainedConfig):
    """Configuration for the behavior cloning regression action decoder.

    Mirrors :class:`PI0LiteFlowConfig` so the two policies share the same
    upstream pipeline (encoder, dataset, training loop, evaluation).
    """

    horizon: int = 16
    action_dim: int = 7
    cond_dim: int = 256
    n_action_steps: int = 1

    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.0

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
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

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

        This policy consumes a precomputed conditioning vector rather than raw
        image features, so local unit tests and early experiments can run
        without a full LeRobot dataset feature map.
        """

        action_feature = getattr(self, "action_feature", None)
        output_features = getattr(self, "output_features", None)
        if output_features and action_feature is None:
            raise ValueError("BCRegressionPolicy requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None
