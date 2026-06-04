from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .configuration_regression import BCRegressionConfig

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION
except Exception:  # pragma: no cover - exercised when LeRobot is unavailable locally.
    ACTION = "action"

    class PreTrainedPolicy(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()


COND_KEY = "observation.cond"


class BCRegressionDecoder(nn.Module):
    """MLP that directly regresses action chunks from a conditioning vector.

    The decoder mirrors :class:`CFMActionDecoder` in shape conventions and
    statistics handling so the two baselines can share the same encoder,
    dataset stats, and evaluation harness. The only methodological difference
    is the learning objective: this module trains with mean squared error
    against the expert action chunk (behavior cloning regression) rather than
    flow matching velocities.
    """

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 7,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        action_mean: Tensor | None = None,
        action_std: Tensor | None = None,
        action_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.horizon = horizon
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.action_eps = action_eps

        layers: list[nn.Module] = [nn.Linear(cond_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)

        mean = torch.zeros(action_dim) if action_mean is None else action_mean.detach().float()
        std = torch.ones(action_dim) if action_std is None else action_std.detach().float()
        self.register_buffer("action_mean", self._format_stat(mean))
        self.register_buffer("action_std", self._format_stat(std).clamp_min(action_eps))

    def _format_stat(self, stat: Tensor) -> Tensor:
        stat = stat.detach().float()
        if stat.numel() == 1:
            stat = stat.expand(self.action_dim)
        if stat.shape == (self.action_dim,):
            return stat.reshape(1, 1, self.action_dim)
        if stat.shape == (1, self.action_dim):
            return stat.reshape(1, 1, self.action_dim)
        if stat.shape == (self.horizon, self.action_dim):
            return stat.reshape(1, self.horizon, self.action_dim)
        if stat.shape == (1, self.horizon, self.action_dim):
            return stat
        raise ValueError(
            "action stat must be scalar, (action_dim,), (horizon, action_dim), "
            f"or (1, horizon, action_dim); got {tuple(stat.shape)}"
        )

    def set_action_stats(self, action_mean: Tensor, action_std: Tensor) -> None:
        self.action_mean.copy_(self._format_stat(action_mean).to(self.action_mean.device))
        self.action_std.copy_(
            self._format_stat(action_std).to(self.action_std.device).clamp_min(self.action_eps)
        )

    def normalize_actions(self, actions: Tensor) -> Tensor:
        return (actions - self.action_mean.to(dtype=actions.dtype)) / self.action_std.to(dtype=actions.dtype)

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        return actions * self.action_std.to(dtype=actions.dtype) + self.action_mean.to(dtype=actions.dtype)

    def forward(self, cond: Tensor) -> Tensor:
        """Predict a normalized action chunk from the conditioning vector."""
        if cond.ndim != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}), got {tuple(cond.shape)}")
        batch_size = cond.shape[0]
        return self.net(cond).reshape(batch_size, self.horizon, self.action_dim)

    def loss(
        self,
        actions: Tensor,
        cond: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._validate_action_shape(actions, "actions")
        if cond.shape != (actions.shape[0], self.cond_dim):
            raise ValueError(f"cond must have shape {(actions.shape[0], self.cond_dim)}, got {tuple(cond.shape)}")

        target = self.normalize_actions(actions)
        prediction = self.forward(cond)

        squared_error = (prediction - target).pow(2)
        if action_is_pad is None:
            loss = squared_error.mean()
        else:
            if action_is_pad.shape != actions.shape[:2]:
                raise ValueError(
                    f"action_is_pad must have shape {tuple(actions.shape[:2])}, got {tuple(action_is_pad.shape)}"
                )
            valid = (~action_is_pad.bool()).to(dtype=squared_error.dtype, device=squared_error.device)
            denom = (valid.sum() * self.action_dim).clamp_min(1.0)
            loss = (squared_error * valid[..., None]).sum() / denom

        return {
            "loss": loss,
            "mse_loss": loss.detach(),
            "pred_action_norm": prediction.detach().norm(dim=-1).mean(),
            "target_action_norm": target.detach().norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def predict(self, cond: Tensor, denormalize: bool = True) -> Tensor:
        prediction = self.forward(cond)
        return self.denormalize_actions(prediction) if denormalize else prediction

    def _validate_action_shape(self, value: Tensor, name: str) -> None:
        if value.ndim != 3 or value.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"{name} must have shape (B, {self.horizon}, {self.action_dim}), got {tuple(value.shape)}")


def _extract_stat_value(stats: Mapping[str, Any], key: str) -> Tensor | None:
    if key not in stats:
        return None
    value = stats[key]
    if isinstance(value, Tensor):
        return value
    if hasattr(value, "values"):
        return torch.as_tensor(value.values)
    return torch.as_tensor(value)


def _extract_action_stats(dataset_stats: Mapping[str, Any] | None, action_dim: int) -> tuple[Tensor, Tensor]:
    if not dataset_stats:
        return torch.zeros(action_dim), torch.ones(action_dim)

    action_stats = dataset_stats.get(ACTION) or dataset_stats.get("action")
    if isinstance(action_stats, Mapping):
        mean = _extract_stat_value(action_stats, "mean")
        std = _extract_stat_value(action_stats, "std")
        if std is None:
            std = _extract_stat_value(action_stats, "scale")
        if mean is not None and std is not None:
            return mean.float(), std.float()

    return torch.zeros(action_dim), torch.ones(action_dim)


class BCRegressionPolicy(PreTrainedPolicy):
    """LeRobot-style behavior cloning regression policy.

    Interface mirrors :class:`PI0LiteFlowPolicy` so the two methods can plug
    into the same training and evaluation harness.
    """

    config_class = BCRegressionConfig
    name = "bc_regression"

    def __init__(self, config: BCRegressionConfig, dataset_stats: dict[str, Any] | None = None) -> None:
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        action_mean, action_std = _extract_action_stats(dataset_stats, config.action_dim)
        self.decoder = BCRegressionDecoder(
            horizon=config.horizon,
            action_dim=config.action_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            action_mean=action_mean,
            action_std=action_std,
            action_eps=config.action_eps,
        )
        self._cached_action_chunk: Tensor | None = None
        self._cached_action_step = 0

    def reset(self) -> None:
        self._cached_action_chunk = None
        self._cached_action_step = 0

    def get_optim_params(self) -> dict[str, Any]:
        return {"params": self.parameters()}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        actions = batch[ACTION]
        cond = self._get_cond(batch)
        return self.decoder.loss(actions, cond, batch.get("action_is_pad"))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        cond = self._get_cond(batch)
        return self.decoder.predict(cond, denormalize=True)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if self._cached_action_chunk is None or self._cached_action_step >= self.config.n_action_steps:
            self._cached_action_chunk = self.predict_action_chunk(batch, **kwargs)
            self._cached_action_step = 0
        action = self._cached_action_chunk[:, self._cached_action_step]
        self._cached_action_step += 1
        return action

    def _get_cond(self, batch: dict[str, Tensor]) -> Tensor:
        if COND_KEY in batch:
            cond = batch[COND_KEY]
        elif "cond" in batch:
            cond = batch["cond"]
        else:
            raise KeyError(f"Expected conditioning vector at batch['{COND_KEY}']")
        if cond.ndim != 2 or cond.shape[-1] != self.config.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.config.cond_dim}), got {tuple(cond.shape)}")
        return cond
