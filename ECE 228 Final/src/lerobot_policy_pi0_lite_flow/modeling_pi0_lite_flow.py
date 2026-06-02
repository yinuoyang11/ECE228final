from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .configuration_pi0_lite_flow import PI0LiteFlowConfig

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION
except Exception:  # pragma: no cover - exercised when LeRobot is unavailable locally.
    ACTION = "action"

    class PreTrainedPolicy(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()


COND_KEY = "observation.cond"


def _sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.ndim != 1:
        raise ValueError(f"t must have shape (B,) or (B, 1), got {tuple(t.shape)}")

    half_dim = dim // 2
    if half_dim == 0:
        return t[:, None]

    exponent = -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype)
    exponent = exponent / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ResidualFiLMBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + gamma) + beta
        return x + self.net(h)


class MLPFiLMVelocityNet(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.time_dim = hidden_dim
        self.input = nn.Linear(horizon * action_dim, hidden_dim)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualFiLMBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon * action_dim))

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        batch_size = x_t.shape[0]
        h = self.input(x_t.reshape(batch_size, self.horizon * self.action_dim))
        time_emb = _sinusoidal_time_embedding(t, self.time_dim).to(dtype=cond.dtype)
        cond_emb = self.cond(torch.cat([cond, time_emb], dim=-1))
        for block in self.blocks:
            h = block(h, cond_emb)
        velocity = self.output(h).reshape(batch_size, self.horizon, self.action_dim)
        return velocity


class TemporalTransformerVelocityNet(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.time_dim = hidden_dim
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        time_emb = _sinusoidal_time_embedding(t, self.time_dim).to(dtype=cond.dtype)
        cond_emb = self.cond(torch.cat([cond, time_emb], dim=-1))[:, None, :]
        tokens = self.action_in(x_t) + self.pos + cond_emb
        return self.head(self.encoder(tokens))


class CFMActionDecoder(nn.Module):
    """Conditional flow matching decoder for continuous action chunks."""

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 7,
        cond_dim: int = 256,
        architecture: str = "mlp_film",
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 3,
        transformer_dim_feedforward: int = 1024,
        action_mean: Tensor | None = None,
        action_std: Tensor | None = None,
        action_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.architecture = architecture
        self.action_eps = action_eps

        if architecture == "mlp_film":
            self.velocity_net = MLPFiLMVelocityNet(
                horizon=horizon,
                action_dim=action_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif architecture == "temporal_transformer":
            self.velocity_net = TemporalTransformerVelocityNet(
                horizon=horizon,
                action_dim=action_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

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

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        self._validate_action_shape(x_t, "x_t")
        if cond.shape != (x_t.shape[0], self.cond_dim):
            raise ValueError(f"cond must have shape {(x_t.shape[0], self.cond_dim)}, got {tuple(cond.shape)}")
        return self.velocity_net(x_t, t, cond)

    def loss(self, actions: Tensor, cond: Tensor, action_is_pad: Tensor | None = None) -> dict[str, Tensor]:
        self._validate_action_shape(actions, "actions")
        if cond.shape != (actions.shape[0], self.cond_dim):
            raise ValueError(f"cond must have shape {(actions.shape[0], self.cond_dim)}, got {tuple(cond.shape)}")

        x1 = self.normalize_actions(actions)
        x0 = torch.randn_like(x1)
        t = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t_view = t.reshape(-1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        pred_velocity = self.forward(x_t, t, cond)

        squared_error = (pred_velocity - target_velocity).pow(2)
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
            "fm_loss": loss.detach(),
            "pred_velocity_norm": pred_velocity.detach().norm(dim=-1).mean(),
            "target_velocity_norm": target_velocity.detach().norm(dim=-1).mean(),
        }

    def sample(
        self,
        cond: Tensor,
        num_steps: int = 8,
        generator: torch.Generator | None = None,
        denormalize: bool = True,
    ) -> Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}), got {tuple(cond.shape)}")

        batch_size = cond.shape[0]
        x = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=cond.device,
            dtype=cond.dtype,
            generator=generator,
        )
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=cond.device, dtype=cond.dtype)
            x = x + dt * self.forward(x, t, cond)
        return self.denormalize_actions(x) if denormalize else x

    def _validate_action_shape(self, value: Tensor, name: str) -> None:
        expected = (value.shape[0], self.horizon, self.action_dim)
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


class PI0LiteFlowPolicy(PreTrainedPolicy):
    config_class = PI0LiteFlowConfig
    name = "pi0_lite_flow"

    def __init__(self, config: PI0LiteFlowConfig, dataset_stats: dict[str, Any] | None = None) -> None:
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        action_mean, action_std = _extract_action_stats(dataset_stats, config.action_dim)
        self.decoder = CFMActionDecoder(
            horizon=config.horizon,
            action_dim=config.action_dim,
            cond_dim=config.cond_dim,
            architecture=config.decoder_arch,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            transformer_nhead=config.transformer_nhead,
            transformer_num_layers=config.transformer_num_layers,
            transformer_dim_feedforward=config.transformer_dim_feedforward,
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
        num_steps = int(kwargs.get("num_steps", self.config.inference_steps))
        return self.decoder.sample(cond, num_steps=num_steps, generator=kwargs.get("generator"))

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
