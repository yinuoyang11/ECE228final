"""Discretized autoregressive action-token model baseline.

This module implements a causal Transformer that predicts discretized action
tokens autoregressively, following the OpenVLA paradigm.  Each continuous action
dimension is binned into ``num_bins`` discrete tokens, and the model is trained
with standard cross-entropy (next-token prediction) loss.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .action_tokenizer import ActionTokenizer
from .configuration_autoregressive import PI0LiteAutoregressiveConfig

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION
except Exception:  # pragma: no cover - exercised when LeRobot is unavailable locally.
    ACTION = "action"

    class PreTrainedPolicy(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()


COND_KEY = "observation.cond"


# ---------------------------------------------------------------------------
# Autoregressive decoder
# ---------------------------------------------------------------------------


class AutoregressiveActionDecoder(nn.Module):
    """Causal Transformer that autoregressively predicts discretized action tokens.

    Architecture (OpenVLA-inspired, lightweight):

    1. The conditioning vector is projected and used as the first "start" token.
    2. Each action token is embedded and summed with a learned positional embedding.
    3. A causal Transformer processes the full sequence.
    4. A linear head produces logits over ``num_bins`` for each position.

    During training, teacher forcing is used.  During inference, tokens are
    sampled one at a time (greedy or with temperature).

    The total sequence length is ``1 + horizon * action_dim`` (condition token
    plus all action tokens).  The model predicts token ``i+1`` from positions
    ``0..i``.
    """

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 7,
        cond_dim: int = 256,
        num_bins: int = 256,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim

        # Total number of action tokens in a chunk.
        self.seq_len = horizon * action_dim

        # Token embedding for discrete action bins.
        self.token_embedding = nn.Embedding(num_bins, hidden_dim)

        # Positional embedding: position 0 = condition, 1..seq_len = action tokens.
        self.pos_embedding = nn.Embedding(self.seq_len + 1, hidden_dim)

        # Project the condition vector to the transformer hidden dim.
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Causal Transformer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head: predict logits over num_bins.
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_bins),
        )

        # Initialise weights.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply small-normal initialisation (GPT-style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create an upper-triangular causal attention mask.

        Returns:
            ``(seq_len, seq_len)`` float tensor with ``-inf`` above the diagonal
            and ``0`` on and below.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, token_ids: Tensor, cond: Tensor) -> Tensor:
        """Run the model in teacher-forcing mode.

        Args:
            token_ids: ``(B, seq_len)`` int64 action token IDs.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            ``(B, seq_len + 1, num_bins)`` logits for each position.
            Position ``i`` predicts the token at position ``i`` (i.e. the
            first output position predicts ``token_ids[:, 0]``).
        """
        B = token_ids.shape[0]
        device = token_ids.device

        # Embed condition → (B, 1, hidden_dim).
        cond_token = self.cond_proj(cond).unsqueeze(1)

        # Embed action tokens → (B, seq_len, hidden_dim).
        action_tokens = self.token_embedding(token_ids)

        # Concatenate: [cond_token, action_tokens] → (B, seq_len + 1, hidden_dim).
        full_seq = torch.cat([cond_token, action_tokens], dim=1)

        # Add positional embeddings.
        positions = torch.arange(full_seq.shape[1], device=device)
        full_seq = full_seq + self.pos_embedding(positions)

        # Apply causal mask and run transformer.
        mask = self._causal_mask(full_seq.shape[1], device)
        hidden = self.transformer(full_seq, mask=mask)

        # Predict logits at every position.
        logits = self.output_head(hidden)  # (B, seq_len + 1, num_bins)

        return logits

    def loss(self, token_ids: Tensor, cond: Tensor) -> dict[str, Tensor]:
        """Compute teacher-forced cross-entropy loss.

        The model sees ``[cond, tok_0, ..., tok_{n-2}]`` and predicts
        ``[tok_0, tok_1, ..., tok_{n-1}]``.

        Args:
            token_ids: ``(B, seq_len)`` target token IDs.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            Dict with keys ``"loss"`` (scalar), ``"ce_loss"`` (detached),
            ``"accuracy"`` (detached, token-level accuracy).
        """
        logits = self.forward(token_ids, cond)  # (B, seq_len + 1, num_bins)

        # The targets for the first seq_len output positions.
        # logits[:, 0] should predict token_ids[:, 0] (output of cond_token position)
        # logits[:, i] should predict token_ids[:, i] for i in [0, seq_len-1]
        # We discard logits[:, -1] (nothing to predict after last token).
        pred_logits = logits[:, :-1]  # (B, seq_len, num_bins)
        targets = token_ids  # (B, seq_len)

        # Flatten for cross-entropy.
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, self.num_bins),
            targets.reshape(-1),
        )

        # Token-level accuracy.
        with torch.no_grad():
            predicted_tokens = pred_logits.argmax(dim=-1)  # (B, seq_len)
            accuracy = (predicted_tokens == targets).float().mean()

        return {
            "loss": ce_loss,
            "ce_loss": ce_loss.detach(),
            "accuracy": accuracy,
        }

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively sample action tokens.

        Args:
            cond: ``(B, cond_dim)`` conditioning vector.
            temperature: Sampling temperature.  ``0`` = greedy argmax.
            generator: Optional RNG for reproducibility.

        Returns:
            ``(B, seq_len)`` sampled token IDs (int64).
        """
        B = cond.shape[0]
        device = cond.device

        # Start with just the condition token.
        cond_token = self.cond_proj(cond).unsqueeze(1)  # (B, 1, hidden_dim)
        generated = torch.zeros(B, 0, dtype=torch.long, device=device)

        for step in range(self.seq_len):
            # Build current sequence.
            if step == 0:
                tokens_seq = cond_token  # (B, 1, hidden_dim)
            else:
                action_emb = self.token_embedding(generated)  # (B, step, hidden_dim)
                tokens_seq = torch.cat([cond_token, action_emb], dim=1)

            # Add positions.
            positions = torch.arange(tokens_seq.shape[1], device=device)
            tokens_seq = tokens_seq + self.pos_embedding(positions)

            # Causal mask.
            mask = self._causal_mask(tokens_seq.shape[1], device)
            hidden = self.transformer(tokens_seq, mask=mask)

            # Get logits for the last position.
            logits = self.output_head(hidden[:, -1])  # (B, num_bins)

            if temperature <= 0.0:
                # Greedy.
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(
                    probs, num_samples=1, generator=generator
                )  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated  # (B, seq_len)


# ---------------------------------------------------------------------------
# Dataset stats extraction
# ---------------------------------------------------------------------------


def _extract_stat_value(stats: Mapping[str, Any], key: str) -> Tensor | None:
    if key not in stats:
        return None
    value = stats[key]
    if isinstance(value, Tensor):
        return value
    if hasattr(value, "values"):
        return torch.as_tensor(value.values)
    return torch.as_tensor(value)


def _extract_action_bounds(
    dataset_stats: Mapping[str, Any] | None, action_dim: int
) -> tuple[Tensor | None, Tensor | None]:
    """Extract min/max action bounds from dataset stats.

    Falls back to mean ± 3*std if min/max are not available.
    """
    if not dataset_stats:
        return None, None

    action_stats = dataset_stats.get(ACTION) or dataset_stats.get("action")
    if isinstance(action_stats, Mapping):
        min_val = _extract_stat_value(action_stats, "min")
        max_val = _extract_stat_value(action_stats, "max")
        if min_val is not None and max_val is not None:
            return min_val.float(), max_val.float()
        # Fallback: use mean ± 3*std.
        mean = _extract_stat_value(action_stats, "mean")
        std = _extract_stat_value(action_stats, "std")
        if mean is not None and std is not None:
            return (mean - 3 * std).float(), (mean + 3 * std).float()
    return None, None


# ---------------------------------------------------------------------------
# LeRobot policy wrapper
# ---------------------------------------------------------------------------


class AutoregressivePolicy(PreTrainedPolicy):
    """Discretized autoregressive action-token policy (LeRobot interface).

    This policy tokenises continuous action chunks into discrete bins and
    trains a causal Transformer with cross-entropy loss.  At inference it
    samples tokens autoregressively and decodes them back to continuous
    actions.

    Interface is identical to ``PI0LiteFlowPolicy``.
    """

    config_class = PI0LiteAutoregressiveConfig
    name = "pi0_lite_autoregressive"

    def __init__(
        self,
        config: PI0LiteAutoregressiveConfig,
        dataset_stats: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

        # Action tokenizer.
        self.tokenizer = ActionTokenizer(
            num_bins=config.num_bins,
            action_dim=config.action_dim,
            horizon=config.horizon,
        )

        # Set tokenizer bounds from dataset stats if available.
        action_min, action_max = _extract_action_bounds(
            dataset_stats, config.action_dim
        )
        if action_min is not None and action_max is not None:
            self.tokenizer.set_bounds(action_min, action_max)

        # Autoregressive decoder.
        self.decoder = AutoregressiveActionDecoder(
            horizon=config.horizon,
            action_dim=config.action_dim,
            cond_dim=config.cond_dim,
            num_bins=config.num_bins,
            hidden_dim=config.hidden_dim,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Action chunk cache for multi-step execution.
        self._cached_action_chunk: Tensor | None = None
        self._cached_action_step = 0

    def reset(self) -> None:
        """Reset the action chunk cache (call at episode start)."""
        self._cached_action_chunk = None
        self._cached_action_step = 0

    def get_optim_params(self) -> dict[str, Any]:
        return {"params": self.parameters()}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the training loss (teacher-forced cross-entropy).

        Args:
            batch: Must contain ``ACTION`` (continuous actions, ``(B, H, D)``)
                and ``COND_KEY`` (conditioning vector, ``(B, cond_dim)``).

        Returns:
            Dict with ``"loss"``, ``"ce_loss"``, and ``"accuracy"`` keys.
        """
        actions = batch[ACTION]  # (B, H, D)
        cond = self._get_cond(batch)  # (B, cond_dim)

        # Tokenize actions.
        token_ids = self.tokenizer.encode(actions)  # (B, H * D)

        return self.decoder.loss(token_ids, cond)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs
    ) -> Tensor:
        """Predict a full action chunk via autoregressive sampling.

        Returns:
            ``(B, horizon, action_dim)`` continuous actions.
        """
        cond = self._get_cond(batch)
        temperature = kwargs.get("temperature", self.config.inference_temperature)
        generator = kwargs.get("generator", None)

        # Sample token IDs.
        token_ids = self.decoder.sample(
            cond, temperature=temperature, generator=generator
        )  # (B, H * D)

        # Decode back to continuous actions.
        return self.tokenizer.decode(token_ids)  # (B, H, D)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs
    ) -> Tensor:
        """Select a single action (with chunk caching).

        Returns:
            ``(B, action_dim)`` single action for the current step.
        """
        if (
            self._cached_action_chunk is None
            or self._cached_action_step >= self.config.n_action_steps
        ):
            self._cached_action_chunk = self.predict_action_chunk(batch, **kwargs)
            self._cached_action_step = 0

        action = self._cached_action_chunk[:, self._cached_action_step]
        self._cached_action_step += 1
        return action

    def _get_cond(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract the conditioning vector from the batch."""
        if COND_KEY in batch:
            cond = batch[COND_KEY]
        elif "cond" in batch:
            cond = batch["cond"]
        else:
            raise KeyError(f"Expected conditioning vector at batch['{COND_KEY}']")
        if cond.ndim != 2 or cond.shape[-1] != self.config.cond_dim:
            raise ValueError(
                f"cond must have shape (B, {self.config.cond_dim}), got {tuple(cond.shape)}"
            )
        return cond
