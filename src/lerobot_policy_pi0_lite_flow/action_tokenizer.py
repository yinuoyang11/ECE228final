"""Action tokenizer for discretizing continuous actions into discrete bins.

Follows the OpenVLA approach: each action dimension is independently binned
into a fixed number of uniform bins. Token IDs map to bin centers for decoding.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionTokenizer(nn.Module):
    """Discretizes continuous actions into tokens and decodes them back.

    Each of the ``action_dim`` dimensions is independently binned into
    ``num_bins`` uniform bins.  The bin boundaries span from per-dimension
    ``action_min`` to ``action_max`` (set from data or manually).

    Encoding maps a continuous value to the nearest bin index (clipped to
    ``[0, num_bins - 1]``).  Decoding maps a bin index back to the center
    of that bin.
    """

    def __init__(
        self,
        num_bins: int = 256,
        action_dim: int = 7,
        horizon: int = 16,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.horizon = horizon

        # Per-dimension bounds — default to [-1, 1].
        self.register_buffer("action_min", -torch.ones(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))

    # ------------------------------------------------------------------
    # Bound management
    # ------------------------------------------------------------------

    def set_bounds(self, action_min: Tensor, action_max: Tensor) -> None:
        """Set per-dimension bounds from tensors of shape ``(action_dim,)``."""
        if action_min.shape != (self.action_dim,):
            raise ValueError(f"action_min must have shape ({self.action_dim},), got {tuple(action_min.shape)}")
        if action_max.shape != (self.action_dim,):
            raise ValueError(f"action_max must have shape ({self.action_dim},), got {tuple(action_max.shape)}")
        self.action_min.copy_(action_min.detach().float())
        self.action_max.copy_(action_max.detach().float())

    def set_bounds_from_data(self, actions: Tensor) -> None:
        """Compute bounds from data using the 1st and 99th percentiles.

        Args:
            actions: Tensor of shape ``(N, ..., action_dim)`` where the last
                dimension is the action dimension.  All leading dimensions are
                flattened before computing percentiles.
        """
        flat = actions.reshape(-1, self.action_dim).float()
        lo = torch.quantile(flat, 0.01, dim=0)
        hi = torch.quantile(flat, 0.99, dim=0)
        # Ensure min < max (add small epsilon if equal).
        eps = 1e-6
        hi = torch.where(hi - lo < eps, lo + eps, hi)
        self.set_bounds(lo, hi)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, actions: Tensor) -> Tensor:
        """Bin continuous actions to token IDs.

        Args:
            actions: ``(B, H, action_dim)`` continuous actions.

        Returns:
            ``(B, H * action_dim)`` token IDs (int64), values in
            ``[0, num_bins - 1]``.
        """
        B, H, D = actions.shape
        if D != self.action_dim:
            raise ValueError(f"Last dim must be {self.action_dim}, got {D}")

        lo = self.action_min.to(actions.device)  # (D,)
        hi = self.action_max.to(actions.device)  # (D,)

        # Normalize to [0, 1] then scale to [0, num_bins - 1].
        normalized = (actions - lo) / (hi - lo)  # (B, H, D)
        bin_ids = (normalized * (self.num_bins - 1)).round().long()
        bin_ids = bin_ids.clamp(0, self.num_bins - 1)

        return bin_ids.reshape(B, H * D)

    def decode(self, token_ids: Tensor, horizon: int | None = None) -> Tensor:
        """Map token IDs back to continuous action values (bin centers).

        Args:
            token_ids: ``(B, H * action_dim)`` token IDs.
            horizon: Override horizon (defaults to ``self.horizon``).

        Returns:
            ``(B, H, action_dim)`` continuous action values.
        """
        H = horizon if horizon is not None else self.horizon
        B = token_ids.shape[0]
        seq_len = token_ids.shape[1]
        if seq_len != H * self.action_dim:
            raise ValueError(
                f"Expected seq_len={H * self.action_dim}, got {seq_len}"
            )

        lo = self.action_min.to(token_ids.device).float()  # (D,)
        hi = self.action_max.to(token_ids.device).float()  # (D,)

        # Reshape to (B, H, D)
        bin_ids = token_ids.reshape(B, H, self.action_dim).float()

        # Map bin index to bin center in [lo, hi].
        normalized = bin_ids / max(self.num_bins - 1, 1)
        actions = normalized * (hi - lo) + lo

        return actions

    def round_trip_error(self, actions: Tensor) -> Tensor:
        """Compute the encode→decode round-trip error (for diagnostics).

        Returns the L1 error per sample, shape ``(B,)``.
        """
        token_ids = self.encode(actions)
        reconstructed = self.decode(token_ids, horizon=actions.shape[1])
        return (actions - reconstructed).abs().mean(dim=(1, 2))

    @property
    def bin_width(self) -> Tensor:
        """Per-dimension bin width, shape ``(action_dim,)``."""
        return (self.action_max - self.action_min) / max(self.num_bins - 1, 1)
