from __future__ import annotations

from typing import Any

import torch


class IdentityProcessorPipeline:
    """Small fallback used when LeRobot processor classes are unavailable."""

    def __call__(self, value):
        return value


def make_pi0_lite_flow_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Return no-op processors for a precomputed-condition policy.

    The decoder expects `batch["observation.cond"]` and internally normalizes
    actions, so the package does not need observation or action processors for
    the first implementation phase.
    """

    try:
        from lerobot.processor import PolicyProcessorPipeline

        return PolicyProcessorPipeline([]), PolicyProcessorPipeline([])
    except Exception:  # pragma: no cover - local fallback without LeRobot.
        return IdentityProcessorPipeline(), IdentityProcessorPipeline()
