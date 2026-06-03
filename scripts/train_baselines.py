#!/usr/bin/env python3
"""Train and compare all three action-generation baselines.

Usage:
    PYTHONPATH=src python scripts/train_baselines.py [OPTIONS]

Options:
    --epochs        Number of training epochs (default: 200)
    --batch_size    Batch size (default: 64)
    --horizon       Action chunk horizon (default: 16)
    --action_dim    Action dimensionality (default: 7)
    --cond_dim      Conditioning vector dim (default: 256)
    --num_bins      Number of bins for autoregressive (default: 256)
    --lr            Learning rate (default: 1e-3)
    --hidden_dim    Hidden dimension for all models (default: 128)
    --seed          Random seed (default: 42)
    --save_dir      Directory to save checkpoints (default: checkpoints/)
    --device        Device to train on (default: auto-detect)

This script generates synthetic demonstration data and trains three
policies with identical data:
  1. Flow matching (conditional flow matching action decoder)
  2. Behavior cloning regression (MSE baseline)
  3. Discretized autoregressive action-token model (cross-entropy baseline)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Add src to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_policy_pi0_lite_flow.action_tokenizer import ActionTokenizer
from lerobot_policy_pi0_lite_flow.configuration_autoregressive import (
    PI0LiteAutoregressiveConfig,
)
from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig
from lerobot_policy_pi0_lite_flow.modeling_autoregressive import AutoregressivePolicy
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import PI0LiteFlowPolicy


# ---------------------------------------------------------------------------
# Behavior Cloning Regression Baseline
# ---------------------------------------------------------------------------


class BCRegressionNet(nn.Module):
    """Behavior cloning regression model (MSE loss).

    Given a conditioning vector, directly predicts continuous action chunks
    using an MLP with residual blocks.
    """

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 7,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        layers = [nn.Linear(cond_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: Tensor) -> Tensor:
        """Predict action chunk from conditioning vector.

        Args:
            cond: (B, cond_dim)

        Returns:
            (B, horizon, action_dim)
        """
        return self.net(cond).reshape(-1, self.horizon, self.action_dim)

    def loss(self, actions: Tensor, cond: Tensor) -> dict[str, Tensor]:
        pred = self.forward(cond)
        mse = F.mse_loss(pred, actions)
        return {"loss": mse, "mse_loss": mse.detach()}

    @torch.no_grad()
    def predict(self, cond: Tensor) -> Tensor:
        return self.forward(cond)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    num_samples: int,
    horizon: int,
    action_dim: int,
    cond_dim: int,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Generate synthetic demonstrations with structured correlations.

    Creates conditioning vectors and corresponding action chunks where
    the actions depend on the conditioning in a learnable way:
      - A random linear map from cond to "target" action
      - Smooth temporal structure (actions change slowly over horizon)
      - Some noise to make it realistic
    """
    torch.manual_seed(seed)

    # Random projection from cond_dim to action_dim.
    W = torch.randn(cond_dim, action_dim) * 0.1
    b = torch.randn(action_dim) * 0.5

    # Generate conditioning vectors.
    cond = torch.randn(num_samples, cond_dim)

    # Generate "target" actions from cond.
    base_action = cond @ W + b  # (N, action_dim)

    # Create smooth action chunks with temporal structure.
    t = torch.linspace(0, 1, horizon).unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
    freq = torch.randn(num_samples, 1, action_dim) * 2  # Random frequency per sample
    phase = torch.randn(num_samples, 1, action_dim) * math.pi

    actions = (
        base_action.unsqueeze(1)
        + 0.3 * torch.sin(2 * math.pi * freq * t + phase)
        + 0.05 * torch.randn(num_samples, horizon, action_dim)
    )

    return cond, actions


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    method: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.  Returns average metrics."""
    model.train()
    total_loss = 0.0
    total_extra = {}
    num_batches = 0

    for cond_batch, action_batch in dataloader:
        cond_batch = cond_batch.to(device)
        action_batch = action_batch.to(device)
        optimizer.zero_grad()

        if method == "flow":
            batch = {"action": action_batch, "observation.cond": cond_batch}
            output = model.forward(batch)
        elif method == "autoregressive":
            batch = {"action": action_batch, "observation.cond": cond_batch}
            output = model.forward(batch)
        elif method == "regression":
            output = model.loss(action_batch, cond_batch)
        else:
            raise ValueError(f"Unknown method: {method}")

        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += output["loss"].item()
        for k, v in output.items():
            if k != "loss" and isinstance(v, Tensor):
                total_extra[k] = total_extra.get(k, 0.0) + v.item()
        num_batches += 1

    avg = {"loss": total_loss / num_batches}
    for k, v in total_extra.items():
        avg[k] = v / num_batches
    return avg


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    method: str,
    cond: Tensor,
    actions: Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate action prediction quality."""
    model.eval()
    cond = cond.to(device)
    actions = actions.to(device)

    batch = {"observation.cond": cond}

    # Predict action chunks.
    t0 = time.time()
    if method == "flow":
        pred = model.predict_action_chunk(batch)
    elif method == "autoregressive":
        pred = model.predict_action_chunk(batch)
    elif method == "regression":
        pred = model.predict(cond)
    else:
        raise ValueError(f"Unknown method: {method}")
    inference_time = (time.time() - t0) / max(cond.shape[0], 1)

    # Metrics.
    mse = F.mse_loss(pred, actions).item()
    l1 = F.l1_loss(pred, actions).item()

    # Action smoothness: average jerk (3rd derivative approximation).
    if pred.shape[1] >= 3:
        jerk = torch.diff(pred, n=2, dim=1).abs().mean().item()
    else:
        jerk = 0.0

    return {
        "mse": mse,
        "l1": l1,
        "jerk": jerk,
        "inference_time_per_sample": inference_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train all three baselines")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_eval", type=int, default=500)
    args = parser.parse_args()

    # Device.
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate data.
    print("Generating synthetic demonstration data...")
    train_cond, train_actions = generate_synthetic_data(
        args.num_train, args.horizon, args.action_dim, args.cond_dim, seed=args.seed
    )
    eval_cond, eval_actions = generate_synthetic_data(
        args.num_eval, args.horizon, args.action_dim, args.cond_dim, seed=args.seed + 1
    )

    train_dataset = TensorDataset(train_cond, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Dataset stats for tokenizer bounds.
    action_min = train_actions.reshape(-1, args.action_dim).min(dim=0).values
    action_max = train_actions.reshape(-1, args.action_dim).max(dim=0).values
    action_mean = train_actions.reshape(-1, args.action_dim).mean(dim=0)
    action_std = train_actions.reshape(-1, args.action_dim).std(dim=0)

    dataset_stats = {
        "action": {
            "min": action_min,
            "max": action_max,
            "mean": action_mean,
            "std": action_std,
        }
    }

    # --------------- Build models ----------------

    # 1. Flow matching.
    flow_config = PI0LiteFlowConfig(
        horizon=args.horizon,
        action_dim=args.action_dim,
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim,
        num_layers=4,
        inference_steps=8,
    )
    flow_stats = {"action": {"mean": action_mean, "std": action_std}}
    flow_policy = PI0LiteFlowPolicy(flow_config, dataset_stats=flow_stats).to(device)

    # 2. Autoregressive.
    ar_config = PI0LiteAutoregressiveConfig(
        horizon=args.horizon,
        action_dim=args.action_dim,
        cond_dim=args.cond_dim,
        num_bins=args.num_bins,
        hidden_dim=args.hidden_dim,
        nhead=4,
        num_layers=4,
        dim_feedforward=args.hidden_dim * 4,
        dropout=0.1,
    )
    ar_policy = AutoregressivePolicy(ar_config, dataset_stats=dataset_stats).to(device)

    # 3. BC Regression.
    bc_model = BCRegressionNet(
        horizon=args.horizon,
        action_dim=args.action_dim,
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim,
        num_layers=4,
    ).to(device)

    models = {
        "flow": (flow_policy, "flow"),
        "autoregressive": (ar_policy, "autoregressive"),
        "regression": (bc_model, "regression"),
    }

    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for name, (model, _) in models.items()
    }

    # Print model sizes.
    print("\n" + "=" * 60)
    print("Model sizes:")
    for name, (model, _) in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} parameters")
    print("=" * 60 + "\n")

    # --------------- Training ----------------

    history = {name: [] for name in models}

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        for name, (model, method) in models.items():
            metrics = train_one_epoch(model, method, train_loader, optimizers[name], device)
            history[name].append(metrics)

        if epoch % 20 == 0 or epoch == 1:
            losses = {n: history[n][-1]["loss"] for n in models}
            loss_str = " | ".join(f"{n}: {v:.4f}" for n, v in losses.items())
            print(f"Epoch {epoch:4d} | {loss_str}")

    # --------------- Evaluation ----------------

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)

    eval_results = {}
    for name, (model, method) in models.items():
        metrics = evaluate_model(model, method, eval_cond, eval_actions, device)
        eval_results[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    # --------------- Save ----------------

    # Save checkpoints.
    for name, (model, _) in models.items():
        torch.save(model.state_dict(), save_dir / f"{name}_model.pt")

    # Save training history and eval results.
    results = {
        "args": vars(args),
        "eval_results": eval_results,
        "training_history": {
            name: [{"epoch": i + 1, **m} for i, m in enumerate(hist)]
            for name, hist in history.items()
        },
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCheckpoints and results saved to {save_dir}/")

    # --------------- Summary Table ----------------

    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'MSE':>10} {'L1':>10} {'Jerk':>10} {'Inf Time':>12}")
    print("-" * 60)
    for name, metrics in eval_results.items():
        print(
            f"{name:<20} {metrics['mse']:>10.6f} {metrics['l1']:>10.6f} "
            f"{metrics['jerk']:>10.6f} {metrics['inference_time_per_sample']:>10.4f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
