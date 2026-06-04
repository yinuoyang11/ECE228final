#!/usr/bin/env python3
"""Train the Discretized Autoregressive Action-Token Model Baseline on LIBERO Task 20.

Usage:
    PYTHONPATH=src python scripts/train_baselines.py [OPTIONS]
"""
from __future__ import annotations

import argparse
import json
import random
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split

# Add src to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_policy_pi0_lite_flow.representation_encoder import RepresentationEncoder, RepresentationEncoderConfig
from lerobot_policy_pi0_lite_flow.configuration_autoregressive import PI0LiteAutoregressiveConfig
from lerobot_policy_pi0_lite_flow.modeling_autoregressive import AutoregressivePolicy
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset

def collate_libero(batch: list[dict]) -> dict:
    """Custom collate function for LIBEROActionChunkDataset outputs."""
    return {
        'images': torch.stack([b['images'] for b in batch]),       # (B, 2, 3, 224, 224)
        'state': torch.stack([b['state'] for b in batch]),         # (B, 8)
        'instruction': [b['instruction'] for b in batch],          # list of str
        'action': torch.stack([b['action'] for b in batch]),       # (B, H, 7)
    }

def train_one_epoch(
    model: nn.Module,
    encoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    encoder.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        state = batch['state'].to(device)
        action_batch = batch['action'].to(device)
        instructions = batch['instruction']
        
        optimizer.zero_grad()

        encoder_input = {
            "images": images,
            "state": state,
            "instruction": instructions,
        }
        cond_batch = encoder(encoder_input)

        fwd_batch = {"action": action_batch, "observation.cond": cond_batch}
        output = model.forward(fwd_batch)

        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += output["loss"].item()
        if "accuracy" in output:
            total_acc += output["accuracy"].item()
        num_batches += 1

        if num_batches % 20 == 0 or num_batches == 1 or num_batches == len(dataloader):
            print(f"  Batch {num_batches:3d}/{len(dataloader):3d} | Loss: {output['loss'].item():.4f}", flush=True)

    return {
        "loss": total_loss / max(1, num_batches),
        "accuracy": total_acc / max(1, num_batches)
    }

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    encoder: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    encoder.eval()
    
    total_mse = 0.0
    total_l1 = 0.0
    total_samples = 0
    total_inf_time = 0.0
    
    for batch in eval_loader:
        images = batch['images'].to(device)
        state = batch['state'].to(device)
        actions = batch['action'].to(device)
        instructions = batch['instruction']
        batch_size = state.size(0)
        
        encoder_input = {
            "images": images,
            "state": state,
            "instruction": instructions,
        }
        cond = encoder(encoder_input)
        batch_dict = {"observation.cond": cond}
        
        t0 = time.time()
        pred = model.predict_action_chunk(batch_dict)
        inf_time = time.time() - t0
        total_inf_time += inf_time
        
        total_mse += F.mse_loss(pred, actions, reduction='sum').item()
        total_l1 += F.l1_loss(pred, actions, reduction='sum').item()
        total_samples += batch_size

    action_elements = eval_loader.dataset[0]['action'].numel()
    
    return {
        "mse": total_mse / max(1, total_samples * action_elements),
        "l1": total_l1 / max(1, total_samples * action_elements),
        "inference_time_per_sample": total_inf_time / max(1, total_samples),
    }

def main():
    parser = argparse.ArgumentParser(description="Train Autoregressive Baseline on HuggingFaceVLA/libero task 20")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Load LIBERO dataset via teammate's adapter ----------------
    print("\nLoading HuggingFaceVLA/libero dataset using teammate's LIBEROActionChunkDataset...")
    print("Using Task 20 specifically as requested!")
    
    import glob
    import os
    from datasets import Dataset, concatenate_datasets
    
    arrow_pattern = os.path.expanduser("~/.cache/huggingface/datasets/HuggingFaceVLA___libero/default/0.0.0/86958911c0f959db2bbbdb107eb3e17c5f9c798e/*.arrow")
    arrow_files = sorted(glob.glob(arrow_pattern))
    
    if arrow_files:
        print(f"Found {len(arrow_files)} compiled Arrow files in local cache!")
        print("Loading dataset directly from local Arrow files to bypass download...")
        shards = [Dataset.from_file(f) for f in arrow_files]
        hf_ds = concatenate_datasets(shards)
    else:
        print("Local Arrow cache not found. Downloading/loading dataset from HF Hub...")
        from datasets import load_dataset
        hf_ds = load_dataset("HuggingFaceVLA/libero", split="train")
    
    full_dataset = LIBEROActionChunkDataset(
        base_dataset=hf_ds,
        horizon=args.horizon,
        task_indices=[20],
    )
    
    # Train / eval split
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    eval_size = dataset_size - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    print(f"Total Task 20 frames: {dataset_size}")
    print(f"Train samples: {train_size} | Eval samples: {eval_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_libero,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_libero,
    )

    # Note: To avoid iterating over the entire image dataset which is slow, we use a fixed approximation
    # for dataset statistics based on typical LIBERO ranges, or we can just fetch a few batches to estimate it.
    print("Estimating action statistics from the first few batches...")
    all_actions = []
    for i, batch in enumerate(train_loader):
        all_actions.append(batch["action"].reshape(-1, args.action_dim))
        if i >= 10:  # use 10 batches (320 samples) to estimate stats
            break
            
    all_actions = torch.cat(all_actions, dim=0)
    action_min = all_actions.min(dim=0).values
    action_max = all_actions.max(dim=0).values
    action_mean = all_actions.mean(dim=0)
    action_std = all_actions.std(dim=0)
    
    action_max = torch.where(action_max == action_min, action_max + 1e-6, action_max)
    action_std = action_std.clamp_min(1e-6)

    dataset_stats = {
        "action": {
            "min": action_min,
            "max": action_max,
            "mean": action_mean,
            "std": action_std,
        }
    }

    # --------------- Build Autoregressive Model ----------------
    print("\nBuilding Autoregressive Model & Representation Encoder...")
    enc_cfg = RepresentationEncoderConfig(state_dim=args.state_dim, cond_dim=args.cond_dim)
    encoder = RepresentationEncoder(enc_cfg).to(device)

    dummy_batch = {
        "images": torch.zeros(1, 2, 3, 224, 224, device=device),
        "state": torch.zeros(1, args.state_dim, device=device),
        "instruction": ["dummy task description"],
    }
    encoder(dummy_batch)

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
    model = AutoregressivePolicy(ar_config, dataset_stats=dataset_stats).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()), 
        lr=args.lr, weight_decay=1e-4
    )

    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total params: {n_params + enc_params:,} (Decoder: {n_params:,}, Encoder: {enc_params:,})")

    # --------------- Training ----------------
    history = []
    
    if not args.eval_only:
        print(f"Training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                model=model,
                encoder=encoder,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device
            )
            history.append(metrics)
    
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}")
    
        torch.save(model.state_dict(), save_dir / "autoregressive_model.pt")
        torch.save(encoder.state_dict(), save_dir / "autoregressive_encoder.pt")
    else:
        print("Loading checkpoints for evaluation...")
        model.load_state_dict(torch.load(save_dir / "autoregressive_model.pt", map_location=device))
        encoder.load_state_dict(torch.load(save_dir / "autoregressive_encoder.pt", map_location=device))

    print("\n" + "=" * 60)
    print("Evaluation Results on HuggingFaceVLA/libero (Task 20):")
    print("=" * 60)

    metrics = evaluate_model(
        model=model,
        encoder=encoder,
        eval_loader=eval_loader,
        device=device
    )
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"L1: {metrics['l1']:.6f}")
    print(f"Inference Time: {metrics['inference_time_per_sample']:.4f}s")

    results = {
        "args": vars(args),
        "eval_results": {"autoregressive": metrics},
        "training_history": {"autoregressive": [{"epoch": i + 1, **m} for i, m in enumerate(history)]},
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCheckpoints and results saved to {save_dir}/")

if __name__ == "__main__":
    main()
