#!/usr/bin/env python3
"""Train the frozen-CLIP discretized autoregressive LIBERO baseline."""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_policy_pi0_lite_flow.autoregressive_experiment import (  # noqa: E402
    ACTION,
    COND_KEY,
    build_encoder,
    build_policy,
    cache_frozen_features,
    cached_batch_to_device,
    compute_action_bounds,
    condition_from_cached,
    load_libero_action_dataset,
    resolve_device,
    save_json,
    split_by_episode,
    tensor_dict_to_lists,
    trainable_encoder_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--task-indices", type=int, nargs="+", default=[20])
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--encoder", choices=("clip", "mock"), default="clip")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/autoregressive"))
    parser.add_argument("--run-name", default="ar_clip_task20_3000")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    step: int,
    args_dict: dict[str, Any],
    split: dict[str, list[int]],
    action_stats: dict[str, torch.Tensor],
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "step": step,
            "args": args_dict,
            "split": split,
            "action_stats": tensor_dict_to_lists(action_stats),
            "encoder_state_dict": trainable_encoder_state_dict(encoder),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args_dict = vars(args).copy()
    args_dict["device"] = str(device)
    args_dict["output_dir"] = str(args.output_dir)

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    print("Loading LIBERO dataset and task metadata...", flush=True)
    raw_dataset, dataset, task_descriptions = load_libero_action_dataset(
        args.repo_id,
        args.horizon,
        args.task_indices,
    )
    missing_descriptions = [index for index in args.task_indices if not task_descriptions.get(index)]
    if missing_descriptions:
        raise RuntimeError(f"Missing task descriptions for task indices: {missing_descriptions}")

    split = split_by_episode(dataset, args.eval_fraction, args.seed)
    train_indices = split["train_indices"]
    eval_indices = split["eval_indices"]
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_indices = eval_indices[: args.max_eval_samples]
    if not train_indices or not eval_indices:
        raise RuntimeError("Train and eval subsets must both contain samples")

    args_dict["task_descriptions"] = {
        str(index): task_descriptions[index] for index in args.task_indices
    }
    args_dict["train_samples"] = len(train_indices)
    args_dict["eval_samples"] = len(eval_indices)
    args_dict["train_episode_ids"] = split["train_episode_ids"]
    args_dict["eval_episode_ids"] = split["eval_episode_ids"]
    save_json(run_dir / "args.json", args_dict)
    print(
        f"Selected {len(dataset)} frames from {len(split['train_episode_ids']) + len(split['eval_episode_ids'])} episodes; "
        f"train={len(train_indices)}, eval={len(eval_indices)}",
        flush=True,
    )
    print(f"Instruction: {task_descriptions[args.task_indices[0]]}", flush=True)

    print("Computing train-split action tokenizer bounds without decoding images...", flush=True)
    action_stats = compute_action_bounds(raw_dataset, split["train_raw_indices"], args.action_dim)

    print(f"Building {args.encoder} representation encoder and autoregressive policy...", flush=True)
    encoder = build_encoder(args_dict).to(device)
    policy = build_policy(args_dict, action_stats).to(device)

    print("Precomputing frozen image/text features...", flush=True)
    cache_start = time.perf_counter()
    train_cache = cache_frozen_features(
        encoder,
        Subset(dataset, train_indices),
        device,
        args.batch_size,
        args.num_workers,
    )
    eval_cache = cache_frozen_features(
        encoder,
        Subset(dataset, eval_indices),
        device,
        args.batch_size,
        args.num_workers,
    )
    torch.save(train_cache.tensors, run_dir / "train_features.pt")
    torch.save(eval_cache.tensors, run_dir / "eval_features.pt")
    print(f"Feature cache complete in {time.perf_counter() - cache_start:.1f}s", flush=True)

    if args.encoder == "clip":
        encoder.image_encoder.to("cpu")
        encoder.text_encoder.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()

    train_loader = DataLoader(
        train_cache,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    trainable_params = [
        param
        for param in list(encoder.parameters()) + list(policy.parameters())
        if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    trainable_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters: {trainable_count:,}", flush=True)

    history: list[dict[str, float]] = []
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["step", "loss", "ce_loss", "accuracy", "grad_norm"],
            lineterminator="\n",
        )
        writer.writeheader()
        iterator = iter(train_loader)
        encoder.train()
        policy.train()
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = cached_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            cond = condition_from_cached(encoder, batch)
            output = policy(
                {
                    ACTION: batch[ACTION],
                    "action_is_pad": batch["action_is_pad"],
                    COND_KEY: cond,
                }
            )
            output["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            row = {
                "step": step,
                "loss": float(output["loss"].detach().cpu()),
                "ce_loss": float(output["ce_loss"].cpu()),
                "accuracy": float(output["accuracy"].cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            history.append(row)
            writer.writerow(row)
            metrics_file.flush()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"step {step:4d}/{args.steps} | loss {row['loss']:.4f} | "
                    f"token acc {row['accuracy']:.4f} | grad {row['grad_norm']:.3f}",
                    flush=True,
                )
            if step % args.save_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoint_{step:06d}.pt",
                    step,
                    args_dict,
                    split,
                    action_stats,
                    encoder,
                    policy,
                    optimizer,
                    history,
                )

    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        args.steps,
        args_dict,
        split,
        action_stats,
        encoder,
        policy,
        optimizer,
        history,
    )
    print(f"Saved final checkpoint: {run_dir / 'checkpoint_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
