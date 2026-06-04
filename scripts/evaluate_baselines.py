#!/usr/bin/env python3
"""Evaluate a trained frozen-CLIP autoregressive LIBERO baseline checkpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_policy_pi0_lite_flow.autoregressive_experiment import (  # noqa: E402
    ACTION,
    COND_KEY,
    CachedFeatureDataset,
    build_encoder,
    build_policy,
    cached_batch_to_device,
    collate_raw,
    condition_from_cached,
    load_libero_action_dataset,
    load_trainable_encoder_state_dict,
    masked_action_error_sums,
    resolve_device,
    save_json,
    synchronize,
    tensor_dict_from_lists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/autoregressive_clip_task20_3000"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latency-samples", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def evaluate_quality(
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    dataset: CachedFeatureDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    encoder.eval()
    policy.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mse_sum = 0.0
    l1_sum = 0.0
    action_count = 0
    token_loss_sum = 0.0
    token_correct_sum = 0.0
    token_count = 0

    for batch in loader:
        batch = cached_batch_to_device(batch, device)
        cond = condition_from_cached(encoder, batch)
        actions = batch[ACTION]
        action_is_pad = batch["action_is_pad"]

        output = policy({ACTION: actions, "action_is_pad": action_is_pad, COND_KEY: cond})
        valid_tokens = int((~action_is_pad).sum().item() * actions.shape[-1])
        token_loss_sum += float(output["ce_loss"].cpu()) * valid_tokens
        token_correct_sum += float(output["accuracy"].cpu()) * valid_tokens
        token_count += valid_tokens

        prediction = policy.predict_action_chunk({COND_KEY: cond})
        batch_mse, batch_l1, batch_count = masked_action_error_sums(prediction, actions, action_is_pad)
        mse_sum += float(batch_mse.cpu())
        l1_sum += float(batch_l1.cpu())
        action_count += int(batch_count.item())

    return {
        "mse": mse_sum / max(action_count, 1),
        "l1": l1_sum / max(action_count, 1),
        "token_cross_entropy": token_loss_sum / max(token_count, 1),
        "token_accuracy": token_correct_sum / max(token_count, 1),
        "valid_action_values": action_count,
        "valid_tokens": token_count,
    }


@torch.no_grad()
def measure_decoder_latency(
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    dataset: CachedFeatureDataset,
    device: torch.device,
    samples: int,
) -> dict[str, float]:
    encoder.eval()
    policy.eval()
    durations = []
    limit = min(samples, len(dataset))
    for index in range(limit + 3):
        item = dataset[index % limit]
        batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
        cond = condition_from_cached(encoder, batch)
        synchronize(device)
        start = time.perf_counter()
        policy.predict_action_chunk({COND_KEY: cond})
        synchronize(device)
        if index >= 3:
            durations.append(time.perf_counter() - start)
    return _latency_summary(durations)


@torch.no_grad()
def measure_end_to_end_latency(
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    raw_dataset: Subset,
    device: torch.device,
    samples: int,
) -> dict[str, float]:
    encoder.eval()
    policy.eval()
    durations = []
    limit = min(samples, len(raw_dataset))
    for index in range(limit + 3):
        item = raw_dataset[index % limit]
        batch = collate_raw([item])
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        synchronize(device)
        start = time.perf_counter()
        cond = encoder(
            {
                "images": images,
                "state": state,
                "instruction": batch["instruction"],
            }
        )
        policy.predict_action_chunk({COND_KEY: cond})
        synchronize(device)
        if index >= 3:
            durations.append(time.perf_counter() - start)
    return _latency_summary(durations)


def _latency_summary(durations: list[float]) -> dict[str, float]:
    if not durations:
        raise ValueError("At least one latency sample is required")
    return {
        "mean_ms": statistics.mean(durations) * 1000.0,
        "median_ms": statistics.median(durations) * 1000.0,
        "samples": len(durations),
    }


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(steps, [row["loss"] for row in history], color="#d97706", linewidth=1.5)
    axes[0].set(title="Autoregressive Training Loss", xlabel="Step", ylabel="Cross-Entropy")
    axes[1].plot(steps, [row["accuracy"] for row in history], color="#2563eb", linewidth=1.5)
    axes[1].set(title="Token Prediction Accuracy", xlabel="Step", ylabel="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=220)
    plt.close(fig)


def plot_eval_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    values = [metrics["quality"]["mse"], metrics["quality"]["l1"]]
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    bars = axis.bar(["MSE", "L1"], values, color=["#dc2626", "#0f766e"], width=0.55)
    axis.set(title="Held-Out LIBERO Task 20 Action Error", ylabel="Error")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "eval_comparison.png", dpi=220)
    plt.close(fig)


def plot_latency(metrics: dict[str, Any], output_dir: Path) -> None:
    values = [
        metrics["decoder_latency"]["median_ms"],
        metrics["end_to_end_latency"]["median_ms"],
    ]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(["Decoder only", "Raw image/text end-to-end"], values, color=["#d97706", "#2563eb"], width=0.55)
    axis.set(title="Autoregressive Chunk Inference Latency", ylabel="Median latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} ms", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "inference_latency.png", dpi=220)
    plt.close(fig)


def write_latex_table(metrics: dict[str, Any], output_dir: Path) -> None:
    quality = metrics["quality"]
    decoder_ms = metrics["decoder_latency"]["median_ms"]
    end_to_end_ms = metrics["end_to_end_latency"]["median_ms"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Autoregressive action generation on held-out LIBERO Task 20 episodes.}",
        r"\label{tab:ar-task20}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & MSE $\downarrow$ & L1 $\downarrow$ & Decoder (ms) $\downarrow$ & End-to-end (ms) $\downarrow$ \\",
        r"\midrule",
        (
            f"CLIP Autoregressive & {quality['mse']:.6f} & {quality['l1']:.6f} & "
            f"{decoder_ms:.1f} & {end_to_end_ms:.1f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / "results_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    batch_size = args.batch_size or int(train_args["batch_size"])
    action_stats = tensor_dict_from_lists(checkpoint["action_stats"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    print("Rebuilding dataset, frozen CLIP encoder, and policy...", flush=True)
    _, raw_action_dataset, _ = load_libero_action_dataset(
        train_args["repo_id"],
        int(train_args["horizon"]),
        train_args["task_indices"],
    )
    encoder = build_encoder(train_args).to(device)
    policy = build_policy(train_args, action_stats).to(device)
    load_trainable_encoder_state_dict(encoder, checkpoint["encoder_state_dict"])
    policy.load_state_dict(checkpoint["policy_state_dict"])

    eval_cache_path = args.checkpoint.parent / "eval_features.pt"
    if not eval_cache_path.exists():
        raise FileNotFoundError(f"Missing feature cache: {eval_cache_path}")
    eval_cache = CachedFeatureDataset(torch.load(eval_cache_path, map_location="cpu", weights_only=True))

    print(f"Evaluating {len(eval_cache)} held-out samples...", flush=True)
    quality = evaluate_quality(encoder, policy, eval_cache, device, batch_size, args.num_workers)
    print("Measuring decoder-only latency...", flush=True)
    decoder_latency = measure_decoder_latency(encoder, policy, eval_cache, device, args.latency_samples)

    eval_indices = checkpoint["split"]["eval_indices"]
    max_eval_samples = train_args.get("max_eval_samples")
    if max_eval_samples is not None:
        eval_indices = eval_indices[: int(max_eval_samples)]
    raw_eval_dataset = Subset(raw_action_dataset, eval_indices)
    print("Measuring raw image/text end-to-end latency...", flush=True)
    end_to_end_latency = measure_end_to_end_latency(
        encoder,
        policy,
        raw_eval_dataset,
        device,
        args.latency_samples,
    )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "task_indices": train_args["task_indices"],
        "eval_episode_ids": checkpoint["split"]["eval_episode_ids"],
        "quality": quality,
        "decoder_latency": decoder_latency,
        "end_to_end_latency": end_to_end_latency,
    }
    save_json(args.output_dir / "metrics.json", metrics)
    plot_training_curves(checkpoint["training_history"], args.output_dir)
    plot_eval_metrics(metrics, args.output_dir)
    plot_latency(metrics, args.output_dir)
    write_latex_table(metrics, args.output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved evaluation artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
