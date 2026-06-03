from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import ACTION  # noqa: E402
from lerobot_policy_pi0_lite_flow.qwenvl_flow import (  # noqa: E402
    QwenVLFlowConfig,
    QwenVLFlowPolicy,
    QwenVLTokenEncoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-QwenVL token flow action head on LIBERO.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--task-indices", nargs="+", type=int, default=[20])
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--qwen-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--qwen-max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=896)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-inference-timesteps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", default="runs/qwenvl_flow")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    validate_args(args)

    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    log_path = run_dir / "metrics.csv"
    save_args(args, run_dir)

    dataset = LIBEROActionChunkDataset(repo_id=args.repo_id, horizon=args.horizon, task_indices=args.task_indices)
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"dataset_samples={len(dataset)} task_indices={args.task_indices}")

    qwen_encoder = QwenVLTokenEncoder(
        model_name=args.qwen_model,
        min_pixels=args.qwen_min_pixels,
        max_pixels=args.qwen_max_pixels,
        dtype=args.qwen_dtype,
        device=device,
    )
    action_stats = load_action_stats(args.repo_id)
    config = QwenVLFlowConfig(
        qwen_model=args.qwen_model,
        context_dim=qwen_encoder.output_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_inference_timesteps=args.num_inference_timesteps,
    )
    policy = QwenVLFlowPolicy(config, dataset_stats={ACTION: action_stats}).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    qwen_trainable = sum(param.numel() for param in qwen_encoder.parameters() if param.requires_grad)
    head_trainable = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    print(f"run_dir={run_dir}")
    print(f"device={device} qwen_trainable_params={qwen_trainable} head_trainable_params={head_trainable}")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    printed_context_shape = False
    while step < args.steps:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                encoded = qwen_encoder(batch)
            if not printed_context_shape:
                print(f"context_tokens_shape={tuple(encoded['context_tokens'].shape)}")
                printed_context_shape = True
            policy_batch = {
                "context_tokens": encoded["context_tokens"],
                "context_attention_mask": encoded["context_attention_mask"],
                "state": batch["state"].float(),
                ACTION: batch[ACTION].float(),
                "action_is_pad": batch["action_is_pad"],
            }
            output = policy(policy_batch)
            loss = output["loss"] / args.grad_accum_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue

            grad_norm = None
            if args.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                row = {
                    "step": step,
                    "micro_step": micro_step,
                    "loss": float(output["loss"].detach().cpu()),
                    "fm_loss": float(output["fm_loss"].detach().cpu()),
                    "pred_velocity_norm": float(output["pred_velocity_norm"].detach().cpu()),
                    "target_velocity_norm": float(output["target_velocity_norm"].detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else "",
                }
                append_metrics(log_path, row)
                print(" ".join(f"{key}={value}" for key, value in row.items()))
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(run_dir / f"checkpoint_step_{step:06d}.pt", step, policy, optimizer, args, config)
            if step >= args.steps:
                break

    save_checkpoint(run_dir / "checkpoint_final.pt", step, policy, optimizer, args, config)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive")
    if args.task_indices is None or not args.task_indices:
        raise ValueError("task-indices must contain at least one task")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_dir: str, run_name: str | None) -> Path:
    root = Path(output_dir)
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / name
    if run_dir.exists():
        suffix = 1
        while (root / f"{name}_{suffix:03d}").exists():
            suffix += 1
        run_dir = root / f"{name}_{suffix:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_args(args: argparse.Namespace, run_dir: Path) -> None:
    with (run_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)


def load_action_stats(repo_id: str) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo_id=repo_id, filename="meta/stats.json", repo_type="dataset")
    with open(path, encoding="utf-8") as file:
        stats = json.load(file)["action"]
    action_stats = {key: torch.tensor(stats[key], dtype=torch.float32) for key in ("mean", "std")}
    print(f"action_mean={action_stats['mean'].tolist()} action_std={action_stats['std'].tolist()}")
    return action_stats


def append_metrics(path: Path, row: dict[str, int | float | str]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    step: int,
    policy: QwenVLFlowPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    config: QwenVLFlowConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "args": vars(args),
            "config": vars(config),
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"saved_checkpoint={path}")


if __name__ == "__main__":
    main()
