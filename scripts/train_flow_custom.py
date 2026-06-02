from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig  # noqa: E402
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import ACTION, COND_KEY, PI0LiteFlowPolicy  # noqa: E402
from lerobot_policy_pi0_lite_flow.representation_encoder import (  # noqa: E402
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal custom LIBERO training loop for pi0-lite flow.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--task-indices", nargs="+", type=int, default=None, help="Optional LIBERO task indices to train.")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP image/text encoders instead of mocks.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--finetune-clip-vision-layers", type=int, default=0, help="Unfreeze the last N CLIP vision layers. Use -1 for all.")
    parser.add_argument("--finetune-clip-text-layers", type=int, default=0, help="Unfreeze the last N CLIP text layers. Use -1 for all.")
    parser.add_argument("--clip-lr", type=float, default=1e-5, help="Learning rate for unfrozen CLIP parameters.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", default="runs/pi0_lite_flow")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Initialize encoder and policy weights from a checkpoint.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    validate_clip_args(args)

    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    log_path = run_dir / "metrics.csv"
    save_args(args, run_dir)

    dataset = LIBEROActionChunkDataset(repo_id=args.repo_id, horizon=args.horizon, task_indices=args.task_indices)
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"dataset_samples={len(dataset)} task_indices={args.task_indices or 'all'}")

    image_encoder = None
    text_encoder = None
    image_dim = 512
    text_dim = 512
    if args.use_clip:
        image_encoder = CLIPImageEncoder(
            args.clip_model,
            trainable_layers=args.finetune_clip_vision_layers,
        )
        text_encoder = CLIPTextEncoder(
            args.clip_model,
            trainable_layers=args.finetune_clip_text_layers,
        )
        image_dim = image_encoder.output_dim
        text_dim = text_encoder.output_dim

    encoder = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=image_dim,
            text_dim=text_dim,
            state_dim=args.state_dim,
            cond_dim=args.cond_dim,
            num_views=2,
        ),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    ).to(device)
    policy = PI0LiteFlowPolicy(
        PI0LiteFlowConfig(
            horizon=args.horizon,
            action_dim=args.action_dim,
            cond_dim=args.cond_dim,
        )
    ).to(device)
    if args.init_checkpoint:
        load_initial_weights(Path(args.init_checkpoint), encoder, policy, device)

    clip_params = [
        param
        for name, param in encoder.named_parameters()
        if param.requires_grad and _is_clip_backbone_key(name)
    ]
    base_params = [
        param
        for name, param in encoder.named_parameters()
        if param.requires_grad and not _is_clip_backbone_key(name)
    ]
    base_params.extend(param for param in policy.parameters() if param.requires_grad)
    trainable_params = [*base_params, *clip_params]
    optimizer_groups = [{"params": base_params, "lr": args.lr}]
    if clip_params:
        optimizer_groups.append({"params": clip_params, "lr": args.clip_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    encoder.train()
    policy.train()
    print(f"run_dir={run_dir}")
    print(
        f"device={device} trainable_params={sum(param.numel() for param in trainable_params)} "
        f"clip_trainable_params={sum(param.numel() for param in clip_params)}"
    )
    step = 0
    while step < args.steps:
        for batch in loader:
            step += 1
            batch = _move_batch_to_device(batch, device)
            cond = encoder(batch)
            policy_batch = {
                COND_KEY: cond,
                ACTION: batch[ACTION].float(),
                "action_is_pad": batch["action_is_pad"],
            }

            output = policy.forward(policy_batch)
            loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if args.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm)
            optimizer.step()

            if step % args.log_every == 0:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "fm_loss": float(output["fm_loss"].detach().cpu()),
                    "pred_velocity_norm": float(output["pred_velocity_norm"].detach().cpu()),
                    "target_velocity_norm": float(output["target_velocity_norm"].detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else "",
                }
                append_metrics(log_path, row)
                print(" ".join(f"{key}={value}" for key, value in row.items()))
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(run_dir / f"checkpoint_step_{step:06d}.pt", step, encoder, policy, optimizer, args)
            if step >= args.steps:
                break

    save_checkpoint(run_dir / "checkpoint_final.pt", step, encoder, policy, optimizer, args)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_clip_args(args: argparse.Namespace) -> None:
    layer_counts = (args.finetune_clip_vision_layers, args.finetune_clip_text_layers)
    if any(layer_count < -1 for layer_count in layer_counts):
        raise ValueError("CLIP finetune layer counts must be -1, 0, or positive integers")
    if any(layer_counts) and not args.use_clip:
        raise ValueError("CLIP finetuning requires --use-clip")
    if args.clip_lr <= 0:
        raise ValueError("clip-lr must be positive")


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
    encoder: RepresentationEncoder,
    policy: PI0LiteFlowPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "step": step,
            "args": vars(args),
            "encoder": trainable_state_dict(encoder),
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"saved_checkpoint={path}")


def load_initial_weights(
    path: Path,
    encoder: RepresentationEncoder,
    policy: PI0LiteFlowPolicy,
    device: torch.device,
) -> None:
    checkpoint = torch.load(path, map_location=device)
    _, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected}")
    policy.load_state_dict(checkpoint["policy"])
    print(f"initialized_from={path}")


def trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, param in module.named_parameters() if param.requires_grad}
    return {
        name: value
        for name, value in module.state_dict().items()
        if name in trainable_names or not _is_clip_backbone_key(name)
    }


def _is_clip_backbone_key(name: str) -> bool:
    return name.startswith("image_encoder.model.") or name.startswith("text_encoder.model.")


if __name__ == "__main__":
    main()
