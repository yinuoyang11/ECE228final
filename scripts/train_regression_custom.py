"""Custom LIBERO training loop for the BC regression baseline.

This mirrors scripts/train_flow_custom.py so the regression baseline is trained
on the exact same data pipeline, encoder, and optimizer setup as the flow
matching policy. The only differences are the policy class (BCRegressionPolicy)
and an optional ``--freeze-encoder`` switch for loading a pretrained encoder
checkpoint and training only the regression decoder.
"""

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

from lerobot_policy_pi0_lite_flow.configuration_regression import BCRegressionConfig  # noqa: E402
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_regression import (  # noqa: E402
    ACTION,
    COND_KEY,
    BCRegressionPolicy,
)
from lerobot_policy_pi0_lite_flow.representation_encoder import (  # noqa: E402
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom LIBERO training loop for the BC regression baseline.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--local-dir", default=None, help="Path to locally downloaded LIBERO dataset (skips HF Hub download).")
    parser.add_argument("--task-indices", nargs="+", type=int, default=None, help="Optional LIBERO task indices to train.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="Explicit LIBERO episode_index list. Bypasses task-based filtering. "
        "Useful when the dataset's task metadata is unreliable (HuggingFaceVLA/libero).",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP image/text encoders instead of mocks.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--finetune-clip-vision-layers",
        type=int,
        default=0,
        help="Unfreeze the last N CLIP vision layers. Use -1 for all.",
    )
    parser.add_argument(
        "--finetune-clip-text-layers",
        type=int,
        default=0,
        help="Unfreeze the last N CLIP text layers. Use -1 for all.",
    )
    parser.add_argument("--clip-lr", type=float, default=1e-5, help="Learning rate for unfrozen CLIP parameters.")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the entire representation encoder; train only the regression decoder. "
        "Requires --init-checkpoint to provide encoder weights for a fair comparison.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize encoder (and optionally policy) weights from a flow or regression checkpoint.",
    )
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

    dataset = LIBEROActionChunkDataset(
        repo_id=args.repo_id,
        horizon=args.horizon,
        task_indices=args.task_indices,
        episodes=args.episodes,
        root=args.local_dir,
    )
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
    policy = BCRegressionPolicy(
        BCRegressionConfig(
            horizon=args.horizon,
            action_dim=args.action_dim,
            cond_dim=args.cond_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        dataset_stats={ACTION: load_action_stats(args.repo_id, local_dir=args.local_dir)},
    ).to(device)
    if args.init_checkpoint:
        load_initial_weights(Path(args.init_checkpoint), encoder, policy, device)

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("encoder_frozen=True")

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
    if not trainable_params:
        raise RuntimeError("No trainable parameters; check --freeze-encoder configuration.")

    optimizer_groups = [{"params": base_params, "lr": args.lr}] if base_params else []
    if clip_params:
        optimizer_groups.append({"params": clip_params, "lr": args.clip_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    if not args.freeze_encoder:
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
            if args.freeze_encoder:
                with torch.no_grad():
                    cond = encoder(batch)
            else:
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
                    "mse_loss": float(output["mse_loss"].detach().cpu()),
                    "pred_action_norm": float(output["pred_action_norm"].detach().cpu()),
                    "target_action_norm": float(output["target_action_norm"].detach().cpu()),
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
    if args.freeze_encoder and not args.init_checkpoint:
        raise ValueError(
            "--freeze-encoder requires --init-checkpoint so the encoder uses pretrained weights instead of random init."
        )
    if args.freeze_encoder and any(layer_counts):
        raise ValueError("--freeze-encoder is incompatible with CLIP finetuning layer counts")


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


def load_action_stats(repo_id: str, local_dir: str | None = None) -> dict[str, torch.Tensor]:
    if local_dir is not None:
        path = Path(local_dir) / "meta" / "stats.json"
    else:
        path = hf_hub_download(repo_id=repo_id, filename="meta/stats.json", repo_type="dataset")
    with open(path, encoding="utf-8") as file:
        stats = json.load(file)["action"]
    action_stats = {key: torch.tensor(stats[key], dtype=torch.float32) for key in ("mean", "std")}
    print(
        f"action_mean={action_stats['mean'].tolist()} "
        f"action_std={action_stats['std'].tolist()}"
    )
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
    encoder: RepresentationEncoder,
    policy: BCRegressionPolicy,
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
    policy: BCRegressionPolicy,
    device: torch.device,
) -> None:
    """Load encoder weights from any flow/regression checkpoint; policy only loads if shapes match."""
    checkpoint = torch.load(path, map_location=device)
    _, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("image_encoder.model.", "text_encoder.model."))]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected}")
    policy_state = checkpoint.get("policy")
    if policy_state is not None:
        try:
            policy.load_state_dict(policy_state, strict=False)
            print(f"initialized_policy_from={path}")
        except RuntimeError as exc:
            print(f"skipped_policy_init (incompatible state dict): {exc}")
    print(f"initialized_encoder_from={path}")


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
