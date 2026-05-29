from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig  # noqa: E402
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import ACTION, PI0LiteFlowPolicy  # noqa: E402
from lerobot_policy_pi0_lite_flow.representation_encoder import (  # noqa: E402
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline validation for the custom pi0-lite flow pipeline.")
    parser.add_argument("checkpoint", help="Path to checkpoint_final.pt or checkpoint_step_*.pt")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--skip-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-steps", type=int, default=None, help="Flow Euler inference steps. Defaults to config.")
    parser.add_argument("--output", default=None, help="Optional CSV path for the aggregate metrics.")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    repo_id = args.repo_id or train_args.get("repo_id", "HuggingFaceVLA/libero")
    horizon = int(train_args.get("horizon", 16))
    action_dim = int(train_args.get("action_dim", 7))
    state_dim = int(train_args.get("state_dim", 8))
    cond_dim = int(train_args.get("cond_dim", 256))
    clip_model = train_args.get("clip_model", "openai/clip-vit-base-patch32")
    use_clip = bool(train_args.get("use_clip", True))
    skip_samples = int(args.skip_samples if args.skip_samples is not None else train_args.get("max_samples", 0))

    dataset = LIBEROActionChunkDataset(repo_id=repo_id, horizon=horizon)
    start = min(skip_samples, len(dataset))
    end = min(start + args.num_samples, len(dataset))
    dataset = Subset(dataset, range(start, end))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    image_encoder = None
    text_encoder = None
    image_dim = 512
    text_dim = 512
    if use_clip:
        image_encoder = CLIPImageEncoder(clip_model, freeze=True)
        text_encoder = CLIPTextEncoder(clip_model, freeze=True)
        image_dim = image_encoder.output_dim
        text_dim = text_encoder.output_dim

    encoder = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=image_dim,
            text_dim=text_dim,
            state_dim=state_dim,
            cond_dim=cond_dim,
            num_views=2,
        ),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    ).to(device)
    missing, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("image_encoder.model.", "text_encoder.model."))]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected}")

    policy = PI0LiteFlowPolicy(
        PI0LiteFlowConfig(
            horizon=horizon,
            action_dim=action_dim,
            cond_dim=cond_dim,
            inference_steps=int(args.num_steps or train_args.get("inference_steps", 8)),
        )
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])

    encoder.eval()
    policy.eval()

    totals = {
        "mse_sum": 0.0,
        "l1_sum": 0.0,
        "smoothness_sum": 0.0,
        "num_values": 0,
        "num_smooth_values": 0,
        "num_batches": 0,
        "latency_sum": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            valid = (~batch["action_is_pad"].bool()).to(device)
            expert = batch[ACTION].float()

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            cond = encoder(batch)
            pred = policy.decoder.sample(cond, num_steps=policy.config.inference_steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            squared = (pred - expert).pow(2) * valid[..., None]
            absolute = (pred - expert).abs() * valid[..., None]
            totals["mse_sum"] += float(squared.sum().cpu())
            totals["l1_sum"] += float(absolute.sum().cpu())
            totals["num_values"] += int(valid.sum().item() * expert.shape[-1])

            if expert.shape[1] > 1:
                pred_delta = pred[:, 1:] - pred[:, :-1]
                valid_delta = (valid[:, 1:] & valid[:, :-1]).to(pred_delta.dtype)
                smooth = pred_delta.pow(2).sum(dim=-1) * valid_delta
                totals["smoothness_sum"] += float(smooth.sum().cpu())
                totals["num_smooth_values"] += int(valid_delta.sum().item())

            totals["num_batches"] += 1
            totals["latency_sum"] += elapsed / max(expert.shape[0], 1)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "repo_id": repo_id,
        "start_index": start,
        "num_samples": end - start,
        "mse": totals["mse_sum"] / max(totals["num_values"], 1),
        "l1": totals["l1_sum"] / max(totals["num_values"], 1),
        "smoothness": totals["smoothness_sum"] / max(totals["num_smooth_values"], 1),
        "latency_sec_per_sample": totals["latency_sum"] / max(totals["num_batches"], 1),
    }
    print(json.dumps(metrics, indent=2))

    if args.output:
        write_metrics_csv(Path(args.output), metrics)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return output


def write_metrics_csv(path: Path, metrics: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()

