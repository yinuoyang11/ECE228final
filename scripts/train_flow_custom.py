from __future__ import annotations

import argparse
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
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-clip", action="store_true", help="Use frozen CLIP image/text encoders instead of mocks.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = LIBEROActionChunkDataset(repo_id=args.repo_id, horizon=args.horizon)
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    image_encoder = None
    text_encoder = None
    image_dim = 512
    text_dim = 512
    if args.use_clip:
        image_encoder = CLIPImageEncoder(args.clip_model, freeze=True)
        text_encoder = CLIPTextEncoder(args.clip_model, freeze=True)
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

    trainable_params = [param for param in list(encoder.parameters()) + list(policy.parameters()) if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    encoder.train()
    policy.train()
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
            optimizer.step()

            if step % args.log_every == 0:
                print(f"step={step} loss={loss.item():.6f}")
            if step >= args.steps:
                break


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


if __name__ == "__main__":
    main()
