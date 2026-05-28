from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.representation_encoder import (  # noqa: E402
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CLIP-backed representation encoder smoke test.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    image_encoder = CLIPImageEncoder(args.model_name, freeze=True)
    text_encoder = CLIPTextEncoder(args.model_name, freeze=True)
    config = RepresentationEncoderConfig(
        image_dim=image_encoder.output_dim,
        text_dim=text_encoder.output_dim,
        state_dim=8,
        cond_dim=256,
        num_views=2,
    )
    encoder = RepresentationEncoder(config, image_encoder=image_encoder, text_encoder=text_encoder)
    base_instructions = ["pick up the cube", "open the drawer"]
    batch = {
        "images": torch.randint(0, 256, (args.batch_size, 2, 3, 224, 224), dtype=torch.uint8),
        "state": torch.randn(args.batch_size, 8),
        "instruction": [base_instructions[idx % len(base_instructions)] for idx in range(args.batch_size)],
    }

    with torch.no_grad():
        cond = encoder(batch)
    print(f"cond shape: {tuple(cond.shape)}")
    print(f"has nan: {torch.isnan(cond).any().item()}")


if __name__ == "__main__":
    main()
