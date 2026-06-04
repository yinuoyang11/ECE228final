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
from lerobot_policy_pi0_lite_flow.configuration_autoregressive import PI0LiteAutoregressiveConfig  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_autoregressive import AutoregressivePolicy  # noqa: E402
from lerobot_policy_pi0_lite_flow.autoregressive_experiment import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CLIP-backed representation encoder smoke test.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)

    image_encoder = CLIPImageEncoder(args.model_name, freeze=True)
    text_encoder = CLIPTextEncoder(args.model_name, freeze=True)
    config = RepresentationEncoderConfig(
        image_dim=image_encoder.output_dim,
        text_dim=text_encoder.output_dim,
        state_dim=8,
        cond_dim=256,
        num_views=2,
    )
    encoder = RepresentationEncoder(config, image_encoder=image_encoder, text_encoder=text_encoder).to(device)
    base_instructions = ["pick up the cube", "open the drawer"]
    batch = {
        "images": torch.randint(0, 256, (args.batch_size, 2, 3, 224, 224), dtype=torch.uint8, device=device),
        "state": torch.randn(args.batch_size, 8, device=device),
        "instruction": [base_instructions[idx % len(base_instructions)] for idx in range(args.batch_size)],
    }

    encoder.train()
    with torch.no_grad():
        features = encoder.extract_features(batch)
        cond = encoder(batch)
        cached_cond = encoder.forward_from_features(features["image_features"], batch["state"], features["text_features"])
    if not torch.allclose(cond, cached_cond, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Raw and cached CLIP feature paths do not match")

    policy_config = PI0LiteAutoregressiveConfig(
        horizon=2,
        action_dim=7,
        cond_dim=256,
        num_bins=32,
        hidden_dim=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
    )
    policy = AutoregressivePolicy(policy_config).to(device)
    output = policy(
        {
            "action": torch.randn(args.batch_size, 2, 7, device=device),
            "action_is_pad": torch.zeros(args.batch_size, 2, dtype=torch.bool, device=device),
            "observation.cond": cond,
        }
    )
    output["loss"].backward()
    reloaded = AutoregressivePolicy(policy_config).to(device)
    reloaded.load_state_dict(policy.state_dict())

    print(f"device: {device}")
    print(f"cond shape: {tuple(cond.shape)}")
    print(f"has nan: {torch.isnan(cond).any().item()}")
    print(f"loss: {output['loss'].item():.4f}")


if __name__ == "__main__":
    main()
