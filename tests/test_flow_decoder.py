import torch

from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import (
    ACTION,
    COND_KEY,
    CFMActionDecoder,
    PI0LiteFlowPolicy,
)


def make_decoder(architecture: str) -> CFMActionDecoder:
    return CFMActionDecoder(
        horizon=16,
        action_dim=7,
        cond_dim=256,
        architecture=architecture,
        hidden_dim=64,
        num_layers=2,
        transformer_nhead=4,
        transformer_num_layers=1,
        transformer_dim_feedforward=128,
    )


def test_decoder_architectures_preserve_shape():
    for arch in ["mlp_film", "temporal_transformer"]:
        decoder = make_decoder(arch)
        x_t = torch.randn(3, 16, 7)
        t = torch.rand(3)
        cond = torch.randn(3, 256)
        assert decoder(x_t, t, cond).shape == (3, 16, 7)


def test_loss_is_finite_and_backpropagates_with_mask():
    decoder = make_decoder("mlp_film")
    actions = torch.randn(4, 16, 7)
    cond = torch.randn(4, 256)
    action_is_pad = torch.zeros(4, 16, dtype=torch.bool)
    action_is_pad[:, -3:] = True

    output = decoder.loss(actions, cond, action_is_pad)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in decoder.parameters())

    all_pad_output = decoder.loss(actions, cond, torch.ones(4, 16, dtype=torch.bool))
    assert torch.isfinite(all_pad_output["loss"])
    assert all_pad_output["loss"].item() == 0.0


def test_normalization_round_trip():
    mean = torch.arange(7, dtype=torch.float32)
    std = torch.arange(1, 8, dtype=torch.float32)
    decoder = CFMActionDecoder(action_mean=mean, action_std=std)
    actions = torch.randn(2, 16, 7)
    restored = decoder.denormalize_actions(decoder.normalize_actions(actions))
    assert torch.allclose(actions, restored, atol=1e-6)


def test_sampling_is_deterministic_with_seeded_generator():
    decoder = make_decoder("mlp_film")
    cond = torch.randn(2, 256)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    sample_a = decoder.sample(cond, num_steps=4, generator=generator_a)
    sample_b = decoder.sample(cond, num_steps=4, generator=generator_b)

    assert sample_a.shape == (2, 16, 7)
    assert torch.allclose(sample_a, sample_b)


def test_mlp_can_overfit_fixed_flow_batch():
    torch.manual_seed(7)
    decoder = CFMActionDecoder(horizon=2, action_dim=2, cond_dim=4, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-3)

    cond = torch.randn(32, 4)
    actions = torch.stack([cond[:, :2], cond[:, 2:]], dim=1)
    x1 = decoder.normalize_actions(actions)
    x0 = torch.randn_like(x1)
    t = torch.linspace(0.05, 0.95, steps=32)
    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    with torch.no_grad():
        initial = (decoder(x_t, t, cond) - target).pow(2).mean().item()

    for _ in range(120):
        optimizer.zero_grad()
        loss = (decoder(x_t, t, cond) - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    final = (decoder(x_t, t, cond) - target).pow(2).mean().item()
    assert final < initial * 0.5


def test_policy_forward_predict_and_select_action():
    config = PI0LiteFlowConfig(
        horizon=16,
        action_dim=7,
        cond_dim=256,
        hidden_dim=64,
        num_layers=2,
        inference_steps=2,
        n_action_steps=2,
    )
    dataset_stats = {ACTION: {"mean": torch.ones(7), "std": torch.full((7,), 2.0)}}
    policy = PI0LiteFlowPolicy(config, dataset_stats=dataset_stats)
    batch = {
        ACTION: torch.randn(3, 16, 7),
        COND_KEY: torch.randn(3, 256),
        "action_is_pad": torch.zeros(3, 16, dtype=torch.bool),
    }

    output = policy.forward(batch)
    assert torch.isfinite(output["loss"])

    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (3, 16, 7)

    first = policy.select_action(batch)
    second = policy.select_action(batch)
    assert first.shape == (3, 7)
    assert second.shape == (3, 7)

    policy.reset()
    assert policy._cached_action_chunk is None
