import torch

from lerobot_policy_pi0_lite_flow.configuration_regression import BCRegressionConfig
from lerobot_policy_pi0_lite_flow.modeling_regression import (
    ACTION,
    COND_KEY,
    BCRegressionDecoder,
    BCRegressionPolicy,
)


def make_decoder(**overrides) -> BCRegressionDecoder:
    defaults = dict(
        horizon=16,
        action_dim=7,
        cond_dim=256,
        hidden_dim=64,
        num_layers=2,
    )
    defaults.update(overrides)
    return BCRegressionDecoder(**defaults)


def test_decoder_forward_shape():
    decoder = make_decoder()
    cond = torch.randn(3, 256)
    assert decoder(cond).shape == (3, 16, 7)


def test_loss_is_finite_and_backpropagates_with_mask():
    decoder = make_decoder()
    actions = torch.randn(4, 16, 7)
    cond = torch.randn(4, 256)
    action_is_pad = torch.zeros(4, 16, dtype=torch.bool)
    action_is_pad[:, -3:] = True

    output = decoder.loss(actions, cond, action_is_pad)
    assert torch.isfinite(output["loss"])
    assert "mse_loss" in output
    output["loss"].backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in decoder.parameters())

    all_pad_output = decoder.loss(actions, cond, torch.ones(4, 16, dtype=torch.bool))
    assert torch.isfinite(all_pad_output["loss"])
    assert all_pad_output["loss"].item() == 0.0


def test_normalization_round_trip():
    mean = torch.arange(7, dtype=torch.float32)
    std = torch.arange(1, 8, dtype=torch.float32)
    decoder = BCRegressionDecoder(action_mean=mean, action_std=std)
    actions = torch.randn(2, 16, 7)
    restored = decoder.denormalize_actions(decoder.normalize_actions(actions))
    assert torch.allclose(actions, restored, atol=1e-6)


def test_predict_applies_denormalization():
    mean = torch.full((7,), 3.0)
    std = torch.full((7,), 2.0)
    decoder = BCRegressionDecoder(
        horizon=4,
        action_dim=7,
        cond_dim=16,
        hidden_dim=32,
        num_layers=2,
        action_mean=mean,
        action_std=std,
    )
    cond = torch.randn(2, 16)
    normalized = decoder(cond)
    denormalized = decoder.predict(cond)
    expected = decoder.denormalize_actions(normalized)
    assert torch.allclose(denormalized, expected, atol=1e-6)


def test_decoder_can_overfit_fixed_batch():
    torch.manual_seed(7)
    decoder = BCRegressionDecoder(horizon=2, action_dim=2, cond_dim=4, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-3)

    cond = torch.randn(32, 4)
    actions = torch.stack([cond[:, :2], cond[:, 2:]], dim=1)

    with torch.no_grad():
        initial = decoder.loss(actions, cond)["loss"].item()

    for _ in range(120):
        optimizer.zero_grad()
        loss = decoder.loss(actions, cond)["loss"]
        loss.backward()
        optimizer.step()

    final = decoder.loss(actions, cond)["loss"].item()
    assert final < initial * 0.5


def test_policy_forward_predict_and_select_action():
    config = BCRegressionConfig(
        horizon=16,
        action_dim=7,
        cond_dim=256,
        hidden_dim=64,
        num_layers=2,
        n_action_steps=2,
    )
    dataset_stats = {ACTION: {"mean": torch.ones(7), "std": torch.full((7,), 2.0)}}
    policy = BCRegressionPolicy(config, dataset_stats=dataset_stats)
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


def test_policy_accepts_dataset_stats_with_action_alias():
    config = BCRegressionConfig(
        horizon=4,
        action_dim=3,
        cond_dim=8,
        hidden_dim=16,
        num_layers=2,
    )
    dataset_stats = {"action": {"mean": torch.zeros(3), "std": torch.ones(3)}}
    policy = BCRegressionPolicy(config, dataset_stats=dataset_stats)
    assert policy.decoder.action_mean.shape == (1, 1, 3)
    assert policy.decoder.action_std.shape == (1, 1, 3)


def test_policy_uses_cond_fallback_key():
    config = BCRegressionConfig(horizon=4, action_dim=3, cond_dim=8, hidden_dim=16, num_layers=2)
    policy = BCRegressionPolicy(config)
    batch = {
        ACTION: torch.randn(2, 4, 3),
        "cond": torch.randn(2, 8),
    }
    output = policy.forward(batch)
    assert torch.isfinite(output["loss"])
