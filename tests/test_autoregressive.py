"""Unit tests for the discretized autoregressive action-token baseline.

Mirrors the structure of test_flow_decoder.py to keep the test suite consistent.
"""
import torch

from lerobot_policy_pi0_lite_flow.action_tokenizer import ActionTokenizer
from lerobot_policy_pi0_lite_flow.configuration_autoregressive import (
    PI0LiteAutoregressiveConfig,
)
from lerobot_policy_pi0_lite_flow.modeling_autoregressive import (
    ACTION,
    COND_KEY,
    AutoregressiveActionDecoder,
    AutoregressivePolicy,
)


# ---------------------------------------------------------------------------
# Action Tokenizer tests
# ---------------------------------------------------------------------------


def test_tokenizer_encode_decode_shape():
    tok = ActionTokenizer(num_bins=256, action_dim=7, horizon=16)
    actions = torch.randn(3, 16, 7)

    token_ids = tok.encode(actions)
    assert token_ids.shape == (3, 16 * 7)
    assert token_ids.dtype == torch.long
    assert (token_ids >= 0).all() and (token_ids < 256).all()

    reconstructed = tok.decode(token_ids)
    assert reconstructed.shape == (3, 16, 7)


def test_tokenizer_round_trip_within_bin_width():
    tok = ActionTokenizer(num_bins=256, action_dim=7, horizon=4)
    # Set known bounds.
    tok.set_bounds(torch.full((7,), -1.0), torch.full((7,), 1.0))

    actions = torch.rand(8, 4, 7) * 2 - 1  # Uniform in [-1, 1].
    token_ids = tok.encode(actions)
    reconstructed = tok.decode(token_ids, horizon=4)

    # Max error per dimension should be at most half a bin width.
    max_bin_width = (2.0 / 255)  # range / (num_bins - 1)
    max_error = (actions - reconstructed).abs().max().item()
    assert max_error <= max_bin_width + 1e-6, f"Max error {max_error} exceeds {max_bin_width}"


def test_tokenizer_set_bounds_from_data():
    tok = ActionTokenizer(num_bins=256, action_dim=3, horizon=2)
    actions = torch.randn(100, 2, 3) * 5 + 2  # mean ≈ 2, std ≈ 5
    tok.set_bounds_from_data(actions)

    assert tok.action_min.shape == (3,)
    assert tok.action_max.shape == (3,)
    assert (tok.action_max > tok.action_min).all()


def test_tokenizer_clamps_out_of_range():
    tok = ActionTokenizer(num_bins=256, action_dim=2, horizon=1)
    tok.set_bounds(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))

    # Values well outside bounds.
    actions = torch.tensor([[[10.0, -10.0]]])  # (1, 1, 2)
    token_ids = tok.encode(actions)
    assert (token_ids >= 0).all() and (token_ids < 256).all()


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------


def make_decoder(**kwargs) -> AutoregressiveActionDecoder:
    defaults = dict(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return AutoregressiveActionDecoder(**defaults)


def test_decoder_forward_shape():
    decoder = make_decoder()
    token_ids = torch.randint(0, 16, (2, 4 * 3))  # (B, H*D)
    cond = torch.randn(2, 32)

    logits = decoder(token_ids, cond)
    assert logits.shape == (2, 4 * 3 + 1, 16)  # (B, seq_len+1, num_bins)


def test_loss_is_finite_and_backpropagates():
    decoder = make_decoder()
    token_ids = torch.randint(0, 16, (4, 4 * 3))
    cond = torch.randn(4, 32)

    output = decoder.loss(token_ids, cond)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in decoder.parameters()
    )


def test_causal_decoder_future_token_does_not_change_earlier_logits():
    decoder = make_decoder()
    decoder.eval()
    cond = torch.randn(1, 32)
    tokens_a = torch.randint(0, 16, (1, 12))
    tokens_b = tokens_a.clone()
    changed_index = 7
    tokens_b[:, changed_index] = (tokens_b[:, changed_index] + 1) % 16

    logits_a = decoder(tokens_a, cond)
    logits_b = decoder(tokens_b, cond)

    assert torch.allclose(logits_a[:, : changed_index + 1], logits_b[:, : changed_index + 1])


def test_padded_tokens_do_not_change_loss_or_accuracy():
    decoder = make_decoder()
    decoder.eval()
    cond = torch.randn(2, 32)
    tokens_a = torch.randint(0, 16, (2, 12))
    tokens_b = tokens_a.clone()
    tokens_b[:, 8:] = (tokens_b[:, 8:] + 3) % 16
    token_is_pad = torch.zeros_like(tokens_a, dtype=torch.bool)
    token_is_pad[:, 8:] = True

    output_a = decoder.loss(tokens_a, cond, token_is_pad=token_is_pad)
    output_b = decoder.loss(tokens_b, cond, token_is_pad=token_is_pad)

    assert torch.allclose(output_a["loss"], output_b["loss"])
    assert torch.allclose(output_a["accuracy"], output_b["accuracy"])


def test_sample_shape():
    decoder = make_decoder()
    cond = torch.randn(3, 32)
    samples = decoder.sample(cond, temperature=0.0)
    assert samples.shape == (3, 4 * 3)
    assert samples.dtype == torch.long
    assert (samples >= 0).all() and (samples < 16).all()


def test_sample_deterministic_greedy():
    decoder = make_decoder()
    cond = torch.randn(2, 32)
    s1 = decoder.sample(cond, temperature=0.0)
    s2 = decoder.sample(cond, temperature=0.0)
    assert torch.equal(s1, s2)


def test_sample_with_temperature():
    decoder = make_decoder()
    cond = torch.randn(2, 32)
    samples = decoder.sample(cond, temperature=1.0)
    assert samples.shape == (2, 4 * 3)


# ---------------------------------------------------------------------------
# Decoder overfitting test
# ---------------------------------------------------------------------------


def test_decoder_can_overfit_small_batch():
    """The decoder should be able to overfit a fixed batch of token sequences."""
    torch.manual_seed(42)
    decoder = make_decoder(
        horizon=2, action_dim=2, num_bins=8,
        hidden_dim=64, num_layers=2, nhead=4,
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)

    # Fixed batch: 16 samples, each with 4 tokens (H=2, D=2).
    cond = torch.randn(16, 32)
    token_ids = torch.randint(0, 8, (16, 2 * 2))

    # Record initial loss.
    initial_output = decoder.loss(token_ids, cond)
    initial_loss = initial_output["loss"].item()

    # Train for a few steps.
    for _ in range(100):
        optimizer.zero_grad()
        output = decoder.loss(token_ids, cond)
        output["loss"].backward()
        optimizer.step()

    final_output = decoder.loss(token_ids, cond)
    final_loss = final_output["loss"].item()

    assert final_loss < initial_loss * 0.5, (
        f"Expected loss to drop by at least 50%: {initial_loss:.4f} → {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


def test_policy_forward_and_loss():
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0,
    )
    stats = {
        ACTION: {
            "min": torch.full((3,), -2.0),
            "max": torch.full((3,), 2.0),
        }
    }
    policy = AutoregressivePolicy(config, dataset_stats=stats)

    batch = {
        ACTION: torch.randn(3, 4, 3),  # (B, H, D)
        COND_KEY: torch.randn(3, 32),
    }

    output = policy.forward(batch)
    assert torch.isfinite(output["loss"])
    assert "accuracy" in output


def test_policy_forward_accepts_action_padding_mask():
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.0,
    )
    policy = AutoregressivePolicy(config)
    output = policy(
        {
            ACTION: torch.randn(2, 4, 3),
            "action_is_pad": torch.tensor([[False, False, True, True], [False, False, False, True]]),
            COND_KEY: torch.randn(2, 32),
        }
    )

    assert torch.isfinite(output["loss"])


def test_policy_predict_action_chunk():
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
    )
    policy = AutoregressivePolicy(config)

    batch = {COND_KEY: torch.randn(2, 32)}
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 4, 3)


def test_policy_select_action_with_caching():
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
        n_action_steps=2,
    )
    policy = AutoregressivePolicy(config)

    batch = {COND_KEY: torch.randn(2, 32)}

    first = policy.select_action(batch)
    second = policy.select_action(batch)
    assert first.shape == (2, 3)
    assert second.shape == (2, 3)

    # After n_action_steps, cache should be refreshed.
    policy.reset()
    assert policy._cached_action_chunk is None


def test_policy_without_dataset_stats():
    """Policy should work with default tokenizer bounds when no stats are given."""
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
    )
    policy = AutoregressivePolicy(config)

    batch = {
        ACTION: torch.randn(2, 4, 3) * 0.5,  # Within default [-1, 1] range.
        COND_KEY: torch.randn(2, 32),
    }
    output = policy.forward(batch)
    assert torch.isfinite(output["loss"])


def test_empty_output_feature_map_does_not_require_lerobot_action_feature():
    config = PI0LiteAutoregressiveConfig()
    config.output_features = {}

    config.validate_features()


def test_policy_with_mean_std_stats():
    """Policy should fallback to mean ± 3*std when min/max not available."""
    config = PI0LiteAutoregressiveConfig(
        horizon=4, action_dim=3, cond_dim=32, num_bins=16,
        hidden_dim=64, nhead=4, num_layers=2, dim_feedforward=128,
    )
    stats = {
        ACTION: {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
        }
    }
    policy = AutoregressivePolicy(config, dataset_stats=stats)

    # Bounds should be [-3, 3].
    assert torch.allclose(policy.tokenizer.action_min, torch.full((3,), -3.0))
    assert torch.allclose(policy.tokenizer.action_max, torch.full((3,), 3.0))
