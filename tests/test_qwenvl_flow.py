import torch
from torch import nn
from types import MethodType

from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import ACTION
from lerobot_policy_pi0_lite_flow.qwenvl_flow import (
    QwenVLFlowConfig,
    QwenVLFlowPolicy,
    QwenVLTokenEncoder,
    TokenFlowActionHead,
    _find_visual_lora_target_modules,
)


def make_config() -> QwenVLFlowConfig:
    return QwenVLFlowConfig(
        context_dim=11,
        embed_dim=16,
        hidden_dim=32,
        horizon=3,
        action_dim=2,
        state_dim=5,
        num_heads=4,
        num_layers=2,
        num_inference_timesteps=3,
    )


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    config = make_config()
    return {
        "context_tokens": torch.randn(batch_size, 7, config.context_dim),
        "context_attention_mask": torch.tensor(
            [[True, True, True, True, False, False, False], [True, True, True, True, True, True, False]][:batch_size]
        ),
        "state": torch.randn(batch_size, config.state_dim),
        ACTION: torch.randn(batch_size, config.horizon, config.action_dim),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


def test_token_flow_action_head_loss_backpropagates_with_context_mask():
    config = make_config()
    head = TokenFlowActionHead(config)
    batch = make_batch()

    output = head.loss(
        actions=batch[ACTION],
        context_tokens=batch["context_tokens"],
        context_attention_mask=batch["context_attention_mask"],
        state=batch["state"],
        action_is_pad=batch["action_is_pad"],
    )

    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in head.parameters())


def test_qwenvl_flow_policy_sample_shape_and_range():
    config = make_config()
    policy = QwenVLFlowPolicy(config)
    batch = make_batch()

    chunk = policy.predict_action_chunk(batch)
    first = policy.select_action(batch)
    second = policy.select_action(batch)

    assert chunk.shape == (2, config.horizon, config.action_dim)
    assert first.shape == (2, config.action_dim)
    assert second.shape == (2, config.action_dim)
    assert torch.all(chunk <= 1.0)
    assert torch.all(chunk >= -1.0)


def test_qwenvl_action_normalization_round_trip():
    config = make_config()
    mean = torch.tensor([0.5, -0.25])
    std = torch.tensor([2.0, 0.5])
    head = TokenFlowActionHead(config, action_mean=mean, action_std=std)
    actions = torch.randn(2, config.horizon, config.action_dim)

    restored = head.denormalize_actions(head.normalize_actions(actions))

    assert torch.allclose(actions, restored, atol=1e-6)


def test_qwenvl_policy_checkpoint_state_loads(tmp_path):
    config = make_config()
    policy = QwenVLFlowPolicy(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"config": vars(config), "policy": policy.state_dict()}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored = QwenVLFlowPolicy(QwenVLFlowConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["policy"])

    chunk = restored.predict_action_chunk(make_batch(batch_size=1))

    assert chunk.shape == (1, config.horizon, config.action_dim)


def test_qwen_visual_lora_target_discovery_only_selects_visual_linears():
    class FakeQwen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Module()
            self.visual.patch_embed = nn.Linear(3, 4)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4)])

    targets = _find_visual_lora_target_modules(FakeQwen())

    assert targets == ["visual.patch_embed"]


def test_qwen_trainable_safety_rejects_non_visual_or_non_lora_params():
    class FakeQwen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Module()
            self.visual.lora_A = nn.Linear(2, 2)
            self.model = nn.Module()
            self.model.bad_lora = nn.Linear(2, 2)

    encoder = QwenVLTokenEncoder.__new__(QwenVLTokenEncoder)
    nn.Module.__init__(encoder)
    encoder.model = FakeQwen()
    for name, param in encoder.model.named_parameters():
        param.requires_grad = "lora" in name

    try:
        encoder._assert_only_lora_trainable()
    except RuntimeError as exc:
        assert "bad_lora" in str(exc)
    else:
        raise AssertionError("Expected non-visual LoRA trainable parameter to be rejected")


def test_qwen_forward_detaches_only_in_frozen_mode():
    source = torch.randn(2, 3, requires_grad=True)

    def fake_forward_impl(self, batch):
        return {
            "context_tokens": source * 2,
            "context_attention_mask": torch.ones(2, 3, dtype=torch.bool),
        }

    encoder = QwenVLTokenEncoder.__new__(QwenVLTokenEncoder)
    nn.Module.__init__(encoder)
    encoder._forward_impl = MethodType(fake_forward_impl, encoder)

    encoder.lora_enabled = False
    frozen = encoder.forward({})
    assert not frozen["context_tokens"].requires_grad

    encoder.lora_enabled = True
    trainable = encoder.forward({})
    assert trainable["context_tokens"].requires_grad
