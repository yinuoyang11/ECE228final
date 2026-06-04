from __future__ import annotations

import math
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from .modeling_pi0_lite_flow import ACTION


@dataclass
class QwenVLFlowConfig:
    qwen_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    context_dim: int = 2048
    embed_dim: int = 896
    hidden_dim: int = 1024
    horizon: int = 16
    action_dim: int = 7
    state_dim: int = 8
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.0
    num_inference_timesteps: int = 20
    n_action_steps: int = 1
    action_eps: float = 1e-6

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.n_action_steps <= 0 or self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must be in [1, horizon]")


class QwenVLTokenEncoder(nn.Module):
    """Frozen Qwen2.5-VL encoder that returns vision-language token features."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        dtype: str = "bfloat16",
        device: torch.device | str | None = None,
        lora_enabled: bool = False,
        lora_scope: str = "vision",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_adapter_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional heavy dependency.
            raise ImportError("Install transformers with Qwen2.5-VL support to use QwenVLTokenEncoder.") from exc
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:  # pragma: no cover - depends on the installed transformers version.
            from transformers import AutoModelForImageTextToText as Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model_kwargs = {"trust_remote_code": True}
        torch_dtype = _resolve_torch_dtype(dtype)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        self.lora_enabled = lora_enabled or lora_adapter_path is not None
        self.lora_scope = lora_scope
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        for param in self.model.parameters():
            param.requires_grad = False
        if lora_adapter_path is not None:
            self._load_lora_adapter(lora_adapter_path)
        elif lora_enabled:
            self._enable_lora(lora_scope=lora_scope, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if device is not None:
            self.model.to(device)
        self.model.train(mode=self.lora_enabled)
        self.output_dim = int(getattr(self.model.config, "hidden_size", 0) or self.model.config.text_config.hidden_size)
        if self.lora_enabled:
            self._assert_only_lora_trainable()

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        with torch.no_grad() if not self.lora_enabled else nullcontext():
            return self._forward_impl(batch)

    def _forward_impl(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        images = _require_tensor(batch, "images")
        instructions = batch.get("instruction")
        if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
            raise TypeError("batch['instruction'] must be a sequence of strings")
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 2:
            raise ValueError(f"QwenVLTokenEncoder expects two views, got {images.shape[1]}")

        messages = [
            _make_qwen_message(
                agentview=_tensor_chw_to_pil(images[index, 0]),
                wrist=_tensor_chw_to_pil(images[index, 1]),
                instruction=str(instructions[index]),
            )
            for index in range(images.shape[0])
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        image_inputs = _extract_qwen_images(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if isinstance(value, Tensor) else value for key, value in inputs.items()}
        base_model = getattr(self.model, "model", self.model)
        outputs = base_model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            attention_mask = attention_mask.bool()
        hidden = hidden if self.lora_enabled else hidden.detach()
        return {
            "context_tokens": hidden,
            "context_attention_mask": attention_mask.detach(),
        }

    def save_lora_adapter(self, path: str | Path) -> None:
        if not self.lora_enabled:
            return
        if not hasattr(self.model, "save_pretrained"):
            raise RuntimeError("Qwen model does not support save_pretrained for LoRA adapter saving")
        self.model.save_pretrained(str(path))

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _enable_lora(self, lora_scope: str, r: int, alpha: int, dropout: float) -> None:
        if lora_scope != "vision":
            raise ValueError(f"Only vision LoRA is supported in this path, got {lora_scope!r}")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional heavy dependency.
            raise ImportError("Install peft>=0.17.0 to enable QwenVL LoRA fine-tuning.") from exc

        target_modules = _find_visual_lora_target_modules(self.model)
        if not target_modules:
            raise RuntimeError("No Qwen visual Linear modules found for LoRA injection")
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        self.lora_enabled = True
        self._assert_only_lora_trainable()

    def _load_lora_adapter(self, path: str | Path) -> None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional heavy dependency.
            raise ImportError("Install peft>=0.17.0 to load a QwenVL LoRA adapter.") from exc

        self.model = PeftModel.from_pretrained(self.model, str(path), is_trainable=False)
        self.lora_enabled = False
        for param in self.model.parameters():
            param.requires_grad = False

    def _assert_only_lora_trainable(self) -> None:
        bad = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad
            and ("lora" not in name.lower() or not (name.startswith("visual.") or ".visual." in name))
        ]
        if bad:
            preview = ", ".join(bad[:5])
            raise RuntimeError(f"Only visual LoRA params may be trainable; found: {preview}")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._make_pe(max_len), persistent=False)

    def forward(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        if seq_len > self.pe.shape[1]:
            self.pe = self._make_pe(seq_len).to(self.pe.device)
        pe = self.pe[:, :seq_len]
        if device is not None or dtype is not None:
            pe = pe.to(device=device, dtype=dtype)
        return pe

    def _make_pe(self, max_len: int) -> Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.dim))
        pe = torch.zeros(max_len, self.dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)


class ActionTokenEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.pos = SinusoidalPositionalEncoding(embed_dim, max_len=horizon)

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.horizon:
            raise ValueError(f"Expected actions [B, {self.horizon}, D], got {tuple(actions.shape)}")
        tokens = self.net(actions)
        return tokens + self.pos(self.horizon, device=actions.device, dtype=tokens.dtype)


class CrossAttentionFlowBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        action_tokens: Tensor,
        context_tokens: Tensor,
        time_emb: Tensor,
        context_attention_mask: Tensor | None,
    ) -> Tensor:
        x = action_tokens
        self_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self_out
        key_padding_mask = None
        if context_attention_mask is not None:
            key_padding_mask = ~context_attention_mask.bool()
        cross_out, _ = self.cross_attn(
            self.norm2(x),
            context_tokens,
            context_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + cross_out
        x = x + self.ff(self.norm3(x) + time_emb[:, None, :])
        return x


class TokenFlowActionHead(nn.Module):
    """Flow matching denoiser that cross-attends action tokens to VLM tokens."""

    def __init__(
        self,
        config: QwenVLFlowConfig,
        action_mean: Tensor | None = None,
        action_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context_proj = nn.Linear(config.context_dim, config.embed_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )
        self.action_encoder = ActionTokenEncoder(
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
            horizon=config.horizon,
        )
        self.time_pos_enc = SinusoidalPositionalEncoding(config.embed_dim, max_len=1000)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionFlowBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    hidden_dim=config.embed_dim * 4,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.action_dim)
        mean = torch.zeros(config.action_dim) if action_mean is None else action_mean.detach().float()
        std = torch.ones(config.action_dim) if action_std is None else action_std.detach().float()
        self.register_buffer("action_mean", _format_action_stat(mean, config.action_dim))
        self.register_buffer("action_std", _format_action_stat(std, config.action_dim).clamp_min(config.action_eps))

    def normalize_actions(self, actions: Tensor) -> Tensor:
        return (actions - self.action_mean.to(dtype=actions.dtype)) / self.action_std.to(dtype=actions.dtype)

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        return actions * self.action_std.to(dtype=actions.dtype) + self.action_mean.to(dtype=actions.dtype)

    def velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        context_tokens: Tensor,
        state: Tensor,
        context_attention_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_actions(x_t, "x_t")
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t must have shape ({x_t.shape[0]},), got {tuple(t.shape)}")
        context = self.context_proj(context_tokens.to(dtype=x_t.dtype))
        state_token = self.state_encoder(state.to(dtype=x_t.dtype)).unsqueeze(1)
        context = torch.cat([context, state_token], dim=1)
        if context_attention_mask is not None:
            state_mask = torch.ones((context_attention_mask.shape[0], 1), dtype=torch.bool, device=context_attention_mask.device)
            context_attention_mask = torch.cat([context_attention_mask.bool(), state_mask], dim=1)

        time_index = (t.clamp(0, 1) * 999).long()
        time_table = self.time_pos_enc(1000, device=x_t.device, dtype=x_t.dtype).squeeze(0)
        time_emb = time_table.index_select(0, time_index)
        x = self.action_encoder(x_t)
        for block in self.blocks:
            x = block(x, context, time_emb, context_attention_mask)
        return self.output(self.norm_out(x))

    def loss(
        self,
        actions: Tensor,
        context_tokens: Tensor,
        state: Tensor,
        context_attention_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._validate_actions(actions, "actions")
        x1 = self.normalize_actions(actions)
        x0 = torch.rand_like(x1) * 2.0 - 1.0
        t = torch.distributions.Beta(2.0, 2.0).sample((actions.shape[0],)).to(device=actions.device, dtype=actions.dtype)
        t = t.clamp(0.02, 0.98)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(x_t, t, context_tokens, state, context_attention_mask)
        squared_error = (pred_velocity - target_velocity).pow(2)
        if action_is_pad is None:
            loss = squared_error.mean()
        else:
            if action_is_pad.shape != actions.shape[:2]:
                raise ValueError(f"action_is_pad must have shape {tuple(actions.shape[:2])}, got {tuple(action_is_pad.shape)}")
            valid = (~action_is_pad.bool()).to(dtype=squared_error.dtype, device=squared_error.device)
            loss = (squared_error * valid[..., None]).sum() / (valid.sum() * self.config.action_dim).clamp_min(1.0)
        return {
            "loss": loss,
            "fm_loss": loss.detach(),
            "pred_velocity_norm": pred_velocity.detach().norm(dim=-1).mean(),
            "target_velocity_norm": target_velocity.detach().norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        context_tokens: Tensor,
        state: Tensor,
        context_attention_mask: Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        steps = int(num_steps or self.config.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        batch_size = context_tokens.shape[0]
        sample_dtype = next(self.parameters()).dtype
        x = torch.rand(
            batch_size,
            self.config.horizon,
            self.config.action_dim,
            device=context_tokens.device,
            dtype=sample_dtype,
            generator=generator,
        ) * 2.0 - 1.0
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((batch_size,), step * dt, device=context_tokens.device, dtype=context_tokens.dtype)
            x = x + dt * self.velocity(x, t, context_tokens, state, context_attention_mask)
        return self.denormalize_actions(x).clamp(-1.0, 1.0)

    def _validate_actions(self, actions: Tensor, name: str) -> None:
        expected = (self.config.horizon, self.config.action_dim)
        if actions.ndim != 3 or actions.shape[1:] != expected:
            raise ValueError(f"{name} must have shape (B, {expected[0]}, {expected[1]}), got {tuple(actions.shape)}")


class QwenVLFlowPolicy(nn.Module):
    def __init__(
        self,
        config: QwenVLFlowConfig,
        dataset_stats: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        action_mean, action_std = _extract_action_stats(dataset_stats, config.action_dim)
        self.config = config
        self.action_head = TokenFlowActionHead(config, action_mean=action_mean, action_std=action_std)
        self._cached_action_chunk: Tensor | None = None
        self._cached_action_step = 0

    def reset(self) -> None:
        self._cached_action_chunk = None
        self._cached_action_step = 0

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return self.action_head.loss(
            actions=batch[ACTION],
            context_tokens=batch["context_tokens"],
            context_attention_mask=batch.get("context_attention_mask"),
            state=batch["state"],
            action_is_pad=batch.get("action_is_pad"),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: Mapping[str, Tensor], num_steps: int | None = None) -> Tensor:
        return self.action_head.sample(
            context_tokens=batch["context_tokens"],
            context_attention_mask=batch.get("context_attention_mask"),
            state=batch["state"],
            num_steps=num_steps,
        )

    @torch.no_grad()
    def select_action(self, batch: Mapping[str, Tensor], num_steps: int | None = None) -> Tensor:
        if self._cached_action_chunk is None or self._cached_action_step >= self.config.n_action_steps:
            self._cached_action_chunk = self.predict_action_chunk(batch, num_steps=num_steps)
            self._cached_action_step = 0
        action = self._cached_action_chunk[:, self._cached_action_step]
        self._cached_action_step += 1
        return action


def _make_qwen_message(agentview: Image.Image, wrist: Image.Image, instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": agentview},
                {"type": "image", "image": wrist},
                {"type": "text", "text": f"Robot instruction: {instruction}"},
            ],
        }
    ]


def _find_visual_lora_target_modules(model: nn.Module) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith("visual.") or ".visual." in name:
            targets.append(name)
    return sorted(targets)


def _extract_qwen_images(messages: list[list[dict[str, Any]]]) -> list[Image.Image]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return [
            content["image"]
            for message in messages
            for turn in message
            for content in turn["content"]
            if content.get("type") == "image"
        ]
    image_inputs: list[Image.Image] = []
    for message in messages:
        images, _ = process_vision_info(message)
        image_inputs.extend(images)
    return image_inputs


def _tensor_chw_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected image [C, H, W], got {tuple(image.shape)}")
    if image.shape[0] not in (1, 3):
        raise ValueError(f"Expected channel-first image, got {tuple(image.shape)}")
    if image.max() <= 2:
        image = image * 255.0
    array = image.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def _require_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch['{key}'] must be a torch.Tensor")
    return value


def _resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _format_action_stat(stat: Tensor, action_dim: int) -> Tensor:
    stat = stat.detach().float()
    if stat.shape != (action_dim,):
        raise ValueError(f"action stat must have shape ({action_dim},), got {tuple(stat.shape)}")
    return stat.reshape(1, 1, action_dim)


def _extract_stat_value(stats: Mapping[str, Any], key: str) -> Tensor | None:
    if key not in stats:
        return None
    value = stats[key]
    if isinstance(value, Tensor):
        return value
    if hasattr(value, "values"):
        return torch.as_tensor(value.values)
    return torch.as_tensor(value)


def _extract_action_stats(dataset_stats: Mapping[str, Any] | None, action_dim: int) -> tuple[Tensor, Tensor]:
    if not dataset_stats:
        return torch.zeros(action_dim), torch.ones(action_dim)
    action_stats = dataset_stats.get(ACTION) or dataset_stats.get("action")
    if isinstance(action_stats, Mapping):
        mean = _extract_stat_value(action_stats, "mean")
        std = _extract_stat_value(action_stats, "std")
        if mean is not None and std is not None:
            return mean.float(), std.float()
    return torch.zeros(action_dim), torch.ones(action_dim)
