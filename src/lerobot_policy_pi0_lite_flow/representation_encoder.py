from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modeling_pi0_lite_flow import COND_KEY


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@dataclass
class RepresentationEncoderConfig:
    """Config for the upstream observation encoder.

    The default output width intentionally matches PI0LiteFlowConfig.cond_dim.
    """

    image_dim: int = 512
    text_dim: int = 512
    state_dim: int = 8
    state_embed_dim: int = 128
    cond_dim: int = 256
    fusion_hidden_dim: int = 512
    num_views: int = 2
    use_language: bool = True
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MockImageEncoder(nn.Module):
    """Tiny image encoder for local shape tests without downloading CLIP weights."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.LazyLinear(output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, C, H, W], got {tuple(images.shape)}")
        batch_size, num_views = images.shape[:2]
        flat_images = images.reshape(batch_size * num_views, *images.shape[2:]).float()
        if flat_images.max() > 2:
            flat_images = flat_images / 255.0
        features = self.net(flat_images)
        return features.reshape(batch_size, num_views, self.output_dim)


class MockTextEncoder(nn.Module):
    """Deterministic text encoder for tests that should not require model downloads."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(32, output_dim)

    def forward(self, instructions: Sequence[str], device: torch.device | None = None) -> Tensor:
        rows = []
        for instruction in instructions:
            encoded = instruction.encode("utf-8", errors="ignore")[:32]
            row = torch.zeros(32, dtype=torch.float32)
            if encoded:
                row[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.float32) / 255.0
            rows.append(row)
        features = torch.stack(rows, dim=0)
        if device is not None:
            features = features.to(device)
        return self.proj(features)


class CLIPImageEncoder(nn.Module):
    """CLIP image encoder backed by Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        trainable_layers: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as exc:  # pragma: no cover - dependency is optional for core tests.
            raise ImportError("Install transformers to use CLIPImageEncoder.") from exc

        self.model = CLIPModel.from_pretrained(model_name)
        self.output_dim = self.model.config.projection_dim
        trainable_layers = 0 if freeze and trainable_layers is None else trainable_layers
        _set_clip_branch_trainable(
            self.model,
            branch_name="vision_model",
            projection_name="visual_projection",
            final_norm_name="post_layernorm",
            trainable_layers=-1 if trainable_layers is None else trainable_layers,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, C, H, W], got {tuple(images.shape)}")
        batch_size, num_views = images.shape[:2]
        flat_images = images.reshape(batch_size * num_views, *images.shape[2:])
        pixel_values = preprocess_clip_images(flat_images)

        grad_enabled = any(param.requires_grad for param in self.model.vision_model.parameters())
        with torch.set_grad_enabled(grad_enabled):
            features = self.model.get_image_features(pixel_values=pixel_values)
        features = _clip_output_to_tensor(features, projection=getattr(self.model, "visual_projection", None))
        return features.reshape(batch_size, num_views, -1)

    def train(self, mode: bool = True) -> CLIPImageEncoder:
        super().train(mode)
        if not any(param.requires_grad for param in self.model.parameters()):
            self.model.eval()
        return self


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder backed by Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        trainable_layers: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError as exc:  # pragma: no cover - dependency is optional for core tests.
            raise ImportError("Install transformers to use CLIPTextEncoder.") from exc

        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.projection_dim
        trainable_layers = 0 if freeze and trainable_layers is None else trainable_layers
        _set_clip_branch_trainable(
            self.model,
            branch_name="text_model",
            projection_name="text_projection",
            final_norm_name="final_layer_norm",
            trainable_layers=-1 if trainable_layers is None else trainable_layers,
        )

    def forward(self, instructions: Sequence[str], device: torch.device | None = None) -> Tensor:
        encoded = self.tokenizer(list(instructions), padding=True, truncation=True, return_tensors="pt")
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}

        grad_enabled = any(param.requires_grad for param in self.model.text_model.parameters())
        with torch.set_grad_enabled(grad_enabled):
            features = self.model.get_text_features(**encoded)
        return _clip_output_to_tensor(features, projection=getattr(self.model, "text_projection", None))

    def train(self, mode: bool = True) -> CLIPTextEncoder:
        super().train(mode)
        if not any(param.requires_grad for param in self.model.parameters()):
            self.model.eval()
        return self


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = MLP(state_dim, max(128, embed_dim), embed_dim)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state.float())


class RepresentationEncoder(nn.Module):
    """Encode image, state, and language observations into batch[observation.cond]."""

    def __init__(
        self,
        config: RepresentationEncoderConfig | None = None,
        image_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RepresentationEncoderConfig()
        self.image_encoder = image_encoder or MockImageEncoder(self.config.image_dim)
        self.text_encoder = text_encoder or MockTextEncoder(self.config.text_dim)
        self.state_encoder = StateEncoder(self.config.state_dim, self.config.state_embed_dim)

        image_input_dim = self.config.image_dim * self.config.num_views
        self.image_projection = MLP(
            image_input_dim,
            self.config.fusion_hidden_dim,
            self.config.image_dim,
            self.config.dropout,
        )

        fusion_input_dim = self.config.image_dim + self.config.state_embed_dim
        if self.config.use_language:
            fusion_input_dim += self.config.text_dim
        self.fusion = MLP(
            fusion_input_dim,
            self.config.fusion_hidden_dim,
            self.config.cond_dim,
            self.config.dropout,
        )

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        features = self.extract_features(batch)
        return self.forward_from_features(
            image_features=features["image_features"],
            state=_require_tensor(batch, "state"),
            text_features=features.get("text_features"),
        )

    def extract_features(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Extract image and text backbone features for optional frozen caching."""
        images = _require_tensor(batch, "images")

        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, V, C, H, W] or [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != self.config.num_views:
            raise ValueError(f"Configured num_views={self.config.num_views}, got {images.shape[1]}")

        image_features = self.image_encoder(images)
        features = {"image_features": image_features}
        if self.config.use_language:
            instructions = batch.get("instruction")
            if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
                raise TypeError("batch['instruction'] must be a sequence of strings")
            features["text_features"] = self.text_encoder(instructions, device=images.device)
        return features

    def forward_from_features(
        self,
        image_features: Tensor,
        state: Tensor,
        text_features: Tensor | None = None,
    ) -> Tensor:
        """Fuse cached backbone features with the trainable state encoder."""
        if image_features.ndim != 3:
            raise ValueError(
                f"Expected image_features [B, V, D], got {tuple(image_features.shape)}"
            )
        if image_features.shape[1] != self.config.num_views:
            raise ValueError(f"Configured num_views={self.config.num_views}, got {image_features.shape[1]}")
        if image_features.shape[2] != self.config.image_dim:
            raise ValueError(f"Configured image_dim={self.config.image_dim}, got {image_features.shape[2]}")

        image_emb = self.image_projection(image_features.flatten(start_dim=1))
        state_emb = self.state_encoder(state)
        pieces = [image_emb, state_emb]
        if self.config.use_language:
            if text_features is None:
                raise ValueError("text_features are required when use_language=True")
            if text_features.ndim != 2 or text_features.shape[-1] != self.config.text_dim:
                raise ValueError(
                    f"Expected text_features [B, {self.config.text_dim}], got {tuple(text_features.shape)}"
                )
            pieces.append(text_features)

        return self.fusion(torch.cat(pieces, dim=-1))

    def add_condition_to_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(batch)
        output[COND_KEY] = self.forward(batch)
        return output


def preprocess_clip_images(images: Tensor) -> Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB images, got {images.shape[1]} channels")
    images = images.float()
    if images.max() > 2:
        images = images / 255.0
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = CLIP_MEAN.to(device=images.device, dtype=images.dtype)
    std = CLIP_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def _require_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch['{key}'] must be a torch.Tensor")
    return value


def _clip_output_to_tensor(output: Any, projection: nn.Module | None = None) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        pooled = output.pooler_output
        if projection is not None and hasattr(projection, "in_features") and pooled.shape[-1] == projection.in_features:
            return projection(pooled)
        return pooled
    raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")


def _set_clip_branch_trainable(
    model: nn.Module,
    branch_name: str,
    projection_name: str,
    final_norm_name: str,
    trainable_layers: int,
) -> None:
    if trainable_layers < -1:
        raise ValueError("trainable_layers must be -1, 0, or a positive integer")

    for param in model.parameters():
        param.requires_grad = False
    if trainable_layers == 0:
        model.eval()
        return

    branch = getattr(model, branch_name)
    projection = getattr(model, projection_name)
    if trainable_layers == -1:
        modules = [branch, projection]
    else:
        layers = branch.encoder.layers
        if trainable_layers > len(layers):
            raise ValueError(f"Cannot unfreeze {trainable_layers} layers; {branch_name} only has {len(layers)}")
        modules = [*layers[-trainable_layers:], getattr(branch, final_norm_name), projection]

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
