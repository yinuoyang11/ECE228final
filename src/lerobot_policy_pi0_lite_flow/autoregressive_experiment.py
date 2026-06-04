"""Shared utilities for the CLIP autoregressive LIBERO baseline experiment."""
from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .configuration_autoregressive import PI0LiteAutoregressiveConfig
from .libero_adapter import LIBEROActionChunkDataset
from .modeling_autoregressive import ACTION, COND_KEY, AutoregressivePolicy
from .representation_encoder import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    MockImageEncoder,
    MockTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


def resolve_device(requested: str) -> torch.device:
    """Resolve a device, raising when an explicitly requested accelerator is unavailable."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_task_descriptions(repo_id: str) -> dict[int, str]:
    """Load task_index -> language instruction from the dataset metadata."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/tasks.parquet")
    rows = pq.read_table(path).to_pylist()
    descriptions: dict[int, str] = {}
    for row in rows:
        task_index = int(row["task_index"])
        description = row.get("task") or row.get("instruction") or row.get("__index_level_0__")
        if description:
            descriptions[task_index] = str(description)
    return descriptions


def load_libero_action_dataset(
    repo_id: str,
    horizon: int,
    task_indices: Sequence[int],
) -> tuple[Any, LIBEROActionChunkDataset, dict[int, str]]:
    from datasets import load_dataset

    raw_dataset = load_dataset(repo_id, split="train")
    descriptions = load_task_descriptions(repo_id)
    dataset = LIBEROActionChunkDataset(
        base_dataset=raw_dataset,
        horizon=horizon,
        task_indices=task_indices,
        task_descriptions=descriptions,
    )
    return raw_dataset, dataset, descriptions


def raw_indices(dataset: LIBEROActionChunkDataset) -> list[int]:
    if dataset.selected_indices is None:
        return list(range(len(dataset.dataset)))
    return list(dataset.selected_indices)


def split_by_episode(
    dataset: LIBEROActionChunkDataset,
    eval_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Return disjoint train/eval local indices and episode IDs."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    episode_column = _column(dataset.dataset, "episode_index")
    selected_raw = raw_indices(dataset)
    episode_ids = sorted({int(episode_column[index]) for index in selected_raw})
    if len(episode_ids) < 2:
        raise ValueError("At least two episodes are required for an episode-level split")

    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    eval_count = max(1, round(len(episode_ids) * eval_fraction))
    eval_count = min(eval_count, len(episode_ids) - 1)
    eval_episodes = set(episode_ids[:eval_count])
    train_episodes = set(episode_ids[eval_count:])

    train_local = []
    eval_local = []
    for local_index, raw_index in enumerate(selected_raw):
        episode_id = int(episode_column[raw_index])
        if episode_id in train_episodes:
            train_local.append(local_index)
        elif episode_id in eval_episodes:
            eval_local.append(local_index)

    if train_episodes & eval_episodes:
        raise AssertionError("Train and eval episodes overlap")
    return {
        "train_indices": train_local,
        "eval_indices": eval_local,
        "train_episode_ids": sorted(train_episodes),
        "eval_episode_ids": sorted(eval_episodes),
        "train_raw_indices": [selected_raw[index] for index in train_local],
        "eval_raw_indices": [selected_raw[index] for index in eval_local],
    }


def compute_action_bounds(raw_dataset: Any, train_raw_indices: Sequence[int], action_dim: int) -> dict[str, Tensor]:
    """Compute 1st/99th percentile action bounds without decoding images."""
    action_source = raw_dataset.select(train_raw_indices).select_columns(["action"])
    actions = torch.as_tensor(action_source["action"], dtype=torch.float32).reshape(-1, action_dim)
    lo = torch.quantile(actions, 0.01, dim=0)
    hi = torch.quantile(actions, 0.99, dim=0)
    hi = torch.where(hi - lo < 1e-6, lo + 1e-6, hi)
    return {"min": lo, "max": hi, "mean": actions.mean(dim=0), "std": actions.std(dim=0)}


def collate_raw(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "instruction": [str(item["instruction"]) for item in batch],
        ACTION: torch.stack([item[ACTION] for item in batch]),
        "action_is_pad": torch.stack([item["action_is_pad"] for item in batch]),
    }


class CachedFeatureDataset(Dataset):
    """In-memory tensors produced by frozen image/text backbones."""

    def __init__(self, tensors: Mapping[str, Tensor]) -> None:
        if not tensors:
            raise ValueError("CachedFeatureDataset requires tensors")
        lengths = {tensor.shape[0] for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("All cached tensors must have the same leading dimension")
        self.tensors = dict(tensors)
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


@torch.no_grad()
def cache_frozen_features(
    encoder: RepresentationEncoder,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> CachedFeatureDataset:
    """Precompute frozen CLIP image/text features while preserving trainable state input."""
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_raw,
    )
    cached: dict[str, list[Tensor]] = {
        "image_features": [],
        "text_features": [],
        "state": [],
        ACTION: [],
        "action_is_pad": [],
    }
    for batch in loader:
        images = batch["images"].to(device)
        features = encoder.extract_features(
            {
                "images": images,
                "instruction": batch["instruction"],
            }
        )
        cached["image_features"].append(features["image_features"].cpu())
        if "text_features" in features:
            cached["text_features"].append(features["text_features"].cpu())
        cached["state"].append(batch["state"].cpu())
        cached[ACTION].append(batch[ACTION].cpu())
        cached["action_is_pad"].append(batch["action_is_pad"].cpu())

    tensors = {key: torch.cat(parts, dim=0) for key, parts in cached.items() if parts}
    return CachedFeatureDataset(tensors)


def cached_batch_to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def condition_from_cached(encoder: RepresentationEncoder, batch: Mapping[str, Tensor]) -> Tensor:
    return encoder.forward_from_features(
        image_features=batch["image_features"],
        state=batch["state"],
        text_features=batch.get("text_features"),
    )


def build_encoder(args: Mapping[str, Any]) -> RepresentationEncoder:
    config = RepresentationEncoderConfig(
        image_dim=512,
        text_dim=512,
        state_dim=int(args["state_dim"]),
        cond_dim=int(args["cond_dim"]),
        num_views=2,
        use_language=True,
        dropout=float(args.get("encoder_dropout", 0.0)),
    )
    if args["encoder"] == "clip":
        image_encoder = CLIPImageEncoder(args["clip_model"], freeze=True)
        text_encoder = CLIPTextEncoder(args["clip_model"], freeze=True)
    elif args["encoder"] == "mock":
        image_encoder = MockImageEncoder(config.image_dim)
        text_encoder = MockTextEncoder(config.text_dim)
    else:
        raise ValueError(f"Unknown encoder: {args['encoder']}")
    return RepresentationEncoder(config, image_encoder=image_encoder, text_encoder=text_encoder)


def build_policy(args: Mapping[str, Any], action_stats: Mapping[str, Tensor]) -> AutoregressivePolicy:
    config = PI0LiteAutoregressiveConfig(
        horizon=int(args["horizon"]),
        action_dim=int(args["action_dim"]),
        cond_dim=int(args["cond_dim"]),
        num_bins=int(args["num_bins"]),
        hidden_dim=int(args["hidden_dim"]),
        nhead=int(args["nhead"]),
        num_layers=int(args["num_layers"]),
        dim_feedforward=int(args["dim_feedforward"]),
        dropout=float(args["dropout"]),
    )
    return AutoregressivePolicy(config, dataset_stats={ACTION: dict(action_stats)})


def trainable_encoder_state_dict(encoder: RepresentationEncoder) -> dict[str, Tensor]:
    """Exclude frozen CLIP backbones from checkpoints."""
    trainable_names = {name for name, param in encoder.named_parameters() if param.requires_grad}
    return {
        name: value
        for name, value in encoder.state_dict().items()
        if name in trainable_names or not (
            name.startswith("image_encoder.model.") or name.startswith("text_encoder.model.")
        )
    }


def load_trainable_encoder_state_dict(encoder: RepresentationEncoder, state_dict: Mapping[str, Tensor]) -> None:
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    bad_missing = [
        key
        for key in missing
        if not (key.startswith("image_encoder.model.") or key.startswith("text_encoder.model."))
    ]
    if bad_missing:
        raise RuntimeError(f"Missing non-CLIP encoder checkpoint keys: {bad_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected}")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tensor_dict_to_lists(values: Mapping[str, Tensor]) -> dict[str, list[float]]:
    return {key: value.detach().cpu().tolist() for key, value in values.items()}


def tensor_dict_from_lists(values: Mapping[str, Sequence[float]]) -> dict[str, Tensor]:
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}


def masked_action_error_sums(prediction: Tensor, target: Tensor, action_is_pad: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    valid = (~action_is_pad.bool()).unsqueeze(-1).expand_as(target)
    difference = prediction - target
    count = valid.sum()
    mse_sum = (difference.square() * valid).sum()
    l1_sum = (difference.abs() * valid).sum()
    return mse_sum, l1_sum, count


def _column(dataset: Any, key: str) -> Sequence[Any]:
    if hasattr(dataset, "column_names") and key in dataset.column_names:
        return dataset[key]
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and key in hf_dataset.column_names:
        return hf_dataset[key]
    raise KeyError(f"Dataset has no column {key!r}")
