from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset
from torch import Tensor


DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_INSTRUCTION_KEYS = ("task", "instruction", "language_instruction", "prompt")


def libero_samples_to_encoder_batch(
    samples: Sequence[Mapping[str, Any]],
    image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
    state_key: str = "observation.state",
    instruction_keys: Sequence[str] = DEFAULT_INSTRUCTION_KEYS,
) -> dict[str, Any]:
    """Convert LeRobot LIBERO samples into RepresentationEncoder input format."""

    if not samples:
        raise ValueError("At least one sample is required")

    images = []
    states = []
    instructions = []
    for sample in samples:
        images.append(_stack_sample_images(sample, image_keys))
        states.append(_to_tensor(sample[state_key]).float())
        instructions.append(_read_instruction(sample, instruction_keys))

    return {
        "images": torch.stack(images, dim=0),
        "state": torch.stack(states, dim=0),
        "instruction": instructions,
    }


class LIBEROActionChunkDataset(Dataset):
    """Wrap a LeRobot LIBERO dataset and add future action chunks.

    Each item keeps the current observation at time t and returns actions
    [a_t, ..., a_{t+H-1}] without crossing episode boundaries. Timesteps past the
    end of an episode are zero-filled and marked in action_is_pad.
    """

    def __init__(
        self,
        repo_id: str = "HuggingFaceVLA/libero",
        horizon: int = 16,
        base_dataset: Any | None = None,
        task_indices: Sequence[int] | None = None,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = "observation.state",
        action_key: str = "action",
        instruction_keys: Sequence[str] = DEFAULT_INSTRUCTION_KEYS,
        task_descriptions: Mapping[int, str] | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        self.horizon = horizon
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.instruction_keys = tuple(instruction_keys)
        self.task_descriptions = dict(task_descriptions or {})
        self.dataset = base_dataset if base_dataset is not None else self._load_lerobot_dataset(repo_id)
        self.episode_ranges = _episode_ranges(self.dataset)
        self.index_to_episode_end = _index_to_episode_end(self.episode_ranges)
        self.action_source = _make_action_source(self.dataset, action_key)
        self.task_indices = tuple(sorted(set(task_indices))) if task_indices is not None else None
        self.selected_indices = _indices_for_tasks(self.dataset, self.episode_ranges, self.task_indices)

    def __len__(self) -> int:
        return len(self.selected_indices) if self.selected_indices is not None else len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.selected_indices is not None:
            index = self.selected_indices[index]
        sample = self.dataset[index]
        action_chunk, action_is_pad = self._action_chunk(index)
        batch = libero_samples_to_encoder_batch(
            [sample],
            image_keys=self.image_keys,
            state_key=self.state_key,
            instruction_keys=self.instruction_keys,
        )
        instruction = batch["instruction"][0]
        if not instruction and self.task_descriptions:
            task_index = _read_task_index(sample)
            instruction = self.task_descriptions.get(task_index, "")
        return {
            "images": batch["images"].squeeze(0),
            "state": batch["state"].squeeze(0),
            "instruction": instruction,
            self.action_key: action_chunk,
            "action_is_pad": action_is_pad,
        }

    def _action_chunk(self, index: int) -> tuple[Tensor, Tensor]:
        episode_end = self.index_to_episode_end[index]
        first_action = _to_tensor(_get_action(self.action_source, index, self.action_key)).float()
        action_shape = first_action.shape
        action_chunk = torch.zeros((self.horizon, *action_shape), dtype=first_action.dtype)
        action_is_pad = torch.ones(self.horizon, dtype=torch.bool)

        for offset in range(self.horizon):
            future_index = index + offset
            if future_index >= episode_end:
                break
            action_chunk[offset] = _to_tensor(_get_action(self.action_source, future_index, self.action_key)).float()
            action_is_pad[offset] = False

        return action_chunk, action_is_pad

    @staticmethod
    def _load_lerobot_dataset(repo_id: str):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - exercised only when LeRobot is missing.
            raise ImportError("Install lerobot[dataset] to load LIBEROActionChunkDataset by repo_id.") from exc
        return LeRobotDataset(repo_id)


def _stack_sample_images(sample: Mapping[str, Any], image_keys: Sequence[str]) -> Tensor:
    views = []
    for key in image_keys:
        if key in sample:
            views.append(_image_to_chw(sample[key]))
    if not views:
        available = ", ".join(sample.keys())
        raise KeyError(f"No image keys found among {tuple(image_keys)}. Available keys: {available}")
    return torch.stack(views, dim=0)


def _image_to_chw(image: Any) -> Tensor:
    tensor = _to_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        return tensor
    if tensor.shape[-1] in (1, 3):
        return tensor.permute(2, 0, 1).contiguous()
    raise ValueError(f"Cannot infer channel dimension for image shape {tuple(tensor.shape)}")


def _read_instruction(sample: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return str(value)
    return ""


def _read_task_index(sample: Mapping[str, Any]) -> int:
    if "task_index" not in sample:
        raise KeyError("Sample has no task_index for task description lookup")
    return int(_to_tensor(sample["task_index"]).item())


def _to_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    # Handle PIL images returned by HuggingFace datasets
    if hasattr(value, "mode") and hasattr(value, "size") and callable(getattr(value, "convert", None)):
        import numpy as np
        value = np.array(value)
    return torch.as_tensor(value)


def _episode_ranges(dataset: Any) -> list[tuple[int, int]]:
    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None)
    if episodes is not None:
        ranges = []
        for row in episodes:
            if "dataset_from_index" in row and "dataset_to_index" in row:
                ranges.append((int(row["dataset_from_index"]), int(row["dataset_to_index"])))
        if ranges:
            return ranges

    # Check if we can extract the column directly (fast path for HuggingFace Dataset)
    if hasattr(dataset, "column_names") and "episode_index" in dataset.column_names:
        try:
            episode_indices = dataset["episode_index"]
            ranges = []
            if len(episode_indices) > 0:
                current_start = 0
                current_episode = int(episode_indices[0])
                for idx, episode in enumerate(episode_indices):
                    episode_val = int(episode)
                    if episode_val != current_episode:
                        ranges.append((current_start, idx))
                        current_episode = episode_val
                        current_start = idx
                ranges.append((current_start, len(episode_indices)))
            return ranges
        except Exception:
            pass

    # Also check if it wraps a hf_dataset (LeRobotDataset style)
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "column_names") and "episode_index" in hf_dataset.column_names:
        try:
            episode_indices = hf_dataset["episode_index"]
            ranges = []
            if len(episode_indices) > 0:
                current_start = 0
                current_episode = int(episode_indices[0])
                for idx, episode in enumerate(episode_indices):
                    episode_val = int(episode)
                    if episode_val != current_episode:
                        ranges.append((current_start, idx))
                        current_episode = episode_val
                        current_start = idx
                ranges.append((current_start, len(episode_indices)))
            return ranges
        except Exception:
            pass

    ranges = []
    current_start = 0
    current_episode = None
    for idx in range(len(dataset)):
        episode = int(_to_tensor(dataset[idx]["episode_index"]).item())
        if current_episode is None:
            current_episode = episode
            current_start = idx
        elif episode != current_episode:
            ranges.append((current_start, idx))
            current_episode = episode
            current_start = idx
    if current_episode is not None:
        ranges.append((current_start, len(dataset)))
    return ranges


def _index_to_episode_end(ranges: Sequence[tuple[int, int]]) -> list[int]:
    if not ranges:
        return []
    total = max(end for _, end in ranges)
    ends = [0] * total
    for start, end in ranges:
        for idx in range(start, end):
            ends[idx] = end
    return ends


def _make_action_source(dataset: Any, action_key: str) -> Any:
    if hasattr(dataset, "select_columns"):
        try:
            return dataset.select_columns([action_key])
        except Exception:
            pass
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "select_columns"):
        try:
            return hf_dataset.select_columns([action_key])
        except Exception:
            return dataset
    return dataset


def _get_action(source: Any, index: int, action_key: str) -> Any:
    return source[index][action_key]


def _indices_for_tasks(
    dataset: Any,
    episode_ranges: Sequence[tuple[int, int]],
    task_indices: Sequence[int] | None,
) -> list[int] | None:
    if task_indices is None:
        return None
    if not task_indices:
        raise ValueError("task_indices must contain at least one task index")
    if any(task_index < 0 for task_index in task_indices):
        raise ValueError("task_indices must be non-negative")

    selected_tasks = set(task_indices)
    episode_rows = _episode_rows(dataset)
    if len(episode_rows) == len(episode_ranges):
        episode_task_indices = [_episode_task_index(row) for row in episode_rows]
        if all(task_index is not None for task_index in episode_task_indices):
            return [
                index
                for (start, end), task_index in zip(episode_ranges, episode_task_indices, strict=True)
                if task_index in selected_tasks
                for index in range(start, end)
            ]

    # Fast path for HuggingFace Dataset
    if hasattr(dataset, "column_names") and "task_index" in dataset.column_names:
        try:
            task_indices_col = dataset["task_index"]
            return [
                index
                for index, task_val in enumerate(task_indices_col)
                if int(task_val) in selected_tasks
            ]
        except Exception:
            pass

    # Fast path for LeRobotDataset wrapper
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "column_names") and "task_index" in hf_dataset.column_names:
        try:
            task_indices_col = hf_dataset["task_index"]
            return [
                index
                for index, task_val in enumerate(task_indices_col)
                if int(task_val) in selected_tasks
            ]
        except Exception:
            pass

    task_source = _make_task_source(dataset)
    return [
        index
        for index in range(len(dataset))
        if int(_to_tensor(task_source[index]["task_index"]).item()) in selected_tasks
    ]


def _episode_rows(dataset: Any) -> list[Mapping[str, Any]]:
    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None)
    if episodes is None:
        return []
    try:
        return list(episodes)
    except TypeError:
        return []


def _episode_task_index(row: Mapping[str, Any]) -> int | None:
    for key in ("task_index", "stats/task_index/min"):
        if key not in row:
            continue
        value = _to_tensor(row[key])
        if value.numel() == 1:
            return int(value.item())
    return None


def _make_task_source(dataset: Any) -> Any:
    if hasattr(dataset, "select_columns"):
        try:
            return dataset.select_columns(["task_index"])
        except Exception:
            pass
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "select_columns"):
        try:
            return hf_dataset.select_columns(["task_index"])
        except Exception:
            pass
    return dataset
