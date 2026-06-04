from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch import Tensor

from . import _hf_compat  # noqa: F401 - side effect: patch HF symlink on Windows


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
        episodes: Sequence[int] | None = None,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        state_key: str = "observation.state",
        action_key: str = "action",
        instruction_keys: Sequence[str] = DEFAULT_INSTRUCTION_KEYS,
        root: str | Path | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if task_indices is not None and episodes is not None:
            raise ValueError("pass either task_indices or episodes, not both")

        self.horizon = horizon
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.instruction_keys = tuple(instruction_keys)
        self.task_indices = tuple(sorted(set(task_indices))) if task_indices is not None else None
        self.episodes = sorted(set(int(e) for e in episodes)) if episodes is not None else None
        self.root = Path(root) if root is not None else None
        if base_dataset is not None:
            self.dataset = base_dataset
        else:
            self.dataset = self._load_lerobot_dataset(
                repo_id,
                task_indices=self.task_indices,
                episodes=self.episodes,
                root=self.root,
            )
        self.episode_ranges = _episode_ranges(self.dataset)
        self.index_to_episode_end = _index_to_episode_end(self.episode_ranges)
        self.action_source = _make_action_source(self.dataset, action_key)
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
        return {
            "images": batch["images"].squeeze(0),
            "state": batch["state"].squeeze(0),
            "instruction": batch["instruction"][0],
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
    def _load_lerobot_dataset(
        repo_id: str,
        task_indices: Sequence[int] | None = None,
        episodes: Sequence[int] | None = None,
        root: Path | None = None,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - exercised only when LeRobot is missing.
            raise ImportError("Install lerobot[dataset] to load LIBEROActionChunkDataset by repo_id.") from exc

        if root is not None:
            # Local dataset: pass root directly, no HF Hub interaction needed.
            common_kwargs: dict = {"root": root}
        else:
            # force_cache_sync=True is a workaround for a lerobot 0.5.1 bug:
            # ``DatasetReader.try_load`` only catches ``FileNotFoundError`` /
            # ``NotADirectoryError``. When the local cache contains zero data
            # parquet shards (or any are missing), the underlying ``datasets``
            # library raises a ``ValueError("Instruction 'train' corresponds to no
            # data!")`` which bubbles up uncaught, so the automatic download path
            # inside ``LeRobotDataset.__init__`` never runs. Forcing cache sync
            # skips ``try_load`` and calls ``_download`` directly. On subsequent
            # runs the already-cached files are validated via HEAD requests in a
            # few seconds and only missing shards are downloaded.
            common_kwargs = {"force_cache_sync": True}

        if episodes is not None:
            return LeRobotDataset(repo_id, episodes=list(episodes), **common_kwargs)

        if task_indices is None:
            return LeRobotDataset(repo_id, **common_kwargs)

        target_episodes = resolve_task_episodes(repo_id, task_indices, local_root=root)
        if not target_episodes:
            raise ValueError(
                f"No episodes match task_indices {sorted(set(task_indices))} in {repo_id!r}."
            )
        return LeRobotDataset(repo_id, episodes=target_episodes, **common_kwargs)


def resolve_task_episodes(
    repo_id: str,
    task_indices: Iterable[int],
    local_root: str | Path | None = None,
) -> list[int]:
    """Return episode indices whose ``task_index`` is in ``task_indices``.

    If ``local_root`` is given, reads meta parquet files directly from that
    directory (no network access). Otherwise downloads ``meta/*`` files from
    the Hugging Face Hub.
    """

    import pyarrow.parquet as pq

    selected = {int(t) for t in task_indices}
    if not selected:
        raise ValueError("task_indices must contain at least one task index")
    if any(t < 0 for t in selected):
        raise ValueError("task_indices must be non-negative")

    if local_root is not None:
        local_meta_root = Path(local_root)
    else:
        from huggingface_hub import snapshot_download
        local_meta_root = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/*"],
        ))

    meta_dir = local_meta_root / "meta" / "episodes"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Expected meta/episodes/ under {local_meta_root}")

    parquet_files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet files under {meta_dir}")

    episodes: list[int] = []
    for pf in parquet_files:
        tbl = pq.read_table(pf, columns=["episode_index", "stats/task_index/min"])
        ep_col = tbl.column("episode_index").to_pylist()
        task_col = tbl.column("stats/task_index/min").to_pylist()
        for ep_idx, task_value in zip(ep_col, task_col, strict=True):
            if hasattr(task_value, "__len__") and not isinstance(task_value, (str, bytes)):
                task_int = int(task_value[0])
            else:
                task_int = int(task_value)
            if task_int in selected:
                episodes.append(int(ep_idx))

    return sorted(set(episodes))


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


def _to_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value)


def _episode_ranges(dataset: Any) -> list[tuple[int, int]]:
    dataset_len = len(dataset)

    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None)
    if episodes is not None:
        ranges = []
        for row in episodes:
            if "dataset_from_index" in row and "dataset_to_index" in row:
                ranges.append((int(row["dataset_from_index"]), int(row["dataset_to_index"])))
        # Only trust meta-derived ranges when they reference indices within the
        # actual loaded dataset. When the caller passed ``episodes=[...]`` to
        # LeRobotDataset, the underlying PyArrow table is filtered/reindexed but
        # ``meta.episodes`` still describes the unfiltered dataset, so the
        # from/to indices would point past ``len(dataset)``.
        if ranges and max(end for _, end in ranges) <= dataset_len:
            return ranges

    episode_column = _scan_episode_index_column(dataset, dataset_len)
    ranges = []
    current_start = 0
    current_episode = None
    for idx, episode in enumerate(episode_column):
        if current_episode is None:
            current_episode = episode
            current_start = idx
        elif episode != current_episode:
            ranges.append((current_start, idx))
            current_episode = episode
            current_start = idx
    if current_episode is not None:
        ranges.append((current_start, dataset_len))
    return ranges


def _scan_episode_index_column(dataset: Any, dataset_len: int) -> list[int]:
    """Read the ``episode_index`` column as a fast list[int] without decoding
    images. Falls back to per-row access if the column shortcut is unavailable."""

    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            return [int(v) for v in hf_dataset.with_format(None)["episode_index"]]
        except Exception:
            try:
                return [int(v) for v in hf_dataset["episode_index"]]
            except Exception:
                pass

    return [int(_to_tensor(dataset[idx]["episode_index"]).item()) for idx in range(dataset_len)]


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
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "select_columns"):
        try:
            return hf_dataset.select_columns(["task_index"])
        except Exception:
            pass
    return dataset
