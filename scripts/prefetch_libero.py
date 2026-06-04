"""Pre-fetch LIBERO data for selected task indices, with live, line-buffered logging.

Why this exists
---------------
`LeRobotDataset(repo_id)` blindly downloads every parquet shard (LIBERO is ~32 GB).
For local development we only need a small subset (e.g. Task 20 = ~1-2 GB). This
script first resolves the episode indices that belong to the requested tasks
(via meta/, a few MB), then instantiates `LeRobotDataset(repo_id, episodes=...)`
which only downloads the chunk parquet files that contain those episodes.

Logs are flushed line-by-line so you can `Get-Content -Wait logs/prefetch_libero.log`
in another shell to follow the download progress.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lerobot_policy_pi0_lite_flow  # noqa: F401 - triggers HF symlink patch

from lerobot_policy_pi0_lite_flow.libero_adapter import (
    LIBEROActionChunkDataset,
    resolve_task_episodes,
)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch LIBERO data for selected task indices.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument(
        "--task-indices",
        nargs="+",
        type=int,
        required=True,
        help="LIBERO task indices to pre-fetch (e.g. --task-indices 20).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Action chunk horizon. Only affects __len__/sample shapes, not the download volume.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to mirror stdout to. Defaults to logs/prefetch_libero_<ts>.log.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file) if args.log_file else (
        ROOT / "logs" / f"prefetch_libero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _TeeStream(sys.stdout, open(log_path, "a", encoding="utf-8", buffering=1))
    sys.stderr = sys.stdout

    print(f"[{_now()}] log_file={log_path}")
    print(f"[{_now()}] repo_id={args.repo_id} task_indices={args.task_indices}")

    t0 = time.time()
    print(f"[{_now()}] Step 1/3: resolving episodes for tasks {args.task_indices} (downloads only meta/*)...")
    episodes = resolve_task_episodes(args.repo_id, args.task_indices)
    print(f"[{_now()}]   -> resolved {len(episodes)} episode(s) in {time.time() - t0:.1f}s")
    if not episodes:
        raise SystemExit("No episodes match the requested task_indices.")
    print(f"[{_now()}]   first 20: {episodes[:20]}")

    t1 = time.time()
    print(f"[{_now()}] Step 2/3: constructing LeRobotDataset (triggers parquet downloads for selected episodes)...")
    print(f"[{_now()}]   This is the slow part. Expect ~1-2 GB per task on first run.")
    dataset = LIBEROActionChunkDataset(
        repo_id=args.repo_id,
        horizon=args.horizon,
        task_indices=args.task_indices,
    )
    print(f"[{_now()}]   -> LIBEROActionChunkDataset ready: {len(dataset)} samples in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"[{_now()}] Step 3/3: smoke check by reading the first sample...")
    sample = dataset[0]
    keys = sorted(sample.keys())
    print(f"[{_now()}]   sample keys: {keys}")
    print(f"[{_now()}]   images shape: {tuple(sample['images'].shape)}")
    print(f"[{_now()}]   state shape:  {tuple(sample['state'].shape)}")
    print(f"[{_now()}]   action shape: {tuple(sample['action'].shape)}")
    print(f"[{_now()}]   instruction:  {sample['instruction']!r}")
    print(f"[{_now()}]   -> sample loaded in {time.time() - t2:.2f}s")

    print(f"[{_now()}] DONE. Total wall time: {time.time() - t0:.1f}s")


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


if __name__ == "__main__":
    main()
