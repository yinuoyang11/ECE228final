from __future__ import annotations

import argparse
import json

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Read LIBERO dataset metadata without downloading the full dataset.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    info_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="meta/info.json",
        repo_type="dataset",
        revision=args.revision,
    )
    with open(info_path, encoding="utf-8") as file:
        info = json.load(file)

    print(f"repo_id: {args.repo_id}")
    for key in ("codebase_version", "robot_type", "total_tasks", "total_episodes", "total_frames", "fps"):
        print(f"{key}: {info.get(key, 'unknown')}")
    if args.list_tasks:
        print("tasks:")
        for task_index, description in read_tasks(args.repo_id, args.revision):
            print(f"  {task_index:02d}: {description}")


def read_tasks(repo_id: str, revision: str | None) -> list[tuple[int, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for --list-tasks. Install the project requirements first.") from exc

    tasks_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/tasks.parquet",
        repo_type="dataset",
        revision=revision,
    )
    rows = pq.read_table(tasks_path).to_pylist()
    return [(int(row["task_index"]), str(row["__index_level_0__"])) for row in rows]


if __name__ == "__main__":
    main()
