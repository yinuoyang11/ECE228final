from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def describe_value(value: object) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return f"type={type(value).__name__}, shape={shape}, dtype={dtype}"


def import_lerobot_dataset():
    import_errors = []
    for import_stmt in (
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        "from lerobot.datasets import LeRobotDataset",
        "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset",
    ):
        namespace = {}
        try:
            exec(import_stmt, namespace)
            print(f"Imported LeRobotDataset with: {import_stmt}")
            return namespace["LeRobotDataset"]
        except Exception:
            import_errors.append((import_stmt, traceback.format_exc()))

    print("Could not import LeRobotDataset from the installed LeRobot package.")
    for import_stmt, error in import_errors:
        print(f"\n--- {import_stmt} ---")
        print(error)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LeRobot LIBERO sample keys and shapes.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id)
    sample = dataset[args.index]

    print(f"repo_id: {args.repo_id}")
    print(f"num_samples: {len(dataset)}")
    print("sample keys:")
    for key in sorted(sample.keys()):
        print(f"  {key}: {describe_value(sample[key])}")


if __name__ == "__main__":
    main()

