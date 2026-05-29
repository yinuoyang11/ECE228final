from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from train_flow_custom.py metrics.csv.")
    parser.add_argument("metrics_csv", help="Path to metrics.csv")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to loss.png next to metrics_csv.")
    parser.add_argument("--smooth", type=int, default=10, help="Moving average window for smoothed loss.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc

    metrics_path = Path(args.metrics_csv)
    rows = read_metrics(metrics_path)
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    smoothed = moving_average(losses, args.smooth)

    output_path = Path(args.output) if args.output else metrics_path.with_name("loss.png")
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, alpha=0.35, label="loss")
    if smoothed:
        plt.plot(steps[-len(smoothed) :], smoothed, label=f"loss ma{args.smooth}")
    plt.xlabel("step")
    plt.ylabel("flow matching loss")
    plt.title(metrics_path.parent.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"saved_plot={output_path}")


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return []
    return [sum(values[idx - window : idx]) / window for idx in range(window, len(values) + 1)]


if __name__ == "__main__":
    main()

