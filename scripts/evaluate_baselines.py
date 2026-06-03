#!/usr/bin/env python3
"""Evaluate and compare trained baselines.  Generates plots and tables.

Usage:
    PYTHONPATH=src python scripts/evaluate_baselines.py [OPTIONS]

Options:
    --results_file  Path to results.json from train_baselines.py (default: checkpoints/results.json)
    --output_dir    Directory for output plots (default: results/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def plot_training_curves(history: dict, output_dir: Path) -> None:
    """Plot training loss curves for all methods."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {"flow": "#6366f1", "autoregressive": "#f97316", "regression": "#10b981"}
    labels = {
        "flow": "Flow Matching (CFM)",
        "autoregressive": "Autoregressive (Discrete Tokens)",
        "regression": "BC Regression (MSE)",
    }

    for method_name, records in history.items():
        epochs = [r["epoch"] for r in records]
        losses = [r["loss"] for r in records]
        color = colors.get(method_name, "#888888")
        label = labels.get(method_name, method_name)
        ax.plot(epochs, losses, label=label, color=color, linewidth=2, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title("Training Loss Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves → {out_path}")


def plot_eval_bars(eval_results: dict, output_dir: Path) -> None:
    """Plot grouped bar chart of evaluation metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available — skipping plots.")
        return

    methods = list(eval_results.keys())
    metrics = ["mse", "l1", "jerk"]
    metric_labels = {"mse": "MSE ↓", "l1": "L1 ↓", "jerk": "Jerk ↓"}
    colors = {"flow": "#6366f1", "autoregressive": "#f97316", "regression": "#10b981"}
    pretty_names = {
        "flow": "Flow Matching",
        "autoregressive": "Autoregressive",
        "regression": "BC Regression",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [eval_results[m].get(metric, 0) for m in methods]
        bars = ax.bar(
            range(len(methods)),
            values,
            color=[colors.get(m, "#888888") for m in methods],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([pretty_names.get(m, m) for m in methods], fontsize=10)
        ax.set_title(metric_labels.get(metric, metric), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars.
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Evaluation Metrics Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = output_dir / "eval_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation comparison → {out_path}")


def plot_inference_latency(eval_results: dict, output_dir: Path) -> None:
    """Plot inference latency comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(eval_results.keys())
    times = [eval_results[m].get("inference_time_per_sample", 0) * 1000 for m in methods]
    colors = {"flow": "#6366f1", "autoregressive": "#f97316", "regression": "#10b981"}
    pretty_names = {
        "flow": "Flow Matching",
        "autoregressive": "Autoregressive",
        "regression": "BC Regression",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        range(len(methods)),
        times,
        color=[colors.get(m, "#888888") for m in methods],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([pretty_names.get(m, m) for m in methods], fontsize=11)
    ax.set_xlabel("Inference Time (ms/sample)", fontsize=12)
    ax.set_title("Inference Latency Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}ms",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    out_path = output_dir / "inference_latency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inference latency → {out_path}")


def print_results_table(eval_results: dict) -> None:
    """Print a formatted results table."""
    pretty_names = {
        "flow": "Flow Matching (CFM)",
        "autoregressive": "Autoregressive Tokens",
        "regression": "BC Regression (MSE)",
    }

    print("\n" + "=" * 75)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Method':<25} {'MSE ↓':>10} {'L1 ↓':>10} {'Jerk ↓':>10} {'Latency':>12}")
    print("-" * 75)
    for name, metrics in eval_results.items():
        pname = pretty_names.get(name, name)
        latency_ms = metrics.get("inference_time_per_sample", 0) * 1000
        print(
            f"{pname:<25} "
            f"{metrics.get('mse', 0):>10.6f} "
            f"{metrics.get('l1', 0):>10.6f} "
            f"{metrics.get('jerk', 0):>10.6f} "
            f"{latency_ms:>10.2f}ms"
        )
    print("=" * 75)

    # Analysis.
    if len(eval_results) >= 2:
        mse_ranking = sorted(eval_results.items(), key=lambda x: x[1].get("mse", float("inf")))
        print(f"\n🏆 Best MSE: {pretty_names.get(mse_ranking[0][0], mse_ranking[0][0])}")

        l1_ranking = sorted(eval_results.items(), key=lambda x: x[1].get("l1", float("inf")))
        print(f"🏆 Best L1:  {pretty_names.get(l1_ranking[0][0], l1_ranking[0][0])}")

        jerk_ranking = sorted(eval_results.items(), key=lambda x: x[1].get("jerk", float("inf")))
        print(f"🏆 Smoothest: {pretty_names.get(jerk_ranking[0][0], jerk_ranking[0][0])}")

        latency_ranking = sorted(
            eval_results.items(),
            key=lambda x: x[1].get("inference_time_per_sample", float("inf")),
        )
        print(f"🏆 Fastest: {pretty_names.get(latency_ranking[0][0], latency_ranking[0][0])}")


def generate_latex_table(eval_results: dict, output_dir: Path) -> None:
    """Generate a LaTeX table for the project report."""
    pretty_names = {
        "flow": "Flow Matching (CFM)",
        "autoregressive": "Autoregressive Tokens",
        "regression": "BC Regression (MSE)",
    }

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of action-generation methods on synthetic benchmark.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Method & MSE $\downarrow$ & L1 $\downarrow$ & Jerk $\downarrow$ & Latency (ms) \\",
        r"\midrule",
    ]

    # Find best values for bolding.
    best_mse = min(m.get("mse", float("inf")) for m in eval_results.values())
    best_l1 = min(m.get("l1", float("inf")) for m in eval_results.values())
    best_jerk = min(m.get("jerk", float("inf")) for m in eval_results.values())

    for name, metrics in eval_results.items():
        pname = pretty_names.get(name, name)
        mse = metrics.get("mse", 0)
        l1 = metrics.get("l1", 0)
        jerk = metrics.get("jerk", 0)
        latency = metrics.get("inference_time_per_sample", 0) * 1000

        mse_str = f"\\textbf{{{mse:.4f}}}" if abs(mse - best_mse) < 1e-8 else f"{mse:.4f}"
        l1_str = f"\\textbf{{{l1:.4f}}}" if abs(l1 - best_l1) < 1e-8 else f"{l1:.4f}"
        jerk_str = f"\\textbf{{{jerk:.4f}}}" if abs(jerk - best_jerk) < 1e-8 else f"{jerk:.4f}"

        lines.append(f"{pname} & {mse_str} & {l1_str} & {jerk_str} & {latency:.2f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = output_dir / "results_table.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare baselines")
    parser.add_argument(
        "--results_file", type=str, default="checkpoints/results.json",
        help="Path to results.json from train_baselines.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for output plots and tables",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run train_baselines.py first to generate results.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    eval_results = results.get("eval_results", {})
    training_history = results.get("training_history", {})

    # Print table.
    print_results_table(eval_results)

    # Generate plots.
    if training_history:
        plot_training_curves(training_history, output_dir)
    if eval_results:
        plot_eval_bars(eval_results, output_dir)
        plot_inference_latency(eval_results, output_dir)
        generate_latex_table(eval_results, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
