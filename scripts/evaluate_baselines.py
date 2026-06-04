import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(results: dict, save_dir: Path):
    """Plot training loss and accuracy."""
    history = results["training_history"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if "autoregressive" in history:
        epochs = [e["epoch"] for e in history["autoregressive"]]
        losses = [e["loss"] for e in history["autoregressive"]]
        accs = [e["accuracy"] for e in history["autoregressive"]]
        
        ax1.plot(epochs, losses, label="Autoregressive", color="tab:orange", lw=2)
        ax2.plot(epochs, accs, label="Autoregressive", color="tab:orange", lw=2)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curve")
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Token Prediction Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_eval_metrics(results: dict, save_dir: Path):
    """Bar chart comparison of evaluation metrics."""
    eval_res = results["eval_results"]
    
    if "autoregressive" not in eval_res:
        return
        
    metrics = ["MSE", "L1 Error"]
    autoregressive_vals = [
        eval_res["autoregressive"]["mse"],
        eval_res["autoregressive"]["l1"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, autoregressive_vals, width, label='Autoregressive Token', color="tab:orange")
    
    ax.set_ylabel('Error Value')
    ax.set_title('Evaluation Metrics on LIBERO Task 20')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(save_dir / "eval_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_latency(results: dict, save_dir: Path):
    """Plot inference latency."""
    eval_res = results["eval_results"]
    if "autoregressive" not in eval_res:
        return
        
    methods = ['Autoregressive Token']
    times = [eval_res["autoregressive"]["inference_time_per_sample"] * 1000] # ms
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, times, color=["tab:orange"], width=0.4)
    
    ax.set_ylabel('Inference Time (ms / chunk)')
    ax.set_title('Inference Latency')
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    fig.savefig(save_dir / "inference_latency.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_latex_table(results: dict, save_dir: Path):
    """Generate a LaTeX table for the report."""
    eval_res = results["eval_results"]
    if "autoregressive" not in eval_res:
        return
        
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Action-generation performance on LIBERO Task 20.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Method & MSE $\downarrow$ & L1 $\downarrow$ & Inf. Time (ms) $\downarrow$ \\",
        r"\midrule"
    ]
    
    m = eval_res["autoregressive"]
    lines.append(
        f"Autoregressive & {m['mse']:.4f} & {m['l1']:.4f} & {m['inference_time_per_sample']*1000:.1f} \\\\"
    )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    with open(save_dir / "results_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(checkpoint_dir / "results.json", "r") as f:
        results = json.load(f)
        
    print("Generating plots and tables for Autoregressive model...")
    plot_training_curves(results, results_dir)
    plot_eval_metrics(results, results_dir)
    plot_latency(results, results_dir)
    generate_latex_table(results, results_dir)
    print(f"Done! Outputs saved to {results_dir}/")

if __name__ == "__main__":
    main()
