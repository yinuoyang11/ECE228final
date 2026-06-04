"""Publication-quality figures for BC Regression trained on LIBERO Tasks 20+21+22."""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT    = Path(__file__).resolve().parents[1]

# ── palette (consistent with paper-style aesthetics) ──────────────────────────
P_BLUE   = "#2563EB"
P_TEAL   = "#0891B2"
P_GREEN  = "#059669"
P_AMBER  = "#D97706"
P_RED    = "#DC2626"
P_VIOLET = "#7C3AED"
LIGHT_BG = "#F8FAFC"
PANEL_BG = "#FFFFFF"
GRID_CLR = "#E2E8F0"
TEXT_M   = "#1E293B"
TEXT_S   = "#64748B"

RCPARAMS = {
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.7,
    "text.color":        TEXT_M,
    "axes.labelcolor":   TEXT_M,
    "xtick.color":       TEXT_S,
    "ytick.color":       TEXT_S,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  GRID_CLR,
}
plt.rcParams.update(RCPARAMS)


# ── I/O helpers ───────────────────────────────────────────────────────────────
def read_train_csv(path: Path):
    steps, losses = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            losses.append(float(row["mse_loss"]))
    return np.array(steps), np.array(losses)


def read_eval_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def read_rollout_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ema(arr: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def savefig(fig, path: Path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── load all data ─────────────────────────────────────────────────────────────
def load_data(res_dir: Path, rollout_dir: Path):
    steps, losses = read_train_csv(res_dir / "metrics.csv")
    eval_m        = read_eval_csv(res_dir / "eval_metrics.csv")
    rollout_stats = None
    rpath = rollout_dir / "rollout_summary.json"
    if rpath.exists():
        rollout_stats = read_rollout_json(rpath)
    return steps, losses, eval_m, rollout_stats


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Training Loss Curve
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_loss(steps, losses, out_dir):
    smooth = ema(losses, alpha=0.12)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.13)

    ax.fill_between(steps, losses, alpha=0.10, color=P_BLUE)
    ax.plot(steps, losses,  color=P_BLUE, lw=1.0, alpha=0.40, label="Raw MSE loss")
    ax.plot(steps, smooth,  color=P_BLUE, lw=2.4, label="EMA smoothed")

    # annotate minimum
    i_min = int(np.argmin(smooth))
    ax.annotate(
        f"Min: {smooth[i_min]:.4f}",
        xy=(steps[i_min], smooth[i_min]),
        xytext=(40, 30), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=TEXT_S, lw=1.3),
        fontsize=10, color=P_BLUE, fontweight="bold",
    )
    # annotate final
    ax.annotate(
        f"Final: {smooth[-1]:.4f}",
        xy=(steps[-1], smooth[-1]),
        xytext=(-130, 30), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=TEXT_S, lw=1.3),
        fontsize=10, color=P_BLUE, fontweight="bold",
    )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_xlim(0, steps[-1]); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title(
        "BC Regression — Training Loss Curve\n"
        "LIBERO Tasks 20, 21, 22  ·  6,000 steps  ·  CLIP ViT-B/32  ·  batch 32",
        fontsize=13, fontweight="bold", pad=12, color=TEXT_M,
    )

    savefig(fig, out_dir / "fig1_training_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Evaluation Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_eval_metrics(eval_m, out_dir):
    METRIC_INFO = [
        ("Overall MSE",   "mse",                  P_BLUE),
        ("L1 Distance",   "l1",                   P_TEAL),
        ("Position MSE",  "position_mse",          P_GREEN),
        ("Rotation MSE",  "rotation_mse",          P_VIOLET),
        ("Gripper MSE",   "gripper_mse",           P_AMBER),
        ("Smoothness",    "smoothness",            P_RED),
    ]
    labels = [m[0] for m in METRIC_INFO]
    values = [eval_m[m[1]] for m in METRIC_INFO]
    colors = [m[2] for m in METRIC_INFO]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.14)

    bars = ax.bar(labels, values, color=colors, width=0.55,
                  zorder=3, edgecolor="white", linewidth=1.4)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.013,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=TEXT_M,
        )

    ax.set_ylabel("Value  (lower is better)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.22)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title(
        "BC Regression — Offline Evaluation Metrics\n"
        "LIBERO Tasks 20, 21, 22  ·  1,024 held-out samples",
        fontsize=13, fontweight="bold", pad=12,
    )

    savefig(fig, out_dir / "fig2_eval_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Gripper Accuracy (donut) + Latency (gauge)
# ══════════════════════════════════════════════════════════════════════════════
def fig_gripper_and_latency(eval_m, out_dir):
    acc = eval_m["gripper_sign_accuracy"]
    lat_ms = eval_m["latency_sec_per_sample"] * 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.87, bottom=0.08, wspace=0.35)

    # --- donut ---
    wedge_colors = [P_GREEN, GRID_CLR]
    ax1.pie([acc, 1-acc], colors=wedge_colors, startangle=90,
            wedgeprops={"width": 0.40, "edgecolor": "white", "linewidth": 2.5})
    ax1.text(0, 0.08, f"{acc*100:.1f}%",
             ha="center", va="center", fontsize=30, fontweight="bold", color=P_GREEN)
    ax1.text(0, -0.20, "Gripper\nOpen/Close\nAccuracy",
             ha="center", va="center", fontsize=11.5, color=TEXT_S, linespacing=1.5)
    ax1.set_title("Gripper Sign Accuracy", fontsize=13, fontweight="bold", pad=16)

    # --- bar for latency comparison ---
    ax2.axis("off")
    # Draw a styled info box
    latency_label = f"{lat_ms:.2f} ms"
    throughput = 1000 / lat_ms

    info_items = [
        ("Inference Latency",  latency_label,      P_BLUE),
        ("Throughput",         f"{throughput:.1f} Hz", P_TEAL),
        ("Gripper Accuracy",   f"{acc*100:.1f}%",  P_GREEN),
        ("Overall MSE",        f"{eval_m['mse']:.5f}", P_AMBER),
        ("Position MSE",       f"{eval_m['position_mse']:.5f}", P_VIOLET),
        ("Rotation MSE",       f"{eval_m['rotation_mse']:.6f}", P_RED),
    ]
    row_h = 0.13
    y0    = 0.92
    for i, (k, v, c) in enumerate(info_items):
        y = y0 - i * row_h
        bg = "#F0F9FF" if i % 2 == 0 else PANEL_BG
        rect = FancyBboxPatch((0.05, y - row_h + 0.015), 0.90, row_h - 0.010,
                               boxstyle="round,pad=0.008",
                               facecolor=bg, edgecolor=GRID_CLR,
                               transform=ax2.transAxes, clip_on=False, zorder=2)
        ax2.add_patch(rect)
        ax2.text(0.12, y - row_h/2 + 0.015, k,
                 transform=ax2.transAxes, ha="left", va="center",
                 fontsize=11, color=TEXT_M)
        ax2.text(0.88, y - row_h/2 + 0.015, v,
                 transform=ax2.transAxes, ha="right", va="center",
                 fontsize=11.5, fontweight="bold", color=c)

    ax2.set_title("Key Performance Metrics", fontsize=13, fontweight="bold", pad=16)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.suptitle(
        "BC Regression · LIBERO Tasks 20+21+22 — Gripper & Latency Summary",
        fontsize=13, fontweight="bold", y=1.01,
    )
    savefig(fig, out_dir / "fig3_gripper_latency.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Rollout MSE per Scenario (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig_rollout_mse(rollout_stats: list[dict], out_dir):
    task_colors = {20: P_BLUE, 21: P_GREEN, 22: P_AMBER}
    labels  = [s["scenario"] for s in rollout_stats]
    mses    = [s["avg_mse"] for s in rollout_stats]
    colors  = [task_colors.get(s["task"], P_VIOLET) for s in rollout_stats]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.22)

    bars = ax.bar(labels, mses, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(mses)*0.012,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # task legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=P_BLUE,  label="Task 20 – pick up orange juice"),
        Patch(facecolor=P_GREEN, label="Task 21 – pick up ketchup"),
        Patch(facecolor=P_AMBER, label="Task 22 – pick up cream cheese"),
    ]
    ax.legend(handles=legend_elems, fontsize=10, loc="upper right")

    ax.axhline(np.mean(mses), color=TEXT_S, lw=1.4, ls="--", zorder=4,
               label=f"Mean = {np.mean(mses):.4f}")

    ax.set_ylabel("Avg Step MSE  (predicted vs expert)", fontsize=12)
    ax.set_ylim(0, max(mses) * 1.28)
    ax.tick_params(axis="x", labelrotation=28, labelsize=9.5)
    ax.set_title(
        "BC Regression — Rollout MSE Across 10 Scenarios\n"
        "LIBERO Tasks 20, 21, 22  ·  Each video ~200 frames",
        fontsize=13, fontweight="bold", pad=12,
    )
    savefig(fig, out_dir / "fig4_rollout_mse.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Full Dashboard (2×3 grid)
# ══════════════════════════════════════════════════════════════════════════════
def fig_dashboard(steps, losses, eval_m, rollout_stats, out_dir):
    smooth = ema(losses, alpha=0.12)

    METRIC_INFO = [
        ("mse",            "Overall MSE",  P_BLUE),
        ("l1",             "L1 Distance",  P_TEAL),
        ("position_mse",   "Pos. MSE",     P_GREEN),
        ("rotation_mse",   "Rot. MSE",     P_VIOLET),
        ("gripper_mse",    "Grip. MSE",    P_AMBER),
        ("smoothness",     "Smoothness",   P_RED),
    ]
    bar_vals   = [eval_m[m[0]] for m in METRIC_INFO]
    bar_labels = [m[1] for m in METRIC_INFO]
    bar_colors = [m[2] for m in METRIC_INFO]
    acc        = eval_m["gripper_sign_accuracy"]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(LIGHT_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.90, bottom=0.07)

    # ── (0,0) training loss ──────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.fill_between(steps, losses, alpha=0.10, color=P_BLUE)
    ax00.plot(steps, losses, color=P_BLUE, lw=0.9, alpha=0.35)
    ax00.plot(steps, smooth, color=P_BLUE, lw=2.2, label="EMA smooth")
    ax00.set_xlabel("Step", fontsize=10)
    ax00.set_ylabel("MSE Loss", fontsize=10)
    ax00.set_xlim(0, steps[-1]); ax00.set_ylim(bottom=0)
    ax00.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax00.set_title("Training Loss", fontsize=11, fontweight="bold")
    ax00.legend(fontsize=9)

    # ── (0,1) eval bar ───────────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    bars = ax01.bar(bar_labels, bar_vals, color=bar_colors, width=0.6,
                    zorder=3, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, bar_vals):
        ax01.text(bar.get_x()+bar.get_width()/2,
                  bar.get_height()+max(bar_vals)*0.016,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax01.set_ylabel("Value (lower = better)", fontsize=10)
    ax01.set_ylim(0, max(bar_vals)*1.28)
    ax01.tick_params(axis="x", labelrotation=20, labelsize=8.5)
    ax01.set_title("Offline Eval Metrics", fontsize=11, fontweight="bold")

    # ── (0,2) donut ──────────────────────────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.pie([acc, 1-acc], colors=[P_GREEN, GRID_CLR], startangle=90,
             wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2})
    ax02.text(0, 0.07, f"{acc*100:.1f}%",
              ha="center", va="center", fontsize=26, fontweight="bold", color=P_GREEN)
    ax02.text(0, -0.22, "Gripper Accuracy",
              ha="center", va="center", fontsize=10, color=TEXT_S)
    ax02.set_title("Gripper Sign Accuracy", fontsize=11, fontweight="bold")

    # ── (1,0:2) rollout MSE per scenario ────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0:2])
    if rollout_stats:
        task_colors = {20: P_BLUE, 21: P_GREEN, 22: P_AMBER}
        r_labels = [s["scenario"] for s in rollout_stats]
        r_mses   = [s["avg_mse"] for s in rollout_stats]
        r_colors = [task_colors.get(s["task"], P_VIOLET) for s in rollout_stats]
        rb = ax10.bar(r_labels, r_mses, color=r_colors, width=0.6, zorder=3,
                      edgecolor="white", linewidth=1)
        for bar, val in zip(rb, r_mses):
            ax10.text(bar.get_x()+bar.get_width()/2,
                      bar.get_height()+max(r_mses)*0.014,
                      f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax10.axhline(np.mean(r_mses), color=TEXT_S, lw=1.4, ls="--")
        ax10.set_ylim(0, max(r_mses)*1.28)
        ax10.tick_params(axis="x", labelrotation=25, labelsize=8.5)
    else:
        ax10.text(0.5, 0.5, "No rollout data yet", ha="center", va="center",
                  transform=ax10.transAxes, fontsize=12, color=TEXT_S)
    ax10.set_ylabel("Avg Step MSE", fontsize=10)
    ax10.set_title("Rollout MSE — 10 Scenarios", fontsize=11, fontweight="bold")

    # ── (1,2) summary stat table ──────────────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")
    lat_ms = eval_m.get("latency_sec_per_sample", 0) * 1000
    table_rows = [
        ("Metric",              "Value"),
        ("Overall MSE",         f"{eval_m['mse']:.5f}"),
        ("L1 Distance",         f"{eval_m['l1']:.5f}"),
        ("Position MSE",        f"{eval_m['position_mse']:.5f}"),
        ("Rotation MSE",        f"{eval_m['rotation_mse']:.6f}"),
        ("Gripper MSE",         f"{eval_m['gripper_mse']:.5f}"),
        ("Gripper Accuracy",    f"{acc*100:.1f}%"),
        ("Smoothness",          f"{eval_m['smoothness']:.5f}"),
        ("Latency (ms/sample)", f"{lat_ms:.2f}"),
    ]
    rh = 0.088; y0 = 0.97
    for ri, (k, v) in enumerate(table_rows):
        y = y0 - ri * rh
        is_hdr = ri == 0
        bg = P_BLUE if is_hdr else ("#EFF6FF" if ri%2==0 else PANEL_BG)
        fc = "white" if is_hdr else TEXT_M
        rect = FancyBboxPatch((0.02, y-rh+0.005), 0.96, rh-0.006,
                               boxstyle="round,pad=0.006",
                               facecolor=bg, edgecolor=GRID_CLR,
                               transform=ax12.transAxes, clip_on=False)
        ax12.add_patch(rect)
        ax12.text(0.08, y-rh/2+0.005, k, transform=ax12.transAxes,
                  ha="left", va="center", fontsize=9,
                  fontweight="bold" if is_hdr else "normal", color=fc)
        ax12.text(0.92, y-rh/2+0.005, v, transform=ax12.transAxes,
                  ha="right", va="center", fontsize=9, fontweight="bold", color=fc)
    ax12.set_title("Full Metric Summary", fontsize=11, fontweight="bold")

    fig.suptitle(
        "BC Regression — LIBERO Tasks 20+21+22\n"
        '"pick up orange juice / ketchup / cream cheese and place it in the basket"  ·  6,000 steps',
        fontsize=13, fontweight="bold", color=TEXT_M, y=0.975,
    )
    savefig(fig, out_dir / "fig5_dashboard.png", dpi=180)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir",     default="results/task202122_regression",
                        help="Directory with metrics.csv and eval_metrics.csv")
    parser.add_argument("--rollout-dir", default="results/rollouts")
    parser.add_argument("--out-dir",     default="results/figures")
    args = parser.parse_args()

    res_dir     = ROOT / args.res_dir
    rollout_dir = ROOT / args.rollout_dir
    out_dir     = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    steps, losses, eval_m, rollout_stats = load_data(res_dir, rollout_dir)

    print("Generating figures ...")
    fig_training_loss(steps, losses, out_dir)
    fig_eval_metrics(eval_m, out_dir)
    fig_gripper_and_latency(eval_m, out_dir)
    if rollout_stats:
        fig_rollout_mse(rollout_stats, out_dir)
    fig_dashboard(steps, losses, eval_m, rollout_stats, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
