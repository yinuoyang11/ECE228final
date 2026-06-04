"""Generate publication-quality figures for LIBERO Task 20 – BC Regression only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "task20_regression"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
PRIMARY   = "#2563EB"   # blue
ACCENT    = "#10B981"   # emerald
WARN      = "#F59E0B"   # amber
DANGER    = "#EF4444"   # red
LIGHT_BG  = "#F8FAFC"
PANEL_BG  = "#FFFFFF"
GRID_CLR  = "#E2E8F0"
TEXT_MAIN = "#1E293B"
TEXT_SUB  = "#64748B"

PLT_STYLE = {
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.8,
    "text.color":        TEXT_MAIN,
    "axes.labelcolor":   TEXT_MAIN,
    "xtick.color":       TEXT_SUB,
    "ytick.color":       TEXT_SUB,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}
plt.rcParams.update(PLT_STYLE)


# ── helpers ────────────────────────────────────────────────────────────────────
def read_train_csv(path: Path) -> tuple[list[float], list[float]]:
    steps, losses = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            losses.append(float(row["mse_loss"]))
    return steps, losses


def read_eval_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) if v.replace(".", "").lstrip("-").isdigit() else v
            for k, v in row.items()}


def smoothed(values: list[float], w: int = 5) -> list[float]:
    arr = np.array(values, dtype=float)
    out = np.convolve(arr, np.ones(w) / w, mode="valid")
    # prepend raw values so array length matches input
    return list(arr[:w-1]) + list(out)


# ── load data ──────────────────────────────────────────────────────────────────
steps, losses = read_train_csv(RES_DIR / "metrics.csv")
eval_m        = read_eval_csv(RES_DIR / "eval_metrics.csv")

smooth_loss = smoothed(losses, w=7)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Training Loss Curve
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(11, 5))
fig1.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)

ax.fill_between(steps, losses, alpha=0.12, color=PRIMARY)
ax.plot(steps, losses,       color=PRIMARY,  lw=1.2, alpha=0.45, label="Raw loss")
ax.plot(steps, smooth_loss,  color=PRIMARY,  lw=2.5, label="Smoothed (7-step MA)")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_xlim(0, max(steps))
ax.set_ylim(0)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=11, framealpha=0.9)
ax.set_title(
    "BC Regression — Training Loss Curve\n"
    "LIBERO Task 20 · \"pick up the orange juice and place it in the basket\"",
    fontsize=13, fontweight="bold", pad=12, color=TEXT_MAIN,
)

# annotation: final loss
ax.annotate(
    f"Final loss: {losses[-1]:.4f}",
    xy=(steps[-1], losses[-1]),
    xytext=(-120, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color=TEXT_SUB, lw=1.5),
    fontsize=11, color=PRIMARY, fontweight="bold",
)

fig1.savefig(OUT_DIR / "task20_reg_training_loss.png", dpi=180, bbox_inches="tight")
plt.close(fig1)
print("Saved: task20_reg_training_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Eval Metric Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
metric_labels = ["Overall\nMSE", "L1\nDistance", "Position\nMSE", "Rotation\nMSE", "Gripper\nMSE", "Smoothness"]
metric_keys   = ["mse", "l1", "position_mse", "rotation_mse", "gripper_mse", "smoothness"]
values        = [eval_m[k] for k in metric_keys]
bar_colors    = [PRIMARY, ACCENT, PRIMARY, ACCENT, WARN, ACCENT]

fig2, ax = plt.subplots(figsize=(12, 6))
fig2.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.15)

bars = ax.bar(metric_labels, values, color=bar_colors, width=0.55, zorder=3,
              edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(values) * 0.012,
        f"{val:.5f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT_MAIN,
    )

ax.set_ylabel("Value (lower is better)", fontsize=12)
ax.set_ylim(0, max(values) * 1.22)
ax.set_title(
    "BC Regression — Offline Evaluation Metrics\n"
    "LIBERO Task 20 · 1000 held-out samples · checkpoint_final.pt",
    fontsize=13, fontweight="bold", pad=12,
)
ax.tick_params(axis="x", labelsize=11)

fig2.savefig(OUT_DIR / "task20_reg_eval_metrics.png", dpi=180, bbox_inches="tight")
plt.close(fig2)
print("Saved: task20_reg_eval_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Gripper Sign Accuracy (prominent single stat)
# ══════════════════════════════════════════════════════════════════════════════
acc = eval_m["gripper_sign_accuracy"]

fig3, ax = plt.subplots(figsize=(7, 5))
fig3.subplots_adjust(left=0.12, right=0.88, top=0.85, bottom=0.12)

# donut chart
wedge_vals = [acc, 1 - acc]
wedge_colors = [ACCENT, GRID_CLR]
wedges, _ = ax.pie(
    wedge_vals, colors=wedge_colors,
    startangle=90,
    wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
)
ax.text(0, 0, f"{acc*100:.2f}%", ha="center", va="center",
        fontsize=26, fontweight="bold", color=ACCENT)
ax.text(0, -0.22, "Gripper Sign\nAccuracy", ha="center", va="center",
        fontsize=12, color=TEXT_SUB)
ax.set_title(
    "Gripper Open/Close Accuracy\nBC Regression · LIBERO Task 20",
    fontsize=13, fontweight="bold", pad=14,
)

fig3.savefig(OUT_DIR / "task20_reg_gripper_accuracy.png", dpi=180, bbox_inches="tight")
plt.close(fig3)
print("Saved: task20_reg_gripper_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Summary Dashboard (2×2 combined)
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(16, 10))
fig4.patch.set_facecolor(LIGHT_BG)

gs = fig4.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                        left=0.07, right=0.97, top=0.91, bottom=0.08)

# ── (0,0) training loss ──────────────────────────────────────────────────────
ax00 = fig4.add_subplot(gs[0, 0])
ax00.fill_between(steps, losses, alpha=0.10, color=PRIMARY)
ax00.plot(steps, losses,      color=PRIMARY, lw=1.0, alpha=0.35)
ax00.plot(steps, smooth_loss, color=PRIMARY, lw=2.2, label="Smoothed loss")
ax00.set_xlabel("Training Step", fontsize=10)
ax00.set_ylabel("MSE Loss", fontsize=10)
ax00.set_xlim(0, max(steps)); ax00.set_ylim(0)
ax00.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax00.set_title("Training Loss Curve", fontsize=11, fontweight="bold")
ax00.annotate(f"Final: {losses[-1]:.4f}",
              xy=(steps[-1], losses[-1]), xytext=(-100, 25),
              textcoords="offset points",
              arrowprops=dict(arrowstyle="->", color=TEXT_SUB, lw=1.2),
              fontsize=9, color=PRIMARY, fontweight="bold")

# ── (0,1) eval bar ───────────────────────────────────────────────────────────
ax01 = fig4.add_subplot(gs[0, 1])
short_labels = ["MSE", "L1", "Pos-MSE", "Rot-MSE", "Grip-MSE", "Smooth"]
bars01 = ax01.bar(short_labels, values, color=bar_colors, width=0.55,
                   zorder=3, edgecolor="white", linewidth=1)
for bar, val in zip(bars01, values):
    ax01.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.015,
              f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax01.set_ylabel("Value (lower is better)", fontsize=10)
ax01.set_ylim(0, max(values)*1.25)
ax01.set_title("Offline Evaluation Metrics", fontsize=11, fontweight="bold")
ax01.tick_params(axis="x", labelsize=9)

# ── (1,0) donut ───────────────────────────────────────────────────────────────
ax10 = fig4.add_subplot(gs[1, 0])
ax10.pie([acc, 1-acc], colors=[ACCENT, GRID_CLR], startangle=90,
         wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 2})
ax10.text(0,  0.05, f"{acc*100:.2f}%", ha="center", va="center",
          fontsize=22, fontweight="bold", color=ACCENT)
ax10.text(0, -0.22, "Gripper Accuracy", ha="center", va="center",
          fontsize=10, color=TEXT_SUB)
ax10.set_title("Gripper Sign Accuracy", fontsize=11, fontweight="bold")

# ── (1,1) stat card table ─────────────────────────────────────────────────────
ax11 = fig4.add_subplot(gs[1, 1])
ax11.axis("off")
stat_rows = [
    ["Metric",              "Value"],
    ["Overall MSE",         f"{eval_m['mse']:.5f}"],
    ["L1 Distance",         f"{eval_m['l1']:.5f}"],
    ["Position MSE",        f"{eval_m['position_mse']:.5f}"],
    ["Rotation MSE",        f"{eval_m['rotation_mse']:.6f}"],
    ["Gripper MSE",         f"{eval_m['gripper_mse']:.5f}"],
    ["Gripper Accuracy",    f"{eval_m['gripper_sign_accuracy']*100:.2f}%"],
    ["Smoothness",          f"{eval_m['smoothness']:.5f}"],
    ["Latency (ms/sample)", f"{eval_m['latency_sec_per_sample']*1000:.2f}"],
]
col_widths = [0.62, 0.38]
row_h = 0.085
x0, y0 = 0.02, 0.97
for r_i, row in enumerate(stat_rows):
    y = y0 - r_i * row_h
    is_header = r_i == 0
    bg = PRIMARY if is_header else ("#EFF6FF" if r_i % 2 == 0 else PANEL_BG)
    fc = "white" if is_header else TEXT_MAIN
    rect = FancyBboxPatch((x0, y - row_h + 0.005), 0.96, row_h - 0.006,
                           boxstyle="round,pad=0.005",
                           facecolor=bg, edgecolor=GRID_CLR,
                           transform=ax11.transAxes, clip_on=False)
    ax11.add_patch(rect)
    cx = x0
    for col_i, (cell, cw) in enumerate(zip(row, col_widths)):
        ax11.text(cx + cw / 2, y - row_h / 2 + 0.005, cell,
                  transform=ax11.transAxes,
                  ha="center", va="center",
                  fontsize=9.5, fontweight="bold" if is_header or col_i == 0 else "normal",
                  color=fc)
        cx += cw
ax11.set_title("Full Metric Summary", fontsize=11, fontweight="bold")

fig4.suptitle(
    "BC Regression — LIBERO Task 20\n"
    "\"pick up the orange juice and place it in the basket\"  ·  3000 steps  ·  CLIP ViT-B/32  ·  1000 eval samples",
    fontsize=13, fontweight="bold", color=TEXT_MAIN, y=0.998,
)

fig4.savefig(OUT_DIR / "task20_reg_dashboard.png", dpi=180, bbox_inches="tight")
plt.close(fig4)
print("Saved: task20_reg_dashboard.png")
