# BC Regression — LIBERO Action Prediction

Behavioral Cloning (BC) Regression baseline for the [ECE 228](https://ece228.ucsd.edu/) final project.
The model learns to predict 7-DOF robot actions from dual camera images, proprioceptive state, and a
natural-language task instruction, using a frozen CLIP encoder + lightweight MLP decoder trained
with MSE loss.

---

## Project Structure

```
ECE228final-Regression/
├── src/
│   └── lerobot_policy_pi0_lite_flow/
│       ├── libero_adapter.py          # LIBEROActionChunkDataset + task filtering
│       ├── representation_encoder.py  # CLIP image/text + state → cond vector
│       ├── configuration_regression.py
│       ├── modeling_regression.py     # BCRegressionPolicy (MLP, MSE loss)
│       └── ...
├── scripts/
│   ├── train_regression_custom.py     # BC Regression training
│   ├── eval_regression_custom.py      # Offline evaluation
│   ├── batch_rollout.py               # Generate 10 rollout videos
│   ├── plot_202122_results.py         # Paper-style figures
│   ├── visualize_rollout.py           # Single-task rollout video
│   ├── train_flow_custom.py           # (Flow Matching reference)
│   ├── eval_flow_custom.py            # (Flow Matching reference)
│   └── prefetch_libero.py             # Download LIBERO from HuggingFace
├── results/
│   ├── task202122_regression/         # Checkpoints + metrics
│   │   ├── checkpoint_final.pt
│   │   ├── metrics.csv                # Per-step training loss
│   │   ├── eval_metrics.csv           # Offline eval results
│   │   └── args.json
│   ├── rollouts/                      # 10 MP4 rollout videos
│   └── figures/                       # Publication figures (fig1-fig5)
└── tests/
    └── test_regression.py
```

---

## Quick Start

### 1. Environment

```powershell
conda create -n cuda-ml python=3.12
conda activate cuda-ml
pip install -e ".[dev,encoder]"
```

### 2. Dataset

The full LIBERO dataset (~32 GB) should be downloaded locally once.
With a locally cached copy at `D:/AA_Graduate/datasets/libero`, all scripts
accept `--local-dir` to avoid repeated HuggingFace network calls:

```powershell
# Download via HuggingFace Hub (one-time)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceVLA/libero', repo_type='dataset',
                  local_dir='D:/AA_Graduate/datasets/libero')
"
```

---

## Training

Train BC Regression on LIBERO tasks 20, 21, 22:

```powershell
python scripts\train_regression_custom.py `
  --local-dir "D:/AA_Graduate/datasets/libero" `
  --task-indices 20 21 22 `
  --steps 6000 `
  --batch-size 32 `
  --use-clip `
  --lr 1e-4 `
  --run-name task202122_regression `
  --output-dir results `
  --log-every 100 `
  --save-every 2000
```

Checkpoints are saved to `results/task202122_regression/`.

---

## Evaluation

Run offline evaluation on held-out samples:

```powershell
python scripts\eval_regression_custom.py `
  "results/task202122_regression/checkpoint_final.pt" `
  --local-dir "D:/AA_Graduate/datasets/libero" `
  --num-samples 1024 `
  --output "results/task202122_regression/eval_metrics.csv"
```

---

## Rollout Videos (Offline Visualization)

Generate 10 offline visualization videos covering different scenarios across tasks 20, 21, 22.
These are **not** simulation rollouts — the robot frames come from the expert demonstrations
in the dataset, with the model's predicted actions overlaid as a bar chart.

```powershell
python scripts\batch_rollout.py `
  --checkpoint "results/task202122_regression/checkpoint_final.pt" `
  --local-dir "D:/AA_Graduate/datasets/libero" `
  --frames 200 `
  --fps 10 `
  --out-dir "results/rollouts"
```

Videos are saved as `results/rollouts/Task20-ep0.mp4`, `Task21-ep11.mp4`, etc.
Each video shows agent-view | wrist-view + a per-DOF action bar (predicted vs expert).

---

## Simulation Rollout — Success Rate (Linux Server)

To get a real **task success rate**, run the simulation rollout script on a Linux server
with MuJoCo + LIBERO + ffmpeg installed:

```bash
# Install requirements (server)
pip install -e /path/to/LIBERO   # LIBERO from source
pip install robosuite==1.4.0

# Run 10 episodes per task, tasks 20+21+22
python scripts/sim_rollout_regression.py \
  results/task202122_regression/checkpoint_final.pt \
  --dataset-task-indices 20 21 22 \
  --episodes-per-task 10 \
  --suite libero_object \
  --video-dir results/sim_rollout \
  --device cuda
```

Output `results/sim_rollout/rollout_metrics.json` will contain:
- `success_rate` — fraction of episodes where the task was completed
- `successes` / `episodes` — raw counts
- Per-episode `success`, `steps`, `video` path

Server requirements:
- Linux (Ubuntu 20.04+)
- NVIDIA GPU + CUDA
- `MUJOCO_GL=egl` (set automatically by the script)
- `ffmpeg` on PATH

---

## Plotting

Generate publication-quality figures from training + eval + rollout results:

```powershell
python scripts\plot_202122_results.py `
  --res-dir "results/task202122_regression" `
  --rollout-dir "results/rollouts" `
  --out-dir "results/figures"
```

Five figures are produced:

| File | Content |
|------|---------|
| `fig1_training_loss.png` | Training loss curve (EMA smoothed) |
| `fig2_eval_metrics.png`  | Offline eval metric bar chart |
| `fig3_gripper_latency.png` | Gripper accuracy donut + latency table |
| `fig4_rollout_mse.png`   | Per-scenario rollout MSE across 10 videos |
| `fig5_dashboard.png`     | Combined 2×3 summary dashboard |

---

## Results — Tasks 20+21+22

**Training config:** 6,000 steps · batch 32 · lr 1e-4 · CLIP ViT-B/32 (frozen) · 19,588 samples

| Metric | Value |
|--------|------:|
| Final training loss (MSE) | **0.0689** |
| Overall eval MSE | **0.01180** |
| L1 Distance | 0.04980 |
| Position MSE | 0.00806 |
| Rotation MSE | 0.000302 |
| Gripper MSE | 0.05749 |
| **Gripper sign accuracy** | **98.3%** |
| Smoothness | 0.01488 |
| Inference latency | 9.43 ms (106 Hz) |

**Tasks:**
- Task 20: *pick up the orange juice and place it in the basket*
- Task 21: *pick up the ketchup and place it in the basket*
- Task 22: *pick up the cream cheese and place it in the basket*

---

## Architecture

```
observation.images  (B, 2, 3, 256, 256)  ──►  CLIP ViT-B/32 (frozen) ──► z_img (×2)
observation.state   (B, 8)               ──►  state MLP               ──► z_state
task instruction    list[str]            ──►  CLIP text (frozen)       ──► z_text

concat + linear projection ──► cond  (B, 256)

cond  ──►  BCRegressionPolicy (4-layer MLP, hidden 256, MSE)  ──►  action (B, horizon, 7)
```

- **Encoder**: frozen CLIP ViT-B/32, trainable fusion MLP (1.8 M params total)
- **Policy**: 4-layer MLP, `cond_dim=256 → hidden=256 → action_dim×horizon`
- **Loss**: MSE on normalized action chunks (horizon = 16, action dim = 7)

---

## Tests

```powershell
pytest tests/
```
