# pi0-Lite Action Decoder Experiments

This repository contains the shared LIBERO representation encoder and action
decoder experiments for the ECE 228 final project. The autoregressive baseline
uses discretized action tokens and a causal Transformer conditioned on the same
dual-view vision, language, and robot-state representation used by the flow
matching policy.

## Autoregressive Baseline

The baseline predicts a `(horizon, action_dim)` action chunk as a flattened
sequence of discrete action tokens:

1. Each action dimension is uniformly quantized using dataset action bounds.
2. Frozen CLIP image and text encoders extract dual-view visual features and the
   LIBERO task instruction feature.
3. A trainable fusion encoder produces `batch["observation.cond"]`.
4. A causal Transformer predicts action tokens with next-token cross-entropy.
5. Tokens are generated autoregressively and decoded to continuous actions.

The default report experiment uses LIBERO dataset task index `20`:

```text
pick up the orange juice and place it in the basket
```

Train and evaluation splits are created by episode, not by frame, so overlapping
action chunks from one episode never appear in both splits. Padded action steps
are excluded from training and evaluation metrics.

## Environment

```bash
conda create -n ece228-pi0lite python=3.12 -y
conda run -n ece228-pi0lite python -m pip install -e '.[dev,encoder]' matplotlib
```

## Validate CLIP and MPS

```bash
PYTHONPATH=src conda run -n ece228-pi0lite \
  python scripts/smoke_clip_encoder.py --device mps

PYTHONPATH=src conda run -n ece228-pi0lite pytest -q
```

## Train the 3000-Step CLIP Autoregressive Baseline

```bash
PYTHONPATH=src conda run -n ece228-pi0lite \
  python -u scripts/train_baselines.py \
  --task-indices 20 \
  --steps 3000 \
  --batch-size 8 \
  --encoder clip \
  --device mps \
  --save-every 500 \
  --run-name ar_clip_task20_3000
```

Frozen CLIP features are precomputed once and cached inside the run directory.
Only the state encoder, fusion layers, and autoregressive decoder are optimized.

## Evaluate and Generate Figures

```bash
PYTHONPATH=src conda run -n ece228-pi0lite \
  python scripts/evaluate_baselines.py \
  artifacts/autoregressive_clip_task20_3000/checkpoint_final.pt \
  --device mps \
  --output-dir results/autoregressive_clip_task20_3000
```

If the frozen CLIP feature cache is not present beside the checkpoint, the
evaluation script rebuilds it from the held-out episode split stored in the
checkpoint.

The evaluation command writes:

- `metrics.json`
- `args.json`
- `training_metrics.csv`
- `training_curves.png`
- `eval_comparison.png`
- `inference_latency.png`
- `results_table.tex`

## Measure LIBERO Success Rate

The final checkpoint can be evaluated in the LIBERO simulator with the
autoregressive rollout entry point. LIBERO simulation requires Linux/EGL,
`ffmpeg`, and `lerobot[libero]`; it is not part of the basic macOS training
environment.

```bash
INSTALL_LIBERO=1 bash scripts/docker_build.sh

docker run --rm --gpus all --ipc=host \
  --env MUJOCO_GL=egl \
  --env PYOPENGL_PLATFORM=egl \
  --volume "$PWD/artifacts:/workspace/artifacts" \
  --volume "$HOME/.cache/huggingface:/cache/huggingface" \
  ece228-pi0-lite-flow:latest \
  python scripts/eval_autoregressive_libero_rollout.py \
    artifacts/autoregressive_clip_task20_3000/checkpoint_final.pt \
    --suite libero_object \
    --dataset-task-indices 20 \
    --episodes-per-task 10 \
    --device cuda
```

This writes episode videos and `rollout_metrics.json` beside the checkpoint.
The checkpoint is kept in the local `artifacts/` folder and intentionally
ignored by Git; share it separately from the normal source-code repository.

## Shared Encoder and Flow Training

The shared encoder supports mock encoders for tests and explicit CLIP encoders
for real experiments. Flow matching training remains available through
`scripts/train_flow_custom.py`; use `--use-clip` for the CLIP-backed path.

## Core Modules

```text
src/lerobot_policy_pi0_lite_flow/
  action_tokenizer.py
  autoregressive_experiment.py
  configuration_autoregressive.py
  libero_adapter.py
  modeling_autoregressive.py
  modeling_pi0_lite_flow.py
  representation_encoder.py

scripts/
  train_baselines.py
  evaluate_baselines.py
  eval_autoregressive_libero_rollout.py
  smoke_clip_encoder.py
```

## References

- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024.
- Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control", 2024.
- Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", 2023.
