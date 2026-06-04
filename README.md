# Discretized Autoregressive Action-Token Model Baseline (ECE 228 Final)

This package provides a custom policy baseline for language-conditioned robot manipulation, evaluated on the **LIBERO** benchmark (Task 20). It is designed to plug into the overall project pipeline built by the team.

This specific baseline implements a **Discretized Autoregressive Action Token** policy (similar to OpenVLA).

## Architecture Overview

### Upstream: RepresentationEncoder (Teammate)
- Encodes **dual-view RGB images** (third-person + wrist camera) via CLIP vision encoder
- Encodes **language instructions** via CLIP text encoder
- Encodes **robot proprioceptive state** (8-dim) via MLP
- Fuses all modalities into a 256-dim conditioning vector via a learned MLP fusion layer

### Downstream: Autoregressive Decoder (This Baseline)
- Each action dimension is binned into 256 discrete tokens (uniform bins)
- A causal Transformer autoregressively predicts the token sequence
- Trained with cross-entropy loss (next-token prediction)
- At inference, tokens are sampled one at a time and decoded back to continuous values

## Dataset

We natively use the team's `LIBEROActionChunkDataset` adapter, which pulls from the complete LIBERO benchmark:
- **Repo ID**: `HuggingFaceVLA/libero`
- **Task**: Task 20 specifically
- **Observations**: Dual-view RGB images (`image` and `image2`) + 8-dim proprioceptive state
- **Actions**: 7-dim (6 DOF arm + gripper), chunked with horizon=16
- **Split**: 80% train / 20% test, seed=42

## Quick Start

### Environment Setup

```bash
conda create -n ece228-pi0lite python=3.12
conda activate ece228-pi0lite
pip install torch torchvision einops pytest matplotlib transformers datasets
pip install -e .
```

### Train the Autoregressive Baseline

```bash
PYTHONPATH=src python scripts/train_baselines.py --epochs 100 --batch_size 32
```

This seamlessly downloads the **real LIBERO Task 20 image dataset**, trains the Autoregressive model jointly with the RepresentationEncoder, and saves:
- Model checkpoints → `checkpoints/`
- Training history + eval results → `checkpoints/results.json`

### Evaluate and Generate Plots

```bash
PYTHONPATH=src python scripts/evaluate_baselines.py
```
PYTHONPATH=src python scripts/evaluate_baselines.py
```

Generates:
- Training loss curves → `results/training_curves.png`
- Evaluation metric comparison → `results/eval_comparison.png`
- Inference latency comparison → `results/inference_latency.png`
- LaTeX table for report → `results/results_table.tex`

## Project Structure

```
ECE228final/
├── src/lerobot_policy_pi0_lite_flow/
│   ├── __init__.py
│   ├── action_tokenizer.py              # Discretizes actions into tokens
│   ├── configuration_autoregressive.py   # Autoregressive baseline config
│   ├── configuration_pi0_lite_flow.py    # Flow matching config
│   ├── libero_adapter.py                # LIBERO dataset adapter (teammate)
│   ├── modeling_autoregressive.py        # Autoregressive decoder + policy
│   ├── modeling_pi0_lite_flow.py         # Flow matching decoder + policy
│   ├── processor_pi0_lite_flow.py        # Data processors
│   └── representation_encoder.py         # CLIP + state encoder (teammate)
├── scripts/
│   ├── train_baselines.py               # Train all three methods on LIBERO
│   └── evaluate_baselines.py            # Generate comparison plots
├── tests/
│   ├── test_autoregressive.py           # 15 tests for autoregressive baseline
│   └── test_flow_decoder.py             # 6 tests for flow matching
├── checkpoints/                         # Trained model weights
├── results/                             # Generated plots and tables
├── pyproject.toml
├── environment.yml
└── README.md
```

## Usage Examples

### Autoregressive Policy

```python
from lerobot_policy_pi0_lite_flow import (
    PI0LiteAutoregressiveConfig,
    AutoregressivePolicy,
)

config = PI0LiteAutoregressiveConfig(
    horizon=16, action_dim=7, cond_dim=256,
    num_bins=256, hidden_dim=256, nhead=8, num_layers=4,
)
dataset_stats = {
    "action": {
        "min": action_min,  # (action_dim,)
        "max": action_max,  # (action_dim,)
    }
}
policy = AutoregressivePolicy(config, dataset_stats=dataset_stats)

# Training
batch = {"observation.cond": cond, "action": actions}
output = policy.forward(batch)
loss = output["loss"]  # Cross-entropy loss
accuracy = output["accuracy"]  # Token prediction accuracy

# Inference
batch = {"observation.cond": cond}
action_chunk = policy.predict_action_chunk(batch)  # (B, 16, 7)
single_action = policy.select_action(batch)  # (B, 7) with chunk caching
```

### Flow Matching Policy

```python
from lerobot_policy_pi0_lite_flow import PI0LiteFlowConfig, PI0LiteFlowPolicy

config = PI0LiteFlowConfig(
    horizon=16, action_dim=7, cond_dim=256,
    hidden_dim=256, num_layers=4, inference_steps=8,
)
policy = PI0LiteFlowPolicy(config, dataset_stats=stats)

# Same interface as autoregressive policy
output = policy.forward(batch)        # flow matching loss
chunk = policy.predict_action_chunk(batch)  # Euler ODE integration
action = policy.select_action(batch)  # with chunk caching
```

## References

- **π0**: Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", arXiv:2410.24164, 2024.
- **OpenVLA**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", arXiv:2406.09246, 2024.
- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.
- **LIBERO**: Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", NeurIPS 2023.
