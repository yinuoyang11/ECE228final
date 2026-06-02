# π0-Lite: Flow Matching vs Autoregressive vs Regression Action Decoders

This package provides a LeRobot-style custom policy plugin for comparing three action-generation approaches for language-conditioned robot manipulation:

1. **Conditional Flow Matching (CFM)** — the main π0-Lite method
2. **Discretized Autoregressive Action Tokens** — OpenVLA-style baseline
3. **Behavior Cloning Regression (MSE)** — simple MLP baseline

All three methods consume a precomputed conditioning vector `batch["observation.cond"]` and predict continuous action chunks `(B, horizon, action_dim)`.

## Architecture Overview

### Flow Matching (Main Method)
- Trains a velocity field network with the flow matching objective
- At inference, integrates the learned velocity field from Gaussian noise using Euler steps
- Supports both MLP-FiLM and Temporal Transformer architectures

### Discretized Autoregressive Action Tokens (Baseline)
- Each action dimension is binned into 256 discrete tokens (uniform bins)
- A causal Transformer autoregressively predicts the token sequence
- Trained with cross-entropy loss (next-token prediction)
- At inference, tokens are sampled one at a time and decoded back to continuous values
- Follows the **OpenVLA** paradigm (Kim et al., 2024)

### BC Regression (Baseline)
- Directly predicts continuous action chunks with MSE loss
- Simple MLP with residual connections

## Input Format

All policies expect a precomputed fused vision-language conditioning vector:

```python
batch["observation.cond"]  # torch.Tensor, shape: (B, cond_dim)
```

Default `cond_dim = 256`. During training, the batch must also include:

```python
batch = {
    "observation.cond": cond.float(),       # (B, 256) by default
    "action": actions.float(),              # (B, horizon, action_dim)
    "action_is_pad": pad_mask.bool(),       # optional, (B, horizon)
}
```

At inference, only the condition vector is required:

```python
batch = {"observation.cond": cond.float()}
action = policy.select_action(batch)   # (B, action_dim)
```

## Quick Start

### Environment Setup

```bash
conda create -n ece228-pi0lite python=3.12
conda activate ece228-pi0lite
pip install torch einops pytest matplotlib
pip install -e .
```

### Run Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

### Train All Baselines

```bash
PYTHONPATH=src python scripts/train_baselines.py --epochs 200 --hidden_dim 128
```

This trains all three methods on synthetic demonstration data and saves:
- Model checkpoints → `checkpoints/`
- Training history + eval results → `checkpoints/results.json`

### Evaluate and Generate Plots

```bash
pip install matplotlib
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
│   ├── modeling_autoregressive.py        # Autoregressive decoder + policy
│   ├── modeling_pi0_lite_flow.py         # Flow matching decoder + policy
│   └── processor_pi0_lite_flow.py        # Data processors
├── scripts/
│   ├── train_baselines.py               # Train all three methods
│   └── evaluate_baselines.py            # Generate comparison plots
├── tests/
│   ├── test_autoregressive.py           # 15 tests for autoregressive baseline
│   └── test_flow_decoder.py             # 6 tests for flow matching
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
