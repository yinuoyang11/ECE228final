# pi0-lite Flow Action Decoder

This package provides a LeRobot-style custom policy plugin centered on a conditional flow matching action decoder for continuous action chunks.

The current module expects a precomputed conditioning vector in `batch["observation.cond"]` and trains on raw action chunks in `batch["action"]`.

## Vision-Language Condition Input

This policy does not directly consume raw images, text strings, token ids, or
vision-language token sequences. Instead, it expects a precomputed fused
vision-language conditioning vector for each batch item.

The default condition key and shape are:

```python
batch["observation.cond"]  # torch.Tensor, shape: (B, cond_dim)
```

The default configuration uses `cond_dim = 256`, so the default condition tensor
shape is:

```python
batch["observation.cond"].shape == (B, 256)
```

The fallback key `batch["cond"]` is also accepted. During training, the batch
must also include action chunks:

```python
batch = {
    "observation.cond": cond.float(),       # (B, 256) by default
    "action": actions.float(),              # (B, horizon, action_dim)
    "action_is_pad": pad_mask.bool(),       # optional, (B, horizon)
}
```

With the default policy config, actions have shape `(B, 16, 7)`. At inference
time, only the condition vector is required:

```python
batch = {
    "observation.cond": cond.float(),  # (B, 256) by default
}

action = policy.select_action(batch)   # (B, action_dim)
```

If the upstream vision-language encoder outputs a single embedding with a
different width, either set `config.cond_dim` to that width or add a projection
layer before passing the embedding to the policy:

```python
cond = projector(vl_embedding)  # (B, 256)
batch["observation.cond"] = cond
```

If the upstream encoder outputs a token sequence such as `(B, N, D)`, pool or
select a representative token first, then optionally project it to `cond_dim`:

```python
pooled = vl_tokens.mean(dim=1)  # (B, D)
cond = projector(pooled)        # (B, 256)
```

## Environment

```powershell
conda create -n ece228-pi0lite python=3.12
conda activate ece228-pi0lite
pip install lerobot torch einops pytest
pip install -e .
pytest
```

For local core checks before LeRobot is installed, run:

```powershell
$env:PYTHONPATH="src"
pytest
```
