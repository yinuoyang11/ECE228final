# Server Training Guide

This guide targets an Ubuntu server with an NVIDIA GPU. The repository Docker
image uses CUDA 12.8, persists Hugging Face downloads on the host, and writes
checkpoints to the host `runs/` directory.

## 1. Verify the NVIDIA Driver

The host needs a working NVIDIA driver:

```bash
nvidia-smi
```

## 2. Install Docker Engine

Follow the official Docker Engine instructions for Ubuntu. The commands below
configure Docker's apt repository and install the required packages:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the Docker group, or run `newgrp docker` for
the current shell.

## 3. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## 4. Clone and Build

```bash
git clone --branch Encoder https://github.com/yinuoyang11/ECE228final.git
cd ECE228final
bash scripts/docker_build.sh
```

For a future headless LIBERO rollout image with EGL simulator dependencies:

```bash
INSTALL_LIBERO=1 bash scripts/docker_build.sh
```

## 5. Inspect the Dataset

The default dataset is `HuggingFaceVLA/libero`. Read its metadata and list task
indices without downloading the full video dataset:

```bash
docker run --rm ece228-pi0-lite-flow:latest \
  python scripts/inspect_libero_metadata.py --list-tasks
```

The current dataset contains 40 tasks, 1693 episodes, and 273465 frames.

## 6. Run a Smoke Test

This downloads the required dataset shards and CLIP weights on first use:

```bash
GPU_ID=1 STEPS=3 MAX_SAMPLES=16 BATCH_SIZE=2 RUN_NAME=smoke \
  bash scripts/docker_train.sh
```

Check that `runs/pi0_lite_flow/smoke/checkpoint_final.pt` exists.
Set `GPU_ID=1` to expose only host GPU 1 to the container. Omit it to expose
all GPUs.

## 7. Train Selected Tasks

Start with a small related task group:

```bash
GPU_ID=1 STEPS=10000 MAX_SAMPLES=0 BATCH_SIZE=8 SAVE_EVERY=5000 RUN_NAME=tasks_20_21_22 \
  bash scripts/docker_train.sh --task-indices 20 21 22
```

`MAX_SAMPLES=0` means that all frames from the selected tasks are eligible for
sampling. Task indices 20, 21, and 22 are basket pick-and-place tasks.
`SAVE_EVERY=5000` writes an intermediate checkpoint every 5000 optimizer steps.
Set `SAVE_EVERY=0` to keep only `checkpoint_final.pt`.

## 8. Train All Tasks

```bash
STEPS=100000 MAX_SAMPLES=0 BATCH_SIZE=8 SAVE_EVERY=10000 RUN_NAME=clip_flow_full \
  bash scripts/docker_train.sh
```

Use a smaller `BATCH_SIZE` if the GPU runs out of memory. Run long jobs inside
`tmux` so they survive SSH disconnects:

```bash
tmux new -s ece228
```

## 9. Checkpoints and Offline Evaluation

Training artifacts are saved on the host:

```text
runs/pi0_lite_flow/<run-name>/
├── args.json
├── metrics.csv
├── checkpoint_step_*.pt
└── checkpoint_final.pt
```

Run held-out offline evaluation:

```bash
docker run --rm --gpus '"device=1"' --ipc=host \
  --volume "$PWD/runs:/workspace/runs" \
  --volume "$HOME/.cache/huggingface:/cache/huggingface" \
  ece228-pi0-lite-flow:latest \
  python scripts/eval_flow_custom.py \
    runs/pi0_lite_flow/tasks_20_21_22/checkpoint_final.pt \
    --num-samples 512 \
    --batch-size 128 \
    --output runs/pi0_lite_flow/tasks_20_21_22/eval.csv
```

This is offline held-out action prediction evaluation. It reports action MSE,
L1 error, predicted action smoothness, and latency. It does not launch the
LIBERO simulator or measure task success rate.

Delete intermediate checkpoints after a completed run while keeping the final
model:

```bash
find runs/pi0_lite_flow/tasks_20_21_22 \
  -name 'checkpoint_step_*.pt' -delete
```

## 10. Headless Rollout Videos

Build the image with the Linux-only LIBERO simulator dependencies:

```bash
INSTALL_LIBERO=1 bash scripts/docker_build.sh
```

Run headless rollouts on GPU 1 and save a side-by-side agentview and wrist-camera
MP4 for every episode:

```bash
docker run --rm --gpus '"device=1"' --ipc=host \
  --env MUJOCO_GL=egl \
  --env PYOPENGL_PLATFORM=egl \
  --volume "$PWD/runs:/workspace/runs" \
  --volume "$HOME/.cache/huggingface:/cache/huggingface" \
  ece228-pi0-lite-flow:latest \
  python scripts/eval_libero_rollout.py \
    runs/pi0_lite_flow/tasks_20_21_22_bs128/checkpoint_final.pt \
    --suite libero_object \
    --dataset-task-indices 20 21 22 \
    --episodes-per-task 10 \
    --video-view both \
    --video-dir runs/pi0_lite_flow/tasks_20_21_22_bs128/rollout_videos
```

The output directory contains videos and aggregate success-rate metrics:

```text
rollout_videos/
├── libero_object_task00_episode00_success.mp4
├── libero_object_task00_episode01_failure.mp4
└── rollout_metrics.json
```

The `HuggingFaceVLA/libero` dataset task indices and LIBERO benchmark suite-local
task ids are different. Use `--dataset-task-indices` to pass the same global
indices used for training. The rollout script maps them to suite-local ids by
instruction text. Use `--task-ids` only when intentionally passing LIBERO
suite-local ids.

## 11. Fine-Tune CLIP After a Frozen-Encoder Run

Before fine-tuning, rerun the rollout command above with
`--dataset-task-indices 20 21 22`. Older commands that used
`--suite libero_object --task-ids 0 1 2` evaluated different tasks.

If the corrected frozen-CLIP rollout still fails to fit the tasks, warm-start
from the existing checkpoint and unfreeze the final CLIP vision layers. Keep
the text encoder frozen initially because the main missing signal is usually
visual control information:

```bash
nohup env GPU_ID=1 STEPS=5000 MAX_SAMPLES=0 BATCH_SIZE=16 NUM_WORKERS=4 \
  SAVE_EVERY=0 RUN_NAME=tasks_20_21_22_clip_vision_ft \
  bash scripts/docker_train.sh \
    --task-indices 20 21 22 \
    --init-checkpoint runs/pi0_lite_flow/tasks_20_21_22_bs128/checkpoint_final.pt \
    --finetune-clip-vision-layers 2 \
    --finetune-clip-text-layers 0 \
    --clip-lr 1e-5 \
  > train_tasks_20_21_22_clip_vision_ft.log 2>&1 &
```

The fusion layers and flow decoder continue to use `--lr 1e-4`. The unfrozen
CLIP parameters use the smaller `--clip-lr`. CLIP fine-tuning stores additional
backbone weights in the checkpoint and uses more GPU memory than frozen-CLIP
training. Start with `BATCH_SIZE=16` and reduce it if needed.

Use `--finetune-clip-vision-layers -1` only when intentionally fine-tuning the
entire CLIP vision branch. Fine-tune text layers later only if rollout evidence
shows language confusion between tasks.

## References

- Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
