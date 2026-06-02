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

## References

- Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
