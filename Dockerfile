# ─────────────────────────────────────────────────────────────────────────────
# BC Regression — LIBERO Simulation Rollout
#
# Base:  CUDA 12.8 + cuDNN + Ubuntu 24.04
# Uses:  MuJoCo 2.3.7, robosuite 1.4.0, LIBERO (cloned at build time),
#        Python 3.12, ffmpeg for MP4 recording
#
# Build:
#   docker build -t ece228-regression .
#
# Run simulation rollout (10 episodes per task, tasks 20+21+22):
#   docker run --gpus all --rm \
#     -v $(pwd)/results:/workspace/results \
#     ece228-regression \
#     python scripts/sim_rollout_regression.py \
#       results/task202122_regression/checkpoint_final.pt \
#       --dataset-task-indices 20 21 22 \
#       --episodes-per-task 10 \
#       --suite libero_object \
#       --video-dir results/sim_rollout \
#       --device cuda
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# ── system packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        git curl wget ca-certificates \
        ffmpeg \
        # MuJoCo / EGL headless rendering
        libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
        libglfw3-dev libxrandr2 libxinerama-dev libxcursor-dev libxi-dev \
        # robosuite / LIBERO native deps
        build-essential patchelf libopenmpi-dev \
        # misc
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python/pip.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && python -m pip install --upgrade pip setuptools wheel --break-system-packages

# ── MuJoCo 2.3.7 ─────────────────────────────────────────────────────────────
# robosuite 1.4.0 expects mujoco >= 2.3.2
RUN python -m pip install mujoco==2.3.7 --break-system-packages

# ── robosuite 1.4.0 ───────────────────────────────────────────────────────────
# LIBERO requires exactly this version
RUN python -m pip install robosuite==1.4.0 --break-system-packages

# ── LIBERO (from source) ──────────────────────────────────────────────────────
# Clone and install; skip bddl/init_states download at build time –
# they are bundled with the package via the libero.libero module.
RUN git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
 && python -m pip install -e /opt/LIBERO --break-system-packages

# ── project Python deps ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY pyproject.toml requirements.txt ./
COPY src ./src

# Install project with all extras (encoder + dev) but WITHOUT robosuite/mujoco
# overrides – those are already installed above.
RUN python -m pip install -e ".[dev,encoder]" --break-system-packages \
 && python -m pip install matplotlib pyarrow --break-system-packages

# ── copy remaining source ─────────────────────────────────────────────────────
COPY scripts ./scripts
COPY tests   ./tests

# ── runtime environment ───────────────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
# Silence HuggingFace symlink warning on Linux
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Pre-create LIBERO config so it doesn't ask interactive questions
RUN mkdir -p /root/.libero \
 && printf "datasets_default_path: /data/libero_datasets\n" > /root/.libero/config.yaml

CMD ["python", "scripts/sim_rollout_regression.py", "--help"]
