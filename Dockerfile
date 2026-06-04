# ─────────────────────────────────────────────────────────────────────────────
# BC Regression — LIBERO Simulation Rollout
#
# Base:  CUDA 12.8 + cuDNN + Ubuntu 24.04
# Uses:  MuJoCo 3.2.7, robosuite 1.4.0, LIBERO (cloned at build time),
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
        python3.12 python3.12-dev python3.12-venv \
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

ENV PATH=/opt/venv/bin:${PATH}
ENV PYTHONPATH=/opt/LIBERO:${PYTHONPATH}
ENV LIBERO_CONFIG_PATH=/opt/libero-config
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
RUN python3.12 -m venv /opt/venv \
 && python -m pip install --upgrade pip setuptools wheel

# ── MuJoCo ───────────────────────────────────────────────────────────────────
# mujoco==2.3.7 has no Python 3.12 wheel and tries to build from source.
RUN python -m pip install --only-binary=mujoco mujoco==3.2.7

# ── robosuite 1.4.0 ───────────────────────────────────────────────────────────
# LIBERO requires exactly this version
RUN python -m pip install robosuite==1.4.0

# LIBERO's setup.py does not declare its runtime dependencies.
RUN python -m pip install \
    "pyyaml>=6.0" \
    "easydict>=1.9" \
    "cloudpickle>=2.1" \
    "gym==0.25.2" \
    "bddl==1.0.1" \
    "termcolor>=2.0" \
    "imageio>=2.31" \
    "tqdm>=4.64"

# ── LIBERO (from source) ──────────────────────────────────────────────────────
# Clone and install; skip bddl/init_states download at build time –
# they are bundled with the package via the libero.libero module.
RUN git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
 && python -m pip install -e /opt/LIBERO \
 && mkdir -p "${LIBERO_CONFIG_PATH}" /tmp/numba-cache \
 && LIBERO_ROOT="/opt/LIBERO/libero/libero" \
 && printf "assets: %s/assets\nbddl_files: %s/bddl_files\nbenchmark_root: %s\ndatasets: %s/../datasets\ninit_states: %s/init_files\n" "${LIBERO_ROOT}" "${LIBERO_ROOT}" "${LIBERO_ROOT}" "${LIBERO_ROOT}" "${LIBERO_ROOT}" > "${LIBERO_CONFIG_PATH}/config.yaml"

# ── project Python deps ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY pyproject.toml requirements.txt ./
COPY src ./src

# Install project with all extras (encoder + dev) but WITHOUT robosuite/mujoco
# overrides – those are already installed above.
RUN python -m pip install -e ".[dev,encoder]" \
 && python -m pip install matplotlib pyarrow \
 && python -c "import bddl, gym, robosuite, yaml; import libero; from libero.libero import benchmark, get_libero_path; from libero.libero.envs import OffScreenRenderEnv; print('libero_paths=' + str(list(libero.__path__))); print(sorted(benchmark.get_benchmark_dict())); print(get_libero_path('assets')); print(OffScreenRenderEnv)"

# ── copy remaining source ─────────────────────────────────────────────────────
COPY scripts ./scripts
COPY tests   ./tests

# ── runtime environment ───────────────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
# Silence HuggingFace symlink warning on Linux
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

CMD ["python", "scripts/sim_rollout_regression.py", "--help"]
