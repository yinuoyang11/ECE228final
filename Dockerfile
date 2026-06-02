FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    linux-libc-dev \
    libosmesa6 \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY . .

RUN python -m pip install -r requirements.txt
RUN python -c "from importlib.metadata import version; import torch, torchcodec, transformers; from lerobot.datasets.lerobot_dataset import LeRobotDataset; from transformers import CLIPModel, CLIPTokenizer; print('torch=' + torch.__version__ + ' torchcodec=' + version('torchcodec') + ' transformers=' + transformers.__version__)"

ARG INSTALL_LIBERO=0
RUN if [ "${INSTALL_LIBERO}" = "1" ]; then \
        python -m pip install "lerobot[libero]==0.5.1"; \
    fi

CMD ["bash"]
