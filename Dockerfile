# Generic ComfyUI CI worker: the ComfyUI commit under test is checked out at
# request time (src/checkout.py), so this image only bakes the environment.
# Models are NOT baked either — they live on the RunPod network volume,
# wired in via /comfy-config/extra_model_paths.yaml.
# Plain Ubuntu base: torch's cu128 wheels bundle the CUDA runtime, and the
# GPU driver comes from the RunPod host, so a CUDA base image is dead weight.
FROM ubuntu:24.04

ARG PYTHON_VERSION=3.12
ARG TORCH_CHANNEL=https://download.pytorch.org/whl/cu128

# All python/pip below resolve to this venv, including the handler's runtime
# `pip install -r requirements.txt` when a commit changes requirements.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/comfyui \
    EXTRA_MODEL_PATHS=/comfy-config/extra_model_paths.yaml \
    BAKED_REQ_HASH_FILE=/comfy-config/requirements.sha256 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python${PYTHON_VERSION} -m venv /opt/venv \
    && pip install --upgrade pip

# Torch first (large, cache-friendly), pinned to the CUDA channel above.
RUN pip install torch torchvision torchaudio --index-url ${TORCH_CHANNEL}

# Pre-clone ComfyUI and install its requirements; per-request checkout only
# re-runs pip when the target commit's requirements.txt hash differs.
RUN git clone --depth 50 https://github.com/Comfy-Org/ComfyUI ${COMFY_ROOT} \
    && pip install -r ${COMFY_ROOT}/requirements.txt \
    && mkdir -p /comfy-config \
    && sha256sum ${COMFY_ROOT}/requirements.txt | cut -d' ' -f1 > ${BAKED_REQ_HASH_FILE}

RUN pip install runpod pynvml

COPY docker/extra_model_paths.yaml /comfy-config/extra_model_paths.yaml
COPY src/ /worker/src/

WORKDIR /worker/src
CMD ["python", "-u", "handler.py"]
