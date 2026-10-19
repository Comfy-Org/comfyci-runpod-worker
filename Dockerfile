# Generic ComfyUI CI worker: the ComfyUI commit under test is checked out at
# request time (src/checkout.py), so this image only bakes the environment.
# Models are NOT baked either — they live on the RunPod network volume,
# wired in via /comfy-config/extra_model_paths.yaml.
#
# One image per torch build. The build args come from the `images` block of
# manifest/lanes.json, so a lane's environment is data: build-image.yml builds
# every image listed there and tags it ghcr.io/comfy-org/comfyci-runpod-worker:<image>.
# Plain Ubuntu base: torch's CUDA wheels bundle the CUDA runtime and the GPU
# driver comes from the RunPod host, so a CUDA base image is dead weight. Python
# comes from uv (python-build-standalone), so a lane can run a Python that
# Ubuntu does not package.
FROM ubuntu:24.04

ARG IMAGE_ID=cu130-t2141
ARG PYTHON_VERSION=3.13
ARG CUDA_VERSION=13.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_SPEC="torch==2.14.1 torchvision==0.29.1"
# Git sha of this repository the image was built from; recorded in every run.
ARG IMAGE_SHA=dev

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# All python/pip below resolve to this venv, including the handler's runtime
# `pip install -r requirements.txt` when a commit changes requirements.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    COMFY_ROOT=/comfyui \
    EXTRA_MODEL_PATHS=/comfy-config/extra_model_paths.yaml \
    BAKED_REQ_HASH_FILE=/comfy-config/requirements.sha256 \
    COMFY_CONSTRAINTS_FILE=/comfy-config/constraints.txt \
    IMAGE_INFO_FILE=/comfy-config/image.json \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# gcc + libc headers: torch 2.14 dispatches some ops (torch._native, e.g. the
# bmm_outer_product the Qwen3 and Gemma text encoders hit) to Triton kernels, and
# Triton builds its CUDA driver shim with the system C compiler on first use.
# Without one every such workflow fails with "Failed to find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# The lane's interpreter and a seeded venv (pip stays available for the runtime
# requirements install).
RUN uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} --seed /opt/venv \
    && pip install --upgrade pip

# Torch first (large, cache-friendly), pinned to the lane's build and CUDA
# channel. The same pins become a constraints file, so a later requirements
# install can never swap the torch build underneath the lane.
RUN mkdir -p /comfy-config \
    && pip install ${TORCH_SPEC} --index-url ${TORCH_INDEX} \
    && for spec in ${TORCH_SPEC}; do echo "$spec" >> ${COMFY_CONSTRAINTS_FILE}; done

# Pre-clone ComfyUI and install its requirements; per-request checkout only
# re-runs pip when the target commit's requirements.txt hash differs.
RUN git clone --depth 50 https://github.com/Comfy-Org/ComfyUI ${COMFY_ROOT} \
    && pip install -r ${COMFY_ROOT}/requirements.txt -c ${COMFY_CONSTRAINTS_FILE} \
    && sha256sum ${COMFY_ROOT}/requirements.txt | cut -d' ' -f1 > ${BAKED_REQ_HASH_FILE}

# Pinned: SDK behaviour around handler outputs changed between releases.
RUN pip install runpod==1.12.0 nvidia-ml-py

# What this image is, echoed into every run record as its `env` block.
RUN printf '{"image": "%s", "image_sha": "%s", "python": "%s", "cuda": "%s", "torch_spec": "%s", "torch_index": "%s"}\n' \
        "${IMAGE_ID}" "${IMAGE_SHA}" "${PYTHON_VERSION}" "${CUDA_VERSION}" "${TORCH_SPEC}" "${TORCH_INDEX}" \
        > ${IMAGE_INFO_FILE}

COPY docker/extra_model_paths.yaml /comfy-config/extra_model_paths.yaml
COPY src/ /worker/src/

WORKDIR /worker/src
CMD ["python", "-u", "handler.py"]
