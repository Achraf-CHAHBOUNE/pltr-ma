# PLTR-MA (P3) experiment container — GPU-ready, CPU-compatible.
#
# The SAME image runs a fast CPU smoke test locally and the full publication run
# on the cluster; only environment variables change (see DOCKER_README.md).
# run_experiment.py auto-detects CUDA and falls back to CPU when no GPU is visible.
#
# Match BASE_IMAGE to the CUDA version your cluster's driver supports, e.g.:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime -t pltr-ma .
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# deps first so the layer caches across code edits (torch comes from the base image)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# experiment code
COPY code/ /app/

# results are written here; mount a host directory over it to keep the outputs
RUN mkdir -p /results
VOLUME ["/results"]

ENTRYPOINT ["python", "run_experiment.py"]
