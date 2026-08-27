#!/usr/bin/env bash
# =============================================================================
# PLTR-MA — one-command runner.
#
#   ./run.sh smoke     quick 2-minute check that everything works   (do this first)
#   ./run.sh n100      the full N=100 cluster study     (~3-6 h on 16 cores)
#   ./run.sh n10       N=10 study                       (~1 h on 16 cores)
#   ./run.sh n3        N=3 study (reproduces the paper) (under 1 h)
#
# Add `trace` to run on the real Alibaba workload instead of synthetic arrivals:
#   ./run.sh n100 trace
#
# Everything is auto-detected: GPU if present, CPU otherwise; worker count from
# the number of cores. Results land in ./results/<name>/ on this machine.
# =============================================================================
set -euo pipefail

MODE="${1:-}"
ENVKIND="${2:-synthetic}"
IMAGE="pltr-ma:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Windows (Git Bash / MSYS) needs care with paths:
#  - HOST paths handed to docker.exe must be Windows-style  -> use `pwd -W`
#  - CONTAINER paths ("/results") must NOT be rewritten     -> prefix with "//"
# On Linux/macOS both are plain and CP stays empty.
CP=""
HOSTDIR="$HERE"
case "$(uname -s 2>/dev/null || echo Linux)" in
  MINGW*|MSYS*|CYGWIN*)
    HOSTDIR="$(cd "$HERE" && pwd -W 2>/dev/null || echo "$HERE")"
    CP="/"
    ;;
esac

if [ -z "$MODE" ]; then
  sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
fi

# --- 0. sanity -------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not on PATH."
  echo "Install Docker Engine: https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to the Docker daemon."
  echo "Try:  sudo systemctl start docker    (or add your user to the 'docker' group)"
  exit 1
fi

# --- 1. validate arguments BEFORE the expensive build ----------------------
case "$MODE" in
  smoke|n3|n10|n100) ;;
  *) echo "ERROR: unknown mode '$MODE'. Use: smoke | n3 | n10 | n100"; exit 1 ;;
esac
if [ "$ENVKIND" = "trace" ] && [ ! -f "$HERE/code/trace_alibaba.csv" ]; then
  echo "ERROR: real-trace mode needs code/trace_alibaba.csv"
  echo "Generate it once with:"
  echo "  python code/make_trace.py alibaba /path/to/batch_task.csv \\"
  echo "         --out code/trace_alibaba.csv --bucket 10 --max-steps 5000"
  echo "(batch_task.csv comes from https://github.com/alibaba/clusterdata)"
  exit 1
elif [ "$ENVKIND" != "trace" ] && [ "$ENVKIND" != "synthetic" ]; then
  echo "ERROR: unknown workload '$ENVKIND'. Use: synthetic (default) | trace"; exit 1
fi

# --- 2. build (cached after the first time) --------------------------------
# NOTE: must happen BEFORE the GPU probe below, which runs the image.
echo "==> Building image (first run takes a few minutes, then it's cached)"
docker build -t "$IMAGE" "$HOSTDIR"

# --- 2. hardware ------------------------------------------------------------
# DEFAULT IS CPU, DELIBERATELY. The policy networks are ~5k parameters, so GPU
# kernel-launch overhead dominates and a GPU gives no speed-up here (measured).
# What actually helps is running many independent runs in parallel, one per core.
# Forcing CPU also avoids each worker allocating its own CUDA context.
#   Override with:  USE_GPU=1 ./run.sh n100
CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
WORKERS=$(( CORES > 1 ? CORES - 1 : 1 ))

GPU_ARGS=()
DEVICE_ENV=(-e DEVICE=cpu)
DEVICE_NOTE="CPU x $WORKERS workers (recommended — GPU gives no speed-up at this network size)"

if [ "${USE_GPU:-0}" = "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && docker run --rm --gpus all \
       --entrypoint python "$IMAGE" \
       -c "import torch;exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    GPU_ARGS=(--gpus all)
    DEVICE_ENV=()
    DEVICE_NOTE="GPU (forced via USE_GPU=1); workers use the 'spawn' start method"
  else
    echo "WARNING: USE_GPU=1 but no usable GPU was found in the container. Using CPU."
  fi
fi

# --- 3. mode -> settings ---------------------------------------------------
case "$MODE" in
  smoke)
    NAME=smoke;  SERVERS=3;   ITERS=6;   SEEDS=2; ABL=2; GRID=4;  MC=10000; EVALEP=2
    EXTRA=(-e EVAL_EVERY=3 -e V_GRID=0.3,3.0 -e DELTA_GRID=0.005,0.05)
    ETA="about 2 minutes" ;;
  n3)
    NAME=n3;     SERVERS=3;   ITERS=400; SEEDS=10; ABL=5; GRID=8; MC=200000; EVALEP=10
    EXTRA=(); ETA="under 1 hour with $WORKERS workers" ;;
  n10)
    NAME=n10;    SERVERS=10;  ITERS=400; SEEDS=5;  ABL=3; GRID=8; MC=200000; EVALEP=10
    EXTRA=(-e BLOCKS=convergence,pareto); ETA="about 1 hour with $WORKERS workers" ;;
  n100)
    NAME=n100;   SERVERS=100; ITERS=400; SEEDS=5;  ABL=3; GRID=8; MC=200000; EVALEP=10
    EXTRA=(-e BLOCKS=convergence,pareto)
    ETA="~37 core-hours => roughly $(( 37 / (WORKERS>0?WORKERS:1) + 1 ))h with $WORKERS workers" ;;
esac

TRACE_ARGS=()
if [ "$ENVKIND" = "trace" ]; then
  TRACE_ARGS=(-e "TRACE_PATH=${CP}/data/trace_alibaba.csv")
  NAME="${NAME}_trace"
fi

mkdir -p "$HERE/results"

# --- 4. go -----------------------------------------------------------------
cat <<EOF

------------------------------------------------------------------
  PLTR-MA run:  $NAME
  servers (N):  $SERVERS        seeds: $SEEDS
  hardware:     $DEVICE_NOTE
  cores:        $CORES  ->  $WORKERS parallel workers
  workload:     $ENVKIND
  expected:     $ETA
  results ->    $HERE/results/$NAME/
------------------------------------------------------------------

EOF

if [ "$MODE" = "n100" ] || [ "$MODE" = "n10" ]; then
  echo "This is a long run. If you are on SSH, start it inside tmux or screen"
  echo "so it survives a disconnect:    tmux new -s pltr   then re-run this."
  echo
fi

exec docker run --rm "${GPU_ARGS[@]}" \
  -v "$HOSTDIR/results:${CP}/results" \
  -v "$HOSTDIR/code:${CP}/data:ro" \
  -e OUT_DIR="${CP}/results/$NAME" \
  -e N_SERVERS="$SERVERS" \
  -e N_ITERATIONS="$ITERS" \
  -e N_SEEDS="$SEEDS" \
  -e PARETO_SEEDS="$SEEDS" \
  -e BASELINE_SEEDS="$SEEDS" \
  -e ABLATION_SEEDS="$ABL" \
  -e EVAL_EPISODES="$EVALEP" \
  -e PARETO_GRID="$GRID" \
  -e HV_MC="$MC" \
  -e N_WORKERS="$WORKERS" \
  "${DEVICE_ENV[@]}" "${TRACE_ARGS[@]}" "${EXTRA[@]}" \
  "$IMAGE"
