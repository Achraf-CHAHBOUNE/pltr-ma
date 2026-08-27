# PLTR-MA — Quick Start

Constrained multi-agent RL for cloud resource allocation.
Everything runs in one Docker container. **Nothing to install except Docker.**

---

## Run it

```bash
cd p3
chmod +x run.sh

./run.sh smoke      # 1. sanity check      (~2 minutes)
./run.sh n100       # 2. the real study    (~1-2 days)
```

That's it. GPU, CPU, and core count are detected automatically.
Results appear in `results/<name>/`.

> **Long runs over SSH:** start `tmux new -s pltr` first, then run the command,
> so it survives a disconnection. Detach with `Ctrl-B` then `D`; return with
> `tmux attach -t pltr`.

---

## The four modes

| Command | Cluster size | Seeds | Wall-clock on 16 cores |
|---|---|---|---|
| `./run.sh smoke` | 3 | 2 | 2 minutes — proves the pipeline works |
| `./run.sh n3` | 3 | 10 | under 1 hour — reproduces the paper |
| `./run.sh n10` | 10 | 5 | about 1 hour |
| `./run.sh n100` | **100** | 5 | **3–6 hours** |

*(N=100 is ~50 training runs at ~45 min each ≈ 37 core-hours. The runner executes
them in parallel, one per core, so wall-clock time is roughly 37 ÷ (cores − 1).
On 4 cores expect ~12 hours; on 32, closer to 90 minutes.)*

Add `trace` to use the real Alibaba production workload instead of synthetic
arrivals (see "Real workload" below):

```bash
./run.sh n100 trace
```

---

## Hardware notes — please read before allocating a machine

**Cores matter far more than a GPU here.** The policy networks are tiny (~5,000
parameters each), so GPU kernel-launch overhead dominates and a GPU gives almost
no benefit — it can even be slower. What the workload actually needs is **many CPU
cores**, because the runner executes independent training runs in parallel
(one per core, set automatically).

Measured cost per training iteration:

| Cluster size N | sec / iteration | relative |
|---|---|---|
| 3 | 0.23 | 1.0× |
| 10 | 0.87 | 3.7× |
| 30 | 1.70 | 7.4× |
| **100** | **5.37** | **23×** |

Cost grows roughly linearly with N, because every server has its own actor
network updated in sequence.

**Recommended:** 16+ CPU cores, 16 GB RAM, 10 GB disk. A GPU is optional.
Memory scales with parallel workers (~1 GB each at N=100), so if RAM is tight,
lower the worker count rather than the seed count.

### Sanity check on first full run
The N=100 path has been verified end-to-end (all blocks, figures written). One
thing to confirm when the *full* run finishes: the five variants should reach
**different** final evaluation returns. In a 2-iteration smoke test they come out
identical, which is expected — the policies are barely trained and share a seed —
but at 400 iterations they must diverge. If they don't, something is wrong.

---

## Real workload (optional)

The default is a synthetic Poisson arrival process. To replay a real production
trace, convert the raw CSV once — **also inside Docker**, nothing to install:

```bash
# download batch_task.csv from https://github.com/alibaba/clusterdata, then:
docker run --rm --entrypoint python \
  -v "$PWD/code:/app" -v "/path/to/data:/data:ro" \
  pltr-ma:latest make_trace.py alibaba /data/batch_task.csv \
  --out /app/trace_alibaba.csv --bucket 10 --max-steps 5000
```

Then add `trace` to any command: `./run.sh n100 trace`

---

## Dependencies

**Nothing to install — Docker handles everything.** For reference, the image gets
torch from its CUDA base image plus `requirements.txt` (numpy, matplotlib,
gymnasium, pandas). torch is deliberately *not* in `requirements.txt`: listing it
would let pip replace the CUDA build with a CPU-only wheel and disable GPU support.

---

## Output

Each run writes to `results/<name>/`:

- `results.npz` — every metric, per-seed mean **and** standard deviation
- `fig_convergence.png`, `fig_pareto.png`, `fig_violation.png`,
  `fig_ablation.png`, `fig_stability.png`

Figures can be regenerated without re-running: `python code/plot.py results/<name>`

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `cannot talk to the Docker daemon` | `sudo systemctl start docker`, or add your user to the `docker` group |
| CUDA / driver mismatch on build | `docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime -t pltr-ma .` |
| Run seems stuck | It isn't — N=100 takes ~45 min per run and prints only at evaluation checkpoints |
| Plotting step fails | Harmless: `results.npz` is written *before* plotting; regenerate with `python code/plot.py results/<name>` |
| Out of memory | Lower the worker count: `N_WORKERS` is cores−1 by default; each worker needs ~1 GB at N=100 |
| Want to use the GPU anyway | `USE_GPU=1 ./run.sh n100` (expect no speed-up — see hardware notes) |

## What has been tested

Verified on this codebase before hand-off: image builds; full pipeline runs end-to-end
at N=3 and **N=100** (all blocks, `results.npz` + figures written); parallel workers
verified with and without CUDA; argument guards reject bad input before the build.

Three bugs were found and fixed during that testing: parallel workers crashed on any
CUDA machine (`fork` cannot re-initialize CUDA — now uses `spawn`); `plot.py` ignored
its directory argument; and on Windows Git Bash the container paths were rewritten,
sending output outside the project.

Full configuration reference: **`DOCKER_README.md`**
