# PLTR-MA (P3) — running the experiments in Docker

> **Just want to run it? Use `./run.sh` — see `QUICKSTART.md`.**
> `./run.sh smoke` then `./run.sh n100`. It handles build, device, core count and
> mounts for you. This file is the full reference for driving the container manually.

One image runs everything. The **same** container does a 2-minute smoke test on a
laptop and the full multi-seed publication run on a server — only environment
variables change.

### Device policy (important)

`run_experiment.py` auto-detects CUDA, **but `run.sh` deliberately forces `DEVICE=cpu`.**
The policy networks are ~5,000 parameters, so GPU kernel-launch overhead dominates
and a GPU gives no measured speed-up. What helps is running many independent runs
in parallel — one per core, via `N_WORKERS`. Override with `USE_GPU=1 ./run.sh n100`.

If you do use a GPU together with `N_WORKERS > 1`, the pool switches to the `spawn`
start method automatically — a *forked* process cannot re-initialize CUDA and would
crash with `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.

---

## 1. Build

```bash
cd p3
docker build -t pltr-ma .
```

If the cluster needs a different CUDA version, match the base image to its driver:

```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime -t pltr-ma .
```

## 2. Smoke test (CPU, ~2 min) — do this first

```bash
docker run --rm -v "$PWD/results:/results" \
  -e OUT_DIR=/results/smoke \
  -e N_ITERATIONS=6 -e N_SEEDS=2 -e PARETO_SEEDS=2 -e BASELINE_SEEDS=2 \
  -e ABLATION_SEEDS=2 -e EVAL_EVERY=3 -e EVAL_EPISODES=2 \
  -e PARETO_GRID=4 -e HV_MC=10000 \
  -e V_GRID=0.3,3.0 -e DELTA_GRID=0.005,0.05 \
  pltr-ma
```

Expect `results/smoke/results.npz` plus five PNGs. The numbers are meaningless at this
size — this only proves the pipeline runs.

## 3. Full run on the GPU cluster

```bash
docker run --rm --gpus all -v "$PWD/results:/results" \
  -e OUT_DIR=/results/synthetic -e N_SERVERS=3 -e N_SEEDS=10 pltr-ma
```

Or run both environments via compose:

```bash
docker compose run --rm synthetic
docker compose run --rm trace
```

**Note:** the GPU mainly buys *throughput* (many seeds/variants/N in parallel), not
per-run speed — the networks are small and the bottleneck is CPU env-stepping.

## 4. The real workload trace (environment 2)

The trace env needs a normalised CSV. Generate one of:

```bash
# (a) synthetic sample — for testing only, NOT for the paper
python code/make_trace.py synthetic --out code/trace_sample.csv --steps 600

# (b) the real Alibaba cluster-trace-v2018 — for the paper
#     download batch_task.csv from https://github.com/alibaba/clusterdata
python code/make_trace.py alibaba /path/to/batch_task.csv \
       --out code/trace_alibaba.csv --bucket 10 --max-steps 5000
```

Then point the run at it:

```bash
docker run --rm --gpus all -v "$PWD/results:/results" -v "$PWD/code:/data:ro" \
  -e OUT_DIR=/results/trace -e TRACE_PATH=/data/trace_alibaba.csv pltr-ma
```

---

## Configuration (all optional, sensible defaults)

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | auto | `cuda` / `cpu`; auto-detected if unset |
| `TRACE_PATH` | *(unset)* | unset ⇒ synthetic Poisson env; set ⇒ real-trace replay |
| `N_SERVERS` | 3 | cluster size (agents) |
| `ARRIVAL_RATE` | 1.4 | ρ≈1.12 — constraints bind (synthetic env only) |
| `N_ITERATIONS` | 400 | training updates per run |
| `N_SEEDS` | 5 | convergence seeds |
| `PARETO_SEEDS` / `BASELINE_SEEDS` | 5 / 5 | seeds for the front and the fixed-weight baseline |
| `ABLATION_SEEDS` | 3 | seeds for the V / δ ablations |
| `PARETO_GRID` | 8 | simplex resolution of the ω-sweep |
| `BASELINE_GRID` | 0 | >0 ⇒ densify the fixed-weight MORL baseline front |
| `HV_MC` | 200000 | Monte-Carlo samples for hypervolume |
| `COST_LIMIT` / `LAM_LR` / `LAM_INIT` | 0.0 / 0.2 / 2.0 | PPO-Lagrangian baseline |
| `BLOCKS` | `convergence,pareto,ablation` | which blocks to run |
| `N_WORKERS` | 1 | parallel runs; set to (cores − 1). This is the main speed lever. |
| `OUT_DIR` | `/results` | output directory (mount it!) |

## Outputs

`$OUT_DIR/results.npz` (all metrics, per-seed mean **and** std) plus
`fig_convergence`, `fig_violation`, `fig_pareto`, `fig_ablation`, `fig_stability` (PNG).

## Troubleshooting

- **No GPU used?** Check the banner the container prints — it reports
  `CUDA available` and the device. Ensure `--gpus all` and the NVIDIA container toolkit.
- **CUDA/driver mismatch:** rebuild with a `BASE_IMAGE` matching the cluster driver.
- **Plot step fails:** the run is still safe — `results.npz` is written before plotting,
  and figures can be regenerated with `python plot.py`.
