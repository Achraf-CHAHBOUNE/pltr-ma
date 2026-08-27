"""make_trace.py — build a normalised workload-trace CSV for WorkloadTrace.

Two entry points:

  synthetic(...)    generate a small, realistic-looking sample (diurnal load curve +
                    bursts, heterogeneous long-tailed task sizes) for LOCAL TESTING.
                    No download needed; this is what we develop the loader against.

  from_alibaba(...) convert a raw Alibaba cluster-trace-v2018 `batch_task` file into
                    the normalised schema. Run this ON THE CLUSTER once you have the
                    real trace: https://github.com/alibaba/clusterdata (v2018).

Output schema (env units, one row per task):

    arrival_step,cpu,mem,work,slack

`cpu,mem` live in ~[0.1,0.5], `work` in ~[1,5], `slack` in [10,30] — matching
CloudAllocationEnv's task ranges so the trace drops in without rescaling the env.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

# env-native ranges (keep in sync with cloud_env.CloudAllocationEnv)
CPU_RANGE = (0.1, 0.5)
MEM_RANGE = (0.1, 0.5)
WORK_RANGE = (1.0, 5.0)
SLACK_RANGE = (10, 30)


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def _write(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_step", "cpu", "mem", "work", "slack"])
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Synthetic sample (for local testing — no real trace required)
# --------------------------------------------------------------------------- #
def synthetic(path: str = "trace_sample.csv", n_steps: int = 600,
              base_rate: float = 1.4, seed: int = 0) -> str:
    """A realistic-*looking* workload: a diurnal sine load curve with random bursts
    and heterogeneous, long-tailed (lognormal) task sizes. Not a substitute for a
    real trace in the paper — it is only for developing/validating the loader."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_steps):
        # diurnal factor in [0.4, 1.6] + occasional 3x bursts
        diurnal = 1.0 + 0.6 * np.sin(2 * np.pi * t / 240.0)
        burst = 3.0 if rng.random() < 0.03 else 1.0
        rate = base_rate * diurnal * burst
        for _ in range(int(rng.poisson(rate))):
            cpu = _clip(rng.lognormal(mean=-1.4, sigma=0.6), *CPU_RANGE)
            mem = _clip(rng.lognormal(mean=-1.4, sigma=0.6), *MEM_RANGE)
            work = _clip(rng.lognormal(mean=0.7, sigma=0.7), *WORK_RANGE)
            slack = int(rng.integers(SLACK_RANGE[0], SLACK_RANGE[1] + 1))
            rows.append((t, round(cpu, 4), round(mem, 4), round(work, 4), slack))
    _write(path, rows)
    print(f"[synthetic] wrote {len(rows)} tasks over {n_steps} steps -> {path}")
    return path


# --------------------------------------------------------------------------- #
# Real Alibaba cluster-trace-v2018 converter (run on the cluster with the real file)
# --------------------------------------------------------------------------- #
def _robust_scale(x: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    """Map x into [lo, hi] using the 1st/99th percentiles, robustly.

    Two real properties of the Alibaba trace force this:
      * `plan_cpu` is almost constant (p50 = p95 = 100, i.e. 1 core), so a naive
        min-max divides by ~0 and collapses every task onto the range edges. When
        the percentile span is degenerate we map everything to the midpoint.
      * task duration is extremely long-tailed (p50 = 8s, max = 583,886s), so we
        scale it on a log axis to keep the bulk of the mass spread out.
    """
    x = np.asarray(x, dtype=float)
    if log:
        x = np.log1p(np.maximum(x, 0.0))
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    span = p99 - p1
    if span <= 1e-9:                       # degenerate: (near-)constant column
        return np.full(x.shape, 0.5 * (lo + hi))
    return np.clip(lo + (x - p1) / span * (hi - lo), lo, hi)


def from_alibaba(in_path: str, out_path: str = "trace_alibaba.csv",
                 bucket_seconds: float = 10.0, max_steps: int | None = 5000,
                 target_rate: float = 1.4, seed: int = 0) -> str:
    """Convert Alibaba v2018 `batch_task.csv` -> normalised trace for our env.

    Raw schema (no header; verified against the official schema.txt):
        0 task_name, 1 instance_num, 2 job_name, 3 task_type, 4 status,
        5 start_time, 6 end_time, 7 plan_cpu (100 = 1 core), 8 plan_mem (0-100)

    Two calibrations matter and are easy to get wrong:

    * **Cluster-scale subsampling.** The trace comes from a ~4,000-machine cluster
      (~186 tasks per 10s bucket). Our environment has a handful of servers, so we
      keep a random `target_rate / observed_rate` fraction of tasks. Without this
      the env is swamped instantly, every task misses its deadline, and the results
      are meaningless. `target_rate` is the desired MEAN arrivals per step.
    * **Robust normalisation** (see `_robust_scale`): duration is log-scaled
      because it spans five orders of magnitude; near-constant columns collapse to
      the midpoint instead of the range edges.

    What is preserved from the real data: arrival *timing* (bursts, diurnal shape)
    and the task-size/duration *distribution* — which is exactly the realism the
    synthetic Poisson process lacks.
    """
    rng = np.random.default_rng(seed)
    I_START, I_END, I_CPU, I_MEM = 5, 6, 7, 8

    # pandas' C parser handles the 800MB / 14M-row file in seconds; the stdlib csv
    # reader would take minutes and hold millions of Python objects in memory.
    try:
        import pandas as pd
        df = pd.read_csv(in_path, header=None, usecols=[I_START, I_END, I_CPU, I_MEM],
                         names=["start", "end", "cpu", "mem"], engine="c")
        df = df.dropna()
        df = df[df["end"] >= df["start"]]
        starts = df["start"].to_numpy(float); ends = df["end"].to_numpy(float)
        cpus = df["cpu"].to_numpy(float);     mems = df["mem"].to_numpy(float)
    except ImportError:                       # stdlib fallback (slow but works)
        cols = [[], [], [], []]
        with open(in_path, newline="") as f:
            for row in csv.reader(f):
                try:
                    st, en = float(row[I_START]), float(row[I_END])
                    if en < st:
                        continue
                    cols[0].append(st); cols[1].append(en)
                    cols[2].append(float(row[I_CPU])); cols[3].append(float(row[I_MEM]))
                except (ValueError, IndexError):
                    continue
        starts, ends, cpus, mems = (np.asarray(c, float) for c in cols)

    if starts.size == 0:
        raise ValueError(f"no usable rows parsed from {in_path} "
                         f"(check column indices I_START/I_END/I_CPU/I_MEM)")

    steps = ((starts - starts.min()) / bucket_seconds).astype(np.int64)

    # --- pick the DENSEST window of `max_steps` buckets --------------------- #
    # Do NOT just take the first max_steps buckets: a handful of tasks report
    # start_time ~= 0 while the bulk of the workload begins hours later, so the
    # leading window is almost empty (611 tasks instead of ~10^6) and the
    # environment would sit idle for the whole episode.
    if max_steps is not None and int(steps.max()) >= max_steps:
        counts = np.bincount(steps, minlength=int(steps.max()) + 1)
        csum = np.concatenate([[0], np.cumsum(counts)])
        win_sums = csum[max_steps:] - csum[:-max_steps]
        w0 = int(np.argmax(win_sums))
        keep = (steps >= w0) & (steps < w0 + max_steps)
        steps = steps[keep] - w0
        starts, ends, cpus, mems = (a[keep] for a in (starts, ends, cpus, mems))
        print(f"[alibaba] densest {max_steps:,}-bucket window starts at bucket "
              f"{w0:,} ({win_sums[w0]:,} tasks)")
        if steps.size == 0:
            raise ValueError("no tasks fall inside the selected window")

    # --- subsample to our cluster scale ------------------------------------- #
    n_buckets = int(steps.max()) + 1
    observed_rate = steps.size / max(1, n_buckets)
    frac = min(1.0, float(target_rate) / max(observed_rate, 1e-9))
    if frac < 1.0:
        sel = rng.random(steps.size) < frac
        steps, starts, ends, cpus, mems = (a[sel] for a in
                                           (steps, starts, ends, cpus, mems))
    print(f"[alibaba] {observed_rate:.1f} tasks/bucket in the raw trace -> keeping "
          f"{frac*100:.2f}% to hit ~{target_rate} tasks/step "
          f"({steps.size:,} tasks over {n_buckets:,} steps)")

    # --- normalise into the env's units ------------------------------------- #
    cpu_n = _robust_scale(cpus, *CPU_RANGE)
    mem_n = _robust_scale(mems, *MEM_RANGE)
    work_n = _robust_scale(ends - starts, *WORK_RANGE, log=True)   # long-tailed
    slack = rng.integers(SLACK_RANGE[0], SLACK_RANGE[1] + 1, size=steps.size)

    order = np.argsort(steps, kind="stable")
    rows = [(int(steps[i]), round(float(cpu_n[i]), 4), round(float(mem_n[i]), 4),
             round(float(work_n[i]), 4), int(slack[i])) for i in order]
    _write(out_path, rows)
    print(f"[alibaba] wrote {len(rows):,} tasks over "
          f"{rows[-1][0]+1 if rows else 0:,} steps -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a workload-trace CSV.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("synthetic", help="generate a synthetic sample for testing")
    p_s.add_argument("--out", default="trace_sample.csv")
    p_s.add_argument("--steps", type=int, default=600)
    p_s.add_argument("--rate", type=float, default=1.4)
    p_s.add_argument("--seed", type=int, default=0)

    p_a = sub.add_parser("alibaba", help="convert a raw Alibaba batch_task.csv")
    p_a.add_argument("in_path")
    p_a.add_argument("--out", default="trace_alibaba.csv")
    p_a.add_argument("--bucket", type=float, default=10.0)
    p_a.add_argument("--max-steps", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "synthetic":
        synthetic(args.out, args.steps, args.rate, args.seed)
    else:
        from_alibaba(args.in_path, args.out, args.bucket, args.max_steps)
