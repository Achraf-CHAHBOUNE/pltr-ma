"""run_experiment.py — container entrypoint for the PLTR-MA study.

Fully config-driven via environment variables so the SAME image runs a fast CPU
smoke test locally and the full publication run on the GPU cluster with no code
change. Device is auto-detected (CUDA if visible, else CPU) unless DEVICE is set.

Typical uses
------------
Local CPU smoke test:
    N_ITERATIONS=8 N_SEEDS=2 PARETO_SEEDS=2 BASELINE_SEEDS=2 \
    ABLATION_SEEDS=2 EVAL_EPISODES=2 PARETO_GRID=4 HV_MC=20000 \
    OUT_DIR=results python run_experiment.py

Full run on the GPU cluster (defaults already publication-grade):
    N_SERVERS=10 N_SEEDS=10 python run_experiment.py
"""
from __future__ import annotations

import os
import time

import torch

from train import ExperimentConfig, main as run_train
import plot


def _get(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _tuple(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return tuple(float(x) for x in v.replace(" ", "").split(","))


def build_config() -> ExperimentConfig:
    # DEVICE: explicit override, else auto-detect
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    return ExperimentConfig(
        n_servers=_get("N_SERVERS", 3, int),
        arrival_rate=_get("ARRIVAL_RATE", 1.4, float),
        horizon=_get("HORIZON", 150, int),
        n_iterations=_get("N_ITERATIONS", 400, int),
        n_seeds=_get("N_SEEDS", 5, int),
        ablation_seeds=_get("ABLATION_SEEDS", 3, int),
        eval_every=_get("EVAL_EVERY", 20, int),
        eval_episodes=_get("EVAL_EPISODES", 6, int),
        V=_get("V", 0.5, float),
        delta=_get("DELTA", 0.02, float),
        alpha=_get("ALPHA", 1.0, float),
        V_grid=_tuple("V_GRID", (0.1, 0.3, 0.5, 1.0, 3.0)),
        delta_grid=_tuple("DELTA_GRID", (0.002, 0.01, 0.02, 0.05)),
        pareto_grid=_get("PARETO_GRID", 8, int),
        pareto_seeds=_get("PARETO_SEEDS", 5, int),
        baseline_seeds=_get("BASELINE_SEEDS", 5, int),
        baseline_grid=_get("BASELINE_GRID", 0, int),
        hv_mc=_get("HV_MC", 200_000, int),
        n_workers=_get("N_WORKERS", 1, int),
        # environment axis: unset => synthetic Poisson env; set => real-trace replay
        trace_path=os.environ.get("TRACE_PATH") or None,
        # PPO-Lagrangian baseline
        cost_idx=_get("COST_IDX", 0, int),
        cost_limit=_get("COST_LIMIT", 0.0, float),
        lam_lr=_get("LAM_LR", 0.2, float),
        lam_init=_get("LAM_INIT", 2.0, float),
        out_dir=os.environ.get("OUT_DIR", "/results"),
        device=device,
    )


def main():
    xc = build_config()
    blocks = tuple(os.environ.get("BLOCKS", "convergence,pareto,ablation")
                   .replace(" ", "").split(","))

    print("=" * 64)
    print("PLTR-MA experiment runner")
    print(f"  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  device      = {xc.device}")
    print(f"  N servers   = {xc.n_servers}  | seeds = {xc.n_seeds} "
          f"(pareto {xc.pareto_seeds}, baseline {xc.baseline_seeds}, ablation {xc.ablation_seeds})")
    print(f"  iterations  = {xc.n_iterations} | blocks = {blocks}")
    print(f"  environment = {'TRACE: ' + xc.trace_path if xc.trace_path else 'synthetic (Poisson)'}")
    print(f"  out_dir     = {xc.out_dir}")
    print("=" * 64, flush=True)

    t0 = time.time()
    run_train(xc, blocks=blocks)
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)

    try:
        plot.main(xc.out_dir)
        print("[plot] figures written", flush=True)
    except Exception as e:  # plotting must never lose a completed run
        print(f"[plot] WARNING: figure rendering failed ({e}); "
              f"results.npz is safe in {xc.out_dir}", flush=True)


if __name__ == "__main__":
    main()
