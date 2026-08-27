"""
train.py — PLTR-MA experiment orchestration (build step 6; Algorithm 1).

Ties the five mechanisms together and produces every result the paper needs:

  (A) Convergence ablation   : vanilla MAPPO / +Lyapunov / +CPO / full PLTR-MA,
                               each over >= n_seeds seeds, evaluated on a common
                               footing (uniform reference omega) every eval_every
                               iterations -> mean +/- std learning curves.
  (B) Pareto front           : sweep omega over the simplex for the trained full
                               PLTR-MA model -> non-dominated trade-off points.
  (C) V / delta ablation      : final performance vs the Lyapunov weight V and the
                               trust-region radius delta.
  (D) Queue stability        : mean backlog + virtual-queue trace (property P1).

All results are saved to results/ as a single .npz (plus a small JSON of the
config) so plot.py (step 7) can render figures without re-training.

SCALING: everything is driven by ExperimentConfig below. To go from N=3 to
N=50 servers, change `n_servers`; nothing else needs editing. Heavier settings
(more seeds / iterations / eval episodes) are where the Kaggle/GPU-or-many-core
run pays off; the defaults here are a fast smoke configuration.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from cloud_env import CloudAllocationEnv
from mappo import MAPPO, MAPPOConfig, scalarize
from lyapunov import LyapunovTracker
from trust_region import TrustRegionMAPPO
from lagrangian_ppo import LagrangianMAPPO
from trace import WorkloadTrace
from pareto import (ParetoMAPPO, pareto_mask, simplex_grid,
                    simplex_corners_center, hypervolume)


# --------------------------------------------------------------------------- #
# Experiment configuration  (the ONE place to edit for scaling / heavier runs)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    # environment / scaling
    n_servers: int = 3
    arrival_rate: float = 1.4       # rho~1.12: cluster runs hot so constraints bind
    horizon: int = 150

    # training budget
    n_iterations: int = 300
    n_seeds: int = 3
    ablation_seeds: int = 3        # seeds averaged in the V / delta ablations
    eval_every: int = 20
    eval_episodes: int = 6

    # PLTR-MA hyperparameters
    V: float = 0.5             # Lyapunov reward/constraint trade-off
    delta: float = 0.02        # trust-region KL radius
    alpha: float = 1.0         # Dirichlet concentration for omega sampling

    # ablation grids
    V_grid: tuple = (0.1, 0.3, 0.5, 1.0, 3.0)
    delta_grid: tuple = (0.002, 0.01, 0.02, 0.05)
    pareto_grid: int = 8       # simplex resolution for the front sweep
    pareto_seeds: int = 3      # seeds for the PLTR-MA front (hypervolume mean +/- std)
    baseline_seeds: int = 3    # seeds per fixed-weight scalarized-MAPPO policy
    hv_mc: int = 200_000       # Monte-Carlo samples for hypervolume
    baseline_grid: int = 0     # >0 => densify the fixed-weight MORL front to a
                               # simplex_grid(M, baseline_grid); 0 => corners+centre

    # --- P3: environment axis --------------------------------------------- #
    # None => synthetic Poisson arrivals (the conference env). A path to a
    # normalised trace CSV (see make_trace.py) => real-workload replay, which is
    # the second, Q1-credible environment.
    trace_path: str | None = None

    # --- P3: PPO-Lagrangian baseline -------------------------------------- #
    cost_idx: int = 0          # which constraint to bound (0 = capacity c1)
    cost_limit: float = 0.0    # target E[c_k] <= cost_limit
    lam_lr: float = 0.2        # dual-ascent step on the Lagrange multiplier
    lam_init: float = 2.0      # initial multiplier

    # --- parallelism ------------------------------------------------------- #
    # Runs are embarrassingly parallel (independent variant/seed/weight) and the
    # bottleneck is single-threaded Python env-stepping, so PROCESS parallelism is
    # what buys speed. 0/1 => sequential. On Kaggle (~4 cores, 12h cap) use 4.
    n_workers: int = 1

    out_dir: str = "results"
    device: str = "cpu"

    def mappo_cfg(self, seed: int) -> MAPPOConfig:
        return MAPPOConfig(
            n_servers=self.n_servers, arrival_rate=self.arrival_rate,
            horizon=self.horizon, steps_per_update=self.horizon,
            n_iterations=self.n_iterations, seed=seed, device=self.device,
        )


VARIANTS = ("mappo", "lyapunov", "cpo", "lagrangian", "pltr_ma")


def make_env(xc: ExperimentConfig, seed: int) -> CloudAllocationEnv:
    """Fresh environment for ONE algo instance.

    Trace-driven when `xc.trace_path` is set (real-workload replay, P3's second
    environment); otherwise the synthetic Poisson arrival process.
    Each algo needs its own env instance, so this is called per variant/seed.
    """
    trace = WorkloadTrace(xc.trace_path, horizon=xc.horizon) if xc.trace_path else None
    return CloudAllocationEnv(
        n_servers=xc.n_servers, arrival_rate=xc.arrival_rate,
        horizon=xc.horizon, seed=seed, trace=trace,
    )


# --------------------------------------------------------------------------- #
# Variant factory
# --------------------------------------------------------------------------- #
def build_variant(name: str, xc: ExperimentConfig, seed: int,
                  V: float | None = None, delta: float | None = None):
    """Return (algo, tracker_or_None) for a named ablation variant."""
    cfg = xc.mappo_cfg(seed)
    V = xc.V if V is None else V
    delta = xc.delta if delta is None else delta
    env = make_env(xc, seed)          # honours xc.trace_path (environment axis)

    if name == "mappo":
        return MAPPO(cfg, env=env), None
    if name == "lyapunov":
        lyap = LyapunovTracker(V=V)
        return MAPPO(cfg, env=env, reward_fn=lyap.make_reward_fn()), lyap
    if name == "cpo":
        return TrustRegionMAPPO(cfg, delta=delta, env=env), None
    if name == "lagrangian":
        # PPO-Lagrangian safe-RL baseline (learned dual on the constraint).
        return LagrangianMAPPO(cfg, env=env, cost_idx=xc.cost_idx,
                               cost_limit=xc.cost_limit, lam_lr=xc.lam_lr,
                               lam_init=xc.lam_init), None
    if name == "pltr_ma":
        lyap = LyapunovTracker(V=V)
        return ParetoMAPPO(cfg, delta=delta, alpha=xc.alpha, env=env,
                           reward_fn=lyap.make_reward_fn()), lyap
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Common-footing evaluation (uniform reference omega)
# --------------------------------------------------------------------------- #
UNIFORM = None  # filled per-env


def eval_policy(algo, omega, n_ep: int, seed_base: int = 90000):
    """Mean scalar return, per-objective return, cost vector and backlog."""
    M, K = algo.env.M, algo.env.K
    sca, vec, cost, backlog = [], [], [], []
    for e in range(n_ep):
        obs, _ = algo.env.reset(seed=seed_base + e)
        rv = np.zeros(M); cs = np.zeros(K); bl = []
        done = False
        while not done:
            a, _ = algo.act(obs, cond=omega)
            obs, r, term, trunc, info = algo.env.step(a)
            rv += r; cs += info["cost"]; bl.append(info["total_backlog"])
            done = term or trunc
        sca.append(scalarize(rv, omega)); vec.append(rv)
        cost.append(cs / max(1, len(bl))); backlog.append(np.mean(bl))
    return (float(np.mean(sca)), np.mean(vec, 0),
            np.mean(cost, 0), float(np.mean(backlog)))


def fit(algo, xc: ExperimentConfig, omega_sampler):
    """Unified training loop with periodic common-footing evaluation."""
    uni = np.full(algo.env.M, 1.0 / algo.env.M, dtype=np.float32)
    algo._obs, _ = algo.env.reset()
    curve = {"iter": [], "eval_scalar": [], "eval_cost": [],
             "eval_backlog": [], "eval_vec": []}
    for it in range(xc.n_iterations):
        omega = omega_sampler()
        roll = algo.collect_rollout(xc.horizon, omega=omega)
        algo.update(roll)
        if it % xc.eval_every == 0 or it == xc.n_iterations - 1:
            s, v, c, b = eval_policy(algo, uni, xc.eval_episodes)
            curve["iter"].append(it)
            curve["eval_scalar"].append(s)
            curve["eval_vec"].append(v)
            curve["eval_cost"].append(c)
            curve["eval_backlog"].append(b)
    return curve


# --------------------------------------------------------------------------- #
# Parallel execution helper
# --------------------------------------------------------------------------- #
def _worker_init():
    """Pin each worker to one torch thread so N processes don't oversubscribe."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _pmap(fn, items, n_workers: int):
    """map(fn, items), in parallel when n_workers > 1. `fn` must be module-level.

    Start method matters: a *forked* child cannot re-initialize CUDA, so when the
    parent has CUDA available we must use "spawn". Forking is faster to start, so
    we keep it for the CPU case (where it is also perfectly safe).
    """
    if n_workers and n_workers > 1 and len(items) > 1:
        import multiprocessing as mp
        import torch
        if torch.cuda.is_available():
            method = "spawn"        # required: fork + CUDA => RuntimeError
        else:
            method = "fork" if hasattr(os, "fork") else "spawn"
        ctx = mp.get_context(method)
        with ctx.Pool(processes=min(n_workers, len(items)),
                      initializer=_worker_init) as pool:
            return pool.map(fn, items)
    return [fn(x) for x in items]


# --------------------------------------------------------------------------- #
# Experiment blocks
# --------------------------------------------------------------------------- #
def _conv_job(args):
    """One (variant, seed) convergence run. Module-level so it is picklable."""
    name, seed, xc = args
    algo, _ = build_variant(name, xc, seed)
    M = algo.env.M
    sampler = ((lambda a=algo: a.sample_omega()) if name == "pltr_ma"
               else (lambda M=M: np.full(M, 1.0 / M, np.float32)))
    t0 = time.time()
    curve = fit(algo, xc, sampler)
    print(f"  [{name:10s}] seed {seed} done ({time.time()-t0:5.1f}s) "
          f"final eval={curve['eval_scalar'][-1]:.2f}", flush=True)
    return name, seed, curve


def run_convergence(xc: ExperimentConfig) -> dict:
    """(A) all variants x seeds, common-footing learning curves."""
    jobs = [(name, seed, xc) for name in VARIANTS for seed in range(xc.n_seeds)]
    results = _pmap(_conv_job, jobs, xc.n_workers)

    by_variant: dict[str, list] = {n: [] for n in VARIANTS}
    for name, _seed, curve in sorted(results, key=lambda r: (r[0], r[1])):
        by_variant[name].append(curve)

    out = {}
    for name, per_seed in by_variant.items():
        iters = np.array(per_seed[0]["iter"])
        scal = np.array([p["eval_scalar"] for p in per_seed])      # (seeds, T)
        cost = np.array([p["eval_cost"] for p in per_seed])        # (seeds, T, K)
        back = np.array([p["eval_backlog"] for p in per_seed])
        out[name] = {
            "iter": iters,
            "scalar_mean": scal.mean(0), "scalar_std": scal.std(0),
            "cost_mean": cost.mean(0), "cost_std": cost.std(0),    # (T, K)
            "backlog_mean": back.mean(0), "backlog_std": back.std(0),
            "n_seeds": np.array(xc.n_seeds),
        }
    return out


def _train_fixed_weight_baseline(xc: ExperimentConfig, omega: np.ndarray,
                                 seed: int) -> np.ndarray:
    """Train ONE vanilla scalarized MAPPO at a fixed preference `omega` and
    return its mean per-objective return vector. This is the obvious
    alternative to weight-conditioning: one network per weight, retrained."""
    algo, _ = build_variant("mappo", xc, seed=seed)
    fit(algo, xc, lambda w=omega: w.astype(np.float32))
    _, vec, _, _ = eval_policy(algo, omega, xc.eval_episodes)
    return vec


def _front_job(args):
    """One PLTR-MA front (train once, then sweep omega). Module-level = picklable."""
    xc, seed = args
    algo, _ = build_variant("pltr_ma", xc, seed=seed)
    algo.train(verbose=False)
    _, returns, _ = algo.sweep_front(n_grid=xc.pareto_grid, n_ep=xc.eval_episodes)
    print(f"  [pltr_ma front] seed {seed} done ({len(returns)} pts)", flush=True)
    return returns


def _base_job(args):
    """One fixed-weight baseline policy (train at a fixed omega, then evaluate)."""
    xc, omega, seed = args
    vec = _train_fixed_weight_baseline(xc, omega, seed)
    print(f"  [baseline] seed {seed} w={np.round(omega, 2)} done", flush=True)
    return vec


def run_pareto(xc: ExperimentConfig) -> dict:
    """(B) PLTR-MA omega-sweep front vs. a fixed-weight scalarized-MAPPO front.

    Offense, not a standalone scatter: ONE weight-conditioned PLTR-MA network
    (swept over the simplex) is compared, in the same objective space and via
    hypervolume, against k separately-trained fixed-weight MAPPO policies. The
    claim is comparable-or-broader coverage at ~1/k the training cost. Both
    fronts and the HV are reported as mean +/- std over seeds so the comparison
    is a real gap, not noise.
    """
    # --- PLTR-MA front, one per seed (single network, swept) ----------------
    pareto_seeds = max(1, xc.pareto_seeds)
    fronts = np.array(_pmap(_front_job, [(xc, s) for s in range(pareto_seeds)],
                            xc.n_workers))          # (seeds, P, M)

    # --- fixed-weight scalarized-MAPPO baseline front -----------------------
    # Fixed-weight multi-policy front = the MORL baseline (linear scalarization,
    # one network retrained per preference). `baseline_grid > 0` densifies it
    # beyond corners+centre for a stronger P3 comparison.
    M_obj = fronts.shape[-1]
    base_omegas = (simplex_grid(M_obj, xc.baseline_grid) if xc.baseline_grid > 0
                   else simplex_corners_center(M_obj))            # (k, M)
    base_seeds = max(1, xc.baseline_seeds)
    base_jobs = [(xc, w, s) for s in range(base_seeds) for w in base_omegas]
    base_vecs = _pmap(_base_job, base_jobs, xc.n_workers)
    base = np.array(base_vecs).reshape(base_seeds, len(base_omegas), M_obj)

    # --- shared normalization box for a fair hypervolume comparison ---------
    allpts = np.concatenate([fronts.reshape(-1, fronts.shape[-1]),
                             base.reshape(-1, base.shape[-1])], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)

    hv_pltr = np.array([hypervolume(f, lo, hi, n_mc=xc.hv_mc) for f in fronts])
    hv_base = np.array([hypervolume(b, lo, hi, n_mc=xc.hv_mc) for b in base])

    # seed 0 front + its mask for the scatter; means for the baseline markers
    omegas0 = simplex_grid(fronts.shape[-1], xc.pareto_grid)
    returns0 = fronts[0]
    return {
        "omegas": omegas0, "returns": returns0, "mask": pareto_mask(returns0),
        "returns_all": fronts,                       # (seeds, P, M)
        "baseline_omegas": base_omegas,              # (k, M)
        "baseline_returns": base.mean(0),            # (k, M)
        "baseline_returns_all": base,                # (seeds, k, M)
        "hv_pltr_mean": np.array(hv_pltr.mean()), "hv_pltr_std": np.array(hv_pltr.std()),
        "hv_base_mean": np.array(hv_base.mean()), "hv_base_std": np.array(hv_base.std()),
        "hv_ref_lo": lo, "hv_ref_hi": hi,
    }


def _train_uniform(algo, xc):
    """Train a variant at the fixed uniform reference omega (used by ablations)."""
    uni = np.full(algo.env.M, 1.0 / algo.env.M, np.float32)
    algo._obs, _ = algo.env.reset()
    for _ in range(xc.n_iterations):
        algo.update(algo.collect_rollout(xc.horizon, omega=uni))
    return uni


def run_ablation(xc: ExperimentConfig) -> dict:
    """(C) final performance vs V (Lyapunov) and delta (trust region).

    Averaged over `ablation_seeds` seeds — single-seed ablations are dominated by
    run-to-run variance, which buries the V / delta trend (see experiments/test2).
    Stores the full constraint-cost vector so the plot can pick whichever
    constraint is actually binding.
    """
    n_seeds = max(1, xc.ablation_seeds)

    V_res, V_sd = [], []
    for V in xc.V_grid:
        rows = []
        for seed in range(n_seeds):
            algo, lyap = build_variant("lyapunov", xc, seed=seed, V=V)
            uni = _train_uniform(algo, xc)
            s, v, c, b = eval_policy(algo, uni, xc.eval_episodes)
            zmax = lyap.stability_report()["Z_max"].max()
            rows.append((s, c[0], c[1], c[2], b, float(zmax)))
        m, sd = np.mean(rows, axis=0), np.std(rows, axis=0)
        V_res.append((V, *m))          # (V, scalar, c1, c2, c3, backlog, zmax)
        V_sd.append((V, *sd))          # std in the same column layout (col 0 = V, std 0)

    d_res, d_sd = [], []
    for d in xc.delta_grid:
        rows = []
        for seed in range(n_seeds):
            algo, _ = build_variant("cpo", xc, seed=seed, delta=d)
            uni = _train_uniform(algo, xc)
            s, v, c, b = eval_policy(algo, uni, xc.eval_episodes)
            rows.append((s, algo.kl_report()["kl_max"]))
        m, sd = np.mean(rows, axis=0), np.std(rows, axis=0)
        d_res.append((d, *m))          # (delta, scalar, kl_max)
        d_sd.append((d, *sd))

    return {"V": np.array(V_res), "V_std": np.array(V_sd),
            "delta": np.array(d_res), "delta_std": np.array(d_sd),
            "n_seeds": np.array(n_seeds)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(xc: ExperimentConfig | None = None,
         blocks=("convergence", "pareto", "ablation")) -> dict:
    xc = xc or ExperimentConfig()
    os.makedirs(xc.out_dir, exist_ok=True)
    print(f"PLTR-MA experiments | N={xc.n_servers} seeds={xc.n_seeds} "
          f"iters={xc.n_iterations} | blocks={blocks}")

    results: dict = {}
    if "convergence" in blocks:
        print("[A] convergence ablation ...")
        results["convergence"] = run_convergence(xc)
    if "pareto" in blocks:
        print("[B] Pareto-front sweep ...")
        results["pareto"] = run_pareto(xc)
    if "ablation" in blocks:
        print("[C] V / delta ablation ...")
        results["ablation"] = run_ablation(xc)

    # flatten to a savez-friendly dict
    flat = {}
    for blk, d in results.items():
        _flatten(blk, d, flat)
    np.savez(os.path.join(xc.out_dir, "results.npz"), **flat)
    with open(os.path.join(xc.out_dir, "config.json"), "w") as f:
        json.dump(asdict(xc), f, indent=2, default=str)
    print(f"saved -> {os.path.join(xc.out_dir, 'results.npz')}")
    return results


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}", v, out)
    else:
        out[prefix] = np.asarray(obj)


if __name__ == "__main__":
    main()
