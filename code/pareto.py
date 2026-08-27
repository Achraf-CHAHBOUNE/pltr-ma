"""
pareto.py — Weight-conditioned actors for Pareto-front construction (build step 5)
            and the full PLTR-MA agent.

Implements Section IV.C. Both actor and critic take the preference vector
omega in Delta^M as an explicit input (the `cond_dim=M` path already built into
mappo.py). At the start of each training episode omega is sampled from a
Dirichlet distribution Dir(alpha); the augmented reward omega^T r then drives
that episode's update. A single network thereby represents a continuous family
of policies — one per preference. At inference we sweep omega over the simplex
and read off the (approximate) Pareto front.

`ParetoMAPPO` subclasses `TrustRegionMAPPO`, so it is simultaneously the full
PLTR-MA stack:

    full PLTR-MA = Pareto (omega-conditioned)         [this file]
                 + trust region (delta KL bound)       [trust_region.py]
                 + Lyapunov drift-plus-penalty reward   [lyapunov.py]

The four ablation configurations used in the paper are therefore:
    vanilla MAPPO  : MAPPO(cfg)
    +Lyapunov      : MAPPO(cfg, reward_fn=lyap.make_reward_fn())
    +CPO           : TrustRegionMAPPO(cfg, delta)
    full PLTR-MA   : ParetoMAPPO(cfg, delta, reward_fn=lyap.make_reward_fn())
"""

from __future__ import annotations

import itertools

import numpy as np

from mappo import MAPPOConfig
from trust_region import TrustRegionMAPPO


def simplex_grid(M: int, n: int) -> np.ndarray:
    """All points (k_1,...,k_M)/n with sum k = n — a uniform grid on Delta^M."""
    pts = []
    for combo in itertools.combinations_with_replacement(range(M), n):
        w = np.zeros(M)
        for c in combo:
            w[c] += 1
        pts.append(w / n)
    return np.unique(np.array(pts), axis=0)


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (objectives are MAXIMIZED)."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j >= i on all objectives and > on at least one
        dominated = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def simplex_corners_center(M: int) -> np.ndarray:
    """The M one-hot corners of Delta^M plus the uniform centre — the fixed
    preference weights at which the scalarized-MAPPO baseline front is trained."""
    pts = list(np.eye(M, dtype=np.float32))
    pts.append(np.full(M, 1.0 / M, np.float32))
    return np.array(pts, dtype=np.float32)


def hypervolume(points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                n_mc: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume of a MAXIMIZED point set, normalized into the
    unit cube by the shared (lo, hi) box so two fronts are directly comparable.

    Objectives are scaled (p - lo) / (hi - lo) and clipped to [0, 1]; the
    reference point is the origin. HV is the fraction of the unit cube that is
    dominated by at least one (non-dominated) point — i.e. the dominated volume
    in [0, 1]^M. Deterministic given `seed`. lo/hi MUST be identical across the
    methods being compared (pass the nadir/ideal of their union)."""
    P = (np.asarray(points, float) - lo) / np.maximum(hi - lo, 1e-12)
    P = np.clip(P, 0.0, 1.0)
    P = P[pareto_mask(P)]                       # only non-dominated points contribute
    if len(P) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    S = rng.random((n_mc, P.shape[1]))
    dominated = np.zeros(n_mc, dtype=bool)
    for p in P:                                 # p dominates s iff p >= s in every objective
        dominated |= np.all(p >= S, axis=1)
    return float(dominated.mean())


class ParetoMAPPO(TrustRegionMAPPO):
    """Full PLTR-MA: weight-conditioned actors/critic + trust region + Lyapunov."""

    def __init__(self, cfg: MAPPOConfig, delta: float = 0.02, alpha: float = 1.0,
                 reward_fn=None, **kwargs):
        # force omega-conditioning (cond_dim = M); ignore any caller cond_dim
        kwargs.pop("cond_dim", None)
        super().__init__(cfg, delta=delta, cond_dim=cfg_M(cfg), reward_fn=reward_fn, **kwargs)
        self.alpha = alpha
        self.M = self.env.M

    def sample_omega(self) -> np.ndarray:
        return np.random.dirichlet(np.full(self.M, self.alpha)).astype(np.float32)

    # ------------------------------------------------------------------ #
    def train(self, n_iterations: int | None = None, verbose: bool = True) -> dict:
        """Like MAPPO.train but draws a fresh omega ~ Dir(alpha) each episode."""
        n_iterations = n_iterations or self.cfg.n_iterations
        self._obs, _ = self.env.reset()
        history = {"iter": [], "omega": [], "scalar_return": [], "ret_vec": []}

        for it in range(n_iterations):
            omega = self.sample_omega()
            roll = self.collect_rollout(self.cfg.steps_per_update, omega=omega)
            self.update(roll)
            self.kl_history.append(float(self.last_kl.max()))

            vec = (np.mean(roll["ep_returns_vec"], axis=0)
                   if roll["ep_returns_vec"] else np.full(self.M, np.nan))
            sca = float(np.dot(omega, vec))
            history["iter"].append(it)
            history["omega"].append(omega)
            history["scalar_return"].append(sca)
            history["ret_vec"].append(vec)

            if verbose and (it % max(1, n_iterations // 10) == 0 or it == n_iterations - 1):
                print(f"iter {it:4d} | omega {np.round(omega,2)} | "
                      f"scalar {sca:8.2f} | vec {np.round(vec,1)}")
        return history

    # ------------------------------------------------------------------ #
    def evaluate_omega(self, omega: np.ndarray, n_ep: int = 5) -> np.ndarray:
        """Mean per-objective vector return when acting under preference omega."""
        omega = np.asarray(omega, np.float32)
        vecs = []
        for e in range(n_ep):
            obs, _ = self.env.reset(seed=30000 + e)
            acc = np.zeros(self.M)
            done = False
            while not done:
                a, _ = self.act(obs, cond=omega)
                obs, rvec, term, trunc, _ = self.env.step(a)
                acc += rvec
                done = term or trunc
            vecs.append(acc)
        return np.mean(vecs, axis=0)

    def sweep_front(self, n_grid: int = 8, n_ep: int = 5):
        """Sweep omega over the simplex; return (omegas, returns, pareto_mask)."""
        omegas = simplex_grid(self.M, n_grid)
        returns = np.array([self.evaluate_omega(w, n_ep) for w in omegas])
        return omegas, returns, pareto_mask(returns)


def cfg_M(cfg) -> int:
    """M is fixed by the env (=3); exposed as a helper so __init__ stays one line."""
    from cloud_env import CloudAllocationEnv
    return CloudAllocationEnv.M


__all__ = ["ParetoMAPPO", "simplex_grid", "pareto_mask",
           "simplex_corners_center", "hypervolume"]
