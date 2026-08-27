"""stats.py — reproducible significance testing for the P3 hypervolume comparison.

Turns the raw per-seed Pareto fronts saved by train.py into the head-to-head
hypervolume statistics reported in the paper: per-seed hypervolume, a two-sided
Mann-Whitney U test (non-parametric, no normality assumption — appropriate for
n=5..10 seeds), Cohen's d effect size, and the win count. No scipy dependency;
the U statistic uses the standard tie-corrected normal approximation.

Run:  python stats.py <results_dir_or_npz> [more_dirs...]
e.g.  python stats.py results_synthetic results_trace results_trace_n10
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from pareto import hypervolume


# --------------------------------------------------------------------------- #
# Loading (accepts a results.npz OR a directory tree of .npy files)
# --------------------------------------------------------------------------- #
def load_results(path: str) -> dict:
    """Load a results bundle written by train.py, from either a `.npz` file or a
    directory of `.npy` files (the layout produced when the .npz is unpacked)."""
    if os.path.isfile(path) and path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    # try results.npz inside the directory first
    npz = os.path.join(path, "results.npz")
    if os.path.isfile(npz):
        z = np.load(npz, allow_pickle=True)
        return {k: z[k] for k in z.files}
    # else walk the .npy tree
    R = {}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".npy"):
                key = os.path.relpath(os.path.join(root, f), path)
                key = key.replace(os.sep, "/")[:-4]
                R[key] = np.load(os.path.join(root, f), allow_pickle=True)
    if not R:
        raise FileNotFoundError(f"no results.npz or .npy files found under {path}")
    return R


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mann_whitney_u(x, y) -> tuple[float, float, float]:
    """Two-sided Mann-Whitney U test via the tie-corrected normal approximation.
    Returns (U, z, p). Suitable for the small seed counts used here."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    tie_term = 0.0
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
            tie_term += c ** 3 - c
    U1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (U1 - mu) / sd if sd > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(U1), float(z), float(p)


def cohens_d(x, y) -> float:
    """Pooled-SD standardized mean difference (x - y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("inf")


def hv_per_seed(R: dict, which: str, n_mc: int = 200_000):
    """Per-seed hypervolume for `which` in {'pltr','base'}, on the SHARED
    normalization box stored with the run (so the two methods are comparable)."""
    key = "pareto/returns_all" if which == "pltr" else "pareto/baseline_returns_all"
    lo, hi = R["pareto/hv_ref_lo"], R["pareto/hv_ref_hi"]
    return np.array([hypervolume(f, lo, hi, n_mc=n_mc) for f in R[key]])


def hypervolume_report(R: dict, n_mc: int = 200_000) -> dict:
    """Full head-to-head: per-seed HV, ratio, Mann-Whitney p, Cohen's d, wins."""
    hv_p = hv_per_seed(R, "pltr", n_mc)
    hv_b = hv_per_seed(R, "base", n_mc)
    U, z, p = mann_whitney_u(hv_p, hv_b)
    return {
        "n_seeds": len(hv_p),
        "k_baseline": int(len(R["pareto/baseline_omegas"])),
        "hv_pltr": hv_p, "hv_base": hv_b,
        "pltr_mean": float(hv_p.mean()), "pltr_std": float(hv_p.std()),
        "base_mean": float(hv_b.mean()), "base_std": float(hv_b.std()),
        "ratio": float(hv_p.mean() / max(hv_b.mean(), 1e-12)),
        "U": U, "z": z, "p_value": p,
        "cohens_d": cohens_d(hv_p, hv_b),
        "wins": int((hv_p > hv_b).sum()),
    }


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def print_report(path: str, n_mc: int = 200_000) -> dict:
    R = load_results(path)
    rep = hypervolume_report(R, n_mc)
    print(f"\n=== {path} ===")
    print(f"  seeds={rep['n_seeds']}  baseline policies k={rep['k_baseline']}")
    print(f"  PLTR-MA      HV = {rep['pltr_mean']:.3f} +/- {rep['pltr_std']:.3f}")
    print(f"  {rep['k_baseline']}x fixed-wt  HV = {rep['base_mean']:.3f} +/- {rep['base_std']:.3f}")
    print(f"  ratio = {rep['ratio']:.2f}x   Mann-Whitney U={rep['U']:.0f} "
          f"p={rep['p_value']:.4f} {_stars(rep['p_value'])}")
    print(f"  Cohen's d = {rep['cohens_d']:.2f}   wins = {rep['wins']}/{rep['n_seeds']}")
    return rep


if __name__ == "__main__":
    paths = sys.argv[1:] or ["results"]
    for p in paths:
        print_report(p)
