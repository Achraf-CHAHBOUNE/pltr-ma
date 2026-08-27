"""figures_journal.py — the new figures P3 needs beyond the per-run plots.

Produces three journal figures from the saved result bundles:

  fig_scaling.png   grouped bars of PLTR-MA vs fixed-weight hypervolume at N=3 and
                    N=10 (real trace), annotated with the ratio and significance.
                    This is the paper's strongest visual: the advantage GROWS with
                    scale (1.30x -> 1.57x).
  fig_two_env.png   PLTR-MA vs fixed-weight hypervolume on the synthetic and the
                    real-trace environment (both N=3) — shows the win transfers
                    from the toy simulator to a real workload.
  fig_hv_scatter.png per-seed hypervolume (PLTR-MA vs fixed-weight) for each setting;
                    every point above the diagonal is a seed PLTR-MA wins. Makes the
                    9/10, 10/10, 5/5 sweeps visible at a glance.

Usage:
  python figures_journal.py SYN=<dir> TRACE=<dir> TRACE_N10=<dir> [OUT=<dir>]
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats import load_results, hypervolume_report, hv_per_seed, _stars

PLTR_C, BASE_C = "tab:blue", "tab:gray"


def _bar(ax, groups, reps, title):
    """Grouped PLTR vs baseline HV bars, one group per (label, report)."""
    x = np.arange(len(groups))
    w = 0.38
    p = [r["pltr_mean"] for r in reps]
    b = [r["base_mean"] for r in reps]
    pe = [r["pltr_std"] for r in reps]
    be = [r["base_std"] for r in reps]
    ax.bar(x - w / 2, p, w, yerr=pe, capsize=4, color=PLTR_C, label="PLTR-MA (1 network)")
    ax.bar(x + w / 2, b, w, yerr=be, capsize=4, color=BASE_C,
           label=f"{reps[0]['k_baseline']}x fixed-weight (retrained)")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Hypervolume (dominated volume)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    top = max(max(np.array(p) + pe), max(np.array(b) + be))
    # annotate each group just above its OWN taller bar (not the global top)
    for i, r in enumerate(reps):
        y = max(p[i] + pe[i], b[i] + be[i])
        ax.annotate("", xy=(i - w / 2, y + top * 0.02), xytext=(i + w / 2, y + top * 0.02),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
        ax.text(i, y + top * 0.05, f"{r['ratio']:.2f}x {_stars(r['p_value'])}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, top * 1.28)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)


def fig_scaling(trace, trace_n10, out):
    reps = [hypervolume_report(trace), hypervolume_report(trace_n10)]
    labels = [f"N=3\n({reps[0]['n_seeds']} seeds)", f"N=10\n({reps[1]['n_seeds']} seeds)"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    _bar(ax, labels, reps, "Scaling on the real Alibaba trace")
    _save(fig, out, "fig_scaling.png")


def fig_two_env(syn, trace, out):
    reps = [hypervolume_report(syn), hypervolume_report(trace)]
    labels = ["Synthetic\n(Poisson)", "Real trace\n(Alibaba)"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    _bar(ax, labels, reps, "Hypervolume advantage transfers to a real workload (N=3)")
    _save(fig, out, "fig_two_env.png")


def fig_hv_scatter(settings, out):
    fig, ax = plt.subplots(figsize=(5, 5))
    markers = ["o", "s", "^", "D"]
    lim = 0.0
    for (name, R), mk in zip(settings, markers):
        hp, hb = hv_per_seed(R, "pltr"), hv_per_seed(R, "base")
        ax.scatter(hb, hp, marker=mk, s=45, alpha=0.8, label=name)
        lim = max(lim, hp.max(), hb.max())
    lim *= 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.6, lw=1)
    ax.text(lim * 0.62, lim * 0.52, "PLTR-MA wins\n(above diagonal)",
            fontsize=8, color="tab:blue", rotation=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("fixed-weight hypervolume (per seed)")
    ax.set_ylabel("PLTR-MA hypervolume (per seed)")
    ax.set_title("Per-seed hypervolume: every point above = a seed won")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    _save(fig, out, "fig_hv_scatter.png")


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv):
    args = dict(a.split("=", 1) for a in argv if "=" in a)
    syn = load_results(args["SYN"])
    trace = load_results(args["TRACE"])
    trace_n10 = load_results(args["TRACE_N10"])
    out = args.get("OUT", "journal_figs")
    print("Rendering journal figures ...")
    fig_scaling(trace, trace_n10, out)
    fig_two_env(syn, trace, out)
    fig_hv_scatter([("Synthetic N=3", syn), ("Real trace N=3", trace),
                    ("Real trace N=10", trace_n10)], out)
    print("done.")


if __name__ == "__main__":
    main(sys.argv[1:])
