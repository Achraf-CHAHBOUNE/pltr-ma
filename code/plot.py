"""
plot.py — Figures for the PLTR-MA paper (build step 7).

Reads results/results.npz (written by train.py) and renders, into results/:

  fig_convergence.png : eval scalar return vs iteration for the four ablation
                        variants (mean +/- std band over seeds) — paper Fig. 2(a).
  fig_pareto.png      : Pareto front recovered by sweeping omega at inference;
                        non-dominated points highlighted — paper Fig. 2(b).
  fig_ablation.png    : (left) SLA cost & return vs Lyapunov weight V;
                        (right) realized KL vs trust-region radius delta — Fig. 2(c).
  fig_stability.png   : mean queue backlog vs iteration per variant — empirical
                        Lyapunov stability (property P1).

Uses a headless Agg backend so it runs unchanged on Kaggle. No retraining here.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANT_LABELS = {
    "mappo": "MAPPO (vanilla)",
    "lyapunov": "MAPPO + Lyapunov",
    "cpo": "MAPPO + TR (trust region)",
    "lagrangian": "PPO-Lagrangian (safe-RL)",
    "pltr_ma": "PLTR-MA (full)",
}
VARIANT_ORDER = ("mappo", "lyapunov", "cpo", "lagrangian", "pltr_ma")


def load(out_dir: str = "results") -> dict:
    """Load a results bundle from either a `.npz` file, a directory containing
    one, or a directory tree of `.npy` files (the layout of the released
    `results/` folders). Mirrors stats.load_results so both entry points accept
    the same inputs."""
    if os.path.isfile(out_dir) and out_dir.endswith(".npz"):
        z = np.load(out_dir, allow_pickle=True)
        return {k: z[k] for k in z.files}

    npz = os.path.join(out_dir, "results.npz")
    if os.path.isfile(npz):
        z = np.load(npz, allow_pickle=True)
        return {k: z[k] for k in z.files}

    R = {}
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".npy"):
                key = os.path.relpath(os.path.join(root, f), out_dir)
                R[key.replace(os.sep, "/")[:-4]] = np.load(
                    os.path.join(root, f), allow_pickle=True)
    if not R:
        raise FileNotFoundError(
            f"no results.npz or .npy files found under {out_dir}")
    return R


# --------------------------------------------------------------------------- #
def plot_convergence(R, out_dir):
    plt.figure(figsize=(6, 4))
    for name in VARIANT_ORDER:
        it = R.get(f"convergence/{name}/iter")
        if it is None:
            continue
        mu = R[f"convergence/{name}/scalar_mean"]
        sd = R[f"convergence/{name}/scalar_std"]
        line, = plt.plot(it, mu, label=VARIANT_LABELS[name], linewidth=2)
        plt.fill_between(it, mu - sd, mu + sd, alpha=0.15, color=line.get_color())
    plt.xlabel("Training iteration")
    plt.ylabel("Eval scalar return (uniform $\\omega$)")
    plt.title("Convergence: ablation over PLTR-MA mechanisms")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_dir, "fig_convergence.png")


def plot_violation(R, out_dir):
    """Constraint-violation cost per variant across training, mean +/- std band.
    This is the figure that makes the constrained variants' benefit visible —
    scalar return alone overlaps because the variants are tuned to matched return."""
    cost_names = ["capacity $c_1$", "queue-overflow $c_2$", "SLA $c_3$"]
    # pick the binding constraint = the one whose final value spreads most across variants
    finals = {}
    for name in VARIANT_ORDER:
        cm = R.get(f"convergence/{name}/cost_mean")
        if cm is not None:
            finals[name] = np.asarray(cm)[-1]        # (K,)
    if not finals:
        return
    spread = np.stack(list(finals.values())).std(0)  # (K,)
    bind = int(np.argmax(spread))

    plt.figure(figsize=(6, 4))
    for name in VARIANT_ORDER:
        it = R.get(f"convergence/{name}/iter")
        if it is None:
            continue
        mu = np.asarray(R[f"convergence/{name}/cost_mean"])[:, bind]
        line, = plt.plot(it, mu, label=VARIANT_LABELS[name], linewidth=2)
        sd_key = f"convergence/{name}/cost_std"
        if sd_key in R:
            sd = np.asarray(R[sd_key])[:, bind]
            plt.fill_between(it, mu - sd, mu + sd, alpha=0.15, color=line.get_color())
    plt.xlabel("Training iteration")
    plt.ylabel(f"Constraint-violation cost ({cost_names[bind]})")
    plt.title("Constraint violation across training (mean $\\pm$ std over seeds)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_dir, "fig_violation.png")


def plot_pareto(R, out_dir):
    if "pareto/returns" not in R:
        return
    ret = R["pareto/returns"]                 # (P, 3) returns (higher=better)
    mask = R["pareto/mask"].astype(bool)
    # plot the two most-varying objectives; color by the third
    spread = ret.std(0)
    ax_x, ax_y = np.argsort(spread)[::-1][:2]
    ax_c = ({0, 1, 2} - {ax_x, ax_y}).pop()
    labels = ["latency", "energy", "SLA"]

    plt.figure(figsize=(6, 4.5))
    sc = plt.scatter(ret[:, ax_x], ret[:, ax_y], c=ret[:, ax_c],
                     cmap="viridis", s=30, alpha=0.6, label="PLTR-MA (dominated)")
    plt.scatter(ret[mask, ax_x], ret[mask, ax_y], facecolors="none",
                edgecolors="red", s=90, linewidths=1.8, label="PLTR-MA (Pareto-optimal)")

    # overlay the fixed-weight scalarized-MAPPO baseline front
    base = R.get("pareto/baseline_returns")
    if base is not None:
        base = np.asarray(base)
        plt.scatter(base[:, ax_x], base[:, ax_y], marker="D", s=80,
                    facecolors="none", edgecolors="black", linewidths=1.8,
                    label="fixed-weight MAPPO (per-weight retrain)")

    plt.colorbar(sc, label=f"{labels[ax_c]} return")
    plt.xlabel(f"{labels[ax_x]} return (higher = better)")
    plt.ylabel(f"{labels[ax_y]} return (higher = better)")

    title = "Pareto front via $\\omega$-sweep (PLTR-MA)"
    hv_p, hv_ps = R.get("pareto/hv_pltr_mean"), R.get("pareto/hv_pltr_std")
    hv_b, hv_bs = R.get("pareto/hv_base_mean"), R.get("pareto/hv_base_std")
    if hv_p is not None and hv_b is not None:
        k = len(base) if base is not None else 0
        title += (f"\nHV: PLTR-MA {float(hv_p):.3f}$\\pm${float(hv_ps):.3f}"
                  f"  vs  {k}$\\times$fixed-weight {float(hv_b):.3f}$\\pm${float(hv_bs):.3f}")
    plt.title(title, fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_dir, "fig_pareto.png")


def plot_ablation(R, out_dir):
    if "ablation/V" not in R:
        return
    V = R["ablation/V"]          # cols: V, scalar, c1, c2, c3, backlog, Zmax
    d = R["ablation/delta"]      # cols: delta, scalar, kl_max
    V_sd = R.get("ablation/V_std")       # same column layout, std (col 0 ~ 0)
    d_sd = R.get("ablation/delta_std")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # auto-pick the binding constraint = the one that varies most across V
    c_cols = V[:, 2:5]
    names = ["capacity $c_1$", "queue-overflow $c_2$", "SLA $c_3$"]
    bind = int(np.argmax(c_cols.std(0)))
    c_err = V_sd[:, 2 + bind] if V_sd is not None else None
    ax1.errorbar(V[:, 0], V[:, 2 + bind], yerr=c_err, fmt="o-", color="tab:red",
                 capsize=3, label=names[bind])
    ax1.set_xlabel("Lyapunov weight $V$")
    ax1.set_ylabel(f"constraint cost {names[bind]}", color="tab:red")
    ax1.set_xscale("log")
    ax1b = ax1.twinx()
    s_err = V_sd[:, 1] if V_sd is not None else None
    ax1b.errorbar(V[:, 0], V[:, 1], yerr=s_err, fmt="s--", color="tab:blue",
                  capsize=3, label="scalar return")
    ax1b.set_ylabel("eval scalar return", color="tab:blue")
    ax1.set_title("Lyapunov trade-off: $V$ vs constraint/return")
    ax1.grid(alpha=0.3)

    kl_err = d_sd[:, 2] if d_sd is not None else None
    ax2.errorbar(d[:, 0], d[:, 2], yerr=kl_err, fmt="o-", color="tab:green",
                 capsize=3, label="realized max KL")
    ax2.plot(d[:, 0], d[:, 0], "k:", alpha=0.6, label="$\\delta$ (radius)")
    ax2.set_xlabel("Trust-region radius $\\delta$")
    ax2.set_ylabel("realized KL")
    ax2.set_title("Trust region: KL respects $\\delta$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(out_dir, "fig_ablation.png")


def plot_stability(R, out_dir):
    plt.figure(figsize=(6, 4))
    plotted = False
    for name in VARIANT_ORDER:
        it = R.get(f"convergence/{name}/iter")
        if it is None:
            continue
        bl = R[f"convergence/{name}/backlog_mean"]
        plt.plot(it, bl, label=VARIANT_LABELS[name], linewidth=2)
        plotted = True
    if not plotted:
        return
    plt.xlabel("Training iteration")
    plt.ylabel("Mean queue backlog")
    plt.title("Empirical queue stability (property P1)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_dir, "fig_stability.png")


def _save(out_dir, name):
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


def main(out_dir: str = "results"):
    R = load(out_dir)
    print("Rendering figures ...")
    plot_convergence(R, out_dir)
    plot_violation(R, out_dir)
    plot_pareto(R, out_dir)
    plot_ablation(R, out_dir)
    plot_stability(R, out_dir)
    print("done.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
