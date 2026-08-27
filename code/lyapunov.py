"""
lyapunov.py — Lyapunov drift-plus-penalty constraint handling (build step 3).

Implements Section IV.A of the paper as a lightweight tracker that plugs into
the MAPPO backbone through its `reward_fn` hook (no edits to mappo.py).

For each of the K constraints we maintain a virtual queue Z_k:

    Z_k(t+1) = max{ 0, Z_k(t) + c_k(t) - d_k }          (Eq. IV.A)

where c_k(t) is the per-step constraint cost from the env (info["cost"]) and
d_k is the operator tolerance. The drift-plus-penalty rule replaces the plain
scalarized reward  r_t = omega^T r_t  with the augmented per-step reward

    r_t^aug = V * r_t  -  sum_k Z_k(t) * c_k(t)

i.e. we MAXIMIZE  V * reward - drift, which is the negative of the cost
Q^L = E[ sum_t gamma^t ( Delta(t) - V * omega^T r_t ) ] that the paper's
centralized critic estimates. Small V => constraints dominate (tight queues,
O(V) backlog); large V => reward dominates (O(1/V) optimality gap). The penalty
uses the current queue level Z_k(t) and then advances the queue with the
observed cost — the standard single-slot drift-plus-penalty update.

The tracker also records a per-step trace of the virtual queues so we can
verify mean-rate stability empirically (paper property P1 / Section V).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LyapunovTracker:
    K: int = 3
    # operator tolerances d_k. Set to zero so every constraint violation builds
    # its virtual queue (the constraints are then always active and the Lyapunov
    # weight V genuinely trades constraint satisfaction against reward). With
    # slack tolerances the cluster satisfies everything and V has no leverage
    # (see experiments/test2 -> test3).
    d: tuple[float, ...] = (0.0, 0.0, 0.0)
    V: float = 1.0

    Z: np.ndarray = field(init=False)
    Z_trace: list = field(default_factory=list, init=False)
    _last_step: int = field(default=0, init=False)

    def __post_init__(self):
        self.d = np.asarray(self.d, dtype=np.float64)
        assert len(self.d) == self.K
        self.Z = np.zeros(self.K, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.Z = np.zeros(self.K, dtype=np.float64)

    def penalty(self, cost: np.ndarray) -> float:
        """Drift penalty term  sum_k Z_k(t) * c_k(t)  using the current queues."""
        return float(np.dot(self.Z, cost))

    def advance(self, cost: np.ndarray) -> None:
        """Virtual-queue update  Z_k <- max(0, Z_k + c_k - d_k)."""
        self.Z = np.maximum(0.0, self.Z + np.asarray(cost, np.float64) - self.d)
        self.Z_trace.append(self.Z.copy())

    def augmented_reward(self, scalar_reward: float, cost: np.ndarray) -> float:
        p = self.penalty(cost)
        self.advance(cost)
        return self.V * scalar_reward - p

    # ------------------------------------------------------------------ #
    def make_reward_fn(self):
        """Return a closure matching MAPPO's reward_fn(info, scalar_reward).

        Resets the virtual queues at each episode boundary (detected via
        info["step"] == 1, since the env increments t to 1 on the first step
        after reset). Per-episode reset is the standard choice for episodic
        training and matches the per-episode queue-stability check.
        """
        def fn(info: dict, scalar_reward: float) -> float:
            if info["step"] == 1:
                self.reset()
            return self.augmented_reward(scalar_reward, info["cost"])
        return fn

    # ------------------------------------------------------------------ #
    def stability_report(self) -> dict:
        tr = np.asarray(self.Z_trace) if self.Z_trace else np.zeros((1, self.K))
        return {
            "Z_mean": tr.mean(axis=0),
            "Z_max": tr.max(axis=0),
            "Z_final": tr[-1],
            "bounded": bool(np.all(np.isfinite(tr)) and tr.max() < 1e4),
        }


__all__ = ["LyapunovTracker"]
