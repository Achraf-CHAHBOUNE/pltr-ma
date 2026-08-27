"""lagrangian_ppo.py — PPO-Lagrangian constrained-RL baseline (P3 head-to-head).

The standard safe-RL baseline (the primary baseline used across the Safety Gym
benchmark): augment the reward with a *learned* Lagrange multiplier on the
constraint cost, and update that multiplier by dual ascent on the realized
constraint violation. It is deliberately distinct from PLTR-MA's Lyapunov
drift-plus-penalty, which uses a queueing-theoretic *dynamic* dual (virtual
queues) rather than a learned one — so the comparison isolates "learned dual vs
Lyapunov dual" rather than re-running an ablation of our own mechanism.

Wired onto the same CTDE MAPPO backbone, env, evaluation and hypervolume harness
as every other variant, so it drops straight into the P3 comparison table.
"""
from __future__ import annotations

import numpy as np

from mappo import MAPPO, MAPPOConfig


class LagrangianMAPPO(MAPPO):
    """MAPPO + PPO-Lagrangian: reward is scalarized_return - lambda * cost[cost_idx];
    lambda is updated each iteration by projected dual ascent toward cost_limit."""

    def __init__(self, cfg: MAPPOConfig, cost_idx: int = 0, cost_limit: float = 0.0,
                 lam_lr: float = 0.05, lam_init: float = 1.0, lam_max: float = 100.0,
                 **kwargs):
        # attributes needed by the reward hook must exist BEFORE super().__init__,
        # since the hook is registered there (it is only *called* during rollouts).
        self.cost_idx = int(cost_idx)
        self.cost_limit = float(cost_limit)
        self.lam_lr = float(lam_lr)
        self.lam = float(lam_init)
        self.lam_max = float(lam_max)
        self.lam_history: list[float] = []
        kwargs.pop("reward_fn", None)                 # this baseline owns the reward hook
        super().__init__(cfg, reward_fn=self._augment, **kwargs)

    def _augment(self, info, scalar: float) -> float:
        """Lagrangian-augmented per-step reward: r - lambda * c_k."""
        return float(scalar) - self.lam * float(info["cost"][self.cost_idx])

    def update(self, roll: dict) -> dict:
        logs = super().update(roll)
        # projected dual ascent on the multiplier toward the cost limit
        jc = float(np.asarray(roll["cost"])[:, self.cost_idx].mean())
        self.lam = float(np.clip(self.lam + self.lam_lr * (jc - self.cost_limit),
                                 0.0, self.lam_max))
        self.lam_history.append(self.lam)
        logs["lam"] = self.lam
        logs["jc"] = jc
        return logs

    def lagrangian_report(self) -> dict:
        lam = np.asarray(self.lam_history) if self.lam_history else np.zeros(1)
        return {"cost_idx": self.cost_idx, "cost_limit": self.cost_limit,
                "lam_final": float(lam[-1]), "lam_mean": float(lam.mean())}


__all__ = ["LagrangianMAPPO"]
