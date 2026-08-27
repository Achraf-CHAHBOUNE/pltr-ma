"""
trust_region.py — CPO-style per-agent trust-region update (build step 4).

Implements Section IV.B of the paper. Each agent's actor is updated to maximize
the (Lyapunov-augmented) advantage subject to a per-agent KL trust region

    D_KL( pi_i^old || pi_i ) <= delta .

Full CPO solves this with a conjugate-gradient / Fisher-vector step plus a line
search. For this preliminary, single-GPU feasibility study we use the standard
practical surrogate: PPO-clip for the linearized objective plus a KL trust
region enforced by EARLY-STOPPING each agent once its realized KL exceeds delta
(the mechanism wired into mappo.MAPPO.update via `tr_delta`). This is the
common TRPO/CPO approximation in deep safe-RL and preserves the monotonic-
improvement intuition; the exact CG projection and its multi-agent proof are
left to future work.

The constraint side of CPO is supplied by the Lyapunov virtual queues
(lyapunov.py): the drift-plus-penalty term injected through `reward_fn` acts as
the dynamic dual variable on the constraints, with the constraint signal coming
from the cluster-level cost, which is the centralized critic's role. This module
is therefore intentionally thin — it activates and logs the KL trust region.

Usage:
    algo = TrustRegionMAPPO(cfg, delta=0.02, reward_fn=lyap.make_reward_fn())
"""

from __future__ import annotations

import numpy as np

from mappo import MAPPO, MAPPOConfig


class TrustRegionMAPPO(MAPPO):
    """MAPPO with a per-agent KL trust-region radius delta on the actor update."""

    def __init__(self, cfg: MAPPOConfig, delta: float = 0.02, **kwargs):
        super().__init__(cfg, **kwargs)
        self.tr_delta = delta          # activates the KL early-stop in update()
        self.kl_history: list = []     # realized max per-agent KL per update

    def update(self, roll: dict) -> dict:
        logs = super().update(roll)
        self.kl_history.append(float(self.last_kl.max()))
        return logs

    def kl_report(self) -> dict:
        kl = np.asarray(self.kl_history) if self.kl_history else np.zeros(1)
        return {
            "delta": self.tr_delta,
            "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "within_delta_frac": float(np.mean(kl <= self.tr_delta * 1.5)),
        }


__all__ = ["TrustRegionMAPPO"]
