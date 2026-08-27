"""
mappo.py — CTDE MAPPO backbone for PLTR-MA (build step 2).

Vanilla Multi-Agent PPO with Centralized-Training / Decentralized-Execution:

  * N decentralized actors  pi_i(a_i | o_i ; theta_i)  — one per agent, each
    sees ONLY its own 9-dim local observation (heterogeneous servers => no
    parameter sharing).
  * one centralized critic  V(s ; phi)  — sees the flat joint observation
    (global state, dim = N * OBS_DIM). Used only during training; discarded
    at execution. This is the standard MAPPO value critic (Yu et al., 2022).
  * PPO-clip policy update + GAE(lambda) advantages, shared team advantage
    across agents (the env reward is a cooperative cluster-level signal).

Forward-compatibility for later PLTR-MA steps (kept deliberately minimal):

  * Actor/Critic accept an optional `cond_dim` conditioning input. It is 0
    here; pareto.py (step 5) sets cond_dim=M and feeds the preference vector
    omega. Nothing else changes.
  * The vector reward (M=3) from cloud_env is scalarized by `scalarize()`
    with a fixed preference `omega`. Vanilla MAPPO uses the uniform omega;
    step 5 will randomize it per episode. The augmented per-step reward can
    also be overridden by a hook (`reward_fn`) so lyapunov.py (step 3) can
    inject the drift-plus-penalty term without touching this file.

This is the backbone validated in Week 1 (converged to the theoretical max on
a toy CTDE task); here it is ported onto the cloud allocation env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cloud_env import CloudAllocationEnv


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class MAPPOConfig:
    # --- environment ---------------------------------------------------- #
    n_servers: int = 3
    arrival_rate: float = 2.0
    horizon: int = 200

    # --- optimization --------------------------------------------------- #
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatches: int = 4
    hidden: int = 64            # decentralized actors (always see one 9-dim obs)
    critic_hidden: int | None = None   # None => auto-scale with N (see auto_critic_hidden)

    # --- run control ---------------------------------------------------- #
    n_iterations: int = 400          # one rollout (= one episode) per iteration
    steps_per_update: int = 200      # rollout length; matches horizon by default
    seed: int = 0
    device: str = "cpu"

    # --- multi-objective preference (vanilla MAPPO: uniform & fixed) ---- #
    omega: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """Two-hidden-layer MLP with orthogonal init (PPO-standard)."""
    layers = [
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    ]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    # smaller gain on the output head (standard PPO trick)
    nn.init.orthogonal_(net[-1].weight, 0.01)
    return net


class Actor(nn.Module):
    """Decentralized policy for one agent: local obs (+ optional omega) -> logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = _mlp(obs_dim + cond_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = obs if self.cond_dim == 0 else torch.cat([obs, cond], dim=-1)
        return self.net(x)

    def dist(self, obs, cond=None) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(obs, cond))


class Critic(nn.Module):
    """Centralized state-value V(s) (+ optional omega)."""

    def __init__(self, state_dim: int, hidden: int, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = _mlp(state_dim + cond_dim, 1, hidden)

    def forward(self, state: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = state if self.cond_dim == 0 else torch.cat([state, cond], dim=-1)
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def auto_critic_hidden(state_dim: int, actor_hidden: int = 64, cap: int = 512) -> int:
    """Critic width that keeps pace with the joint-state input.

    The centralized critic sees `state_dim = N * OBS_DIM`, which grows linearly in
    the number of agents (27 at N=3, 90 at N=10, 450 at N=50), while each
    decentralized actor always sees a single 9-dim observation. Holding the critic
    at 64 units would force a ~7x compression of the joint state at N=50, so a weak
    large-N result could not be distinguished from plain critic underfitting — a
    confound a reviewer would (rightly) attack. Widening the critic is free at
    deployment: under CTDE it is discarded after training.

    Rounds up to a power of two, floored at `actor_hidden` and capped at `cap`, so
    small-N behaviour is byte-identical to before (N=3 -> 64).
    """
    import math
    w = 2 ** math.ceil(math.log2(max(int(state_dim), 32)))
    return int(min(cap, max(actor_hidden, w)))


def scalarize(reward_vec: np.ndarray, omega: np.ndarray) -> float:
    """Linear scalarization omega^T r of the (M,) vector reward."""
    return float(np.dot(omega, reward_vec))


def compute_gae(rewards, values, last_value, gamma, lam):
    """Generalized Advantage Estimation. rewards/values are 1-D arrays of len T."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


# --------------------------------------------------------------------------- #
# MAPPO agent
# --------------------------------------------------------------------------- #
class MAPPO:
    def __init__(
        self,
        cfg: MAPPOConfig,
        env: CloudAllocationEnv | None = None,
        cond_dim: int = 0,
        reward_fn: Callable | None = None,
    ):
        """
        cond_dim : width of an extra conditioning vector fed to actors+critic
                   (0 for vanilla MAPPO; M for the weight-conditioned variant).
        reward_fn: optional hook (step_info, scalar_reward) -> augmented scalar
                   reward. Lets lyapunov.py inject drift-plus-penalty without
                   editing this file. None => use the plain scalarized reward.
        """
        self.cfg = cfg
        self.cond_dim = cond_dim
        self.reward_fn = reward_fn
        self.device = torch.device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = env or CloudAllocationEnv(
            n_servers=cfg.n_servers, arrival_rate=cfg.arrival_rate,
            horizon=cfg.horizon, seed=cfg.seed,
        )
        self.N = self.env.N
        obs_dim = self.env.OBS_DIM
        n_act = self.env.N_ACTIONS
        state_dim = self.N * obs_dim

        self.actors = [
            Actor(obs_dim, n_act, cfg.hidden, cond_dim).to(self.device)
            for _ in range(self.N)
        ]
        # actors stay small (deployed, 9-dim obs); the critic scales with N because
        # its input is the joint state and it is discarded at execution time.
        self.critic_hidden = (cfg.critic_hidden if cfg.critic_hidden
                              else auto_critic_hidden(state_dim, cfg.hidden))
        self.critic = Critic(state_dim, self.critic_hidden, cond_dim).to(self.device)

        self.actor_opts = [
            torch.optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in self.actors
        ]
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.omega = np.asarray(cfg.omega, dtype=np.float32)

        # Per-agent KL trust-region radius (CPO-style). None => vanilla PPO-clip.
        # trust_region.py (step 4) sets this to delta; when an agent's update
        # exceeds the radius we freeze it for the rest of the update, enforcing
        # D_KL(pi_old || pi) <~ delta and preserving the monotonic-improvement
        # bound. self.last_kl records the realized per-agent KL each update.
        self.tr_delta: float | None = None
        self.last_kl = np.zeros(self.N)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: np.ndarray, cond: np.ndarray | None = None):
        """obs: (N, obs_dim). Returns actions, logprobs (both (N,))."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        cond_t = self._cond_tensor(cond, batch=self.N)
        actions, logps = [], []
        for i, actor in enumerate(self.actors):
            d = actor.dist(obs_t[i:i + 1], None if cond_t is None else cond_t[i:i + 1])
            a = d.sample()
            actions.append(int(a.item()))
            logps.append(float(d.log_prob(a).item()))
        return np.array(actions), np.array(logps, dtype=np.float32)

    def _cond_tensor(self, cond, batch):
        if self.cond_dim == 0:
            return None
        c = np.broadcast_to(cond, (batch, self.cond_dim)).astype(np.float32)
        return torch.as_tensor(c, device=self.device)

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    def collect_rollout(self, n_steps: int, omega: np.ndarray | None = None):
        """Run the env for n_steps, auto-resetting at episode boundaries."""
        omega = self.omega if omega is None else np.asarray(omega, np.float32)
        cond = omega if self.cond_dim > 0 else None

        buf = {k: [] for k in
               ("obs", "state", "actions", "logps", "rew", "val", "done", "cost")}
        ep_returns_vec, ep_scalar = [], []          # completed-episode logging
        cur_vec = np.zeros(self.env.M)
        cur_scalar = 0.0

        obs = self._obs
        for _ in range(n_steps):
            state = obs.reshape(-1)
            with torch.no_grad():
                v = self.critic(
                    torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                    self._cond_tensor(cond, 1),
                ).item()
            actions, logps = self.act(obs, cond)
            nobs, rvec, term, trunc, info = self.env.step(actions)

            sca = scalarize(rvec, omega)
            aug = self.reward_fn(info, sca) if self.reward_fn is not None else sca

            buf["obs"].append(obs.copy())
            buf["state"].append(state.copy())
            buf["actions"].append(actions.copy())
            buf["logps"].append(logps.copy())
            buf["rew"].append(aug)
            buf["val"].append(v)
            buf["done"].append(term or trunc)
            buf["cost"].append(np.asarray(info["cost"], dtype=np.float32))  # (K,) per-step constraint cost

            cur_vec += rvec
            cur_scalar += sca
            obs = nobs
            if term or trunc:
                ep_returns_vec.append(cur_vec.copy())
                ep_scalar.append(cur_scalar)
                cur_vec = np.zeros(self.env.M)
                cur_scalar = 0.0
                obs, _ = self.env.reset()

        self._obs = obs
        # bootstrap value for the final state
        with torch.no_grad():
            last_v = self.critic(
                torch.as_tensor(obs.reshape(-1), dtype=torch.float32,
                                device=self.device).unsqueeze(0),
                self._cond_tensor(cond, 1),
            ).item()

        rollout = {k: np.asarray(v) for k, v in buf.items()}
        rollout["last_value"] = last_v
        rollout["omega"] = omega
        rollout["ep_returns_vec"] = ep_returns_vec
        rollout["ep_scalar"] = ep_scalar
        return rollout

    # ------------------------------------------------------------------ #
    # Update (PPO-clip)
    # ------------------------------------------------------------------ #
    def update(self, roll: dict) -> dict:
        cfg = self.cfg
        adv, returns = compute_gae(
            roll["rew"], roll["val"], roll["last_value"], cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = len(adv)
        obs = torch.as_tensor(roll["obs"], dtype=torch.float32, device=self.device)        # (T,N,obs)
        state = torch.as_tensor(roll["state"], dtype=torch.float32, device=self.device)    # (T,N*obs)
        actions = torch.as_tensor(roll["actions"], dtype=torch.long, device=self.device)   # (T,N)
        old_logps = torch.as_tensor(roll["logps"], dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        cond = roll["omega"] if self.cond_dim > 0 else None
        cond_obs = self._cond_tensor(cond, T) if cond is not None else None  # (T,cond)

        mb_size = max(1, T // cfg.minibatches)
        logs = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_updates = 0

        # Trust-region bookkeeping: per-agent realized KL and a freeze flag.
        kl_accum = np.zeros(self.N)
        kl_count = np.zeros(self.N)
        frozen = np.zeros(self.N, dtype=bool)

        for _ in range(cfg.update_epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, mb_size):
                mb = idx[start:start + mb_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                # --- critic ---
                v_pred = self.critic(
                    state[mb_t], None if cond_obs is None else cond_obs[mb_t]
                )
                v_loss = F.mse_loss(v_pred, ret_t[mb_t])
                self.critic_opt.zero_grad()
                (cfg.value_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                # --- actors (per-agent, shared team advantage) ---
                for i, actor in enumerate(self.actors):
                    if frozen[i]:                       # trust-region: agent done
                        continue
                    d = actor.dist(
                        obs[mb_t, i],
                        None if cond_obs is None else cond_obs[mb_t],
                    )
                    new_logp = d.log_prob(actions[mb_t, i])
                    ratio = torch.exp(new_logp - old_logps[mb_t, i])

                    # standard positive PPO KL estimator E[(r-1) - log r]
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    if self.tr_delta is not None and approx_kl > self.tr_delta:
                        frozen[i] = True                # exceeded radius -> freeze
                        kl_accum[i] += approx_kl; kl_count[i] += 1
                        continue

                    surr1 = ratio * adv_t[mb_t]
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[mb_t]
                    ent = d.entropy().mean()
                    pi_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * ent

                    self.actor_opts[i].zero_grad()
                    pi_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    self.actor_opts[i].step()

                    logs["pi_loss"] += pi_loss.item()
                    logs["entropy"] += ent.item()
                    logs["kl"] += approx_kl
                    kl_accum[i] += approx_kl; kl_count[i] += 1
                logs["v_loss"] += v_loss.item()
                n_updates += 1

        for k in logs:
            logs[k] /= max(1, n_updates) * (self.N if k in ("pi_loss", "entropy", "kl") else 1)
        self.last_kl = kl_accum / np.maximum(1, kl_count)
        logs["max_kl"] = float(self.last_kl.max())
        return logs

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def train(self, n_iterations: int | None = None, verbose: bool = True) -> dict:
        n_iterations = n_iterations or self.cfg.n_iterations
        self._obs, _ = self.env.reset()
        history = {"iter": [], "scalar_return": [], "ret_latency": [],
                   "ret_energy": [], "ret_sla": [], "mean_backlog": []}

        for it in range(n_iterations):
            roll = self.collect_rollout(self.cfg.steps_per_update)
            self.update(roll)

            if roll["ep_scalar"]:
                sca = float(np.mean(roll["ep_scalar"]))
                vec = np.mean(roll["ep_returns_vec"], axis=0)
            else:  # rollout shorter than an episode — fall back to summary
                sca, vec = float("nan"), np.full(self.env.M, np.nan)
            summ = self.env.episode_summary()

            history["iter"].append(it)
            history["scalar_return"].append(sca)
            history["ret_latency"].append(float(vec[0]))
            history["ret_energy"].append(float(vec[1]))
            history["ret_sla"].append(float(vec[2]))
            history["mean_backlog"].append(summ["mean_backlog"])

            if verbose and (it % max(1, n_iterations // 10) == 0 or it == n_iterations - 1):
                print(f"iter {it:4d} | scalar_return {sca:8.2f} | "
                      f"latency {vec[0]:8.1f} energy {vec[1]:8.1f} sla {vec[2]:7.1f} | "
                      f"meanQ {summ['mean_backlog']:.2f}")
        return history


__all__ = ["MAPPO", "MAPPOConfig", "Actor", "Critic", "scalarize", "compute_gae"]
