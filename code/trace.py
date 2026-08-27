"""trace.py — real-workload trace driver for CloudAllocationEnv (P3 / Q1 credibility).

Replaces the synthetic Poisson arrival process with task arrivals *replayed from a
real cluster trace* (e.g. Alibaba cluster-trace-v2018), so evaluation rests on a
realistic, heterogeneous, time-varying workload rather than a hand-tuned analytical
process. The environment dynamics are otherwise identical; only WHERE the arriving
tasks come from changes. This is what turns the "custom toy simulator" limitation of
the synthetic-only setting into a real-workload evaluation.

CSV schema (one row per task), already normalised into the env's units:

    arrival_step,cpu,mem,work,slack

Produce this file with make_trace.py — either `synthetic()` (a small realistic sample
for local testing, no download) or `from_alibaba()` (convert the real trace on the
cluster; https://github.com/alibaba/clusterdata).
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np


class WorkloadTrace:
    """Replays task arrivals from a normalised trace CSV, bucketed by arrival step.

    A single long trace serves many episodes: `reset(rng)` picks a random window
    offset so different seeds/episodes see different slices of the workload, which
    gives genuine per-seed variety (needed for the multi-seed error bars) from one
    real trace.
    """

    def __init__(self, path: str, horizon: int, loop: bool = True):
        self.path = path
        self.horizon = int(horizon)
        self.loop = loop

        by_step: dict[int, list] = defaultdict(list)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                s = int(float(row["arrival_step"]))
                by_step[s].append((
                    float(row["cpu"]), float(row["mem"]),
                    float(row["work"]), float(row["slack"]),
                ))
        self._by_step = dict(by_step)
        self.max_step = max(self._by_step) if self._by_step else 0
        self.n_tasks = sum(len(v) for v in self._by_step.values())
        self.offset = 0

    def reset(self, rng: np.random.Generator) -> None:
        """Pick a random window start so each episode replays a different slice."""
        span = self.max_step - self.horizon
        self.offset = int(rng.integers(0, span + 1)) if span > 0 else 0

    def tasks_at(self, step: int):
        """Return [(cpu, mem, work, slack), ...] arriving at this episode step."""
        s = self.offset + int(step)
        if self.loop and self.max_step > 0:
            s = s % (self.max_step + 1)
        return self._by_step.get(s, [])

    def __repr__(self) -> str:
        return (f"WorkloadTrace(tasks={self.n_tasks}, span={self.max_step} steps, "
                f"horizon={self.horizon}, loop={self.loop})")


__all__ = ["WorkloadTrace"]
