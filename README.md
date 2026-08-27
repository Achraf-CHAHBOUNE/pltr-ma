# PLTR-MA

Reference implementation and reproduction package for **PLTR-MA** (Pareto
Lyapunov–Trust-Region Multi-Agent), a constrained multi-agent reinforcement
learning architecture for cloud resource allocation.

A single preference-conditioned policy network is trained across the whole space
of objective weightings, so an operator selects an operating point on the
latency / energy / SLA trade-off at run time by setting a preference vector —
with no retraining. Constraint satisfaction is handled by a centralized Lyapunov
drift-plus-penalty critic that supplies joint constraint signals to decentralized
per-agent updates.

## Reproducing the paper

Everything runs in one container. **Docker is the only prerequisite.**

```bash
chmod +x run.sh
./run.sh smoke      # 2-minute sanity check — do this first
./run.sh n3         # the N=3 study reported in the paper
./run.sh n10        # the N=10 scaling study
```

Add `trace` to run on the real Alibaba workload instead of synthetic arrivals:

```bash
./run.sh n3 trace
```

See **[QUICKSTART.md](QUICKSTART.md)** for hardware guidance and troubleshooting,
and **[DOCKER_README.md](DOCKER_README.md)** for the full configuration reference.

> **Hardware note:** the policy networks are ~5,000 parameters, so a GPU gives no
> measured speed-up — kernel-launch overhead dominates. What helps is CPU cores:
> the runner executes independent training runs in parallel, one per core.
> 16 cores and 16 GB RAM is a comfortable target.

## Regenerating the reported statistics

Every number in the paper can be recomputed from the saved per-seed results,
which are included in `results/`:

```bash
python code/stats.py results/results_synthetic results/results_trace results/results_trace_n10
```

This prints the per-setting hypervolume means, the two-sided Mann–Whitney U test,
Cohen's *d*, and the per-seed win counts — i.e. the contents of the paper's
hypervolume table.

Figures can be regenerated without re-running any training:

```bash
python code/plot.py results/results_synthetic
```

## Repository contents

| Path | Contents |
|---|---|
| `code/cloud_env.py` | The cloud-allocation environment (Gymnasium API) |
| `code/mappo.py` | CTDE MAPPO backbone |
| `code/lyapunov.py` | Virtual queues and drift-plus-penalty |
| `code/trust_region.py` | Per-agent KL trust-region update |
| `code/pareto.py` | Preference conditioning, front sweep, hypervolume |
| `code/lagrangian_ppo.py` | PPO-Lagrangian safe-RL baseline |
| `code/train.py`, `run_experiment.py` | Experiment driver |
| `code/stats.py` | Significance testing (no SciPy dependency) |
| `code/plot.py`, `figures_journal.py` | Figures |
| `code/make_trace.py`, `trace.py` | Alibaba trace conversion and replay |
| `results/` | Per-seed results for all three reported settings |

## The real workload

The paper's second environment replays arrivals from the **Alibaba
cluster-trace-v2018**, which is third-party data available from
[alibaba/clusterdata](https://github.com/alibaba/clusterdata). We distribute the
conversion script rather than a redistributed copy of the trace:

```bash
python code/make_trace.py alibaba /path/to/batch_task.csv \
       --out code/trace_alibaba.csv --bucket 10 --max-steps 5000
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{chahboune2026pltrma,
  title   = {Operator-Tunable Multi-Objective Cloud Resource Management:
             A Data-Driven Architecture with Constraint Guarantees},
  author  = {Chahboune, Achraf and Jadli, Aissam and Bahassine, Said},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
