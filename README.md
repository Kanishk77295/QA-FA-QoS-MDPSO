# QA-FA-QoS-MDPSO

**Queue-Aware and Failure-Aware Fitness for Multi-Objective PSO Cloud Scheduling:
Replication, Ablation and Failure Analysis of QoS-MDPSO**

Repository: https://github.com/Kanishk77295/QA-FA-QoS-MDPSO

Authors: Vatsal Sharma, Harshit, Kanishk Gupta
Department of Information Technology, Netaji Subhas University of Technology, Main Campus, Delhi, India.

This project re-implements QoS-MDPSO (Ouyang & Zhang, *Cluster Computing* 28:727, 2025,
doi:10.1007/s10586-025-05532-5) together with its baselines (Min-Min, DPSO) and examines its model.
The base paper's deadline test (Eq. 9) compares a task's execution time with its deadline and ignores
queueing on single-core VMs. We contribute:

* **QA-QoS-MDPSO** - identical search, queue-aware (earliest-deadline-first completion time), normalized, additively penalized fitness.
* **FA-QoS-MDPSO** - the QA fitness plus expected retries and losses from a VM failure model (credibility = per-attempt success probability).
* A one-at-a-time **ablation** of the five QoS-MDPSO mechanisms, with fitness-evaluation counts.
* A **Monte Carlo failure simulator** that tests whether credibility predicts task survival.

## Repository layout

| Path | Purpose |
|---|---|
| `config.py` | Every experimental parameter, each marked `PAPER` (from the base paper) or `ASSUMED` |
| `environment.py` | Seeded generator of tasks and VMs; batch evaluator (makespan, cost, credibility, Eq. 9 misses, true queue-aware misses) |
| `fitness.py` | `paper_fitness` (Eq. 11, sign-corrected), `qa_fitness`, `fa_fitness` |
| `algorithms.py` | Min-Min, DPSO, QoS-MDPSO (with ablation switches), QA- and FA-QoS-MDPSO |
| `failures.py` | Monte Carlo VM-failure simulator (3 attempts, 200 scenarios per schedule) |
| `run_experiments.py` | Task-length sweep, task-count sweep, penalty sensitivity, convergence |
| `run_extensions.py` | Ablation and failure experiments |
| `analyze.py`, `analyze_extensions.py` | Summary tables, % improvements, Wilcoxon tests, figures |
| `results/` | Raw per-run CSVs and the `stats.txt` / `stats_extensions.txt` outputs used in the paper |
| `figures/` | PNG figures produced by the analysis scripts |
| `paper/` | LaTeX source (Springer Nature `sn-jnl` template, `sn-basic` references), compiled PDF, scripts that draw the paper figures |

## Reproducing the results

Python 3.9+ is required.

```bash
pip install -r requirements.txt
python run_experiments.py      # results/exp_length.csv, exp_tasks.csv, exp_lambda.csv, convergence.csv
python run_extensions.py       # results/exp_ablation.csv, exp_failures.csv
python analyze.py              # results/stats.txt, results/summary_*.csv, figures/*.png
python analyze_extensions.py   # results/stats_extensions.txt, figures/ablation.png, failures_*.png
```

Run every script from the repository root. All runs are seeded (seeds 0-9); seed `s` builds the same
environment for every algorithm, so comparisons are paired. The CSV files in `results/` come from a run on
Ubuntu in VirtualBox; re-running `analyze.py` and `analyze_extensions.py` on them reproduces the shipped
`stats*.txt` exactly. A fresh `run_experiments.py` / `run_extensions.py` run gives the same schedules
(fixed seeds) but different `runtime_s` values.

### Building the paper

```bash
cd paper
python make_diagrams.py && python make_figures.py   # draws the vector figures from ../results
latexmk -pdf main.tex
```

`sn-jnl.cls` and `sn-basic.bst` are the Springer Nature authoring-template files.

## Key parameters

Taken from the base paper: 100 tasks (sweep 40-120), task length 4x10^6 MI (sweep 2-16x10^6), deadlines U[3x10^4, 5x10^4] s,
12 VMs from six templates (100-2600 MIPS, 0.06-1.0 $/h), c1 = c2 = 2, inertia weight 0.4-0.9.
Assumed (not reported in the base paper): swarm size 30, 200 iterations, objective weights 1/3 each,
budget U(0.8, 1.2) x mean task cost, round-and-clip discretization.
Ours: penalty weight lambda = 1, failure model with K = 3 attempts.

## Headline results (10 seeds per setting)

* At the default setting QoS-MDPSO has 37.6 true deadline misses out of 100 tasks; the base paper's Eq. 9 counts 0.6.
* QA-QoS-MDPSO reduces true misses by 27.4% / 58.9% (length / count sweeps) and makespan by 5.6% / 17.1%, at 8.3% / 11.7% lower credibility.
* Credibility predicts task survival under failures (r = 0.889); FA-QoS-MDPSO completes 84.3% of tasks versus 78.8% for QoS-MDPSO.
* A textbook Min-Min beats all PSO variants on makespan and true misses; the spbest mechanism hurts the base paper's fitness.

## Citation

Please cite the paper in `paper/` and the base paper above.
