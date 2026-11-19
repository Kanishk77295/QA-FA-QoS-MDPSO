"""
run_extensions.py - Extension experiments.

Exp 5: Ablation of QoS-MDPSO's five mechanisms (paper fitness)  -> results/exp_ablation.csv
Exp 6: VM-failure Monte Carlo on every algorithm's schedule      -> results/exp_failures.csv
       scenario "paper":    VM credibility F ~ U(0, 1)      (paper's range)
       scenario "reliable": VM credibility F ~ U(0.9, 1.0)  (realistic cloud VMs)
Task-count sweep 40..120, 10 seeds each, same environments as run_experiments.py.
"""
import os, time
import numpy as np, pandas as pd
import config as cfg
from environment import CloudEnv
from algorithms import (min_min, dpso, qos_mdpso, qa_qos_mdpso, fa_qos_mdpso,
                        CountingFitness)
from fitness import paper_fitness
from failures import simulate

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------- Exp 5: ablation
VARIANTS = {
    "Full QoS-MDPSO": {},
    "no chaotic w": {"chaos": False},
    "no spbest": {"spbest": False},
    "no mbest": {"mbest": False},
    "no adaptive position": {"adaptive": False},
    "no worst replacement": {"replace": False},
    "all five off": {"chaos": False, "spbest": False, "mbest": False,
                     "adaptive": False, "replace": False},
}
rows = []
for n in cfg.TASK_COUNTS:
    for s in cfg.SEEDS:
        env = CloudEnv(n=n, seed=s)
        for name, mech in VARIANTS.items():
            fit = CountingFitness(paper_fitness)
            t0 = time.time()
            pos, curve = qos_mdpso(env, fitness_fn=fit, seed=s, mech=mech)
            r = env.evaluate(pos)
            rows.append(dict(variant=name, n_tasks=n, seed=s, runtime_s=time.time() - t0,
                             evals=fit.evals, paper_fitness=paper_fitness(env, pos)[0][0],
                             makespan_h=r["makespan"][0], tec=r["tec"][0], tc=r["tc"][0],
                             deadline_miss_true=r["md_true"][0], budget_miss=r["mb"][0]))
    print("ablation n =", n, flush=True)
pd.DataFrame(rows).to_csv("results/exp_ablation.csv", index=False)

# ---------------------------------------------------------------- Exp 6: failures
ALGS = {
    "Min-Min": lambda e, s: min_min(e),
    "DPSO": lambda e, s: dpso(e, seed=s),
    "QoS-MDPSO": lambda e, s: qos_mdpso(e, seed=s),
    "QA-QoS-MDPSO (ours)": lambda e, s: qa_qos_mdpso(e, seed=s),
    "FA-QoS-MDPSO (ours, failure-aware)": lambda e, s: fa_qos_mdpso(e, seed=s),
}
rows = []
for scenario in ["paper", "reliable"]:
    for n in cfg.TASK_COUNTS:
        for s in cfg.SEEDS:
            env = CloudEnv(n=n, seed=s)
            if scenario == "reliable":
                env.F = 0.9 + 0.1 * env.F                      # same VMs, high reliability
                env.tc_ub = (env.f * env.F.max()).sum()
            for name, fn in ALGS.items():
                pos, _ = fn(env, s)
                r = env.evaluate(pos)
                sim = simulate(env, pos, seed=s)
                rows.append(dict(scenario=scenario, algorithm=name, n_tasks=n, seed=s,
                                 makespan_h=r["makespan"][0], tec=r["tec"][0], tc=r["tc"][0],
                                 deadline_miss_true=r["md_true"][0], **sim))
        print("failures", scenario, "n =", n, flush=True)
pd.DataFrame(rows).to_csv("results/exp_failures.csv", index=False)
print("all done")
