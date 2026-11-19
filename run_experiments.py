"""
run_experiments.py - runs every algorithm on every setting for every seed.

Exp 1: vary task length (paper Fig. 4)      -> results/exp_length.csv
Exp 2: vary number of tasks (paper Fig. 5)  -> results/exp_tasks.csv
Exp 3: penalty weight sensitivity (ours)    -> results/exp_lambda.csv
Exp 4: convergence curves (default setting) -> results/convergence.csv
Same seed => same cloud environment for all algorithms (fair comparison).
"""
import os, time
import numpy as np, pandas as pd
import config as cfg
from environment import CloudEnv
from algorithms import ALGORITHMS, dpso, qos_mdpso, qa_qos_mdpso
from fitness import paper_fitness

os.makedirs("results", exist_ok=True)


def one_run(env, name, fn, seed, **extra):
    t0 = time.time()
    pos, _ = fn(env, seed)
    rt = time.time() - t0
    r = env.evaluate(pos)
    return dict(algorithm=name, seed=seed, runtime_s=rt,
                makespan_h=r["makespan"][0], tec=r["tec"][0], tc=r["tc"][0],
                deadline_miss_paper=r["md_paper"][0], deadline_miss_true=r["md_true"][0],
                budget_miss=r["mb"][0], paper_fitness=paper_fitness(env, pos)[0][0], **extra)


rows = []
for L in cfg.TASK_LENGTHS:
    for s in cfg.SEEDS:
        env = CloudEnv(n=cfg.N_TASKS_DEFAULT, avg_len=L, seed=s)
        for name, fn in ALGORITHMS.items():
            rows.append(one_run(env, name, fn, s, task_length=L))
    print("length", L, "done", flush=True)
pd.DataFrame(rows).to_csv("results/exp_length.csv", index=False)

rows = []
for n in cfg.TASK_COUNTS:
    for s in cfg.SEEDS:
        env = CloudEnv(n=n, seed=s)
        for name, fn in ALGORITHMS.items():
            rows.append(one_run(env, name, fn, s, n_tasks=n))
    print("tasks", n, "done", flush=True)
pd.DataFrame(rows).to_csv("results/exp_tasks.csv", index=False)

rows = []
for lam in [0.5, 1, 2, 5, 10]:
    cfg.LAMBDA_PENALTY = lam
    for s in cfg.SEEDS:
        env = CloudEnv(seed=s)
        rows.append(one_run(env, "QA-QoS-MDPSO (ours)",
                            lambda e, sd: qa_qos_mdpso(e, seed=sd), s, lam=lam))
cfg.LAMBDA_PENALTY = 1.0
pd.DataFrame(rows).to_csv("results/exp_lambda.csv", index=False)
print("lambda done", flush=True)

# convergence: best paper-fitness found so far, re-scored on the paper fitness
rows = []
for s in cfg.SEEDS[:5]:
    env = CloudEnv(seed=s)
    for name, fn in [("DPSO", lambda e: dpso(e, seed=s)),
                     ("QoS-MDPSO", lambda e: qos_mdpso(e, seed=s))]:
        _, curve = fn(env)
        rows += [dict(algorithm=name, seed=s, iteration=i, best_fitness=v)
                 for i, v in enumerate(curve)]
pd.DataFrame(rows).to_csv("results/convergence.csv", index=False)
print("all done")
