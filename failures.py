"""
failures.py - Monte Carlo simulation of VM failures (our extension).

The paper optimises credibility (TC) but never tests whether a high-TC schedule
actually fails less. Here credibility gets a concrete meaning:

  * Each execution attempt of task i on VM j SUCCEEDS with probability F_j.
  * A failure is detected when the attempt ends; the task is re-run on the SAME VM
    (time and money spent again), up to MAX_ATTEMPTS attempts.
  * A task that fails all attempts is FAILED (lost); its later queue-mates still run.
  * Tasks inside a VM run in EDF order, as in environment.evaluate().

For each schedule we draw N_MC random failure scenarios and report averages.
"""
import numpy as np

MAX_ATTEMPTS = 3      # ASSUMED: original run + 2 retries
N_MC = 200            # Monte Carlo scenarios per schedule


def simulate(env, pos, n_mc=N_MC, max_attempts=MAX_ATTEMPTS, seed=0):
    rng = np.random.default_rng(seed + 7000)
    pos = np.asarray(pos)
    n = env.n
    rows = np.arange(n)
    p_ok = env.F[pos]                                   # per-attempt success probability
    e = env.E[rows, pos]                                # seconds per attempt
    c = env.C[rows, pos]                                # dollars per attempt

    # attempts needed until first success ~ Geometric(p_ok); capped at max_attempts
    need = rng.geometric(np.clip(p_ok, 1e-9, 1.0), size=(n_mc, n))
    ok = need <= max_attempts                           # task eventually succeeded
    used = np.minimum(need, max_attempts)               # attempts actually executed

    t = used * e[None, :]                               # (n_mc, n) time spent per task
    cost = (used * c[None, :]).sum(1)

    # queue-aware completion times, EDF order within each VM
    order = env.edf
    finish = np.zeros((n_mc, n))
    for j in np.unique(pos):
        idx = order[pos[order] == j]                    # tasks on VM j, in EDF order
        finish[:, idx] = np.cumsum(t[:, idx], axis=1)
    makespan_h = finish.max(1) / 3600.0
    late_or_lost = (~ok) | (finish > env.d[None, :])    # missed = lost OR finished late

    first_try_fail = (need > 1).sum(1)                  # attempts that failed at least once
    crit = env.f
    return dict(
        success_rate=ok.mean(),                                         # share of tasks completed
        crit_weighted_success=(ok * crit).sum(1).mean() / crit.sum(),   # importance-weighted
        lost_tasks=(~ok).sum(1).mean(),
        lost_critical=((~ok) & (crit[None, :] >= 0.7)).sum(1).mean(),   # lost tasks with f >= 0.7
        tasks_with_retries=first_try_fail.mean(),
        eff_makespan_h=makespan_h.mean(),
        eff_cost=cost.mean(),
        eff_deadline_miss=late_or_lost.sum(1).mean(),
    )
