"""
fitness.py - the two fitness functions compared in this project (lower = better).

paper_fitness : Eq. (11) of Ouyang & Zhang (2025), with the sign typo fixed
                (cost +, credibility -), using Eq. (9) deadline check (no queueing).
qa_fitness    : OUR queue-aware, normalised fitness.
"""
import numpy as np
import config as cfg


def paper_fitness(env, P):
    r = env.evaluate(P)
    num = cfg.ALPHA * r["makespan"] + cfg.BETA * r["tec"] - cfg.GAMMA * r["tc"]
    den = 1.0 + np.exp(-np.sqrt(r["md_paper"] + r["mb"]))
    return num / den, r


def qa_fitness(env, P):
    r = env.evaluate(P)
    ms_n = r["makespan"] / env.ms_lb           # >= 1, closer to 1 is better
    tec_n = r["tec"] / env.tec_lb              # >= 1
    tc_gap = 1.0 - r["tc"] / env.tc_ub         # in [0,1], 0 is best
    viol = (r["md_true"] + r["mb"]) / env.n    # fraction of tasks violating a constraint
    fit = (cfg.ALPHA * ms_n + cfg.BETA * tec_n + cfg.GAMMA * tc_gap
           + cfg.LAMBDA_PENALTY * viol)
    return fit, r


def fa_fitness(env, P, K=None):
    """Failure-aware queue-aware fitness (extension 2).
    Uses the failure model of failures.py IN EXPECTATION:
      expected attempts on VM j  a_j = (1 - (1-F_j)^K) / F_j   (geometric, capped at K)
      probability a task is lost    q_j = (1-F_j)^K
    Times and costs are scaled by a_j; expected lost tasks join the violation count."""
    from failures import MAX_ATTEMPTS
    K = K or MAX_ATTEMPTS
    F = np.clip(env.F, 1e-9, 1.0)
    a = (1 - (1 - F) ** K) / F                     # (m,)
    q = (1 - F) ** K
    r = env.evaluate(P, mult=np.broadcast_to(a, (env.n, env.m)))
    lost = q[np.atleast_2d(P)].sum(1)
    ms_n = r["makespan"] / env.ms_lb
    tec_n = r["tec"] / env.tec_lb
    tc_gap = 1.0 - r["tc"] / env.tc_ub
    viol = (r["md_true"] + r["mb"] + lost) / env.n
    fit = (cfg.ALPHA * ms_n + cfg.BETA * tec_n + cfg.GAMMA * tc_gap
           + cfg.LAMBDA_PENALTY * viol)
    return fit, r
