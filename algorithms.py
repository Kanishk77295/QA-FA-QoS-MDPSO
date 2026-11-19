"""
algorithms.py - Min-Min, DPSO, QoS-MDPSO (paper) and QA-QoS-MDPSO (ours).

Every algorithm returns (best_position, convergence_curve).
QoS-MDPSO and QA-QoS-MDPSO share ONE implementation; only the fitness
function passed in differs. That isolates our contribution.
"""
import numpy as np
import config as cfg
from fitness import paper_fitness, qa_fitness, fa_fitness


def _discretise(X, m):
    """Continuous PSO position -> valid VM index (round + clip to [0, m-1])."""
    return np.clip(np.rint(X), 0, m - 1).astype(int)


# ------------------------------------------------------------------ Min-Min
def min_min(env, *_):
    """Classic Min-Min: repeatedly schedule the task whose earliest
    possible completion time is smallest, on the VM giving that time."""
    ready = np.zeros(env.m)
    left = np.ones(env.n, bool)
    pos = np.zeros(env.n, int)
    for _ in range(env.n):
        ct = ready[None, :] + env.E                    # completion time matrix
        ct[~left] = np.inf
        i, j = np.unravel_index(np.argmin(ct), ct.shape)
        pos[i], ready[j], left[i] = j, ct[i, j], False
    return pos, None


# ------------------------------------------------------------------ DPSO
def dpso(env, fitness_fn=paper_fitness, seed=0, N=cfg.SWARM_SIZE, T=cfg.MAX_ITER):
    """Standard discrete PSO: Eqs. (4)-(8), linear inertia weight Eq. (6)."""
    rng = np.random.default_rng(seed + 1000)
    n, m = env.n, env.m
    X = rng.integers(0, m, (N, n)).astype(float)
    V = rng.uniform(-m / 2, m / 2, (N, n))
    fit, _ = fitness_fn(env, _discretise(X, m))
    pbest, pfit = X.copy(), fit.copy()
    g = np.argmin(pfit); gbest, gfit = pbest[g].copy(), pfit[g]
    curve = []
    for t in range(T):
        w = cfg.W_MAX - (cfg.W_MAX - cfg.W_MIN) * t / T                   # Eq. 6
        r1, r2 = rng.random((N, n)), rng.random((N, n))
        V = w * V + cfg.C1 * r1 * (pbest - X) + cfg.C2 * r2 * (gbest - X)  # Eq. 4
        V = np.clip(V, -m / 2, m / 2)
        X = _discretise(X + V, m).astype(float)                           # Eq. 5
        fit, _ = fitness_fn(env, X.astype(int))
        better = fit < pfit
        pbest[better], pfit[better] = X[better], fit[better]               # Eq. 7
        g = np.argmin(pfit)
        if pfit[g] < gfit: gbest, gfit = pbest[g].copy(), pfit[g]          # Eq. 8
        curve.append(gfit)
    return gbest.astype(int), np.array(curve)


# ------------------------------------------------------------------ QoS-MDPSO
ALL_ON = dict(chaos=True, spbest=True, mbest=True, adaptive=True, replace=True)


class CountingFitness:
    """Wraps a fitness function and counts how many schedules it scores."""
    def __init__(self, fn):
        self.fn, self.evals = fn, 0

    def __call__(self, env, P):
        self.evals += np.atleast_2d(P).shape[0]
        return self.fn(env, P)


def qos_mdpso(env, fitness_fn=paper_fitness, seed=0, N=cfg.SWARM_SIZE, T=cfg.MAX_ITER,
              mech=None):
    """QoS-MDPSO following Algorithm 1 / Eqs. (12)-(24) of the paper.

    mech: ablation switches (default: all five mechanisms ON = the paper's algorithm).
      chaos=False    -> linear inertia weight, Eq. (6)       instead of Eqs. (12)-(13)
      spbest=False   -> learn from own pbest                  instead of spbest, Eqs. (14)-(15)
      mbest=False    -> learn from gbest                      instead of mbest, Eq. (16)
      adaptive=False -> always X = X + V, Eq. (19)            instead of Eqs. (20)-(21)
      replace=False  -> no worst-particle replacement         instead of Eqs. (22)-(24)
    """
    mech = {**ALL_ON, **(mech or {})}
    rng = np.random.default_rng(seed + 2000)
    n, m = env.n, env.m
    X = rng.integers(0, m, (N, n)).astype(float)
    V = rng.uniform(-m / 2, m / 2, (N, n))
    fit, _ = fitness_fn(env, _discretise(X, m))
    pbest, pfit = X.copy(), fit.copy()
    g = np.argmin(pfit); gbest, gfit = pbest[g].copy(), pfit[g]

    R = rng.uniform(0.01, 0.99)
    while R in (0.25, 0.5, 0.75): R = rng.uniform(0.01, 0.99)              # Eq. 12 exclusions
    curve = []
    for t in range(T):
        # --- inertia weight: chaotic Eqs. (12)-(13), or linear Eq. (6) if ablated
        if mech["chaos"]:
            R = 4 * R * (1 - R)
            w = cfg.W_MIN * R + t * (cfg.W_MAX - cfg.W_MIN) / T
        else:
            w = cfg.W_MAX - (cfg.W_MAX - cfg.W_MIN) * t / T

        # --- diversity-enhanced learning: spbest, Eqs. (14)-(15)
        if mech["spbest"]:
            a = rng.integers(0, N, N)
            b = (a + rng.integers(1, N, N)) % N                            # b != a
            c = np.where(pfit[a] < pfit[b], a, b)                          # cpbest
            use_c = pfit[c] < pfit
            spbest = np.where(use_c[:, None], pbest[c], pbest)
        else:
            spbest = pbest

        # --- adaptive mainstream learning: mbest, Eq. (16)
        mbest = pbest.mean(0) if mech["mbest"] else gbest

        # --- velocity, Eq. (17)
        r1, r2 = rng.random((N, n)), rng.random((N, n))
        V = w * V + r1 * cfg.C1 * (spbest - X) + r2 * cfg.C2 * (mbest - X)
        V = np.clip(V, -m / 2, m / 2)

        # --- adaptive position update, Eqs. (20)-(21)
        if mech["adaptive"]:
            cur_fit, _ = fitness_fn(env, _discretise(X, m))
            p = cur_fit / (np.mean(cur_fit) + 1e-12)
            use_eq19 = p < rng.random(N)          # Eq. 21: p < rand -> X + V
            Xnew = np.where(use_eq19[:, None], X + V, w * X + (1 - w) * V)
        else:
            Xnew = X + V
        X = _discretise(Xnew, m).astype(float)

        # --- end replacement of the worst particle, Eqs. (22)-(24)
        if mech["replace"]:
            i = rng.integers(0, N, N)
            j = (i + rng.integers(1, N, N)) % N
            nbest = _discretise((pbest[i] - pbest[j]) * rng.random((N, n)) + gbest, m)
            nfit, _ = fitness_fn(env, nbest)
            for k in range(N):
                worst = np.argmax(pfit)                                    # Eq. 22
                if nfit[k] < pfit[worst]:                                  # Eq. 24
                    pbest[worst], pfit[worst] = nbest[k], nfit[k]
                    X[worst] = nbest[k]

        # --- personal / global best update (Algorithm 1, lines 23-31)
        fit, _ = fitness_fn(env, X.astype(int))
        better = fit < pfit
        pbest[better], pfit[better] = X[better], fit[better]
        g = np.argmin(pfit)
        if pfit[g] < gfit: gbest, gfit = pbest[g].copy(), pfit[g]
        curve.append(gfit)
    return gbest.astype(int), np.array(curve)


def qa_qos_mdpso(env, seed=0, **kw):
    """OUR variant: identical search, queue-aware normalised fitness."""
    return qos_mdpso(env, fitness_fn=qa_fitness, seed=seed, **kw)


def fa_qos_mdpso(env, seed=0, **kw):
    """Extension 2: identical search, failure-aware + queue-aware fitness."""
    return qos_mdpso(env, fitness_fn=fa_fitness, seed=seed, **kw)


ALGORITHMS = {
    "Min-Min": lambda env, seed: min_min(env),
    "DPSO": lambda env, seed: dpso(env, seed=seed),
    "QoS-MDPSO": lambda env, seed: qos_mdpso(env, seed=seed),
    "QA-QoS-MDPSO (ours)": lambda env, seed: qa_qos_mdpso(env, seed=seed),
}
