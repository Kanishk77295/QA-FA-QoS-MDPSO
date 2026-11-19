"""
environment.py - builds a synthetic cloud (tasks + VMs) and evaluates schedules.

A schedule ("position") is an integer vector of length n: pos[i] = VM index of task i.
evaluate() works on a whole batch of schedules at once (shape B x n) for speed.
"""
import numpy as np
import config as cfg


class CloudEnv:
    def __init__(self, n=cfg.N_TASKS_DEFAULT, avg_len=cfg.AVG_LENGTH_DEFAULT,
                 m=cfg.N_VMS, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.m = n, m

        # ---- tasks: t_i = {l_i, d_i, b_i, f_i}  (Section 3.1) ----
        self.l = rng.uniform(avg_len - cfg.LENGTH_SPREAD, avg_len + cfg.LENGTH_SPREAD, n)
        self.d = rng.uniform(cfg.AVG_DEADLINE - cfg.DEADLINE_SPREAD,
                             cfg.AVG_DEADLINE + cfg.DEADLINE_SPREAD, n)
        self.f = rng.uniform(0, 1, n)

        # ---- VMs: vm_j = {MIPS_j, P_j, F_j}; template picked at random ----
        tpl = rng.integers(0, len(cfg.VM_MIPS), m)
        self.mips = np.array(cfg.VM_MIPS, float)[tpl]
        self.price = np.array(cfg.VM_PRICE_PER_HOUR, float)[tpl]
        self.F = rng.uniform(0, 1, m)

        # ---- matrices E (seconds) and C (dollars): e_ij = l_i/MIPS_j, c_ij = e_ij*P_j ----
        self.E = self.l[:, None] / self.mips[None, :]
        self.C = self.E * self.price[None, :] / 3600.0          # price is per hour

        # ---- budget (ASSUMED, paper gives no range) ----
        tpl_cost = self.l[:, None] * (np.array(cfg.VM_PRICE_PER_HOUR) /
                                      np.array(cfg.VM_MIPS))[None, :] / 3600.0
        self.b = tpl_cost.mean(1) * rng.uniform(*cfg.BUDGET_FACTOR, n)

        # ---- reference values used by OUR normalised fitness ----
        self.ms_lb = self.l.sum() / self.mips.sum() / 3600.0     # perfect-split makespan (h)
        self.tec_lb = self.C.min(1).sum()                       # every task on its cheapest VM
        self.tc_ub = (self.f * self.F.max()).sum()              # every task on most credible VM

        # EDF execution order inside each VM (earliest deadline first)
        self.edf = np.argsort(self.d, kind="stable")

    def evaluate(self, P, mult=None):
        """P: (B, n) int array. Returns dict of per-schedule metrics (arrays of length B).
        mult: optional (n, m) multiplier on time and cost (used for expected retries)."""
        P = np.atleast_2d(P)
        B, n, m = P.shape[0], self.n, self.m
        rows = np.arange(n)
        E = self.E if mult is None else self.E * mult
        C = self.C if mult is None else self.C * mult

        T = E[rows, P]                                          # (B,n) exec time of each task
        Cst = C[rows, P]

        key = (np.arange(B)[:, None] * m + P).ravel()           # (schedule, vm) group id
        loads = np.bincount(key, weights=T.ravel(), minlength=B * m).reshape(B, m)
        makespan_h = loads.max(1) / 3600.0                      # Eq. 1 (in hours)
        tec = Cst.sum(1)                                        # Eq. 2
        tc = (self.f[None, :] * self.F[P]).sum(1)               # Eq. 3

        md_paper = (T > self.d[None, :]).sum(1)                 # Eq. 9 as written: e_ij <= d_i
        mb = (Cst > self.b[None, :]).sum(1)                     # Eq. 10

        # ---- TRUE deadline misses: tasks queue on their VM, run in EDF order ----
        Po = P[:, self.edf]
        To = T[:, self.edf]
        do = np.broadcast_to(self.d[self.edf], (B, n))
        k = (np.arange(B)[:, None] * m + Po).ravel()
        idx = np.argsort(k, kind="stable")
        ks, ts, ds = k[idx], To.ravel()[idx], do.ravel()[idx]
        cs = np.cumsum(ts)
        start = np.r_[True, ks[1:] != ks[:-1]]
        base = np.maximum.accumulate(np.where(start, np.arange(len(ks)), 0))
        finish = cs - (cs[base] - ts[base])                     # completion time incl. waiting
        late = finish > ds
        md_true = np.bincount(ks[late] // m, minlength=B)

        return dict(makespan=makespan_h, tec=tec, tc=tc,
                    md_paper=md_paper, mb=mb, md_true=md_true)
