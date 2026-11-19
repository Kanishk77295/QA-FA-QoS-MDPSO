"""
config.py - every experimental parameter in one place.

PAPER = value taken from Ouyang & Zhang (2025), Table 3 / Section 4-5.
ASSUMED = the paper does not state it; we chose it and document it in the report.
"""

# ---------------- Environment (Table 3) ----------------
N_TASKS_DEFAULT = 100            # PAPER
AVG_LENGTH_DEFAULT = 4e6         # PAPER  (MI); l_i in [l-1e6, l+1e6]
LENGTH_SPREAD = 1e6              # PAPER
AVG_DEADLINE = 4e4               # PAPER  (s);  d_i in [d-1e4, d+1e4]
DEADLINE_SPREAD = 1e4            # PAPER
N_VMS = 12                       # PAPER
VM_MIPS = [100, 200, 400, 800, 1300, 2600]       # PAPER
VM_PRICE_PER_HOUR = [0.06, 0.12, 0.24, 0.48, 0.5, 1.0]  # PAPER
BUDGET_FACTOR = (0.8, 1.2)       # ASSUMED: b_i = U(0.8,1.2) x task's mean cost over the 6 VM templates

# ---------------- PSO (Section 4-5) ----------------
SWARM_SIZE = 30                  # ASSUMED (paper does not report N)
MAX_ITER = 200                   # ASSUMED (paper does not report T_max)
C1 = C2 = 2.0                    # PAPER
W_MAX, W_MIN = 0.9, 0.4          # PAPER
ALPHA = BETA = GAMMA = 1 / 3     # ASSUMED (paper only says alpha+beta+gamma=1)

# ---------------- Our contribution ----------------
LAMBDA_PENALTY = 1.0             # weight of the true-violation penalty in the queue-aware fitness

# ---------------- Experiments ----------------
SEEDS = list(range(10))                          # 10 independent runs per setting
TASK_LENGTHS = [2e6, 4e6, 6e6, 8e6, 10e6, 12e6, 14e6, 16e6]   # PAPER Fig. 4 x-axis
TASK_COUNTS = [40, 60, 80, 100, 120]                         # PAPER Fig. 5 x-axis
