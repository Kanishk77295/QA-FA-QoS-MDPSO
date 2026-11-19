"""
analyze.py - summary tables, % improvements, Wilcoxon tests and figures.
Outputs: results/summary_*.csv, results/stats.txt, figures/*.png
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

os.makedirs("figures", exist_ok=True)
ORDER = ["Min-Min", "DPSO", "QoS-MDPSO", "QA-QoS-MDPSO (ours)"]
COL = {"Min-Min": "#9a9a9a", "DPSO": "#4C78A8", "QoS-MDPSO": "#F58518",
       "QA-QoS-MDPSO (ours)": "#54A24B"}
MK = {"Min-Min": "^", "DPSO": "o", "QoS-MDPSO": "s", "QA-QoS-MDPSO (ours)": "D"}
METRICS = [("makespan_h", "Makespan (hours) ↓"), ("tec", "Total execution cost ($) ↓"),
           ("tc", "Total credibility ↑"), ("deadline_miss_true", "True deadline misses ↓"),
           ("deadline_miss_paper", "Deadline misses counted by paper's Eq. 9"),
           ("budget_miss", "Budget violations ↓")]
out = open("results/stats.txt", "w")


def say(*a):
    print(*a); print(*a, file=out)


def pct(new, old, higher_better=False):
    return (new - old) / old * 100 * (1 if higher_better else -1)


for exp, xcol, xlabel, xscale in [("length", "task_length", "Average task length (×10⁶ MI)", 1e6),
                                  ("tasks", "n_tasks", "Number of tasks", 1)]:
    df = pd.read_csv(f"results/exp_{exp}.csv")
    summ = df.groupby([xcol, "algorithm"]).mean(numeric_only=True).reset_index()
    summ.to_csv(f"results/summary_{exp}.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (mcol, title) in zip(axes.ravel(), METRICS):
        for a in ORDER:
            s = summ[summ.algorithm == a]
            ax.plot(s[xcol] / xscale, s[mcol], marker=MK[a], color=COL[a], label=a, lw=2)
        ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Experiment: varying {'task length' if exp=='length' else 'number of tasks'} "
                 f"(mean of 10 seeds)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"figures/exp_{exp}.png", dpi=150); plt.close(fig)

    say(f"\n===== {exp.upper()}: mean over all settings and seeds =====")
    g = df.groupby("algorithm").mean(numeric_only=True).loc[ORDER]
    say(g[["makespan_h", "tec", "tc", "deadline_miss_true", "deadline_miss_paper",
           "budget_miss", "paper_fitness", "runtime_s"]].round(3).to_string())

    # paper-style headline: QoS-MDPSO vs DPSO, averaged per setting
    p = summ.pivot(index=xcol, columns="algorithm")
    say(f"\n-- QoS-MDPSO vs DPSO (avg % improvement over settings; paper reports "
        f"{'19.69 / 4.39 / 7.04' if exp=='length' else '14.08 / 2.74 / 2.89'} for MS/TEC/TC)")
    for m, hb in [("makespan_h", 0), ("tec", 0), ("tc", 1)]:
        say(f"   {m:12s} {pct(p[m]['QoS-MDPSO'], p[m]['DPSO'], hb).mean():7.2f} %")
    say("-- OURS vs QoS-MDPSO (avg % improvement over settings)")
    for m, hb in [("makespan_h", 0), ("tec", 0), ("tc", 1), ("deadline_miss_true", 0),
                  ("budget_miss", 0)]:
        say(f"   {m:20s} {pct(p[m]['QA-QoS-MDPSO (ours)'], p[m]['QoS-MDPSO'], hb).mean():7.2f} %")

    # Wilcoxon signed-rank: ours vs QoS-MDPSO, paired by (setting, seed)
    say("-- Wilcoxon signed-rank, OURS vs QoS-MDPSO, paired runs (n="
        f"{df[xcol].nunique()*df.seed.nunique()})")
    a = df[df.algorithm == "QA-QoS-MDPSO (ours)"].sort_values([xcol, "seed"])
    b = df[df.algorithm == "QoS-MDPSO"].sort_values([xcol, "seed"])
    for m in ["makespan_h", "tec", "tc", "deadline_miss_true", "budget_miss"]:
        st = wilcoxon(a[m].values, b[m].values)
        say(f"   {m:20s} median diff {np.median(a[m].values-b[m].values):8.3f}  p = {st.pvalue:.2e}")

# lambda sensitivity
lam = pd.read_csv("results/exp_lambda.csv").groupby("lam").mean(numeric_only=True)
say("\n===== LAMBDA SENSITIVITY (default setting, 10 seeds) =====")
say(lam[["makespan_h", "tec", "tc", "deadline_miss_true", "budget_miss"]].round(2).to_string())
fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.plot(lam.index, lam.deadline_miss_true, "D-", color=COL["QA-QoS-MDPSO (ours)"], lw=2,
         label="True deadline misses")
ax1.set_xscale("log"); ax1.set_xlabel("Penalty weight λ"); ax1.set_ylabel("True deadline misses")
ax2 = ax1.twinx()
ax2.plot(lam.index, lam.tc, "o--", color="#B279A2", lw=2, label="Total credibility")
ax2.set_ylabel("Total credibility")
fig.legend(loc="upper center", ncol=2, fontsize=9); ax1.grid(alpha=.3)
ax1.set_title("Trade-off: deadline penalty vs credibility", pad=28)
fig.tight_layout(); fig.savefig("figures/lambda_tradeoff.png", dpi=150); plt.close(fig)

# convergence
cv = pd.read_csv("results/convergence.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
for a in ["DPSO", "QoS-MDPSO"]:
    c = cv[cv.algorithm == a].groupby("iteration").best_fitness.mean()
    ax.plot(c.index, c.values, color=COL[a], lw=2, label=a)
ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness (paper Eq. 11) ↓")
ax.set_title("Convergence (default setting, mean of 5 seeds)"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/convergence.png", dpi=150); plt.close(fig)

# headline bar chart: default setting n=100, l=4e6
d = pd.read_csv("results/exp_length.csv")
d = d[d.task_length == 4e6].groupby("algorithm").agg(["mean", "std"]).loc[ORDER]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (m, t) in zip(axes, [("makespan_h", "Makespan (h) ↓"), ("tec", "Cost ($) ↓"),
                             ("tc", "Credibility ↑"), ("deadline_miss_true", "True deadline misses ↓")]):
    ax.bar(range(4), d[m]["mean"], yerr=d[m]["std"], color=[COL[a] for a in ORDER], capsize=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Min-Min", "DPSO", "QoS-\nMDPSO", "Ours"])
    ax.set_title(t); ax.grid(axis="y", alpha=.3)
fig.suptitle("Default setting (n=100, l=4×10⁶ MI): mean ± std over 10 seeds")
fig.tight_layout(); fig.savefig("figures/default_setting.png", dpi=150); plt.close(fig)
say("\nDefault setting means:\n" + d.xs("mean", axis=1, level=1)[
    ["makespan_h", "tec", "tc", "deadline_miss_true", "deadline_miss_paper", "budget_miss"]].round(2).to_string())
out.close()
