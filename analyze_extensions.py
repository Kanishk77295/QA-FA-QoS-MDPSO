"""
analyze_extensions.py - tables, Wilcoxon tests and figures for the extensions.
Outputs: results/stats_extensions.txt, figures/ablation.png, figures/failures_*.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

out = open("results/stats_extensions.txt", "w")


def say(*a):
    print(*a); print(*a, file=out)


def wtest(df, key, a, b, col, by=("n_tasks", "seed")):
    x = df[df[key] == a].sort_values(list(by))[col].values
    y = df[df[key] == b].sort_values(list(by))[col].values
    d = x - y
    if np.all(d == 0):
        return np.median(d), 1.0
    return np.median(d), wilcoxon(x, y).pvalue


# ================================================================ ablation
ab = pd.read_csv("results/exp_ablation.csv")
VORDER = ["Full QoS-MDPSO", "no chaotic w", "no spbest", "no mbest",
          "no adaptive position", "no worst replacement", "all five off"]
g = ab.groupby("variant").mean(numeric_only=True).loc[VORDER]
say("===== ABLATION (task-count sweep 40-120, 10 seeds, paper fitness; mean over 50 runs) =====")
say(g[["paper_fitness", "makespan_h", "tec", "tc", "deadline_miss_true", "evals", "runtime_s"]]
    .round(3).to_string())
say("\nChange vs Full when the mechanism is REMOVED (positive fitness change = removal hurts)")
full = ab[ab.variant == "Full QoS-MDPSO"]
for v in VORDER[1:]:
    md, p = wtest(ab, "variant", v, "Full QoS-MDPSO", "paper_fitness")
    pct = (g.loc[v, "paper_fitness"] - g.loc["Full QoS-MDPSO", "paper_fitness"]) / \
        g.loc["Full QoS-MDPSO", "paper_fitness"] * 100
    say(f"   {v:22s} fitness {pct:+7.2f}%   median diff {md:+7.3f}   p = {p:.2e}")

fig, ax = plt.subplots(figsize=(8, 4.2))
base = g.loc["Full QoS-MDPSO", "paper_fitness"]
vals = (g["paper_fitness"] - base) / base * 100
cols = ["#888" if v == "Full QoS-MDPSO" else ("#E45756" if vals[v] > 0 else "#54A24B") for v in VORDER]
ax.barh(range(len(VORDER)), vals.values, color=cols)
ax.set_yticks(range(len(VORDER))); ax.set_yticklabels(VORDER); ax.invert_yaxis()
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Change in paper fitness vs full QoS-MDPSO (%)  →  removal hurts")
ax.set_title("Ablation: effect of removing each mechanism (50 paired runs)")
ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig("figures/ablation.png", dpi=150)
plt.close(fig)

# ================================================================ failures
fl = pd.read_csv("results/exp_failures.csv")
AORDER = ["Min-Min", "DPSO", "QoS-MDPSO", "QA-QoS-MDPSO (ours)", "FA-QoS-MDPSO (ours, failure-aware)"]
SHORT = ["Min-Min", "DPSO", "QoS-MDPSO", "QA (ours)", "FA (ours)"]
COL = ["#9a9a9a", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]
COLS = ["tc", "deadline_miss_true", "success_rate", "crit_weighted_success", "lost_tasks",
        "lost_critical", "eff_makespan_h", "eff_cost", "eff_deadline_miss"]
for sc in ["paper", "reliable"]:
    d = fl[fl.scenario == sc]
    say(f"\n===== FAILURES, scenario '{sc}' (task-count sweep, 10 seeds, 200 MC each; mean over 50) =====")
    say(d.groupby("algorithm").mean(numeric_only=True).loc[AORDER][COLS].round(3).to_string())
    say("-- Wilcoxon vs QoS-MDPSO (paired by n_tasks, seed)")
    for a in AORDER[3:]:
        for c in ["success_rate", "lost_tasks", "eff_makespan_h", "eff_cost", "eff_deadline_miss"]:
            md, p = wtest(d, "algorithm", a, "QoS-MDPSO", c)
            say(f"   {a[:12]:12s} {c:20s} median diff {md:+8.3f}  p = {p:.2e}")
    say("-- Correlation across all runs: TC vs success_rate  r = "
        f"{np.corrcoef(d.tc / d.n_tasks, d.success_rate)[0, 1]:.3f} (TC per task)")

    gm = d.drop(columns="scenario").groupby("algorithm").agg(["mean", "std"]).loc[AORDER]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (c, t) in zip(axes, [("success_rate", "Tasks completed (share) ↑"),
                                 ("lost_tasks", "Tasks lost after 3 attempts ↓"),
                                 ("eff_makespan_h", "Makespan with re-runs (h) ↓"),
                                 ("eff_deadline_miss", "Deadline missed or lost ↓")]):
        ax.bar(range(5), gm[c]["mean"], yerr=gm[c]["std"], color=COL, capsize=3)
        ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=25, ha="right", fontsize=9)
        ax.set_title(t, fontsize=11); ax.grid(axis="y", alpha=.3)
    fig.suptitle(f"Under VM failures — credibility range "
                 f"{'U(0,1), paper' if sc=='paper' else 'U(0.9,1), reliable VMs'} "
                 f"(mean ± std, 50 runs × 200 scenarios)")
    fig.tight_layout(); fig.savefig(f"figures/failures_{sc}.png", dpi=150); plt.close(fig)

# TC vs success scatter (paper scenario)
d = fl[fl.scenario == "paper"]
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for a, s, c in zip(AORDER, SHORT, COL):
    x = d[d.algorithm == a]
    ax.scatter(x.tc / x.n_tasks, x.success_rate, s=16, color=c, label=s, alpha=.75)
ax.set_xlabel("Total credibility per task (TC / n)"); ax.set_ylabel("Share of tasks completed")
ax.set_title("Does credibility predict surviving failures?"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/failures_tc_vs_success.png", dpi=150); plt.close(fig)
out.close()
