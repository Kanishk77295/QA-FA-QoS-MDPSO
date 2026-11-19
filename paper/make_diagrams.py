import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
plt.rcParams.update({"font.size":9,"font.family":"serif","axes.grid":True,"grid.alpha":.3,"pdf.fonttype":42})
C={"Min-Min":"#777777","DPSO":"#3b6fa0","QoS-MDPSO":"#e8821c","QA (ours)":"#3a9a47","FA (ours)":"#a35c9e"}
def save(n): plt.tight_layout(); plt.savefig(n+".pdf"); plt.close()

# ---- data (Tables A1, A2, 8, 9, 10 of the project report)
L=[2,4,6,8,10,12,14,16]
mis_len={"Min-Min":[0.0,4.2,26.9,49.9,64.0,71.1,77.3,82.6],"DPSO":[12.7,35.5,46.8,56.9,66.8,72.0,77.3,80.6],
"QoS-MDPSO":[7.7,37.6,55.7,63.5,76.4,77.4,82.7,86.3],"QA (ours)":[0.9,22.5,32.4,55.9,63.5,70.9,77.5,81.9]}
ms_len={"Min-Min":[5.82,11.40,17.03,22.66,28.31,33.93,39.54,45.14],"DPSO":[17.25,33.80,45.23,62.06,74.39,89.09,99.81,121.49],
"QoS-MDPSO":[12.85,26.62,39.39,49.68,71.11,79.79,92.13,104.36],"QA (ours)":[11.81,22.81,37.60,49.64,62.19,79.38,86.32,105.83]}
N=[40,60,80,100,120]
mis_n={"Min-Min":[0.0,0.0,2.2,4.2,12.6],"DPSO":[0.9,6.5,23.7,35.5,46.7],"QoS-MDPSO":[1.9,8.8,28.0,37.6,57.0],"QA (ours)":[0.0,2.2,15.4,22.5,37.3]}
ms_n={"Min-Min":[4.98,6.60,9.91,11.40,13.71],"DPSO":[10.25,15.26,28.48,33.80,34.12],"QoS-MDPSO":[10.63,14.88,28.91,26.62,31.81],"QA (ours)":[8.42,11.18,24.05,22.81,29.11]}
def sweep(x,a,b,xl,name):
    f,ax=plt.subplots(1,2,figsize=(7.2,2.9))
    for k in a: ax[0].plot(x,a[k],marker="o",ms=3.5,color=C[k],label=k)
    for k in b: ax[1].plot(x,b[k],marker="o",ms=3.5,color=C[k],label=k)
    ax[0].set_ylabel("True deadline misses (of n)"); ax[1].set_ylabel("Makespan (h)")
    for A in ax: A.set_xlabel(xl)
    ax[0].legend(fontsize=7); save(name)
sweep(L,mis_len,ms_len,r"Average task length ($\times10^6$ MI)","fig_len")
sweep(N,mis_n,ms_n,"Number of tasks","fig_cnt")

# default setting bars
alg=["Min-Min","DPSO","QoS-MDPSO","QA (ours)"]
d={"Makespan (h)":[11.40,33.80,26.62,22.81],"Cost (\\$)":[50.18,52.66,50.04,49.7],"Credibility":[24.93,30.90,31.30,27.5],"True misses":[4.2,35.5,37.6,22.5]}
f,ax=plt.subplots(1,4,figsize=(7.4,2.5))
for a,(k,v) in zip(ax,d.items()):
    a.bar(range(4),v,color=[C[x] for x in alg]); a.set_title(k,fontsize=8); a.set_xticks(range(4)); a.set_xticklabels(["MM","DPSO","QoS","QA"],fontsize=7)
    for i,y in enumerate(v): a.text(i,y,f"{y:g}",ha="center",va="bottom",fontsize=6)
save("fig_default")

# lambda
lam=[0.5,1,2,5,10]; tm=[23.2,22.5,20.4,14.6,13.2]; tc=[27.33,27.50,26.75,25.06,25.99]
f,a=plt.subplots(figsize=(3.6,2.7)); a.plot(lam,tm,"o-",color=C["QA (ours)"],label="True misses"); a.set_xscale("log"); a.set_xlabel(r"Penalty weight $\lambda$"); a.set_ylabel("True deadline misses")
b=a.twinx(); b.plot(lam,tc,"s--",color=C["FA (ours)"],label="Total credibility"); b.set_ylabel("Total credibility"); b.grid(False)
h=a.get_legend_handles_labels(); h2=b.get_legend_handles_labels(); a.legend(h[0]+h2[0],h[1]+h2[1],fontsize=7,loc="lower left"); save("fig_lambda")

# ablation
v=["no chaotic w","no spbest","no mbest","no adaptive pos.","no worst repl.","all five off"]; c=[12.39,-5.98,5.27,16.50,13.14,22.54]
f,a=plt.subplots(figsize=(4.6,2.6)); a.barh(v[::-1],c[::-1],color=["#d9534f" if x>0 else "#3a9a47" for x in c[::-1]]); a.axvline(0,color="k",lw=.8)
a.set_xlabel("Change in paper fitness when removed (%)"); 
for i,x in enumerate(c[::-1]): a.text(x+(0.4 if x>0 else -0.4),i,f"{x:+.2f}",va="center",ha="left" if x>0 else "right",fontsize=7)
a.set_xlim(-10,27); save("fig_ablation")

# failures
done=[72.7,79.6,78.8,74.5,84.3]; lost=[21.6,16.4,16.6,20.2,12.3]; names=["Min-Min","DPSO","QoS-MDPSO","QA (ours)","FA (ours)"]
f,ax=plt.subplots(1,2,figsize=(6.4,2.6))
ax[0].bar(range(5),done,color=[C[x] for x in names]); ax[0].set_ylabel("Tasks completed (%)"); ax[0].set_ylim(60,90)
ax[1].bar(range(5),lost,color=[C[x] for x in names]); ax[1].set_ylabel("Tasks lost (of 100)")
for A,vv in zip(ax,[done,lost]):
    A.set_xticks(range(5)); A.set_xticklabels(["MM","DPSO","QoS","QA","FA"],fontsize=8)
    for i,y in enumerate(vv): A.text(i,y,f"{y:g}",ha="center",va="bottom",fontsize=7)
save("fig_fail")

# convergence is not available as numbers -> not drawn.

# worked example
f,ax=plt.subplots(figsize=(5.6,2.0))
ax.barh(1,2,left=0,color="#4a78a8",edgecolor="k"); ax.text(1,1,"T1",ha="center",va="center",color="w")
ax.barh(1,1,left=2,color="#e8821c",edgecolor="k"); ax.text(2.5,1,"T2",ha="center",va="center")
ax.barh(0,6,left=0,color="#3a9a47",edgecolor="k"); ax.text(3,0,"T3",ha="center",va="center",color="w")
ax.axvline(2.5,color="r",ls="--"); ax.text(2.55,1.45,"T2 deadline 2.5 s",color="r",fontsize=7)
ax.set_yticks([0,1]); ax.set_yticklabels(["VM2","VM1"]); ax.set_xlabel("Time (s)"); ax.set_ylim(-.5,1.8); ax.set_xlim(0,8); ax.grid(axis="y",alpha=0)
save("fig_example")

# boxes
def box(a,x,y,w,h,t,fc="#eef3f8",ec="#345",fs=7.5,ls="-"):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,ls=ls,lw=1)); a.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arrow(a,x1,y1,x2,y2): a.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",lw=1))
# loop flowchart
f,a=plt.subplots(figsize=(6.2,3.9)); a.set_xlim(0,10); a.set_ylim(0,10); a.axis("off")
steps=["Generate environment for seed s (tasks l,d,b,f; VMs MIPS,P,F)","Initialise N particles (random VM index per task, random velocity)","Evaluate fitness of every particle","Chaotic inertia weight w(t) (Eqs. 12-13); spbest, mbest (Eqs. 14-16)","Velocity (Eq. 17), adaptive position (Eq. 21); round and clip to VM indices","Replace worst personal best (Eqs. 22-24)","Update pbest and gbest","t < Tmax ?  (yes: loop back)"]
ys=[9.0,7.85,6.7,5.55,4.4,3.25,2.1,0.95]
for i,(s,y) in enumerate(zip(steps,ys)):
    fc="#d9efd9" if i==2 else "#eef3f8"
    box(a,0.6,y,5.6,0.8,s,fc=fc,ec="#2a2" if i==2 else "#345")
    if i<7: arrow(a,3.4,y,3.4,ys[i+1]+0.82)
box(a,6.9,6.3,2.9,1.5,"Fitness module\n(the only part that differs)\n- QoS-MDPSO: Eq. 11\n- QA: EDF completion times,\n  normalised, additive penalty\n- FA: QA + expected retries/loss",fc="#fff7dd",ec="#a80",fs=6.5,ls="--")
a.annotate("",xy=(6.2,7.1),xytext=(6.9,7.1),arrowprops=dict(arrowstyle="->",ls="--",lw=1,color="#a80"))
a.annotate("",xy=(0.6,5.95),xytext=(0.6,1.35),arrowprops=dict(arrowstyle="-",lw=0)); 
a.plot([0.6,0.25,0.25,0.6],[1.35,1.35,5.95,5.95],color="k",lw=1); arrow(a,0.25,5.95,0.62,5.95)
box(a,6.9,0.95,2.9,0.8,"Output gbest schedule",fc="#eee")
arrow(a,6.2,1.35,6.9,1.35)
save("fig_flow")

# architecture
f,a=plt.subplots(figsize=(6.4,2.9)); a.set_xlim(0,12); a.set_ylim(0,6); a.axis("off")
box(a,0.2,3.9,2.4,1.5,"config.py\nall parameters\n(PAPER / ASSUMED)",fs=7)
box(a,3.2,3.9,2.6,1.5,"environment.py\nseeded generator of\ntasks and VMs;\nqueue-aware evaluator",fs=7)
box(a,6.4,3.9,2.6,1.5,"fitness.py\npaper / QA / FA\nfitness functions",fs=7,fc="#fff7dd",ec="#a80")
box(a,9.6,3.9,2.2,1.5,"algorithms.py\nMin-Min, DPSO,\nQoS-MDPSO (+ablation)",fs=7)
box(a,3.2,1.5,2.6,1.5,"run_experiments.py\nsweeps, lambda,\nablation, convergence",fs=7)
box(a,6.4,1.5,2.6,1.5,"failures.py\nMonte Carlo failure\nsimulator (200 scenarios)",fs=7)
box(a,9.6,1.5,2.2,1.5,"analyze.py\nWilcoxon tests,\ntables, figures",fs=7,fc="#d9efd9",ec="#2a2")
for (x1,y1,x2,y2) in [(2.6,4.65,3.2,4.65),(5.8,4.65,6.4,4.65),(9.0,4.65,9.6,4.65),(10.7,3.9,10.7,3.0),(4.5,3.9,4.5,3.0),(5.8,2.25,6.4,2.25),(9.0,2.25,9.6,2.25)]: arrow(a,x1,y1,x2,y2)
a.text(6,0.6,"Seed s builds the same environment for every algorithm, so all comparisons are paired.",ha="center",fontsize=7.5,style="italic")
save("fig_arch")
