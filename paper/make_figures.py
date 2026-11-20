import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U="../results/"
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.grid":True,"grid.alpha":.3,"pdf.fonttype":42})
C={"Min-Min":"#777777","DPSO":"#3b6fa0","QoS-MDPSO":"#e8821c","QA-QoS-MDPSO (ours)":"#3a9a47","FA-QoS-MDPSO (ours, failure-aware)":"#a35c9e"}
M={"Min-Min":"^","DPSO":"o","QoS-MDPSO":"s","QA-QoS-MDPSO (ours)":"D"}
SH={"Min-Min":"Min-Min","DPSO":"DPSO","QoS-MDPSO":"QoS-MDPSO","QA-QoS-MDPSO (ours)":"QA (ours)","FA-QoS-MDPSO (ours, failure-aware)":"FA (ours)"}
def save(n): plt.tight_layout(); plt.savefig(n+".pdf"); plt.close()
L=pd.read_csv(U+"exp_length.csv"); T=pd.read_csv(U+"exp_tasks.csv")
print(L.algorithm.unique())
def sweep(df,x,xl,name,xs=1):
    cols=[("makespan_h","Makespan (h)"),("tec","Total cost (\\$)"),("tc","Total credibility"),("deadline_miss_true","True deadline misses"),("deadline_miss_paper","Misses counted by Eq. 9"),("budget_miss","Budget violations")]
    f,ax=plt.subplots(2,3,figsize=(7.4,4.6))
    g=df.groupby(["algorithm",x]).mean(numeric_only=True).reset_index()
    for a,(c,t) in zip(ax.ravel(),cols):
        for k in M:
            d=g[g.algorithm==k]; a.plot(d[x]/xs,d[c],marker=M[k],ms=3.5,color=C[k],label=SH[k])
        a.set_title(t,fontsize=8); a.set_xlabel(xl,fontsize=7)
    ax[0,0].legend(fontsize=6); save(name)
sweep(L,"task_length",r"Average task length ($\times10^6$ MI)","fig_len",1e6)
sweep(T,"n_tasks","Number of tasks","fig_cnt")
# default
D=T[T.n_tasks==100]; al=list(M)
cols=[("makespan_h","Makespan (h)"),("tec","Cost (\\$)"),("tc","Credibility"),("deadline_miss_true","True deadline misses")]
f,ax=plt.subplots(1,4,figsize=(7.4,2.5))
for a,(c,t) in zip(ax,cols):
    m=[D[D.algorithm==k][c].mean() for k in al]; s=[D[D.algorithm==k][c].std() for k in al]
    a.bar(range(4),m,yerr=s,capsize=2,color=[C[k] for k in al]); a.set_title(t,fontsize=8); a.set_xticks(range(4)); a.set_xticklabels(["MM","DPSO","QoS","QA"],fontsize=7)
save("fig_default")
# convergence
V=pd.read_csv(U+"convergence.csv"); g=V.groupby(["algorithm","iteration"]).best_fitness.mean().reset_index()
f,a=plt.subplots(figsize=(4.4,2.8))
for k,c in [("DPSO",C["DPSO"]),("QoS-MDPSO",C["QoS-MDPSO"])]:
    d=g[g.algorithm==k]; a.plot(d.iteration,d.best_fitness,color=c,label=k,lw=1.6)
a.set_xlabel("Iteration"); a.set_ylabel("Best paper fitness"); a.legend(); save("fig_conv")
print(g.groupby("algorithm").best_fitness.agg(["first","last"]), g[(g.algorithm=="QoS-MDPSO")&(g.iteration==35)])
# lambda
Lm=pd.read_csv(U+"exp_lambda.csv").groupby("lam").mean(numeric_only=True).reset_index()
f,a=plt.subplots(figsize=(3.8,2.7)); a.plot(Lm.lam,Lm.deadline_miss_true,"D-",color=C["QA-QoS-MDPSO (ours)"],label="True misses"); a.set_xscale("log"); a.set_xlabel(r"Penalty weight $\lambda$"); a.set_ylabel("True deadline misses")
b=a.twinx(); b.plot(Lm.lam,Lm.tc,"o--",color="#a35c9e",label="Total credibility"); b.set_ylabel("Total credibility"); b.grid(False)
h=a.get_legend_handles_labels(); h2=b.get_legend_handles_labels(); a.legend(h[0]+h2[0],h[1]+h2[1],fontsize=6.5,loc="lower left"); save("fig_lambda")
# ablation
A=pd.read_csv(U+"exp_ablation.csv"); m=A.groupby("variant").paper_fitness.mean(); full=m["Full QoS-MDPSO"]
order=["no chaotic w","no spbest","no mbest","no adaptive position","no worst replacement","all five off"]
ch=[(m[v]/full-1)*100 for v in order]
f,a=plt.subplots(figsize=(4.6,2.6)); a.barh(order[::-1],ch[::-1],color=["#d9534f" if x>0 else "#3a9a47" for x in ch[::-1]]); a.axvline(0,color="k",lw=.8)
a.set_xlabel("Change in paper fitness when removed (%)"); a.set_xlim(-10,27)
for i,x in enumerate(ch[::-1]): a.text(x+(0.4 if x>0 else -0.4),i,f"{x:+.2f}",va="center",ha="left" if x>0 else "right",fontsize=7)
save("fig_ablation"); print(ch)
# failures
F=pd.read_csv(U+"exp_failures.csv"); order=list(SH)
def fb(scn,name,ttl):
    d=F[F.scenario==scn]; cols=[("success_rate","Tasks completed (share)"),("lost_tasks","Tasks lost after 3 attempts"),("eff_makespan_h","Makespan with re-runs (h)"),("eff_deadline_miss","Deadline missed or lost")]
    f,ax=plt.subplots(1,4,figsize=(7.4,2.6))
    for a,(c,t) in zip(ax,cols):
        m=[d[d.algorithm==k][c].mean() for k in order]; s=[d[d.algorithm==k][c].std() for k in order]
        a.bar(range(5),m,yerr=s,capsize=2,color=[C[k] for k in order]); a.set_title(t,fontsize=7.5); a.set_xticks(range(5)); a.set_xticklabels(["MM","DPSO","QoS","QA","FA"],fontsize=7)
    save(name)
fb("paper","fig_fail",""); fb("reliable","fig_rel","")
d=F[F.scenario=="paper"]
f,a=plt.subplots(figsize=(4.4,3.0))
for k in order:
    q=d[d.algorithm==k]; a.scatter(q.tc/q.n_tasks,q.success_rate,s=10,alpha=.7,color=C[k],label=SH[k])
a.set_xlabel("Total credibility per task (TC / n)"); a.set_ylabel("Share of tasks completed"); a.legend(fontsize=6.5)
print(np.corrcoef(d.tc/d.n_tasks,d.success_rate)[0,1]); save("fig_scatter")
