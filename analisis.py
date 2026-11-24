#!/usr/bin/env python3
"""Analisis estadistico de results.csv (anteproyecto: Optimizacion del calculo de
matrices inversas mediante procesos e hilos jerarquizados).

Uso:  python3 analisis.py [results.csv] [carpeta_salida]
Genera tablas CSV, figuras PNG y resumen.txt en la carpeta de salida (por defecto 'resultados/').
"""
import sys, os, itertools, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "resultados"
ALPHA = 0.05
os.makedirs(OUT, exist_ok=True)
LOG = []
def say(s=""):
    print(s); LOG.append(str(s))

# ------------------------------------------------------------------ carga
d = pd.read_csv(CSV)
d = d[d.status == "ok"].copy()
d["cfg"] = d.apply(lambda r: "seq" if r.version == "seq" else f"{r.version}-P{int(r.P)}", axis=1)
d["logt"] = np.log(d.time_s)
NS = sorted(d.n.unique())
CFGS = ["seq"] + [f"{v}-P{p}" for v in ["threads", "procs", "hier"] for p in [2, 4, 8]]
VERS = ["threads", "procs", "hier"]
PS = sorted(d[d.version != "seq"].P.unique())

say("=" * 70); say("1. VALIDACION DE CORRECCION"); say("=" * 70)
chk = d[d.residual >= 0]
say(f"Corridas totales: {len(d)}; con residuo calculado: {len(chk)}")
say(f"Residuo maximo: {chk.residual.max():.3e}  (umbral 1e-8) -> "
    f"{'CUMPLE' if chk.residual.max() < 1e-8 else 'NO CUMPLE'}")
say(f"Configuraciones con residuo verificado: {chk.groupby(['n','cfg']).ngroups} de {d.groupby(['n','cfg']).ngroups}")

# ------------------------------------------------------------------ resumen
def ci95(x):
    x = np.asarray(x); k = len(x)
    if k < 2: return (np.nan, np.nan)
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf(0.975, k - 1)
    return (m - h, m + h)

rows = []
for (n, cfg), g in d.groupby(["n", "cfg"]):
    lo, hi = ci95(g.time_s)
    rows.append(dict(n=n, cfg=cfg, version=g.version.iloc[0], P=1 if cfg == "seq" else int(g.P.iloc[0]),
                     reps=len(g), media=g.time_s.mean(), mediana=g.time_s.median(), desv=g.time_s.std(),
                     ic95_lo=lo, ic95_hi=hi, cv_pct=100 * g.time_s.std() / g.time_s.mean(),
                     rss_mb=g.maxrss_kb.median() / 1024, ctx_vol=g.vol_ctx.median(), ctx_invol=g.invol_ctx.median()))
R = pd.DataFrame(rows)
T1 = R[R.cfg == "seq"].set_index("n").media
R["speedup"] = R.apply(lambda r: T1[r.n] / r.media, axis=1)
R["eficiencia"] = R.speedup / R.P
R["cfg"] = pd.Categorical(R.cfg, CFGS, ordered=True)
R = R.sort_values(["n", "cfg"]).reset_index(drop=True)
R.to_csv(f"{OUT}/tabla_resumen.csv", index=False)

say(); say("=" * 70); say("2. RESUMEN (media, IC95%, aceleracion S=T1/TP, eficiencia E=S/P)"); say("=" * 70)
for n in NS:
    say(f"\n--- n = {n} ---")
    t = R[R.n == n][["cfg", "reps", "media", "ic95_lo", "ic95_hi", "cv_pct", "speedup", "eficiencia"]]
    say(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ------------------------------------------------------------------ supuestos
say(); say("=" * 70); say("3. SUPUESTOS (sobre log(tiempo))"); say("=" * 70)
sw = []
for (n, cfg), g in d.groupby(["n", "cfg"]):
    if len(g) >= 3:
        sw.append(dict(n=n, cfg=cfg, p=stats.shapiro(g.logt).pvalue))
SW = pd.DataFrame(sw); SW.to_csv(f"{OUT}/shapiro.csv", index=False)
frac_norm = (SW.p > ALPHA).mean()
say(f"Shapiro-Wilk: {int((SW.p > ALPHA).sum())} de {len(SW)} configuraciones son compatibles con normalidad (p>{ALPHA}) -> {100*frac_norm:.0f}%")
lev = {}
for n in NS:
    groups = [g.logt.values for _, g in d[d.n == n].groupby("cfg")]
    lev[n] = stats.levene(*groups, center="median").pvalue
    say(f"Levene (homogeneidad de varianzas) n={n}: p = {lev[n]:.3g} -> {'se cumple' if lev[n] > ALPHA else 'NO se cumple'}")
normal_ok = frac_norm >= 0.95 and all(p > ALPHA for p in lev.values())
say(f"\nDecision: {'se cumplen los supuestos -> ANOVA + Tukey' if normal_ok else 'NO se cumplen los supuestos -> Kruskal-Wallis + Mann-Whitney con Holm (los ANOVA se reportan solo como referencia)'}")

# ------------------------------------------------------------------ omnibus
say(); say("=" * 70); say("4. PRUEBAS OMNIBUS"); say("=" * 70)
om = []
for n in NS:
    g = d[d.n == n]
    groups = [x.logt.values for _, x in g.groupby("cfg")]
    kw = stats.kruskal(*groups)
    f = stats.f_oneway(*groups)
    om.append(dict(n=n, kruskal_H=kw.statistic, kruskal_p=kw.pvalue, anova_F=f.statistic, anova_p=f.pvalue))
    say(f"n={n}: Kruskal-Wallis H={kw.statistic:.1f} p={kw.pvalue:.3g} | ANOVA 1 via F={f.statistic:.1f} p={f.pvalue:.3g}")
pd.DataFrame(om).to_csv(f"{OUT}/omnibus.csv", index=False)

say("\nANOVA factorial sobre log(tiempo), solo versiones paralelas: n x version x P")
par = d[d.version != "seq"].copy()
par["n_"] = par.n.astype(str); par["P_"] = par.P.astype(int).astype(str)
mod = ols("logt ~ C(n_) * C(version) * C(P_)", data=par).fit()
an = sm.stats.anova_lm(mod, typ=2)
an["eta2_parcial"] = an["sum_sq"] / (an["sum_sq"] + an.loc["Residual", "sum_sq"])
an.to_csv(f"{OUT}/anova_factorial.csv")
say(an.to_string(float_format=lambda x: f"{x:.4g}"))

# ------------------------------------------------------------------ hipotesis principal
say(); say("=" * 70)
say("5. HIPOTESIS PRINCIPAL  H0: T(hier) = T(otra)   H1: T(hier) < T(otra)   (alfa=0.05, Holm)"); say("=" * 70)
def cliff(a, b):  # prob(a<b) - prob(a>b): + => 'a' mas rapido
    a = np.asarray(a); b = np.asarray(b)
    lt = (a[:, None] < b[None, :]).sum(); gt = (a[:, None] > b[None, :]).sum()
    return (lt - gt) / (len(a) * len(b))
def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (b.mean() - a.mean()) / sp
H = []
for n in NS:
    g = d[d.n == n]
    for P in PS:
        h = g[g.cfg == f"hier-P{P}"].time_s.values
        for other in [f"threads-P{P}", f"procs-P{P}", "seq"]:
            o = g[g.cfg == other].time_s.values
            mw = stats.mannwhitneyu(h, o, alternative="less")
            tw = stats.ttest_ind(np.log(h), np.log(o), equal_var=False, alternative="less")
            H.append(dict(n=n, P=P, vs=other, mediana_hier=np.median(h), mediana_otra=np.median(o),
                          cambio_pct=100 * (np.median(h) / np.median(o) - 1),
                          p_mw=mw.pvalue, p_welch_log=tw.pvalue, cliff_delta=cliff(h, o), d_cohen=cohen_d(np.log(h), np.log(o))))
H = pd.DataFrame(H)
H["p_holm"] = multipletests(H.p_mw, method="holm")[1]
H["signif"] = np.where(H.p_holm < ALPHA, "SI (H1)", "no")
H.to_csv(f"{OUT}/hipotesis_hier.csv", index=False)
for vs in ["seq", "threads", "procs"]:
    say(f"\n--- hier vs {vs} ---")
    t = H[H.vs.str.startswith(vs)][["n", "P", "mediana_hier", "mediana_otra", "cambio_pct", "p_holm", "cliff_delta", "signif"]]
    say(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# Sentido contrario: cuando la otra version es mas rapida
say("\nComparacion en sentido contrario (H1': T(otra) < T(hier)) para hilos y procesos, con Holm:")
Hc = []
for n in NS:
    g = d[d.n == n]
    for P in PS:
        h = g[g.cfg == f"hier-P{P}"].time_s.values
        for v in ["threads", "procs"]:
            o = g[g.cfg == f"{v}-P{P}"].time_s.values
            Hc.append(dict(n=n, P=P, otra=v, p=stats.mannwhitneyu(o, h, alternative="less").pvalue))
Hc = pd.DataFrame(Hc); Hc["p_holm"] = multipletests(Hc.p, method="holm")[1]
Hc["otra_mas_rapida"] = np.where(Hc.p_holm < ALPHA, "SI", "no")
Hc.to_csv(f"{OUT}/hipotesis_contraria.csv", index=False)
say(Hc.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ------------------------------------------------------------------ Amdahl
say(); say("=" * 70); say("6. AMDAHL: f estimada con P=2 -> aceleracion predicha para P=4 y 8 vs observada"); say("=" * 70)
A = []
for n in NS:
    for v in VERS:
        S = {int(r.P): r.speedup for _, r in R[(R.n == n) & (R.version == v)].iterrows()}
        f2 = min(max((1 - 1 / S[2]) / (1 - 1 / 2), 0), 1)
        for P in [4, 8]:
            pred = 1 / ((1 - f2) + f2 / P)
            A.append(dict(n=n, version=v, f_est_P2=f2, P=P, S_pred=pred, S_obs=S[P], obs_sobre_pred=S[P] / pred))
A = pd.DataFrame(A); A.to_csv(f"{OUT}/amdahl.csv", index=False)
say(A.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ------------------------------------------------------------------ figuras
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})
col = {"seq": "k", "threads": "tab:blue", "procs": "tab:orange", "hier": "tab:green"}

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for v in ["seq"] + VERS:
    for P in ([1] if v == "seq" else [4]):
        t = R[(R.version == v) & (R.P == P)]
        ax.errorbar(t.n, t.media, yerr=[t.media - t.ic95_lo, t.ic95_hi - t.media], marker="o", capsize=3, color=col[v],
                    label="seq" if v == "seq" else f"{v} (P=4)")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("Tiempo de pared (s)")
ax.set_title("Tiempo vs tamano (P=4, media e IC 95%)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_tiempo_vs_n.png"); plt.close(fig)

fig, axs = plt.subplots(1, len(NS), figsize=(4 * len(NS), 3.8), sharey=True)
for ax, n in zip(np.atleast_1d(axs), NS):
    for v in VERS:
        t = R[(R.n == n) & (R.version == v)]
        ax.plot(t.P, t.speedup, marker="o", color=col[v], label=v)
    ax.plot(PS, PS, "k--", lw=.8, label="ideal")
    ax.set_title(f"n = {n}"); ax.set_xlabel("P"); ax.set_xticks(PS)
np.atleast_1d(axs)[0].set_ylabel("Aceleracion S = T1/TP"); np.atleast_1d(axs)[0].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_aceleracion.png"); plt.close(fig)

fig, axs = plt.subplots(1, len(NS), figsize=(4 * len(NS), 3.8), sharey=True)
for ax, n in zip(np.atleast_1d(axs), NS):
    for v in VERS:
        t = R[(R.n == n) & (R.version == v)]
        ax.plot(t.P, t.eficiencia, marker="o", color=col[v], label=v)
    ax.axhline(1, color="k", ls="--", lw=.8); ax.set_title(f"n = {n}"); ax.set_xlabel("P"); ax.set_xticks(PS)
np.atleast_1d(axs)[0].set_ylabel("Eficiencia E = S/P"); np.atleast_1d(axs)[0].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_eficiencia.png"); plt.close(fig)

big = [n for n in NS if n >= 2048]
fig, axs = plt.subplots(1, max(len(big), 1), figsize=(5.2 * max(len(big), 1), 4), squeeze=False)
for ax, n in zip(axs[0], big):
    order = [c for c in CFGS if c in set(d[d.n == n].cfg)]
    ax.boxplot([d[(d.n == n) & (d.cfg == c)].time_s for c in order], showfliers=True)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=70)
    ax.set_title(f"Distribucion del tiempo, n = {n}"); ax.set_ylabel("s")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_boxplot.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, (colname, ttl) in zip(axs, [("rss_mb", "Memoria maxima (RSS, MB)"), ("ctx_invol", "Cambios de contexto involuntarios (mediana)")]):
    for v in ["seq"] + VERS:
        t = R[(R.version == v) & ((R.P == 4) | (v == "seq"))]
        ax.plot(t.n, t[colname], marker="o", color=col[v], label="seq" if v == "seq" else f"{v} (P=4)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(ttl); ax.set_xlabel("n"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_memoria_contexto.png"); plt.close(fig)

fig, axs = plt.subplots(1, len(NS), figsize=(4 * len(NS), 3.8), sharey=True)
for ax, n in zip(np.atleast_1d(axs), NS):
    for v in VERS:
        t = A[(A.n == n) & (A.version == v)]
        ax.plot(t.P, t.S_obs, marker="o", color=col[v], label=f"{v} obs")
        ax.plot(t.P, t.S_pred, marker="x", ls=":", color=col[v], label=f"{v} Amdahl")
    ax.set_title(f"n = {n}"); ax.set_xlabel("P"); ax.set_xticks([4, 8])
np.atleast_1d(axs)[0].set_ylabel("Aceleracion"); np.atleast_1d(axs)[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_amdahl.png"); plt.close(fig)

with open(f"{OUT}/resumen.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
say(f"\nListo. Resultados en '{OUT}/'")
