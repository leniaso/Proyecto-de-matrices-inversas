#!/usr/bin/env python3
"""Comparaciones por pares de las versiones nuevas (results_pool.csv).

Uso: python3 analisis_comparaciones.py results_pool.csv [carpeta_salida]

Familias (mismo n y mismo P, Mann-Whitney con correccion de Holm sobre TODAS las pruebas):
  A. procs_pool     < procs            (el pool de procesos reduce el costo de fork?)
  B. procs          < procs_pipe       (memoria compartida es mas rapida que pipes?)
  C. threads_barrier < threads         (el pool con barreras reduce el costo de crear hilos?)
  D. threads_queue  vs threads_barrier (cola dinamica vs reparto estatico; prueba bilateral)
cambio_pct = (mediana_X / mediana_Y - 1)*100 ; negativo => X mas rapida que Y.
cliff_delta > 0 => X tiende a ser mas rapida que Y (X<Y en mas pares de lo contrario).
"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

CSV = sys.argv[1] if len(sys.argv) > 1 else "results_pool.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "resultados_comparaciones"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(CSV); d = d[d.status == "ok"]

def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] < b[None, :]).sum() - (a[:, None] > b[None, :]).sum()) / (len(a) * len(b))

FAM = [("A", "procs_pool", "procs", "less"), ("B", "procs", "procs_pipe", "less"),
       ("C", "threads_barrier", "threads", "less"), ("D", "threads_queue", "threads_barrier", "two-sided")]
rows = []
for n in sorted(d.n.unique()):
    for P in sorted(d[d.version != "seq"].P.unique()):
        for fam, x, y, alt in FAM:
            a = d[(d.n == n) & (d.P == P) & (d.version == x)].time_s.values
            b = d[(d.n == n) & (d.P == P) & (d.version == y)].time_s.values
            if len(a) < 3 or len(b) < 3: continue
            p = stats.mannwhitneyu(a, b, alternative=alt).pvalue
            rows.append(dict(familia=fam, X=x, Y=y, n=n, P=P, reps_X=len(a), reps_Y=len(b),
                             mediana_X=np.median(a), mediana_Y=np.median(b),
                             cambio_pct=100 * (np.median(a) / np.median(b) - 1),
                             cliff_delta=cliff(a, b), p=p, alternativa=alt))
R = pd.DataFrame(rows)
R["p_holm"] = multipletests(R.p, method="holm")[1]
R["signif"] = np.where(R.p_holm < 0.05, "SI", "no")
R.to_csv(f"{OUT}/comparaciones_pares.csv", index=False)
pd.set_option("display.width", 200)
for fam, x, y, alt in FAM:
    t = R[R.familia == fam]
    print(f"\n=== {fam}: {x} vs {y}  ({alt}) ===")
    print(t[["n", "P", "mediana_X", "mediana_Y", "cambio_pct", "cliff_delta", "p_holm", "signif"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nPruebas totales (Holm): {len(R)}")
