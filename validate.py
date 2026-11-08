#!/usr/bin/env python3
"""Compara la inversa del programa en C contra numpy.linalg.inv (LAPACK).
Uso: DUMP=/tmp/t ./matinv hier 1000 4 64 7 && python3 validate.py /tmp/t 1000"""
import sys, numpy as np
pref, n = sys.argv[1], int(sys.argv[2])
M   = np.fromfile(f"{pref}_M.bin").reshape(n, n)
Inv = np.fromfile(f"{pref}_Inv.bin").reshape(n, n)
ref = np.linalg.inv(M)
res = np.linalg.norm(M @ Inv - np.eye(n), np.inf)
dif = np.linalg.norm(Inv - ref, np.inf) / np.linalg.norm(ref, np.inf)
print(f"residuo inf: {res:.3e}   error relativo vs LAPACK: {dif:.3e}")
sys.exit(0 if res < 1e-8 else 1)
