#!/usr/bin/env python3
"""fit6: solve for effective mu/V of the 2D Wiener from n-linearity (m33 7x7 grids)."""
import numpy as np

def load(path):
    lines = open(path).read().splitlines()
    src = np.array([[float(v) for v in lines[1+k].split()] for k in range(7)])
    grids = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith('==') and 'status' in ln:
            name = ln[3:].split(' status')[0]
            if name.startswith('const'):
                i += 1; continue
            grids[name] = np.array([[float(v) for v in lines[i+1+k].split()] for k in range(7)])
            i += 8
        else:
            i += 1
    return src, grids

src, g = load('wiener2.txt')
n1 = g['m33-a11-n1']; n05 = g['m33-a11-n05']
H, W = src.shape

def clamp(v, lo, hi): return min(max(v, lo), hi)

def win(x, y, mw=3, mh=3, ax=1, ay=1):
    cols = [clamp(x+ax-(mw-1)+i, 0, W-1) for i in range(mw)]
    rows = [clamp(y+ay-(mh-1)+j, 0, H-1) for j in range(mh)]
    return np.array([[src[r, c] for c in cols] for r in rows])

print(f'{"xy":>4} {"in":>5} {"icept":>9} {"slope":>9} {"mu":>8} {"Vpop":>8} '
      f'{"Veff@mu":>9} {"Vh":>7} {"Vv":>7} {"mu*":>8} {"Veff/mu*":>9}')
for y in range(H):
    for x in range(W):
        w = win(x, y)
        in_ = src[y, x]
        o1 = n1[y, x]; o05 = n05[y, x]
        slope = (o1 - o05) / 0.5
        icept = 2*o05 - o1
        mu = w.mean(); V = w.var()
        # marginal variances (center row / center col of the window)
        Vh = w[1, :].var(); Vv = w[:, 1].var()
        # effective V assuming mu: out = in - n*(in-mu)/Veff
        Veff = (in_ - mu) / (in_ - o1) if abs(in_ - o1) > 1e-9 else float('nan')
        # effective mu assuming V: out = in - n*(in-mu*)/V
        mus = in_ - (in_ - o1) * V if V > 1e-9 else float('nan')
        print(f'{x},{y:<3} {in_:5g} {icept:9.5g} {slope:9.5g} {mu:8.5g} {V:8.5g} '
              f'{Veff:9.5g} {Vh:7.4g} {Vv:7.4g} {mus:8.5g} {(V/ (in_-o1) if abs(in_-o1)>1e-9 else float("nan")):9.5g}')
