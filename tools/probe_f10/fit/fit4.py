#!/usr/bin/env python3
"""Correlate n=0 output (suspected V_temp) with k = in-out(n=1).
   If out(n>0) = in - n*(in-mu_t)/V_t and n=0 outputs V_t (or V_t^2),
   then mu_t = in - k*V_t is recoverable and should look like a mean."""
import numpy as np

def parse(path):
    grids, src = {}, None
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln == 'src:':
            src = np.array([[float(v) for v in lines[i+1+k].split()] for k in range(7)])
            i += 8
        elif ln.startswith('==') and 'status' in ln and not ln.startswith('== const'):
            name = ln[3:].split(' status')[0]
            grids[name] = np.array([[float(v) for v in lines[i+1+k].split()] for k in range(7)])
            i += 8
        else:
            i += 1
    return src, grids

src, g = parse('wiener2.txt')
H, W = src.shape

def stats(x, y, mw, mh, ax, ay):
    cols = [min(max(x + ax - (mw-1) + i, 0), W-1) for i in range(mw)]
    rows = [min(max(y + ay - (mh-1) + j, 0), H-1) for j in range(mh)]
    s = [src[r, c] for r in rows for c in cols]
    n = len(s); mu = sum(s)/n; V = sum((v-mu)**2 for v in s)/n
    return mu, V

k = src - g['m33-a11-n1']        # out = in - k*n  =>  k = (in-mu_t)/V_t
o0 = g['m33-a11-n0']             # suspected V_t or V_t^2
print(f"{'(x,y)':>6} {'in':>5} {'mu':>8} {'V':>8} {'o0':>10} {'o0-V':>9} {'o0-V2':>10} {'k':>8} {'mu_t=o0':>8} {'mu_t=sq':>8}")
for y in range(H):
    for x in range(W):
        mu, V = stats(x, y, 3, 3, 1, 1)
        kk = k[y, x]
        mut1 = src[y,x] - kk*o0[y,x]          # if o0 = V_t
        mut2 = src[y,x] - kk*np.sqrt(max(o0[y,x],0))  # if o0 = V_t^2
        print(f"({x},{y}) {src[y,x]:5g} {mu:8.3f} {V:8.3f} {o0[y,x]:10.4f} {o0[y,x]-V:9.3f} {o0[y,x]-V*V:10.3f} {kk:8.4f} {mut1:8.3f} {mut2:8.3f}")
