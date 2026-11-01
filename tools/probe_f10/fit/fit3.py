#!/usr/bin/env python3
"""k = in - out(n=1): exact rational forms; look for structure."""
import numpy as np
from fractions import Fraction as F

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

src, grids = parse('wiener2.txt')
H, W = src.shape

def stats(x, y, mw, mh, ax, ay):
    cols = [min(max(x + ax - (mw-1) + i, 0), W-1) for i in range(mw)]
    rows = [min(max(y + ay - (mh-1) + j, 0), H-1) for j in range(mh)]
    s = [src[r, c] for r in rows for c in cols]
    n = len(s); mu = sum(s)/n; V = sum((v-mu)**2 for v in s)/n
    return mu, V, s

for gname, mw, mh, ax, ay in [('m33-a11-n1', 3, 3, 1, 1), ('m55-a22-n1', 5, 5, 2, 2)]:
    g = grids[gname]
    print(f'=== {gname}: k = in - out(n=1) as rationals ===')
    for y in range(H):
        row = []
        for x in range(W):
            k = src[y, x] - g[y, x]
            f = F(k).limit_denominator(100000)
            mu, V, s = stats(x, y, mw, mh, ax, ay)
            pred = (src[y, x] - mu) / V
            row.append(f'{f.numerator:>7}/{f.denominator:<6}({float(f):8.4g},pred {pred:8.4g})')
        print(f'y={y}: ' + ' '.join(row))
