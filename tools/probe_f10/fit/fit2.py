#!/usr/bin/env python3
"""Search stats-window/primary-pixel hypotheses for Wiener: out = in' - n^2*(in'-mu)/V."""
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

def samples(x, y, mw, mh, ax, ay, border='replicate'):
    out = []
    for j in range(mh):
        for i in range(mw):
            cx = x + ax - (mw - 1) + i
            cy = y + ay - (mh - 1) + j
            if border == 'replicate':
                cx = min(max(cx, 0), W - 1); cy = min(max(cy, 0), H - 1)
                out.append(src[cy, cx])
            elif border == 'zero':
                out.append(src[cy, cx] if (0 <= cx < W and 0 <= cy < H) else 0.0)
            elif border == 'raw':
                out.append(src[cy, cx] if (0 <= cx < W and 0 <= cy < H) else None)
    return out

def mu_var(s, ndiv=None):
    ss = [v for v in s if v is not None]
    n = ndiv or len(ss)
    if not ss: return None, None
    mu = sum(ss) / len(ss)
    V = sum((v - mu) ** 2 for v in ss) / n
    return mu, V

# c(x,y) from n=1 grid: c = in - out  (since out = in - c*n^2)
cases = [('m33-a11-n1', 3, 3, 1, 1), ('m55-a22-n1', 5, 5, 2, 2)]
for gname, mw, mh, ax, ay in cases:
    g = grids[gname]
    print(f'=== {gname}: c = in - out(n=1) ===')
    cs = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            cs[y, x] = src[y, x] - g[y, x]
    # search: stats window shifted by (dx,dy), primary pixel src(x+px, y+py), divisor variants
    best = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for px in range(-2, 3):
                for py in range(-2, 3):
                    for border in ('replicate', 'zero'):
                        for nd_off in (0, -1, 1):
                            errs = []
                            ok = True
                            for y in range(H):
                                for x in range(W):
                                    ix, iy = x + px, y + py
                                    ix = min(max(ix, 0), W - 1); iy = min(max(iy, 0), H - 1)
                                    in_ = src[iy, ix]
                                    s = samples(x + dx, y + dy, mw, mh, ax, ay, border)
                                    if any(v is None for v in s): ok = False; break
                                    mu, V = mu_var(s, len(s) + nd_off)
                                    if V is None or abs(V) < 1e-9: ok = False; break
                                    cpred = (in_ - mu) / V
                                    errs.append(abs(cpred - cs[y, x]))
                                if not ok: break
                            if ok:
                                best.append((max(errs), dx, dy, px, py, border, nd_off))
    best.sort()
    for b in best[:6]:
        print(f'  maxerr={b[0]:.6g} stats_win_shift=({b[1]},{b[2]}) primary=({b[3]},{b[4]}) border={b[5]} ndiv_off={b[6]}')
