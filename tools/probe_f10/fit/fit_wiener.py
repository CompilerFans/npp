#!/usr/bin/env python3
"""Fit NPP 13.4 nppiFilterWienerBorder semantics from probe data."""
import itertools, sys
import numpy as np

def parse(path):
    grids, src, consts = {}, None, {}
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln == 'src:':
            src = np.array([[float(v) for v in lines[i+1+k].split()] for k in range(7)])
            i += 8
        elif ln.startswith('==') and 'status' in ln:
            name = ln[3:].split(' status')[0]
            if name.startswith('const'):
                parts = ln.split()
                n = float(parts[2].split('=')[1])
                consts[n] = (float(parts[-2].split('=')[1]), float(parts[-1].split('=')[1]))
                i += 1
            else:
                grids[name] = np.array([[float(v) for v in lines[i+1+k].split()] for k in range(7)])
                i += 8
        else:
            i += 1
    return src, grids, consts

src, grids, consts = parse(sys.argv[1] if len(sys.argv) > 1 else 'wiener2.txt')
H, W = src.shape

def window(x, y, mw, mh, ax, ay):
    cols = [min(max(x + ax - (mw - 1) + i, 0), W - 1) for i in range(mw)]
    rows = [min(max(y + ay - (mh - 1) + j, 0), H - 1) for j in range(mh)]
    return [src[r, c] for r in rows for c in cols]

def stats(x, y, mw, mh, ax, ay):
    s = window(x, y, mw, mh, ax, ay)
    n = len(s)
    mu = sum(s) / n
    var = sum((v - mu) ** 2 for v in s) / n
    return mu, var, s

def report(name, mw, mh, ax, ay):
    g = grids[name]
    print(f'--- {name} (mask {mw}x{mh} anchor {ax},{ay}) ---')
    print(f'{"x,y":>6} {"out":>12} {"mu":>10} {"V":>10} {"V2":>10} {"in":>6}  cands')
    for y in range(H):
        for x in range(W):
            mu, V, s = stats(x, y, mw, mh, ax, ay)
            out = g[y, x]
            in_ = src[y, x]
            print(f'{x},{y:>4} {out:12.6g} {mu:10.5g} {V:10.5g} {V*V:10.5g} {in_:6g}  '
                  f'out-mu={out-mu:9.4g} out/V={out/V if V else float("nan"):8.4g}')

# candidate: linear fit of out on feature basis
def linfit(name, mw, mh, ax, ay, feats, tag):
    g = grids[name]
    A, b = [], []
    for y in range(H):
        for x in range(W):
            mu, V, s = stats(x, y, mw, mh, ax, ay)
            in_ = src[y, x]
            env = {'mu': mu, 'V': V, 'in': in_, '1': 1.0, 'V2': V*V, 'mu2': mu*mu,
                   'in2': in_*in_, 'muV': mu*V, 'inV': in_*V, 'V3': V*V*V,
                   's': V*V+mu*mu, 'sq': sum(v*v for v in s)/len(s)}
            A.append([env[f] for f in feats])
            b.append(g[y, x])
    A, b = np.array(A), np.array(b)
    coef, res, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    err = np.abs(pred - b).max()
    print(f'{tag:34s} maxabs err {err:12.6g}   coefs ' + ' '.join(f'{f}={c:.6g}' for f, c in zip(feats, coef)))
    return err

if __name__ == '__main__':
    what = sys.argv[2] if len(sys.argv) > 2 else 'table'
    if what == 'table':
        report('m33-a11-n0', 3, 3, 1, 1)
    elif what == 'fit':
        for name, mw, mh, ax, ay in [('m33-a11-n0', 3, 3, 1, 1), ('m31-n0', 3, 1, 1, 0),
                                     ('m13-n0', 1, 3, 0, 1), ('m55-a22-n0', 5, 5, 2, 2)]:
            for feats in (['1', 'mu', 'V'], ['1', 'V'], ['mu', 'V', 'V2', 'in', 'in2', 'mu2', 'muV'],
                          ['1', 'mu', 'V', 'V2', 'in', 'in2', 'mu2', 'muV', 'inV', 'sq']):
                linfit(name, mw, mh, ax, ay, feats, f'{name} ~ ' + '+'.join(feats))

def rat(v, maxd=10000):
    from fractions import Fraction
    f = Fraction(v).limit_denominator(maxd)
    return f

def analyze_n0():
    """Try to identify the n=0 special path on multiple grids."""
    print('=== n=0 path: exact rational forms ===')
    for name, mw, mh, ax, ay in [('m33-a11-n0', 3, 3, 1, 1), ('m31-n0', 3, 1, 1, 0), ('m13-n0', 1, 3, 0, 1)]:
        g = grids[name]
        print(f'--- {name} row0 ---')
        for x in range(W):
            out = g[0, x]
            mu, V, s = stats(x, 0, mw, mh, ax, ay)
            f = rat(out)
            print(f'x={x} out={out:.9g} = {f} (den {f.denominator}) mu={mu:.4g} V={V:.4g} samples={s}')

def gain_fit():
    """For n>0: solve g = (out-mu)/(in-mu) per pixel, compare to V-based forms."""
    for nname, n in [('m33-a11-n025', 0.25), ('m33-a11-n05', 0.5), ('m33-a11-n1', 1.0)]:
        g = grids[nname]
        print(f'=== gain fit {nname} ===')
        for y in range(H):
            for x in range(W):
                mu, V, s = stats(x, y, 3, 3, 1, 1)
                in_ = src[y, x]
                out = g[y, x]
                if abs(in_ - mu) < 1e-9:
                    continue
                gg = (out - mu) / (in_ - mu)
                cands = {
                    '1-n2/V': 1 - n*n/V, '1-n/V': 1 - n/V, 'V/(V+n2)': V/(V+n*n),
                    '(V-n2)/V': (V-n*n)/V, '1-n2/V2': 1 - n*n/(V*V),
                    '(V2-n2)/V2': (V*V-n*n)/(V*V), 'V2/(V2+n2)': V*V/(V*V+n*n),
                    '1-n4/V2': 1 - n**4/(V*V), '(V-n2)/(V+n2)': (V-n*n)/(V+n*n),
                }
                best = min(cands, key=lambda k: abs(cands[k]-gg))
                print(f'{x},{y} out={out:9.5g} mu={mu:7.4g} V={V:8.5g} g={gg:9.6g} best={best}({cands[best]:9.6g}) '
                      f'Veff_n2={n*n/max(1e-12,1-gg):9.5g}')
