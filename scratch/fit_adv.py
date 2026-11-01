#!/usr/bin/env python3
"""Parse f9c_rmap.out impulse maps -> per-dst-cell tap multisets; look for structure."""
import re
from collections import defaultdict

def parse(path):
    runs, cur = [], None
    hdr = re.compile(r'^== rmap (\S+).*off=\((\d+),(\d+)\) src=(\d+)x(\d+) roi=(\d+)x(\d+) mask=(\d+)x(\d+) a=\((\d+),(\d+)\)')
    imp = re.compile(r'^  imp\((\d+),(\d+)\): (.*)$')
    hit = re.compile(r'\((\d+),(\d+)\)=(\d+)')
    for line in open(path):
        m = hdr.match(line)
        if m:
            cur = dict(name=m.group(1), offx=int(m.group(2)), offy=int(m.group(3)),
                       sw=int(m.group(4)), sh=int(m.group(5)), W=int(m.group(6)),
                       H=int(m.group(7)), mw=int(m.group(8)), mh=int(m.group(9)),
                       ax=int(m.group(10)), ay=int(m.group(11)), imp={})
            runs.append(cur); continue
        m = imp.match(line)
        if m and cur is not None:
            j, r = int(m.group(1)), int(m.group(2))
            cur['imp'][(j, r)] = {((int(a), int(b)), int(w)) for a, b, w in hit.findall(m.group(3))}
    return runs

runs = parse('/home/cjxu/worktrees/mcMPP-f9/scratch/remote_out/f9c_rmap.out')
for run in runs:
    print(f"== {run['name']} off=({run['offx']},{run['offy']}) src={run['sw']}x{run['sh']} "
          f"roi={run['W']}x{run['H']} mask={run['mw']}x{run['mh']} a=({run['ax']},{run['ay']}) ==")
    cell = defaultdict(dict)
    for (j, r), hits in run['imp'].items():
        for (xy, w) in hits:
            cell[xy][(j, r)] = w
    for y in range(run['H']):
        for x in range(run['W']):
            taps = cell.get((x, y), {})
            if not taps: continue
            total = sum(taps.values())
            s = ' '.join(f"({j},{r})x{w//10}" for (j, r), w in sorted(taps.items()))
            print(f"  dst({x},{y}) [{total//10} visible]: {s}")
    print()
