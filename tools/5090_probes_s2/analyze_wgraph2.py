#!/usr/bin/env python3
"""Stage-2 WMSSSIM analysis: full functor decode + per-level partials,
accumulator evolution and divisor solve, to pin the reduction formula.

usage: analyze_wgraph2.py <snapdir> <w> <h> <pws_base_hex>
"""
import sys, struct, math, os

snap=sys.argv[1]; W=int(sys.argv[2]); H=int(sys.argv[3])
base=int(sys.argv[4],16)

def rd(name):
    with open(os.path.join(snap,name),'rb') as f: return f.read()
def getf(buf,off): return struct.unpack_from('<f',buf,off)[0]
def u64(b,off=0): return int.from_bytes(b[off:off+8],'little')
def u32(b,off=0): return int.from_bytes(b[off:off+4],'little')
def f32(b,off=0): return struct.unpack_from('<f',b,off)[0]

# ---------- params.txt ----------
nodes=[]; cur=None
for line in open(os.path.join(snap,'params.txt')):
    line=line.strip()
    if line.startswith('node '):
        idx=int(line.split()[1])
        parts=dict(p.split('=',1) for p in line.split()[2:] if '=' in p)
        cur={'idx':idx,'grid':parts['grid'],'block':parts['block'],
             'smem':int(parts['smem']),'params':[]}
        nodes.append(cur)
    elif line.startswith('p') and cur is not None:
        sz=int(line.split()[1].split('=')[1])
        cur['params'].append((sz,bytes.fromhex(line.split('hex=')[1].strip())))
reds=[n for n in nodes if n['smem']==3072 and len(n['params'])==2]
sigs=[n for n in nodes if len(n['params'])==2 and n['smem']!=3072]
decs=[n for n in nodes if len(n['params'])==11]
print(f"nodes={len(nodes)} reds={len(reds)} sigs={len(sigs)} decs={len(decs)}")

# ---------- images / model ----------
img_a=rd('img_a.bin'); img_b=rd('img_b.bin')
A=[[img_a[y*W+x] for x in range(W)] for y in range(H)]
B=[[img_b[y*W+x] for x in range(W)] for y in range(H)]
ax=list(range(-5,6))
w1=[math.exp(-(x*x)/(2*1.5*1.5)) for x in ax]
s=sum(w1); w1=[v/s for v in w1]
def conv(im):
    h=len(im); w=len(im[0])
    out=[[0.0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc=0.0
            for dy in range(-5,6):
                yy=min(h-1,max(0,y+dy)); wy=w1[dy+5]; row=im[yy]
                for dx in range(-5,6):
                    acc+=wy*w1[dx+5]*row[min(w-1,max(0,x+dx))]
            out[y][x]=acc
    return out
def dec2(im):
    h=len(im); w=len(im[0]); nw=(w+1)//2; nh=(h+1)//2
    return [[(im[max(0,2*y-1)][max(0,2*x-1)]+im[max(0,2*y-1)][min(w-1,2*x)]
             +im[min(h-1,2*y)][max(0,2*x-1)]+im[min(h-1,2*y)][min(w-1,2*x)])/4.0
            for x in range(nw)] for y in range(nh)]
C1,C2=6.5025,58.5225
WT=[0.0448,0.2856,0.3001,0.2363,0.1333]

# ---------- full red functor decode ----------
print("\n== red functors (full decode) ==")
for i,r in enumerate(reds):
    sz,blob=r['params'][0]
    print(f" red{i} node={r['idx']} grid={r['grid']} block={r['block']} sz={sz} blobsz={len(blob)}")
    for off in range(0,min(sz,len(blob)),8):
        v=u64(blob,off)
        note=f"ptr+0x{v-base:x}" if 0x1000<v<0x7fffffffffff and v>=base else ""
        print(f"   +{off:3d}: u32s=({u32(blob,off):<12},{u32(blob,off+4):<12}) f32s=({f32(blob,off):.7g},{f32(blob,off+4):.7g}) {note}")

# ---------- signal args ----------
print("\n== signal kernel args (arena from p0, p1 gives tail) ==")
sigargs=[]
for i,sg in enumerate(sigs):
    sz,b0=sg['params'][0]; b1=sg['params'][1][1]
    arena=bytearray(b0[:32])
    if len(b1)>24: arena+=b1[24:]
    args=[]
    for off in range(0,len(arena)//8*8,8):
        v=u64(arena,off)
        note=f"ptr+0x{v-base:x}" if 0x1000<v<0x7fffffffffff and v>=base else ""
        args.append(v)
        print(f" sig{i} node={sg['idx']} grid={sg['grid']} block={sg['block']} smem={sg['smem']}: a[{off//8}]={v:<22} {note} u32s=({u32(arena,off)},{u32(arena,off+4)})")
    sigargs.append(args)

# ---------- decimate calls ----------
print("\n== decimate calls ==")
for i,d in enumerate(decs):
    d1=u64(d['params'][5][1])-base; d2=u64(d['params'][7][1])-base
    dw=u32(d['params'][9][1]); dh=u32(d['params'][9][1],4)
    print(f" dec{i} node={d['idx']}: dst {dw}x{dh} d1=+0x{d1:x} d2=+0x{d2:x} last={u32(d['params'][10][1])}")

# ---------- per-level partials + model row sums ----------
print("\n== partials vs model row sums ==")
a,b=A,B
for lev in range(5):
    r=reds[lev]
    pptr=u64(r['params'][0][1],56)-base
    hh=len(a); ww=len(a[0])
    buf=rd(f"prefix_{r['idx']:02d}.bin")
    npart=sigargs[lev][0] if lev<len(sigargs) else hh
    P=[getf(buf,pptr+4*k) for k in range(npart)]
    mu_a=conv(a); mu_b=conv(b)
    ea2=conv([[a[y][x]*a[y][x] for x in range(ww)] for y in range(hh)])
    eb2=conv([[b[y][x]*b[y][x] for x in range(ww)] for y in range(hh)])
    eab=conv([[a[y][x]*b[y][x] for x in range(ww)] for y in range(hh)])
    rows_abs=[]; rows_sgn=[]; rows_l=[]
    for y in range(hh):
        sa=0.0; sb=0.0; sab=0.0; sa2=0.0; sb2=0.0; mu=0.0; mv=0.0; ml=0.0
        for x in range(ww):
            va=ea2[y][x]-mu_a[y][x]**2; vb=eb2[y][x]-mu_b[y][x]**2
            cov=eab[y][x]-mu_a[y][x]*mu_b[y][x]
            sa+=(2*abs(cov)+C2)/(va+vb+C2)
            sb+=(2*cov+C2)/(va+vb+C2)
            l=(2*mu_a[y][x]*mu_b[y][x]+C1)/(mu_a[y][x]**2+mu_b[y][x]**2+C1)
            ml+=l*(2*abs(cov)+C2)/(va+vb+C2)
        rows_abs.append(sa); rows_sgn.append(sb); rows_l.append(ml)
    npnx=hh*ww
    print(f"-- level {lev} ({ww}x{hh}) partials@+0x{pptr:x} n={npart}")
    print("   NPP  :", " ".join(f"{v:9.5f}" for v in P))
    print("   |cov|:", " ".join(f"{v:9.5f}" for v in rows_abs))
    print("   cov  :", " ".join(f"{v:9.5f}" for v in rows_sgn))
    if lev==4:
        print("   l*cs :", " ".join(f"{v:9.5f}" for v in rows_l))
    da=max(abs(P[y]-rows_abs[y]) for y in range(npart)) if npart==hh else -1
    ds=max(abs(P[y]-rows_sgn[y]) for y in range(npart)) if npart==hh else -1
    print(f"   maxdiff |cov|={da:.3e} cov={ds:.3e}  sumNPP={sum(P):.6f} sumAbs={sum(rows_abs):.6f}")
    if lev<4: a=dec2(a); b=dec2(b)

# ---------- accumulator evolution ----------
print("\n== accumulators (pWS+0..+0x18) after each signal ==")
for lev,sg in enumerate(sigs):
    buf=rd(f"prefix_{sg['idx']:02d}.bin")
    acc=[getf(buf,4*k) for k in range(6)]
    print(f" after sig{lev} (prefix_{sg['idx']:02d}): " + " ".join(f"{v:.8g}" for v in acc))
buf18=rd("prefix_18.bin")
print(" final pWS+0..0x18: " + " ".join(f"{getf(buf18,4*k):.8g}" for k in range(6)))
out=rd('out.bin')
print(f" out.bin: f32={struct.unpack_from('<f',out,0)[0]:.10g} f64={struct.unpack_from('<d',out,0)[0]:.10g}")

# ---------- divisor solve ----------
print("\n== divisor / exponent solve ==")
a,b=A,B
for lev in range(5):
    r=reds[lev]; sg=sigs[lev]
    pptr=u64(r['params'][0][1],56)-base
    buf=rd(f"prefix_{r['idx']:02d}.bin")
    npart=sigargs[lev][0]
    P=[getf(buf,pptr+4*k) for k in range(npart)]
    S=sum(P)
    hh=len(a); ww=len(a[0]); npx=hh*ww
    abuf=rd(f"prefix_{sg['idx']:02d}.bin")
    accv=getf(abuf,4*lev)
    w=WT[lev]
    mean=S/npx
    # acc = mean^w ?
    print(f" lev{lev}: S={S:.6f} npx={npx} mean={mean:.8f} acc={accv:.8f} mean^w={mean**w:.8f} "
          f"S/acc^(1/w)={S/(accv**(1.0/w)) if accv>0 else float('nan'):.4f}")
    if lev<4: a=dec2(a); b=dec2(b)
prod=1.0
for lev in range(5):
    abuf=rd(f"prefix_{sigs[lev]['idx']:02d}.bin")
    prod*=getf(abuf,4*lev)
print(f" product of accs = {prod:.10g}")
