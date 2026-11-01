#!/usr/bin/env python3
"""Stage-4: test hypothesis  per-pixel value = cs^w (levels 0-3), (l*cs)^w (level 4);
acc = mean over all pixels; final = product of accs.
usage: analyze_wgraph4.py <snapdir> <w> <h> <pws_base_hex>
"""
import sys, struct, math, os
snap=sys.argv[1]; W=int(sys.argv[2]); H=int(sys.argv[3]); base=int(sys.argv[4],16)
def rd(n):
    with open(os.path.join(snap,n),'rb') as f: return f.read()
def getf(b,o): return struct.unpack_from('<f',b,o)[0]
def u64(b,o=0): return int.from_bytes(b[o:o+8],'little')
def u32(b,o=0): return int.from_bytes(b[o:o+4],'little')
nodes=[]; cur=None
for line in open(os.path.join(snap,'params.txt')):
    line=line.strip()
    if line.startswith('node '):
        idx=int(line.split()[1]); parts=dict(p.split('=',1) for p in line.split()[2:] if '=' in p)
        cur={'idx':idx,'smem':int(parts['smem']),'params':[]}; nodes.append(cur)
    elif line.startswith('p') and cur is not None:
        cur['params'].append((int(line.split()[1].split('=')[1]),bytes.fromhex(line.split('hex=')[1].strip())))
reds=[n for n in nodes if n['smem']==3072 and len(n['params'])==2]
sigs=[n for n in nodes if len(n['params'])==2 and n['smem']!=3072]

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

a,b=A,B
accs=[]; accs_npp=[]
for lev in range(5):
    r=reds[lev]
    blob=r['params'][0][1]
    w_lev=u32(blob,0); h_lev=u32(blob,4)
    wl=f32w=struct.unpack_from('<f',blob,76)[0]
    el=struct.unpack_from('<f',blob,72)[0]  # l-exponent field
    pptr=u64(blob,56)-base
    buf=rd(f"prefix_{r['idx']:02d}.bin")
    P=[getf(buf,pptr+4*k) for k in range(h_lev)]
    mu_a=conv(a); mu_b=conv(b)
    ea2=conv([[a[y][x]*a[y][x] for x in range(w_lev)] for y in range(h_lev)])
    eb2=conv([[b[y][x]*b[y][x] for x in range(w_lev)] for y in range(h_lev)])
    eab=conv([[a[y][x]*b[y][x] for x in range(w_lev)] for y in range(h_lev)])
    # candidate per-pixel values
    rows_cspw=[]; rows_cs=[]; rows_lcs_pw=[]; rows_l_pw=[]
    for y in range(h_lev):
        s1=s2=s3=s4=0.0
        for x in range(w_lev):
            va=ea2[y][x]-mu_a[y][x]**2; vb=eb2[y][x]-mu_b[y][x]**2
            cov=eab[y][x]-mu_a[y][x]*mu_b[y][x]
            cs=(2*abs(cov)+C2)/(va+vb+C2)
            csg=(2*cov+C2)/(va+vb+C2)
            l=(2*mu_a[y][x]*mu_b[y][x]+C1)/(mu_a[y][x]**2+mu_b[y][x]**2+C1)
            s1+=cs**wl
            s2+=cs
            s3+=(l*cs)**wl
            s4+=l**wl if el!=0 else 0.0
        rows_cspw.append(s1); rows_cs.append(s2); rows_lcs_pw.append(s3); rows_l_pw.append(s4)
    tag = "cs^w" if lev<4 else "(l*cs)^w"
    d_cspw=max(abs(P[y]-rows_cspw[y]) for y in range(h_lev))
    d_cs=max(abs(P[y]-rows_cs[y]) for y in range(h_lev))
    d_lcs=max(abs(P[y]-rows_lcs_pw[y]) for y in range(h_lev))
    print(f"level {lev} ({w_lev}x{h_lev}) w={wl} l-exp={el}: maxdiff cs^w={d_cspw:.3e} raw_cs={d_cs:.3e} (l*cs)^w={d_lcs:.3e}")
    if h_lev<=8:
        print("  NPP   :", " ".join(f"{v:8.5f}" for v in P))
        print("  "+tag+":", " ".join(f"{v:8.5f}" for v in (rows_cspw if lev<4 else rows_lcs_pw)))
    npx=w_lev*h_lev
    mv=sum(rows_lcs_pw if lev==4 else rows_cspw)/npx
    accs.append(mv)
    abuf=rd(f"prefix_{sigs[lev]['idx']:02d}.bin")
    accs_npp.append(getf(abuf,4*lev))
    if lev<4: a=dec2(a); b=dec2(b)

print("\nacc model vs NPP:")
prod=1.0; prodn=1.0
for lev in range(5):
    print(f"  lev{lev}: model={accs[lev]:.8f} npp={accs_npp[lev]:.8f} diff={accs[lev]-accs_npp[lev]:+.2e}")
    prod*=accs[lev]; prodn*=accs_npp[lev]
out=struct.unpack_from('<f',rd('out.bin'),0)[0]
print(f"product model={prod:.8f} product(npp accs)={prodn:.8f} out.bin={out:.8f}")
