#!/usr/bin/env python3
"""Analyze wgraph prefix snapshots: decode NPP WMSSSIM intermediates.
usage: analyze_wgraph.py <snapdir> <w> <h> <pws_base_hex>
"""
import sys, struct, math, os

snap=sys.argv[1]; W=int(sys.argv[2]); H=int(sys.argv[3])
base=int(sys.argv[4],16)

def rd(name):
    with open(os.path.join(snap,name),'rb') as f: return f.read()

img_a=rd('img_a.bin'); img_b=rd('img_b.bin')
A=[[img_a[y*W+x] for x in range(W)] for y in range(H)]
B=[[img_b[y*W+x] for x in range(W)] for y in range(H)]

def getf(buf,off): return struct.unpack_from('<f',buf,off)[0]
def region(buf,off,w,h,elemstride=None):
    st=elemstride if elemstride else w
    return [[getf(buf,off+4*(y*st+x)) for x in range(w)] for y in range(h)]

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
                yy=min(h-1,max(0,y+dy))
                for dx in range(-5,6):
                    xx=min(w-1,max(0,x+dx))
                    acc+=w1[dy+5]*w1[dx+5]*im[yy][xx]
            out[y][x]=acc
    return out
C1,C2=6.5025,58.5225

# ---------- parse params.txt ----------
def parse_params(snap):
    nodes=[]
    cur=None
    for line in open(os.path.join(snap,'params.txt')):
        line=line.strip()
        if line.startswith('node '):
            idx=int(line.split()[1])
            parts=dict(p.split('=',1) for p in line.split()[2:] if '=' in p)
            cur={'idx':idx,'func':parts['func'],'grid':parts['grid'],'block':parts['block'],
                 'smem':int(parts['smem']),'params':[]}
            nodes.append(cur)
        elif line.startswith('p') and cur is not None:
            sz=int(line.split()[1].split('=')[1])
            hexs=line.split('hex=')[1].strip()
            cur['params'].append((sz,bytes.fromhex(hexs)))
    return nodes
nodes=parse_params(snap)
def u64(b,off=0): return int.from_bytes(b[off:off+8],'little')
def u32(b,off=0): return int.from_bytes(b[off:off+4],'little')
def f32(b,off=0): return struct.unpack_from('<f',b,off)[0]

print(f"nodes={len(nodes)}")
# classify by param count
filts=[n for n in nodes if len(n['params'])==7]
reds=[n for n in nodes if len(n['params'])==2 and n['smem']==3072]
sigs=[n for n in nodes if len(n['params'])==2 and n['smem']!=3072]
decs=[n for n in nodes if len(n['params'])==11]
print(f"filters={len(filts)} reductions={len(reds)} signals={len(sigs)} decimates={len(decs)}")

# ---------- 1. decimate calls ----------
print("\n== decimate calls ==")
for i,d in enumerate(decs):
    a=u64(d['params'][0][1]); sa=u32(d['params'][1][1])
    b=u64(d['params'][2][1]); sb=u32(d['params'][3][1])
    sw=u32(d['params'][4][1]); sh=u32(d['params'][4][1],4)
    d1=u64(d['params'][5][1]); sd1=u32(d['params'][6][1])
    d2=u64(d['params'][7][1]); sd2=u32(d['params'][8][1])
    dw=u32(d['params'][9][1]); dh=u32(d['params'][9][1],4)
    last=u32(d['params'][10][1])
    print(f" dec{i}: src({sw}x{sh}) a=+0x{a-base:x} st={sa} b=+0x{b-base:x} st={sb}"
          f" -> dst({dw}x{dh}) d1=+0x{d1-base:x} st={sd1} d2=+0x{d2-base:x} st={sd2} last={last}")

def dec1d(v,mode):
    n=len(v)
    if mode=='shift': no=(n+1)//2; return [ (v[max(0,2*i-1)]+v[min(n-1,2*i)])/2 for i in range(no) ]
    if mode=='std':   no=(n+1)//2; return [ (v[2*i]+v[min(n-1,2*i+1)])/2 for i in range(no) ]
    if mode=='shift_floor': no=n//2; return [ (v[max(0,2*i-1)]+v[2*i])/2 for i in range(no) ]
    if mode=='std_floor':   no=n//2; return [ (v[2*i]+v[2*i+1])/2 for i in range(no) ]
def dec2(im,mode):
    h=len(im); w=len(im[0])
    tmp=[dec1d(im[y],mode) for y in range(h)]
    w2=len(tmp[0])
    cols=[[tmp[y][x] for y in range(h)] for x in range(w2)]
    cols2=[dec1d(c,mode) for c in cols]
    return [[cols2[x][y] for x in range(w2)] for y in range(len(cols2[0]))]

srcA,srcB=A,B
print("\n== decimated images vs hypotheses ==")
for i,d in enumerate(decs):
    dw=u32(d['params'][9][1]); dh=u32(d['params'][9][1],4)
    d1=u64(d['params'][5][1])-base; d2=u64(d['params'][7][1])-base
    buf=rd(f"prefix_{d['idx']:02d}.bin")
    im1=region(buf,d1,dw,dh); im2=region(buf,d2,dw,dh)
    print(f"-- level{i+1} A ({dw}x{dh}) --")
    for row in im1: print("   "," ".join(f"{v:8.3f}" for v in row))
    for mode in ['shift','std','shift_floor','std_floor']:
        p1=dec2(srcA,mode); p2=dec2(srcB,mode)
        d_=max(abs(im1[y][x]-p1[y][x]) for y in range(dh) for x in range(dw))
        d2_=max(abs(im2[y][x]-p2[y][x]) for y in range(dh) for x in range(dw))
        print(f"   mode={mode}: A maxdiff={d_:.3e}  B maxdiff={d2_:.3e}")
    srcA=dec2(srcA,'shift'); srcB=dec2(srcB,'shift')

# ---------- 2. ImageReduction functor decode + partials ----------
print("\n== ImageReduction functors ==")
for i,r in enumerate(reds):
    blob=r['params'][0][1]; sz=r['params'][0][0]
    print(f" red{i} (node {r['idx']}) p0 size={sz}:")
    for off in range(0,min(sz,88),8):
        v=u64(blob,off)
        f1=f32(blob,off); f2=f32(blob,off+4)
        note=""
        if 0x1000<v<0x7fffffffffff: note=f"ptr +0x{v-base:x}" if v>=base else "ptr?"
        print(f"   +{off:3d}: u64=0x{v:016x} ({v})  f32s=({f1:.6g},{f2:.6g}) {note}")

# partials: functor output ptr — dump them from the snapshot AFTER this red node
def model_levels(a,b,nlev=5):
    """returns list of (mu_a,mu_b,eab,ea2,eb2) maps per level"""
    out=[]
    for j in range(nlev):
        mu_a=conv(a); mu_b=conv(b)
        ea2=conv([[a[y][x]*a[y][x] for x in range(len(a[0]))] for y in range(len(a))])
        eb2=conv([[b[y][x]*b[y][x] for x in range(len(b[0]))] for y in range(len(b))])
        eab=conv([[a[y][x]*b[y][x] for x in range(len(a[0]))] for y in range(len(a))])
        out.append((mu_a,mu_b,eab,ea2,eb2))
        if j<nlev-1: a=dec2(a,'shift'); b=dec2(b,'shift')
    return out

lv=model_levels(A,B,5)
print("\n== reduction partial sums vs model (level 0) ==")
r0=reds[0]
blob=r0['params'][0][1]
# guess: p0 has size at +0, 5 ptrs at +8..+40, stride +48, outptr +56?
outptr=u64(blob,56)
print(f"guessed partials ptr: +0x{outptr-base:x}")
buf=rd(f"prefix_{r0['idx']:02d}.bin")
mu_a,mu_b,eab,ea2,eb2=lv[0]
hh,ww=H,W
sa=[[ea2[y][x]-mu_a[y][x]**2 for x in range(ww)] for y in range(hh)]
sb=[[eb2[y][x]-mu_b[y][x]**2 for x in range(ww)] for y in range(hh)]
sab=[[eab[y][x]-mu_a[y][x]*mu_b[y][x] for x in range(ww)] for y in range(hh)]
cs=[[ (2*abs(sab[y][x])+C2)/(sa[y][x]+sb[y][x]+C2) for x in range(ww)] for y in range(hh)]
l=[[ (2*mu_a[y][x]*mu_b[y][x]+C1)/(mu_a[y][x]**2+mu_b[y][x]**2+C1) for x in range(ww)] for y in range(hh)]
csn=[[ (2*sab[y][x]+C2)/(sa[y][x]+sb[y][x]+C2) for x in range(ww)] for y in range(hh)]
tot_cs=sum(cs[y][x] for y in range(hh) for x in range(ww))
tot_lcs=sum(l[y][x]*cs[y][x] for y in range(hh) for x in range(ww))
tot_csn=sum(csn[y][x] for y in range(hh) for x in range(ww))
# valid-only sums
vtot=0.0; vtotl=0.0
if hh>=11 and ww>=11:
    vtot=sum(cs[y][x] for y in range(5,hh-5) for x in range(5,ww-5))
    vtotl=sum(l[y][x]*cs[y][x] for y in range(5,hh-5) for x in range(5,ww-5))
print(f"model sums: cs(all)={tot_cs:.6f} cs(valid)={vtot:.6f} l*cs(all)={tot_lcs:.6f} l*cs(valid)={vtotl:.6f} csnocov(all)={tot_csn:.6f}")
print(f"means: cs(all)={tot_cs/(hh*ww):.8f} cs(valid)={vtot/max(1,(hh-10)*(ww-10)):.8f}")
print("partials dump (+0x%x..):"% (outptr-base))
for k in range(8):
    v=getf(buf,(outptr-base)+4*k)
    print(f"   partial[{k}] = {v:.6f}")
print("pWS head (first 64 bytes as f32):")
for k in range(16):
    print(f"   +{4*k:3d}: {getf(buf,4*k):.8g}")
