#!/usr/bin/env python3
"""Stage-3: real level-4 maps, decimated images all levels, final value.
usage: analyze_wgraph3.py <snapdir> <w> <h> <pws_base_hex>
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
decs=[n for n in nodes if len(n['params'])==11]

# ---- real level-4 maps (after red4 = node 17 -> prefix_17)
r4=reds[4]; blob=r4['params'][0][1]
w4=u32(blob,0); h4=u32(blob,4)
mptr=[u64(blob,8+8*i)-base for i in range(5)]
stride=u64(blob,48)//4
buf=rd('prefix_17.bin')
print(f"level4 maps ({w4}x{h4}) stride={stride} ptrs=" + " ".join(f"+0x{p:x}" for p in mptr))
names=['mu_a','mu_b','E_ab ','E_a2 ','E_b2 ']
for i,p in enumerate(mptr):
    vals=[getf(buf,p+stride*y*4) for y in range(h4)] if h4>0 else []
    row=[getf(buf,p+4*x) for x in range(w4)]
    print(f"  {names[i]}: row0={['%.6g'%v for v in row]} allrows={vals if h4>1 else ''}")
# derived
mua=getf(buf,mptr[0]); mub=getf(buf,mptr[1]); eab=getf(buf,mptr[2]); ea2=getf(buf,mptr[3]); eb2=getf(buf,mptr[4])
va=ea2-mua*mua; vb=eb2-mub*mub; cov=eab-mua*mub
print(f"  derived: mu_a={mua:.6g} mu_b={mub:.6g} cov={cov:.6g} va={va:.6g} vb={vb:.6g}")
print(f"  partial4={getf(buf,u64(blob,56)-base):.6g}")

# ---- decimated images, all levels, vs model chain
img_a=rd('img_a.bin'); img_b=rd('img_b.bin')
A=[[img_a[y*W+x] for x in range(W)] for y in range(H)]
B=[[img_b[y*W+x] for x in range(W)] for y in range(H)]
def dec2(im):
    h=len(im); w=len(im[0]); nw=(w+1)//2; nh=(h+1)//2
    return [[(im[max(0,2*y-1)][max(0,2*x-1)]+im[max(0,2*y-1)][min(w-1,2*x)]
             +im[min(h-1,2*y)][max(0,2*x-1)]+im[min(h-1,2*y)][min(w-1,2*x)])/4.0
            for x in range(nw)] for y in range(nh)]
def dec2std(im):
    h=len(im); w=len(im[0]); nw=w//2; nh=h//2
    return [[(im[2*y][2*x]+im[2*y][2*x+1]+im[2*y+1][2*x]+im[2*y+1][2*x+1])/4.0
            for x in range(nw)] for y in range(nh)]
ma,mb=A,B
for i,d in enumerate(decs):
    dw=u32(d['params'][9][1]); dh=u32(d['params'][9][1],4)
    d1=u64(d['params'][5][1])-base; d2=u64(d['params'][7][1])-base
    s1=u32(d['params'][6][1]); s2=u32(d['params'][8][1])
    buf=rd(f"prefix_{d['idx']:02d}.bin")
    im1=[[getf(buf,d1+4*(y*s1+x)) for x in range(dw)] for y in range(dh)]
    im2=[[getf(buf,d2+4*(y*s2+x)) for x in range(dw)] for y in range(dh)]
    p1=dec2(ma); p2=dec2(mb)
    q1=dec2std(ma); q2=dec2std(mb)
    e1=max(abs(im1[y][x]-p1[y][x]) for y in range(dh) for x in range(dw))
    e2=max(abs(im2[y][x]-p2[y][x]) for y in range(dh) for x in range(dw))
    f1=max(abs(im1[y][x]-q1[y][x]) for y in range(dh) for x in range(dw)) if q1 else -1
    print(f"dec{i}: {dw}x{dh} stride=({s1},{s2}) shift_clamp maxerr A={e1:.3e} B={e2:.3e} std_floor={f1:.3e}")
    if dw<=4:
        print("   A:", [[f'{v:g}' for v in r] for r in im1])
        print("   B:", [[f'{v:g}' for v in r] for r in im2])
        print("   modelA:", [[f'{v:g}' for v in r] for r in p1])
    ma,mb=p1,p2

# ---- final output
out=rd('out.bin')
print(f"out.bin len={len(out)} f32={struct.unpack_from('<f',out,0)[0]:.10g}")
