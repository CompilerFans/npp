import re
lines = open('wiener2.txt').read().splitlines()
src=[]; cases={}; cur=None; const=[]
for ln in lines:
    if ln.startswith('src:'):
        cur='src'; continue
    if ln.startswith('== '):
        name=ln[3:].split(' status')[0]; cases[name]=[]; cur=name; continue
    if cur and ln.strip():
        cases[cur].append([float(t) for t in ln.split()])
src=cases.pop('src')
W=len(src[0]); H=len(src)
def G(name): 
    g=cases[name]; assert len(g)==H and len(g[0])==W, (name,len(g)); return g
n0=G('m33-a11-n0'); n025=G('m33-a11-n025'); n05=G('m33-a11-n05'); n1=G('m33-a11-n1')
# model: out(n) = in - n*(in-mu)/V  (V>n); check linearity: out(0.5)-out(0.25) == out(1)-out(0.5)
print("pixel in   out(.25)      out(.5)      out(1)       d1          d2      linear?  slope=(in-out(1))  V_from_n0  mu_solved  V>n all?")
bad=0
for y in range(H):
    for x in range(W):
        i=src[y][x]; a,b,c=n025[y][x],n05[y][x],n1[y][x]
        d1=b-a; d2=c-b
        lin = abs(d1-d2)<1e-4
        if not lin: bad+=1
        slope=(i-c)/1.0
        V=n0[y][x]
        mu=i-slope*V if V!=0 else None
        ok = V>1
        print(f"({x},{y}) {i:3d} {a:11.6f} {b:11.6f} {c:11.6f} {d1:10.6f} {d2:10.6f} {str(lin):5s} {slope:12.6f} {V:10.4f} {str(mu):12s} {ok}")
print("nonlinear count:",bad)
