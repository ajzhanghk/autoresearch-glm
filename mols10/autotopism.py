import json, glob, numpy as np
from itertools import permutations, product
from collections import defaultdict
N=10
def inv(p):
    q=[0]*N
    for i,v in enumerate(p): q[v]=i
    return q
def comp(a,b): return [a[b[i]] for i in range(N)]
def normalize(L):
    # permute columns so row 0 becomes identity: L'[r,c]=L[r, p0inv[c]]
    p0=list(L[0]); p0inv=inv(p0)
    return np.array([[int(L[r,p0inv[c]]) for c in range(N)] for r in range(N)])
def cycles(p):
    seen=[False]*N; cs=[]
    for i in range(N):
        if not seen[i]:
            c=[]; j=i
            while not seen[j]: seen[j]=True; c.append(j); j=p[j]
            cs.append(c)
    return cs
def conjugators(p, k):
    cp=cycles(p); ck=cycles(k)
    bp=defaultdict(list); bk=defaultdict(list)
    for c in cp: bp[len(c)].append(c)
    for c in ck: bk[len(c)].append(c)
    if {l:len(v) for l,v in bp.items()}!={l:len(v) for l,v in bk.items()}: return
    lens=sorted(bp); cs=[]
    for l in lens:
        cs.append((l,bp[l],bk[l],list(permutations(range(len(bk[l])))),list(product(range(l),repeat=len(bp[l])))))
    def gen(i,beta):
        if i==len(cs): yield list(beta); return
        l,pcs,kcs,perms,rots=cs[i]
        for perm in perms:
            for rot in rots:
                b2=beta[:]
                for pi,pc in enumerate(pcs):
                    kc=kcs[perm[pi]]; off=rot[pi]
                    for t in range(l): b2[pc[t]]=kc[(t+off)%l]
                yield from gen(i+1,b2)
    yield from gen(0,[0]*N)
def has_nontrivial_autotopism(L):
    L=normalize(L)  # row 0 = identity now
    P=[list(L[r]) for r in range(N)]
    rowset={tuple(P[r]):r for r in range(N)}
    P1=P[1]
    for a0 in range(N):
        Pa0inv=inv(P[a0])
        for rp in range(N):
            K=comp(Pa0inv,P[rp])
            for beta in conjugators(P1,K):
                binv=inv(beta); gamma=comp(P[a0],beta); ok=True; alpha=[]
                for r in range(N):
                    tr=rowset.get(tuple(comp(comp(gamma,P[r]),binv)))
                    if tr is None: ok=False; break
                    alpha.append(tr)
                if ok and len(set(alpha))==N and not (a0==0 and beta==list(range(N))):
                    return True
    return False
files=sorted(glob.glob('mols10/results/odls_pairs/pair_*.json'), key=lambda f:int(f.split('_')[-1].split('.')[0]))
sym=0; asym=[]
for f in files:
    d=json.load(open(f)); L=np.array(d['L'],dtype=int)
    if has_nontrivial_autotopism(L): sym+=1
    else: asym.append(f.split('/')[-1])
print(f"ct>=4 squares: {len(files)} total, {sym} nontrivial-autotopism (MMM-excluded), {len(asym)} trivial")
print("TRIVIAL-autotopism high-ct squares:", asym if asym else "NONE — all MMM-excluded")
