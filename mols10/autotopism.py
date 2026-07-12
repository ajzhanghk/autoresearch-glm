"""Autotopism-group triviality test for order-10 Latin squares.
An autotopism is (alpha,beta,gamma) with L[alpha(r),beta(c)]=gamma(L(r,c)).
Determined by (alpha(0), beta) once the square is normalized to identity
first row; beta ranges over conjugators of row 1 to P[a0]^{-1} P[r'],
found directly by cycle-matching (fast). Triviality is isotopy-invariant.
"""
import numpy as np
from itertools import permutations, product
from collections import defaultdict
N=10
def inv(p):
    q=[0]*N
    for i,v in enumerate(p): q[v]=i
    return q
def comp(a,b): return [a[b[i]] for i in range(N)]
def normalize(L):
    p0inv=inv([int(x) for x in L[0]])
    return np.array([[int(L[r,p0inv[c]]) for c in range(N)] for r in range(N)])
def _cycles(p):
    seen=[False]*N; cs=[]
    for i in range(N):
        if not seen[i]:
            c=[]; j=i
            while not seen[j]: seen[j]=True; c.append(j); j=p[j]
            cs.append(c)
    return cs
def _conjugators(p,k):
    cp=_cycles(p); ck=_cycles(k); bp=defaultdict(list); bk=defaultdict(list)
    for c in cp: bp[len(c)].append(c)
    for c in ck: bk[len(c)].append(c)
    if {l:len(v) for l,v in bp.items()}!={l:len(v) for l,v in bk.items()}: return
    lens=sorted(bp); css=[]
    for l in lens:
        css.append((l,bp[l],bk[l],list(permutations(range(len(bk[l])))),list(product(range(l),repeat=len(bp[l])))))
    def gen(i,beta):
        if i==len(css): yield list(beta); return
        l,pcs,kcs,perms,rots=css[i]
        for perm in perms:
            for rot in rots:
                b2=beta[:]
                for pi,pc in enumerate(pcs):
                    kc=kcs[perm[pi]]; off=rot[pi]
                    for t in range(l): b2[pc[t]]=kc[(t+off)%l]
                yield from gen(i+1,b2)
    yield from gen(0,[0]*N)
def has_nontrivial_autotopism(L):
    L=normalize(L); P=[[int(x) for x in L[r]] for r in range(N)]
    rowset={tuple(P[r]):r for r in range(N)}; P1=P[1]
    for a0 in range(N):
        Pa0inv=inv(P[a0])
        for rp in range(N):
            K=comp(Pa0inv,P[rp])
            for beta in _conjugators(P1,K):
                binv=inv(beta); gamma=comp(P[a0],beta); ok=True; alpha=[]
                for r in range(N):
                    tr=rowset.get(tuple(comp(comp(gamma,P[r]),binv)))
                    if tr is None: ok=False; break
                    alpha.append(tr)
                if ok and len(set(alpha))==N and not (a0==0 and beta==list(range(N))):
                    return True
    return False
