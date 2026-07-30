#!/usr/bin/env python3
"""
Three-point counting relaxation for OA(n^2, k, n, 2) index 1  <=>  (k-2)-MOLS(n).

SAFETY PRINCIPLE (no false proofs): every constraint below is a PROVABLY
NECESSARY condition satisfied by any genuine such OA. Hence the feasible
region is an OUTER relaxation, and LP-INFEASIBLE ==> rigorous nonexistence.
The only failure mode is a too-weak bound (feasible when actually
nonexistent), which we report honestly as "no obstruction found".

Model of the OA as a NET: N = n^2 points, k coordinates. Coordinate c is a
partition of the points into q=n classes of size n ("symbol classes").
Strength-2 index-1  <=>  any two coordinates' partitions are orthogonal
<=> any two distinct points agree (same class) in AT MOST ONE coordinate.

THREE-POINT VARIABLES.
For an ORDERED triple of DISTINCT points (x,y,z), its "type" tau records, for
each coordinate c, the partition pattern of {x,y,z} into classes:
  0: all three in the same class          ("S")
  1: x=y  in a class, z separate          ("xy")
  2: x=z  in a class, y separate          ("xz")
  3: y=z  in a class, x separate          ("yz")
  4: all three in distinct classes        ("D")
so tau in {0..4}^k. y_tau >= 0 = number of ordered distinct triples of type tau.

NECESSARY CONSTRAINTS (all exact, from index-1 OA axioms):
 (C0) y_tau >= 0.
 (C1) sum_tau y_tau = N(N-1)(N-2).
 (C2) At-most-one-agreement per pair: for any two of the three points, the set
      of coordinates where they share a class has size <= 1. In tau this means:
      #coords with pattern in {S, xy} (x,y together) <= 1;  likewise xz, yz.
      (S makes all three pairs agree at that coord.)  We ENFORCE these as
      structural (types violating them are excluded, y_tau forced 0).
 (C3) Per-coordinate class-size counts (exact). Fix coordinate c. Classifying
      ordered triples by their pattern at c:
        #(all-same at c)     = n * n(n-1)(n-2)            [q=n classes, size n]
        #(exactly x=y at c)  = n * [n(n-1)] * (N-n)       [pair in a class, z outside]
        (xz, yz same by symmetry)
        #(all-diff at c)     = N(N-1)(N-2) - above
      => sum over tau with tau_c = p  of y_tau  equals these exact values.
 (C4) Pair marginal (exact, from index 1). For the ordered pair (x,y), the
      number of coords where they agree is 0 or 1, with EXACT totals:
        #ordered pairs agreeing at a given coord c = n * n(n-1) = n^2(n-1)
      Collapsing z out of y_tau must reproduce the exact pair-agreement counts.

If this LP is INFEASIBLE -> the OA (hence (k-2)-MOLS(n)) does not exist.
"""
import sys, itertools
import numpy as np
from scipy.optimize import linprog

def solve(n, k, verbose=True):
    N = n*n
    q = n
    patterns = range(5)  # S, xy, xz, yz, D
    # pair-agreement per pattern: which of the 3 unordered pairs (0:xy,1:xz,2:yz)
    # share a class at that coordinate
    pair_agree = {0:{0,1,2}, 1:{0}, 2:{1}, 3:{2}, 4:set()}
    # enumerate feasible types tau in {0..4}^k with C2 (each pair agrees in <=1 coord)
    types = []
    for tau in itertools.product(patterns, repeat=k):
        agree_count = [0,0,0]  # xy, xz, yz
        for c in tau:
            for p in pair_agree[c]:
                agree_count[p]+=1
        if all(a<=1 for a in agree_count):
            types.append(tau)
    T = len(types)
    if verbose: print(f"  OA({N},{k},{n},2): {T} feasible triple-types (after C2 pruning)")
    idx = {t:i for i,t in enumerate(types)}

    rows=[]; rhs=[]  # equality constraints A_eq y = b_eq
    # (C1)
    r=np.ones(T); rows.append(r); rhs.append(N*(N-1)*(N-2))
    # (C3) per-coordinate pattern totals
    tot = N*(N-1)*(N-2)
    per = {
        0: n * n*(n-1)*(n-2),         # all-same
        1: n * (n*(n-1)) * (N-n),     # exactly x=y
        2: n * (n*(n-1)) * (N-n),     # exactly x=z
        3: n * (n*(n-1)) * (N-n),     # exactly y=z
    }
    per[4] = tot - per[0]-per[1]-per[2]-per[3]  # all-diff
    for c in range(k):
        for p in patterns:
            r=np.zeros(T)
            for t in types:
                if t[c]==p: r[idx[t]]=1
            rows.append(r); rhs.append(per[p])
    A_eq=np.array(rows); b_eq=np.array(rhs, dtype=float)
    # (C5) TWO-COORDINATE joint three-point counts (exact, from orthogonality).
    # By index-1, points <-> (c-class, c'-class) grid [n]x[n] for any pair c,c'.
    # Labeled patterns have block structure; blocks(S)=1, blocks(xy/xz/yz)=2,
    # blocks(D)=3. A labeled pair {i,j} is a coincident point iff grouped
    # together in BOTH P@c and P@c' -> impossible (distinct points) -> count 0.
    # Otherwise count = n_(r) * n_(s), r=#blocks(P@c), s=#blocks(P@c').
    blocks_of = {0:[[0,1,2]], 1:[[0,1],[2]], 2:[[0,2],[1]], 3:[[1,2],[0]],
                 4:[[0],[1],[2]]}
    def nblk(p): return len(blocks_of[p])
    def together(p, i, j):  # are labeled points i,j in same block of pattern p
        return any(i in b and j in b for b in blocks_of[p])
    def falling(n_, r):
        v=1
        for t in range(r): v*= (n_-t)
        return v
    def joint_count(p, pp):
        for (i,j) in [(0,1),(0,2),(1,2)]:
            if together(p,i,j) and together(pp,i,j):
                return 0
        return falling(n, nblk(p)) * falling(n, nblk(pp))
    extra_rows=[]; extra_rhs=[]
    for c in range(k):
        for cp in range(c+1,k):
            for p in patterns:
                for pp in patterns:
                    val=joint_count(p,pp)
                    r=np.zeros(T)
                    for t in types:
                        if t[c]==p and t[cp]==pp: r[idx[t]]=1
                    extra_rows.append(r); extra_rhs.append(val)
    if extra_rows:
        A_eq=np.vstack([A_eq, np.array(extra_rows)])
        b_eq=np.concatenate([b_eq, np.array(extra_rhs, dtype=float)])
    if verbose: print(f"  constraints: {A_eq.shape[0]} (incl. {len(extra_rows)} two-coord joint)")
    # feasibility LP: min 0 s.t. A_eq y = b_eq, y>=0
    res=linprog(c=np.zeros(T), A_eq=A_eq, b_eq=b_eq, bounds=[(0,None)]*T,
                method='highs')
    status = res.status  # 0 optimal(feasible), 2 infeasible
    if verbose:
        print(f"  LP status: {res.message}")
        if status==0: print("  => FEASIBLE (no 3-point counting obstruction)")
        elif status==2: print("  => INFEASIBLE => (k-2)-MOLS(n) DOES NOT EXIST")
    return status

if __name__=="__main__":
    print("=== VALIDATION: OA(36,4,6,2) = 2-MOLS(6) = Euler 36 officers (NONEXISTENT) ===")
    solve(6,4)
    print("=== sanity: OA(9,3,3,2) = 1-MOLS(3) (EXISTS, expect feasible) ===")
    solve(3,3)
    print("=== sanity: OA(16,3,4,2) = 1-MOLS(4) (EXISTS) ===")
    solve(4,3)
    print("=== TARGET: OA(100,5,10,2) = 3-MOLS(10) (OPEN) ===")
    solve(10,5)
