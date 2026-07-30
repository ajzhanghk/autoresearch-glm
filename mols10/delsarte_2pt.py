#!/usr/bin/env python3
"""
Two-point Delsarte (LP) baseline for OA(n^2, k, n, 2) index 1
  <=>  (k-2)-MOLS(n).
Distinct codewords agree in <= 1 coordinate (strength 2, index 1), so in a
length-k code the Hamming distance between distinct words lies in {k-1, k}.

We compute the inner distribution a_i (i=0..k) forced by the OA axioms and
check the Delsarte dual (MacWilliams) nonnegativity a'_j >= 0.

Validation targets:
  n=6, k=4 : 2-MOLS(6) = Euler 36 officers = NONEXISTENT (Tarry 1901)
  n=10,k=5 : 3-MOLS(10) = OPEN (this project)
If the 2-point LP is feasible for BOTH, the 2-point bound cannot separate
them, and the three-point SDP (next script) is the discriminator to build,
validated on the n=6 case.
"""
from fractions import Fraction
from math import comb

def krawtchouk(q, n, j, i):
    # q-ary Krawtchouk K_j(i)
    return sum((-1)**h * (q-1)**(j-h) * comb(i, h) * comb(n-i, j-h)
               for h in range(min(i, j) + 1))

def two_point(n, k):
    q = n
    N = n*n  # |C| for index-1 strength-2 OA
    # distinct-codeword distances live in {k-1, k}; solve inner distribution
    # a_{k-1}, a_k from the strength-2 (dual a'_1=a'_2=0) conditions.
    # a_0 = 1 (self), sum of a over distinct = N-1.
    # Unknowns a4:=a_{k-1}, a5:=a_k  (names for k=5); general d1=k-1,d2=k.
    d1, d2 = k-1, k
    # dual conditions a'_1 = a'_2 = 0:
    # a'_j = (1/N) sum_i K_j(i) a_i ; require =0 for j=1,2 (strength 2)
    A = [[krawtchouk(q,k,j,d1), krawtchouk(q,k,j,d2)] for j in (1,2)]
    b = [-krawtchouk(q,k,j,0) for j in (1,2)]
    det = A[0][0]*A[1][1]-A[0][1]*A[1][0]
    a_d1 = Fraction(b[0]*A[1][1]-b[1]*A[0][1], det)
    a_d2 = Fraction(A[0][0]*b[1]-A[1][0]*b[0], det)
    a = {0:Fraction(1), d1:a_d1, d2:a_d2}
    tot = a_d1 + a_d2
    print(f"  OA({N},{k},{n},2): a_{d1}={a_d1}, a_{d2}={a_d2}, sum={tot} (need {N-1})")
    ok = (a_d1>=0 and a_d2>=0 and tot==N-1)
    feasible = ok
    for j in range(k+1):
        ap = sum(krawtchouk(q,k,j,i)*a.get(i,Fraction(0)) for i in range(k+1))/N
        if ap < 0: feasible=False
        print(f"    a'_{j} = {ap}")
    print(f"  => 2-point LP: {'FEASIBLE (no obstruction)' if feasible else 'INFEASIBLE => nonexistent'}")
    return feasible

print("=== n=6, k=4  (2-MOLS(6) = Euler 36 officers, NONEXISTENT) ===")
two_point(6,4)
print("=== n=10, k=5 (3-MOLS(10), OPEN) ===")
two_point(10,5)
