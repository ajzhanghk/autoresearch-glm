# Bright et al. Myrvold-case orthogonal pairs of 10x10 Latin squares

Source: SAT pipeline of Bright, Keita, Stevens, "Myrvold's Results on Orthogonal
Triples of 10 x 10 Latin Squares: A SAT Investigation" (arXiv:2503.10504, EJC 2026).
Scripts: https://github.com/curtisbright/Myrvold-MOLS (commit a0416e5, tag `zenodo`).
The repo contains no solution data; solutions below were re-derived locally by
running the repo's own pipeline (encode.py -> kissat -> decode.py -> verify.py)
for Myrvold's eight unresolved cases {SX, UX, VX, WX, XX, UU, UW, WW} of an
orthogonal pair (A,B) both orthogonal to a hypothetical third square with a
4x4 Latin subsquare.

Each SAT solution is a pair (P,Q) of 10x10 Latin squares that are mutual
"transversal representations": for each row i of P, the cells (k,j) of Q with
Q[k][j]=P[i][j] form a transversal of Q, and the 10 transversals from the 10
rows of P are disjoint.  Writing M[k][j]=i for the transversal index using cell
(k,j) yields a Latin square M orthogonal to Q (and symmetrically for P).  Each
saved pair {"L", "B"} was verified independently in Python: L and B are Latin
squares (every row/column a permutation of 0..9) and orthogonal (all 100
ordered pairs (L[r][c],B[r][c]) distinct).

Columns: transversals(L) / transversals(B) = number of transversals of each
square alone; common = number of transversals of L that are simultaneously
transversals of B (a third square orthogonal to both exists iff 10 disjoint
common transversals exist, so common=0 rules that out immediately for that pair).

| file | Myrvold case | construction | transversals(L) | transversals(B) | common | solver log |
|------|--------------|--------------|-----------------|-----------------|--------|------------|
| pair_1.json | WW | L=Q (TR type W), B=mate derived from rows of P (type W) as transversals of Q | 848 | 784 | 0 | Myrvold-MOLS run.sh WW, kissat seed 1, 3144s CPU (log WW-1.log) |
| pair_2.json | WW | L=P (TR type W), B=mate derived from rows of Q (type W) as transversals of P | 896 | 784 | 0 | Myrvold-MOLS run.sh WW, kissat seed 1, 3144s CPU (log WW-1.log) |
