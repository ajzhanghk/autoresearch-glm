# The N(10) Problem: Finding Three Mutually Orthogonal Latin Squares of Order 10

## 1. Problem Definition

A **Latin square** of order $n$ is an $n \times n$ array filled with $n$ distinct symbols, each occurring exactly once in each row and each column.

Two Latin squares $L_1$ and $L_2$ of order $n$ are **orthogonal** (written $L_1 \perp L_2$) if, when superimposed, every ordered pair of symbols $(a, b)$ with $a \in L_1$, $b \in L_2$ appears in exactly one cell.

The **number** $N(n)$ is defined as the maximum number of mutually orthogonal Latin squares (MOLS) of order $n$ that can simultaneously exist.

**The N(10) problem:** Determine the exact value of $N(10)$.

## 2. Historical Background

| Year | Result | Reference |
|------|--------|-----------|
| 1782 | Euler conjectured $N(n) = 1$ for $n \equiv 2 \pmod{4}$ (Euler's conjecture) | Euler |
| 1960 | $N(10) \geq 2$ — two MOLS of order 10 exist, disproving Euler's conjecture for $n=10$ | Parker [1] |
| 1960 | $N(6) = 1$ proven (Euler's conjecture holds for $n=6$) | Bose, Shrikhande, Parker |
| 1989 | No projective plane of order 10 exists, hence $N(10) \leq 8$ | Lam, Thiel, Swiercz [2] |
| Present | $N(10) \in \{2, 3, 4, 5, 6, 7, 8\}$ — the exact value remains unknown | — |

The best current bounds are:
$$2 \leq N(10) \leq 8$$

Most combinatorialists conjecture $N(10) = 2$, but no proof exists.

## 3. Why N(10) = 2 Is Believed

1. **Algebraic obstruction:** The standard algebraic constructions (based on Galois fields $GF(q)$ for prime powers $q$) give $N(q) \geq q-1$. Since 10 is not a prime power, no such construction applies, and no ad-hoc construction for $N(10) \geq 3$ has been found.

2. **Exhaustive pair analysis:** Every known MOLS-10 pair has a common transversal (CT) count of at most 2. Since $N(10) \geq 3$ requires a pair with $\mathrm{CT} \geq 10$ (see Section 4 of the theory document), this is a strong negative indicator.

3. **Computational searches:** Multiple independent computational searches spanning decades (including this project) have failed to find three MOLS of order 10.

## 4. Our Research Goal

Prove $N(10) \geq 3$ by exhibiting an explicit triple $(L_1, L_2, L_3)$ of mutually orthogonal $10 \times 10$ Latin squares, or accumulate evidence for $N(10) = 2$.

### Equivalence Relations

Finding 3-MOLS of order 10 is equivalent to:
- A **transversal design** $\mathrm{TD}(3, 10)$
- An **orthogonal array** $\mathrm{OA}(100, 3, 10, 2)$
- A **3-net** of order 10
- A **room square** (for special cases)

## 5. Project Scope

This computational research project conducted a systematic search via:
- Parallel simulated annealing (SA) with parallel tempering
- SAT solver encoding (Glucose3 / CDCL)
- Common transversal enumeration and maximization
- Near-miss analysis and reverse search
- Multiple initialization strategies spanning algebraic and random constructions

**Duration:** Sessions 1–7 (2026-06), running 5–7 parallel workers continuously.  
**Codebase:** `mols10/` directory, branch `claude/mols-order-10-search-yfQXK`.

## References

[1] Parker, E. T. (1960). "Orthogonal Latin squares." *Proceedings of the National Academy of Sciences*, 47(6), 859–862.

[2] Lam, C. W. H., Thiel, L., & Swiercz, S. (1989). "The nonexistence of finite projective planes of order 10." *Canadian Journal of Mathematics*, 41(6), 1117–1123.

[3] Wanless, I. M., & Webb, B. S. (2011). "The existence of Latin squares without orthogonal mates." *Designs, Codes and Cryptography*, 60(2), 143–151.

[4] McKay, B. D., Meynert, A., & Myrvold, W. (2007). "Small Latin squares, quasigroups, and loops." *Journal of Combinatorial Designs*, 15(2), 98–119.
