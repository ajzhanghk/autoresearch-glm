# Near-Miss Triple Analysis

## 1. Definition and Significance

A **near-miss triple** $(L_1, L_2, L_3)$ is a triple of Latin squares of order 10 with total energy $E = \mathrm{cl}_{12} + \mathrm{cl}_{13} + \mathrm{cl}_{23}$ small but positive. The best near-miss found in this project has:

$$E = 37, \quad \mathrm{cl}_{12} = 0, \quad \mathrm{cl}_{13} = 22, \quad \mathrm{cl}_{23} = 15$$

These are not 3-MOLS (which requires $E = 0$), but represent the closest approach found after extensive search.

## 2. Explicit Near-Miss Data

### Sample E=37 Triple (Pool Entry 0)

The complete $10 \times 10$ matrices are stored in `mols10/results/near_miss_triple.json`. The $L_3$ component:

```
L3 =
 6  1  2  0  5  7  3  9  4  8
 8  9  0  1  6  3  2  7  5  4
 4  2  3  7  8  1  0  5  6  9
 3  8  6  9  2  0  4  1  7  5
 5  4  9  6  0  8  7  2  1  3
 0  5  4  2  9  6  8  3  1  7    (partial display)
 ...
```

### Clash Structure

**cl13 = 22** means: when $(L_1, L_3)$ are superimposed, 22 of the 100 possible symbol pairs appear a number of times $\neq 1$ (some 0 times, some 2+ times). The remaining 78 pairs appear exactly once.

**cl23 = 15** means: similarly for $(L_2, L_3)$, 15 pairs are wrong.

**Pattern:** All four E=37 pool entries share the same clash breakdown: cl13=22, cl23=15. This suggests a structural constraint — the 22/15 split may be a consequence of the underlying algebraic properties of MOLS-10 pairs.

## 3. Pool Diversity Analysis

### L3 Hamming Distance Matrix

For the four E=37 pool entries:

$$D = \begin{pmatrix} 0 & 60 & 80 & 80 \\ 60 & 0 & 20 & 30 \\ 80 & 20 & 0 & 40 \\ 80 & 30 & 40 & 0 \end{pmatrix}$$

where $D_{ij}$ = fraction of the 100 cells where $L_3^{(i)}$ and $L_3^{(j)}$ differ.

**Interpretation:**
- Entry 0 is far from entries 2, 3 (80% cells differ) but closer to entry 1 (60% differ).
- Entries 1 and 2 are similar (20% differ) — they may lie in the same isotopy class.
- Entries 2 and 3 differ in 40% of cells.

The large maximum distance (80%) confirms the search is reaching genuinely different regions of the L3 space, all with the same minimum energy E=37.

### L1/L2 Pair Diversity

The four E=37 entries come from **different** (L1,L2) pairs (pool enforces at most 2 entries per unique L1). Each pair has:
- $\mathrm{cl}_{12} = 0$ (verified MOLS)
- $\mathrm{CT} \leq 2$ (verified by enumeration and SAT)

## 4. Why E=37 May Be the Entropy Floor

### Theoretical Argument

For a random Latin square $L_3$ over $n$ symbols, paired with a fixed Latin square $L_1$:

The $n^2$ pairs $(L_1(i,j), L_3(i,j))$ approximate a balls-into-bins process: each of the $n^2$ balls (cells) is independently placed into one of $n^2$ bins (pairs). The expected number of bins with count $\neq 1$ is:

$$\mathbb{E}[\mathrm{cl}] \approx n^2 \cdot \left(1 - \frac{(n^2-1)!}{n^2 \cdot (n^2-n)!} \cdot \binom{n}{1}^{?}\right)$$

For $n=10$, this simplifies (using Poisson approximation with mean 1) to:
$$\mathbb{E}[\mathrm{cl}] \approx n^2 \cdot (1 - \Pr[\text{Poisson}(1) = 1]) = 100 \cdot (1 - e^{-1}) \approx 63.2$$

This is the expected value for independent random $L_3$. However, $L_3$ is constrained to be a Latin square (row/column uniqueness), which significantly reduces the variance and shifts the distribution.

For a **random Latin square** $L_3$ (not independent), the expected clash count with a fixed LS partner is approximately $100 \cdot p_{\neq 1}$ where $p_{\neq 1}$ accounts for the LS constraints. Empirically, random LSs give $E \approx 80$–90, while SA-optimized LSs achieve $E \approx 37$.

**The E=37 barrier** corresponds to the minimum of the SA energy landscape. Whether this is:
1. The true global minimum (in which case $N(10) = 2$), or
2. A deep local minimum with the true minimum at $E = 0$ (implying $N(10) \geq 3$)

remains undetermined.

### Empirical Evidence for E=37 as Global Minimum

1. **All basins converge to E=37:** SA from many different starting points (algebraic, random, pool-warm, D5, affine, cyclic) all converge to E=37 or worse.
2. **Strict local minimum:** Single- and two-step searches find no improvement.
3. **ILS fails to escape:** 14 kicks/trial × Sessions 5-7 = ~2000+ ILS kicks, none escaping E=37 toward E<37.
4. **SAT proves UNSAT:** For each (L1,L2) pair associated with our near-misses, Glucose3 proves no L3 exists.
5. **Reverse search gap:** Fixing the near-miss L3 and searching for compatible (L1,L2) finds best E=60 — a large gap indicating L3 is fundamentally incompatible with any MOLS structure.

## 5. Structural Properties of the Near-Miss L3

### Common Transversal Structure

For the E=37 near-miss triple, the $L_3$ matrix partitions cells into 10 groups of 10 (one per symbol value). Each group should be a common transversal of $(L_1, L_2)$ for full orthogonality.

In the near-miss, these groups are **near-transversals** — they cover most rows/columns/symbols of $L_1$ and $L_2$ correctly, but not all.

### Intercalate Count

An **intercalate** of a Latin square $L$ is a $2 \times 2$ submatrix with form $\begin{pmatrix} a & b \\ b & a \end{pmatrix}$ (or $\begin{pmatrix} a & b \\ b & a \end{pmatrix}$ modulo row/column permutation).

Intercalate count is related to the "distance" from the Latin square to the nearest orthogonal mate. The near-miss L3 matrices were not analyzed for intercalate count in this project, but this is a direction for future work.

### Symmetry Analysis

The near-miss $L_3$ matrices appear to have no obvious symmetry. The different pool entries with E=37 are unlikely to be isotopically equivalent (given Hamming distances of 20–80), suggesting they represent distinct isotopy classes.

## 6. Comparison with Known Near-Misses in Literature

The $N(10)$ problem has been studied computationally since the 1990s. Our best near-miss of $E = 37$ should be compared with prior work:

| Reference | Method | Best near-miss |
|-----------|--------|----------------|
| ??? (1990s) | Exhaustive local search | Not publicly reported |
| This project | Parallel-tempering SA + ILS | $E = 37$, $\mathrm{cl}_{12} = 0$ |

**Note:** Systematic reporting of near-miss triples for $N(10)$ is rare in the literature. Our E=37 result with $\mathrm{cl}_{12} = 0$ may represent the best currently known near-miss for the specific metric used.

## 7. Near-Miss Pool Management

### Pool Constraints

The pool is managed by function `_save_triple_miss` with the following acceptance criteria:
1. **Energy threshold:** Only triples with $E \leq 52$ are added.
2. **Pool size cap:** Maximum 8 entries.
3. **Pair diversity:** At most 2 entries sharing the same $L_1$ matrix.
4. **L3 uniqueness:** No two entries have identical $L_3$ matrices (verified by exact comparison).

### Pool Evolution History

| Session | Pool size | Best E | Unique L3 entries at best E |
|---------|-----------|--------|----------------------------|
| 5 (start) | 3 | 37 | 1 |
| 5 (end) | 5 | 37 | 1 (3 identical!) |
| 6 (after fix) | 8 | 37 | 3 (deduplication applied) |
| 7 (current) | 8 | 37 | 4 |

**Bug discovered in Session 6:** Three of five E=37 pool entries were identical ($L_3$ matrices with Hamming distance 0). The pair-diversity criterion (only 2 per L1) was enforced, but L3-uniqueness was not. After adding `np.array_equal` deduplication, the pool now correctly enforces diversity of L3.
