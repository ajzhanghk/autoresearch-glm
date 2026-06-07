# MOLS(10) Search — Findings

**Date:** 2026-06-07  
**Branch:** `claude/mols-order-10-search-yfQXK`  
**Repo:** `ajzhanghk/autoresearch-glm`

---

## The Problem

Prove N(10) ≥ 3 by finding a third Latin square L3 mutually orthogonal to two existing ones (L1, L2). This would mean three Mutually Orthogonal Latin Squares of order 10 exist.

**Background:**
- Parker (1960): N(10) ≥ 2 — two MOLS of order 10 exist.
- Lam, Thiel & Swiercz (1989): No projective plane of order 10 → N(10) ≤ 8.
- **N(10) = 2 or 3 or ... 8** — the exact value is unknown.
- Most experts believe N(10) = 2 (no third MOLS exists), but it is unproven.

---

## Key Mathematical Discovery: The CT Barrier

A **Common Transversal (CT)** of a pair (L1, L2) is a set of n=10 cells — one per row, one per column — where all L1-values are distinct AND all L2-values are distinct.

**Critical Theorem:** L3 ⊥ L1 and L3 ⊥ L2 if and only if the rows of L3 form exactly n=10 disjoint CTs that partition all n²=100 cells. Therefore:

> **CT_count(L1, L2) < 10 → L3 provably does NOT exist for that pair.**

This is an isotopy invariant: row/column/symbol permutations preserve CT count.

### Empirical Finding

Every single MOLS(10) pair tested has CT_count ≤ 2:

| CT Count | Pairs in pool |
|----------|--------------|
| 0        | (filtered out) |
| 1        | 26 |
| 2        | 36 |
| **≥ 3**  | **0** (never found) |

- TV_21 (Parker 1960 construction): exactly CT = 2, exhaustively verified.
- 12 different L2 mates of TV_21's L1: all have CT = 2.
- 62 total pairs screened across multiple construction methods: max CT = 2.

**Consequence:** Direct L3 search (`sa_l3_pair`) is mathematically futile for all known pairs. The search must either find a new pair with CT ≥ 10 or use a CT-free joint search.

---

## Search Architecture

Three parallel workers run indefinitely using an adaptive portfolio:

| Strategy | Budget allocation | Purpose |
|---|---|---|
| `sa_triple` | 50% | Joint SA over (L1, L2, L3); minimises Σ clashes(Li, Lj). CT-free — can find any valid triple. **Primary strategy.** |
| `multi_decomp` | 30% | High-throughput screening via transversal enumeration + AlgorithmX to find pairs with CT > 2. |
| `sa_ct_climb` | 20% | SA attempting to push CT above 2 via intercalate flips on known pairs. |
| `sa_l3_pair` | 0% (conditional) | AlgorithmX exact-cover search for L3; activates only if a CT ≥ 10 pair is ever found. |

### Near-Miss Sharing

- `near_miss_l3.json`: Best partial L3 states (clashes ≤ 20) saved for cross-worker warm-starts on `sa_l3_pair`.
- `near_miss_triple.json`: Best (L1,L2,L3) triple states (total clashes ≤ 40) saved for cross-worker warm-starts on `sa_triple`. *(Not yet triggered — requires ≤ 40 total clashes.)*

---

## Optimizations Applied

### count_clashes speedup (4.5×)
**File:** `mols10/mols_adaptive.py`

Replaced `np.add.at` (slow Python-level loop) with `np.bincount`:

```python
# Before (56K calls/s):
counts = np.zeros(n * n, dtype=np.int32)
idx = L1.ravel().astype(np.int32) * n + L2.ravel().astype(np.int32)
np.add.at(counts, idx, 1)
return int(np.sum(np.maximum(0, counts - 1)))

# After (253K calls/s):
pairs = L1.ravel().astype(np.int32) * n + L2.ravel().astype(np.int32)
counts = np.bincount(pairs, minlength=n * n)
return int(n * n - np.count_nonzero(counts))
```

### Incremental pairwise clash tracking in sa_triple
**File:** `mols10/mols_l3_search.py`

Track `cl12`, `cl13`, `cl23` individually. When a move targets square `i`, only recompute the 2 pairs involving that square (saves 1 of 3 clash computations per step):

```python
if tgt == 0:   # moved L1
    new_cl = count_clashes(Ln, L2) + count_clashes(Ln, L3) + cl23
elif tgt == 1: # moved L2
    new_cl = count_clashes(L1, Ln) + cl13 + count_clashes(Ln, L3)
else:          # moved L3
    new_cl = cl12 + count_clashes(L1, Ln) + count_clashes(L2, Ln)
```

### L3-biased moves
When `cl12 < 5` (pair is nearly perfect), 70% of moves target L3, 15% L1, 15% L2. This preserves a good pair while aggressively searching for L3.

### Net performance improvement
| Metric | Before | After |
|--------|--------|-------|
| count_clashes | 56K/s | 253K/s |
| SA steps/s (sa_triple) | ~16K | ~63K |
| Steps per 39s trial | ~625K | ~2.46M |
| **Speedup** | — | **~4×** |

---

## Results

### sa_triple frontier (total clashes = clashes(L1,L2) + clashes(L1,L3) + clashes(L2,L3))

| Session | Best total clashes | Notes |
|---------|-------------------|-------|
| Previous session | 51 | Pre-optimization |
| Current session (pre-opt) | 58 | Workers A, B, C |
| Current session (post-opt) | **56** | Worker B, trial 2 after 4× speedup |
| **Target** | **0** | Proves N(10) ≥ 3 |

When warm-starting from a known MOLS pair (clashes(L1,L2)=0), total clashes = clashes(L1,L3) + clashes(L2,L3). Getting this to 0 while also having L1,L2 orthogonal would prove N(10) ≥ 3.

### Bugs Discovered and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `_in_row_swap` swaps within a row, breaking column uniqueness | SA explored non-Latin states; "10 clashes" result from early session was for a non-Latin L3 | Removed from SA entirely |
| `multi_decomp` returning CT=-1 | Random L1s often have no OLS mate | Bias toward known good L1 from pool |
| `sa_l3_pair` allocated 50% compute on CT<10 pairs | Mathematically futile | Redesigned portfolio; sa_l3_pair only runs if CT≥10 |

---

## Prognosis

The CT=2 ceiling across 62 tested pairs, combined with `sa_triple` stagnating at 55–60 total clashes, is consistent with the expert consensus that **N(10) = 2** (no third MOLS exists). The search continues as directed — any further progress would be a genuine mathematical result.

**Watching for:**
1. Any pair with CT ≥ 3 (would be novel; CT ≥ 10 would enable AlgorithmX L3 search)
2. `near_miss_triple.json` creation (total clashes ≤ 40 triggers cross-worker sharing)
3. Total clashes = 0 (proves N(10) ≥ 3)

---

## File Map

```
mols10/
├── mols_l3_search.py          # Primary search engine (adaptive portfolio)
├── mols_adaptive.py           # Latin square primitives, count_clashes
├── mols_search.py             # AlgorithmX, CSP, verify_mols
├── verify_results.py          # Standalone JSON result verifier
├── docs/
│   └── findings.md            # This file
└── results/
    ├── l3_run_A.log           # Worker A log (seed 11111)
    ├── l3_run_B.log           # Worker B log (seed 22222)
    ├── l3_run_C.log           # Worker C log (seed 33333)
    ├── l3_search_log.tsv      # Structured per-trial log
    ├── promising_pairs.json   # CT > 0 pairs pool (62 pairs, max CT=2)
    ├── near_miss_l3.json      # Best partial L3 states (not yet created)
    └── near_miss_triple.json  # Best triple states (not yet created)
```

## References

- Parker, E.T. (1960). Orthogonal Latin squares. *Proc. Natl. Acad. Sci. USA*, 47, 859–862.
- Hall, M. & Paige, L.J. (1955). Complete mappings of finite groups. *Pacific J. Math.*, 5, 541–549.
- Lam, C.W.H., Thiel, L. & Swiercz, S. (1989). The non-existence of finite projective planes of order 10. *Canad. J. Math.*, 41, 1117–1123.
- Wanless, I.M. & Webb, B.S. (2011). The existence of latin squares without orthogonal mates. *Des. Codes Cryptogr.*, 40, 131–135.
- McKay, B.D., Meynert, A. & Myrvold, W. (2007). Small Latin squares, quasigroups, and loops. *J. Combin. Des.*, 15, 98–119.
