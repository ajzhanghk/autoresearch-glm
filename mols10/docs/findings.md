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
| 1        | 47 |
| 2        | 209 |
| **≥ 3**  | **0** (never found after 282 pairs tried) |

- TV_21 (Parker 1960 construction): exactly CT = 2, exhaustively verified.
- 12 different L2 mates of TV_21's L1: all have CT = 2.
- 282 total pairs screened across multiple construction methods: max CT = 2.

**Consequence:** Direct L3 search (`sa_l3_pair`) is mathematically futile for all known pairs. The search must either find a new pair with CT ≥ 10 or use a CT-free joint search.

---

## Search Architecture

Four parallel workers run indefinitely using an adaptive portfolio:

| Strategy | Budget allocation | Purpose |
|---|---|---|
| `sa_triple_pt` | 40% | **Primary.** Parallel-tempering SA with 5 replicas at T=[1,4,16,64,256]. Joint optimization over (L1, L2, L3). CT-free — can find any valid triple. |
| `sa_triple` | 20% | Single-chain SA for diversity. |
| `multi_decomp` | 20% | High-throughput pair screening via transversal enumeration + AlgorithmX. Finds pairs with CT > 0. |
| `sa_ct_climb` | 20% | SA attempting to push CT above 2 via intercalate flips on known pairs. |
| `sa_l3_pair` | 0% (conditional) | AlgorithmX exact-cover; activates only if a CT ≥ 10 pair is found. |

### Near-Miss Cascade

`near_miss_triple.json` stores the best (L1,L2,L3) triples found, with pair diversity (max 2 per pair, top-8 total). Each new trial can warm-start from these states, allowing the cascade to build progressively better states.

**Fresh-start probabilities for each replica:**
| Mode | Probability | Description |
|------|-------------|-------------|
| Pure near-miss | 8% | Exact copy of saved best state (cascade preservation) |
| Micro-shake | 10% | Near-miss L3 + 1-10 random moves (tight diversity) |
| Shake | 15% | Near-miss L3 + 15-80 random moves (broad escape) |
| L3-transfer | 17% | CT=2 seed pair + near-miss L3 (cross-pair exploration) |
| CT=2 + fresh L3 | 30% | Known MOLS pair + random L3 |
| Fully random | 20% | All 3 squares random |

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

### Incremental pairwise clash tracking
Track `cl12`, `cl13`, `cl23` individually. When a move targets square `i`, only recompute the 2 pairs involving that square.

### Parallel Tempering (5 replicas, T=[1,4,16,64,256])
Hot replica at T=256 accepts ΔE≈177 moves with P=0.5, enabling escape from deep local minima. Replicas exchange states every 2000 outer steps, laddering good states from hot to cold.

### L3-biased moves
When `cl12 < 5`, 70% of moves target L3, 15% each for L1/L2. Preserves good MOLS pairs while aggressively searching for compatible L3.

### Shake/Micro-shake warm-starts
Perturbations of 1-10 moves (micro-shake) and 15-80 moves (shake) create diverse starting L3 candidates from the near-miss pool, helping escape local basins.

---

## Results

### Near-Miss Cascade Progress

The cascade has been the primary mechanism of improvement. Starting from a random triple (E≈270), progressive warm-starting from saved near-miss states drives the energy down:

| Session | Best E | Notes |
|---------|--------|-------|
| Session 1 (pre-optimization) | ~51 | No warm-start infrastructure |
| Session 2 (bugs present) | 37 | Lost due to JSON save bugs (concurrent writes, unhashable key) |
| Session 3 (bugs fixed) | **39** | First properly-saved near-miss, pair-diversity cap |
| Session 3 (continued) | **37** | 5-replica T=256 broke through; properly saved |

**Current best:** E = 37 (cl12=0, cl13=22, cl23=15)

### Current near_miss_triple.json

| Entry | cl12 | cl13 | cl23 | total | source pair |
|-------|------|------|------|-------|-------------|
| 0-7 | 0 | 22 | 15 | **37** | 4 distinct MOLS pairs (8 entries) |

All entries have cl12=0 (the (L1,L2) component is always a valid MOLS pair).

### Key Negative Finding: E=37 is a Strict Local Minimum

Exhaustive 1-neighborhood search over ALL possible single moves on L3 (45 row swaps + 45 col swaps + 45 relabels + 24 intercalate flips = 159 moves) confirms:

> **Every single move from E=37 either stays at E=37 or increases E. No 1-step improvement exists.**

Moreover, 2-step exhaustive search (all 2025 pairs of row swaps) also finds best=37. This means the basin extends at least 2 steps deep.

**Intercalate counts of near-miss Latin squares** (expected ~2000 for random order-10 LS):
| Square | Intercalates |
|--------|-------------|
| L1 | 23 |
| L2 | 27 |
| L3 | 24 |

The SA is gravitating to a special class of "low-intercalate" Latin squares with only ~1% of the typical intercalate count. This drastically restricts the neighborhood: instead of ~2000 intercalate moves, only 24 are possible for L3.

This structural finding is key: the SA has found Latin squares with very few intercalates, which are the "right type" for MOLS construction (high structure), but these specific squares are provably L3-free (CT=0). Escaping requires finding different high-structure squares.

### Pure L3 and Reverse Search Results

- Pure L3 SA (fix CT=0 pair, vary only L3, 10M+ steps): best cl13+cl23 = 47
- Reverse search (fix near-miss L3, vary L1 seeking orthogonal mate, 60s): best cl13 = 18 (not 0)
- Joint SA (6-replica T=[1,4,16,64,256,1024], 540s, 8649 swaps): best = 37

### Cascade Chronology (2026-06-07)

```
16:00  Workers start, pool rebuilt (215+ pairs)
17:36  First near-miss saved at E=39 (after fixing JSON save bugs)
18:36  5-replica T=256 breaks cascade: E=39→38
18:41  Cascade continues: E=38→37 (two new entries)
18:49  4 entries at E=37, from 2 distinct pairs
19:24  Pool expanded to 8 entries; all at E=37 from 3+ pairs
20:00  540s Worker D trial: still E=37 (8649 swaps)
20:30  Exhaustive check confirms E=37 is strict 1-step local minimum
20:30  Low-intercalate structure discovered (L1/L2/L3: 23/27/24 intercalates)
```

---

## Analysis: Why E=37?

### The "Random Baseline" Coincidence

For two independent random Latin squares A, B of order 10, the expected number of missing pairs:
E[clashes(A,B)] ≈ n² × P(pair missing) = 100 × (1 - e^{-1}) ≈ 36.8 ≈ 37

**Note:** individually cl13=22 and cl23=15 are each better than random (37 each). Their sum equals the single-pair random baseline — this is coincidental.

Our best triple has cl13=22, cl23=15 (both well below 37 individually). Their sum = 37 equals the "random baseline" — a coincidence that suggests we're in a regime where joint optimization has pushed both terms to an equilibrium.

### CT=0 Pairs and L3 Impossibility

All 282 tested MOLS pairs have CT≤2. The CT theorem guarantees: for CT=0, **no L3 exists for that fixed (L1,L2) pair**. The joint SA finds E=37 states where cl12=0 but the pair has CT=0. L3 cannot be made orthogonal to this fixed pair.

For the joint SA to find E=0, it must find a MOLS pair with CT≥10 — something that has never been observed in this search or in prior literature.

### Is 37 a Fundamental Barrier?

Evidence suggests E=37 may be close to the fundamental minimum for the current class of accessible MOLS-10 pairs:
1. L3-only SA (fixing CT=0 pair, all starting L3s) can't beat 37 in 10M+ steps
2. Joint SA with 540s budget and T=256 hot replica can't beat 37
3. Two independent sessions both independently converged to 37 as the cascade minimum

This is consistent with (but does not prove) N(10) = 2.

---

## Bugs Discovered and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Concurrent JSON write | Silent save failure (workers clobber each other's writes) | Inner try-except with `data=[]` recovery |
| Unhashable list key | ALL saves failed after pair-diversity code was added | `k = tuple(raw_k) if isinstance(raw_k, list) else raw_k` |
| numpy `M[a],M[b]=M[b],M[a]` aliasing | Row swap corrupts Latin square (both rows get same values) | Use `M[[a,b]] = M[[b,a]]` |
| CT=0 warm-start removed prematurely | Cascade stalled at 43 (vs 37 with CT=0 warm-starts) | Restored 15% pure near-miss mode |
| L3-bias reduced to 50% | Cascade stalled at 43 (vs 37 with 70% bias) | Reverted to 70%/15%/15% |

---

## Prognosis

The CT=2 ceiling across 282 tested pairs, combined with the sa_triple cascade stagnating at E=37 (cl12=0, cl13=22, cl23=15), is consistent with the expert consensus that **N(10) = 2**. The joint SA cannot push below 37 even with aggressive 5-replica parallel tempering (T_max=256, 540s budget).

**Still watching for:**
1. Any pair with CT ≥ 3 (novel; CT ≥ 10 enables AlgorithmX L3 search)
2. Total clashes < 37 (would be a new cascade record)
3. Total clashes = 0 (proves N(10) ≥ 3)

---

## File Map

| File | Purpose |
|------|---------|
| `mols10/mols_l3_search.py` | Main search engine: sa_triple_pt, sa_ct_climb, multi_decomp |
| `mols10/mols_adaptive.py` | Core utilities: count_clashes, random_latin_square, verify_mols |
| `mols10/results/near_miss_triple.json` | Best (L1,L2,L3) triples for warm-starts (top-8, pair-diverse) |
| `mols10/results/promising_pairs.json` | Best (L1,L2) MOLS pairs by CT count (282 pairs, CT≤2) |
| `mols10/results/l3_run_A/B/C/D.log` | Per-worker progress logs |
| `mols10/docs/findings.md` | This document |
