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
| Session 4 (bugs introduced) | ~~22~~ ~~20~~ | **FAKE** — invalid L3 from cyclic offset=5 bug |
| Session 5 (bugs fixed) | **37** | Restored valid pool; D5/k-scramble/super-shake running |
| Session 6 (current) | **37** | ILS, bigscramble, reverse search, L3-diversity; E=37 holds |

**Current best (verified valid):** E = 37 (cl12=0, cl13=22, cl23=15)

### Current near_miss_triple.json (Session 6)

| Entry | total E | cl12 | Notes |
|-------|---------|------|-------|
| 0–2 | **37** | 0 | 3 entries, all with distinct L3 matrices (Hamming dist 20-80) |
| 3–4 | 43 | 0 | 2 new entries from distinct L3 warm-starts |
| 5–7 | 46-48 | 0 | 3 further entries; all 8 have unique L3s (enforced by new save logic) |

All entries verified: cl12=0, valid Latin squares, no duplicate L3 matrices.

### Pool L3 Diversity Finding (2026-06-07 ~21:50)

Discovered that 3 of 5 E=37 pool entries had IDENTICAL L3 matrices. The pair-diversity
criterion (max 2 per L1 pair) ensures diverse L1/L2 pairs but does NOT prevent repeated L3s.

**Fix**: `_save_triple_miss` now also rejects entries with L3 identical to any existing pool L3.
Pool pruned from 8→5 unique-L3 entries, then grew back to 8 with genuinely diverse L3s.

### Critical Bug Discovery: Invalid Latin Squares in Pool (2026-06-07 ~20:00-21:30)

The cyclic L3 init code used `r.randint(1, n-1)` which picks offsets with gcd(offset,n)>1.
For n=10, offset=5 gives L[i,j]=(i+5j)%10 which has rows with only 2 distinct values — NOT a Latin square.

The SA with an invalid L3 still runs and finds "low clash" states, but the clash counts are meaningless (the count_clashes function also compares against an invalid square). All 8 entries saved as "E=22" and "E=20" had invalid L3 squares. The LS validity check was missing from `_save_triple_miss`.

**Fixes applied:**
- Cyclic mode: `valid_offsets = [k for k in range(1, n) if gcd(k, n) == 1]` (only offsets 1,3,7,9 for n=10)
- `_save_triple_miss`: validates all rows and columns of L1, L2, L3 before saving
- `fresh_state`: validates initial triple, falls back to random LS if invalid

### Key Negative Finding: E=37 is a Strict Local Minimum

Exhaustive 1-neighborhood search over ALL possible single moves on L3 (45 row swaps + 45 col swaps + 45 relabels + 24 intercalate flips = 159 moves) confirms:

> **Every single move from E=37 either stays at E=37 or increases E. No 1-step improvement exists.**

Moreover, 2-step exhaustive search (all 2025 pairs of row swaps) also finds best=37. This means the basin extends at least 2 steps deep.

**Intercalate counts of near-miss Latin squares** (actual computation for n=10):
| Square | Intercalates | Notes |
|--------|-------------|-------|
| L1 | 23 | Near-miss at E=37 |
| L2 | 27 | Near-miss at E=37 |
| L3 (near-miss E=37) | 24 | Very restricted neighborhood |
| Cyclic Z10 (offsets 1,3,7,9) | 25 | Valid algebraic starts |
| D5 Cayley table (dihedral group) | 125 | Non-abelian, much richer |
| Affine (a*i+b*j)%10, valid (a,b) | 25 | Same family as cyclic |

Note: "~2000 intercalates for random LS" figure in earlier notes was incorrect. All valid cyclic/affine squares of order 10 have exactly 25 intercalates.

**New structural innovation (Session 5):** D5 Cayley table has 125 intercalates, giving the SA a much richer 125-move intercalate neighborhood vs. 24 for near-miss. Combined with k-scramble moves (3-5 row permutations), super-shake (100-400 moves from near-miss), and full isotopy for all algebraic init modes, this explores regions inaccessible to prior 1-2 step searches.

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
20:30  Low-intercalate structure (L1/L2/L3: 23/27/24 intercalates) identified
~20:30 Cyclic L3 mode introduced (BUG: invalid offset=5 allowed → fake "cascade")
~20:57 FAKE E=22 and E=20 "discoveries" — all invalid L3s (2 distinct values/row)
21:30  Bug discovered; all fake entries purged; valid pool (E=37) restored
21:30  Bug fixes: valid_offsets, LS validation in save/init, D5 mode, k-scramble
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
2. Joint SA with 540s budget, 7 replicas (T=[1,4,16,64,256,1024,4096]), 28M+ moves/trial: still E=37
3. Three independent sessions all independently converged to 37 as the cascade minimum
4. k-scramble moves (3-5 row permutations, exploring 3+-step neighborhood): no improvement
5. D5 Cayley table starts (125 intercalates, non-abelian structure): cascade back to E=37
6. Affine L3 starts (all valid offsets explored): cascade back to E=37
7. Super-shake (100-400 random moves from near-miss): cascade back to E=37

This is consistent with (but does not prove) N(10) = 2.

---

## Bugs Discovered and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Concurrent JSON write | Silent save failure | Inner try-except with `data=[]` recovery |
| Unhashable list key | ALL saves failed after pair-diversity code | `k = tuple(raw_k) if isinstance(raw_k, list) else raw_k` |
| numpy `M[a],M[b]=M[b],M[a]` aliasing | Row swap corrupts Latin square | Use `M[[a,b]] = M[[b,a]]` |
| CT=0 warm-start removed prematurely | Cascade stalled at 43 | Restored 15% pure near-miss mode |
| L3-bias reduced to 50% | Cascade stalled at 43 | Reverted to 70%/15%/15% |
| Cyclic mode offset bug (session 4) | **CRITICAL**: gcd(offset,n)>1 gives invalid LS; ALL "E=22" and "E=20" results were fake | Restrict to valid_offsets = {k: gcd(k,n)==1} = {1,3,7,9} for n=10 |
| No LS validation in save | Invalid L3 squares entered warm-start pool | Added row/column validation in _save_triple_miss and fresh_state |
| Duplicate L3 in pool | 3/5 E=37 entries identical; warm-starts converge to same minimum | Added L3-deduplication in _save_triple_miss |
| Dead super-shake code (session 6) | super-shake condition `roll<0.93` shadowed by D5 block; unreachable | Corrected probability chain boundaries |
| ILS step-count threshold | Resets on local descent, rarely triggers | Switched to time-based (8s) global stagnation tracking |

---

## Session 6 Additions (2026-06-07)

### New algorithmic features:
- **bigscramble move (k=6-9 rows)**: Larger jumps than kscramble (k=3-5); provides escape attempts beyond the 3-5 row neighborhood.
- **ILS (Iterated Local Search)**: Time-based stagnation trigger (8s after last best_E improvement); shakes L3 with 25-70 random moves on the cold replica. Activates ~14× per 97s trial.
- **Reverse search worker** (`mols_reverse_search.py`): Dedicated 6th worker fixing near-miss L3, searching for compatible (L1, L2) via 7-replica PT. Best found: E=64-65 (baseline=111 for random).
- **multi_decomp 50% random L1**: Increased from 1/3 to 1/2 to explore non-Parker L1 spaces more thoroughly (isotopy of Parker L1 is CT-invariant, so can only ever produce CT≤2 pairs).
- **L3-diversity enforcement**: Pool now rejects entries with duplicate L3 matrices (was: 3/5 E=37 entries identical).

### New negative findings:
- Reverse search best_E=64-65 (vs. 0 needed): near-miss L3 is not compatible with any pair (L1, L2) within the SA's reach.
- All 310 stored MOLS pairs have CT≤2 including 124 from multi_decomp random L1 starts (not just Parker-family).
- ILS with 14 kicks per trial (fresh L3 perturbations every ~8s) also cannot break E=37.

---

## Prognosis

The CT=2 ceiling across 310 tested pairs (124 from random L1 starts, not just Parker), combined with the sa_triple cascade stagnating at E=37 despite:
- 7-replica PT (T_max=4096)
- ILS shaking (14 times/trial, 25-70 moves per shake)
- bigscramble moves (k=6-9)
- D5/affine/cyclic init modes + reverse search
- Dedicated reverse search showing best (L1,L2) compatible with near-miss L3 is E=64 (vs. 0 needed)

...is consistent with the expert consensus that **N(10) = 2**.

**Search status (Session 6):**
- 6 workers running continuously: 4 × 97s + 1 × 600s + 1 × 120s reverse
- 7-replica parallel tempering T=[1,4,16,64,256,1024,4096]
- init modes: near-miss warm-start (8%), micro-shake (10%), shake (15%), L3-transfer (17%),
  CT=2 pair+fresh L3 (23%), cyclic Z10 (7%), affine (8%), D5 Cayley (5%), super-shake (3%),
  reverse (2%), random (2%)
- Moves: row/col/relabel/intercalate/kscramble(3-5)/bigscramble(6-9)
- Pool enforces both pair-diversity (max 2 per L1) and L3-uniqueness

**Still watching for:**
1. Any pair with CT ≥ 3 (novel; CT ≥ 10 enables AlgorithmX L3 search)
2. Total clashes < 37 (would be new cascade record)
3. Total clashes = 0 (proves N(10) ≥ 3)

---

## File Map

| File | Purpose |
|------|---------|
| `mols10/mols_l3_search.py` | Main search engine: sa_triple_pt, multi_decomp; 6-worker adaptive |
| `mols10/mols_reverse_search.py` | Dedicated reverse-search: fix L3, find compatible (L1,L2) |
| `mols10/mols_adaptive.py` | Core utilities: count_clashes, random_latin_square, verify_mols |
| `mols10/results/near_miss_triple.json` | Best (L1,L2,L3) triples for warm-starts (top-8, pair+L3 diverse) |
| `mols10/results/promising_pairs.json` | Best (L1,L2) MOLS pairs by CT count (310 pairs, CT≤2) |
| `mols10/results/l3_v3_s{42,137,271,503}.log` | Per-worker logs (97s budget) |
| `mols10/results/l3_longrun2.log` | Long-run worker log (600s budget) |
| `mols10/results/l3_reverse.log` | Reverse-search worker log |
| `mols10/docs/findings.md` | This document |
