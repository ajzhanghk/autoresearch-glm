# Appendix: Data Tables and Raw Results

## A. The Four E=37 Near-Miss Triples

All four triples share $\mathrm{cl}_{12} = 0$, $\mathrm{cl}_{13} = 22$, $\mathrm{cl}_{23} = 15$.
Full matrices are in `mols10/results/near_miss_triple.json`.

### Triple 0 — L3 Matrix (found 2026-06-07 18:41)

```
Row 0:  6  1  2  0  5  7  3  9  4  8
Row 1:  8  9  0  1  6  3  2  7  5  4
Row 2:  4  2  3  7  8  1  0  5  6  9
Row 3:  (see JSON)
...
Row 9:  (see JSON)
```

### L3 Pairwise Hamming Distance Matrix

```
        T0   T1   T2   T3
T0 [     0   60   80   80 ]
T1 [    60    0   20   30 ]
T2 [    80   20    0   40 ]
T3 [    80   30   40    0 ]
```

Values are out of 100 cells (1 = all cells differ, 0 = identical).

---

## B. CT Distribution Across 311 MOLS-10 Pairs

| CT | Count | Fraction |
|----|-------|---------|
| 1  | 58    | 18.6%   |
| 2  | 253   | 81.4%   |
| ≥3 | 0     | 0%      |

### CT=1 Pairs (N=58)

These 58 pairs have exactly 1 common transversal. By the CT Necessity Theorem, no $L_3$ exists. The single CT is verified to not form a valid partition.

### CT=2 Pairs (N=253)

Includes TV-21 and all variants. The two CTs per pair are explicitly enumerable in < 1 second.

---

## C. Energy Distribution Histogram (Simulated Annealing)

Distribution of best energies found per SA trial (97-second budget, averaged across 5 workers × 18 trials = ~90 trials):

| Energy range | Trial frequency |
|-------------|----------------|
| E = 37      | ~80%           |
| E ∈ [38,42] | ~12%           |
| E ∈ [43,50] | ~6%            |
| E > 50      | ~2%            |

---

## D. Reverse Search Energy Distribution

Distribution of best energies found per 120-second reverse search trial (L3 fixed, optimizing L1/L2):

| Energy range | Frequency |
|-------------|-----------|
| E ≤ 59      | 0%        |
| E ∈ [60,62] | ~10%      |
| E ∈ [63,65] | ~80%      |
| E > 65      | ~10%      |

All trials with $E \geq 60$ (required: $E = 0$). Gap of ≥60 energy units.

---

## E. SAT Worker Statistics

### Phase 1 (311 existing pairs)
- Total solve time: ~13 seconds
- Average per pair: 42ms
- Min: 8ms, Max: 68ms
- All UNSAT

### Phase 2 (new pairs, first 6 hours)
- Total trials: ~880
- MOLS pairs found: ~608 (68.9%)
- UNSAT certified: 604 (99.3% of MOLS pairs)
- SAT (L3 found): 0
- Timeouts: 0
- Rate: ~100 UNSAT certifications/hour

---

## F. Session Timeline

| Session | Date | Key changes | Best E |
|---------|------|------------|--------|
| 1 | Early 2026 | Initial 4-replica PT | ~80 |
| 2 | 2026 | Intercalate moves | ~60 |
| 3 | 2026 | Warm-start pool; kscramble | ~50 |
| 4 | 2026 | 6-replica PT; bigscramble | ~43 |
| 5 | 2026-06-07 | 7-replica PT; time-based ILS | **37** |
| 6 | 2026-06-07 | D5/affine/cyclic inits; L3 dedup fix | **37** |
| 7 | 2026-06-08 | SAT worker; pool-jump ILS; reverse worker | **37** |

---

## G. Bugs Table

| Bug | Session discovered | Effect | Fix |
|-----|------------------|--------|-----|
| Dead super-shake code | 6 | super-shake never executed (unreachable branch behind D5 check) | Reordered probability conditions |
| ILS step-count threshold | 5→6 | ILS fired every 0.5s (~2000 kicks/trial) instead of 8s interval | Replaced with time-based ILS_INTERVAL=8s |
| Global vs local stagnation | 6 | ILS counter reset on any accepted descent move (oscillated at E=37-38) | Track only global best improvement time |
| L3 pool duplicates | 6 | 3/5 E=37 entries had Hamming=0 (identical L3) | Added np.array_equal L3 deduplication in _save_triple_miss |
| Wrong pair file in SAT worker | 7 | _load_pairs() returned 5 pairs instead of 311 | Changed to load promising_pairs.json directly |

---

## H. Code Metrics

| File | Lines | Purpose |
|------|-------|---------|
| mols_l3_search.py | ~2300 | Primary search engine |
| mols_reverse_search.py | ~275 | Reverse search worker |
| mols_sat_worker.py | ~195 | SAT exact solver worker |
| mols_adaptive.py | ~400 | Core utilities |

Total: ~3170 lines of Python.

---

## I. Hardware and Runtime

- **Machine:** Linux 6.18.5, single server
- **Cores available:** 8+
- **Workers:** 7 parallel (one per core)
- **Total SA steps (estimated):** 84,000 steps/second × runtime hours
- **Pool size at project end:** 8 triples (E=37 to E=44)
- **Pairs pool size:** 311

---

## J. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $N(n)$ | Maximum number of MOLS of order $n$ |
| $L_i$ | Latin square $i$ (10×10 array, symbols 0–9) |
| $L_i \perp L_j$ | $L_i$ and $L_j$ are orthogonal |
| $\mathrm{cl}(A,B)$ | Clash count: pairs appearing ≠1 times |
| $E$ | Total energy $= \mathrm{cl}_{12} + \mathrm{cl}_{13} + \mathrm{cl}_{23}$ |
| $\mathrm{CT}(L_1,L_2)$ | Common transversal count of pair |
| SA | Simulated annealing |
| PT | Parallel tempering (replica exchange SA) |
| ILS | Iterated local search |
| CDCL | Conflict-driven clause learning (SAT algorithm) |
| MOLS | Mutually orthogonal Latin squares |
