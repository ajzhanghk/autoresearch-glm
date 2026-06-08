# Implementation Details

## 1. Codebase Structure

```
mols10/
├── mols_adaptive.py          # Core utilities (count_clashes, verify_mols, random LS)
├── mols_l3_search.py         # Primary search engine (5000+ lines)
│   ├── sa_triple()           # Single-chain SA on (L1,L2,L3)
│   ├── sa_triple_pt()        # 7-replica parallel-tempering SA [PRIMARY]
│   ├── sa_ct_climb()         # SA maximizing CT count on MOLS pairs
│   ├── multi_decomp()        # Transversal decomposition pair generation
│   ├── AdaptiveSearch        # Outer loop: portfolio of strategies
│   ├── _save_triple_miss()   # Pool management (energy + diversity constraints)
│   ├── _load_triple_misses() # Pool reader
│   ├── _load_pairs()         # MOLS pair reader
│   └── enumerate_common_transversals()  # CT enumeration
├── mols_reverse_search.py    # Fix L3, search for compatible (L1,L2)
├── mols_sat_worker.py        # Glucose3 exact SAT search for L3
└── results/
    ├── near_miss_triple.json # Pool of best (L1,L2,L3) triples
    ├── promising_pairs.json  # Pool of best MOLS pairs by CT
    ├── mols_pairs_collection.json  # Seed pairs (5 entries)
    ├── l3_v3_s{42,137,271,503}.log  # Per-worker SA logs
    ├── l3_longrun2.log       # Long-run worker log (seed=999)
    ├── l3_reverse.log        # Reverse search log
    └── l3_sat.log            # SAT worker log
```

## 2. Core Data Structures

### Latin Square Representation

All Latin squares are stored as `numpy.ndarray` with dtype `np.int8`, shape `(10, 10)`, values in `{0, 1, ..., 9}`.

### Near-Miss Pool Entry (JSON)

```json
{
  "ts": "2026-06-07T18:41:38",
  "clashes": 37,
  "cl12": 0,
  "L1_key": [0,1,2,...],
  "L1": [[...], ...],
  "L2": [[...], ...],
  "L3": [[...], ...]
}
```

### Promising Pair Entry (JSON)

```json
{
  "L1": [[...], ...],
  "L2": [[...], ...],
  "ct": 2
}
```

## 3. Key Algorithms — Implementation Details

### count_clashes (C extension via numpy)

```python
def count_clashes(A: np.ndarray, B: np.ndarray, n: int) -> int:
    """Count pairs (a,b) appearing != 1 time when A,B superimposed."""
    pair_counts = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            pair_counts[A[i,j], B[i,j]] += 1
    return int(np.sum(pair_counts != 1))
```

Implemented with vectorized numpy operations for performance.

### Intercalate Flip

```python
def _intercalate_flip(L, r1, r2, c1, c2):
    """Flip a 2x2 intercalate if one exists at (r1,r2) x (c1,c2)."""
    a, b = L[r1,c1], L[r1,c2]
    c, d = L[r2,c1], L[r2,c2]
    if a == d and b == c:  # intercalate pattern [a,b; b,a]
        Ln = L.copy()
        Ln[r1,c1], Ln[r1,c2] = b, a
        Ln[r2,c1], Ln[r2,c2] = a, b
        return Ln
    return None
```

### Common Transversal Enumeration

```python
def enumerate_common_transversals(L1, L2, n, max_count, deadline):
    """Backtracking enumeration of CTs with time cap."""
    cts = []
    def backtrack(row, rows_used, cols_used, l1_vals, l2_vals, cells):
        if len(cts) >= max_count or time.time() > deadline:
            return
        if row == n:
            cts.append(cells.copy())
            return
        for col in range(n):
            if col in cols_used:
                continue
            v1, v2 = L1[row,col], L2[row,col]
            if v1 in l1_vals or v2 in l2_vals:
                continue
            cells.append((row,col))
            backtrack(row+1, rows_used|{row}, cols_used|{col},
                     l1_vals|{v1}, l2_vals|{v2}, cells)
            cells.pop()
    backtrack(0, set(), set(), set(), set(), [])
    return cts
```

## 4. Pool Management

### Pool Diversity Invariants

The function `_save_triple_miss` maintains three invariants:

1. **Energy sorting:** Entries sorted by `clashes` ascending.
2. **Pair diversity:** For any L1 key (hash of L1 first row), at most 2 pool entries.
3. **L3 uniqueness:** No two entries have $\mathrm{np.array\_equal}(e_i.L3, e_j.L3)$.

```python
def _save_triple_miss(L1, L2, L3, stats):
    with _pool_lock:
        data = _load_pool()
        # New entry
        entry = {
            "ts": now(), "clashes": E,
            "cl12": cl12, "L1_key": L1[0].tolist(),
            "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist()
        }
        data.append(entry)
        data.sort(key=lambda e: e["clashes"])
        # Apply diversity filter
        seen_pairs = {}
        seen_l3 = []
        diverse = []
        for e in data:
            k = tuple(e["L1_key"])
            if seen_pairs.get(k, 0) >= 2:
                continue
            l3_arr = np.array(e["L3"])
            if any(np.array_equal(l3_arr, prev) for prev in seen_l3):
                continue
            diverse.append(e)
            seen_pairs[k] = seen_pairs.get(k, 0) + 1
            seen_l3.append(l3_arr)
            if len(diverse) == 8:
                break
        write_pool(diverse)
```

## 5. Worker Configuration

### Launch Commands (Session 7)

```bash
# 5 main SA workers
nohup .venv/bin/python mols10/mols_l3_search.py --seed 42  >> results/l3_v3_s42.log  &
nohup .venv/bin/python mols10/mols_l3_search.py --seed 137 >> results/l3_v3_s137.log &
nohup .venv/bin/python mols10/mols_l3_search.py --seed 271 >> results/l3_v3_s271.log &
nohup .venv/bin/python mols10/mols_l3_search.py --seed 503 >> results/l3_v3_s503.log &
nohup .venv/bin/python mols10/mols_l3_search.py --seed 999 >> results/l3_longrun2.log &

# Reverse search worker
nohup .venv/bin/python mols10/mols_reverse_search.py --seed 7777 \
      --budget 120 >> results/l3_reverse.log &

# SAT worker
nohup .venv/bin/python mols10/mols_sat_worker.py --seed 11111 \
      --timeout 20 >> results/l3_sat.log &
```

### Resource Usage

Each worker:
- CPU: 1 core (single-threaded Python)
- Memory: ~200MB (numpy arrays + pool JSON)
- Throughput: ~12,000 SA steps/second (combined over 7 replicas)

Total: ~84,000 SA steps/second across all 5 main workers.

## 6. SAT Solver Implementation

```python
def build_l3_sat(L1: np.ndarray, L2: np.ndarray, n: int = 10) -> Glucose3:
    solver = Glucose3()
    def var(i, j, k): return i*n*n + j*n + k + 1

    # C1: cell exactly-one
    for i in range(n):
        for j in range(n):
            add_exactly_one(solver, [var(i,j,k) for k in range(n)])

    # C2: row uniqueness
    for i in range(n):
        for k in range(n):
            add_exactly_one(solver, [var(i,j,k) for j in range(n)])

    # C3: column uniqueness
    for j in range(n):
        for k in range(n):
            add_exactly_one(solver, [var(i,j,k) for i in range(n)])

    # C4: orthogonality with L1
    for a in range(n):
        for b in range(n):
            group = [var(i,j,b) for i in range(n)
                              for j in range(n) if L1[i,j] == a]
            add_exactly_one(solver, group)

    # C5: orthogonality with L2
    for a in range(n):
        for b in range(n):
            group = [var(i,j,b) for i in range(n)
                              for j in range(n) if L2[i,j] == a]
            add_exactly_one(solver, group)

    return solver
```

With timeout via background thread calling `solver.interrupt()`.

## 7. Bugs Discovered and Fixed

| Bug | Session | Impact | Fix |
|-----|---------|--------|-----|
| Dead super-shake code (unreachable branch) | 6 | Super-shake never fired | Reordered probability boundaries |
| ILS step-count threshold too frequent | 5→6 | ILS fired every 0.5s instead of 8s | Replaced with time-based trigger |
| ILS local vs global stagnation | 6 | Counter reset on any accepted move | Track global best improvement only |
| L3 duplicates in pool | 6 | 3/5 E=37 entries were identical | Added np.array_equal L3 dedup |
| Wrong pair file in SAT worker | 7 | Loaded 5 pairs instead of 311 | Changed to `promising_pairs.json` |

## 8. Reproducibility

### Random Seeds

All workers use seeded Python `random.Random` objects. The outer seed (e.g., 42) seeds the worker's global RNG, which in turn seeds per-replica RNGs and per-trial seeds. Results are reproducible given the same seed and same pool state at trial start.

### Pool Dependence

Because workers share the pool file (read/written with file locks), the pool state at any trial start depends on the history of all prior workers. Exact reproducibility of the search path requires recording the pool state at each trial start (not currently implemented).

### Environment

- Python 3.11+
- numpy 1.24+
- python-sat 1.9.dev4 (Glucose3, RC2)
- OS: Linux 6.18.5
- Hardware: Single machine, 8+ cores available
