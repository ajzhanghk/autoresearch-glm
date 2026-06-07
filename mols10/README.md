# MOLS of Order 10 — Adaptive Search

This sub-project applies an **autoresearch-style adaptive search** to the
combinatorial problem of finding Mutually Orthogonal Latin Squares (MOLS) of
order 10.  It demonstrates the same Edit → Run → Observe → Keep/Discard loop
used in the parent repository's GLM feature-discovery experiments, here applied
to mathematical search instead of tabular modelling.

---

## Background

A **Latin square** of order *n* is an *n × n* array filled with *n* symbols
(0 … n−1) such that every symbol appears exactly once in each row and each
column.  Two Latin squares *L₁* and *L₂* are **orthogonal** if, when
superimposed, all *n²* ordered pairs *(L₁[i,j], L₂[i,j])* are distinct.
*N(n)* denotes the maximum number of mutually orthogonal Latin squares of
order *n*.

| Order | N(n) | Status |
|-------|------|--------|
| 2 | 1 | proven |
| 3 | 2 | proven |
| 5, 7 | 4, 6 | prime → n−1 |
| 6 | 1 | proven (Tarry 1901) |
| **10** | **≥ 2** | **N(10) ≥ 2 proven (Parker 1960); N(10) ≥ 3 is open** |

Whether **N(10) ≥ 3** remains one of the oldest open problems in combinatorics.
The non-existence of a projective plane of order 10 was proved by Lam, Thiel &
Swiercz (1989) via exhaustive computer search (CPU-years), ruling out 9 MOLS
but not 3.

---

## Key Algorithmic Insight: Hall-Paige Trap

Early attempts used cyclic or group-based Latin square generators.  Every group
of order 10 — cyclic **Z₁₀** and dihedral **D₅** — fails the **Hall-Paige
condition** (their Sylow-2 subgroup is cyclic and not contained in the
commutator subgroup), so their Cayley tables have **no orthogonal mate**.  Any
square isotopic to such a Cayley table also has no mate.

**Fix:** Use **random backtracking** to generate Latin squares that are
genuinely outside any group isotopy class.  Random 10×10 Latin squares
generated this way have ~800 transversals each, making an OLS mate findable.

---

## Method: Transversal Exact-Cover

The key observation:

> A Latin square *L₁* has an orthogonal mate *L₂* **if and only if** its *n²*
> cells can be partitioned into *n* disjoint **transversals** (one entry per
> row, per column, and per symbol).

**Algorithm:**

1. **Generate** a random *n×n* Latin square via row-by-row backtracking.
2. **Enumerate** all transversals of *L₁* — typically ~800 for a random 10×10
   square, found in ~0.1 s.
3. **Exact cover (Algorithm X):** use MRV-guided backtracking to select *n* = 10
   pairwise disjoint transversals covering all *n²* cells.  Runs in 0.5–10 s.
4. **Build *L₂*:** assign color *k* ∈ {0 … 9} to every cell in transversal *k*.
   Orthogonality is guaranteed by construction.

This completely replaces the earlier Simulated Annealing approach, which got
stuck at 2 clashes because it was inadvertently working with squares in the
Hall-Paige-forbidden isotopy class.

### Complexity

| Step | Cost |
|------|------|
| Latin square generation | O(n²) per try, rarely fails |
| Transversal enumeration | ~0.1 s for n=10 (~800 results) |
| Algorithm X exact cover | 0.5–10 s for n=10 |
| Total per attempt | **2–10 s** |

---

## Verified Results

Five independent MOLS(10) pairs were found and saved in `results/`.

| Seed | Time (s) | Verified |
|------|----------|----------|
| 42 | 2.1 | ✓ |
| 123 | 3.5 | ✓ |
| 777 | 9.0 | ✓ |
| 2024 | 7.7 | ✓ |
| 31415 | 4.0 | ✓ |

### Example: seed=42

**L₁**
```
2 0 8 6 9 4 1 7 3 5
3 6 0 4 5 8 9 2 1 7
5 9 2 8 1 7 3 0 4 6
4 8 3 2 7 6 5 9 0 1
0 7 5 1 6 3 8 4 9 2
6 4 9 5 8 2 0 1 7 3
7 2 4 0 3 1 6 5 8 9
1 5 6 3 4 9 7 8 2 0
8 1 7 9 0 5 2 3 6 4
9 3 1 7 2 0 4 6 5 8
```

**L₂** (orthogonal mate — all 100 pairs (L₁[i,j], L₂[i,j]) are distinct)
```
2 0 8 9 7 1 5 4 3 6
4 7 6 2 3 5 8 0 9 1
9 4 1 7 8 3 6 2 0 5
7 3 9 8 5 6 0 1 4 2
1 9 5 0 2 7 4 8 6 3
0 5 2 4 1 9 3 6 7 8
8 6 3 5 0 4 1 7 2 9
3 8 4 1 6 0 2 9 5 7
6 1 0 3 9 2 7 5 8 4
5 2 7 6 4 8 9 3 1 0
```

---

## File Structure

```
mols10/
├── README.md                      # this file
├── mols_adaptive.py               # search engine (transversal + SA + SAT + CSP)
├── mols_search.py                 # Phase-2 CSP solver (find L3 given L1,L2)
├── verify_results.py              # standalone result verification
└── results/
    ├── mols_pair_seed42.json      # single verified pair (seed 42)
    └── mols_pairs_collection.json # 5 verified pairs (seeds 42,123,777,2024,31415)
```

---

## Usage

### Quick start — find a MOLS(10) pair

```bash
cd mols10/
python mols_adaptive.py --n 10 --strategy tv --save-dir /tmp/mols_run
```

`--strategy tv` uses the transversal exact-cover method (default, fastest).
A pair is typically found in 2–10 seconds.

### Verify saved results

```bash
python verify_results.py                          # verify all results/
python verify_results.py results/mols_pair_seed42.json
```

### Adaptive portfolio search (all strategies)

```bash
python mols_adaptive.py --n 10 --strategy adaptive --eval-budget 120 --save-dir /tmp/mols_adaptive
```

Runs a portfolio of transversal (tv), simulated annealing (sa), SAT (sat), and
CSP (csp) strategies, mutating the best-performing configuration each trial.

### Phase 2 — search for a third MOLS

```bash
python mols_adaptive.py --n 10 --skip-pair --save-dir /tmp/mols_run
```

Loads the saved `mols_pair.json` and searches for *L₃* orthogonal to both *L₁*
and *L₂* using backtracking CSP from `mols_search.py`.  No solution for n=10
has been found computationally; this is consistent with the conjecture that
N(10) = 2, but remains unproven.

---

## Strategy Comparison

| Strategy | Idea | Phase-1 Result |
|----------|------|----------------|
| `tv` | Transversal exact-cover (Algorithm X) | **2–10 s ✓** |
| `sa` | Simulated annealing on (L₁, L₂) clash count | Stuck at 2 clashes (Hall-Paige trap) |
| `sat` | Glucose4 CDCL SAT solver (522K clauses) | Timeout >8 min |
| `csp` | Coupled backtracking CSP (bitmask FC+MRV) | Timeout |
| `adaptive` | Portfolio of all above with parameter mutation | Finds via `tv` |

---

## Research Context

This work is part of the **autoresearch-glm** project, which applies an
autonomous Edit → Run → Observe → Keep/Discard loop to scientific discovery.
The MOLS search illustrates that the same adaptive framework can be applied to
pure mathematical search problems:

- **Edit:** Mutate search strategy parameters or switch algorithms.
- **Run:** Execute the search for a fixed time budget.
- **Observe:** Record best clash count (or "found") as the metric.
- **Keep/Discard:** Retain the configuration if it improves the frontier.

The Hall-Paige insight — that all groups of order 10 have no complete mapping —
was itself discovered adaptively: observing that all searches with
cyclic/group-based generators failed, investigating the mathematical reason,
and switching to random backtracking generation.

---

## Dependencies

```bash
pip install numpy pysat   # pysat only needed for --strategy sat
```

Python 3.11+.

---

## References

- Parker, E.T. (1960). *Orthogonal Latin squares.* Proc. AMS 10, 946–951.
- Hall, M. & Paige, L.J. (1955). *Complete mappings of finite groups.* Pacific J. Math.
- Lam, C.W.H., Thiel, L., & Swiercz, S. (1989). *The nonexistence of finite projective planes of order 10.* Canad. J. Math.
- Knuth, D.E. (2000). *Dancing links.* Millennial Perspectives in Computer Science.
