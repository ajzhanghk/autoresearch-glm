# Search Algorithms

## Overview

The search employs five primary algorithms operating concurrently across 7 parallel workers:

| Algorithm | Workers | Purpose |
|-----------|---------|---------|
| `sa_triple_pt` | 5 | Joint SA with parallel tempering on $(L_1,L_2,L_3)$ |
| `multi_decomp` | (shared) | High-throughput pair generation via transversal enumeration |
| `sa_ct_climb` | (shared) | SA maximizing $\mathrm{CT}(L_1,L_2)$ via intercalate flips |
| `reverse_search` | 1 | Fix near-miss $L_3$; search for compatible $(L_1,L_2)$ |
| `sat_worker` | 1 | SAT (Glucose3) exact solver for each MOLS pair |

---

## Algorithm 1: Parallel-Tempering Simulated Annealing — `sa_triple_pt`

### Overview

The primary search strategy. Jointly optimizes all three squares $(L_1, L_2, L_3)$ to minimize $E = \mathrm{cl}_{12} + \mathrm{cl}_{13} + \mathrm{cl}_{23}$.

### Parallel Tempering (Replica Exchange)

$K = 7$ replicas run at temperatures $T_i$ in geometric progression:

$$T = (1.0,\ 4.0,\ 16.0,\ 64.0,\ 256.0,\ 1024.0,\ 4096.0)$$

**Acceptance rule** for each replica $i$: a move proposing energy change $\Delta E$ is accepted with probability:
$$P(\text{accept}) = \min\left(1,\ e^{-\Delta E / T_i}\right)$$

**Replica exchange:** Every 2000 outer steps, adjacent replicas $(i, i+1)$ are swapped with probability:
$$P(\text{swap}) = \min\left(1,\ e^{(E_i - E_{i+1})(1/T_i - 1/T_{i+1})}\right)$$

This allows low-energy states discovered at high temperature to flow toward the cold replica, and allows the cold replica to escape local minima via the hot replicas.

### Move Types

All moves are applied to one of $\{L_1, L_2, L_3\}$, selected by:
- If $\mathrm{cl}_{12} < 5$: target $L_3$ with probability 0.70, $L_1$ with 0.15, $L_2$ with 0.15.
- Otherwise: target uniformly at random.

| Move | Description | Neighborhood size |
|------|-------------|-------------------|
| `row` | Swap two rows of the target square | $\binom{10}{2} = 45$ |
| `col` | Swap two columns | 45 |
| `relabel` | Swap two symbol values everywhere | 45 |
| `intercalate` | Flip a $2\times 2$ intercalate submatrix | Up to $100 \cdot 25 = 2500$ |
| `kscramble` | Permute $k \in [3,5]$ randomly chosen rows | $\binom{10}{k} \cdot k!$ |
| `bigscramble` | Permute $k \in [6,9]$ randomly chosen rows | $\binom{10}{k} \cdot k!$ |

All moves preserve the Latin square property of the modified square.

### Iterated Local Search (ILS)

**Trigger:** Time-based — fires when both conditions hold:
1. No global energy improvement in $\geq 8$ seconds.
2. No ILS kick in $\geq 8$ seconds.

**Cold-replica perturbation (30% pool-jump, 70% random shake):**

- *Pool-jump (30%):* Replace $L_3$ of the cold replica with a near-miss pool entry (with 0–15 random move micro-shake), while keeping the current $(L_1, L_2)$. This injects diverse $L_3$ basins into the search.
- *Random shake (70%):* Apply 25–70 random LS-preserving moves to $L_3$ of the cold replica.

**Frequency:** Approximately 14 ILS kicks per 97-second trial.

### Initialization Strategies

Fresh initial states for each trial are drawn from a probability mixture:

| Strategy | Probability | Description |
|----------|-------------|-------------|
| Pure near-miss | 8% | Exact near-miss triple from pool (top-5 by energy) |
| Micro-shake | 10% | Near-miss $L_3$ + 1–10 random moves |
| Shake | 15% | Near-miss $L_3$ + 15–80 moves; possibly fresh $(L_1,L_2)$ |
| L3-transfer | 17% | Near-miss $L_3$ + fresh isotopy variant of seed pair |
| CT=2 pair + fresh $L_3$ | 23% | Known MOLS pair (isotopy) + random $L_3$ |
| Cyclic $L_3$ | 7% | $L_3[i,j] = (i + \omega \cdot j) \bmod 10$ with $\gcd(\omega, 10)=1$, full isotopy |
| Affine $L_3$ | 8% | $L_3[i,j] = (ai + bj) \bmod 10$, $\gcd(a,10)=\gcd(b,10)=1$, full isotopy |
| D5 Cayley table | 5% | Cayley table of dihedral group $D_5$ with full isotopy |
| Super-shake | 3% | Near-miss $L_3$ + 100–400 moves |
| Reverse search | 2% | Fix near-miss $L_3$; randomize $(L_1, L_2)$ |
| Fully random | 2% | Three independent random Latin squares |

### Hot-Replica Reinitialisation

Every 200,000 outer steps, the hottest replica ($T = 4096$) is fully reinitialised from a fresh state to prevent it from becoming trapped.

### Trial Budget

- Standard workers (seeds 42, 137, 271, 503): 97 seconds per trial.
- Long-run worker (seed 999): 540 seconds per trial (less frequent reinit, deeper descent).

---

## Algorithm 2: Multi-Decomposition — `multi_decomp`

### Purpose

High-throughput generation of diverse MOLS pairs $(L_1, L_2)$ scored by $\mathrm{CT}(L_1, L_2)$.

### Method

1. **Transversal enumeration:** Given $L_1$, enumerate transversals of $L_1$ using a backtracking algorithm capped at 10,000 transversals per square.
2. **Algorithm X (Dancing Links):** Use Knuth's Algorithm X to find a partition of the 100 cells into 10 disjoint transversals of $L_1$. Each valid partition yields an orthogonal mate $L_2$.
3. **CT scoring:** Compute $\mathrm{CT}(L_1, L_2)$ for each generated pair. Pairs with $\mathrm{CT} \geq 1$ are saved to `promising_pairs.json`.

### L1 Diversity

$L_1$ is generated as:
- **50%:** Isotopy variant of the Parker TV-21 square (known MOLS-10 base).
- **50%:** Fully random Latin square (multi-decomp then searched for orthogonal mate).

This split was increased from 33% random in Session 6 to explore non-Parker isotopy classes.

---

## Algorithm 3: CT-Climbing SA — `sa_ct_climb`

### Purpose

Directly maximise $\mathrm{CT}(L_1, L_2)$ for a fixed MOLS pair via intercalate flips.

### Method

Starting from a MOLS pair $(L_1, L_2)$:
1. Propose an intercalate flip on $L_2$ (with 85% probability) or $L_1$ (15%).
2. Verify the flip maintains $\mathrm{cl}_{12} = 0$.
3. Compute new $\mathrm{CT}$; accept with SA probability at temperature $T$.

**Move validity:** Only flips that maintain $\mathrm{cl}_{12} = 0$ are accepted (intercalate-flip on $L_2$ while $L_1$ is fixed preserves orthogonality only for specific cell configurations).

**Cooling:** $T_0 = 1.5$, cooling factor $0.99998$ per step.

**Restart policy:** After 80,000 steps without improvement, return to best state. Every 4th restart, generate a fully fresh pair.

---

## Algorithm 4: Reverse Search

### Motivation

If the near-miss $L_3$ (with $E=37$) happens to be a valid third MOLS square for some MOLS pair $(L_1, L_2)$ not yet found, fixing $L_3$ and searching for compatible $(L_1, L_2)$ might be easier.

### Method

Fix $L_3 = L_3^*$ (from the near-miss pool). Run 7-replica PT over $(L_1, L_2)$ only:

**Energy:** $E_{\text{rev}} = \mathrm{cl}_{12} + \mathrm{cl}_{13} + \mathrm{cl}_{23}$ (where $L_3$ is fixed).

**Move bias:** When $\mathrm{cl}_{12} < 5$ (pair nearly orthogonal), prefer moving the square with higher clash count against $L_3^*$.

**ILS:** After 10 seconds of stagnation, apply 20–60 row-permutation moves to $L_1$.

**Budget:** 120 seconds per trial, continuous (worker seed 7777).

### Key Result

Best energy found: $E_{\text{rev}} = 60$ (Session 7) vs. required $E_{\text{rev}} = 0$.

The consistent gap ($E_{\text{rev}} \geq 60$ across 18+ trials with different $L_3^*$ seeds) strongly indicates the near-miss $L_3$ matrices are **not** compatible with any existing MOLS-10 pair structure.

---

## Algorithm 5: SAT Worker (Glucose3)

### Encoding

The 3-MOLS existence question for fixed $(L_1, L_2)$ is encoded as a CNF satisfiability problem (see `02_mathematical_theory.md` Section 3 for full details).

### Implementation

- **Solver:** Glucose3 (CDCL algorithm), from the `python-sat` library.
- **Variables:** 1,000 Boolean variables.
- **Clauses:** ~23,000 clauses.
- **Timeout:** 20 seconds per instance (never triggered in practice — all instances resolve in < 50ms).

### Operational Protocol

**Phase 1:** Test all existing promising pairs from `promising_pairs.json`.

**Phase 2:** Continuously generate new MOLS pairs via:
- Isotopy variants of existing pairs (70% of pair generation attempts)
- Fresh random Latin squares with SA to find orthogonal mate (30%)

For each pair, run Glucose3. Results: SAT (found $L_3$!) or UNSAT (certified non-existence).

### Performance

- Phase 1: 311 pairs tested, all UNSAT, average 42ms per instance.
- Phase 2: ~600 pairs/hour tested (pair generation dominates at 0.02s/pair + 0.04s SAT).
- Total pairs tested as of Session 7: 311 (Phase 1) + 600+ (Phase 2, ongoing).

---

## Algorithm Interaction and Data Flow

```
near_miss_triple.json ←→ sa_triple_pt (warm starts)
                     ←→ reverse_search (fixed L3)
                     ←→ sat_worker (Phase 1 seed)

promising_pairs.json ←→ multi_decomp (adds pairs)
                    ←→ sa_ct_climb (warm starts)
                    ←→ sat_worker (Phase 1 input)
                    ←→ sa_triple_pt (isotopy variants)
```

All workers share both `near_miss_triple.json` (best triples) and `promising_pairs.json` (best pairs) via file I/O with atomic JSON writes. A pool management function enforces:
- Maximum 8 entries in the triple pool.
- Pair diversity: at most 2 entries per unique $L_1$.
- $L_3$ uniqueness: no two pool entries share the same $L_3$ matrix.
