#!/usr/bin/env python3
"""
mols_crossover_search.py — New algorithmic approaches to escape the E=37 barrier.

Three complementary strategies not present in mols_l3_search.py:

1. ROW-REPLACE SA: Each SA move replaces an entire row of L3 via random SDR
   sampling (10 cells per move vs 2-4 for intercalate/row-swap). This can
   bridge the strict 2-step local minimum that blocks cell-level SA.

2. CROSSOVER: Combine two pool E=37 L3 matrices at a random split row,
   complete the remaining rows via SDR (maintaining LS property), then
   refine with SA. Explores convex combinations of E=37 basins.

3. PARTIAL COMPLETION: Fix the best k rows of an E=37 triple (those causing
   fewest clashes with L1/L2) and systematically backtrack over the worst
   rows to find a lower-energy completion.

Usage:
    python mols_crossover_search.py --seed 4242 --budget 90
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
TRIPLE_MISS_FILE = RESULTS_DIR / "near_miss_triple.json"
PROMISING_FILE   = RESULTS_DIR / "promising_pairs.json"
LOG_FILE         = RESULTS_DIR / "l3_crossover.log"
N = 10

# ---------------------------------------------------------------------------
# Core utilities (self-contained, no dependency on mols_adaptive.py)
# ---------------------------------------------------------------------------

def count_clashes(A: np.ndarray, B: np.ndarray, n: int = N) -> int:
    """Number of (a,b) symbol pairs that never appear in superposition A,B."""
    pairs = A.ravel().astype(np.int32) * n + B.ravel().astype(np.int32)
    counts = np.bincount(pairs, minlength=n * n)
    return int(n * n - np.count_nonzero(counts))


def is_valid_ls(L: np.ndarray, n: int = N) -> bool:
    expected = set(range(n))
    for i in range(n):
        if set(L[i, :].tolist()) != expected:
            return False
        if set(L[:, i].tolist()) != expected:
            return False
    return True


def make_pair_counts(A: np.ndarray, B: np.ndarray, n: int = N) -> np.ndarray:
    """Return flat pair-count array: counts[a*n+b] = #cells where A=a, B=b."""
    pairs = A.ravel().astype(np.int32) * n + B.ravel().astype(np.int32)
    return np.bincount(pairs, minlength=n * n).astype(np.int32)


def clashes_from_counts(counts: np.ndarray, n: int = N) -> int:
    return int(n * n - np.count_nonzero(counts))


def build_pair_counts_row(L1_row: np.ndarray, L3_row: np.ndarray, n: int = N) -> np.ndarray:
    """Pair count contribution from a single row."""
    pc = np.zeros(n * n, dtype=np.int32)
    for j in range(n):
        pc[int(L1_row[j]) * n + int(L3_row[j])] += 1
    return pc


def random_valid_row(L3: np.ndarray, row_idx: int, n: int, rng: random.Random):
    """Sample a random valid replacement for row row_idx of L3.

    Returns a new row (np.ndarray of shape (n,)) that completes the LS, or
    None if no valid assignment found (rare: ~1% failure rate).
    """
    # For each column j: symbols NOT already used in that column (excluding current row)
    available: list[list[int]] = []
    for j in range(n):
        used = {int(L3[i, j]) for i in range(n) if i != row_idx}
        available.append([s for s in range(n) if s not in used])

    # Random SDR via greedy on shuffled column order
    cols = list(range(n))
    rng.shuffle(cols)
    new_row = np.zeros(n, dtype=np.int8)
    used_syms: set[int] = set()

    for j in cols:
        choices = [s for s in available[j] if s not in used_syms]
        if not choices:
            return None
        s = rng.choice(choices)
        new_row[j] = s
        used_syms.add(s)

    return new_row


def update_pair_counts(pc: np.ndarray,
                       L_ref_row: np.ndarray,
                       old_l3_row: np.ndarray,
                       new_l3_row: np.ndarray,
                       n: int = N) -> tuple[np.ndarray, int]:
    """Update pair counts and return delta clashes when L3 row changes.

    Returns (updated_pc, delta_cl) where delta_cl = new_clashes - old_clashes.
    Operates in-place on pc.
    """
    delta = 0
    for j in range(n):
        a = int(L_ref_row[j])
        b_old = int(old_l3_row[j])
        b_new = int(new_l3_row[j])
        if b_old == b_new:
            continue
        # Remove old pair
        idx_old = a * n + b_old
        if pc[idx_old] == 1:
            delta += 1   # was exactly-1 → becomes 0 (newly missing)
        pc[idx_old] -= 1
        # Add new pair
        idx_new = a * n + b_new
        if pc[idx_new] == 0:
            delta -= 1   # was 0 → becomes 1 (no longer missing)
        pc[idx_new] += 1
    return pc, delta


# ---------------------------------------------------------------------------
# Pool I/O
# ---------------------------------------------------------------------------
_pool_lock = threading.Lock()


def load_pool() -> list[dict]:
    try:
        return json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
    except (json.JSONDecodeError, OSError):
        return []


def save_to_pool(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                 clashes: int, source: str = "crossover") -> bool:
    """Save a triple to the shared near-miss pool if it improves diversity/energy."""
    # Validate
    n = N
    for sq in (L1, L2, L3):
        if not is_valid_ls(sq, n):
            return False
    if count_clashes(L1, L2, n) != 0:
        return False

    with _pool_lock:
        try:
            data = load_pool()
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "clashes": clashes,
                "cl12": 0,
                "source": source,
                "L1_key": L1[0].tolist(),
                "L1": L1.tolist(),
                "L2": L2.tolist(),
                "L3": L3.tolist(),
            }
            data.append(entry)
            data.sort(key=lambda e: e["clashes"])

            # Diversity filter: ≤2 entries per L1 fingerprint, L3 unique
            seen_pairs: dict = {}
            seen_l3: list = []
            diverse: list = []
            for e in data:
                raw_k = e.get("L1_key")
                k = tuple(raw_k) if isinstance(raw_k, list) else raw_k
                if seen_pairs.get(k, 0) >= 2:
                    continue
                l3_arr = np.array(e["L3"], dtype=np.int8)
                if any(np.array_equal(l3_arr, prev) for prev in seen_l3):
                    continue
                diverse.append(e)
                seen_pairs[k] = seen_pairs.get(k, 0) + 1
                seen_l3.append(l3_arr)
                if len(diverse) == 8:
                    break
            TRIPLE_MISS_FILE.write_text(json.dumps(diverse, indent=2))
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Strategy 1: Row-replacement SA
# ---------------------------------------------------------------------------

def row_replace_sa(L1: np.ndarray, L2: np.ndarray, L3_init: np.ndarray,
                   n: int, budget_s: float, rng: random.Random,
                   T_init: float = 8.0, T_min: float = 0.05) -> tuple[np.ndarray, int]:
    """SA where each move replaces an entire row of L3 (10-cell neighbourhood).

    Uses efficient delta-clash computation via pair-count tracking.
    """
    L3 = L3_init.copy()

    # Maintain separate pair-count arrays for L1-L3 and L2-L3
    pc13 = make_pair_counts(L1, L3, n)
    pc23 = make_pair_counts(L2, L3, n)
    cl13 = clashes_from_counts(pc13, n)
    cl23 = clashes_from_counts(pc23, n)
    E = cl13 + cl23

    best_E = E
    best_L3 = L3.copy()

    t0 = time.time()
    T = T_init
    steps = 0
    elapsed = 0.0
    # Cooling: reach T_min at ~80% of budget
    cool_steps = max(1, int(budget_s * 2000))  # estimate steps
    alpha = (T_min / T_init) ** (1.0 / cool_steps)

    while True:
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            break

        # Propose: replace a random row
        row_idx = rng.randint(0, n - 1)
        new_row = random_valid_row(L3, row_idx, n, rng)
        if new_row is None:
            continue

        old_row = L3[row_idx].copy()

        # Compute delta for L1-L3 pair counts
        pc13_trial = pc13.copy()
        _, d13 = update_pair_counts(pc13_trial, L1[row_idx], old_row, new_row, n)
        pc23_trial = pc23.copy()
        _, d23 = update_pair_counts(pc23_trial, L2[row_idx], old_row, new_row, n)
        dE = d13 + d23

        if dE < 0 or (T > 1e-9 and rng.random() < math.exp(-dE / T)):
            L3[row_idx] = new_row
            pc13 = pc13_trial
            pc23 = pc23_trial
            E += dE
            cl13 += d13
            cl23 += d23
            if E < best_E:
                best_E = E
                best_L3 = L3.copy()
                if best_E == 0:
                    return best_L3, 0

        T = max(T_min, T * alpha)
        steps += 1

    return best_L3, best_E


# ---------------------------------------------------------------------------
# Strategy 2: Crossover between two E=37 pool entries
# ---------------------------------------------------------------------------

def complete_ls_from_prefix(prefix_rows: np.ndarray, n: int,
                             rng: random.Random, max_attempts: int = 50
                             ) -> np.ndarray | None:
    """Given first k rows of a LS, sample a random valid completion for rows k..n-1."""
    k = prefix_rows.shape[0]
    if k >= n:
        return prefix_rows

    # Column usage: for each column j, set of symbols already placed
    col_used: list[set] = [set() for _ in range(n)]
    for i in range(k):
        for j in range(n):
            col_used[j].add(int(prefix_rows[i, j]))

    for _ in range(max_attempts):
        L = prefix_rows.copy()
        col_used_trial = [s.copy() for s in col_used]
        success = True

        for i in range(k, n):
            # Available symbols per column for row i
            avail = [list(set(range(n)) - col_used_trial[j]) for j in range(n)]
            # Random SDR
            cols = list(range(n))
            rng.shuffle(cols)
            new_row = np.zeros(n, dtype=np.int8)
            used_syms: set = set()
            row_ok = True
            for j in cols:
                choices = [s for s in avail[j] if s not in used_syms]
                if not choices:
                    row_ok = False
                    break
                s = rng.choice(choices)
                new_row[j] = s
                used_syms.add(s)
            if not row_ok:
                success = False
                break
            # Accept row
            extra = np.zeros((1, n), dtype=np.int8)
            extra[0] = new_row
            L = np.vstack([L, extra])
            for j in range(n):
                col_used_trial[j].add(int(new_row[j]))

        if success and L.shape[0] == n:
            return L

    return None


def crossover_search(pool_entries: list[dict], L1_fixed: np.ndarray,
                     L2_fixed: np.ndarray, n: int, budget_s: float,
                     rng: random.Random) -> tuple[np.ndarray | None, int]:
    """Combine two pool L3 matrices at random row splits, refine with SA."""
    if len(pool_entries) < 2:
        return None, 999

    best_E = 999
    best_L3 = None
    t0 = time.time()
    trial = 0

    while time.time() - t0 < budget_s:
        # Pick two distinct pool entries (prefer low energy)
        e_a, e_b = rng.sample(pool_entries[:min(6, len(pool_entries))], 2)
        L3_a = np.array(e_a["L3"], dtype=np.int8)
        L3_b = np.array(e_b["L3"], dtype=np.int8)

        # Random split point
        split = rng.randint(1, n - 1)

        # Take rows 0..split-1 from L3_a; complete from L3_b's rows, then SDR
        prefix = L3_a[:split].copy()
        L3_cross = complete_ls_from_prefix(prefix, n, rng)

        if L3_cross is None or not is_valid_ls(L3_cross, n):
            trial += 1
            continue

        # Quick energy check
        E_cross = count_clashes(L1_fixed, L3_cross, n) + count_clashes(L2_fixed, L3_cross, n)

        # Brief SA refinement (5s) using row-replace SA
        sa_budget = min(5.0, (budget_s - (time.time() - t0)) * 0.4)
        if sa_budget > 0.5:
            L3_ref, E_ref = row_replace_sa(
                L1_fixed, L2_fixed, L3_cross, n, sa_budget, rng, T_init=3.0
            )
        else:
            L3_ref, E_ref = L3_cross, E_cross

        if E_ref < best_E:
            best_E = E_ref
            best_L3 = L3_ref.copy()
            if best_E == 0:
                return best_L3, 0

        trial += 1

    return best_L3, best_E


# ---------------------------------------------------------------------------
# Strategy 3: Partial completion — fix good rows, backtrack over bad rows
# ---------------------------------------------------------------------------

def row_clash_contribution(L1: np.ndarray, L2: np.ndarray,
                           L3: np.ndarray, row_idx: int, n: int) -> int:
    """Count how many clash pairs involve row row_idx of L3."""
    # Full pair counts
    pc13 = make_pair_counts(L1, L3, n)
    pc23 = make_pair_counts(L2, L3, n)

    # Remove row contribution, count change in clashes
    pc13_without = pc13.copy()
    pc23_without = pc23.copy()
    for j in range(n):
        idx13 = int(L1[row_idx, j]) * n + int(L3[row_idx, j])
        idx23 = int(L2[row_idx, j]) * n + int(L3[row_idx, j])
        pc13_without[idx13] -= 1
        pc23_without[idx23] -= 1

    cl_full = clashes_from_counts(pc13, n) + clashes_from_counts(pc23, n)
    cl_without = clashes_from_counts(pc13_without, n) + clashes_from_counts(pc23_without, n)
    return cl_full - cl_without


def partial_completion_search(L1: np.ndarray, L2: np.ndarray,
                               L3_seed: np.ndarray, n: int,
                               budget_s: float, rng: random.Random,
                               fix_fraction: float = 0.5
                               ) -> tuple[np.ndarray, int]:
    """Fix the best rows of L3_seed, replace worst rows with SA-refined SDR completion."""
    # Rank rows by their clash contribution (ascending = rows that help most)
    row_scores = [
        (row_clash_contribution(L1, L2, L3_seed, i, n), i)
        for i in range(n)
    ]
    row_scores.sort()

    n_fix = max(1, int(n * fix_fraction))
    fixed_rows = sorted([idx for _, idx in row_scores[:n_fix]])
    free_rows  = sorted([idx for _, idx in row_scores[n_fix:]])

    # Build prefix: extract fixed rows in their original positions
    # We fix the "good" rows and let SDR completion fill the rest
    prefix_rows = L3_seed[fixed_rows].copy()

    best_E = count_clashes(L1, L3_seed, n) + count_clashes(L2, L3_seed, n)
    best_L3 = L3_seed.copy()

    t0 = time.time()
    attempts = 0

    while time.time() - t0 < budget_s:
        # Complete the LS with fixed rows + random SDR for free rows
        # Build a fresh n×n matrix, placing fixed rows, then filling freely
        L3_try = np.zeros((n, n), dtype=np.int8)
        L3_try[fixed_rows] = prefix_rows

        # Fill free rows one by one via SDR
        col_used = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                col_used[j].add(int(L3_try[i, j]))
        # Remove zeros from col_used (they are unfilled cells)
        # Rebuild properly
        col_used = [set() for _ in range(n)]
        for i in fixed_rows:
            for j in range(n):
                col_used[j].add(int(L3_try[i, j]))

        ok = True
        order = free_rows[:]
        rng.shuffle(order)
        for i in order:
            avail = [list(set(range(n)) - col_used[j]) for j in range(n)]
            cols = list(range(n)); rng.shuffle(cols)
            new_row = np.zeros(n, dtype=np.int8)
            used_syms: set = set()
            row_ok = True
            for j in cols:
                choices = [s for s in avail[j] if s not in used_syms]
                if not choices:
                    row_ok = False; break
                s = rng.choice(choices)
                new_row[j] = s
                used_syms.add(s)
            if not row_ok:
                ok = False; break
            L3_try[i] = new_row
            for j in range(n):
                col_used[j].add(int(new_row[j]))

        if not ok:
            attempts += 1
            continue

        # Quick SA refinement on L3_try
        sa_budget = min(3.0, (budget_s - (time.time() - t0)) * 0.3)
        if sa_budget > 0.3:
            L3_ref, E_ref = row_replace_sa(
                L1, L2, L3_try, n, sa_budget, rng, T_init=2.0
            )
        else:
            E_ref = count_clashes(L1, L3_try, n) + count_clashes(L2, L3_try, n)
            L3_ref = L3_try

        if E_ref < best_E:
            best_E = E_ref
            best_L3 = L3_ref.copy()
            if best_E == 0:
                return best_L3, 0

        attempts += 1

    return best_L3, best_E


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run_forever(seed: int, trial_budget: float = 90.0) -> None:
    rng = random.Random(seed)
    trial = 0
    session_best = 999

    _log(f"crossover_search started: seed={seed} budget={trial_budget}s")

    while True:
        trial += 1
        pool = load_pool()
        if not pool:
            _log("Pool empty, sleeping 10s...")
            time.sleep(10)
            continue

        # Only use E=37 (or best available) entries with cl12=0
        best_e = pool[0]["clashes"]
        best_entries = [e for e in pool if e["clashes"] == best_e and e.get("cl12", 1) == 0]
        if not best_entries:
            best_entries = [e for e in pool if e.get("cl12", 1) == 0]
        if not best_entries:
            best_entries = pool

        # Pick a reference (L1, L2) from the best entry
        ref = rng.choice(best_entries[:min(5, len(best_entries))])
        L1 = np.array(ref["L1"], dtype=np.int8)
        L2 = np.array(ref["L2"], dtype=np.int8)
        cl12_ref = count_clashes(L1, L2, N)
        if cl12_ref != 0:
            _log(f"Warning: ref pair not orthogonal (cl12={cl12_ref}), skipping")
            continue

        # Split budget among strategies
        strategy_roll = rng.random()
        t0 = time.time()

        if strategy_roll < 0.40:
            # Strategy 1: Row-replace SA on best pool L3
            L3_seed = np.array(ref["L3"], dtype=np.int8)
            strategy_name = "row_replace_sa"
            L3_out, E_out = row_replace_sa(
                L1, L2, L3_seed, N, trial_budget, rng,
                T_init=6.0 + rng.uniform(-2, 2),
                T_min=0.02
            )

        elif strategy_roll < 0.70:
            # Strategy 2: Crossover between pool entries
            strategy_name = "crossover"
            L3_out, E_out = crossover_search(
                best_entries, L1, L2, N, trial_budget, rng
            )
            if L3_out is None:
                L3_out = np.array(ref["L3"], dtype=np.int8)
                E_out = count_clashes(L1, L3_out, N) + count_clashes(L2, L3_out, N)

        else:
            # Strategy 3: Partial completion (vary fix fraction)
            strategy_name = "partial_completion"
            fix_frac = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            L3_seed = np.array(ref["L3"], dtype=np.int8)
            L3_out, E_out = partial_completion_search(
                L1, L2, L3_seed, N, trial_budget, rng, fix_fraction=fix_frac
            )

        elapsed = time.time() - t0

        if E_out < session_best:
            session_best = E_out

        _log(
            f"trial={trial:4d}  strategy={strategy_name:20s}  "
            f"E={E_out}  session_best={session_best}  elapsed={elapsed:.1f}s"
        )

        if E_out == 0:
            _log("=" * 60)
            _log("*** L3 FOUND! MOLS-10 TRIPLE EXISTS! ***")
            _log("=" * 60)
            # Save immediately
            saved = save_to_pool(L1, L2, L3_out, 0, source="crossover_FOUND")
            _log(f"Pool save: {saved}")
            # Write dedicated file
            found_file = RESULTS_DIR / "MOLS10_FOUND.json"
            found_file.write_text(json.dumps({
                "ts": datetime.now().isoformat(),
                "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3_out.tolist(),
                "cl12": 0, "cl13": 0, "cl23": 0, "E": 0, "strategy": strategy_name
            }, indent=2))
            _log(f"Written to {found_file}")
            sys.exit(0)  # alert other processes via exit code

        if E_out < best_e:
            saved = save_to_pool(L1, L2, L3_out, E_out, source=strategy_name)
            if saved:
                _log(f"  *** New best E={E_out} saved to pool!")

        elif E_out == best_e:
            # Still worth saving for diversity
            save_to_pool(L1, L2, L3_out, E_out, source=strategy_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossover/row-replace L3 search")
    parser.add_argument("--seed",   type=int,   default=4242)
    parser.add_argument("--budget", type=float, default=90.0,
                        help="Seconds per trial")
    args = parser.parse_args()

    run_forever(seed=args.seed, trial_budget=args.budget)
