#!/usr/bin/env python3
"""
mols_cpsat_worker.py — CP-SAT optimization to find minimum E = cl13+cl23.

For fixed (L1, L2) from the near-miss pool (cl12=0), this worker uses OR-Tools
CP-SAT to minimize the number of missing symbol pairs in L1⊕L3 and L2⊕L3.
If the optimal is 0, we have 3-MOLS of order 10.

Key encoding insight: for fixed L1, the pair (a,b) is covered by L1⊕L3 iff
L3[i, c(i,a)] = b for some row i, where c(i,a) is the unique column in row i
with L1[i,c(i,a)] = a. This needs only n=10 indicator variables per (a,b) pair,
not n²=100.

If CP-SAT finds optimal=0 → N(10)≥3 proven (with verifiable optimality certificate).
If optimal≥1 (or UNSAT) → lower bound for this pair confirmed.

Usage:
    python mols_cpsat_worker.py --seed 5678 --timeout 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
TRIPLE_MISS_FILE = RESULTS_DIR / "near_miss_triple.json"
FOUND_FILE       = RESULTS_DIR / "MOLS10_FOUND.json"
LOG_FILE         = RESULTS_DIR / "l3_cpsat.log"
N = 10


def count_clashes(A: np.ndarray, B: np.ndarray, n: int = N) -> int:
    pairs = A.ravel().astype(np.int32) * n + B.ravel().astype(np.int32)
    return int(n * n - np.count_nonzero(np.bincount(pairs, minlength=n * n)))


def load_pool() -> list[dict]:
    try:
        return json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
    except (json.JSONDecodeError, OSError):
        return []


def save_found(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray, seed: int):
    result = {
        "found": True, "timestamp": datetime.now().isoformat(), "seed": seed,
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "cl12": count_clashes(L1, L2), "cl13": count_clashes(L1, L3),
        "cl23": count_clashes(L2, L3),
    }
    FOUND_FILE.write_text(json.dumps(result, indent=2))


POOL_LOCK = RESULTS_DIR / "triple_miss.lock"


def save_to_pool(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                 cl12: int, cl13: int, cl23: int, seed: int) -> bool:
    """Add a new near-miss triple to the pool. Returns True if pool was updated."""
    import fcntl
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12,
        "L1_key": str(L1.tolist()), "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": seed, "solver": "cpsat",
    }
    lock_fd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best_in_pool = min((p["clashes"] for p in pool), default=999)
        if E > best_in_pool:
            return False  # Don't add worse entries
        pool.append(entry)
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def build_col_index(L: np.ndarray, n: int = N) -> np.ndarray:
    """col_of[a][i] = column j such that L[i,j] = a."""
    col_of = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            col_of[int(L[i, j]), i] = j
    return col_of


def relabel_L3_canonical(L3: np.ndarray, n: int = N) -> np.ndarray:
    """Relabel L3 symbols so that L3[0] = [0,1,...,n-1].

    Symbol permutation is an isometry: cl13(L1, σ(L3)) = cl13(L1, L3).
    This allows symmetry breaking without losing any solutions.
    """
    perm = np.zeros(n, dtype=np.int32)
    for j in range(n):
        perm[int(L3[0, j])] = j  # perm[old_symbol] = new_symbol
    return perm[L3.astype(np.int32)].astype(np.int8)


def build_and_solve(L1: np.ndarray, L2: np.ndarray, n: int = N,
                    hint_L3: np.ndarray | None = None,
                    timeout_s: float = 120.0,
                    num_workers: int = 4,
                    symbreak: bool = True,
                    random_seed: int | None = None,
                    ) -> tuple[int, np.ndarray | None, bool]:
    """
    CP-SAT model for min cl13 + cl23 with fixed L1, L2 (cl12=0).

    Efficient encoding: n indicator variables per (a,b) pair, not n².
    Symmetry breaking: fix L3[0][j]=j (symbol relabeling; cuts n! redundancy).
    Returns (best_obj, best_L3, timed_out, is_optimal).
    """
    model = cp_model.CpModel()

    # Decision variables: L3[i][j] ∈ {0..n-1}
    L3v = [[model.new_int_var(0, n - 1, f"L3_{i}_{j}") for j in range(n)]
           for i in range(n)]

    # Latin square constraints
    for i in range(n):
        model.add_all_different(L3v[i])
    for j in range(n):
        model.add_all_different([L3v[i][j] for i in range(n)])

    # Symmetry breaking: fix first row to identity
    if symbreak:
        for j in range(n):
            model.add(L3v[0][j] == j)

    # Warm start from near-miss hint (relabeled to canonical form)
    if hint_L3 is not None:
        hint_canon = relabel_L3_canonical(hint_L3, n)
        for i in range(n):
            for j in range(n):
                model.add_hint(L3v[i][j], int(hint_canon[i, j]))

    # Precompute: for each symbol a, which column in each row contains a
    col_of_1 = build_col_index(L1, n)  # col_of_1[a][i] = j with L1[i,j]=a
    col_of_2 = build_col_index(L2, n)  # col_of_2[a][i] = j with L2[i,j]=a

    # Coverage variables and constraints (efficient: only n cells per pair)
    covered13 = []  # one bool per (a,b): is pair (a,b) in L1⊕L3?
    covered23 = []

    for a in range(n):
        for b in range(n):
            # L1⊕L3 pair (a,b): exists row i with L3[i, col_of_1[a,i]] = b
            cov = model.new_bool_var(f"c13_{a}_{b}")
            indicators = []
            for i in range(n):
                j = int(col_of_1[a, i])
                ind = model.new_bool_var(f"i13_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                indicators.append(ind)
            # cov = OR(indicators)
            model.add_bool_or(indicators).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in indicators]).only_enforce_if(
                cov.negated())
            covered13.append(cov)

    for a in range(n):
        for b in range(n):
            cov = model.new_bool_var(f"c23_{a}_{b}")
            indicators = []
            for i in range(n):
                j = int(col_of_2[a, i])
                ind = model.new_bool_var(f"i23_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                indicators.append(ind)
            model.add_bool_or(indicators).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in indicators]).only_enforce_if(
                cov.negated())
            covered23.append(cov)

    # Objective: minimize uncovered pairs (missing pairs = clashes)
    uncovered = [x.negated() for x in covered13] + [x.negated() for x in covered23]
    obj = model.new_int_var(0, 2 * n * n, "obj")
    model.add(obj == sum(uncovered))
    model.minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False
    if hasattr(solver.parameters, 'random_seed') and random_seed is not None:
        solver.parameters.random_seed = random_seed

    status = solver.solve(model)
    timed_out = (status == cp_model.UNKNOWN)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best_obj = int(solver.objective_value)
        L3_sol = np.array([[solver.value(L3v[i][j]) for j in range(n)]
                           for i in range(n)], dtype=np.int8)
        is_optimal = (status == cp_model.OPTIMAL)
        return best_obj, L3_sol, timed_out, is_optimal
    return -1, None, timed_out, False


def log(msg: str, log_fp=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n"); log_fp.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--timeout", type=float, default=120,
                        help="CP-SAT timeout per pair (seconds)")
    parser.add_argument("--workers", type=int, default=4,
                        help="CP-SAT parallel search workers")
    parser.add_argument("--save-threshold", type=int, default=37,
                        help="Save solutions with E <= this value to pool")
    parser.add_argument("--solver-seed", type=int, default=None,
                        help="Random seed for CP-SAT solver (for reproducibility)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, "a") as lfp:
        log("=" * 60, lfp)
        log(f"CP-SAT L3 optimizer — seed={args.seed}  timeout={args.timeout}s  "
            f"workers={args.workers}", lfp)
        log("=" * 60, lfp)

        trial = 0
        session_best_obj = 999
        optimal_count = 0

        while True:
            trial += 1

            pool = load_pool()
            if not pool:
                log(f"trial={trial:4d}  no pool entries, sleeping 30s", lfp)
                time.sleep(30)
                continue

            entry = rng.choice(pool)
            L1 = np.array(entry["L1"], dtype=np.int8).reshape(N, N)
            L2 = np.array(entry["L2"], dtype=np.int8).reshape(N, N)
            hint_L3 = np.array(entry["L3"], dtype=np.int8).reshape(N, N)

            log(f"trial={trial:4d}  pool_E={entry['clashes']}  "
                f"timeout={args.timeout}s  time={datetime.now().strftime('%H:%M:%S')}", lfp)

            t0 = time.time()
            result = build_and_solve(L1, L2, N, hint_L3=hint_L3,
                                     timeout_s=args.timeout, num_workers=args.workers,
                                     random_seed=args.solver_seed)
            elapsed = time.time() - t0
            best_obj, L3_sol, timed_out, is_optimal = result

            status_str = "OPTIMAL" if is_optimal else ("FEASIBLE" if not timed_out else "TIMEOUT")

            if L3_sol is not None:
                cl13 = count_clashes(L1, L3_sol)
                cl23 = count_clashes(L2, L3_sol)
                E = cl13 + cl23
                if E < session_best_obj:
                    session_best_obj = E
                if is_optimal:
                    optimal_count += 1

                log(f"  {status_str}: E={E}  cl13={cl13}  cl23={cl23}  "
                    f"session_best={session_best_obj}  optimal_proven={is_optimal}  "
                    f"elapsed={elapsed:.1f}s", lfp)

                if E == 0:
                    log(f"!!! N(10) >= 3 FOUND !!! seed={args.seed} trial={trial}", lfp)
                    save_found(L1, L2, L3_sol, args.seed)
                    print("MOLS10_FOUND written — N(10) >= 3 proven!", flush=True)
                    sys.exit(0)

                # Save improved solutions to pool
                cl12 = count_clashes(L1, L2)
                if E <= args.save_threshold:
                    saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23, args.seed)
                    if saved:
                        log(f"  SAVED to pool: E={E}  cl12={cl12}  cl13={cl13}  cl23={cl23}",
                            lfp)

                if is_optimal and E > 0:
                    log(f"  PROVEN: min E >= {E} for this pair "
                        f"(cl12=0, optimal certificate in {elapsed:.1f}s)", lfp)
            else:
                log(f"  {status_str}: no solution  elapsed={elapsed:.1f}s", lfp)

            if trial % 3 == 0:
                log(f"  [status] trials={trial}  optimal_proven={optimal_count}  "
                    f"session_best={session_best_obj}", lfp)


if __name__ == "__main__":
    main()
