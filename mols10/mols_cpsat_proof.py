#!/usr/bin/env python3
"""
mols_cpsat_proof.py — CP-SAT feasibility check: can E ≤ target for a fixed (L1,L2)?

For each pool entry (L1, L2) with cl12=0, this worker checks whether there exists
an L3 with cl13 + cl23 ≤ target. This is a SAT decision problem (not optimization),
which is typically faster to resolve than finding the true minimum.

Outcomes per pair:
  SAT   → found L3 with E ≤ target  (saved to pool; break E=37 barrier!)
  UNSAT → proven E > target for this pair  (lower bound established)
  TIMEOUT → inconclusive for this pair

Usage:
    python mols_cpsat_proof.py --seed 9999 --timeout 300 --target 35
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
LOG_FILE         = RESULTS_DIR / "l3_cpsat_proof.log"
POOL_LOCK        = RESULTS_DIR / "triple_miss.lock"
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


def save_to_pool(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                 cl12: int, cl13: int, cl23: int, seed: int) -> bool:
    import fcntl
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12,
        "L1_key": str(L1.tolist()), "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": seed, "solver": "cpsat_proof",
    }
    lock_fd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best_in_pool = min((p["clashes"] for p in pool), default=999)
        if E > best_in_pool:
            return False
        pool.append(entry)
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def build_col_index(L: np.ndarray, n: int = N) -> np.ndarray:
    col_of = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            col_of[int(L[i, j]), i] = j
    return col_of


def relabel_L3_canonical(L3: np.ndarray, n: int = N) -> np.ndarray:
    """Relabel L3 so L3[0] = [0,...,n-1]. Symbol permutation preserves clashes."""
    perm = np.zeros(n, dtype=np.int32)
    for j in range(n):
        perm[int(L3[0, j])] = j
    return perm[L3.astype(np.int32)].astype(np.int8)


def feasibility_check(L1: np.ndarray, L2: np.ndarray, target: int,
                      n: int = N, hint_L3: np.ndarray | None = None,
                      timeout_s: float = 300.0,
                      num_workers: int = 4,
                      symbreak: bool = True,
                      ) -> tuple[str, np.ndarray | None]:
    """
    Check whether there exists L3 with cl13 + cl23 <= target.

    Adds hard constraint: sum(covered13) + sum(covered23) >= 2*n^2 - target.
    Symmetry breaking: fix L3[0][j]=j (cuts n! ≈ 3.6M redundancy).
    Returns ('SAT', L3_sol), ('UNSAT', None), or ('TIMEOUT', None).
    """
    model = cp_model.CpModel()

    L3v = [[model.new_int_var(0, n - 1, f"L3_{i}_{j}") for j in range(n)]
           for i in range(n)]

    for i in range(n):
        model.add_all_different(L3v[i])
    for j in range(n):
        model.add_all_different([L3v[i][j] for i in range(n)])

    # Symmetry breaking: fix first row to identity
    if symbreak:
        for j in range(n):
            model.add(L3v[0][j] == j)

    if hint_L3 is not None:
        hint_canon = relabel_L3_canonical(hint_L3, n)
        for i in range(n):
            for j in range(n):
                model.add_hint(L3v[i][j], int(hint_canon[i, j]))

    col_of_1 = build_col_index(L1, n)
    col_of_2 = build_col_index(L2, n)

    covered13 = []
    covered23 = []

    for a in range(n):
        for b in range(n):
            cov = model.new_bool_var(f"c13_{a}_{b}")
            indicators = []
            for i in range(n):
                j = int(col_of_1[a, i])
                ind = model.new_bool_var(f"i13_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                indicators.append(ind)
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

    # Hard feasibility constraint: must cover at least 2*n^2 - target pairs
    min_covered = 2 * n * n - target
    model.add(sum(covered13) + sum(covered23) >= min_covered)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False

    status = solver.solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        L3_sol = np.array([[solver.value(L3v[i][j]) for j in range(n)]
                           for i in range(n)], dtype=np.int8)
        return "SAT", L3_sol
    elif status == cp_model.INFEASIBLE:
        return "UNSAT", None
    else:
        return "TIMEOUT", None


def log(msg: str, log_fp=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n"); log_fp.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--timeout", type=float, default=300,
                        help="CP-SAT timeout per pair (seconds)")
    parser.add_argument("--workers", type=int, default=4,
                        help="CP-SAT parallel search workers")
    parser.add_argument("--target", type=int, default=35,
                        help="Feasibility target: check if E <= target is achievable")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, "a") as lfp:
        log("=" * 60, lfp)
        log(f"CP-SAT proof mode — seed={args.seed}  timeout={args.timeout}s  "
            f"workers={args.workers}  target={args.target}", lfp)
        log(f"Checking: does there exist L3 with cl13+cl23 <= {args.target}?", lfp)
        log("=" * 60, lfp)

        trial = 0
        sat_count = 0
        unsat_count = 0
        timeout_count = 0

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
            pool_E = entry["clashes"]

            log(f"trial={trial:4d}  pool_E={pool_E}  target={args.target}  "
                f"timeout={args.timeout}s  time={datetime.now().strftime('%H:%M:%S')}", lfp)

            t0 = time.time()
            status, L3_sol = feasibility_check(
                L1, L2, args.target, N, hint_L3=hint_L3,
                timeout_s=args.timeout, num_workers=args.workers)
            elapsed = time.time() - t0

            if status == "SAT":
                sat_count += 1
                cl12 = count_clashes(L1, L2)
                cl13 = count_clashes(L1, L3_sol)
                cl23 = count_clashes(L2, L3_sol)
                E = cl13 + cl23
                log(f"  SAT: E={E}  cl12={cl12}  cl13={cl13}  cl23={cl23}  "
                    f"elapsed={elapsed:.1f}s", lfp)

                if E == 0:
                    log(f"!!! N(10) >= 3 FOUND !!! seed={args.seed} trial={trial}", lfp)
                    save_found(L1, L2, L3_sol, args.seed)
                    print("MOLS10_FOUND written — N(10) >= 3 proven!", flush=True)
                    sys.exit(0)

                saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23, args.seed)
                if saved:
                    log(f"  SAVED to pool: new best E={E}  (was pool_E={pool_E})", lfp)

                # If we found E <= target, try even lower target
                if E <= args.target:
                    log(f"  -> target achieved! E={E} <= {args.target}", lfp)

            elif status == "UNSAT":
                unsat_count += 1
                log(f"  UNSAT: PROVEN E > {args.target} for this (L1,L2) pair  "
                    f"elapsed={elapsed:.1f}s", lfp)
                log(f"  Lower bound: min cl13+cl23 >= {args.target+1} for this pair", lfp)
            else:
                timeout_count += 1
                log(f"  TIMEOUT: inconclusive  elapsed={elapsed:.1f}s", lfp)

            if trial % 5 == 0:
                log(f"  [status] trials={trial}  SAT={sat_count}  "
                    f"UNSAT={unsat_count}  TIMEOUT={timeout_count}", lfp)


if __name__ == "__main__":
    main()
