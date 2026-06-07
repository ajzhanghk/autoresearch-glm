#!/usr/bin/env python3
"""
mols_sat_worker.py — Direct SAT search for L3.

Encodes "L3 is a Latin square orthogonal to both L1 and L2" as CNF
and solves with Glucose3. This is a complete algorithm:
  - SAT  → L3 found, N(10) ≥ 3 proven
  - UNSAT → that specific (L1, L2) pair cannot have any orthogonal mate

Strategy:
  1. Test all pairs from promising_pairs.json (expected fast UNSAT for CT≤2 pairs)
  2. Generate fresh MOLS pairs via isotopy variants and random SA walks
  3. Run SAT for each new pair
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from pysat.solvers import Glucose3

sys.path.insert(0, str(Path(__file__).parent.parent))

from mols10.mols_adaptive import count_clashes, random_latin_square, verify_mols
from mols10.mols_l3_search import (
    _save_triple_miss, isotopy_variant,
    _row_swap, _col_swap, _relabel,
)

RESULTS_DIR = Path(__file__).parent / "results"
PROMISING_FILE = RESULTS_DIR / "promising_pairs.json"
N = 10


def _load_promising_pairs() -> list[tuple]:
    """Load all 310 promising MOLS pairs from promising_pairs.json."""
    data = json.loads(PROMISING_FILE.read_text())
    pairs = []
    for e in data:
        L1 = np.array(e["L1"], dtype=np.int8)
        L2 = np.array(e["L2"], dtype=np.int8)
        pairs.append((L1, L2))
    return pairs


def var(i: int, j: int, k: int, n: int = N) -> int:
    """1-indexed SAT variable for L3[i,j]=k."""
    return i * n * n + j * n + k + 1


def add_exactly_one(solver: Glucose3, lits: list[int]) -> None:
    """Add at-least-one and at-most-one (pairwise) clauses."""
    solver.add_clause(lits)
    for a in range(len(lits)):
        for b in range(a + 1, len(lits)):
            solver.add_clause([-lits[a], -lits[b]])


def build_l3_sat(L1: np.ndarray, L2: np.ndarray, n: int = N) -> Glucose3:
    """Build Glucose3 instance encoding 'L3 is a LS orthogonal to L1 and L2'."""
    solver = Glucose3()

    # 1. Each cell has exactly one value
    for i in range(n):
        for j in range(n):
            add_exactly_one(solver, [var(i, j, k) for k in range(n)])

    # 2. Row uniqueness (each value appears exactly once per row)
    for i in range(n):
        for k in range(n):
            add_exactly_one(solver, [var(i, j, k) for j in range(n)])

    # 3. Column uniqueness
    for j in range(n):
        for k in range(n):
            add_exactly_one(solver, [var(i, j, k) for i in range(n)])

    # 4. Orthogonality with L1: for each (a,b), exactly one cell with L1=a, L3=b
    for a in range(n):
        for b in range(n):
            group = [var(i, j, b) for i in range(n) for j in range(n) if L1[i, j] == a]
            add_exactly_one(solver, group)

    # 5. Orthogonality with L2: for each (a,b), exactly one cell with L2=a, L3=b
    for a in range(n):
        for b in range(n):
            group = [var(i, j, b) for i in range(n) for j in range(n) if L2[i, j] == a]
            add_exactly_one(solver, group)

    return solver


def find_l3_sat(
    L1: np.ndarray,
    L2: np.ndarray,
    timeout_sec: float = 60.0,
    n: int = N,
) -> tuple[bool, Optional[np.ndarray], float]:
    """
    Try to find L3 orthogonal to both L1 and L2.

    Returns (sat, L3, elapsed_sec).
      sat=True  → L3 found and verified
      sat=False → no L3 exists for this pair (proven by SAT)
    """
    solver = build_l3_sat(L1, L2, n)

    interrupted = [False]

    def _timeout():
        time.sleep(timeout_sec)
        if not interrupted[0]:
            solver.interrupt()

    t = threading.Thread(target=_timeout, daemon=True)
    t.start()

    t0 = time.time()
    try:
        sat = solver.solve_limited(expect_interrupt=True)
    except Exception:
        sat = None
    elapsed = time.time() - t0
    interrupted[0] = True

    if sat is True:
        model = set(solver.get_model())
        L3 = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if var(i, j, k) in model:
                        L3[i, j] = k
                        break
        solver.delete()
        return True, L3, elapsed
    else:
        solver.delete()
        # sat=False → UNSAT; sat=None → interrupted (timeout)
        is_unsat = (sat is False)
        return is_unsat, None, elapsed


def quick_mols_pair(
    n: int = N,
    rng: random.Random = None,
    max_steps: int = 500_000,
    temp: float = 0.3,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Fast SA to find a random MOLS pair (cl12=0).
    Returns (L1, L2) or None if not found within max_steps.
    """
    if rng is None:
        rng = random.Random()

    seed_pairs = _load_promising_pairs()
    if seed_pairs and rng.random() < 0.7:
        sp = rng.choice(seed_pairs)
        L1 = sp[0].copy()
        L2 = sp[1].copy()
        L1, L2 = isotopy_variant(L1, L2, n, random.Random(rng.random()))
    else:
        L1 = random_latin_square(n, random.Random(rng.random()))
        L2 = random_latin_square(n, random.Random(rng.random()))

    cl12 = count_clashes(L1, L2, n)
    best = cl12

    for step in range(max_steps):
        if cl12 == 0:
            return L1, L2
        # Move L2 only (L1 fixed)
        mv = rng.randint(0, 2)
        a, b = rng.sample(range(n), 2)
        if mv == 0:
            L2n = _row_swap(L2, a, b)
        elif mv == 1:
            L2n = _col_swap(L2, a, b)
        else:
            L2n = _relabel(L2, a, b)
        new_cl = count_clashes(L1, L2n, n)
        delta = new_cl - cl12
        T = temp * (0.9999 ** step)
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
            L2 = L2n
            cl12 = new_cl
            if cl12 < best:
                best = cl12
    return None


def save_solution(L1, L2, L3, method="sat_worker"):
    sol = {
        "found": True,
        "method": method,
        "ts": datetime.now().isoformat(),
        "L1": L1.tolist(),
        "L2": L2.tolist(),
        "L3": L3.tolist(),
    }
    path = RESULTS_DIR / "mols3_solution.json"
    path.write_text(json.dumps(sol, indent=2))
    print(f"  ★★★ SOLUTION SAVED to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11111)
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="SAT timeout per pair (seconds)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = N

    print(f"========================================================")
    print(f"MOLS SAT Worker — complete L3 search via Glucose3")
    print(f"seed={args.seed}  sat_timeout={args.timeout}s")
    print(f"========================================================")

    # Phase 1: Test all existing promising pairs
    pairs = _load_promising_pairs()
    print(f"\n[Phase 1] Testing {len(pairs)} existing promising pairs...")
    phase1_sat = 0
    phase1_unsat = 0
    phase1_timeout = 0
    for idx, sp in enumerate(pairs):
        L1 = np.array(sp[0], dtype=np.int8)
        L2 = np.array(sp[1], dtype=np.int8)
        cl12 = count_clashes(L1, L2, n)
        if cl12 != 0:
            continue  # skip non-MOLS pairs

        t0 = time.time()
        is_proven_unsat, L3, elapsed = find_l3_sat(L1, L2, timeout_sec=args.timeout)

        if L3 is not None:
            phase1_sat += 1
            ok, _ = verify_mols([L1, L2, L3])
            print(f"  [{idx:3d}] SAT!  elapsed={elapsed:.2f}s  verify={ok}")
            if ok:
                save_solution(L1, L2, L3)
                print(f"\n{'='*60}")
                print(f"★★★ N(10) ≥ 3 PROVEN in Phase 1! ★★★")
                print(f"{'='*60}")
                sys.exit(0)
        elif is_proven_unsat:
            phase1_unsat += 1
            if idx % 50 == 0:
                print(f"  [{idx:3d}] UNSAT  elapsed={elapsed:.2f}s  (so far: unsat={phase1_unsat})")
        else:
            phase1_timeout += 1
            print(f"  [{idx:3d}] TIMEOUT({args.timeout:.0f}s)  elapsed={elapsed:.2f}s")

    print(f"\n[Phase 1 done] pairs={len(pairs)}  SAT={phase1_sat}  "
          f"UNSAT={phase1_unsat}  TIMEOUT={phase1_timeout}")

    # Phase 2: Continuous new-pair generation + SAT
    print(f"\n[Phase 2] Generating new MOLS pairs + SAT search...")
    trial = 0
    session_sat = 0
    session_unsat = 0
    session_timeout = 0
    session_no_mols = 0

    while True:
        trial += 1
        pair_seed = rng.randint(0, 2 ** 31)
        pair_rng = random.Random(pair_seed)

        t_gen = time.time()
        pair = quick_mols_pair(n=n, rng=pair_rng, max_steps=300_000)
        t_gen_elapsed = time.time() - t_gen

        if pair is None:
            session_no_mols += 1
            if trial % 10 == 0:
                print(f"  trial={trial:5d}  no_mols={session_no_mols}  "
                      f"unsat={session_unsat}  timeout={session_timeout}  "
                      f"time={datetime.now().strftime('%H:%M:%S')}")
            continue

        L1, L2 = pair
        cl12 = count_clashes(L1, L2, n)
        if cl12 != 0:
            session_no_mols += 1
            continue

        t_sat = time.time()
        is_proven_unsat, L3, sat_elapsed = find_l3_sat(L1, L2, timeout_sec=args.timeout)
        t_total = time.time() - t_sat + t_gen_elapsed

        if L3 is not None:
            session_sat += 1
            ok, _ = verify_mols([L1, L2, L3])
            print(f"\n  trial={trial:5d}  SAT!  gen={t_gen_elapsed:.1f}s  "
                  f"sat={sat_elapsed:.2f}s  verify={ok}")
            if ok:
                save_solution(L1, L2, L3)
                print(f"\n{'='*60}")
                print(f"★★★ N(10) ≥ 3 PROVEN in Phase 2! ★★★")
                print(f"{'='*60}")
                sys.exit(0)
        elif is_proven_unsat:
            session_unsat += 1
        else:
            session_timeout += 1

        if trial % 20 == 0:
            print(f"  trial={trial:5d}  sat={session_sat}  unsat={session_unsat}  "
                  f"timeout={session_timeout}  no_mols={session_no_mols}  "
                  f"time={datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
