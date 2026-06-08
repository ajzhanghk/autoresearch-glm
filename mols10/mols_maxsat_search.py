#!/usr/bin/env python3
"""
mols_maxsat_search.py — MAX-SAT guided L3 search worker.

Strategy: Given a MOLS pair (L1, L2) from the pool, use weighted MAX-SAT
(RC2 solver) to find an L3 that maximises the number of orthogonality
constraints satisfied simultaneously.

Unlike exact SAT (which returns UNSAT when no perfect L3 exists), MAX-SAT
returns the *closest possible* L3 — the one violating the fewest constraints.
This directly minimises the clash energy E = cl13 + cl23.

Encoding:
  Variables x[i,j,k] = 1 iff L3[i,j] = k  (1000 Boolean vars, n=10)
  Hard clauses (weight=∞):  C1 cell-unique, C2 row-unique, C3 col-unique
  Soft clauses (weight=1):  C4 orth-L1 exactly-one, C5 orth-L2 exactly-one
    (each orth constraint contributes 1 soft clause per (symbol,value) pair)

The MAX-SAT optimum = max satisfied soft clauses → min violated = min clashes.

Two modes per trial:
  1. FULL: encode all orthogonality constraints as soft — finds true MAX-SAT
     optimum (slowest but strongest).
  2. GUIDED: start from current best E=37 L3, encode only the clashing pairs
     as soft constraints — faster local repair via partial MAX-SAT.

Usage:
    python mols_maxsat_search.py --seed 1337 --timeout 30
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
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
TRIPLE_MISS_FILE = RESULTS_DIR / "near_miss_triple.json"
PROMISING_FILE   = RESULTS_DIR / "promising_pairs.json"
LOG_FILE         = RESULTS_DIR / "l3_maxsat.log"
N = 10

# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def count_clashes(A: np.ndarray, B: np.ndarray, n: int = N) -> int:
    pairs = A.ravel().astype(np.int32) * n + B.ravel().astype(np.int32)
    counts = np.bincount(pairs, minlength=n * n)
    return int(n * n - np.count_nonzero(counts))


def is_valid_ls(L: np.ndarray, n: int = N) -> bool:
    expected = set(range(n))
    for i in range(n):
        if set(L[i].tolist()) != expected or set(L[:, i].tolist()) != expected:
            return False
    return True


def var(i: int, j: int, k: int, n: int = N) -> int:
    return i * n * n + j * n + k + 1


def add_exactly_one_hard(wcnf: WCNF, lits: list[int]) -> None:
    wcnf.append(lits)  # at-least-one (hard)
    for a in range(len(lits)):
        for b in range(a + 1, len(lits)):
            wcnf.append([-lits[a], -lits[b]])  # at-most-one (hard)


def add_exactly_one_soft(wcnf: WCNF, lits: list[int], weight: int = 1) -> None:
    """Soft at-least-one + hard at-most-one (standard weighted partial MAX-SAT)."""
    wcnf.append(lits, weight=weight)  # soft at-least-one
    for a in range(len(lits)):        # hard at-most-one (keeps it a valid LS)
        for b in range(a + 1, len(lits)):
            wcnf.append([-lits[a], -lits[b]])


# ---------------------------------------------------------------------------
# MAX-SAT L3 search
# ---------------------------------------------------------------------------

def build_maxsat_l3(L1: np.ndarray, L2: np.ndarray, n: int = N,
                    L3_hint: np.ndarray | None = None) -> WCNF:
    """Build WCNF for MAX-SAT search of L3 closest to orthogonal with L1, L2.

    Hard:  C1 (cell unique), C2 (row unique), C3 (col unique) — always SAT.
    Soft:  For each ordered pair (a,b), "pair (a,b) appears at least once in
           L1⊕L3 superposition" (weight=1) and same for L2⊕L3.

    With only LS hard constraints, any valid LS satisfies hard clauses.
    Optimal MAX-SAT = maximize covered (a,b) pairs = minimize missing pairs
    = minimize clash count E.  Cost = number of violated soft = E_opt.
    """
    wcnf = WCNF()

    # C1: each cell has exactly one symbol (hard)
    for i in range(n):
        for j in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for k in range(n)])

    # C2: each symbol appears exactly once per row (hard)
    for i in range(n):
        for k in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for j in range(n)])

    # C3: each symbol appears exactly once per column (hard)
    for j in range(n):
        for k in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for i in range(n)])

    # C4: soft at-least-one for each L1⊕L3 pair (a,b)
    # Violated soft clause = pair (a,b) never appears = missing pair = +1 clash
    for a in range(n):
        for b in range(n):
            group = [var(i, j, b) for i in range(n) for j in range(n) if L1[i, j] == a]
            if group:
                wcnf.append(group, weight=1)

    # C5: soft at-least-one for each L2⊕L3 pair (a,b)
    for a in range(n):
        for b in range(n):
            group = [var(i, j, b) for i in range(n) for j in range(n) if L2[i, j] == a]
            if group:
                wcnf.append(group, weight=1)

    return wcnf


def extract_l3_from_model(model: list[int], n: int = N) -> np.ndarray:
    model_set = set(model)
    L3 = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if var(i, j, k) in model_set:
                    L3[i, j] = k
                    break
    return L3


def maxsat_l3_search(L1: np.ndarray, L2: np.ndarray, n: int,
                     timeout_s: float,
                     L3_hint: np.ndarray | None = None
                     ) -> tuple[np.ndarray | None, int, bool]:
    """Run RC2 MAX-SAT to find best L3. Returns (L3, cost, timed_out)."""
    wcnf = build_maxsat_l3(L1, L2, n, L3_hint)

    result_holder: list = [None, None, True]  # [L3, cost, timed_out]

    def _solve():
        try:
            with RC2(wcnf) as rc2:
                model = rc2.compute()
                if model is not None:
                    L3 = extract_l3_from_model(model, n)
                    cost = rc2.cost  # number of violated soft clauses
                    result_holder[0] = L3
                    result_holder[1] = cost
                    result_holder[2] = False
        except Exception:
            pass

    t = threading.Thread(target=_solve, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result_holder[0], result_holder[1], result_holder[2]


# ---------------------------------------------------------------------------
# Guided MAX-SAT: repair only the clashing pairs
# ---------------------------------------------------------------------------

def find_clashing_pairs(L_ref: np.ndarray, L3: np.ndarray, n: int) -> list[tuple[int,int]]:
    """Return (a,b) pairs that are missing from the L_ref/L3 superposition."""
    pairs = L_ref.ravel().astype(np.int32) * n + L3.ravel().astype(np.int32)
    counts = np.bincount(pairs, minlength=n * n)
    missing = []
    for idx in range(n * n):
        if counts[idx] == 0:
            missing.append((idx // n, idx % n))
    return missing


def build_guided_maxsat_l3(L1: np.ndarray, L2: np.ndarray,
                            L3_seed: np.ndarray, n: int = N) -> WCNF:
    """Guided MAX-SAT: hard LS validity only; soft clauses focus on the
    specific missing pairs from L3_seed (the 22+15=37 violations) plus
    soft preferences for cells that already contribute unique pairs.

    Smaller soft-clause count than full MAX-SAT → faster RC2 convergence.
    """
    wcnf = WCNF()

    # Hard: LS validity (C1, C2, C3)
    for i in range(n):
        for j in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for k in range(n)])
    for i in range(n):
        for k in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for j in range(n)])
    for j in range(n):
        for k in range(n):
            add_exactly_one_hard(wcnf, [var(i, j, k) for i in range(n)])

    # Soft: missing pairs from L3_seed vs L1 (weight=2 — fixing L1 clash is priority)
    missing13 = find_clashing_pairs(L1, L3_seed, n)
    for (a, b) in missing13:
        group = [var(i, j, b) for i in range(n) for j in range(n) if L1[i, j] == a]
        if group:
            wcnf.append(group, weight=2)

    # Soft: missing pairs from L3_seed vs L2 (weight=1)
    missing23 = find_clashing_pairs(L2, L3_seed, n)
    for (a, b) in missing23:
        group = [var(i, j, b) for i in range(n) for j in range(n) if L2[i, j] == a]
        if group:
            wcnf.append(group, weight=1)

    # Soft: soft preference for keeping cells that already contribute unique pairs
    pair_counts13 = np.bincount(
        L1.ravel().astype(np.int32)*n + L3_seed.ravel().astype(np.int32),
        minlength=n*n)
    pair_counts23 = np.bincount(
        L2.ravel().astype(np.int32)*n + L3_seed.ravel().astype(np.int32),
        minlength=n*n)
    for i in range(n):
        for j in range(n):
            k = int(L3_seed[i, j])
            p13 = pair_counts13[int(L1[i, j])*n + k]
            p23 = pair_counts23[int(L2[i, j])*n + k]
            if p13 == 1 and p23 == 1:
                wcnf.append([var(i, j, k)], weight=3)

    return wcnf


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
                 clashes: int, source: str = "maxsat") -> bool:
    if not is_valid_ls(L3) or count_clashes(L1, L2) != 0:
        return False
    with _pool_lock:
        try:
            data = load_pool()
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "clashes": clashes, "cl12": 0, "source": source,
                "L1_key": L1[0].tolist(),
                "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
            }
            data.append(entry)
            data.sort(key=lambda e: e["clashes"])
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


def run_forever(seed: int, timeout_s: float = 30.0) -> None:
    rng = random.Random(seed)
    trial = 0
    session_best = 999

    _log(f"maxsat_search started: seed={seed} timeout={timeout_s}s/trial")

    while True:
        trial += 1
        pool = load_pool()
        if not pool:
            _log("Pool empty, sleeping 15s...")
            time.sleep(15)
            continue

        best_e = pool[0]["clashes"]
        best_entries = [e for e in pool if e["clashes"] == best_e and e.get("cl12", 1) == 0]
        if not best_entries:
            best_entries = pool[:3]

        ref = rng.choice(best_entries[:min(5, len(best_entries))])
        L1 = np.array(ref["L1"], dtype=np.int8)
        L2 = np.array(ref["L2"], dtype=np.int8)
        L3_seed = np.array(ref["L3"], dtype=np.int8)

        if count_clashes(L1, L2) != 0:
            continue

        # Alternate between full and guided MAX-SAT
        use_guided = (trial % 3 != 0)
        strategy = "guided_maxsat" if use_guided else "full_maxsat"

        t0 = time.time()
        if use_guided:
            wcnf = build_guided_maxsat_l3(L1, L2, L3_seed)
            result_holder: list = [None, None, True]

            def _solve_guided(wcnf=wcnf, rh=result_holder):
                try:
                    with RC2(wcnf) as rc2:
                        model = rc2.compute()
                        if model is not None:
                            rh[0] = extract_l3_from_model(model)
                            rh[1] = rc2.cost
                            rh[2] = False
                except Exception:
                    pass

            t = threading.Thread(target=_solve_guided, daemon=True)
            t.start()
            t.join(timeout=timeout_s)
            L3_out, cost, timed_out = result_holder
        else:
            L3_out, cost, timed_out = maxsat_l3_search(L1, L2, N, timeout_s, L3_seed)

        elapsed = time.time() - t0

        if L3_out is None or not is_valid_ls(L3_out):
            _log(f"trial={trial:4d}  {strategy}  no valid L3 (timed_out={timed_out})  {elapsed:.1f}s")
            continue

        cl13 = count_clashes(L1, L3_out)
        cl23 = count_clashes(L2, L3_out)
        E_out = cl13 + cl23

        if E_out < session_best:
            session_best = E_out

        _log(
            f"trial={trial:4d}  {strategy:15s}  E={E_out}"
            f"  cl13={cl13}  cl23={cl23}"
            f"  cost={cost}  session_best={session_best}"
            f"  timed_out={timed_out}  elapsed={elapsed:.1f}s"
        )

        if E_out == 0:
            _log("=" * 60)
            _log("*** L3 FOUND! N(10) >= 3 PROVEN! ***")
            _log("=" * 60)
            save_to_pool(L1, L2, L3_out, 0, source="maxsat_FOUND")
            found_file = RESULTS_DIR / "MOLS10_FOUND.json"
            found_file.write_text(json.dumps({
                "ts": datetime.now().isoformat(), "strategy": strategy,
                "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3_out.tolist(),
                "cl12": 0, "cl13": 0, "cl23": 0, "E": 0,
            }, indent=2))
            _log(f"Written to {found_file}")
            sys.exit(0)

        if E_out <= best_e:
            saved = save_to_pool(L1, L2, L3_out, E_out, source=strategy)
            if saved and E_out < best_e:
                _log(f"  *** New pool best E={E_out}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int,   default=1337)
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="MAX-SAT timeout per trial (seconds)")
    args = parser.parse_args()
    run_forever(seed=args.seed, timeout_s=args.timeout)
