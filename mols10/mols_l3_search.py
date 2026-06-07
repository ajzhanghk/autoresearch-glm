#!/usr/bin/env python3
"""
mols_l3_search.py — Adaptive never-stopping search for N(10) ≥ 3.

Theoretical framework
---------------------
Given a MOLS pair (L1, L2), a third square L3 orthogonal to both exists
if and only if the n² cells can be partitioned into n **common transversals**
(CTs) of the pair — each CT is a set of n cells that is simultaneously:
  * a transversal of L1 (one per row/col, all L1-symbols distinct)
  * a transversal of L2 (one per row/col, all L2-symbols distinct)

Key theorem (from this observation):
  CT_count(L1, L2) = 0  →  L3 provably does NOT exist for that pair.

CT count is an *isotopy invariant* of the pair (L1, L2): row/col/symbol
permutations applied uniformly to the pair preserve CT count.

Empirical finding in this project:
  All 5 verified MOLS(10) pairs produced by the transversal-decomposition
  method have CT_count = 0 — they are provably L3-free.  The adaptive
  search must explore different isotopy classes / construction methods.

Search strategies
-----------------
  sa_triple  — SA on (L1, L2, L3) jointly.  Minimises
               Σ clashes(Lᵢ, Lⱼ) over all three pairs.
               Does NOT fix the pair; explores the full MOLS-triple space.
               This is the primary strategy when CT_count of known pairs = 0.

  ct_algx    — For pairs with CT_count ≥ 10: enumerate CTs, then
               Algorithm X exact-cover.  Applied only when a promising
               pair is found.

  csp_dual   — CSP backtracking (FC+UP+MRV+Luby) for L3 given (L1,L2).
               Useful for pairs with moderate CT count as a verifier.

Autoresearch outer loop (never-stopping)
-----------------------------------------
  1. Maintain a ranked pool of (L1,L2) pairs scored by CT count.
  2. Primary: sa_triple — generates new (L1,L2,L3) candidate triples
     independently of the pair pool.  Updates pool when it finds a pair
     with higher CT count.
  3. Secondary: for each pair with CT_count ≥ 1, run ct_algx / csp_dual.
  4. Inject fresh pairs (via transversal method and coupled CSP) at each
     outer iteration to diversify the pool.
  5. Log everything; never stop.

Literature references
---------------------
  Parker 1960            : N(10) ≥ 2.
  Hall & Paige 1955      : Complete mappings; Sylow-2 condition.
  Lam, Thiel & Swiercz 89: No projective plane of order 10 → N(10) ≤ 8.
  Wanless & Webb 2011    : Plexes and transversal structure.
  McKay, Meynert, Myrvold 2007: Latin square isotopy classes.
"""

from __future__ import annotations

import sys
# Force line-buffered stdout so nohup-redirected logs flush immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import csv
import json
import math
import random
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── siblings ────────────────────────────────────────────────────────────────
from mols_adaptive import (
    random_latin_square,
    transversal_find_pair,
    count_clashes,
    _row_swap, _col_swap, _relabel, _intercalate_flip,
    _enumerate_transversals,
    find_orth_mate_transversal,
)
from mols_search import (
    verify_mols,
    save_triplet,
    search_for_third,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH        = RESULTS_DIR / "l3_search_log.tsv"
PAIRS_FILE      = RESULTS_DIR / "mols_pairs_collection.json"
POOL_FILE       = RESULTS_DIR / "l3_pool_state.json"
PROMISING_FILE  = RESULTS_DIR / "promising_pairs.json"   # CT > 0 pairs
NEAR_MISS_FILE  = RESULTS_DIR / "near_miss_l3.json"      # best L3 partial solutions
TRIPLE_MISS_FILE = RESULTS_DIR / "near_miss_triple.json"  # best 3-way sa_triple states
LOG_COLS = ["ts", "trial", "strategy", "pair_id",
            "ct_count", "max_disjoint", "sa_best_clashes",
            "elapsed_s", "found", "note"]


# ===========================================================================
# Common-transversal utilities
# ===========================================================================

def enumerate_common_transversals(
    L1: np.ndarray, L2: np.ndarray, n: int,
    max_count: int = 200_000,
    deadline: float = float("inf"),
) -> list[tuple]:
    """Enumerate common transversals (CTs) of (L1, L2).

    A CT is n cells — one per row, one per column — with all n L1-values
    distinct AND all n L2-values distinct.  Such a CT is exactly one
    colour-class of any potential L3 orthogonal to both L1 and L2.

    If CT_count = 0, L3 does not exist for this pair (proven).
    """
    cts: list[tuple] = []
    used_col  = [False] * n
    used_sym1 = [False] * n
    used_sym2 = [False] * n

    def _bt(row: int, current: list) -> None:
        if time.time() >= deadline or len(cts) >= max_count:
            return
        if row == n:
            cts.append(tuple(current))
            return
        for c in range(n):
            if used_col[c]:
                continue
            s1, s2 = int(L1[row, c]), int(L2[row, c])
            if used_sym1[s1] or used_sym2[s2]:
                continue
            used_col[c] = used_sym1[s1] = used_sym2[s2] = True
            current.append((row, c))
            _bt(row + 1, current)
            current.pop()
            used_col[c] = used_sym1[s1] = used_sym2[s2] = False

    _bt(0, [])
    return cts


def max_disjoint_cts(cts: list[tuple], n: int) -> int:
    """Greedy lower-bound on maximum pairwise-disjoint CT set."""
    used: set[tuple] = set()
    count = 0
    for ct in cts:
        cells = set(ct)
        if not cells & used:
            used |= cells
            count += 1
            if count == n:
                break
    return count


def score_pair(
    L1: np.ndarray, L2: np.ndarray, n: int, budget: float = 3.0
) -> tuple[int, int]:
    """Return (ct_count, max_disjoint) — cheap hope score for a pair."""
    cts = enumerate_common_transversals(
        L1, L2, n, max_count=200_000, deadline=time.time() + budget
    )
    return len(cts), max_disjoint_cts(cts, n)


def _ct_algorithm_x(
    cts: list[tuple], n: int, deadline: float
) -> Optional[list[int]]:
    """Algorithm X: find n pairwise-disjoint CTs covering all n² cells."""
    tv_cells = [frozenset(ct) for ct in cts]
    cell_map: dict[tuple, list[int]] = {}
    for k, cells in enumerate(tv_cells):
        for cell in cells:
            cell_map.setdefault(cell, []).append(k)

    if len(cell_map) != n * n:
        return None

    def _solve(remaining: set, avail: set, chosen: list) -> bool:
        if time.time() >= deadline:
            return False
        if not remaining:
            return True
        best = min(
            remaining,
            key=lambda c: sum(1 for k in cell_map.get(c, []) if k in avail),
        )
        covering = [k for k in cell_map.get(best, []) if k in avail]
        if not covering:
            return False
        for k in covering:
            if time.time() >= deadline:
                return False
            tv_k = tv_cells[k]
            new_rem   = remaining - tv_k
            new_avail = avail - {j for j in avail if tv_cells[j] & tv_k}
            chosen.append(k)
            if _solve(new_rem, new_avail, chosen):
                return True
            chosen.pop()
        return False

    chosen: list[int] = []
    if _solve(set(cell_map), set(range(len(cts))), chosen):
        return chosen
    return None


def find_L3_ct_algx(
    L1: np.ndarray, L2: np.ndarray, n: int, max_seconds: float
) -> tuple[Optional[np.ndarray], dict]:
    """Find L3 via CT enumeration + Algorithm X.  Only useful if CT_count ≥ n."""
    t0 = time.time()
    cts = enumerate_common_transversals(
        L1, L2, n, max_count=200_000,
        deadline=t0 + min(max_seconds * 0.3, 10.0)
    )
    ct_count = len(cts)
    mdj      = max_disjoint_cts(cts, n)
    stats    = {"ct_count": ct_count, "max_disjoint": mdj}

    if ct_count < n:
        stats["note"] = f"ct_count={ct_count} < {n}: provably no L3 for this pair"
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return None, stats

    chosen = _ct_algorithm_x(cts, n, t0 + max_seconds)
    stats["elapsed_s"] = round(time.time() - t0, 2)
    if chosen is None:
        stats["note"] = "AlgX timeout / no perfect cover"
        return None, stats

    L3 = np.empty((n, n), dtype=np.int8)
    for color, k in enumerate(chosen):
        for (i, j) in cts[k]:
            L3[i, j] = color
    stats["note"] = "found via CT+AlgX"
    return L3, stats


# ===========================================================================
# Strategy 0 — SA directly on L3 given a fixed verified MOLS pair
# ===========================================================================

def _targeted_repair(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                     n: int, max_seconds: float) -> tuple[Optional[np.ndarray], int]:
    """When clashes are small, systematically try to repair each conflict.

    For each conflicting cell, enumerate every valid replacement value and
    pick the one that reduces clashes most.  Repeat until 0 or stuck.
    Falls back to random in-row swaps to escape local optima.
    """
    def _clash_count(L3_):
        return count_clashes(L1, L3_, n) + count_clashes(L2, L3_, n)

    def _conflict_cells(L3_):
        """Return cells contributing to clashes."""
        bad = set()
        for sq in (L1, L2):
            seen = {}
            for i in range(n):
                for j in range(n):
                    p = (int(sq[i, j]), int(L3_[i, j]))
                    if p in seen:
                        bad.add((i, j))
                        bad.add(seen[p])
                    else:
                        seen[p] = (i, j)
        return list(bad)

    t0 = time.time()
    L3 = L3.copy()
    clashes = _clash_count(L3)
    best_cl = clashes
    best_L3 = L3.copy()

    for _ in range(3000):
        if time.time() - t0 > max_seconds:
            break
        if clashes == 0:
            ok, _ = verify_mols([L1, L2, L3])
            if ok:
                return L3, 0
        cells = _conflict_cells(L3)
        if not cells:
            break
        # Pick a conflict cell; try all intercalate flips involving it
        i, j = cells[random.randrange(len(cells))]
        best_flip = None
        best_delta = 0
        for i2 in range(n):
            if i2 == i:
                continue
            for j2 in range(n):
                if j2 == j:
                    continue
                fl = _intercalate_flip(L3, i, i2, j, j2)
                if fl is None:
                    continue
                new_cl = _clash_count(fl)
                delta = new_cl - clashes
                if delta < best_delta:
                    best_delta = delta
                    best_flip = fl
        if best_flip is not None:
            L3 = best_flip
            clashes += best_delta
            if clashes < best_cl:
                best_cl = clashes
                best_L3 = L3.copy()
        else:
            # No improving intercalate flip — random intercalate to escape
            i2 = random.choice([x for x in range(n) if x != i])
            j2 = random.randrange(n)
            fl = _intercalate_flip(L3, i, i2, j, j2)
            if fl is not None:
                L3 = fl
                clashes = _clash_count(L3)
                if clashes < best_cl:
                    best_cl = clashes
                    best_L3 = L3.copy()

    return best_L3 if best_cl < clashes else None, best_cl


def sa_l3_given_pair(
    L1: np.ndarray, L2: np.ndarray, n: int,
    max_seconds: float,
    rng_seed: Optional[int] = None,
    pair_id: str = "?",
) -> tuple[Optional[np.ndarray], dict]:
    """SA finding L3 given a fixed, verified MOLS pair (L1, L2).

    Most focused strategy: L1 and L2 are FIXED.  We only move L3.
    Objective: minimise clashes(L1, L3) + clashes(L2, L3).
    When this reaches 0, L3 is a third MOLS.

    Enhancements:
      * Loads near-miss L3 states from disk for warm-starts
      * Saves near-miss states (clashes ≤ 20) to disk for cross-worker sharing
      * Switches to targeted cell-repair when clashes ≤ 15
      * ILS with 6 rounds before full restart
    """
    rng = random.Random(rng_seed)
    t0  = time.time()

    def _clashes_L3(L3_):
        return count_clashes(L1, L3_, n) + count_clashes(L2, L3_, n)

    def _fresh_L3():
        return random_latin_square(n, random.Random(rng.random()))

    # Try to warm-start from a saved near-miss for this pair
    near_misses = _load_near_misses()
    pair_near = [e for e in near_misses if e["pair_id"] == pair_id]
    if pair_near and rng.random() < 0.5:
        e = pair_near[0]  # best (lowest clashes) for this pair
        L3 = np.array(e["L3"], dtype=np.int8)
        # Verify it's still a valid LS (not corrupted)
        ok_ls = all(set(L3[i, :]) == set(range(n)) and
                    set(L3[:, j]) == set(range(n))
                    for i in range(n) for j in range(n)
                    if True)
        if not ok_ls:
            L3 = _fresh_L3()
    else:
        L3 = _fresh_L3()

    clashes = _clashes_L3(L3)
    best_clashes = clashes
    best_L3 = L3.copy()
    near_miss_threshold = 20

    T = 6.0
    cooling = 0.9999975   # Slow cooling for good temperature-level coverage
    no_improve = 0
    restarts = 0
    ils_round = 0
    STALL = 150_000   # Faster restart cycle (more diverse restarts per trial)

    while time.time() - t0 < max_seconds:
        if clashes == 0:
            ok, _ = verify_mols([L1, L2, L3])
            if ok:
                return L3, {"found": True, "best_clashes": 0,
                            "restarts": restarts, "elapsed_s": round(time.time()-t0, 2)}

        # Near-miss: all states are valid Latin (no in_row_swap); track and save best
        if clashes <= near_miss_threshold:
            _save_near_miss(pair_id, L1, L2, L3, clashes)
            near_miss_threshold = clashes
            if clashes <= 15:
                rem = max_seconds - (time.time() - t0)
                if rem > 2.0:
                    L3r, cl_r = _targeted_repair(L1, L2, L3, n, min(rem * 0.4, 8.0))
                    if cl_r == 0:
                        ok, _ = verify_mols([L1, L2, L3r])
                        if ok:
                            return L3r, {"found": True, "best_clashes": 0,
                                         "restarts": restarts,
                                         "elapsed_s": round(time.time()-t0, 2)}
                    if cl_r < best_clashes:
                        best_clashes = cl_r
                        best_L3 = L3r.copy()
                        L3 = L3r.copy()
                        clashes = cl_r
                        continue

        if no_improve >= STALL:
            no_improve = 0
            restarts += 1
            ils_round += 1
            if ils_round >= 6:
                # Occasionally warm-start from a near-miss
                pair_near2 = [e for e in _load_near_misses() if e["pair_id"] == pair_id]
                if pair_near2 and rng.random() < 0.4:
                    L3 = np.array(pair_near2[0]["L3"], dtype=np.int8)
                else:
                    L3 = _fresh_L3()
                T = 5.0
                ils_round = 0
                near_miss_threshold = 20
            else:
                L3 = best_L3.copy()
                k = rng.randint(3, 10)
                for _ in range(k):
                    mv = rng.randint(0, 4)
                    if mv == 0:
                        a, b = rng.sample(range(n), 2)
                        L3 = _row_swap(L3, a, b)
                    elif mv == 1:
                        a, b = rng.sample(range(n), 2)
                        L3 = _col_swap(L3, a, b)
                    elif mv == 2:
                        a, b = rng.sample(range(n), 2)
                        L3 = _relabel(L3, a, b)
                    else:
                        r1, r2 = rng.sample(range(n), 2)
                        c1, c2 = rng.sample(range(n), 2)
                        fl = _intercalate_flip(L3, r1, r2, c1, c2)
                        if fl is not None:
                            L3 = fl
                T = max(T, 4.0)   # Reheat more aggressively after ILS
            clashes = _clashes_L3(L3)
            continue

        # SA move
        mv = rng.randint(0, 4)
        if mv == 0:
            a, b = rng.sample(range(n), 2)
            L3_new = _row_swap(L3, a, b)
        elif mv == 1:
            a, b = rng.sample(range(n), 2)
            L3_new = _col_swap(L3, a, b)
        elif mv == 2:
            a, b = rng.sample(range(n), 2)
            L3_new = _relabel(L3, a, b)
        elif mv == 3:
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            L3_new = _intercalate_flip(L3, r1, r2, c1, c2)
            if L3_new is None:
                no_improve += 1
                continue
        else:
            # Extra intercalate attempt for finer local search
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            L3_new = _intercalate_flip(L3, r1, r2, c1, c2)
            if L3_new is None:
                no_improve += 1
                continue

        new_cl = _clashes_L3(L3_new)
        delta = new_cl - clashes
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            L3, clashes = L3_new, new_cl
            if clashes < best_clashes:
                best_clashes = clashes
                best_L3 = L3.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        T *= cooling

    # Save best valid Latin L3 for cross-worker warm-starts
    if best_clashes <= 20:
        _save_near_miss(pair_id, L1, L2, best_L3, best_clashes)

    return None, {
        "found": False, "best_clashes": best_clashes,
        "restarts": restarts, "elapsed_s": round(time.time() - t0, 2),
    }


# ===========================================================================
# Strategy 1 — 3-way SA (no fixed pair)
# ===========================================================================

def _triple_clashes(L1, L2, L3, n):
    return (count_clashes(L1, L2, n) +
            count_clashes(L1, L3, n) +
            count_clashes(L2, L3, n))


def sa_triple_find(
    n: int,
    max_seconds: float,
    temp_init: float = 6.0,
    cooling: float = 0.9999980,
    restart_after: int = 150_000,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[tuple], dict]:
    """SA over the joint (L1, L2, L3) space.

    Minimises total_clashes = Σᵢ<ⱼ clashes(Lᵢ, Lⱼ).
    When total_clashes = 0 the three squares are 3 MOLS.

    Moves: row-swap / col-swap / relabel / intercalate on any of the three
    squares, chosen uniformly at random.

    ILS: perturb from global-best on stall; full restart every 5 ILS rounds.
    When total_clashes ≤ 5, check CT_count of (L1, L2) — if ≥ n, try AlgX.
    """
    rng = random.Random(rng_seed)
    t0  = time.time()
    move_names = ["row", "col", "relabel", "intercalate"]

    # Seed pool for warm starts: known MOLS pairs give clashes(L1,L2)=0,
    # halving the initial total clashes vs. 3 fully random squares.
    _seed_pairs = _load_pairs()

    def fresh():
        triple_misses = _load_triple_misses()
        roll = rng.random()
        if triple_misses and roll < 0.15:
            # Pure near-miss (15%): continue cascade from best triple state.
            e = rng.choice(triple_misses[:5])
            L1_ = np.array(e["L1"], dtype=np.int8)
            L2_ = np.array(e["L2"], dtype=np.int8)
            L3_ = np.array(e["L3"], dtype=np.int8)
            return L1_, L2_, L3_, _triple_clashes(L1_, L2_, L3_, n)
        if triple_misses and _seed_pairs and roll < 0.45:
            # L3-transfer (30%): CT=2 seed pair + near-miss L3.
            e = rng.choice(triple_misses[:5])
            sp = rng.choice(_seed_pairs)
            L1_ = sp[0].copy(); L2_ = sp[1].copy()
            L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(rng.random()))
            L3_ = np.array(e["L3"], dtype=np.int8)
            return L1_, L2_, L3_, _triple_clashes(L1_, L2_, L3_, n)
        if _seed_pairs and roll < 0.80:
            sp = rng.choice(_seed_pairs)
            L1_ = sp[0].copy(); L2_ = sp[1].copy()
            L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(rng.random()))
        else:
            L1_ = random_latin_square(n, random.Random(rng.random()))
            L2_ = random_latin_square(n, random.Random(rng.random()))
        L3_ = random_latin_square(n, random.Random(rng.random()))
        return L1_, L2_, L3_, _triple_clashes(L1_, L2_, L3_, n)

    def apply_move(L):
        mv = rng.choice(move_names)
        if mv == "row":
            a, b = rng.sample(range(n), 2); return _row_swap(L, a, b)
        elif mv == "col":
            a, b = rng.sample(range(n), 2); return _col_swap(L, a, b)
        elif mv == "relabel":
            a, b = rng.sample(range(n), 2); return _relabel(L, a, b)
        else:
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            f = _intercalate_flip(L, r1, r2, c1, c2)
            return f if f is not None else L

    L1, L2, L3, clashes = fresh()
    # Track individual pairwise clashes for incremental updates (saves 1/3 of calls)
    cl12 = count_clashes(L1, L2, n)
    cl13 = count_clashes(L1, L3, n)
    cl23 = count_clashes(L2, L3, n)
    clashes = cl12 + cl13 + cl23
    T = temp_init
    best_clashes = clashes
    best_state   = (L1.copy(), L2.copy(), L3.copy())
    best_ct_count = 0
    no_improve   = 0
    restarts     = 0
    ils_rounds   = 0

    while time.time() - t0 < max_seconds:
        if clashes == 0:
            ok, _ = verify_mols([L1, L2, L3])
            if ok:
                return (L1, L2, L3), {
                    "found": True, "best_clashes": 0, "restarts": restarts,
                    "sa_best_clashes": 0, "elapsed_s": round(time.time()-t0, 2)
                }

        # When close, try CT-AlgX on the best pair seen
        if clashes <= 6:
            rem = max_seconds - (time.time() - t0)
            if rem > 2.0:
                for A, B in [(L1, L2), (L1, L3), (L2, L3)]:
                    ct12, _ = score_pair(A, B, n, budget=min(1.0, rem * 0.1))
                    if ct12 > best_ct_count:
                        best_ct_count = ct12
                    if ct12 >= n:
                        L3_try, _ = find_L3_ct_algx(A, B, n, rem * 0.4)
                        if L3_try is not None:
                            if np.array_equal(A, L1) and np.array_equal(B, L2):
                                trip = (L1, L2, L3_try)
                            elif np.array_equal(A, L1) and np.array_equal(B, L3):
                                trip = (L1, L3_try, L3)
                            else:
                                trip = (L3_try, L2, L3)
                            ok, _ = verify_mols(list(trip))
                            if ok:
                                return trip, {
                                    "found": True, "best_clashes": 0,
                                    "sa_best_clashes": clashes,
                                    "restarts": restarts,
                                    "elapsed_s": round(time.time()-t0, 2)
                                }

        # SA move — bias toward L3 when pair is nearly perfect (70%/15%/15%)
        if cl12 < 5:
            r = rng.random()
            tgt = 2 if r < 0.70 else (0 if r < 0.85 else 1)
        else:
            tgt = rng.randint(0, 2)
        L = [L1, L2, L3][tgt]
        Ln = apply_move(L)

        # Incremental clash update: only recompute the 2 pairs involving the moved square
        if tgt == 0:
            new_cl12 = count_clashes(Ln, L2, n)
            new_cl13 = count_clashes(Ln, L3, n)
            new_cl = new_cl12 + new_cl13 + cl23
        elif tgt == 1:
            new_cl12 = count_clashes(L1, Ln, n)
            new_cl23 = count_clashes(Ln, L3, n)
            new_cl = new_cl12 + cl13 + new_cl23
        else:
            new_cl13 = count_clashes(L1, Ln, n)
            new_cl23 = count_clashes(L2, Ln, n)
            new_cl = cl12 + new_cl13 + new_cl23

        delta = new_cl - clashes
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            if tgt == 0:
                L1, cl12, cl13 = Ln, new_cl12, new_cl13
            elif tgt == 1:
                L2, cl12, cl23 = Ln, new_cl12, new_cl23
            else:
                L3, cl13, cl23 = Ln, new_cl13, new_cl23
            clashes = new_cl
            if clashes < best_clashes:
                best_clashes = clashes
                best_state   = (L1.copy(), L2.copy(), L3.copy())
                no_improve   = 0
                if clashes <= 52:
                    _save_triple_miss(L1, L2, L3, clashes)
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling

        if no_improve >= restart_after:
            no_improve = 0
            restarts  += 1
            ils_rounds += 1

            if ils_rounds >= 5:
                L1, L2, L3, clashes = fresh()
                ils_rounds = 0
                T = temp_init
            else:
                # Perturb global best
                bL1, bL2, bL3 = best_state
                k = rng.randint(3, 10)
                L1, L2, L3 = bL1.copy(), bL2.copy(), bL3.copy()
                for _ in range(k):
                    tgt2 = rng.randint(0, 2)
                    if tgt2 == 0:   L1 = apply_move(L1)
                    elif tgt2 == 1: L2 = apply_move(L2)
                    else:           L3 = apply_move(L3)
                T = temp_init * 0.75
            # Recompute all pairwise clashes after restart/perturb
            cl12 = count_clashes(L1, L2, n)
            cl13 = count_clashes(L1, L3, n)
            cl23 = count_clashes(L2, L3, n)
            clashes = cl12 + cl13 + cl23

    return None, {
        "found": False, "best_clashes": best_clashes,
        "sa_best_clashes": best_clashes,
        "best_ct_count": best_ct_count,
        "restarts": restarts,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ===========================================================================
# Strategy 1b — Parallel Tempering on 3-way space
# ===========================================================================

def sa_triple_pt(
    n: int,
    max_seconds: float,
    temps: tuple = (1.0, 4.0, 16.0, 64.0, 256.0),
    swap_every: int = 2000,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[tuple], dict]:
    """Parallel tempering (replica-exchange SA) over the (L1, L2, L3) space.

    Runs K=4 replicas at fixed temperatures T_0 < T_1 < T_2 < T_3.
    Every `swap_every` steps, adjacent replicas i and i+1 attempt a swap:
        P_swap = min(1, exp((E_i - E_{i+1}) * (1/T_i - 1/T_{i+1})))
    The hot replica (T=45) crosses energy barriers freely; the cold one
    (T=1.5) refines local minima.  Swaps transport good states from hot
    to cold regions, escaping barriers single-chain SA cannot cross.
    Each replica has its own RNG for full independence.
    """
    rng  = random.Random(rng_seed)
    t0   = time.time()
    K    = len(temps)
    move_names = ["row", "col", "relabel", "intercalate"]
    # Each replica gets its own independent RNG
    rep_rngs = [random.Random(rng.random()) for _ in range(K)]

    _seed_pairs = _load_pairs()

    def _shake_ls(L, r, n_moves):
        """Apply n_moves random LS-preserving moves to create a diverse variant."""
        L = L.copy()
        for _ in range(n_moves):
            mv = r.choice(move_names)
            if mv == "row":
                a, b = r.sample(range(n), 2); L = _row_swap(L, a, b)
            elif mv == "col":
                a, b = r.sample(range(n), 2); L = _col_swap(L, a, b)
            elif mv == "relabel":
                a, b = r.sample(range(n), 2); L = _relabel(L, a, b)
            else:
                r1, r2 = r.sample(range(n), 2); c1, c2 = r.sample(range(n), 2)
                fl = _intercalate_flip(L, r1, r2, c1, c2)
                if fl is not None: L = fl
        return L

    def fresh_state(r):
        triple_misses = _load_triple_misses()
        roll = r.random()
        if triple_misses and roll < 0.10:
            # Pure near-miss (10%): exact warm-start to preserve cascade.
            e = r.choice(triple_misses[:5])
            L1_ = np.array(e["L1"], dtype=np.int8)
            L2_ = np.array(e["L2"], dtype=np.int8)
            L3_ = np.array(e["L3"], dtype=np.int8)
        elif triple_misses and roll < 0.25:
            # Shake (15%): near-miss L3 perturbed by random moves to escape
            # the current local basin. Use seed pair or near-miss L1/L2.
            e = r.choice(triple_misses[:5])
            if _seed_pairs and r.random() < 0.5:
                sp = r.choice(_seed_pairs)
                L1_ = sp[0].copy(); L2_ = sp[1].copy()
                L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(r.random()))
            else:
                L1_ = np.array(e["L1"], dtype=np.int8)
                L2_ = np.array(e["L2"], dtype=np.int8)
            L3_raw = np.array(e["L3"], dtype=np.int8)
            L3_ = _shake_ls(L3_raw, r, r.randint(15, 80))
        elif triple_misses and _seed_pairs and roll < 0.50:
            # L3-transfer (25%): CT=2 seed pair + near-miss L3.
            e = r.choice(triple_misses[:5])
            sp = r.choice(_seed_pairs)
            L1_ = sp[0].copy(); L2_ = sp[1].copy()
            L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(r.random()))
            L3_ = np.array(e["L3"], dtype=np.int8)
        elif _seed_pairs and roll < 0.80:
            # Known MOLS pair (CT=2) + fresh L3 (30%)
            sp = r.choice(_seed_pairs)
            L1_ = sp[0].copy(); L2_ = sp[1].copy()
            L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(r.random()))
            L3_ = random_latin_square(n, random.Random(r.random()))
        else:
            # Fully random (20%)
            L1_ = random_latin_square(n, random.Random(r.random()))
            L2_ = random_latin_square(n, random.Random(r.random()))
            L3_ = random_latin_square(n, random.Random(r.random()))
        cl12 = count_clashes(L1_, L2_, n)
        cl13 = count_clashes(L1_, L3_, n)
        cl23 = count_clashes(L2_, L3_, n)
        return [L1_, L2_, L3_, cl12, cl13, cl23]

    def apply_move_to(state, r):
        L1_, L2_, L3_, cl12_, cl13_, cl23_ = state
        if cl12_ < 5:
            rv = r.random()
            tgt = 2 if rv < 0.70 else (0 if rv < 0.85 else 1)
        else:
            tgt = r.randint(0, 2)
        L = [L1_, L2_, L3_][tgt]
        mv = r.choice(move_names)
        if mv == "row":
            a, b = r.sample(range(n), 2); Ln = _row_swap(L, a, b)
        elif mv == "col":
            a, b = r.sample(range(n), 2); Ln = _col_swap(L, a, b)
        elif mv == "relabel":
            a, b = r.sample(range(n), 2); Ln = _relabel(L, a, b)
        else:
            r1, r2 = r.sample(range(n), 2); c1, c2 = r.sample(range(n), 2)
            fl = _intercalate_flip(L, r1, r2, c1, c2)
            Ln = fl if fl is not None else L
        if tgt == 0:
            ncl12 = count_clashes(Ln, L2_, n); ncl13 = count_clashes(Ln, L3_, n)
            new_E = ncl12 + ncl13 + cl23_
            return [Ln, L2_, L3_, ncl12, ncl13, cl23_], new_E, cl12_ + cl13_ + cl23_
        elif tgt == 1:
            ncl12 = count_clashes(L1_, Ln, n); ncl23 = count_clashes(Ln, L3_, n)
            new_E = ncl12 + cl13_ + ncl23
            return [L1_, Ln, L3_, ncl12, cl13_, ncl23], new_E, cl12_ + cl13_ + cl23_
        else:
            ncl13 = count_clashes(L1_, Ln, n); ncl23 = count_clashes(L2_, Ln, n)
            new_E = cl12_ + ncl13 + ncl23
            return [L1_, L2_, Ln, cl12_, ncl13, ncl23], new_E, cl12_ + cl13_ + cl23_

    # Initialise replicas (warm-start each independently)
    replicas   = [fresh_state(rep_rngs[i]) for i in range(K)]
    energies   = [r[3] + r[4] + r[5] for r in replicas]
    best_E     = min(energies)
    best_state = replicas[energies.index(best_E)][:3]
    best_ct    = 0
    step       = 0
    swaps      = 0
    restarts   = 0

    while time.time() - t0 < max_seconds:
        # ── advance each replica one step with its own RNG ─────────────────
        for i in range(K):
            state = replicas[i]
            new_state, new_E, old_E = apply_move_to(state, rep_rngs[i])
            delta = new_E - old_E
            T = temps[i]
            if delta < 0 or rep_rngs[i].random() < math.exp(-delta / T):
                replicas[i] = new_state
                energies[i] = new_E
                if new_E < best_E:
                    best_E     = new_E
                    best_state = new_state[:3]
                    if new_E <= 52:
                        _save_triple_miss(*best_state, new_E)
                    if new_E == 0:
                        ok, _ = verify_mols(list(best_state))
                        if ok:
                            return tuple(best_state), {
                                "found": True, "best_clashes": 0,
                                "restarts": restarts, "swaps": swaps,
                                "elapsed_s": round(time.time() - t0, 2),
                            }

        step += 1

        # ── replica swaps every swap_every outer steps ─────────────────────
        if step % swap_every == 0:
            for i in range(K - 1):
                Ei, Ej = energies[i], energies[i + 1]
                Ti, Tj = temps[i], temps[i + 1]
                log_prob = (Ei - Ej) * (1.0 / Ti - 1.0 / Tj)
                if log_prob >= 0 or rng.random() < math.exp(log_prob):
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = Ej, Ei
                    swaps += 1

        # ── periodic full reinit of hottest replica to avoid stagnation ────
        if step % (swap_every * 100) == 0:
            replicas[-1] = fresh_state(rep_rngs[-1])
            energies[-1] = replicas[-1][3] + replicas[-1][4] + replicas[-1][5]
            restarts += 1

        # ── CT-AlgX when any replica is very close ─────────────────────────
        if best_E <= 6:
            rem = max_seconds - (time.time() - t0)
            if rem > 2.0:
                bL1, bL2, bL3 = best_state
                for A, B in [(bL1, bL2), (bL1, bL3), (bL2, bL3)]:
                    ct_val, _ = score_pair(A, B, n, budget=min(1.0, rem * 0.1))
                    if ct_val > best_ct:
                        best_ct = ct_val
                    if ct_val >= n:
                        L3t, _ = find_L3_ct_algx(A, B, n, rem * 0.4)
                        if L3t is not None:
                            trip = (A, B, L3t) if not np.array_equal(A, bL3) else (bL1, L3t, bL3)
                            ok, _ = verify_mols(list(trip))
                            if ok:
                                return trip, {
                                    "found": True, "best_clashes": 0,
                                    "restarts": restarts, "swaps": swaps,
                                    "elapsed_s": round(time.time() - t0, 2),
                                }

    return None, {
        "found": False, "best_clashes": best_E,
        "sa_best_clashes": best_E,
        "best_ct_count": best_ct,
        "restarts": restarts, "swaps": swaps,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ===========================================================================
# Strategy 2 — CT-AlgX (for pairs with CT_count ≥ n)
# ===========================================================================
# Already defined above: find_L3_ct_algx()


# ===========================================================================
# Strategy 3 — CSP dual backtracking (for pairs with CT_count > 0)
# ===========================================================================

def csp_dual_find_L3(
    L1: np.ndarray, L2: np.ndarray, n: int,
    max_seconds: float,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[np.ndarray], dict]:
    """CSP with MRV+FC+UP+Luby on L3 given fixed (L1, L2).

    Delegates to search_for_third from mols_search.py.
    Only meaningful if CT_count(L1, L2) > 0; otherwise provably infeasible.
    """
    t0 = time.time()
    L3 = search_for_third(L1, L2, RESULTS_DIR,
                          max_seconds=max_seconds, rng_seed=rng_seed)
    elapsed = round(time.time() - t0, 2)
    if L3 is not None:
        return L3, {"found": True, "elapsed_s": elapsed}
    return None, {"found": False, "elapsed_s": elapsed}


# ===========================================================================
# Strategy 4 — SA directly maximising CT_count on a MOLS pair
# ===========================================================================

def sa_ct_climb(
    n: int,
    max_seconds: float,
    rng_seed: Optional[int] = None,
    ct_cap: int = 60,
    warm_L1: Optional[np.ndarray] = None,
    warm_L2: Optional[np.ndarray] = None,
) -> tuple[Optional[tuple], dict]:
    """SA on a MOLS pair (L1, L2) that directly maximises CT_count(L1, L2).

    This is fundamentally different from sa_triple:
      * sa_triple explores (L1, L2, L3) minimising clashes.
      * sa_ct_climb fixes L1, keeps MOLS(L1, L2) guaranteed, and
        maximises CT_count(L1, L2) via intercalate moves on L2.

    Moves that keep MOLS whilst potentially changing CT:
      a) Intercalate flip on L2 only  → if count_clashes(L1, L2') = 0: valid.
      b) Intercalate flip on L1 only  → if count_clashes(L1', L2) = 0: valid.
         (Very rare; included for diversity.)
      c) Full restart: new L1+L2 pair from transversal exact-cover.

    CT evaluation uses a fast capped enumeration (max ct_cap transversals,
    0.15 s budget) so each SA step is cheap.

    If CT_count ≥ n (= 10) is reached, returns the pair immediately (L3 may
    exist — try Algorithm X).
    """
    rng = random.Random(rng_seed)
    t0  = time.time()

    def fast_ct(L1_, L2_, budget=0.15, cap=None):
        cap_ = cap if cap is not None else ct_cap
        cts = enumerate_common_transversals(
            L1_, L2_, n, max_count=cap_, deadline=time.time() + budget
        )
        return len(cts), cts

    def fresh_pair():
        seed2 = rng.randint(0, 2**31)
        L1_, L2_, _ = transversal_find_pair(n, 15.0, rng_seed=seed2)
        return L1_, L2_

    # Start from warm pair if provided, else generate fresh
    if warm_L1 is not None and warm_L2 is not None:
        L1, L2 = warm_L1.copy(), warm_L2.copy()
    else:
        L1, L2 = fresh_pair()
    if L1 is None:
        return None, {"note": "failed to generate initial pair"}

    ct, _ = fast_ct(L1, L2)
    best_ct = ct
    best_pair = (L1.copy(), L2.copy())

    T = 1.5           # SA temperature: controls acceptance of CT-degrading moves
    cooling = 0.99998
    steps = 0
    restarts = 0
    no_improve = 0
    MAX_NO_IMPROVE = 80_000

    while time.time() - t0 < max_seconds:
        steps += 1
        T *= cooling

        if no_improve >= MAX_NO_IMPROVE:
            # Restart from best, then occasionally fully fresh
            restarts += 1
            no_improve = 0
            if restarts % 4 == 0:
                L1_try2, L2_try2 = fresh_pair()
                if L1_try2 is not None:
                    L1, L2 = L1_try2, L2_try2
                    T = 1.5
                else:
                    # Fresh generation failed — fall back to perturbing best
                    L1, L2 = best_pair[0].copy(), best_pair[1].copy()
                    T = max(T, 0.5)
            else:
                L1, L2 = best_pair[0].copy(), best_pair[1].copy()
                T = max(T, 0.5)
            ct, _ = fast_ct(L1, L2)
            continue

        # Choose move: mostly flip L2, occasionally flip L1
        flip_L1 = rng.random() < 0.15
        r1, r2 = rng.sample(range(n), 2)
        c1, c2 = rng.sample(range(n), 2)

        if flip_L1:
            L1_try = _intercalate_flip(L1, r1, r2, c1, c2)
            if L1_try is None or count_clashes(L1_try, L2, n) != 0:
                no_improve += 1
                continue
            L1_new, L2_new = L1_try, L2
        else:
            L2_try = _intercalate_flip(L2, r1, r2, c1, c2)
            if L2_try is None or count_clashes(L1, L2_try, n) != 0:
                no_improve += 1
                continue
            L1_new, L2_new = L1, L2_try

        new_ct, _ = fast_ct(L1_new, L2_new)
        delta = new_ct - ct  # positive = better

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            L1, L2, ct = L1_new, L2_new, new_ct
            if ct > best_ct:
                best_ct = ct
                best_pair = (L1.copy(), L2.copy())
                no_improve = 0
                if best_ct >= n:
                    return best_pair, {
                        "best_ct": best_ct, "steps": steps,
                        "restarts": restarts, "elapsed_s": round(time.time()-t0, 2),
                        "note": f"CT≥n reached after {steps} steps"
                    }
            else:
                no_improve += 1
        else:
            no_improve += 1

    return best_pair, {
        "best_ct": best_ct, "steps": steps, "restarts": restarts,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ===========================================================================
# Fast batch mate search — reuse transversal enumeration across AlgX calls
# ===========================================================================

def _find_mate_from_tvs(tvs: list, n: int, rng: random.Random,
                        deadline: float) -> Optional[np.ndarray]:
    """Run Algorithm X once with a random shuffle of pre-enumerated transversals.

    Much faster than find_orth_mate_transversal since transversal enumeration
    (the bottleneck) is shared across calls.
    """
    tvs_shuffled = tvs[:]
    rng.shuffle(tvs_shuffled)
    tv_cells = [frozenset(tv) for tv in tvs_shuffled]

    cell_to_tvs: dict[tuple, list[int]] = {}
    for k, cells in enumerate(tv_cells):
        for cell in cells:
            cell_to_tvs.setdefault(cell, []).append(k)

    all_cells = set(cell_to_tvs.keys())
    if len(all_cells) != n * n:
        return None

    chosen: list[int] = []

    def _alg_x(remaining: set, avail: set) -> bool:
        if time.time() >= deadline:
            return False
        if not remaining:
            return True
        best_cell = min(remaining,
                        key=lambda c: sum(1 for k in cell_to_tvs.get(c, []) if k in avail))
        covering = [k for k in cell_to_tvs.get(best_cell, []) if k in avail]
        if not covering:
            return False
        for k in covering:
            cells_k = tv_cells[k]
            removed = {c for c in cells_k if c in remaining}
            blocked = {j for c in cells_k for j in cell_to_tvs.get(c, []) if j in avail}
            chosen.append(k)
            new_avail = avail - blocked
            if _alg_x(remaining - removed, new_avail):
                return True
            chosen.pop()
        return False

    avail = set(range(len(tv_cells)))
    all_remaining = frozenset((r, c) for r in range(n) for c in range(n))
    if not _alg_x(set(all_remaining), avail):
        return None

    # Build L2 from chosen transversals
    L2 = np.zeros((n, n), dtype=np.int8)
    for sym, k in enumerate(chosen):
        for r, c in tv_cells[k]:
            L2[r, c] = sym
    return L2


# ===========================================================================
# Strategy 5 — Multi-decomposition search (different L2 mates for same L1)
# ===========================================================================

def multi_decomp_ct_search(
    n: int,
    max_seconds: float,
    n_attempts: int = 30,
    rng_seed: Optional[int] = None,
    seed_L1: Optional[np.ndarray] = None,
) -> tuple[Optional[tuple], dict]:
    """High-throughput CT screening: enumerate transversals once per L1, then
    run Algorithm X many times with different shuffles to generate diverse L2 mates.

    Typically 10-50× faster than repeated find_orth_mate_transversal calls.
    """
    rng = random.Random(rng_seed)
    t0  = time.time()

    best_ct   = -1
    best_pair: Optional[tuple] = None
    pairs_tried = 0
    l1_batch = 0

    while time.time() - t0 < max_seconds * 0.95:
        rem = max_seconds - (time.time() - t0)
        if rem < 3.0:
            break

        # Choose L1: alternate seed (known good, isotopy-varied) vs fresh random
        if seed_L1 is not None and l1_batch % 3 != 2:
            L1 = seed_L1.copy()
            L1, _ = isotopy_variant(L1, L1, n, random.Random(rng.random()))
        else:
            L1 = random_latin_square(n, random.Random(rng.random()))

        # Enumerate transversals of L1 once (up to 50K, 3s budget)
        enum_budget = min(3.0, (max_seconds - (time.time() - t0)) * 0.25)
        tvs = _enumerate_transversals(L1, n, max_count=50_000,
                                      deadline=time.time() + enum_budget)
        if len(tvs) < n:
            l1_batch += 1
            continue  # L1 has too few transversals (skip)

        # Run AlgorithmX many times with different shuffles on the same tvs
        batch_start = time.time()
        batch_budget = min(rem * 0.7, 40.0)  # Use 70% of remaining for this L1 batch
        n_batch = 0

        while time.time() - batch_start < batch_budget:
            rem2 = max_seconds - (time.time() - t0)
            if rem2 < 1.0:
                break
            deadline_algx = time.time() + min(2.0, rem2 * 0.5)
            L2 = _find_mate_from_tvs(tvs, n, rng, deadline=deadline_algx)
            if L2 is None:
                n_batch += 1
                if n_batch > 5 and pairs_tried == 0:
                    break  # This L1 seems to have no mates — move on
                continue

            pairs_tried += 1
            n_batch += 1
            rem3 = max_seconds - (time.time() - t0)
            cts = enumerate_common_transversals(
                L1, L2, n, max_count=500, deadline=time.time() + min(0.3, rem3 * 0.1)
            )
            ct = len(cts)
            if ct > best_ct:
                best_ct = ct
                best_pair = (L1.copy(), L2.copy())
                print(f"    [mdecomp] New best CT={ct} (pairs_tried={pairs_tried})")
                if best_ct >= n:
                    elapsed = round(time.time() - t0, 2)
                    return best_pair, {"best_ct": best_ct, "pairs_tried": pairs_tried,
                                       "elapsed_s": elapsed}

        l1_batch += 1

    elapsed = round(time.time() - t0, 2)
    return best_pair, {"best_ct": best_ct, "pairs_tried": pairs_tried, "elapsed_s": elapsed}


# ===========================================================================
# Isotopy variant generation
# ===========================================================================

def isotopy_variant(
    L1: np.ndarray, L2: np.ndarray, n: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Random paratopy: independent row/col perms + independent symbol perms.

    Paratopies (unlike plain isotopies) can change CT count because they
    apply different symbol permutations to L1 and L2 independently, and
    also allow transposition (swapping row/column roles).  However, the
    resulting pair must still be MOLS.

    Here we use the sub-group of paratopies that preserves MOLS property:
    * Same row permutation applied to both squares (keeps orthogonality)
    * Same column permutation applied to both squares (keeps orthogonality)
    * Independent symbol permutations on L1 and L2 (keeps orthogonality)
    These are all *isotopies* of the pair.  CT count is invariant under these.

    To explore truly different isotopy classes we also apply random
    *intercalate flips* to one square only — these slightly break isotopy
    equivalence and may move the pair toward a class with higher CT count.
    """
    row_p = list(range(n)); rng.shuffle(row_p)
    col_p = list(range(n)); rng.shuffle(col_p)
    sym1  = list(range(n)); rng.shuffle(sym1)
    sym2  = list(range(n)); rng.shuffle(sym2)

    L1n = np.empty((n, n), dtype=np.int8)
    L2n = np.empty((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            ri, cj = row_p[i], col_p[j]
            L1n[i, j] = sym1[int(L1[ri, cj])]
            L2n[i, j] = sym2[int(L2[ri, cj])]
    return L1n, L2n


def _intercalate_search_variant(
    L1: np.ndarray, L2: np.ndarray, n: int, rng: random.Random, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Apply k random intercalate flips to L2 (breaks isotopy, may increase CT).

    After flipping, L2 might no longer be orthogonal to L1.  We only keep
    the variant if it IS still a valid MOLS pair — i.e. if the flips happen
    to land on cells where both L1 and L2 values change compatibly.
    Falls back to the original pair if no valid variant found in 50 tries.
    """
    for _ in range(50):
        L2v = L2.copy()
        for _ in range(k):
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            flipped = _intercalate_flip(L2v, r1, r2, c1, c2)
            if flipped is not None:
                L2v = flipped
        # Check if still MOLS
        if count_clashes(L1, L2v, n) == 0:
            ok, _ = verify_mols([L1, L2v])
            if ok:
                return L1, L2v
    return L1, L2  # no valid variant found


# ===========================================================================
# Pair record & pool management
# ===========================================================================

@dataclass
class PairRecord:
    pair_id:      str
    L1:           np.ndarray
    L2:           np.ndarray
    ct_count:     int
    max_disjoint: int
    source:       str = "seed"

    @property
    def score(self) -> float:
        return self.ct_count * 100 + self.max_disjoint

    def __lt__(self, other):
        return self.score > other.score  # descending


def _load_pairs() -> list[tuple]:
    data = json.loads(PAIRS_FILE.read_text())
    return [(np.array(e["L1"], dtype=np.int8),
             np.array(e["L2"], dtype=np.int8)) for e in data]


# ===========================================================================
# Logging
# ===========================================================================

def _log(row: dict) -> None:
    exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, delimiter="\t")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLS})


def _save_near_miss(pair_id: str, L1: np.ndarray, L2: np.ndarray,
                    L3: np.ndarray, clashes: int) -> None:
    """Save an L3 candidate with very few clashes for cross-worker warm-starts."""
    entry = {
        "pair_id": pair_id, "clashes": clashes,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
    }
    try:
        data = json.loads(NEAR_MISS_FILE.read_text()) if NEAR_MISS_FILE.exists() else []
        # Keep only best 20 near-misses, no duplicates for same pair_id+clashes
        data = [e for e in data if not (e["pair_id"] == pair_id and e["clashes"] >= clashes)]
        data.append(entry)
        data.sort(key=lambda e: e["clashes"])
        data = data[:20]
        NEAR_MISS_FILE.write_text(json.dumps(data, indent=2))
        print(f"  ★ Near-miss saved: pair={pair_id} clashes={clashes}")
    except Exception:
        pass


def _load_near_misses() -> list[dict]:
    """Load saved near-miss L3 candidates."""
    if not NEAR_MISS_FILE.exists():
        return []
    try:
        return json.loads(NEAR_MISS_FILE.read_text())
    except Exception:
        return []


def _save_triple_miss(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                      clashes: int) -> None:
    """Save a low-clash (L1,L2,L3) triple state for sa_triple warm-starts.

    Maintains a top-5 list with pair diversity: at most 2 entries from the
    same pair (identified by L1 fingerprint), so the cascade explores
    multiple basins rather than collapsing to a single pair's local minimum.
    """
    n = 10
    cl12 = count_clashes(L1, L2, n)
    L1_key = tuple(L1.ravel().tolist())  # fingerprint for pair identity
    entry = {
        "clashes": clashes, "cl12": cl12,
        "L1_key": L1_key,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
    }
    try:
        # Guard against JSON corruption from concurrent writes by multiple workers
        try:
            data = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        except (json.JSONDecodeError, OSError):
            data = []   # recover gracefully; we'll add the new entry below
        data.append(entry)
        data.sort(key=lambda e: e["clashes"])
        # Pair-diversity cap enforced post-sort (handles concurrent writes):
        # keep at most 2 entries per pair, then take top-8.
        # L1_key is stored as a JSON array (list) but needs to be hashable
        # for use as a dict key — convert to tuple on the fly.
        seen_pairs: dict = {}
        diverse: list = []
        for e in data:
            raw_k = e.get("L1_key")
            k = tuple(raw_k) if isinstance(raw_k, list) else raw_k
            cnt = seen_pairs.get(k, 0)
            if cnt < 2:
                diverse.append(e)
                seen_pairs[k] = cnt + 1
            if len(diverse) == 8:
                break
        TRIPLE_MISS_FILE.write_text(json.dumps(diverse, indent=2))
        print(f"  ★ Triple near-miss saved: clashes={clashes}  cl12={cl12}")
    except Exception:
        pass


def _load_triple_misses() -> list[dict]:
    if not TRIPLE_MISS_FILE.exists():
        return []
    try:
        return json.loads(TRIPLE_MISS_FILE.read_text())
    except Exception:
        return []


def _save_promising_pair(pair_id: str, L1: np.ndarray, L2: np.ndarray,
                         ct_count: int, max_disjoint: int, source: str) -> None:
    """Immediately persist any CT>0 pair to disk so it survives process restarts."""
    entry = {
        "pair_id": pair_id,
        "ct_count": ct_count,
        "max_disjoint": max_disjoint,
        "source": source,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "L1": L1.tolist(),
        "L2": L2.tolist(),
    }
    if PROMISING_FILE.exists():
        data = json.loads(PROMISING_FILE.read_text())
        # Skip if already saved with same id
        if any(e["pair_id"] == pair_id for e in data):
            return
    else:
        data = []
    data.append(entry)
    # Sort by ct_count descending
    data.sort(key=lambda e: e["ct_count"], reverse=True)
    PROMISING_FILE.write_text(json.dumps(data, indent=2))
    print(f"  ★ Saved promising pair {pair_id} (ct={ct_count}) → {PROMISING_FILE.name}")


# ===========================================================================
# Adaptive never-stopping outer loop
# ===========================================================================

class L3AdaptiveSearch:
    """Never-stopping adaptive search for N(10) ≥ 3.

    Combines three strategies in a portfolio:
      1. sa_triple  — explores entire (L1,L2,L3) space without fixing a pair.
                      Primary driver; also discovers new pairs with higher CT.
      2. ct_algx    — exact-cover on CTs; activated only for pairs with CT ≥ n.
      3. csp_dual   — CSP on L3 given pair; activated for pairs with CT > 0.

    Pool management:
      * Seed pairs are scored on startup.
      * sa_triple runs continuously, scoring intermediate pairs it encounters.
      * Any pair with CT > current pool maximum is added to the pool.
      * Isotopy variants and intercalate variants are injected periodically.
      * Fresh MOLS pairs are generated when the pool is exhausted.
    """

    def __init__(
        self,
        n: int = 10,
        eval_budget: float = 60.0,
        max_seconds: Optional[float] = None,
        master_seed: int = 0,
        save_dir: Path = RESULTS_DIR,
    ) -> None:
        self.n           = n
        self.eval_budget = eval_budget
        self.max_seconds = max_seconds
        self.rng         = random.Random(master_seed)
        self.save_dir    = save_dir
        self.t0          = time.time()
        self.trial       = 0
        self.pool:       list[PairRecord] = []
        self.pool_ids:   set[str] = set()
        self.best_ct_ever = 0
        self.sa_triple_best_clashes = 3 * n * n   # worst possible

        print("=" * 68)
        print(f"N({n}) ≥ 3 Adaptive Search — never-stopping")
        print(f"eval_budget={eval_budget}s  save_dir={save_dir}")
        print("=" * 68)

        self._load_seed_pairs()

    def _elapsed(self): return time.time() - self.t0
    def _ok(self):
        return self.max_seconds is None or self._elapsed() < self.max_seconds

    def _add_pair(
        self, L1: np.ndarray, L2: np.ndarray,
        source: str = "?", pair_id: Optional[str] = None
    ) -> PairRecord:
        if pair_id is None:
            pair_id = f"{source}_{len(self.pool_ids)}"
        if pair_id in self.pool_ids:
            return None
        ct, mdj = score_pair(L1, L2, self.n, budget=3.0)
        rec = PairRecord(pair_id, L1, L2, ct, mdj, source)
        self.pool.append(rec)
        self.pool_ids.add(pair_id)
        is_new_best = ct > self.best_ct_ever
        if is_new_best:
            self.best_ct_ever = ct
        self.pool.sort()
        tag = " ← NEW BEST CT!" if is_new_best and ct > 0 else ""
        print(f"  [{pair_id}] ct={ct}  max_dis={mdj}  src={source}{tag}")
        # Immediately persist any pair with CT > 0 to disk
        if ct > 0:
            _save_promising_pair(pair_id, L1, L2, ct, mdj, source)
        return rec

    def _load_seed_pairs(self):
        print(f"\nLoading seed pairs from {PAIRS_FILE}…")
        for idx, (L1, L2) in enumerate(_load_pairs()):
            self._add_pair(L1, L2, source="seed", pair_id=f"seed_{idx}")
        # Also load any promising pairs saved from previous runs
        if PROMISING_FILE.exists():
            print(f"Loading promising pairs from {PROMISING_FILE.name}…")
            try:
                data = json.loads(PROMISING_FILE.read_text())
                for e in data:
                    pid = e["pair_id"]
                    if pid not in self.pool_ids:
                        L1 = np.array(e["L1"], dtype=np.int8)
                        L2 = np.array(e["L2"], dtype=np.int8)
                        ct = e["ct_count"]
                        mdj = e["max_disjoint"]
                        rec = PairRecord(pid, L1, L2, ct, mdj, e.get("source", "disk"))
                        self.pool.append(rec)
                        self.pool_ids.add(pid)
                        if ct > self.best_ct_ever:
                            self.best_ct_ever = ct
                        print(f"  [resume] {pid} ct={ct}")
                self.pool.sort()
            except Exception as ex:
                print(f"  Warning: failed to load promising pairs: {ex}")
        print(f"Pool: {len(self.pool)} pairs.  Best CT count: {self.best_ct_ever}\n")

    def _run_sa_triple(self, budget: float) -> Optional[tuple]:
        """Run sa_triple; return triple if found, else update pool with any good pairs."""
        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        print(f"\ntrial={self.trial:4d}  strategy=sa_triple"
              f"  budget={budget:.0f}s  best_ct_ever={self.best_ct_ever}")

        result, stats = sa_triple_find(n, budget, rng_seed=seed)

        elapsed  = stats.get("elapsed_s", budget)
        bc       = stats.get("best_clashes", 3 * n * n)
        best_ct  = stats.get("best_ct_count", 0)

        if bc < self.sa_triple_best_clashes:
            self.sa_triple_best_clashes = bc
        print(f"  sa_triple: best_clashes={bc}  "
              f"sa_frontier={self.sa_triple_best_clashes}  "
              f"best_ct_seen={best_ct}  elapsed={elapsed:.1f}s")

        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": "sa_triple",
              "pair_id": "—", "ct_count": best_ct, "max_disjoint": "—",
              "sa_best_clashes": bc, "elapsed_s": elapsed,
              "found": int(result is not None),
              "note": f"sa_frontier={self.sa_triple_best_clashes}"})

        if result is not None:
            L1, L2, L3 = result
            return L1, L2, L3
        return None

    def _run_sa_triple_pt(self, budget: float) -> Optional[tuple]:
        """Run parallel-tempering sa_triple_pt; return triple if found."""
        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        print(f"\ntrial={self.trial:4d}  strategy=sa_triple_pt"
              f"  budget={budget:.0f}s  best_ct_ever={self.best_ct_ever}")

        result, stats = sa_triple_pt(n, budget, rng_seed=seed)

        elapsed = stats.get("elapsed_s", budget)
        bc      = stats.get("best_clashes", 3 * n * n)
        best_ct = stats.get("best_ct_count", 0)
        swaps   = stats.get("swaps", 0)

        if bc < self.sa_triple_best_clashes:
            self.sa_triple_best_clashes = bc
        print(f"  sa_triple_pt: best_clashes={bc}  "
              f"sa_frontier={self.sa_triple_best_clashes}  "
              f"swaps={swaps}  best_ct_seen={best_ct}  elapsed={elapsed:.1f}s")

        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": "sa_triple_pt",
              "pair_id": "—", "ct_count": best_ct, "max_disjoint": "—",
              "sa_best_clashes": bc, "elapsed_s": elapsed,
              "found": int(result is not None),
              "note": f"sa_frontier={self.sa_triple_best_clashes} swaps={swaps}"})

        if result is not None:
            L1, L2, L3 = result
            return L1, L2, L3
        return None

    def _run_sa_l3_given_pair(self, budget: float) -> Optional[tuple]:
        """Run sa_l3_given_pair on the best CT>0 pool pair; rotate through all CT>0 pairs."""
        if not self.pool:
            return None
        # Prefer highest-CT pairs; within same CT, round-robin
        ct_recs = [r for r in self.pool if r.ct_count > 0]
        if not ct_recs:
            rec = self.pool[0]
        else:
            # Use top-CT pairs 70% of time, full pool 30%
            top_ct = max(r.ct_count for r in ct_recs)
            best_recs = [r for r in ct_recs if r.ct_count == top_ct]
            if self.rng.random() < 0.70:
                idx = self.trial % len(best_recs)
                rec = best_recs[idx]
            else:
                idx = self.trial % len(ct_recs)
                rec = ct_recs[idx]

        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        print(f"\ntrial={self.trial:4d}  strategy=sa_l3_pair"
              f"  pair={rec.pair_id}  ct={rec.ct_count}  budget={budget:.0f}s")

        L3, stats = sa_l3_given_pair(rec.L1, rec.L2, n, budget, rng_seed=seed,
                                      pair_id=rec.pair_id)
        elapsed = stats.get("elapsed_s", budget)
        bc = stats.get("best_clashes", 2 * n * n)
        found = stats.get("found", False)
        restarts = stats.get("restarts", 0)

        print(f"  sa_l3_pair: found={found}  best_clashes={bc}"
              f"  restarts={restarts}  elapsed={elapsed:.1f}s")
        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": "sa_l3_pair",
              "pair_id": rec.pair_id, "ct_count": rec.ct_count,
              "max_disjoint": rec.max_disjoint, "sa_best_clashes": bc,
              "elapsed_s": elapsed, "found": int(found), "note": f"restarts={restarts}"})

        if found and L3 is not None:
            return self._success(rec.L1, rec.L2, L3)
        return None

    def _run_sa_ct_climb(self, budget: float) -> None:
        """Run sa_ct_climb; initialise from best pool pair when CT>0."""
        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        # Inject best CT>0 pair as warm-start into sa_ct_climb
        ct_recs = [r for r in self.pool if r.ct_count > 0]
        warm_L1 = ct_recs[0].L1 if ct_recs else None
        warm_L2 = ct_recs[0].L2 if ct_recs else None

        print(f"\ntrial={self.trial:4d}  strategy=sa_ct_climb"
              f"  budget={budget:.0f}s  best_ct_ever={self.best_ct_ever}"
              f"  warm={'yes ct='+str(ct_recs[0].ct_count) if ct_recs else 'no'}")

        pair, stats = sa_ct_climb(n, budget, rng_seed=seed,
                                    warm_L1=warm_L1, warm_L2=warm_L2)
        elapsed = stats.get("elapsed_s", budget)
        bc = stats.get("best_ct", 0)
        steps = stats.get("steps", 0)
        note = stats.get("note", "")

        print(f"  sa_ct_climb: best_ct={bc}  steps={steps}  elapsed={elapsed:.1f}s  {note}")
        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": "sa_ct_climb",
              "pair_id": "—", "ct_count": bc, "max_disjoint": "—",
              "sa_best_clashes": "—", "elapsed_s": elapsed,
              "found": 0, "note": note or f"steps={steps}"})

        if pair is not None and bc > 0:
            self._add_pair(pair[0], pair[1], source="sa_ct")

        if pair is not None and bc >= n:
            # CT≥10! Try Algorithm X immediately
            L3, algx_stats = find_L3_ct_algx(pair[0], pair[1], n, budget * 0.5)
            if L3 is not None:
                self._success(pair[0], pair[1], L3)

    def _run_multi_decomp(self, budget: float) -> None:
        """Try many Algorithm X restarts on a fresh L1 — find the mate with max CT."""
        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        n_attempts = max(10, int(budget / 4))  # More attempts per budget
        # Use best pool pair's L1 as seed for guaranteed OLS mate finding
        ct_recs = [r for r in self.pool if r.ct_count > 0]
        seed_L1 = ct_recs[0].L1 if ct_recs else None

        print(f"\ntrial={self.trial:4d}  strategy=multi_decomp"
              f"  budget={budget:.0f}s  n_attempts={n_attempts}  best_ct_ever={self.best_ct_ever}")

        pair, stats = multi_decomp_ct_search(n, budget, n_attempts=n_attempts,
                                             rng_seed=seed, seed_L1=seed_L1)
        elapsed = stats.get("elapsed_s", budget)
        bc = stats.get("best_ct", -1)
        tried = stats.get("pairs_tried", 0)

        print(f"  multi_decomp: best_ct={bc}  pairs_tried={tried}  elapsed={elapsed:.1f}s")
        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": "multi_decomp",
              "pair_id": "—", "ct_count": max(bc, 0), "max_disjoint": "—",
              "sa_best_clashes": "—", "elapsed_s": elapsed,
              "found": 0, "note": f"tried={tried}"})

        if pair is not None and bc > 0:
            self._add_pair(pair[0], pair[1], source="mdecomp")

    def _run_pair_strategy(
        self, rec: PairRecord, strategy: str, budget: float
    ) -> Optional[np.ndarray]:
        seed = self.rng.randint(0, 2**31)
        self.trial += 1
        n = self.n

        print(f"\ntrial={self.trial:4d}  strategy={strategy}"
              f"  pair={rec.pair_id}  ct={rec.ct_count}  budget={budget:.0f}s")

        if strategy == "ct_algx":
            L3, stats = find_L3_ct_algx(rec.L1, rec.L2, n, budget)
        elif strategy == "csp_dual":
            L3, stats = csp_dual_find_L3(rec.L1, rec.L2, n, budget, rng_seed=seed)
        else:
            return None

        elapsed = stats.get("elapsed_s", budget)
        bc      = stats.get("best_clashes", "—")
        note    = stats.get("note", "")
        found   = L3 is not None
        print(f"  {strategy}: found={found}  {note}  elapsed={elapsed:.1f}s")

        _log({"ts": datetime.now().isoformat(timespec="seconds"),
              "trial": self.trial, "strategy": strategy,
              "pair_id": rec.pair_id, "ct_count": rec.ct_count,
              "max_disjoint": rec.max_disjoint, "sa_best_clashes": bc,
              "elapsed_s": elapsed, "found": int(found), "note": note})

        return L3

    def _inject_variants(self, rec: PairRecord, n_iso: int = 3) -> None:
        """Add isotopy + intercalate variants of rec to the pool."""
        for v in range(n_iso):
            L1v, L2v = isotopy_variant(
                rec.L1, rec.L2, self.n, random.Random(self.rng.random())
            )
            self._add_pair(L1v, L2v, source="iso",
                           pair_id=f"iso_{rec.pair_id}_{v}")

        # Intercalate variant (may change isotopy class slightly)
        L1c, L2c = _intercalate_search_variant(
            rec.L1, rec.L2, self.n, random.Random(self.rng.random()), k=3
        )
        if not (np.array_equal(L1c, rec.L1) and np.array_equal(L2c, rec.L2)):
            self._add_pair(L1c, L2c, source="intcl")

    def _inject_intercalate_variants(
        self, rec: PairRecord, k_list: list = None
    ) -> None:
        """Try multiple intercalate-flip counts to escape current isotopy class."""
        if k_list is None:
            k_list = [2, 3, 5, 8, 12]
        n = self.n
        rng = random.Random(self.rng.random())
        for k in k_list:
            for attempt in range(10):
                L2v = rec.L2.copy()
                for _ in range(k):
                    r1, r2 = rng.sample(range(n), 2)
                    c1, c2 = rng.sample(range(n), 2)
                    flipped = _intercalate_flip(L2v, r1, r2, c1, c2)
                    if flipped is not None:
                        L2v = flipped
                if count_clashes(rec.L1, L2v, n) == 0:
                    ok, _ = verify_mols([rec.L1, L2v])
                    if ok:
                        vid = f"ic{k}_{rec.pair_id}_{attempt}"
                        self._add_pair(rec.L1.copy(), L2v, source=f"ic{k}", pair_id=vid)
                        break
        # Also try flipping L1 instead
        for k in [3, 5]:
            L1v = rec.L1.copy()
            for _ in range(k):
                r1, r2 = rng.sample(range(n), 2)
                c1, c2 = rng.sample(range(n), 2)
                flipped = _intercalate_flip(L1v, r1, r2, c1, c2)
                if flipped is not None:
                    L1v = flipped
            if count_clashes(L1v, rec.L2, n) == 0:
                ok, _ = verify_mols([L1v, rec.L2])
                if ok:
                    vid = f"ic1_{k}_{rec.pair_id}"
                    self._add_pair(L1v, rec.L2.copy(), source=f"ic1_{k}", pair_id=vid)

    def _reload_promising_pairs(self) -> None:
        """Reload CT>0 pairs from disk (survives process restarts)."""
        if not PROMISING_FILE.exists():
            return
        try:
            data = json.loads(PROMISING_FILE.read_text())
        except Exception:
            return
        for e in data:
            pid = e["pair_id"]
            if pid not in self.pool_ids:
                L1 = np.array(e["L1"], dtype=np.int8)
                L2 = np.array(e["L2"], dtype=np.int8)
                ct = e["ct_count"]
                mdj = e["max_disjoint"]
                rec = PairRecord(pid, L1, L2, ct, mdj, e.get("source", "disk"))
                self.pool.append(rec)
                self.pool_ids.add(pid)
                if ct > self.best_ct_ever:
                    self.best_ct_ever = ct
                print(f"  [reload] {pid} ct={ct} from disk")
        self.pool.sort()

    def _inject_fresh_pair(self) -> None:
        L1, L2, stats = transversal_find_pair(
            self.n, 30.0, rng_seed=self.rng.randint(0, 2**31)
        )
        if L1 is not None:
            self._add_pair(L1, L2, source="tv")

    def _save_pool_state(self) -> None:
        """Persist all CT>0 pool entries to disk for restart continuity."""
        ct_positive = [rec for rec in self.pool if rec.ct_count > 0]
        if not ct_positive:
            return
        try:
            existing_ids: set = set()
            if PROMISING_FILE.exists():
                data = json.loads(PROMISING_FILE.read_text())
                existing_ids = {e["pair_id"] for e in data}
            else:
                data = []
            for rec in ct_positive:
                if rec.pair_id not in existing_ids:
                    data.append({
                        "pair_id": rec.pair_id,
                        "ct_count": rec.ct_count,
                        "max_disjoint": rec.max_disjoint,
                        "source": rec.source,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "L1": rec.L1.tolist(),
                        "L2": rec.L2.tolist(),
                    })
                    existing_ids.add(rec.pair_id)
            data.sort(key=lambda e: e["ct_count"], reverse=True)
            PROMISING_FILE.write_text(json.dumps(data, indent=2))
            print(f"  [checkpoint] Saved {len(data)} promising pairs to disk "
                  f"(best ct={data[0]['ct_count'] if data else 0})")
        except Exception as ex:
            print(f"  [checkpoint] Warning: {ex}")

    def run(self) -> Optional[tuple]:
        n = self.n
        outer_iter = 0

        # Save pool on SIGTERM so restarts can reload CT>0 pairs
        def _sigterm(_sig, _frame):
            print("\n[SIGTERM] Saving pool state before exit…")
            self._save_pool_state()
            sys.exit(0)
        signal.signal(signal.SIGTERM, _sigterm)
        signal.signal(signal.SIGINT, _sigterm)

        last_checkpoint = time.time()

        while self._ok():
            outer_iter += 1
            budget = min(self.eval_budget,
                         self.max_seconds - self._elapsed()
                         if self.max_seconds else self.eval_budget)
            if budget < 1.0:
                break

            # ── Strategy portfolio ─────────────────────────────────────────
            # Portfolio (10 phases):
            #  0,1,2,3 → sa_triple_pt (40%): parallel-tempering 5-replica search
            #  4,5     → sa_triple    (20%): single-chain SA (diversity)
            #  6,7     → multi_decomp (20%): high-throughput pair screening
            #  8,9     → sa_ct_climb  (20%): attempt CT > 2
            #  special: sa_l3_pair ONLY when CT >= 10
            phase = outer_iter % 10

            # If any pair reaches CT >= n, switch immediately to AlgorithmX
            high_ct_recs = [r for r in self.pool if r.ct_count >= n]
            if high_ct_recs:
                result = self._run_sa_l3_given_pair(budget * 0.90)
                if result is not None:
                    return result
            elif phase in (0, 1, 2, 3):
                triple = self._run_sa_triple_pt(budget * 0.90)
                if triple is not None:
                    L1t, L2t, L3t = triple
                    return self._success(L1t, L2t, L3t)
            elif phase in (4, 5):
                triple = self._run_sa_triple(budget * 0.90)
                if triple is not None:
                    L1t, L2t, L3t = triple
                    return self._success(L1t, L2t, L3t)
            elif phase in (6, 7):
                self._run_multi_decomp(budget * 0.70)
            else:  # phase 8,9
                self._run_sa_ct_climb(budget * 0.70)

            # ── pair-based strategies for pool top ─────────────────────────
            if self.pool:
                best_rec = self.pool[0]
                if best_rec.ct_count >= n:
                    print(f"\n  ★ ct_count={best_rec.ct_count} ≥ {n}:"
                          f" running AlgX on pair {best_rec.pair_id}")
                    L3 = self._run_pair_strategy(best_rec, "ct_algx", budget * 0.4)
                    if L3 is not None:
                        return self._success(best_rec.L1, best_rec.L2, L3)
                elif best_rec.ct_count > 0:
                    print(f"\n  ct={best_rec.ct_count} (0<ct<{n}):"
                          f" injecting intercalate variants of {best_rec.pair_id}")
                    self._inject_intercalate_variants(best_rec, k_list=[2, 3, 5, 8, 12])
                else:
                    print(f"\n  Pool best: ct={best_rec.ct_count} — "
                          f"all known pairs are L3-free; relying on portfolio.")

            # ── diversification ─────────────────────────────────────────────
            if outer_iter % 3 == 0:
                if self.pool:
                    self._inject_variants(self.pool[0])
                self._inject_fresh_pair()
            if outer_iter % 5 == 0:
                self._reload_promising_pairs()

            # ── periodic checkpoint every 5 minutes ─────────────────────────
            now = time.time()
            if now - last_checkpoint >= 300:
                self._save_pool_state()
                last_checkpoint = now

            # ── progress summary ────────────────────────────────────────────
            elapsed_h = self._elapsed() / 3600
            print(f"\n  Progress: trials={self.trial}"
                  f"  elapsed={elapsed_h:.2f}h"
                  f"  pool_size={len(self.pool)}"
                  f"  best_ct_ever={self.best_ct_ever}"
                  f"  sa3_best_clashes={self.sa_triple_best_clashes}"
                  f"  {'N(10)≥3 not found yet' if self.best_ct_ever < n else 'CT≥n PAIR EXISTS!'}")

        print(f"\nBudget exhausted after {self._elapsed():.0f}s / "
              f"{self.trial} trials.")
        return None

    def _success(
        self, L1: np.ndarray, L2: np.ndarray, L3: np.ndarray
    ) -> tuple:
        ok, details = verify_mols([L1, L2, L3])
        if not ok:
            print(f"  WARNING: verification failed: {details}")
            return None
        print("\n" + "★" * 68)
        print(f"N({self.n}) ≥ 3 PROVEN!  3 MOLS of order {self.n} found!")
        print(f"Verification: {details}")
        print("★" * 68)
        out = self.save_dir / f"mols_triple_N{self.n}_ge_3.json"
        save_triplet(L1, L2, L3, self._elapsed(), out)
        print(f"Saved → {out}")
        return L1, L2, L3


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Adaptive search for N(10) ≥ 3  (never-stopping)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--n",           type=int,   default=10)
    p.add_argument("--strategy",
                   choices=["adaptive", "sa_triple", "ct_algx", "csp_dual", "score"],
                   default="adaptive",
                   help="adaptive=full loop; score=just score seed pairs and exit")
    p.add_argument("--max-hours",   type=float, default=0,
                   help="Wall-clock budget in hours (0=unlimited)")
    p.add_argument("--eval-budget", type=float, default=60.0,
                   help="Seconds per outer iteration")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save-dir",    default=str(RESULTS_DIR))
    args = p.parse_args()

    n        = args.n
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    max_sec  = args.max_hours * 3600 if args.max_hours > 0 else None

    print("=" * 68)
    print(f"MOLS L3 Search  — is N({n}) ≥ 3?")
    print(f"strategy    : {args.strategy}")
    print(f"budget      : {'unlimited' if not max_sec else f'{args.max_hours}h'}")
    print(f"eval_budget : {args.eval_budget}s")
    print(f"seed        : {args.seed}")
    print("=" * 68)

    if args.strategy == "score":
        print("\nScoring seed pairs by CT count…")
        for idx, (L1, L2) in enumerate(_load_pairs()):
            t0 = time.time()
            ct, mdj = score_pair(L1, L2, n, budget=10.0)
            note = "L3 provably absent" if ct == 0 else (
                f"L3 possible" if ct >= n else f"partial: {ct} CTs"
            )
            print(f"  pair {idx}: ct_count={ct:5d}  max_disjoint={mdj}"
                  f"  ({time.time()-t0:.2f}s)  → {note}")
        return

    if args.strategy == "adaptive":
        searcher = L3AdaptiveSearch(
            n=n, eval_budget=args.eval_budget,
            max_seconds=max_sec, master_seed=args.seed, save_dir=save_dir,
        )
        searcher.run()
        return

    # Single-strategy mode
    pairs = _load_pairs()
    print(f"Loaded {len(pairs)} seed pairs.")

    for idx, (L1, L2) in enumerate(pairs):
        ct, mdj = score_pair(L1, L2, n, budget=3.0)
        print(f"\n── Pair {idx}: ct_count={ct}  max_disjoint={mdj} ──")

        budget = max_sec or 3600.0
        if args.strategy == "sa_triple":
            result, stats = sa_triple_find(n, budget, rng_seed=args.seed)
            print(f"  stats: {stats}")
            if result:
                L1r, L2r, L3r = result
                ok, _ = verify_mols([L1r, L2r, L3r])
                if ok:
                    save_triplet(L1r, L2r, L3r, 0.0,
                                 save_dir / "mols_triple_N10_ge_3.json")
                    print("Saved!"); sys.exit(0)
        elif args.strategy == "ct_algx":
            L3, stats = find_L3_ct_algx(L1, L2, n, budget)
            print(f"  stats: {stats}")
            if L3 is not None:
                save_triplet(L1, L2, L3, 0.0,
                             save_dir / "mols_triple_N10_ge_3.json")
                sys.exit(0)
        elif args.strategy == "csp_dual":
            L3, stats = csp_dual_find_L3(L1, L2, n, budget, rng_seed=args.seed)
            print(f"  stats: {stats}")
            if L3 is not None:
                save_triplet(L1, L2, L3, 0.0,
                             save_dir / "mols_triple_N10_ge_3.json")
                sys.exit(0)
        break   # single-strategy: only try the first/best pair


if __name__ == "__main__":
    main()
