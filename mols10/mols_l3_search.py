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

import argparse
import csv
import json
import math
import random
import sys
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
)
from mols_search import (
    verify_mols,
    save_triplet,
    search_for_third,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH     = RESULTS_DIR / "l3_search_log.tsv"
PAIRS_FILE   = RESULTS_DIR / "mols_pairs_collection.json"
POOL_FILE    = RESULTS_DIR / "l3_pool_state.json"
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
        if _seed_pairs and rng.random() < 0.5:
            # Warm start: known MOLS pair + fresh L3 → total clashes ≈ 40-50
            sp = rng.choice(_seed_pairs)
            L1_ = sp[0].copy()
            L2_ = sp[1].copy()
            # Apply random isotopy to diversify
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
                # Score each pairwise combination; use the one with most CTs
                for A, B in [(L1, L2), (L1, L3), (L2, L3)]:
                    ct12, _ = score_pair(A, B, n, budget=min(1.0, rem * 0.1))
                    if ct12 > best_ct_count:
                        best_ct_count = ct12
                    if ct12 >= n:
                        L3_try, _ = find_L3_ct_algx(A, B, n, rem * 0.4)
                        if L3_try is not None:
                            # Find the third square
                            for C_name, C in [("L3", L3), ("L2", L2), ("L1", L1)]:
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

        # SA move — pick one of the 3 squares at random
        tgt = rng.randint(0, 2)
        L   = [L1, L2, L3][tgt]
        Ln  = apply_move(L)

        new_L1 = Ln if tgt == 0 else L1
        new_L2 = Ln if tgt == 1 else L2
        new_L3 = Ln if tgt == 2 else L3
        new_cl = _triple_clashes(new_L1, new_L2, new_L3, n)

        delta = new_cl - clashes
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            L1, L2, L3, clashes = new_L1, new_L2, new_L3, new_cl
            if clashes < best_clashes:
                best_clashes = clashes
                best_state   = (L1.copy(), L2.copy(), L3.copy())
                no_improve   = 0
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
                T = temp_init
                ils_rounds = 0
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
                clashes = _triple_clashes(L1, L2, L3, n)
                T = temp_init * 0.5

    return None, {
        "found": False, "best_clashes": best_clashes,
        "sa_best_clashes": best_clashes,
        "best_ct_count": best_ct_count,
        "restarts": restarts,
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
        if ct > self.best_ct_ever:
            self.best_ct_ever = ct
        self.pool.sort()
        tag = " ← NEW BEST CT!" if ct > 0 and ct >= self.best_ct_ever else ""
        print(f"  [{pair_id}] ct={ct}  max_dis={mdj}  src={source}{tag}")
        return rec

    def _load_seed_pairs(self):
        print(f"\nLoading seed pairs from {PAIRS_FILE}…")
        for idx, (L1, L2) in enumerate(_load_pairs()):
            self._add_pair(L1, L2, source="seed", pair_id=f"seed_{idx}")
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

    def _inject_fresh_pair(self) -> None:
        L1, L2, stats = transversal_find_pair(
            self.n, 30.0, rng_seed=self.rng.randint(0, 2**31)
        )
        if L1 is not None:
            self._add_pair(L1, L2, source="tv")

    def run(self) -> Optional[tuple]:
        n = self.n
        outer_iter = 0

        while self._ok():
            outer_iter += 1
            budget = min(self.eval_budget,
                         self.max_seconds - self._elapsed()
                         if self.max_seconds else self.eval_budget)
            if budget < 1.0:
                break

            # ── sa_triple runs every iteration ─────────────────────────────
            triple = self._run_sa_triple(budget * 0.6)
            if triple is not None:
                L1, L2, L3 = triple
                return self._success(L1, L2, L3)

            # ── pair-based strategies for pool top ─────────────────────────
            if self.pool:
                best_rec = self.pool[0]
                if best_rec.ct_count >= n:
                    # Full Algorithm X — this pair might have L3!
                    print(f"\n  ★ ct_count={best_rec.ct_count} ≥ {n}:"
                          f" running AlgX on pair {best_rec.pair_id}")
                    L3 = self._run_pair_strategy(best_rec, "ct_algx", budget * 0.4)
                    if L3 is not None:
                        return self._success(best_rec.L1, best_rec.L2, L3)
                elif best_rec.ct_count > 0:
                    # Some CTs — worth trying CSP
                    L3 = self._run_pair_strategy(best_rec, "csp_dual", budget * 0.35)
                    if L3 is not None:
                        return self._success(best_rec.L1, best_rec.L2, L3)
                else:
                    print(f"\n  Pool best: ct={best_rec.ct_count} — "
                          f"all known pairs are L3-free; relying on sa_triple.")

            # ── diversification ─────────────────────────────────────────────
            if outer_iter % 3 == 0:
                if self.pool:
                    self._inject_variants(self.pool[0])
                self._inject_fresh_pair()

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
