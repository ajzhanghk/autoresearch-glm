#!/usr/bin/env python3
"""
mols_search.py — Iterative search for 3 Mutually Orthogonal Latin Squares of order 10.

Background
----------
A Latin square of order n is an n×n array with symbols {0..n-1} where each
symbol appears exactly once per row and column.  Two squares L_A, L_B are
orthogonal if the n² ordered pairs (L_A[i,j], L_B[i,j]) are all distinct.
N(10) ≥ 2 (Parker 1960); whether N(10) ≥ 3 is an open combinatorial problem.

Algorithm overview
------------------
Phase 1 — find MOLS pair (L1, L2):
  Uses a COUPLED CSP: at every cell (i,j) an ordered pair (a,b) is assigned
  to (L1[i,j], L2[i,j]) simultaneously.  Bitmask constraint propagation covers
  both Latin constraints (L1 and L2) and the joint orthogonality constraint in
  a single pass, eliminating the chicken-and-egg problem of the sequential
  approach where most random L1 squares turn out to have no orthogonal mate.

Phase 2 — find third square L3:
  Given fixed (L1, L2), search for L3 orthogonal to both using CSP backtracking
  with bitmask domains and forward checking.

Pruning heuristics (both phases)
---------------------------------
*Bitmask domains*
    row1[i], col1[j] — 10-bit masks: bit k set iff symbol k free in L1
    row2[i], col2[j] — same for L2
    pair_orth[a]     — bit k set iff pair (a,k) unused in (L1, L2)

    Coupled domain of cell (i,j) = for each a ∈ (row1[i] & col1[j]):
        valid_b_for_a = (row2[i] & col2[j]) & pair_orth[a]

*Forward checking (FC)*
    After placing (a,b) at (i,j), scan every unset cell sharing the same row,
    column, or orthogonality index (pair_orth[a] affects all future cells
    assigned a-value a in L1).  Prune branch if any domain becomes empty.

*Minimum Remaining Values (MRV)*
    At each node branch on the unset cell with the fewest valid pairs.

Canonical form reduction
------------------------
L1: first row AND first col fixed to 0..n-1 (fully reduced).
L2: first row fixed to 0..n-1 (symbol normalisation).
L3: first row fixed to 0..n-1.
Two fully-reduced squares cannot be orthogonal (the diagonal pair (k,k) would
appear at both (0,k) and (k,0)), so only L1 is fully reduced.

Performance
-----------
All constraint state kept as Python ints (bitmasks) for O(1) update/query.
Numba @njit applied to small inner helpers when numba is installed.
Checkpoints written to --save-dir every CHECKPOINT_INTERVAL seconds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba JIT
# ---------------------------------------------------------------------------

NUMBA_AVAILABLE = False
try:
    from numba import njit as _njit          # type: ignore
    import numba                             # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    def _njit(fn=None, **kw):               # type: ignore
        return fn if fn is not None else (lambda f: f)

N              = 10
FULL_MASK      = (1 << N) - 1   # 0b1111111111 = 1023

CHECKPOINT_INTERVAL = 300       # seconds between progress saves

# ---------------------------------------------------------------------------
# Bit-manipulation helpers
#
# These are called from Python-level loops, so Python's native built-ins
# (bin().count / int.bit_length) are faster than Numba JIT functions called
# across the Python-C boundary.  Numba is kept as optional for completeness
# but the @_njit decoration is intentionally NOT applied here.
# ---------------------------------------------------------------------------

def _popcount(x: int) -> int:
    """Count set bits using Python's fast native string method."""
    return bin(x).count('1')


def _lsb_index(x: int) -> int:
    """Index of the lowest set bit (0-based) using Python's native bit_length."""
    return (x & -x).bit_length() - 1


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def is_latin_square(L: np.ndarray) -> bool:
    n = L.shape[0]
    expected = set(range(n))
    for i in range(n):
        if set(L[i, :].tolist()) != expected:
            return False
        if set(L[:, i].tolist()) != expected:
            return False
    return True


def are_orthogonal(L_A: np.ndarray, L_B: np.ndarray) -> bool:
    n = L_A.shape[0]
    seen = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            a, b = int(L_A[i, j]), int(L_B[i, j])
            if seen[a, b]:
                return False
            seen[a, b] = True
    return True


def verify_mols(squares: list[np.ndarray]) -> tuple[bool, dict]:
    """
    Verify that every square is a valid Latin square and every pair is orthogonal.
    Returns (all_valid, details_dict).
    """
    details: dict[str, bool] = {}
    for idx, L in enumerate(squares):
        details[f"L{idx+1}_is_latin"] = is_latin_square(L)
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            details[f"L{i+1}_orth_L{j+1}"] = are_orthogonal(squares[i], squares[j])
    return all(details.values()), details


# ---------------------------------------------------------------------------
# Phase 1: COUPLED constraint state for (L1, L2) joint search
# ---------------------------------------------------------------------------

class CoupledPairState:
    """
    Jointly tracks L1 and L2 for a paired CSP search.

    At every cell (i,j) the search assigns an ordered pair (a, b) to
    (L1[i,j], L2[i,j]) simultaneously, keeping both Latin constraints and the
    orthogonality constraint in sync via bitmasks.

    Bitmask invariants:
      row1[i], col1[j]  — bit k set iff symbol k free in row i / col j of L1
      row2[i], col2[j]  — same for L2
      pair_orth[a]      — bit k set iff the ordered pair (a,k) has not yet
                          appeared in (L1, L2)

    Domain of cell (i,j) = set of valid pairs (a,b):
        a ∈ row1[i] & col1[j]
        b ∈ (row2[i] & col2[j]) & pair_orth[a]

    Canonical form enforced during search (not pre-fixed):
      • Row 0: place(0, j, j, j) for all j — both first rows become 0..n-1.
      • Col 0 (i>0): domain restricted to a=i (L1's first col = i-th symbol)
        via the helper domain_count_col0 / pairs_col0.
    """

    __slots__ = ("n", "L1", "L2", "row1", "col1", "row2", "col2", "pair_orth")

    def __init__(self, n: int) -> None:
        self.n = n
        self.L1 = -np.ones((n, n), dtype=np.int8)
        self.L2 = -np.ones((n, n), dtype=np.int8)
        full = (1 << n) - 1
        self.row1      = [full] * n
        self.col1      = [full] * n
        self.row2      = [full] * n
        self.col2      = [full] * n
        self.pair_orth = [full] * n   # pair_orth[a] = bitmask of available b-partners

    # ── placement / undo ────────────────────────────────────────────────────

    def place(self, i: int, j: int, a: int, b: int) -> None:
        self.L1[i, j] = a
        self.L2[i, j] = b
        ba, bb = 1 << a, 1 << b
        self.row1[i] &= ~ba
        self.col1[j] &= ~ba
        self.row2[i] &= ~bb
        self.col2[j] &= ~bb
        self.pair_orth[a] &= ~bb

    def unplace(self, i: int, j: int, a: int, b: int) -> None:
        self.L1[i, j] = -1
        self.L2[i, j] = -1
        ba, bb = 1 << a, 1 << b
        self.row1[i] |= ba
        self.col1[j] |= ba
        self.row2[i] |= bb
        self.col2[j] |= bb
        self.pair_orth[a] |= bb

    # ── domain helpers ───────────────────────────────────────────────────────

    def _b_mask(self, i: int, j: int) -> int:
        """Bitmask of available b-values at (i,j) from Latin constraints."""
        return self.row2[i] & self.col2[j]

    def domain_count(self, i: int, j: int) -> int:
        """Count valid (a,b) pairs for a free interior cell."""
        A1 = self.row1[i] & self.col1[j]
        B_base = self._b_mask(i, j)
        count = 0
        mask_a = A1
        while mask_a:
            a = _lsb_index(mask_a)
            mask_a &= mask_a - 1
            count += _popcount(B_base & self.pair_orth[a])
        return count

    def domain_count_col0(self, i: int) -> int:
        """Count valid b-values for col-0 cell (i,0) where L1[i,0] is forced to i.

        Also checks that the forced L1 value 'i' is still available in row i
        and col 0 (another cell in row i might have already taken value i).
        """
        bit_i = 1 << i
        if not (self.row1[i] & bit_i and self.col1[0] & bit_i):
            return 0   # forced L1 value no longer available → infeasible
        return _popcount(self._b_mask(i, 0) & self.pair_orth[i])

    def pairs_for(self, i: int, j: int) -> list[tuple[int, int]]:
        """List of valid (a,b) pairs for free interior cell (i,j)."""
        A1 = self.row1[i] & self.col1[j]
        B_base = self._b_mask(i, j)
        out: list[tuple[int, int]] = []
        mask_a = A1
        while mask_a:
            a = _lsb_index(mask_a)
            mask_a &= mask_a - 1
            mask_b = B_base & self.pair_orth[a]
            while mask_b:
                b = _lsb_index(mask_b)
                mask_b &= mask_b - 1
                out.append((a, b))
        return out

    def b_mask_col0(self, i: int) -> int:
        """Bitmask of valid b-values for col-0 cell (i,0) [a=i is forced].
        Returns 0 if the forced a-value 'i' is no longer available.
        """
        bit_i = 1 << i
        if not (self.row1[i] & bit_i and self.col1[0] & bit_i):
            return 0
        return self._b_mask(i, 0) & self.pair_orth[i]


# ---------------------------------------------------------------------------
# Phase 1: coupled CSP solver
# ---------------------------------------------------------------------------

class _BacktrackCoupled:
    """
    MRV + FC + Unit Propagation backtracking for CoupledPairState.

    Cell types:
      • col-0 cells (i>0, j=0): L1[i,0] is forced to i (canonical form);
        only b is searched.  Domain = b_mask_col0(i).
      • interior cells (i>0, j>0): both a and b are searched.
        Domain = pairs_for(i,j).

    Pruning pipeline (applied in order at every node):
      1. MRV: branch on cell with smallest domain.
      2. Forward checking (FC): after placing, scan cells sharing row/col/pair_orth;
         prune if any domain becomes 0.
      3. Unit propagation (UP): after FC succeeds, find cells with domain == 1 and
         assign them eagerly; propagate transitively until quiescence.  Eliminates
         many forced choices from the explicit search tree and catches contradictions
         earlier than FC alone.

    Value-order randomisation: if rng is provided, pairs/b-values are shuffled
    before being tried at each node.
    """

    def __init__(self, state: CoupledPairState,
                 rng: Optional[random.Random] = None) -> None:
        self.state  = state
        self.rng    = rng
        self.nodes  = 0
        self.prunes = 0

    @staticmethod
    def _bits_of(mask: int) -> list[int]:
        """Extract bit indices from a bitmask (ascending order)."""
        result = []
        while mask:
            b = _lsb_index(mask)
            result.append(b)
            mask &= mask - 1
        return result

    def _unit_propagate(self, rest: list) -> Optional[list[tuple]]:
        """
        Eagerly assign all cells in *rest* that have exactly one valid option
        (singleton propagation / unit propagation).  Repeats until quiescence.

        Returns the list of (i, j, a, b) forced assignments (may be empty),
        or None if a contradiction (domain == 0) is encountered.  On None the
        state is fully restored to how it was before this call.
        """
        forced:     list[tuple] = []
        forced_set: set         = set()

        changed = True
        while changed:
            changed = False
            for (ri, rj) in rest:
                if (ri, rj) in forced_set:
                    continue
                # Compute domain size
                if rj == 0:
                    cnt = self.state.domain_count_col0(ri)
                else:
                    cnt = self.state.domain_count(ri, rj)

                if cnt == 0:                  # contradiction — undo and signal
                    for fi, fj, fa, fb in reversed(forced):
                        self.state.unplace(fi, fj, fa, fb)
                    return None

                if cnt == 1:                  # forced assignment
                    if rj == 0:
                        b = _lsb_index(self.state.b_mask_col0(ri))
                        a = ri
                    else:
                        pairs = self.state.pairs_for(ri, rj)
                        a, b  = pairs[0]
                    self.state.place(ri, rj, a, b)
                    forced.append((ri, rj, a, b))
                    forced_set.add((ri, rj))
                    changed = True
                    break        # restart scan with updated state

        return forced

    # col-0 cell identifier
    @staticmethod
    def _is_col0(j: int) -> bool:
        return j == 0

    def _domain_count(self, i: int, j: int) -> int:
        if j == 0:
            return self.state.domain_count_col0(i)
        return self.state.domain_count(i, j)

    def _forward_ok(self, rest: list[tuple[int,int]],
                    ci: int, cj: int, placed_a: int) -> bool:
        """Check no unset cell in rest has empty domain after placing at (ci,cj).

        Placing (a, b) at (ci, cj) modifies:
          row1[ci], col1[cj], row2[ci], col2[cj], pair_orth[a].
        Any unset cell sharing row, col, or whose domain involves pair_orth[a]
        may be affected.  We conservatively check all remaining cells; for n=10
        this is at most 90 domain_count calls — fast enough in Python.
        Additionally, col-0 cell (placed_a, 0) — whose forced a-value IS placed_a
        — must be checked whenever pair_orth[placed_a] changes (i.e., always
        after placing a=placed_a anywhere).
        """
        for (ri, rj) in rest:
            affected = (
                ri == ci          # same row of L1 and L2
                or rj == cj       # same col of L1 and L2
                or rj != 0        # any interior cell: pair_orth change may reduce domain
                or ri == placed_a # col-0 cell whose forced a equals placed_a
            )
            if affected and self._domain_count(ri, rj) == 0:
                return False
        return True

    def search(self, remaining: list[tuple[int,int]],
               max_time: Optional[float] = None) -> Optional[bool]:
        """
        Returns True (pair found), False (exhausted), None (timed out).
        """
        if not remaining:
            return True
        if max_time is not None and time.time() >= max_time:
            return None

        self.nodes += 1

        # MRV: pick cell with smallest domain
        best_idx   = 0
        best_count = self.state.n * self.state.n + 1
        for k, (i, j) in enumerate(remaining):
            c = self._domain_count(i, j)
            if c == 0:
                self.prunes += 1
                return False
            if c < best_count:
                best_count = c
                best_idx   = k

        remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
        ci, cj = remaining[0]
        rest   = remaining[1:]

        def _try_placement(a: int, b: int) -> Optional[bool]:
            """Place (a,b) at (ci,cj), run FC + UP, recurse. Undoes forced UP
            assignments after the recursive call.  Returns True/False/None."""
            self.state.place(ci, cj, a, b)
            if not self._forward_ok(rest, ci, cj, a):
                self.state.unplace(ci, cj, a, b)
                return False

            forced = self._unit_propagate(rest)
            if forced is None:                # UP found contradiction
                self.state.unplace(ci, cj, a, b)
                return False

            # Recurse with forced cells removed from the work list
            forced_set  = {(fi, fj) for fi, fj, _, _ in forced}
            filtered    = [c for c in rest if c not in forced_set]
            result      = self.search(filtered, max_time)

            if result is True:
                return True  # solution in current state — do NOT unplace

            # Undo UP forced assignments (reverse order)
            for fi, fj, fa, fb in reversed(forced):
                self.state.unplace(fi, fj, fa, fb)
            self.state.unplace(ci, cj, a, b)
            return result

        if cj == 0:
            # col-0 cell: a=ci is forced; search over b values
            b_values = self._bits_of(self.state.b_mask_col0(ci))
            if self.rng is not None:
                self.rng.shuffle(b_values)
            for b in b_values:
                result = _try_placement(ci, b)
                if result is True:
                    return True
                if result is None:
                    remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
                    return None
        else:
            # interior cell: search over all (a, b) pairs
            pairs = self.state.pairs_for(ci, cj)
            if self.rng is not None:
                self.rng.shuffle(pairs)
            for a, b in pairs:
                result = _try_placement(a, b)
                if result is True:
                    return True
                if result is None:
                    remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
                    return None

        remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
        return False


# ---------------------------------------------------------------------------
# Phase 2: constraint state for third square
# ---------------------------------------------------------------------------

class DualOrthState:
    """
    Mutable constraint state for filling L3 subject to:
      (a) L3 is a Latin square,
      (b) L3 is orthogonal to fixed L1,
      (c) L3 is orthogonal to fixed L2.

    Bitmask invariants:
      row_avail[i]  — bit k set iff k unused in row i of L3
      col_avail[j]  — bit k set iff k unused in col j of L3
      pair13[a]     — bit k set iff pair (a,k) not yet in (L1,L3)
      pair23[b]     — bit k set iff pair (b,k) not yet in (L2,L3)

    Domain of cell (i,j):
        row_avail[i] & col_avail[j] & pair13[L1[i,j]] & pair23[L2[i,j]]
    """

    __slots__ = ("n", "L1", "L2", "L3", "row_avail", "col_avail", "pair13", "pair23")

    def __init__(self, n: int, L1: np.ndarray, L2: np.ndarray) -> None:
        self.n   = n
        self.L1  = L1
        self.L2  = L2
        self.L3  = -np.ones((n, n), dtype=np.int8)
        full = (1 << n) - 1
        self.row_avail = [full] * n
        self.col_avail = [full] * n
        self.pair13    = [full] * n
        self.pair23    = [full] * n

    def domain(self, i: int, j: int) -> int:
        return (self.row_avail[i]
                & self.col_avail[j]
                & self.pair13[int(self.L1[i, j])]
                & self.pair23[int(self.L2[i, j])])

    def place(self, i: int, j: int, v: int) -> None:
        self.L3[i, j] = v
        bit = 1 << v
        self.row_avail[i]               &= ~bit
        self.col_avail[j]               &= ~bit
        self.pair13[int(self.L1[i, j])] &= ~bit
        self.pair23[int(self.L2[i, j])] &= ~bit

    def unplace(self, i: int, j: int, v: int) -> None:
        self.L3[i, j] = -1
        bit = 1 << v
        self.row_avail[i]               |= bit
        self.col_avail[j]               |= bit
        self.pair13[int(self.L1[i, j])] |= bit
        self.pair23[int(self.L2[i, j])] |= bit

    def fix_first_row(self) -> None:
        """Fix L3[0,j] = j for all j (symbol normalisation)."""
        for j in range(self.n):
            self.place(0, j, j)


# ---------------------------------------------------------------------------
# Phase 2: L3 CSP solver
# ---------------------------------------------------------------------------

class _BacktrackDual:
    """
    MRV + FC + UP backtracking for DualOrthState.

    Forward check: after placing at (ci,cj), scan all unset cells sharing
    any of: same row, same col, same L1-value (pair13 affected), same L2-value
    (pair23 affected).

    Unit propagation (UP): after FC succeeds, find all remaining cells with
    domain size 1 and assign them eagerly before recursing.
    """

    def __init__(self, state: DualOrthState, stats: dict,
                 t0: float, save_fn, max_seconds: Optional[float]) -> None:
        self.state       = state
        self.stats       = stats
        self.t0          = t0
        self.save_fn     = save_fn
        self.max_seconds = max_seconds
        self._last_save  = t0
        self._abort      = False

    def _forward_ok(self, rest: list, ci: int, cj: int) -> bool:
        l1v = int(self.state.L1[ci, cj])
        l2v = int(self.state.L2[ci, cj])
        st  = self.state
        for (ri, rj) in rest:
            if (ri == ci or rj == cj
                    or int(st.L1[ri, rj]) == l1v
                    or int(st.L2[ri, rj]) == l2v):
                if st.domain(ri, rj) == 0:
                    return False
        return True

    def _unit_propagate(self, rest: list) -> Optional[list]:
        """Assign all forced cells (domain == 1). Returns forced list or None on contradiction."""
        forced:     list  = []
        forced_set: set   = set()
        changed = True
        while changed:
            changed = False
            for (ri, rj) in rest:
                if (ri, rj) in forced_set:
                    continue
                m   = self.state.domain(ri, rj)
                cnt = _popcount(m)
                if cnt == 0:
                    for fi, fj, fv in reversed(forced):
                        self.state.unplace(fi, fj, fv)
                    return None
                if cnt == 1:
                    v = _lsb_index(m)
                    self.state.place(ri, rj, v)
                    forced.append((ri, rj, v))
                    forced_set.add((ri, rj))
                    self.stats["prune_domain"] += 0   # just propagation, not a prune
                    changed = True
                    break
        return forced

    def search(self, remaining: list) -> bool:
        if self._abort:
            return False
        if not remaining:
            return True

        self.stats["nodes"] += 1

        now = time.time()
        if now - self._last_save >= CHECKPOINT_INTERVAL:
            self.save_fn(self.state, self.stats)
            self._last_save = now
        if self.max_seconds and (now - self.t0) >= self.max_seconds:
            self._abort = True
            return False

        # MRV
        best_idx   = 0
        best_count = self.state.n + 2
        best_mask  = 0
        for k, (i, j) in enumerate(remaining):
            m = self.state.domain(i, j)
            c = _popcount(m)
            if c == 0:
                self.stats["prune_domain"] += 1
                return False
            if c < best_count:
                best_count = c
                best_idx   = k
                best_mask  = m

        remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
        ci, cj = remaining[0]
        rest   = remaining[1:]

        mask = best_mask
        while mask:
            v    = _lsb_index(mask)
            mask &= mask - 1
            self.state.place(ci, cj, v)
            if self._forward_ok(rest, ci, cj):
                forced = self._unit_propagate(rest)
                if forced is not None:
                    forced_set = {(fi, fj) for fi, fj, _ in forced}
                    filtered   = [c for c in rest if c not in forced_set]
                    if self.search(filtered):
                        return True
                    for fi, fj, fv in reversed(forced):
                        self.state.unplace(fi, fj, fv)
                else:
                    self.stats["prune_forward"] += 1
            else:
                self.stats["prune_forward"] += 1
            self.state.unplace(ci, cj, v)

        remaining[0], remaining[best_idx] = remaining[best_idx], remaining[0]
        return False


# ---------------------------------------------------------------------------
# Phase 1 driver: find a MOLS pair via coupled CSP
# ---------------------------------------------------------------------------

def find_mols_pair(
    n: int = 10,
    max_seconds: Optional[float] = None,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find two orthogonal Latin squares L1 (fully reduced), L2 (row-normalised).

    Uses a coupled CSP that assigns pairs (L1[i,j], L2[i,j]) simultaneously,
    giving bidirectional constraint propagation and much stronger pruning than
    the sequential (random L1 → search L2) approach.

    Canonical form:
      Row 0 of both squares: fixed to 0..n-1 via place(0,j,j,j).
      First col of L1:       enforced by restricting col-0 domains to a=i.

    Returns (L1, L2) or (None, None) if max_seconds is exceeded.
    """
    L2_ATTEMPT_SEC = 30.0     # per-restart budget; restart if exceeded

    rng = random.Random(rng_seed)
    t0  = time.time()
    attempt = 0

    while True:
        elapsed = time.time() - t0
        if max_seconds and elapsed >= max_seconds:
            return None, None

        attempt += 1
        print(f"  [pair] attempt {attempt}  elapsed={elapsed:.0f}s", flush=True)

        st = CoupledPairState(n)

        # Fix row 0 of both squares to 0..n-1
        for j in range(n):
            st.place(0, j, j, j)

        # Remaining cells to search:
        #   col-0 cells (i>0, j=0): assign (i, b)  — canonical form for L1
        #   interior cells (i>0, j>0): assign (a, b) freely
        cells = [(i, j) for i in range(1, n) for j in range(n)]
        rng.shuffle(cells)

        bt  = _BacktrackCoupled(st, rng=rng)
        deadline = time.time() + L2_ATTEMPT_SEC
        if max_seconds:
            deadline = min(deadline, t0 + max_seconds)

        result = bt.search(list(cells), max_time=deadline)

        if result is True:
            L1 = st.L1.copy()
            L2 = st.L2.copy()
            print(
                f"         pair found  nodes={bt.nodes:,}  prunes={bt.prunes:,}",
                flush=True,
            )
            return L1, L2
        elif result is False:
            print(
                f"         exhausted (no pair with this cell order)"
                f"  nodes={bt.nodes:,}",
                flush=True,
            )
        else:
            print(
                f"         timed out — shuffling and retrying"
                f"  nodes={bt.nodes:,}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Phase 2 driver: search for the third square
# ---------------------------------------------------------------------------

def search_for_third(
    L1: np.ndarray,
    L2: np.ndarray,
    save_dir: Path,
    max_seconds: Optional[float] = None,
    rng_seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Given two MOLS L1, L2, search for L3 orthogonal to both.

    L3's first row is fixed to 0..n-1 (symbol normalisation).
    Progress checkpointed every CHECKPOINT_INTERVAL seconds.

    Returns L3 if found; None if timed out or search exhausted.
    """
    n   = L1.shape[0]
    rng = random.Random(rng_seed)
    t0  = time.time()

    state = DualOrthState(n, L1, L2)
    state.fix_first_row()

    cells = [(i, j) for i in range(1, n) for j in range(n)]
    rng.shuffle(cells)

    stats: dict = {"nodes": 0, "prune_domain": 0, "prune_forward": 0}

    def save_checkpoint(st: DualOrthState, s: dict) -> None:
        elapsed = round(time.time() - t0, 1)
        cp = {
            "timestamp": datetime.now().isoformat(),
            "elapsed":   elapsed,
            "stats":     s,
            "L3_partial": st.L3.tolist(),
        }
        (save_dir / "checkpoint_l3.json").write_text(json.dumps(cp, indent=2))
        print(
            f"  [ckpt] elapsed={elapsed}s  nodes={s['nodes']:,}"
            f"  prune_domain={s['prune_domain']:,}"
            f"  prune_forward={s['prune_forward']:,}",
            flush=True,
        )

    bt    = _BacktrackDual(state, stats, t0, save_checkpoint, max_seconds)
    found = bt.search(list(cells))
    save_checkpoint(state, stats)

    return state.L3.copy() if found else None


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_pair(L1: np.ndarray, L2: np.ndarray, path: Path) -> None:
    path.write_text(json.dumps({"L1": L1.tolist(), "L2": L2.tolist()}, indent=2))


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(path.read_text())
    return np.array(d["L1"], dtype=np.int8), np.array(d["L2"], dtype=np.int8)


def save_triplet(L1, L2, L3, elapsed: float, path: Path) -> None:
    path.write_text(json.dumps(
        {"found": True, "elapsed": elapsed,
         "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist()},
        indent=2,
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for 3 Mutually Orthogonal Latin Squares of order 10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n",           type=int,   default=10)
    parser.add_argument("--save-dir",    default="mols_progress",
                        help="Directory for checkpoints and results")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Wall-clock time budget in seconds (unlimited if omitted)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="RNG seed for cell-order randomisation")
    parser.add_argument("--skip-pair",   action="store_true",
                        help="Load existing mols_pair.json instead of re-searching")
    args = parser.parse_args()

    n        = args.n
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print("MOLS Order-10 Search")
    print("=" * 50)
    print(f"n            : {n}")
    if NUMBA_AVAILABLE:
        print(f"Numba        : enabled ({numba.__version__})")
    else:
        print("Numba        : disabled — `pip install numba` for inner-loop speedup")
    print(f"Save dir     : {save_dir.resolve()}")
    print(f"Time budget  : {args.max_seconds or 'unlimited'} seconds")
    print(f"RNG seed     : {args.seed}")
    print()

    # ── Phase 1: obtain (L1, L2) ──────────────────────────────────────────
    pair_path = save_dir / "mols_pair.json"

    if pair_path.exists() and args.skip_pair:
        L1, L2 = load_pair(pair_path)
        print(f"Loaded MOLS pair from {pair_path}")
    else:
        print("Phase 1 — coupled CSP search for MOLS pair (L1, L2)…")
        t_pair = time.time()
        L1, L2 = find_mols_pair(n=n, max_seconds=args.max_seconds, rng_seed=args.seed)
        if L1 is None:
            print("ERROR: Could not find a MOLS pair within the time budget.")
            sys.exit(1)
        save_pair(L1, L2, pair_path)
        print(f"MOLS pair found in {time.time() - t_pair:.1f}s  → {pair_path}")

    ok, details = verify_mols([L1, L2])
    if not ok:
        print(f"ERROR: Pair verification failed: {details}")
        sys.exit(1)
    print(f"Pair verified: {details}")
    print()
    print("L1:")
    print(L1)
    print()
    print("L2:")
    print(L2)
    print()

    # ── Phase 2: search for L3 ────────────────────────────────────────────
    print("Phase 2 — searching for L3 orthogonal to both L1 and L2…")
    print(f"Canonical: L3[0, :] = 0..{n-1}  (first row fixed; first col free)")
    print()

    t3      = time.time()
    L3      = search_for_third(
        L1, L2, save_dir=save_dir,
        max_seconds=args.max_seconds, rng_seed=args.seed,
    )
    elapsed = time.time() - t3

    print()
    print("=" * 50)
    if L3 is not None:
        ok, details = verify_mols([L1, L2, L3])
        print(f"SUCCESS — 3 MOLS of order {n} found in {elapsed:.2f}s!")
        print(f"Verification: {details}")
        print()
        print("L3:")
        print(L3)
        result_path = save_dir / "mols_triplet.json"
        save_triplet(L1, L2, L3, elapsed, result_path)
        print(f"Triplet saved to {result_path}")
    else:
        if args.max_seconds:
            print(f"Time budget ({args.max_seconds}s) exhausted — no L3 found for this pair.")
        else:
            print(f"Search complete ({elapsed:.1f}s) — no L3 for this (L1, L2).")
        print()
        print("Tip: delete mols_progress/mols_pair.json and re-run with a different --seed")
        print("     to try a different pair.")


if __name__ == "__main__":
    main()
