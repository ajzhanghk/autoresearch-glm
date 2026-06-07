#!/usr/bin/env python3
"""
mols_adaptive.py — Autoresearch-style adaptive search for MOLS of order 10.

Three complementary search strategies:
  SAT — Encodes MOLS pair as CNF and hands to Glucose4 CDCL SAT solver.
        CDCL's conflict-driven clause learning vastly outperforms hand-rolled
        backtracking for CSPs.  Canonical form constraints are unit clauses.
  SA  — Simulated annealing on (L1, L2) space, minimising orthogonality
        clash count.  Clashes = 0 ↔ valid MOLS pair.  SA escapes local minima
        via Metropolis acceptance; intercalate flips make fine-grained moves.
  CSP — The backtracking coupled-CSP from mols_search.py (Luby restarts).

Autoresearch outer loop (mirrors the GLM experiment loop):
  1. Maintain a portfolio of (strategy, params) configurations.
  2. Evaluate each for EVAL_BUDGET seconds; score: found > low_clashes > high_clashes.
  3. Mutate the best config (one parameter at a time), add to portfolio.
  4. Drop configs that consistently underperform the frontier.
  5. Log every trial to mols_adaptive_results.tsv.
  6. When pair found, hand off to Phase 2 (search for L3).

Usage
-----
    python mols_adaptive.py --n 10 --save-dir mols_adaptive_run
    python mols_adaptive.py --n 10 --strategy sat
    python mols_adaptive.py --n 10 --strategy sa
    python mols_adaptive.py --n 10 --skip-pair   # jump straight to Phase 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reuse Phase-2 solver from mols_search.py
# ---------------------------------------------------------------------------
from mols_search import (
    verify_mols,
    save_pair,
    load_pair,
    save_triplet,
    search_for_third,
)

# ---------------------------------------------------------------------------
# Latin square utilities
# ---------------------------------------------------------------------------

def random_latin_square(n: int, rng: random.Random) -> np.ndarray:
    """Random Latin square via cyclic base + random row/col/symbol permutations."""
    L = np.array([[(i + j) % n for j in range(n)] for i in range(n)], dtype=np.int8)
    rows = list(range(n)); rng.shuffle(rows)
    L = L[rows]
    cols = list(range(n)); rng.shuffle(cols)
    L = L[:, cols]
    perm = list(range(n)); rng.shuffle(perm)
    out = np.empty_like(L)
    for old, new_sym in enumerate(perm):
        out[L == old] = new_sym
    return out


def count_clashes(L1: np.ndarray, L2: np.ndarray, n: int) -> int:
    """Number of repeated (L1[i,j], L2[i,j]) pairs — 0 iff orthogonal."""
    counts = np.zeros(n * n, dtype=np.int32)
    idx = L1.ravel().astype(np.int32) * n + L2.ravel().astype(np.int32)
    np.add.at(counts, idx, 1)
    return int(np.sum(np.maximum(0, counts - 1)))


def canonicalize_pair(L1: np.ndarray, L2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put (L1, L2) into the canonical form expected by search_for_third:
      L1: fully reduced — row 0 = col 0 = 0..n-1
      L2: row-normalised — row 0 = 0..n-1
    Row / column / symbol permutations preserve orthogonality.
    """
    n = L1.shape[0]
    L1 = L1.copy().astype(np.int8)
    L2 = L2.copy().astype(np.int8)

    # 1. Relabel L1 symbols so row 0 = 0..n-1
    pi1 = np.empty(n, dtype=np.int8)
    for j in range(n):
        pi1[int(L1[0, j])] = j
    L1 = pi1[L1]

    # 2. Permute rows so col 0 of L1 = 0..n-1
    row_order = np.empty(n, dtype=int)
    for i in range(n):
        row_order[int(L1[i, 0])] = i
    L1 = L1[row_order]
    L2 = L2[row_order]

    # 3. Relabel L2 symbols so row 0 = 0..n-1
    pi2 = np.empty(n, dtype=np.int8)
    for j in range(n):
        pi2[int(L2[0, j])] = j
    L2 = pi2[L2]

    return L1, L2


# ---------------------------------------------------------------------------
# SA move helpers — every move keeps both squares valid Latin squares
# ---------------------------------------------------------------------------

def _row_swap(L: np.ndarray, r1: int, r2: int) -> np.ndarray:
    L = L.copy(); L[[r1, r2]] = L[[r2, r1]]; return L

def _col_swap(L: np.ndarray, c1: int, c2: int) -> np.ndarray:
    L = L.copy(); L[:, [c1, c2]] = L[:, [c2, c1]]; return L

def _relabel(L: np.ndarray, a: int, b: int) -> np.ndarray:
    L = L.copy()
    ma, mb = (L == a), (L == b)
    L[ma] = b; L[mb] = a
    return L

def _intercalate_flip(L: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> Optional[np.ndarray]:
    """Flip 2×2 intercalate at (r1,r2)×(c1,c2) if it forms a valid intercalate."""
    a, b = int(L[r1, c1]), int(L[r1, c2])
    c, d = int(L[r2, c1]), int(L[r2, c2])
    if a == d and b == c and a != b:
        L = L.copy()
        L[r1, c1], L[r1, c2] = b, a
        L[r2, c1], L[r2, c2] = a, b
        return L
    return None


# ---------------------------------------------------------------------------
# Strategy 1: Simulated annealing
# ---------------------------------------------------------------------------

@dataclass
class SAConfig:
    strategy:      str   = "sa"
    temp_init:     float = 5.0
    cooling:       float = 0.9999990
    restart_after: int   = 100_000
    row_w:         float = 0.30
    col_w:         float = 0.30
    relabel_w:     float = 0.20
    intercalate_w: float = 0.20


def sa_find_pair(
    n: int,
    max_seconds: float,
    cfg: SAConfig,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Minimise count_clashes(L1, L2) via simulated annealing.
    Moves: row-swap, col-swap, global symbol relabel, intercalate flip.
    All moves keep both L1 and L2 as valid Latin squares.
    Returns (L1_canonical, L2_canonical, stats) on success, else (None, None, stats).
    """
    rng = random.Random(rng_seed)
    t0  = time.time()

    total_w   = cfg.row_w + cfg.col_w + cfg.relabel_w + cfg.intercalate_w
    move_ws   = [cfg.row_w / total_w, cfg.col_w / total_w,
                 cfg.relabel_w / total_w, cfg.intercalate_w / total_w]
    move_names = ["row", "col", "relabel", "intercalate"]

    best_clashes = n * n
    total_steps  = 0
    restarts     = 0

    def fresh_start() -> tuple:
        L1_ = random_latin_square(n, random.Random(rng.random()))
        L2_ = random_latin_square(n, random.Random(rng.random()))
        return L1_, L2_, count_clashes(L1_, L2_, n), cfg.temp_init

    L1, L2, clashes, T = fresh_start()
    no_improve = 0

    while True:
        if time.time() - t0 >= max_seconds:
            return None, None, {
                "steps": total_steps, "restarts": restarts,
                "best_clashes": best_clashes, "elapsed": round(time.time()-t0, 1)
            }

        if clashes == 0:
            L1c, L2c = canonicalize_pair(L1, L2)
            return L1c, L2c, {
                "steps": total_steps, "restarts": restarts,
                "best_clashes": 0, "elapsed": round(time.time()-t0, 1)
            }

        move = rng.choices(move_names, weights=move_ws)[0]
        target = rng.randint(1, 2)
        L = L1 if target == 1 else L2

        if move == "row":
            r1, r2 = rng.sample(range(n), 2)
            L_new = _row_swap(L, r1, r2)
        elif move == "col":
            c1, c2 = rng.sample(range(n), 2)
            L_new = _col_swap(L, c1, c2)
        elif move == "relabel":
            a, b = rng.sample(range(n), 2)
            L_new = _relabel(L, a, b)
        else:
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            L_new = _intercalate_flip(L, r1, r2, c1, c2)
            if L_new is None:
                total_steps += 1
                continue

        new_clashes = (count_clashes(L_new, L2, n) if target == 1
                       else count_clashes(L1, L_new, n))
        delta = new_clashes - clashes

        if delta <= 0 or (T > 1e-12 and rng.random() < math.exp(-delta / T)):
            if target == 1: L1 = L_new
            else:           L2 = L_new
            clashes = new_clashes
            if clashes < best_clashes:
                best_clashes = clashes
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= cfg.cooling
        total_steps += 1

        if no_improve >= cfg.restart_after:
            restarts += 1
            L1, L2, clashes, T = fresh_start()
            no_improve = 0


# ---------------------------------------------------------------------------
# Strategy 2: SAT encoding → Glucose4 CDCL solver
# ---------------------------------------------------------------------------

@dataclass
class SATConfig:
    strategy:     str  = "sat"
    canonical:    bool = True   # add canonical-form unit clauses


def _build_mols_cnf(n: int, canonical: bool = True):
    """Build a CNF for the MOLS pair problem.

    Variables (1-indexed for PySAT):
      x1(i,j,k) = n*n*(i) + n*(j) + k + 1           for L1[i,j]=k
      x2(i,j,k) = n*n*n + n*n*(i) + n*(j) + k + 1   for L2[i,j]=k
    Total: 2*n^3 variables.

    Clauses:
      - ALO / AMO per cell (Latin value per cell is unique)
      - Row / column exactly-once for each symbol
      - Orthogonality: for each (a,b), at most one cell (i,j) with L1[i,j]=a, L2[i,j]=b
      - (Optional) canonical-form unit clauses
    """
    N3 = n * n * n

    def v1(i, j, k): return n * n * i + n * j + k + 1
    def v2(i, j, k): return N3 + n * n * i + n * j + k + 1

    clauses = []

    def _amo_pairwise(lits):
        for p in range(len(lits)):
            for q in range(p + 1, len(lits)):
                clauses.append([-lits[p], -lits[q]])

    # ── Canonical form unit clauses ──────────────────────────────────────
    if canonical:
        for j in range(n):
            clauses.append([v1(0, j, j)])   # L1[0,j] = j
            clauses.append([v2(0, j, j)])   # L2[0,j] = j
        for i in range(1, n):
            clauses.append([v1(i, 0, i)])   # L1[i,0] = i  (fully reduced)

    # ── Latin constraints for L1 and L2 ─────────────────────────────────
    for sq, vf in [(1, v1), (2, v2)]:
        for i in range(n):
            for j in range(n):
                # ALO per cell
                clauses.append([vf(i, j, k) for k in range(n)])
                # AMO per cell
                _amo_pairwise([vf(i, j, k) for k in range(n)])
            for k in range(n):
                # Row i: symbol k appears in exactly one column
                clauses.append([vf(i, j, k) for j in range(n)])
                _amo_pairwise([vf(i, j, k) for j in range(n)])
        for j in range(n):
            for k in range(n):
                # Col j: symbol k appears in exactly one row
                clauses.append([vf(i, j, k) for i in range(n)])
                _amo_pairwise([vf(i, j, k) for i in range(n)])

    # ── Orthogonality: ∀(a,b) pair appears at most once ─────────────────
    cells = [(i, j) for i in range(n) for j in range(n)]
    for a in range(n):
        for b in range(n):
            for idx1 in range(len(cells)):
                i, j = cells[idx1]
                for idx2 in range(idx1 + 1, len(cells)):
                    i2, j2 = cells[idx2]
                    clauses.append([-v1(i, j, a), -v2(i, j, b),
                                    -v1(i2, j2, a), -v2(i2, j2, b)])

    n_vars = 2 * N3
    return clauses, n_vars, v1, v2


def sat_find_pair(
    n: int,
    max_seconds: float,
    cfg: SATConfig,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Solve the MOLS pair problem via Glucose4 CDCL SAT solver.

    Interrupts the solver in a background thread after max_seconds.
    Returns (L1, L2, stats) on success, (None, None, stats) on timeout / UNSAT.
    """
    try:
        from pysat.solvers import Glucose4
    except ImportError:
        print("  [SAT] pysat not installed — pip install python-sat", flush=True)
        return None, None, {"elapsed": 0, "reason": "no_pysat"}

    t0 = time.time()
    clauses, n_vars, v1, v2 = _build_mols_cnf(n, canonical=cfg.canonical)
    build_t = time.time() - t0

    print(f"  [SAT] CNF: {n_vars} vars, {len(clauses):,} clauses  "
          f"(build={build_t:.1f}s)", flush=True)

    solver = Glucose4(bootstrap_with=clauses)

    # Interrupt from a timer thread
    interrupted = threading.Event()
    def _timer():
        remaining = max_seconds - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)
        solver.interrupt()
        interrupted.set()

    t_thread = threading.Thread(target=_timer, daemon=True)
    t_thread.start()

    result = solver.solve_limited(expect_interrupt=True)
    elapsed = time.time() - t0

    if result is True:
        model = solver.get_model()
        L1 = np.zeros((n, n), dtype=np.int8)
        L2 = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if model[v1(i, j, k) - 1] > 0:
                        L1[i, j] = k
                    if model[v2(i, j, k) - 1] > 0:
                        L2[i, j] = k
        solver.delete()
        return L1, L2, {"elapsed": round(elapsed, 1), "reason": "found"}

    solver.delete()
    reason = "timeout" if interrupted.is_set() else "unsat"
    return None, None, {"elapsed": round(elapsed, 1), "reason": reason}


# ---------------------------------------------------------------------------
# Strategy 3: CSP with Luby restarts (wrapper around mols_search.py)
# ---------------------------------------------------------------------------

@dataclass
class CSPConfig:
    strategy:  str   = "csp"
    luby_base: float = 10.0    # base unit for Luby restart sequence (seconds)


def _luby(k: int) -> int:
    """k-th term (0-indexed) of the Luby restart sequence."""
    if k == 0:
        return 1
    t = 1
    while t < k + 1:
        t <<= 1
    t >>= 1
    return t >> 1 if k + 1 == t else _luby(k - t + 1)


def csp_find_pair(
    n: int,
    max_seconds: float,
    cfg: CSPConfig,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Coupled CSP with Luby restart schedule from mols_search.py."""
    import mols_search as ms

    rng = random.Random(rng_seed)
    t0  = time.time()
    attempt = 0
    total_nodes = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            return None, None, {"nodes": total_nodes, "attempts": attempt,
                                 "elapsed": round(elapsed, 1), "best_clashes": n*n}

        luby_t   = _luby(attempt) * cfg.luby_base
        deadline = min(time.time() + luby_t,
                       t0 + max_seconds)

        st = ms.CoupledPairState(n)
        for j in range(n):
            st.place(0, j, j, j)

        cells = [(i, j) for i in range(1, n) for j in range(n)]
        rng.shuffle(cells)

        bt = ms._BacktrackCoupled(st, rng=rng)
        result = bt.search(list(cells), max_time=deadline)
        total_nodes += bt.nodes
        attempt += 1

        if result is True:
            return (st.L1.copy(), st.L2.copy(),
                    {"nodes": total_nodes, "attempts": attempt,
                     "elapsed": round(time.time()-t0, 1), "best_clashes": 0})


# ---------------------------------------------------------------------------
# Adaptive autoresearch outer loop
# ---------------------------------------------------------------------------

TSV_COLS = ["trial", "timestamp", "strategy", "params_json",
            "best_clashes", "metric", "elapsed_s", "found"]

MUTATIONS: dict[str, list] = {
    "temp_init":      [1.0, 2.0, 5.0, 8.0, 12.0],
    "cooling":        [0.9999980, 0.9999990, 0.9999995, 0.9999999],
    "restart_after":  [50_000, 100_000, 200_000, 500_000],
    "row_w":          [0.15, 0.25, 0.35, 0.45],
    "col_w":          [0.15, 0.25, 0.35, 0.45],
    "relabel_w":      [0.10, 0.20, 0.30],
    "intercalate_w":  [0.10, 0.20, 0.30, 0.40],
    "luby_base":      [5.0, 10.0, 20.0, 30.0],
    "canonical":      [True, False],
}


def _mutate(cfg, rng: random.Random):
    """Return a copy with one randomly chosen parameter changed."""
    d   = asdict(cfg)
    cls = type(cfg)
    opts = [k for k in d if k in MUTATIONS and k != "strategy"]
    if not opts:
        return cfg
    key = rng.choice(opts)
    choices = [v for v in MUTATIONS[key] if v != d[key]]
    if not choices:
        return cfg
    d[key] = rng.choice(choices)
    return cls(**d)


class AdaptiveSearch:
    """
    Autoresearch outer loop for MOLS pair finding.

    Portfolio: list of (config, frontier_score) tuples.
    Score metric: clashes achieved after eval_budget seconds
    (lower is better; 0 = success and terminates immediately).
    After each trial, mutate the best-scoring config and add to queue.
    Always keep at least one SAT, one SA, and one CSP config in the pool.
    """

    def __init__(self, n: int, save_dir: Path, eval_budget: float = 60.0,
                 master_seed: int = 0, max_seconds: Optional[float] = None) -> None:
        self.n            = n
        self.save_dir     = save_dir
        self.eval_budget  = eval_budget
        self.rng          = random.Random(master_seed)
        self.max_seconds  = max_seconds
        self.t0           = time.time()
        self.trial        = 0

        # Frontier: best clashes per strategy
        self.frontier: dict[str, int] = {"sat": n*n, "sa": n*n, "csp": n*n}
        self.best_cfg: dict[str, object] = {
            "sat": SATConfig(), "sa": SAConfig(), "csp": CSPConfig()
        }

        # Initial queue: SAT first (usually fastest), then SA, then CSP
        self.queue: list = [SATConfig(), SAConfig(), CSPConfig()]

        save_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = save_dir / "mols_adaptive_results.tsv"
        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", newline="") as f:
                csv.writer(f, delimiter="\t").writerow(TSV_COLS)

    def _elapsed(self) -> float:
        return time.time() - self.t0

    def _budget_ok(self) -> bool:
        return self.max_seconds is None or self._elapsed() < self.max_seconds

    def _remaining(self) -> float:
        if self.max_seconds is None:
            return float("inf")
        return max(0.0, self.max_seconds - self._elapsed())

    def _run(self, cfg) -> dict:
        self.trial += 1
        seed  = self.rng.randint(0, 2**31)
        budget = min(self.eval_budget, self._remaining())
        params = asdict(cfg)

        print(f"\n── Trial {self.trial}"
              f"  strategy={cfg.strategy}"
              f"  budget={budget:.0f}s ──────────────────────")
        print(f"   params: {params}", flush=True)

        t_trial = time.time()

        if cfg.strategy == "sat":
            L1, L2, stats = sat_find_pair(self.n, budget, cfg, rng_seed=seed)
        elif cfg.strategy == "sa":
            L1, L2, stats = sa_find_pair(self.n, budget, cfg, rng_seed=seed)
        else:
            L1, L2, stats = csp_find_pair(self.n, budget, cfg, rng_seed=seed)

        elapsed  = time.time() - t_trial
        found    = L1 is not None
        clashes  = 0 if found else stats.get("best_clashes", self.n * self.n)
        metric   = stats.get("steps", stats.get("nodes", stats.get("elapsed", 0)))

        print(f"   result: {'FOUND' if found else f'best_clashes={clashes}'}"
              f"  elapsed={elapsed:.1f}s", flush=True)

        row = [self.trial, datetime.now().isoformat(timespec="seconds"),
               cfg.strategy, json.dumps(params), clashes, metric,
               round(elapsed, 1), int(found)]
        with open(self.tsv_path, "a", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(row)

        return {"L1": L1, "L2": L2, "clashes": clashes, "found": found, "cfg": cfg}

    def run(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print("=" * 60)
        print(f"Adaptive MOLS-{self.n} search")
        print(f"eval_budget={self.eval_budget}s  results={self.tsv_path}")
        print("=" * 60, flush=True)

        while self._budget_ok():
            if not self.queue:
                # Replenish: mutate all frontiers
                for s in ["sat", "sa", "csp"]:
                    self.queue.append(_mutate(self.best_cfg[s], self.rng))

            cfg = self.queue.pop(0)
            res = self._run(cfg)

            if res["found"]:
                print("\n" + "=" * 60)
                print("MOLS PAIR FOUND!")
                print("=" * 60)
                return res["L1"], res["L2"]

            strat = cfg.strategy
            if res["clashes"] < self.frontier[strat]:
                self.frontier[strat] = res["clashes"]
                self.best_cfg[strat] = cfg
                print(f"   [{strat} frontier] clashes={self.frontier[strat]}")
                self.queue.append(_mutate(cfg, self.rng))
            else:
                self.queue.append(_mutate(self.best_cfg[strat], self.rng))

            # Ensure diversity: keep all three strategies represented
            present = {c.strategy for c in self.queue}
            for s, cls in [("sat", SATConfig), ("sa", SAConfig), ("csp", CSPConfig)]:
                if s not in present:
                    self.queue.append(_mutate(self.best_cfg[s], self.rng))

        print("\nTime budget exhausted.")
        return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Adaptive MOLS search (SAT + SA + CSP autoresearch loop)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n",           type=int,   default=10)
    p.add_argument("--save-dir",    default="mols_adaptive_run")
    p.add_argument("--max-hours",   type=float, default=0,
                   help="Wall-clock budget in hours (0 = unlimited)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--eval-budget", type=float, default=120.0,
                   help="Seconds per config trial")
    p.add_argument("--skip-pair",   action="store_true",
                   help="Load existing mols_pair.json and jump to Phase 2")
    p.add_argument("--strategy",
                   choices=["adaptive", "sat", "sa", "csp"],
                   default="adaptive")
    args = p.parse_args()

    n        = args.n
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    max_sec  = args.max_hours * 3600 if args.max_hours > 0 else None

    print("Adaptive MOLS Search")
    print("=" * 60)
    print(f"n            : {n}")
    print(f"strategy     : {args.strategy}")
    print(f"eval budget  : {args.eval_budget}s per trial")
    print(f"save dir     : {save_dir.resolve()}")
    print(f"time budget  : {'unlimited' if not max_sec else f'{args.max_hours}h'}")
    print(f"master seed  : {args.seed}")
    print()

    pair_path = save_dir / "mols_pair.json"

    # ── Phase 1: find (L1, L2) ───────────────────────────────────────────
    if pair_path.exists() and args.skip_pair:
        L1, L2 = load_pair(pair_path)
        print(f"Loaded pair from {pair_path}")
    else:
        if args.strategy == "sat":
            cfg = SATConfig()
            L1, L2, stats = sat_find_pair(n, max_sec or 1e9, cfg, rng_seed=args.seed)
            if L1 is None:
                print(f"SAT: {stats}")
        elif args.strategy == "sa":
            cfg = SAConfig()
            L1, L2, stats = sa_find_pair(n, max_sec or 1e9, cfg, rng_seed=args.seed)
        elif args.strategy == "csp":
            cfg = CSPConfig()
            L1, L2, stats = csp_find_pair(n, max_sec or 1e9, cfg, rng_seed=args.seed)
        else:
            searcher = AdaptiveSearch(
                n=n, save_dir=save_dir,
                eval_budget=args.eval_budget,
                master_seed=args.seed,
                max_seconds=max_sec,
            )
            L1, L2 = searcher.run()

        if L1 is None:
            print("ERROR: Could not find a MOLS pair within the time budget.")
            sys.exit(1)

        save_pair(L1, L2, pair_path)
        print(f"\nMOLS pair saved → {pair_path}")

    ok, details = verify_mols([L1, L2])
    if not ok:
        print(f"ERROR: pair verification failed: {details}")
        sys.exit(1)
    print(f"Pair verified: {details}\n")
    print(f"L1:\n{L1}\nL2:\n{L2}\n")

    # ── Phase 2: find L3 ─────────────────────────────────────────────────
    print("Phase 2 — searching for L3 orthogonal to both L1 and L2…")
    print(f"Canonical: L3[0,:] = 0..{n-1}  (first row fixed)\n")

    t2 = time.time()
    L3 = search_for_third(L1, L2, save_dir,
                          max_seconds=max_sec, rng_seed=args.seed)

    if L3 is None:
        print("Phase 2 timed out — L3 not found in budget.")
        sys.exit(1)

    elapsed = time.time() - t2
    ok, details = verify_mols([L1, L2, L3])
    if ok:
        print("=" * 60)
        print(f"SUCCESS — 3 MOLS of order {n} found in {elapsed:.1f}s!")
        print(f"Verification: {details}\n")
        print(f"L3:\n{L3}")
        trip_path = save_dir / "mols_triplet.json"
        save_triplet(L1, L2, L3, elapsed, trip_path)
        print(f"Triplet saved → {trip_path}")
    else:
        print(f"ERROR: Final verification failed: {details}")
        sys.exit(1)


if __name__ == "__main__":
    main()
