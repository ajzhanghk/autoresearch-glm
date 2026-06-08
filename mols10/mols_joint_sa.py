#!/usr/bin/env python3
"""
mols_joint_sa.py — Joint simulated annealing over all three LS simultaneously.

Key insight: fixing (L1, L2) constrains the search to a manifold where all 316
tested pairs are SAT-UNSAT certified. By letting ALL three LS move, we explore
the full joint space where cl12 > 0 temporarily, potentially finding paths to
E = cl12 + cl13 + cl23 = 0 (3-MOLS of order 10).

Usage:
    python mols_joint_sa.py --seed 2023 --budget 180
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
TRIPLE_MISS_FILE = RESULTS_DIR / "near_miss_triple.json"
FOUND_FILE       = RESULTS_DIR / "MOLS10_FOUND.json"
LOG_FILE         = RESULTS_DIR / "l3_joint_sa.log"
N = 10


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def count_clashes(A: np.ndarray, B: np.ndarray, n: int = N) -> int:
    """Number of (a,b) symbol pairs missing from superposition A,B."""
    pairs = A.ravel().astype(np.int32) * n + B.ravel().astype(np.int32)
    return int(n * n - np.count_nonzero(np.bincount(pairs, minlength=n * n)))


def is_valid_ls(L: np.ndarray, n: int = N) -> bool:
    expected = set(range(n))
    for i in range(n):
        if set(L[i].tolist()) != expected or set(L[:, i].tolist()) != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Pool I/O
# ---------------------------------------------------------------------------

def load_pool() -> list[dict]:
    try:
        return json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
    except (json.JSONDecodeError, OSError):
        return []


def pool_entry_to_matrices(entry: dict, n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def to_arr(key):
        return np.array(entry[key], dtype=np.int8).reshape(n, n)
    return to_arr("L1"), to_arr("L2"), to_arr("L3")


def save_found(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray, seed: int):
    result = {
        "found": True,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "L1": L1.tolist(),
        "L2": L2.tolist(),
        "L3": L3.tolist(),
        "cl12": count_clashes(L1, L2),
        "cl13": count_clashes(L1, L3),
        "cl23": count_clashes(L2, L3),
    }
    FOUND_FILE.write_text(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Move helpers (all preserve LS validity)
# ---------------------------------------------------------------------------

def apply_row_swap(L: np.ndarray, r: int, s: int) -> np.ndarray:
    Ln = L.copy(); Ln[[r, s]] = Ln[[s, r]]; return Ln


def apply_col_swap(L: np.ndarray, c: int, d: int) -> np.ndarray:
    Ln = L.copy(); Ln[:, [c, d]] = Ln[:, [d, c]]; return Ln


def apply_relabel(L: np.ndarray, a: int, b: int) -> np.ndarray:
    Ln = L.copy()
    Ln[L == a] = b; Ln[L == b] = a; return Ln


def apply_intercalate(L: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> np.ndarray | None:
    """Swap the 2×2 intercalate block if structurally valid."""
    a, b = int(L[r1, c1]), int(L[r1, c2])
    if a == b or int(L[r2, c1]) != b or int(L[r2, c2]) != a:
        return None
    Ln = L.copy()
    Ln[r1, c1], Ln[r1, c2] = b, a
    Ln[r2, c1], Ln[r2, c2] = a, b
    return Ln


def random_valid_row(L: np.ndarray, row_idx: int, n: int, rng: random.Random) -> np.ndarray | None:
    """SDR-based random replacement for one row of L."""
    available = []
    for j in range(n):
        used = {int(L[i, j]) for i in range(n) if i != row_idx}
        available.append([s for s in range(n) if s not in used])
    cols = list(range(n)); rng.shuffle(cols)
    new_row = np.zeros(n, dtype=np.int8)
    used_syms: set[int] = set()
    for j in cols:
        choices = [s for s in available[j] if s not in used_syms]
        if not choices: return None
        s = rng.choice(choices); new_row[j] = s; used_syms.add(s)
    return new_row


# ---------------------------------------------------------------------------
# Joint SA over (L1, L2, L3)
# ---------------------------------------------------------------------------

def joint_sa(L1_init: np.ndarray, L2_init: np.ndarray, L3_init: np.ndarray,
             n: int, budget_s: float, rng: random.Random,
             T_init: float = 15.0, T_min: float = 0.01,
             log_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SA over all three LS. Returns (best_L1, best_L2, best_L3, best_E)."""

    L1 = L1_init.copy(); L2 = L2_init.copy(); L3 = L3_init.copy()
    cl12 = count_clashes(L1, L2, n)
    cl13 = count_clashes(L1, L3, n)
    cl23 = count_clashes(L2, L3, n)
    E = cl12 + cl13 + cl23

    best_E = E
    best_L1, best_L2, best_L3 = L1.copy(), L2.copy(), L3.copy()

    t0 = time.time()
    n_steps = 0

    # Move type list with weights
    # (ls_idx, move_name)
    MOVES = [
        (0, "row_swap"), (0, "col_swap"), (0, "relabel"), (0, "intercalate"),
        (1, "row_swap"), (1, "col_swap"), (1, "relabel"), (1, "intercalate"),
        (2, "row_swap"), (2, "col_swap"), (2, "relabel"), (2, "intercalate"),
        (2, "row_replace"),  # SDR-based large jump, only for L3
    ]
    # Bias toward L3 moves (easier search axis) and larger moves
    WEIGHTS = [2, 2, 3, 2,  2, 2, 3, 2,  3, 3, 4, 3, 5]

    elapsed = 0.0
    while elapsed < budget_s:
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / budget_s)
        T = T_init * (T_min / T_init) ** frac

        ls_idx, mv = rng.choices(MOVES, weights=WEIGHTS, k=1)[0]
        LS = [L1, L2, L3][ls_idx]

        # Generate proposed new LS
        Ln = None
        if mv == "row_swap":
            r, s = rng.sample(range(n), 2)
            Ln = apply_row_swap(LS, r, s)
        elif mv == "col_swap":
            c, d = rng.sample(range(n), 2)
            Ln = apply_col_swap(LS, c, d)
        elif mv == "relabel":
            a, b = rng.sample(range(n), 2)
            Ln = apply_relabel(LS, a, b)
        elif mv == "intercalate":
            r1, r2 = rng.sample(range(n), 2)
            c1, c2 = rng.sample(range(n), 2)
            Ln = apply_intercalate(LS, r1, r2, c1, c2)
            if Ln is None:
                continue
        elif mv == "row_replace":
            ri = rng.randint(0, n - 1)
            Ln = apply_row_swap(LS, ri, ri)  # placeholder
            nr = random_valid_row(L3, ri, n, rng)
            if nr is None:
                continue
            Ln = L3.copy(); Ln[ri] = nr

        if Ln is None:
            continue

        # Compute new clashes — only recompute the two pairs affected by ls_idx
        if ls_idx == 0:   # L1 changes → cl12 and cl13 change
            cl12n = count_clashes(Ln, L2, n)
            cl13n = count_clashes(Ln, L3, n)
            cl23n = cl23
        elif ls_idx == 1:  # L2 changes → cl12 and cl23 change
            cl12n = count_clashes(L1, Ln, n)
            cl13n = cl13
            cl23n = count_clashes(Ln, L3, n)
        else:              # L3 changes → cl13 and cl23 change
            cl12n = cl12
            cl13n = count_clashes(L1, Ln, n)
            cl23n = count_clashes(L2, Ln, n)

        En = cl12n + cl13n + cl23n
        d_E = En - E

        if d_E < 0 or rng.random() < math.exp(-d_E / T):
            if ls_idx == 0: L1 = Ln
            elif ls_idx == 1: L2 = Ln
            else: L3 = Ln
            cl12, cl13, cl23 = cl12n, cl13n, cl23n
            E = En

            if E < best_E:
                best_E = E
                best_L1, best_L2, best_L3 = L1.copy(), L2.copy(), L3.copy()
                if log_fn:
                    log_fn(f"  new_best E={best_E} cl12={cl12} cl13={cl13} cl23={cl23} "
                           f"t={elapsed:.1f}s T={T:.3f}")
                if best_E == 0:
                    return best_L1, best_L2, best_L3, 0

        n_steps += 1

    if log_fn:
        rate = n_steps / max(budget_s, 1)
        log_fn(f"  steps={n_steps} ({rate:.0f}/s)")

    return best_L1, best_L2, best_L3, best_E


# ---------------------------------------------------------------------------
# ILS: re-perturb when stuck
# ---------------------------------------------------------------------------

def ils_perturb(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray,
                n: int, rng: random.Random, strength: int = 4
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply 'strength' random LS-valid moves to the worst LS."""
    L1 = L1.copy(); L2 = L2.copy(); L3 = L3.copy()
    lss = [L1, L2, L3]
    for _ in range(strength):
        ls_idx = rng.randint(0, 2)
        mv = rng.choice(["row_swap", "col_swap", "relabel"])
        LS = lss[ls_idx]
        if mv == "row_swap":
            r, s = rng.sample(range(n), 2); LS[[r, s]] = LS[[s, r]]
        elif mv == "col_swap":
            c, d = rng.sample(range(n), 2); LS[:, [c, d]] = LS[:, [d, c]]
        else:
            a, b = rng.sample(range(n), 2)
            ma, mb = (LS == a), (LS == b); LS[ma] = b; LS[mb] = a
    return L1, L2, L3


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def log(msg: str, log_fp=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n"); log_fp.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--budget", type=float, default=180,
                        help="seconds per SA phase")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, "a") as lfp:
        log("=" * 60, lfp)
        log(f"Joint 3-LS SA — seed={args.seed}  budget_per_trial={args.budget}s", lfp)
        log("=" * 60, lfp)

        trial = 0
        session_best = 999

        while True:
            trial += 1

            pool = load_pool()
            if not pool:
                log(f"trial={trial:4d}  no pool entries, sleeping 30s", lfp)
                time.sleep(30)
                continue

            entry = rng.choice(pool)
            L1_init, L2_init, L3_init = pool_entry_to_matrices(entry, N)

            # ILS-style: every 3rd trial perturb L1/L2 to explore new territory
            if trial % 3 == 0 and trial > 1:
                L1_init, L2_init, L3_init = ils_perturb(
                    L1_init, L2_init, L3_init, N, rng, strength=rng.randint(2, 6))

            init_E = (count_clashes(L1_init, L2_init) + count_clashes(L1_init, L3_init)
                      + count_clashes(L2_init, L3_init))

            log(f"trial={trial:4d}  strategy=joint_sa  init_E={init_E}  "
                f"budget={args.budget}s  time={datetime.now().strftime('%H:%M:%S')}", lfp)

            def log_imp(msg): log(msg, lfp)

            t0 = time.time()
            L1, L2, L3, best_E = joint_sa(
                L1_init, L2_init, L3_init, N,
                budget_s=args.budget, rng=rng,
                T_init=15.0, T_min=0.01, log_fn=log_imp,
            )
            elapsed = time.time() - t0

            if best_E < session_best:
                session_best = best_E

            cl12 = count_clashes(L1, L2)
            cl13 = count_clashes(L1, L3)
            cl23 = count_clashes(L2, L3)
            log(f"  done: best_E={best_E}  cl12={cl12}  cl13={cl13}  cl23={cl23}  "
                f"session_best={session_best}  elapsed={elapsed:.1f}s", lfp)

            if best_E == 0:
                log(f"!!! N(10) >= 3 FOUND !!! seed={args.seed} trial={trial}", lfp)
                save_found(L1, L2, L3, args.seed)
                print("MOLS10_FOUND written — N(10) >= 3 proven!", flush=True)
                sys.exit(0)

            if trial % 5 == 0:
                log(f"  [status] pool={len(pool)} entries  session_best={session_best}", lfp)


if __name__ == "__main__":
    main()
