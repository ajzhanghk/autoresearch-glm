#!/usr/bin/env python3
"""
mols_reverse_search.py — Fix near-miss L3, search for compatible (L1, L2).

The near-miss pool has E=37 triples with cl12=0, cl13=22, cl23=15 for
specific (L1_nm, L2_nm, L3_nm). The question: does L3_nm work as a third
MOLS square for ANY pair (L1, L2)? This script fixes L3=L3_nm and runs a
parallel-tempering SA to find L1, L2 such that:
  - cl12 = 0  (L1 ⊥ L2)
  - cl13 = 0  (L1 ⊥ L3)
  - cl23 = 0  (L2 ⊥ L3)

If successful: N(10) ≥ 3 is proven.

Different from main search: L3 is fixed, only L1 and L2 are moved.
This narrows the search space and might find compatible pairs if they exist.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mols10.mols_adaptive import (
    count_clashes,
    random_latin_square,
    verify_mols,
)
from mols10.mols_l3_search import (
    _row_swap, _col_swap, _relabel, _intercalate_flip,
    _load_triple_misses, _load_pairs, _save_triple_miss,
    isotopy_variant,
)

RESULTS_DIR = Path(__file__).parent / "results"
TRIPLE_FILE  = RESULTS_DIR / "near_miss_triple.json"


def sa_l1l2_fixed_l3(
    L3_fixed: np.ndarray,
    n: int,
    max_seconds: float,
    temps: tuple = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0),
    rng_seed: Optional[int] = None,
) -> tuple[Optional[tuple], dict]:
    """Parallel-tempering SA: fix L3, find orthogonal (L1, L2).

    Energy E = cl12 + cl13 + cl23 where L3 is fixed (never moved).
    Only L1 and L2 are varied. E=0 means L3 is a valid third MOLS square.
    """
    rng   = random.Random(rng_seed)
    t0    = time.time()
    K     = len(temps)
    swap_every = 2000
    move_names = ["row", "col", "relabel", "intercalate", "kscramble"]
    rep_rngs = [random.Random(rng.random()) for _ in range(K)]

    seed_pairs = _load_pairs()

    def random_state(r):
        if seed_pairs and r.random() < 0.6:
            sp = r.choice(seed_pairs)
            L1_ = sp[0].copy(); L2_ = sp[1].copy()
            L1_, L2_ = isotopy_variant(L1_, L2_, n, random.Random(r.random()))
        else:
            L1_ = random_latin_square(n, random.Random(r.random()))
            L2_ = random_latin_square(n, random.Random(r.random()))
        cl12_ = count_clashes(L1_, L2_, n)
        cl13_ = count_clashes(L1_, L3_fixed, n)
        cl23_ = count_clashes(L2_, L3_fixed, n)
        return [L1_, L2_, cl12_, cl13_, cl23_]

    def apply_move(state, r):
        L1_, L2_, cl12_, cl13_, cl23_ = state
        # Move bias: if cl12 < 5, prefer optimizing whichever is worse among cl13/cl23
        if cl12_ < 5:
            if cl13_ > cl23_:
                tgt = r.choice([0, 0, 0, 1])  # prefer L1
            else:
                tgt = r.choice([0, 1, 1, 1])  # prefer L2
        else:
            tgt = r.randint(0, 1)
        L = L1_ if tgt == 0 else L2_
        mv = r.choice(move_names)
        if mv == "row":
            a, b = r.sample(range(n), 2); Ln = _row_swap(L, a, b)
        elif mv == "col":
            a, b = r.sample(range(n), 2); Ln = _col_swap(L, a, b)
        elif mv == "relabel":
            a, b = r.sample(range(n), 2); Ln = _relabel(L, a, b)
        elif mv == "kscramble":
            k = r.randint(3, 7)
            rows = r.sample(range(n), k)
            Ln = L.copy(); perm = rows[:]; r.shuffle(perm)
            Ln[rows] = L[perm]
        else:
            r1, r2 = r.sample(range(n), 2); c1, c2 = r.sample(range(n), 2)
            fl = _intercalate_flip(L, r1, r2, c1, c2)
            Ln = fl if fl is not None else L
        if tgt == 0:
            ncl12 = count_clashes(Ln, L2_, n)
            ncl13 = count_clashes(Ln, L3_fixed, n)
            new_E = ncl12 + ncl13 + cl23_
            return [Ln, L2_, ncl12, ncl13, cl23_], new_E, cl12_ + cl13_ + cl23_
        else:
            ncl12 = count_clashes(L1_, Ln, n)
            ncl23 = count_clashes(Ln, L3_fixed, n)
            new_E = ncl12 + cl13_ + ncl23
            return [L1_, Ln, ncl12, cl13_, ncl23], new_E, cl12_ + cl13_ + cl23_

    replicas = [random_state(rep_rngs[i]) for i in range(K)]
    energies = [s[2] + s[3] + s[4] for s in replicas]
    best_E   = min(energies)
    best_idx = energies.index(best_E)
    best_L1  = replicas[best_idx][0].copy()
    best_L2  = replicas[best_idx][1].copy()
    step = 0; swaps = 0; restarts = 0
    last_improve = t0; ils_interval = 10.0

    while time.time() - t0 < max_seconds:
        for i in range(K):
            new_state, new_E, old_E = apply_move(replicas[i], rep_rngs[i])
            delta = new_E - old_E
            T = temps[i]
            if delta < 0 or rep_rngs[i].random() < math.exp(-delta / T):
                replicas[i] = new_state
                energies[i] = new_E
                if new_E < best_E:
                    best_E = new_E
                    best_L1 = new_state[0].copy()
                    best_L2 = new_state[1].copy()
                    last_improve = time.time()
                    if new_E == 0:
                        ok, _ = verify_mols([best_L1, best_L2, L3_fixed])
                        if ok:
                            return (best_L1, best_L2, L3_fixed), {
                                "found": True, "best_E": 0,
                                "swaps": swaps, "elapsed_s": round(time.time()-t0, 2),
                            }

        step += 1

        # ILS: shake cold replica L1 and L2 when stagnated
        now = time.time()
        if now - last_improve >= ils_interval and now - getattr(apply_move, '_last_ils', 0) >= ils_interval:
            apply_move._last_ils = now
            s = replicas[0]
            L1s, L2s = s[0], s[1]
            n_shake = rep_rngs[0].randint(20, 60)
            # Shake L1 by permuting random rows
            L1_shk = L1s.copy()
            for _ in range(n_shake):
                a, b = rep_rngs[0].sample(range(n), 2)
                L1_shk = _row_swap(L1_shk, a, b)
            cl12_shk = count_clashes(L1_shk, L2s, n)
            cl13_shk = count_clashes(L1_shk, L3_fixed, n)
            replicas[0] = [L1_shk, L2s, cl12_shk, cl13_shk, s[4]]
            energies[0] = cl12_shk + cl13_shk + s[4]
            restarts += 1

        if step % swap_every == 0:
            for i in range(K - 1):
                Ei, Ej = energies[i], energies[i + 1]
                Ti, Tj = temps[i], temps[i + 1]
                log_prob = (Ei - Ej) * (1.0 / Ti - 1.0 / Tj)
                if log_prob >= 0 or rng.random() < math.exp(log_prob):
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = Ej, Ei
                    swaps += 1

        if step % (swap_every * 50) == 0:
            replicas[-1] = random_state(rep_rngs[-1])
            energies[-1] = sum(replicas[-1][2:])

    return None, {
        "found": False, "best_E": best_E,
        "swaps": swaps, "restarts": restarts,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--budget", type=float, default=120.0,
                    help="SA budget per trial in seconds")
    args = ap.parse_args()

    n = 10
    rng = random.Random(args.seed)

    print(f"====================================================================")
    print(f"MOLS Reverse Search — fix near-miss L3, find compatible (L1, L2)")
    print(f"seed={args.seed}  budget_per_trial={args.budget}s")
    print(f"====================================================================")

    trial = 0
    session_best = 300

    while True:
        trial += 1
        # Reload near-miss L3 each trial (pool updates from other workers)
        misses = _load_triple_misses()
        if not misses:
            print("[reverse] No near-miss pool yet, waiting...")
            import time as _t; _t.sleep(30)
            continue

        # Pick a near-miss entry (prefer E=37 entries, use top-5)
        e = rng.choice(misses[:min(5, len(misses))])
        L3_fixed = np.array(e["L3"], dtype=np.int8)
        miss_E = e["clashes"]

        seed = rng.randint(0, 2**31)
        print(f"\ntrial={trial:4d}  strategy=reverse  L3_from_E={miss_E}"
              f"  budget={args.budget:.0f}s"
              f"  time={datetime.now().strftime('%H:%M:%S')}")

        result, stats = sa_l1l2_fixed_l3(L3_fixed, n, args.budget, rng_seed=seed)

        elapsed = stats.get("elapsed_s", args.budget)
        best_E  = stats.get("best_E", 300)
        found   = stats.get("found", False)
        swaps   = stats.get("swaps", 0)
        restarts = stats.get("restarts", 0)

        if best_E < session_best:
            session_best = best_E

        print(f"  reverse: found={found}  best_E={best_E}  "
              f"session_best={session_best}  swaps={swaps}  "
              f"ils_kicks={restarts}  elapsed={elapsed:.1f}s")

        if found and result is not None:
            L1, L2, L3 = result
            print(f"\n{'='*60}")
            print(f"★★★ N(10) ≥ 3 PROVEN by reverse search! ★★★")
            print(f"{'='*60}")
            # Save the solution
            sol = {
                "found": True,
                "method": "reverse_search",
                "ts": datetime.now().isoformat(),
                "L1": L1.tolist(),
                "L2": L2.tolist(),
                "L3": L3.tolist(),
            }
            (RESULTS_DIR / "mols3_solution.json").write_text(json.dumps(sol, indent=2))
            print("Solution saved to mols10/results/mols3_solution.json")
            sys.exit(0)

        # Also save as near-miss if good
        if best_E < 300 and result is None:
            from mols10.mols_l3_search import _save_triple_miss
            # best_L1, best_L2 need to be recovered from stats (not currently returned)
            pass


if __name__ == "__main__":
    main()
