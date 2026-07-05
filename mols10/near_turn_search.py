#!/usr/bin/env python3
"""
Near-turn-square search: IC-perturb high-transversal turn-squares to explore
squares OUTSIDE the turn family (which was studied classically) while keeping
transversal counts in the thousands. For each candidate, stream orthogonal
mates and compute ct(L,B), completing the triple whenever ct >= 10.

Usage: python3 near_turn_search.py <instance>
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from turn_square_search import (build_turn_square, count_transversals,
                                enum_transversals, stream_mates_and_ct,
                                try_complete_triple, pattern_hillclimb)

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LOG = REPO / f"mols10/results/nearturn_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE = REPO / f"mols10/results/nearturn_best_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add","-A"], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def apply_random_ic(L, rng, tries=60):
    for _ in range(tries):
        r1, r2 = rng.sample(range(N), 2)
        c1, c2 = rng.sample(range(N), 2)
        v11=int(L[r1,c1]); v12=int(L[r1,c2]); v21=int(L[r2,c1]); v22=int(L[r2,c2])
        if v11==v22 and v12==v21:
            L2 = L.copy()
            L2[r1,c1]=v12; L2[r1,c2]=v11; L2[r2,c1]=v22; L2[r2,c2]=v21
            return L2, True
    return L, False

def main():
    log("=" * 70)
    log(f"Near-Turn-Square Search — instance={INSTANCE}")
    log("IC perturbations of 5504-transversal turn-squares, mate/ct streaming")
    log("=" * 70)

    rng = random.Random(3141592653 + INSTANCE * 104729)

    # Get a 5504 turn-square quickly
    log("Finding a max-transversal turn-square...")
    ranked = pattern_hillclimb(rng, n_restarts=6, n_steps=200)
    cnt, key = ranked[0]
    pat = [list(row) for row in key]
    base = build_turn_square(pat)
    log(f"Base turn-square: {cnt} transversals")

    global_best_ct = -1
    trial = 0
    while True:
        trial += 1
        # Perturb with k IC moves (k in 1..4), keep if transversal count stays high
        k = rng.randint(1, 4)
        L = base.copy()
        moved = 0
        for _ in range(k):
            L, ok = apply_random_ic(L, rng)
            if ok: moved += 1
        if moved == 0:
            continue
        n_trans = count_transversals(L, cap=100000)
        if n_trans < 2000:
            if trial % 50 == 0:
                log(f"trial {trial}: n_trans={n_trans} (below 2000, skipped); best_ct so far={global_best_ct}")
            continue

        trans = enum_transversals(L)
        log(f"trial {trial}: k={moved} IC moves, {len(trans)} transversals; streaming mates (240s)...")
        cb = stream_mates_and_ct(L, trans, time_budget_s=240, max_mates=100000)
        hist_top = sorted(cb.ct_hist.items(), reverse=True)[:6]
        log(f"  {cb.n_mates} mates; best_ct={cb.best_ct}; hist(top)={hist_top}")

        if cb.best_ct > global_best_ct and cb.best_mate_decomp is not None:
            global_best_ct = cb.best_ct
            BEST_FILE.write_text(json.dumps({
                'trial': trial, 'n_trans': len(trans), 'n_mates_seen': cb.n_mates,
                'best_ct': int(cb.best_ct),
                'L': L.tolist(), 'B': cb.best_mate_decomp[1].tolist(),
                'ct_hist': {str(kk): v for kk, v in sorted(cb.ct_hist.items(), reverse=True)},
            }, indent=2))
            log(f"  NEW GLOBAL BEST ct={global_best_ct} (saved)")

        if cb.best_ct >= N:
            sel, B = cb.best_mate_decomp
            log(f"  *** ct >= {N}! Attempting triple completion... ***")
            C = try_complete_triple(L, B)
            if C is not None:
                cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
                log(f"  VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
                if cl12 == 0 and cl13 == 0 and cl23 == 0:
                    log("*** 3-MOLS(10) FOUND via near-turn-square! ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'near_turn_square',
                        'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': 0, 'cl13': 0, 'cl23': 0,
                    }, indent=2))
                    git_push("3-MOLS(10) FOUND via near-turn-square search!")
                    sys.exit(0)

if __name__ == "__main__":
    main()
