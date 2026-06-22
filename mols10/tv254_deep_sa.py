#!/usr/bin/env python3
"""
Ultra-deep SA on the tv_223 E=29 isotopy found by fast_iso.
Runs 1B steps to push toward E<26 or E=0.
Also searches other promising entries.
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

sys.path.insert(0, "/tmp")
from fast_sa_numba import run_sa_fast, make_pc, _run_sa_core

LOG       = REPO / "mols10/results/tv223_deep.log"
BEST_FILE = REPO / "mols10/results/tv223_deep_best.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

T_START = 8.0   # Higher to explore from E=28/29 starting points
T_END   = 0.01
GOAL_E  = 26


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    for f in [str(BEST_FILE.relative_to(REPO)),
              str((REPO / "mols10/results/fast_iso_promising.json").relative_to(REPO))]:
        subprocess.run(["git", "-C", str(REPO), "add", f], check=False)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg], check=False)
    subprocess.run(["git", "-C", str(REPO), "push", "-u", "origin", BRANCH], check=False)


# Load the promising isotopy entries (sorted by E)
promising_file = REPO / "mols10/results/fast_iso_promising.json"
promising = json.loads(promising_file.read_text())
if not promising:
    print("No promising entries found!")
    sys.exit(1)

log("=" * 70)
log(f"Ultra-deep SA on tv_223 E=29 isotopy ({len(promising)} entries)")
log(f"Goal: E < {GOAL_E}")
log("=" * 70)

bests = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
global_best_E = bests.get('global_best_E', GOAL_E)

rng = random.Random(1618033988)
run_num = 0

while True:
    # Cycle through promising entries by E
    for entry in promising:
        pair_id = entry['pair_id']
        E_promise = entry['E']
        L1_iso = np.array(entry['L1_iso'], dtype=np.int8).reshape(N, N)
        L2_iso = np.array(entry['L2_iso'], dtype=np.int8).reshape(N, N)
        L3_best = np.array(entry['L3'], dtype=np.int8).reshape(N, N)

        # Verify energy
        cl13 = count_clashes(L1_iso, L3_best)
        cl23 = count_clashes(L2_iso, L3_best)
        E_check = cl13 + cl23
        if E_check != E_promise:
            log(f"  {pair_id}: energy mismatch {E_check} vs {E_promise}, using {E_check}")
            E_promise = E_check

        run_num += 1
        pair_best = bests.get(f'{pair_id}_iso{entry["iso_num"]}', {}).get('E', E_promise)

        # Vary perturbation: 0-5 IC kicks from current best
        n_perturb = rng.randint(0, 4)
        L3_start = L3_best.copy()
        for _ in range(n_perturb):
            # Find valid IC and apply
            for attempt in range(200):
                r1 = rng.randint(0, N-1); r2 = rng.randint(0, N-1)
                if r1 == r2: continue
                c1 = rng.randint(0, N-1); c2 = rng.randint(0, N-1)
                if c1 == c2: continue
                a, b = int(L3_start[r1, c1]), int(L3_start[r1, c2])
                if a != b and L3_start[r2, c1] == b and L3_start[r2, c2] == a:
                    L3_start[r1, c1] = b; L3_start[r1, c2] = a
                    L3_start[r2, c1] = a; L3_start[r2, c2] = b
                    break

        # 100M step SA run (about 220s at 454k steps/s)
        n_steps = 100_000_000
        seed = rng.randint(1, 2**31 - 1)

        t0 = time.time()
        E_result, L3_result = run_sa_fast(L1_iso, L2_iso, L3_start, n_steps, seed, T_START, T_END)
        elapsed = time.time() - t0

        cl13r = count_clashes(L1_iso, L3_result)
        cl23r = count_clashes(L2_iso, L3_result)
        log(f"run={run_num} {pair_id} iso={entry['iso_num']} perturb={n_perturb}: "
            f"start_E={E_promise+n_perturb*2} → E={E_result} cl13={cl13r} cl23={cl23r} "
            f"t={elapsed:.0f}s ({n_steps/elapsed:.0f} steps/s)")

        if E_result < pair_best:
            pair_best = E_result
            key = f'{pair_id}_iso{entry["iso_num"]}'
            bests[key] = {'E': int(E_result), 'cl13': int(cl13r), 'cl23': int(cl23r),
                           'pair_id': pair_id, 'iso_num': entry['iso_num'],
                           'L3': L3_result.tolist(), 'L1_iso': entry['L1_iso'],
                           'L2_iso': entry['L2_iso']}
            BEST_FILE.write_text(json.dumps(bests, indent=2))
            log(f"  *** New best for {pair_id} iso={entry['iso_num']}: E={E_result}")

            # Update L3 in promising for next iterations
            entry['L3'] = L3_result.tolist()

        if E_result < global_best_E:
            global_best_E = E_result
            log(f"*** NEW GLOBAL BEST E={global_best_E} for {pair_id} iso={entry['iso_num']}! ***")
            bests['global_best_E'] = int(global_best_E)
            BEST_FILE.write_text(json.dumps(bests, indent=2))
            git_push(f"tv223_deep: {pair_id} E={global_best_E} (new global best!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
            if global_best_E == 0:
                log("*** 3-MOLS FOUND! ***"); sys.exit(0)

    # Reload promising to catch new entries from fast_iso
    promising = json.loads(promising_file.read_text())
    promising.sort(key=lambda x: x['E'])
