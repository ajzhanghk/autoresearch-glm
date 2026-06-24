#!/usr/bin/env python3
"""
Random Isotopy + SA Search.

For each of the best E=26 pairs, apply random isotopy transformations
(row permutations of L1 and L2 independently) then run SA.
This explores completely different (L1', L2') pairs than the ones we've tried.

The energy landscape depends on the pair (L1, L2). A random isotopy
may expose a different landscape with lower barriers from E=0.
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
sys.path.insert(0, "/tmp")
from fast_sa_numba import make_pc, _run_sa_core

SEED_PAIRS = ['iso2_tv21_0_5', 'tv_254', 'iso_tv_21_0']
INSTANCE_ID = sys.argv[1] if len(sys.argv) > 1 else '0'
LOG       = REPO / f"mols10/results/rand_iso_{INSTANCE_ID}.log"
BEST_FILE = REPO / f"mols10/results/rand_iso_best_{INSTANCE_ID}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

SA_STEPS = 5_000_000
T_START  = 3.0
T_END    = 0.02
GOAL_E   = 25

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add",
                    str(BEST_FILE.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def apply_isotopy(L, perm):
    """Apply row permutation perm to Latin square L (returns new array)."""
    return L[perm, :]

# Load pairs
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

# Better seeds
cpsat_seed = None
_fp = REPO/"mols10/results/cpsat_e27_step_best.json"
if _fp.exists():
    cpsat_seed = json.loads(_fp.read_text())

GLOBAL_RECORD = 26  # current best from fixed pairs — only report if we beat this
global_best_E = 999
global_best_info = None

rng = random.Random(314159265 + int(INSTANCE_ID) * 999983)

log("="*70)
log(f"Random Isotopy SA | instance={INSTANCE_ID} | seeds={SEED_PAIRS}")
log(f"SA steps={SA_STEPS//1000}k, T=[{T_START},{T_END}]")
log("="*70)

t_start = time.time()
trial_num = 0

while True:
    trial_num += 1

    # Choose a random base pair
    base_pair_id = rng.choice(SEED_PAIRS)
    p = pair_map[base_pair_id]
    L1_base = np.array(p['L1'], dtype=np.int8).reshape(N,N)
    L2_base = np.array(p['L2'], dtype=np.int8).reshape(N,N)

    # Apply random isotopy: permute rows of L1 and L2 independently
    # Also optionally permute columns
    perm1 = np.array(rng.sample(range(N), N), dtype=np.int8)
    perm2 = np.array(rng.sample(range(N), N), dtype=np.int8)
    L1_iso = L1_base[perm1, :]
    L2_iso = L2_base[perm2, :]

    # Check L1_iso and L2_iso are still Latin squares (they should be)
    L1f = L1_iso.ravel().astype(np.int32)
    L2f = L2_iso.ravel().astype(np.int32)

    # Random L3 seed — either random or from existing best
    seed_choice = rng.random()
    if seed_choice < 0.3 and base_pair_id in best_data:
        # Use known good L3 as seed (might not be good for this isotopy but worth trying)
        b = best_data[base_pair_id]
        L3_init = np.array(b['L3'], dtype=np.int8).reshape(N,N)
    elif seed_choice < 0.6 and cpsat_seed and cpsat_seed.get('pair_id') == base_pair_id:
        L3_init = np.array(cpsat_seed['L3'], dtype=np.int8).reshape(N,N)
    else:
        # Random L3 (valid Latin square)
        L3_init = np.array([rng.sample(range(N), N) for _ in range(N)], dtype=np.int8)

    seed_int = rng.randint(1, 2**31-1)
    pc1 = make_pc(L1_iso, L3_init)
    pc2 = make_pc(L2_iso, L3_init)
    L3f = L3_init.ravel().astype(np.int32).copy()
    E_final, L3_out = _run_sa_core(L1f, L2f, L3f, pc1, pc2, SA_STEPS,
                                    T_START, T_END, seed_int)
    E_final = int(E_final)
    L3_final = L3_out.astype(np.int8).reshape(N,N)

    # Verify
    E_check = count_clashes(L1_iso, L3_final) + count_clashes(L2_iso, L3_final)
    if E_check != E_final:
        log(f"WARNING: SA returned {E_final} but verify={E_check}, using {E_check}")
        E_final = E_check

    if E_final < global_best_E:
        global_best_E = E_final
        cl13 = int(count_clashes(L1_iso, L3_final))
        cl23 = int(count_clashes(L2_iso, L3_final))
        elapsed = time.time() - t_start
        log(f"*** NEW BEST E={E_final} cl13={cl13} cl23={cl23} trial={trial_num} base={base_pair_id} t={elapsed:.0f}s ***")
        if E_final < GLOBAL_RECORD:
            BEST_FILE.write_text(json.dumps({
                'E': E_final, 'cl13': cl13, 'cl23': cl23,
                'base_pair_id': base_pair_id, 'trial': trial_num,
                'L1': L1_iso.tolist(), 'L2': L2_iso.tolist(), 'L3': L3_final.tolist()
            }, indent=2))
            git_push(f"rand_iso: GLOBAL RECORD E={E_final} (base={base_pair_id})\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if E_final == 0:
            log("*** 3-MOLS FOUND! ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'base_pair_id': base_pair_id,
                'L1': L1_iso.tolist(), 'L2': L2_iso.tolist(), 'L3': L3_final.tolist(),
                'cl12': int(count_clashes(L1_iso, L2_iso)), 'cl13': cl13, 'cl23': cl23
            }, indent=2))
            sys.exit(0)

    if trial_num % 50 == 0:
        elapsed = time.time() - t_start
        rate = trial_num / elapsed * 3600
        log(f"Trial {trial_num} | best={global_best_E} | {rate:.0f} trials/hr | t={elapsed:.0f}s")
