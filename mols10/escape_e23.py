#!/usr/bin/env python3
"""
Escape from E=23 local minimum using non-IC moves.

E=23 is a strict 4-deep IC local minimum — all IC-reachable states
within 4 moves have E>=23. IC-only SA cannot escape it.

Strategy: use row/column SWAPS of L3 (large jumps in state space)
to land in a different basin, then run fast IC SA from there.
Also try SYMBOL RELABELING (permuting the values in L3).

These are all valid Latin square operations (result is still a LS).
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np
import itertools

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
sys.path.insert(0, "/tmp")
from fast_sa_numba import make_pc, _run_sa_core

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
INSTANCE = sys.argv[2] if len(sys.argv) > 2 else '0'
LOG       = REPO / f"mols10/results/escape_{PAIR_ID}_{INSTANCE}.log"
BEST_FILE = REPO / f"mols10/results/escape_best_{PAIR_ID}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

SA_STEPS_REFINE = 5_000_000  # After each escape, refine with SA
T_START = 2.0
T_END   = 0.02

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add",str(BEST_FILE.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def run_sa(L1f, L2f, L3_init, n_steps, T_start, T_end, seed):
    pc1 = make_pc(L1, L3_init)
    pc2 = make_pc(L2, L3_init)
    L3f = L3_init.ravel().astype(np.int32).copy()
    E, L3_out = _run_sa_core(L1f, L2f, L3f, pc1, pc2, n_steps, T_start, T_end, seed)
    return int(E), L3_out.astype(np.int8).reshape(N, N)

def apply_row_swap(L3, r1, r2):
    L3n = L3.copy()
    L3n[[r1, r2]] = L3n[[r2, r1]]
    return L3n

def apply_col_swap(L3, c1, c2):
    L3n = L3.copy()
    L3n[:, [c1, c2]] = L3n[:, [c2, c1]]
    return L3n

def apply_symbol_relabel(L3, perm):
    """Apply symbol permutation: value v becomes perm[v]."""
    L3n = np.zeros_like(L3)
    for i in range(N):
        for j in range(N):
            L3n[i, j] = perm[int(L3[i, j])]
    return L3n

# Load pair
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L1f = L1.ravel().astype(np.int32)
L2f = L2.ravel().astype(np.int32)

# Load best seed
seed_L3 = None
E_seed = 999
for _fp in [
    REPO/f"mols10/results/escape_best_{PAIR_ID}.json",
    REPO/f"mols10/results/pt_agg_best_{PAIR_ID}_b.json",
    REPO/f"mols10/results/pt_agg_best_{PAIR_ID}_a.json",
    REPO/f"mols10/results/pt_best_{PAIR_ID}.json",
]:
    if _fp.exists():
        _d = json.loads(_fp.read_text())
        if _d.get('pair_id') == PAIR_ID and _d.get('E', 999) < E_seed:
            seed_L3 = np.array(_d['L3'], dtype=np.int8).reshape(N,N)
            E_seed = _d['E']

if seed_L3 is None:
    best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())
    seed_L3 = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
    E_seed = count_clashes(L1, seed_L3) + count_clashes(L2, seed_L3)

global_best_E = E_seed
global_best_L3 = seed_L3.copy()

log("="*70)
log(f"Escape SA — pair={PAIR_ID} inst={INSTANCE} seed_E={global_best_E}")
log(f"SA refine: {SA_STEPS_REFINE//1000}k steps, T=[{T_START},{T_END}]")
log("="*70)

rng = random.Random(271828182 + hash(PAIR_ID + INSTANCE) % 100000)
t_start = time.time()
trial_num = 0

def check_and_save(E_new, L3_new):
    global global_best_E, global_best_L3
    if E_new < global_best_E:
        global_best_E = E_new
        global_best_L3 = L3_new.copy()
        cl13 = int(count_clashes(L1, L3_new))
        cl23 = int(count_clashes(L2, L3_new))
        elapsed = time.time() - t_start
        log(f"*** NEW BEST E={E_new} cl13={cl13} cl23={cl23} trial={trial_num} t={elapsed:.0f}s ***")
        BEST_FILE.write_text(json.dumps({
            'E': E_new, 'cl13': cl13, 'cl23': cl23,
            'pair_id': PAIR_ID, 'trial': trial_num,
            'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_new.tolist()
        }, indent=2))
        git_push(f"escape: {PAIR_ID} E={E_new}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if E_new == 0:
            log("*** 3-MOLS FOUND! ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'pair_id': PAIR_ID,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_new.tolist(),
                'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
            }, indent=2))
            import os; os._exit(0)

while True:
    trial_num += 1
    move_type = rng.random()

    if move_type < 0.40:
        # Random row swap escape
        r1, r2 = rng.sample(range(N), 2)
        L3_escaped = apply_row_swap(global_best_L3, r1, r2)
        escape_label = f"rowswap({r1},{r2})"
    elif move_type < 0.80:
        # Random column swap escape
        c1, c2 = rng.sample(range(N), 2)
        L3_escaped = apply_col_swap(global_best_L3, c1, c2)
        escape_label = f"colswap({c1},{c2})"
    else:
        # Random symbol relabeling
        perm = list(range(N))
        rng.shuffle(perm)
        L3_escaped = apply_symbol_relabel(global_best_L3, perm)
        escape_label = f"relabel"

    E_escaped = count_clashes(L1, L3_escaped) + count_clashes(L2, L3_escaped)

    # Run fast SA from escaped state
    seed_int = rng.randint(1, 2**31-1)
    E_refined, L3_refined = run_sa(L1f, L2f, L3_escaped, SA_STEPS_REFINE,
                                    T_START, T_END, seed_int)

    E_check = count_clashes(L1, L3_refined) + count_clashes(L2, L3_refined)
    if E_check != E_refined:
        E_refined = E_check

    check_and_save(E_refined, L3_refined)

    if trial_num % 10 == 0:
        elapsed = time.time() - t_start
        rate = trial_num / elapsed * 3600
        log(f"Trial {trial_num} | best={global_best_E} | {rate:.0f}/hr | t={elapsed:.0f}s")
