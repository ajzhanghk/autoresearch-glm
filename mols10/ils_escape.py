#!/usr/bin/env python3
"""
Iterated Local Search (ILS) escape from E=23.

Unlike escape_e23.py (which always starts from global best E=23),
ILS accepts slightly worse states as new starting points, diversifying
the escape trajectory. Accepts result if E < current_start_E + DELTA.
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

PAIR_ID  = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
INSTANCE = sys.argv[2] if len(sys.argv) > 2 else '0'
DELTA    = int(sys.argv[3]) if len(sys.argv) > 3 else 4  # Accept if E <= best + DELTA

LOG       = REPO / f"mols10/results/ils_{PAIR_ID}_{INSTANCE}.log"
BEST_FILE = REPO / f"mols10/results/escape_best_{PAIR_ID}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

SA_STEPS  = 3_000_000
T_START   = 2.0
T_END     = 0.05

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

def random_escape(L3, rng):
    """Apply 1-3 random non-IC moves."""
    n_moves = rng.choices([1,2,3], weights=[40,40,20])[0]
    L3c = L3.copy()
    for _ in range(n_moves):
        mt = rng.random()
        if mt < 0.25:
            r1, r2 = rng.sample(range(N), 2)
            L3c[[r1,r2]] = L3c[[r2,r1]]
        elif mt < 0.50:
            c1, c2 = rng.sample(range(N), 2)
            L3c[:,[c1,c2]] = L3c[:,[c2,c1]]
        elif mt < 0.70:
            perm = list(range(N)); rng.shuffle(perm)
            L3n = np.zeros_like(L3c)
            for i in range(N):
                for j in range(N):
                    L3n[i,j] = perm[int(L3c[i,j])]
            L3c = L3n
        elif mt < 0.85:
            perm = list(range(N)); rng.shuffle(perm)
            L3c = L3c[np.array(perm)]
        else:
            perm = list(range(N)); rng.shuffle(perm)
            L3c = L3c[:, np.array(perm)]
    return L3c

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
current_L3 = seed_L3.copy()
current_E = E_seed

log("="*70)
log(f"ILS escape — pair={PAIR_ID} inst={INSTANCE} delta={DELTA} seed_E={global_best_E}")
log(f"SA: {SA_STEPS//1000}k steps, T=[{T_START},{T_END}]  accept if E<=best+{DELTA}")
log("="*70)

rng = random.Random(161803398 + hash(PAIR_ID + INSTANCE) % 100000)
t_start = time.time()
trial_num = 0
accepts = 0

while True:
    trial_num += 1

    # Escape from current ILS state (not always from global best)
    L3_escaped = random_escape(current_L3, rng)
    seed_int = rng.randint(1, 2**31-1)
    E_new, L3_new = run_sa(L1f, L2f, L3_escaped, SA_STEPS, T_START, T_END, seed_int)
    E_check = count_clashes(L1, L3_new) + count_clashes(L2, L3_new)
    if E_check != E_new: E_new = E_check

    # Update global best
    if E_new < global_best_E:
        global_best_E = E_new
        global_best_L3 = L3_new.copy()
        cl13 = int(count_clashes(L1, L3_new))
        cl23 = int(count_clashes(L2, L3_new))
        elapsed = time.time() - t_start
        log(f"*** NEW GLOBAL BEST E={E_new} cl13={cl13} cl23={cl23} trial={trial_num} t={elapsed:.0f}s ***")
        BEST_FILE.write_text(json.dumps({
            'E': E_new, 'cl13': cl13, 'cl23': cl23,
            'pair_id': PAIR_ID, 'trial': trial_num,
            'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_new.tolist()
        }, indent=2))
        git_push(f"ils: {PAIR_ID} E={E_new}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if E_new == 0:
            log("*** 3-MOLS FOUND! ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'pair_id': PAIR_ID,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_new.tolist(),
                'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
            }, indent=2))
            import os; os._exit(0)

    # ILS acceptance: accept if better than current_E, or within DELTA of global best
    if E_new <= global_best_E + DELTA:
        current_L3 = L3_new.copy()
        current_E = E_new
        accepts += 1

    if trial_num % 20 == 0:
        elapsed = time.time() - t_start
        rate = trial_num / elapsed * 3600
        accept_rate = accepts / trial_num * 100
        log(f"Trial {trial_num} | global_best={global_best_E} | current={current_E} | "
            f"{rate:.0f}/hr | accept_rate={accept_rate:.0f}% | t={elapsed:.0f}s")
