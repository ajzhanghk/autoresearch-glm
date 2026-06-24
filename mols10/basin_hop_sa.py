#!/usr/bin/env python3
"""
Basin-Hopping SA for MOLS-10.

Strategy: repeatedly perturb a good state with random IC moves,
then run fast SA from scratch. Accepts new best greedily.
Proven by TV-254 PT finding E=25 at round 14 — that basin exists.
Goal: rediscover E=25 (tv_254) or E<26 (iso2), then push further.
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

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
LOG       = REPO / f"mols10/results/basin_hop_{PAIR_ID}.log"
BEST_FILE = REPO / f"mols10/results/basin_hop_best_{PAIR_ID}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

SA_STEPS   = 2_000_000
T_START    = 2.0
T_END      = 0.05
N_PERTURB_MIN = 5
N_PERTURB_MAX = 30

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

def random_ic_move(L3, rng):
    for _ in range(500):
        r1,r2 = rng.sample(range(N), 2)
        c1,c2 = rng.sample(range(N), 2)
        A,B,C,D = int(L3[r1,c1]),int(L3[r1,c2]),int(L3[r2,c1]),int(L3[r2,c2])
        if A==D and B==C and A!=B:
            L3[r1,c1]=B; L3[r1,c2]=A; L3[r2,c1]=A; L3[r2,c2]=B
            return True
    return False

# Load pair
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L1f = L1.ravel().astype(np.int32)
L2f = L2.ravel().astype(np.int32)

# Load best seed
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())
seed_L3 = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
E_seed = count_clashes(L1, seed_L3) + count_clashes(L2, seed_L3)

# Try to load a better seed
for _fp in [
    REPO/"mols10/results/cpsat_e27_step_best.json",
    REPO/f"mols10/results/pt_best_{PAIR_ID}.json",
    REPO/f"mols10/results/basin_hop_best_{PAIR_ID}.json",
]:
    if _fp.exists():
        _d = json.loads(_fp.read_text())
        if _d.get('pair_id') == PAIR_ID and _d.get('E', 999) < E_seed:
            seed_L3 = np.array(_d['L3'], dtype=np.int8).reshape(N,N)
            E_seed = _d['E']

global_best_E = E_seed
global_best_L3 = seed_L3.copy()

log("="*70)
log(f"Basin-Hopping SA — pair={PAIR_ID} seed_E={global_best_E}")
log(f"SA steps={SA_STEPS//1000}k, T=[{T_START},{T_END}]")
log("="*70)

rng = random.Random(271828 + hash(PAIR_ID) % 100000)
t_start = time.time()
restart_num = 0

while True:
    restart_num += 1
    # Perturb from global best
    n_perturb = rng.randint(N_PERTURB_MIN, N_PERTURB_MAX)
    L3_perturbed = global_best_L3.copy()
    for _ in range(n_perturb):
        random_ic_move(L3_perturbed, rng)

    E_perturbed = count_clashes(L1, L3_perturbed) + count_clashes(L2, L3_perturbed)

    # Run SA
    seed_int = rng.randint(1, 2**31-1)
    pc1 = make_pc(L1, L3_perturbed)
    pc2 = make_pc(L2, L3_perturbed)
    L3f = L3_perturbed.ravel().astype(np.int32).copy()
    E_final, L3_out = _run_sa_core(L1f, L2f, L3f, pc1, pc2, SA_STEPS,
                                    T_START, T_END, seed_int)
    E_final = int(E_final)
    L3_final = L3_out.astype(np.int8).reshape(N,N)

    if E_final < global_best_E:
        global_best_E = E_final
        global_best_L3 = L3_final.copy()
        cl13 = int(count_clashes(L1, L3_final))
        cl23 = int(count_clashes(L2, L3_final))
        elapsed = time.time() - t_start
        log(f"*** NEW BEST E={E_final} cl13={cl13} cl23={cl23} restart={restart_num} perturb={n_perturb} t={elapsed:.0f}s ***")
        BEST_FILE.write_text(json.dumps({
            'E': E_final, 'cl13': cl13, 'cl23': cl23,
            'pair_id': PAIR_ID, 'restart': restart_num,
            'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_final.tolist()
        }, indent=2))
        git_push(f"basin_hop: {PAIR_ID} E={E_final}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if E_final == 0:
            log("*** 3-MOLS FOUND! ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'pair_id': PAIR_ID,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_final.tolist(),
                'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
            }, indent=2))
            sys.exit(0)

    if restart_num % 20 == 0:
        elapsed = time.time() - t_start
        rate = restart_num / elapsed * 3600
        log(f"Restart {restart_num} | best={global_best_E} | {rate:.0f} restarts/hr | t={elapsed:.0f}s")
