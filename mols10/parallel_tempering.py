#!/usr/bin/env python3
"""
Parallel Tempering (Replica Exchange) SA for MOLS-10.

Never tried before. Multiple SA chains at different temperatures run
simultaneously; periodically swap adjacent replicas if Metropolis criterion
allows. High-T replicas explore freely; low-T replicas refine.

This can escape deep local minima that trap standard SA, because:
- High-T replicas visit high-energy states (broad exploration)
- Swaps can "teleport" low-T replicas to different basins
- Then low-T replicas descend to local minima in those new basins

Setup:
  - 8 replicas at temperatures T1 < T2 < ... < T8
  - Each replica runs SA with numba (fast)
  - Every SWAP_INTERVAL steps, try to swap adjacent replicas
  - Focus on best pairs: iso2_tv21_0_5 and tv_254 (both at E=27)
  - Also try iso_tv_21_0 isotopies
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
sys.path.insert(0, "/tmp")
from fast_sa_numba import make_pc, _run_sa_core

LOG       = REPO / "mols10/results/parallel_tempering.log"
BEST_FILE = REPO / "mols10/results/pt_best.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

# Temperature ladder: geometric from T_LOW to T_HIGH
N_REPLICAS  = 8
T_LOW       = 0.05   # cold replica (near-greedy)
T_HIGH      = 12.0   # hot replica (broad exploration)
TEMPS = [T_LOW * (T_HIGH/T_LOW)**(i/(N_REPLICAS-1)) for i in range(N_REPLICAS)]

STEPS_PER_ROUND = 500_000    # SA steps between swap attempts
N_ROUNDS_LOG    = 20         # log every N rounds
GOAL_E          = 26

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    for f in [str(BEST_FILE.relative_to(REPO))]:
        subprocess.run(["git","-C",str(REPO),"add",f], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def run_sa_one_temp(L1f, L2f, L3_init, pc1, pc2, n_steps, T_start, T_end, seed):
    """Run SA at fixed temperature (T_start == T_end for isothermal)."""
    L3f = L3_init.ravel().astype(np.int32).copy()
    E, L3_out = _run_sa_core(L1f, L2f, L3f, pc1, pc2, n_steps, T_start, T_end, seed)
    return int(E), L3_out.astype(np.int8).reshape(N, N)

# Load pairs
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

# Choose which pair to run PT on
PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'iso2_tv21_0_5'
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L1f = L1.ravel().astype(np.int32)
L2f = L2.ravel().astype(np.int32)

# Seed best L3
seed_L3 = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
E_seed = count_clashes(L1, seed_L3) + count_clashes(L2, seed_L3)

log("="*70)
log(f"Parallel Tempering SA — pair={PAIR_ID} seed_E={E_seed}")
log(f"Replicas={N_REPLICAS}, T range=[{T_LOW},{T_HIGH:.1f}]")
log(f"Temps: {[f'{t:.3f}' for t in TEMPS]}")
log(f"Steps/round: {STEPS_PER_ROUND//1000}k")
log("="*70)

rng = random.Random(31415926535 + hash(PAIR_ID) % 10000)

# Initialize replicas: all start from seed + random perturbations
replicas = []  # list of (E, L3) for each replica
for k in range(N_REPLICAS):
    L3_r = seed_L3.copy()
    # Perturb more for hotter replicas
    n_perturb = k * 3
    for _ in range(n_perturb):
        # Random IC move
        for attempt in range(200):
            r1,r2 = rng.sample(range(N), 2)
            c1,c2 = rng.sample(range(N), 2)
            A,B,C,D = int(L3_r[r1,c1]),int(L3_r[r1,c2]),int(L3_r[r2,c1]),int(L3_r[r2,c2])
            if A==D and B==C and A!=B:
                L3_r[r1,c1]=B; L3_r[r1,c2]=A; L3_r[r2,c1]=A; L3_r[r2,c2]=B
                break
    E_r = count_clashes(L1, L3_r) + count_clashes(L2, L3_r)
    replicas.append([E_r, L3_r.copy()])

global_best_E = E_seed
global_best_L3 = seed_L3.copy()

# Load existing best
if BEST_FILE.exists():
    bd = json.loads(BEST_FILE.read_text())
    if bd.get('pair_id') == PAIR_ID and bd.get('E', 999) < global_best_E:
        global_best_E = bd['E']
        global_best_L3 = np.array(bd['L3'], dtype=np.int8).reshape(N,N)

swap_accepted = [0] * (N_REPLICAS - 1)
swap_tried    = [0] * (N_REPLICAS - 1)
round_num = 0
t_start = time.time()

log(f"Initial replica energies: {[r[0] for r in replicas]}")

while True:
    round_num += 1

    # Step 1: Run SA on each replica independently
    new_replicas = []
    for k, (E_k, L3_k) in enumerate(replicas):
        T = TEMPS[k]
        seed_k = rng.randint(1, 2**31-1)
        pc1 = make_pc(L1, L3_k)
        pc2 = make_pc(L2, L3_k)
        E_new, L3_new = run_sa_one_temp(L1f, L2f, L3_k, pc1, pc2,
                                         STEPS_PER_ROUND, T, T, seed_k)
        new_replicas.append([E_new, L3_new])

    replicas = new_replicas

    # Step 2: Attempt replica swaps (adjacent pairs, alternating even/odd)
    # Even-indexed swaps
    for k in range(0, N_REPLICAS-1, 2):
        E_k, L3_k = replicas[k]
        E_k1, L3_k1 = replicas[k+1]
        T_k, T_k1 = TEMPS[k], TEMPS[k+1]
        # Metropolis acceptance: swap if exp((E_k1 - E_k)(1/T_k - 1/T_k1)) > U[0,1]
        delta = (E_k1 - E_k) * (1.0/T_k - 1.0/T_k1)
        swap_tried[k] += 1
        if delta >= 0 or rng.random() < math.exp(delta):
            replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
            swap_accepted[k] += 1

    # Odd-indexed swaps
    for k in range(1, N_REPLICAS-1, 2):
        E_k, L3_k = replicas[k]
        E_k1, L3_k1 = replicas[k+1]
        T_k, T_k1 = TEMPS[k], TEMPS[k+1]
        delta = (E_k1 - E_k) * (1.0/T_k - 1.0/T_k1)
        swap_tried[k] += 1
        if delta >= 0 or rng.random() < math.exp(delta):
            replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
            swap_accepted[k] += 1

    # Step 3: Check for new best (coldest replica is index 0)
    E_cold = replicas[0][0]
    if E_cold < global_best_E:
        global_best_E = E_cold
        global_best_L3 = replicas[0][1].copy()
        cl13 = int(count_clashes(L1, global_best_L3))
        cl23 = int(count_clashes(L2, global_best_L3))
        log(f"*** NEW BEST E={global_best_E} cl13={cl13} cl23={cl23} round={round_num} ***")
        BEST_FILE.write_text(json.dumps({
            'E': global_best_E, 'cl13': cl13, 'cl23': cl23,
            'pair_id': PAIR_ID, 'round': round_num,
            'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': global_best_L3.tolist()
        }, indent=2))
        git_push(f"pt: {PAIR_ID} E={global_best_E} (NEW BEST!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if global_best_E == 0:
            log("*** 3-MOLS FOUND! ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'pair_id': PAIR_ID,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': global_best_L3.tolist(),
                'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
            }, indent=2))
            sys.exit(0)

    # Also check all replicas for global best
    for k, (E_k, L3_k) in enumerate(replicas):
        if E_k < global_best_E:
            global_best_E = E_k
            global_best_L3 = L3_k.copy()
            cl13 = int(count_clashes(L1, L3_k))
            cl23 = int(count_clashes(L2, L3_k))
            log(f"*** NEW BEST (replica {k}) E={global_best_E} cl13={cl13} cl23={cl23} round={round_num} ***")

    if round_num % N_ROUNDS_LOG == 0:
        elapsed = time.time() - t_start
        energies = [r[0] for r in replicas]
        swap_rates = [f"{swap_accepted[k]/max(swap_tried[k],1)*100:.0f}%" for k in range(N_REPLICAS-1)]
        log(f"Round {round_num:4d} | energies={energies} | best={global_best_E} | "
            f"t={elapsed:.0f}s | swaps={swap_rates}")
