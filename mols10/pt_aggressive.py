#!/usr/bin/env python3
"""
Aggressive Parallel Tempering — more replicas, more frequent swaps,
different temperature ranges. Designed to find the E=25 basin for tv_254
that was found previously but lost due to a save bug.
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

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
VARIANT  = sys.argv[2] if len(sys.argv) > 2 else 'a'  # 'a' or 'b' for different configs
LOG       = REPO / f"mols10/results/pt_agg_{PAIR_ID}_{VARIANT}.log"
BEST_FILE = REPO / f"mols10/results/pt_agg_best_{PAIR_ID}_{VARIANT}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

# Variant a: wider T range, smaller steps (more swaps per SA step)
# Variant b: focus on medium T range where E=25 was found (T~1)
if VARIANT == 'a':
    N_REPLICAS  = 10
    T_LOW       = 0.08
    T_HIGH      = 20.0
    STEPS_PER_ROUND = 200_000
else:  # variant b
    N_REPLICAS  = 8
    T_LOW       = 0.3
    T_HIGH      = 5.0
    STEPS_PER_ROUND = 300_000

TEMPS = [T_LOW * (T_HIGH/T_LOW)**(i/(N_REPLICAS-1)) for i in range(N_REPLICAS)]
N_ROUNDS_LOG = 20

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add",str(BEST_FILE.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def run_sa(L1f, L2f, L3_init, pc1, pc2, n_steps, T, seed):
    L3f = L3_init.ravel().astype(np.int32).copy()
    E, L3_out = _run_sa_core(L1f, L2f, L3f, pc1, pc2, n_steps, T, T, seed)
    return int(E), L3_out.astype(np.int8).reshape(N, N)

# Load pair
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L1f = L1.ravel().astype(np.int32)
L2f = L2.ravel().astype(np.int32)

# Load best seed
seed_L3 = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
E_seed = count_clashes(L1, seed_L3) + count_clashes(L2, seed_L3)
for _fp in [
    REPO/"mols10/results/cpsat_e27_step_best.json",
    REPO/f"mols10/results/pt_best_{PAIR_ID}.json",
    REPO/f"mols10/results/pt_agg_best_{PAIR_ID}_{VARIANT}.json",
]:
    if _fp.exists():
        _d = json.loads(_fp.read_text())
        if _d.get('pair_id') == PAIR_ID and _d.get('E', 999) < E_seed:
            seed_L3 = np.array(_d['L3'], dtype=np.int8).reshape(N,N)
            E_seed = _d['E']

global_best_E = E_seed
global_best_L3 = seed_L3.copy()

log("="*70)
log(f"Aggressive PT | pair={PAIR_ID} variant={VARIANT} seed_E={global_best_E}")
log(f"Replicas={N_REPLICAS}, T=[{T_LOW},{T_HIGH:.1f}]")
log(f"Temps: {[f'{t:.3f}' for t in TEMPS]}")
log(f"Steps/round: {STEPS_PER_ROUND//1000}k")
log("="*70)

# Use a different random seed than the regular PT (hash variant)
seed_base = 27182818 + hash(PAIR_ID + VARIANT) % 100000
rng = random.Random(seed_base)

# Initialize replicas
replicas = []
for k in range(N_REPLICAS):
    L3_r = seed_L3.copy()
    n_perturb = k * 5  # more perturbations per replica
    for _ in range(n_perturb):
        for _ in range(200):
            r1,r2 = rng.sample(range(N), 2)
            c1,c2 = rng.sample(range(N), 2)
            A,B,C,D = int(L3_r[r1,c1]),int(L3_r[r1,c2]),int(L3_r[r2,c1]),int(L3_r[r2,c2])
            if A==D and B==C and A!=B:
                L3_r[r1,c1]=B; L3_r[r1,c2]=A; L3_r[r2,c1]=A; L3_r[r2,c2]=B
                break
    E_r = count_clashes(L1, L3_r) + count_clashes(L2, L3_r)
    replicas.append([E_r, L3_r.copy()])

swap_accepted = [0] * (N_REPLICAS - 1)
swap_tried    = [0] * (N_REPLICAS - 1)
round_num = 0
t_start = time.time()

log(f"Initial replica energies: {[r[0] for r in replicas]}")

while True:
    round_num += 1

    # SA on each replica
    new_replicas = []
    for k, (E_k, L3_k) in enumerate(replicas):
        seed_k = rng.randint(1, 2**31-1)
        pc1 = make_pc(L1, L3_k)
        pc2 = make_pc(L2, L3_k)
        E_new, L3_new = run_sa(L1f, L2f, L3_k, pc1, pc2, STEPS_PER_ROUND, TEMPS[k], seed_k)
        new_replicas.append([E_new, L3_new])
    replicas = new_replicas

    # Swaps (both even and odd)
    for phase in [0, 1]:
        for k in range(phase, N_REPLICAS-1, 2):
            E_k, L3_k = replicas[k]
            E_k1, L3_k1 = replicas[k+1]
            T_k, T_k1 = TEMPS[k], TEMPS[k+1]
            delta = (E_k1 - E_k) * (1.0/T_k - 1.0/T_k1)
            swap_tried[k] += 1
            if delta >= 0 or rng.random() < math.exp(delta):
                replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
                swap_accepted[k] += 1

    # Check all replicas for new best — SAVE IMMEDIATELY
    for k, (E_k, L3_k) in enumerate(replicas):
        if E_k < global_best_E:
            global_best_E = E_k
            global_best_L3 = L3_k.copy()
            cl13 = int(count_clashes(L1, L3_k))
            cl23 = int(count_clashes(L2, L3_k))
            log(f"*** NEW BEST (replica {k}) E={global_best_E} cl13={cl13} cl23={cl23} round={round_num} ***")
            BEST_FILE.write_text(json.dumps({
                'E': global_best_E, 'cl13': cl13, 'cl23': cl23,
                'pair_id': PAIR_ID, 'round': round_num, 'variant': VARIANT,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': global_best_L3.tolist()
            }, indent=2))
            git_push(f"pt_agg: {PAIR_ID} E={global_best_E}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
            if global_best_E == 0:
                log("*** 3-MOLS FOUND! ***")
                FOUND_FILE.write_text(json.dumps({
                    'found': True, 'pair_id': PAIR_ID,
                    'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': global_best_L3.tolist(),
                    'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
                }, indent=2))
                sys.exit(0)

    if round_num % N_ROUNDS_LOG == 0:
        elapsed = time.time() - t_start
        energies = [r[0] for r in replicas]
        swap_rates = [f"{swap_accepted[k]/max(swap_tried[k],1)*100:.0f}%" for k in range(N_REPLICAS-1)]
        log(f"Round {round_num:4d} | energies={energies} | best={global_best_E} | "
            f"t={elapsed:.0f}s | swaps={swap_rates}")
