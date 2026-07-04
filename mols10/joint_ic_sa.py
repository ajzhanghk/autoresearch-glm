#!/usr/bin/env python3
"""
Joint IC SA: minimize E_total = cl(L1,L2) + cl(L1,L3) + cl(L2,L3) over ALL THREE squares.

Unlike fixed-pair SA, this allows L1 and L2 to also move via IC moves,
escaping the "fixed L1, L2 orthogonal pair" constraint.

Start from the known E=23 state (L1, L2 fixed, L3 at best known).
Apply IC moves to any of L1, L2, L3 and accept via Metropolis criterion.

E_total = 0 iff (L1, L2, L3) are 3-MOLS(10).
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LOG = REPO / f"mols10/results/joint_ic_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
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

def ic_moves(L):
    """Return list of all valid IC move arguments (r1,r2,c1,c2) for L."""
    moves = []
    for r1 in range(N):
        for r2 in range(r1+1, N):
            for c1 in range(N):
                for c2 in range(c1+1, N):
                    v11=int(L[r1,c1]); v12=int(L[r1,c2])
                    v21=int(L[r2,c1]); v22=int(L[r2,c2])
                    if v11==v22 and v12==v21:
                        moves.append((r1,r2,c1,c2))
    return moves

def apply_ic(L, r1, r2, c1, c2):
    L2 = L.copy()
    L2[r1,c1]=int(L[r1,c2]); L2[r1,c2]=int(L[r1,c1])
    L2[r2,c1]=int(L[r2,c2]); L2[r2,c2]=int(L[r2,c1])
    return L2

def apply_random_ic(L, rng):
    """Apply a random IC move to L. Returns (new_L, success)."""
    # Random tries
    for _ in range(20):
        r1, r2 = rng.sample(range(N), 2)
        c1, c2 = rng.sample(range(N), 2)
        v11=int(L[r1,c1]); v12=int(L[r1,c2]); v21=int(L[r2,c1]); v22=int(L[r2,c2])
        if v11==v22 and v12==v21:
            return apply_ic(L, r1, r2, c1, c2), True
    return L, False

def compute_E(L1, L2, L3):
    """E_total = cl(L1,L2) + cl(L1,L3) + cl(L2,L3)"""
    return count_clashes(L1, L2) + count_clashes(L1, L3) + count_clashes(L2, L3)

# Load starting state from tv_254 best
log("="*70)
log(f"Joint IC SA — instance={INSTANCE}")
log("Minimizing E_total = cl(L1,L2) + cl(L1,L3) + cl(L2,L3)")
log("="*70)

d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L1_init = np.array(d['L1'], dtype=np.int8).reshape(N,N)
L2_init = np.array(d['L2'], dtype=np.int8).reshape(N,N)
L3_init = np.array(d['L3'], dtype=np.int8).reshape(N,N)

E_init = compute_E(L1_init, L2_init, L3_init)
log(f"Starting E_total = {E_init} (cl12={count_clashes(L1_init,L2_init)}, cl13={count_clashes(L1_init,L3_init)}, cl23={count_clashes(L2_init,L3_init)})")

rng = random.Random(141421356 + INSTANCE * 1_000_003)

# Multiple temperature schedules per instance
T_SCHEDULES = [
    (3.0, 0.005, 5_000_000),   # hot → cold
    (5.0, 0.01, 3_000_000),    # very hot
    (2.0, 0.001, 5_000_000),   # moderate
    (1.5, 0.0001, 5_000_000),  # slow cooling
]

L1 = L1_init.copy()
L2 = L2_init.copy()
L3 = L3_init.copy()
current_E = E_init
best_E = E_init
best_state = (L1.copy(), L2.copy(), L3.copy())
t_start = time.time()
step = 0
restart = 0

while True:
    restart += 1
    T0, T_end, N_STEPS = rng.choice(T_SCHEDULES)
    log(f"Restart {restart}: T={T0}→{T_end}, N_steps={N_STEPS}")

    for _ in range(N_STEPS):
        step += 1

        # Choose which square to move (prefer L3 since it matters most for E)
        which = rng.choices([0, 1, 2], weights=[2, 2, 6])[0]

        if which == 0:
            new_L, ok = apply_random_ic(L1, rng)
            if not ok: continue
            # Delta E: cl(L1,L2) and cl(L1,L3) change; cl(L2,L3) unchanged
            old_cl12 = count_clashes(L1, L2)
            old_cl13 = count_clashes(L1, L3)
            new_cl12 = count_clashes(new_L, L2)
            new_cl13 = count_clashes(new_L, L3)
            delta_E = (new_cl12 + new_cl13) - (old_cl12 + old_cl13)
        elif which == 1:
            new_L, ok = apply_random_ic(L2, rng)
            if not ok: continue
            old_cl12 = count_clashes(L1, L2)
            old_cl23 = count_clashes(L2, L3)
            new_cl12 = count_clashes(L1, new_L)
            new_cl23 = count_clashes(new_L, L3)
            delta_E = (new_cl12 + new_cl23) - (old_cl12 + old_cl23)
        else:  # which == 2
            new_L, ok = apply_random_ic(L3, rng)
            if not ok: continue
            old_cl13 = count_clashes(L1, L3)
            old_cl23 = count_clashes(L2, L3)
            new_cl13 = count_clashes(L1, new_L)
            new_cl23 = count_clashes(L2, new_L)
            delta_E = (new_cl13 + new_cl23) - (old_cl13 + old_cl23)

        # SA acceptance
        progress = step / N_STEPS
        T = T0 * (T_end / T0) ** progress

        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
            if which == 0: L1 = new_L.copy()
            elif which == 1: L2 = new_L.copy()
            else: L3 = new_L.copy()
            current_E += delta_E

        if current_E < best_E:
            best_E = current_E
            best_state = (L1.copy(), L2.copy(), L3.copy())
            cl12 = count_clashes(L1, L2)
            cl13 = count_clashes(L1, L3)
            cl23 = count_clashes(L2, L3)
            elapsed = time.time() - t_start
            log(f"Step {step}: NEW BEST E={best_E} (cl12={cl12}, cl13={cl13}, cl23={cl23}) T={T:.4f} t={elapsed:.0f}s")

            if best_E == 0:
                log("*** 3-MOLS(10) FOUND! E=0 ***")
                B1, B2, B3 = best_state
                FOUND_FILE.write_text(json.dumps({
                    'found': True, 'method': 'joint_ic_sa',
                    'L1': B1.tolist(), 'L2': B2.tolist(), 'L3': B3.tolist(),
                    'cl12': int(count_clashes(B1,B2)),
                    'cl13': int(count_clashes(B1,B3)),
                    'cl23': int(count_clashes(B2,B3)),
                    'E': 0,
                }, indent=2))
                git_push("3-MOLS(10) FOUND via joint_ic_sa!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                import os; os._exit(0)

    # End of schedule: log and restart from best state
    elapsed = time.time() - t_start
    cl12 = count_clashes(*best_state[:2])
    cl13 = count_clashes(best_state[0], best_state[2])
    cl23 = count_clashes(best_state[1], best_state[2])
    log(f"Restart {restart} done: best_E={best_E} (cl12={cl12}, cl13={cl13}, cl23={cl23}) t={elapsed:.0f}s")

    # Restart from best state with small perturbation
    L1, L2, L3 = [x.copy() for x in best_state]
    current_E = best_E

    # Occasional large perturbation to escape
    if restart % 5 == 0:
        log(f"Large perturbation at restart {restart}")
        for _ in range(50):
            L3, ok = apply_random_ic(L3, rng)
        current_E = compute_E(L1, L2, L3)
