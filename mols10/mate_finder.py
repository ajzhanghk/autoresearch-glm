#!/usr/bin/env python3
"""
Mate finder: given the best L3 (E=23), find Latin squares L' with cl(L3,L')=0.
If L3 has an orthogonal mate L', then search for L'' with L''⊥L3 AND L''⊥L'.
If found: (L3, L', L'') are 3-MOLS(10)!

Also try: find if L3 already has 2 orthogonal mates from our (L1,L2) pair.
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
LOG = REPO / f"mols10/results/mate_{PAIR_ID}_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE  = REPO / f"mols10/results/escape_best_{PAIR_ID}.json"
BRANCH     = "claude/mols-order-10-search-yfQXK"

SA_STEPS = 5_000_000
T_START  = 2.0
T_END    = 0.01

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add","-A"], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def run_sa_minimize_cl(A, B_init, n_steps, T_start, T_end, seed):
    """Find B to minimize cl(A, B). A is fixed, B is the variable LS."""
    # We use: L1=A, L3=B (variable). L2 not involved.
    # Energy = cl(A, B). Set L2=dummy (use A itself) but only count cl13.
    # Actually, modify to only count cl(A,B):
    # In _run_sa_core: E = cl(L1,L3) + cl(L2,L3). We want only cl(A,B).
    # Workaround: set L2 = A (a Latin square orthogonal to nothing useful),
    # but E will = cl(A,B) + cl(A,B) = 2*cl(A,B). Still minimizes correctly.
    Af = A.ravel().astype(np.int32)
    pc1 = make_pc(A, B_init)
    pc2 = make_pc(A, B_init)  # Same as pc1
    Bf = B_init.ravel().astype(np.int32).copy()
    E2, B_out = _run_sa_core(Af, Af, Bf, pc1, pc2, n_steps, T_start, T_end, seed)
    B_res = B_out.astype(np.int8).reshape(N, N)
    cl_actual = int(count_clashes(A, B_res))
    return cl_actual, B_res

def random_ls(rng):
    """Generate a random valid 10x10 Latin square."""
    L = np.zeros((N, N), dtype=np.int8)
    base = list(range(N))
    for i in range(N):
        row = base.copy()
        rng.shuffle(row)
        L[i] = row
    # Fix columns by swapping within rows
    for j in range(N):
        col = list(L[:, j])
        if len(set(col)) == N: continue
        # Find duplicates and swap
        seen = {}
        for i, v in enumerate(col):
            if v in seen:
                # Find unused value
                unused = [x for x in range(N) if x not in col]
                if unused:
                    L[i, j] = unused[0]
                    col[i] = unused[0]
            else:
                seen[v] = i
    return L  # May not be valid; SA will fix it

def make_random_ls_via_shuffle(rng):
    """Make random LS by row permutations of reduced Latin square."""
    # Start with cyclic LS
    base = np.array([[(i+j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    # Randomly permute rows
    row_perm = list(range(N)); rng.shuffle(row_perm)
    # Randomly permute columns
    col_perm = list(range(N)); rng.shuffle(col_perm)
    # Randomly permute symbols
    sym_perm = list(range(N)); rng.shuffle(sym_perm)
    L = base[row_perm][:, col_perm]
    # Apply symbol permutation
    Ln = np.zeros_like(L)
    for i in range(N):
        for j in range(N):
            Ln[i,j] = sym_perm[int(L[i,j])]
    return Ln

# Load the L3 (the fixed square we're finding mates for)
d = json.loads((REPO/f"mols10/results/pt_best_{PAIR_ID}.json").read_text())
L1 = np.array(d['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(d['L2'], dtype=np.int8).reshape(N,N)
L3_best = np.array(d['L3'], dtype=np.int8).reshape(N,N)
E_cur = d['E']  # = 23

log("="*70)
log(f"Mate finder — pair={PAIR_ID} inst={INSTANCE}")
log(f"Fixed L3 has E={E_cur} (cl13=9, cl23=14)")
log(f"Searching for L' with cl(L3,L')=0 (orthogonal mate)")
log(f"SA: {SA_STEPS//1000}k steps, T=[{T_START},{T_END}]")
log("="*70)

# First check: is L3 already orthogonal to L1 or L2?
cl13 = count_clashes(L1, L3_best)
cl23 = count_clashes(L2, L3_best)
log(f"Current: cl(L1,L3)={cl13}, cl(L2,L3)={cl23}")
log(f"Need to find L' with cl(L3,L')=0")

rng = random.Random(271828182 + hash(PAIR_ID + INSTANCE) % 100000)
t_start = time.time()
trial = 0
best_cl_found = 999
best_mate = None

while True:
    trial += 1

    # Random starting L'
    L_prime_init = make_random_ls_via_shuffle(rng)

    seed_int = rng.randint(1, 2**31-1)
    cl, L_prime = run_sa_minimize_cl(L3_best, L_prime_init, SA_STEPS, T_START, T_END, seed_int)

    if cl < best_cl_found:
        best_cl_found = cl
        best_mate = L_prime.copy()
        elapsed = time.time() - t_start
        log(f"*** NEW BEST MATE: cl(L3,L')={cl} trial={trial} t={elapsed:.0f}s ***")
        if cl == 0:
            log(f"*** FOUND ORTHOGONAL MATE! L3 and L' are orthogonal ***")
            log(f"Now searching for L'' orthogonal to both L3 and L'...")
            # Search for L'' orthogonal to both L3 and L'
            # E = cl(L3,L'') + cl(L',L'') must be 0
            L_prime_f = L_prime.ravel().astype(np.int32)
            L3_f = L3_best.ravel().astype(np.int32)
            for attempt in range(100):
                L_dpp_init = make_random_ls_via_shuffle(rng)
                pc1 = make_pc(L3_best, L_dpp_init)
                pc2 = make_pc(L_prime, L_dpp_init)
                L_dpp_f = L_dpp_init.ravel().astype(np.int32).copy()
                E_dpp, L_dpp_out = _run_sa_core(L3_f, L_prime_f, L_dpp_f, pc1, pc2,
                                                  SA_STEPS, T_START, T_END, rng.randint(1,2**31-1))
                L_dpp = L_dpp_out.astype(np.int8).reshape(N,N)
                E_check = count_clashes(L3_best, L_dpp) + count_clashes(L_prime, L_dpp)
                if E_check != E_dpp: E_dpp = E_check
                log(f"  L'' attempt {attempt+1}: E(L3,L')+E(L',L'')={E_dpp}")
                if E_dpp == 0:
                    log("*** 3-MOLS FOUND via mate search!!! ***")
                    cl_12 = int(count_clashes(L3_best, L_prime))
                    cl_13 = int(count_clashes(L3_best, L_dpp))
                    cl_23 = int(count_clashes(L_prime, L_dpp))
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'mate_finder',
                        'pair_id': PAIR_ID,
                        'L1': L3_best.tolist(), 'L2': L_prime.tolist(), 'L3': L_dpp.tolist(),
                        'cl12': cl_12, 'cl13': cl_13, 'cl23': cl_23
                    }, indent=2))
                    git_push("MOLS10 FOUND via mate_finder!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)

    if trial % 10 == 0:
        elapsed = time.time() - t_start
        rate = trial / elapsed * 3600
        log(f"Trial {trial} | best_cl={best_cl_found} | {rate:.0f}/hr | t={elapsed:.0f}s")
