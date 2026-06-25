#!/usr/bin/env python3
"""
ILS Mate Finder: ILS-style search for orthogonal mate of L3.

Instead of random restarts, uses ILS acceptance:
- Start from best mate found so far (or random)
- Apply escape moves (row/col swap, symbol relabel) to current mate
- SA to minimize cl(L3, L')
- Accept if cl <= best_cl + DELTA

Also saves best mate to file for cross-process sharing.
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
INSTANCE = sys.argv[2] if len(sys.argv) > 2 else '1'
DELTA    = int(sys.argv[3]) if len(sys.argv) > 3 else 5

LOG       = REPO / f"mols10/results/ils_mate_{PAIR_ID}_{INSTANCE}.log"
MATE_BEST = REPO / f"mols10/results/mate_best_{PAIR_ID}.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

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
    """Find B minimizing cl(A, B). Use A as both L1 and L2 in _run_sa_core."""
    Af = A.ravel().astype(np.int32)
    pc1 = make_pc(A, B_init)
    pc2 = make_pc(A, B_init)
    Bf = B_init.ravel().astype(np.int32).copy()
    E2, B_out = _run_sa_core(Af, Af, Bf, pc1, pc2, n_steps, T_start, T_end, seed)
    B_res = B_out.astype(np.int8).reshape(N, N)
    cl_actual = int(count_clashes(A, B_res))
    return cl_actual, B_res

def make_random_ls_via_shuffle(rng):
    base = np.array([[(i+j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    row_perm = list(range(N)); rng.shuffle(row_perm)
    col_perm = list(range(N)); rng.shuffle(col_perm)
    sym_perm = list(range(N)); rng.shuffle(sym_perm)
    L = base[row_perm][:, col_perm]
    Ln = np.zeros_like(L)
    for i in range(N):
        for j in range(N):
            Ln[i,j] = sym_perm[int(L[i,j])]
    return Ln

def escape_mate(L_prime, rng):
    """Apply 1-3 random non-IC moves to L'."""
    n_moves = rng.choices([1,2,3], weights=[40,40,20])[0]
    Lc = L_prime.copy()
    for _ in range(n_moves):
        mt = rng.random()
        if mt < 0.25:
            r1, r2 = rng.sample(range(N), 2)
            Lc[[r1,r2]] = Lc[[r2,r1]]
        elif mt < 0.50:
            c1, c2 = rng.sample(range(N), 2)
            Lc[:,[c1,c2]] = Lc[:,[c2,c1]]
        elif mt < 0.70:
            perm = list(range(N)); rng.shuffle(perm)
            Ln = np.zeros_like(Lc)
            for i in range(N):
                for j in range(N):
                    Ln[i,j] = perm[int(Lc[i,j])]
            Lc = Ln
        elif mt < 0.85:
            perm = list(range(N)); rng.shuffle(perm)
            Lc = Lc[np.array(perm)]
        else:
            perm = list(range(N)); rng.shuffle(perm)
            Lc = Lc[:, np.array(perm)]
    return Lc

# Load L3 (the square we're finding mates for)
d = json.loads((REPO/f"mols10/results/pt_best_{PAIR_ID}.json").read_text())
L1 = np.array(d['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(d['L2'], dtype=np.int8).reshape(N,N)
L3_fixed = np.array(d['L3'], dtype=np.int8).reshape(N,N)
E_fixed = d['E']

log("="*70)
log(f"ILS Mate Finder — pair={PAIR_ID} inst={INSTANCE} delta={DELTA}")
log(f"Fixed L3 has E={E_fixed}")
log(f"SA: {SA_STEPS//1000}k steps, T=[{T_START},{T_END}]  accept if cl<=best+{DELTA}")
log("="*70)

# Load best mate from file if available
rng = random.Random(271828182 + hash(PAIR_ID + INSTANCE + "ils") % 100000)
best_cl = 999
best_mate = None
current_mate = None
current_cl = 999

if MATE_BEST.exists():
    try:
        mb = json.loads(MATE_BEST.read_text())
        if mb.get('pair_id') == PAIR_ID and 'L_prime' in mb:
            best_mate = np.array(mb['L_prime'], dtype=np.int8).reshape(N,N)
            best_cl = mb['cl']
            current_mate = best_mate.copy()
            current_cl = best_cl
            log(f"Loaded existing best mate: cl={best_cl}")
    except Exception as e:
        log(f"Could not load mate file: {e}")

if best_mate is None:
    # Start with random
    current_mate = make_random_ls_via_shuffle(rng)
    log("Starting from random Latin square")

t_start = time.time()
trial = 0
accepts = 0

while True:
    trial += 1

    # ILS escape from current_mate
    if current_mate is not None and trial > 1:
        L_prime_init = escape_mate(current_mate, rng)
    else:
        L_prime_init = make_random_ls_via_shuffle(rng)

    seed_int = rng.randint(1, 2**31-1)
    cl, L_prime = run_sa_minimize_cl(L3_fixed, L_prime_init, SA_STEPS, T_START, T_END, seed_int)

    # Update global best
    if cl < best_cl:
        best_cl = cl
        best_mate = L_prime.copy()
        elapsed = time.time() - t_start
        log(f"*** NEW BEST MATE: cl(L3,L')={cl} trial={trial} t={elapsed:.0f}s ***")
        # Save to file for cross-process sharing
        MATE_BEST.write_text(json.dumps({
            'pair_id': PAIR_ID, 'cl': int(cl), 'trial': trial,
            'L_prime': L_prime.tolist()
        }, indent=2))
        if cl == 0:
            log("*** FOUND ORTHOGONAL MATE! Searching for L'' to complete 3-MOLS ***")
            L_prime_f = L_prime.ravel().astype(np.int32)
            L3_f = L3_fixed.ravel().astype(np.int32)
            for attempt in range(200):
                L_dpp_init = make_random_ls_via_shuffle(rng)
                pc1 = make_pc(L3_fixed, L_dpp_init)
                pc2 = make_pc(L_prime, L_dpp_init)
                L_dpp_f = L_dpp_init.ravel().astype(np.int32).copy()
                E_dpp, L_dpp_out = _run_sa_core(L3_f, L_prime_f, L_dpp_f, pc1, pc2,
                                                  SA_STEPS, T_START, T_END, rng.randint(1,2**31-1))
                L_dpp = L_dpp_out.astype(np.int8).reshape(N,N)
                E_check = count_clashes(L3_fixed, L_dpp) + count_clashes(L_prime, L_dpp)
                if E_check != E_dpp: E_dpp = E_check
                log(f"  L'' attempt {attempt+1}: E(L3,L'')+E(L',L'')={E_dpp}")
                if E_dpp == 0:
                    log("*** 3-MOLS FOUND via ILS mate finder!!! ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'ils_mate_finder',
                        'pair_id': PAIR_ID,
                        'L1': L3_fixed.tolist(), 'L2': L_prime.tolist(), 'L3': L_dpp.tolist(),
                        'cl12': 0, 'cl13': int(count_clashes(L3_fixed, L_dpp)),
                        'cl23': int(count_clashes(L_prime, L_dpp))
                    }, indent=2))
                    git_push("MOLS10 FOUND via ils_mate_finder!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)

    # ILS acceptance
    if cl <= best_cl + DELTA:
        current_mate = L_prime.copy()
        current_cl = cl
        accepts += 1

    if trial % 10 == 0:
        elapsed = time.time() - t_start
        rate = trial / elapsed * 3600
        accept_rate = accepts / trial * 100
        log(f"Trial {trial} | best_cl={best_cl} | current_cl={current_cl} | "
            f"{rate:.0f}/hr | accept={accept_rate:.0f}% | t={elapsed:.0f}s")
