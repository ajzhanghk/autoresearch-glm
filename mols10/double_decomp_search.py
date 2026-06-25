#!/usr/bin/env python3
"""
Search for Latin squares with two ORTHOGONAL transversal decompositions.

If L has two transversal decompositions P1 = {T1,...,T10} and P2 = {S1,...,S10}
such that |Ti ∩ Sj| = 1 for all i,j, then (L, B, C) form 3-MOLS where:
  B[r][c] = i  iff  (r,c) ∈ Ti
  C[r][c] = j  iff  (r,c) ∈ Sj

Strategy:
1. Start from L1 (tv_254, 872 transversals)
2. Apply IC moves to explore LS with many transversals (>= 500)
3. For each candidate, find ALL transversal decompositions via CP-SAT
4. Check if any two decompositions are orthogonal
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else '0'
LOG = REPO / f"mols10/results/ddecomp_{INSTANCE}.log"
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

def count_transversals_fast(L, max_count=5000):
    """Fast bitmask-based transversal counter."""
    count = [0]
    rows = [[int(L[r,c]) for c in range(N)] for r in range(N)]
    def bt(row, col_mask, val_mask):
        if row == N:
            count[0] += 1
            return count[0] >= max_count
        for col in range(N):
            if not (col_mask >> col & 1):
                v = rows[row][col]
                if not (val_mask >> v & 1):
                    if bt(row+1, col_mask|(1<<col), val_mask|(1<<v)):
                        return True
        return False
    bt(0, 0, 0)
    return count[0]

def enum_transversals(L):
    """Enumerate ALL transversals of L."""
    all_t = []
    rows = [[int(L[r,c]) for c in range(N)] for r in range(N)]
    def bt(row, col_mask, val_mask, path):
        if row == N:
            all_t.append(tuple(path)); return
        for col in range(N):
            if not (col_mask >> col & 1):
                v = rows[row][col]
                if not (val_mask >> v & 1):
                    path.append(col)
                    bt(row+1, col_mask|(1<<col), val_mask|(1<<v), path)
                    path.pop()
    bt(0, 0, 0, [])
    return all_t  # list of tuples: (col for row 0, col for row 1, ..., col for row 9)

def find_decompositions(L, transversals, max_decomps=50, timeout_s=60):
    """
    Find up to max_decomps transversal decompositions of L using CP-SAT.
    Returns list of decompositions, each is a list of 10 transversal indices.
    """
    M = len(transversals)
    if M < N: return []

    # Build cell → list of transversal indices covering that cell
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(transversals):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []

    for _ in range(max_decomps):
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(M)]

        # Each cell covered exactly once
        for r in range(N):
            for c in range(N):
                model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)
        model.add(sum(x) == N)

        # Forbid previously found decompositions
        for prev in forbidden:
            model.add(sum(x[i] for i in prev) < N)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_s
        solver.parameters.num_workers = 2
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
        decomps.append(selected)
        forbidden.append(selected)

        if len(decomps) >= max_decomps:
            break

    return decomps

def check_orthogonal_decomps(t1_indices, t2_indices, transversals):
    """
    Check if two transversal decompositions P1 and P2 are orthogonal:
    |Ti ∩ Sj| = 1 for all i,j.
    Each transversal is a tuple: (col for row 0, ..., col for row 9).
    """
    # Build intersection matrix
    # Ti: set of (row, col) pairs in transversal t1_indices[i]
    # For Ti ∩ Sj = {(r,c)}: transversal t1_indices[i] and t2_indices[j]
    # share exactly one row-position.
    # Since each transversal has exactly one cell per row,
    # Ti ∩ Sj = {(r, Ti[r])} if Ti[r] == Sj[r] for exactly one r.

    T1 = [transversals[i] for i in t1_indices]
    T2 = [transversals[j] for j in t2_indices]

    # intersect_matrix[i][j] = list of rows where Ti and Sj agree
    for i, ti in enumerate(T1):
        for j, sj in enumerate(T2):
            matches = [r for r in range(N) if ti[r] == sj[r]]
            if len(matches) != 1:
                return False, None
    return True, (T1, T2)

def build_mols_from_decomps(L, T1, T2):
    """Build B and C from two orthogonal decompositions, verify 3-MOLS."""
    B = np.zeros((N,N), dtype=np.int8)
    C = np.zeros((N,N), dtype=np.int8)
    for i, t in enumerate(T1):
        for row, col in enumerate(t):
            B[row, col] = i
    for j, t in enumerate(T2):
        for row, col in enumerate(t):
            C[row, col] = j
    cl_AB = count_clashes(L, B)
    cl_AC = count_clashes(L, C)
    cl_BC = count_clashes(B, C)
    return B, C, cl_AB, cl_AC, cl_BC

def apply_ic_move(L, r1, r2, c1, c2):
    """Apply IC move if valid. Returns (new_L, success)."""
    v11 = int(L[r1,c1]); v12 = int(L[r1,c2])
    v21 = int(L[r2,c1]); v22 = int(L[r2,c2])
    if v11 == v22 and v12 == v21:
        L2 = L.copy()
        L2[r1,c1] = v12; L2[r1,c2] = v11
        L2[r2,c1] = v22; L2[r2,c2] = v21
        return L2, True
    return L, False

# Load starting LS from tv_254
d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L_start = np.array(d['L1'], dtype=np.int8).reshape(N,N)

log("="*70)
log(f"Double-Decomp Search — instance={INSTANCE}")
log("Searching for LS with two orthogonal transversal decompositions")
log("="*70)

rng = random.Random(161803398 + int(INSTANCE) * 999983)
current_L = L_start.copy()
current_n_trans = count_transversals_fast(current_L, max_count=5000)
best_n_trans = current_n_trans
best_L = current_L.copy()
log(f"Starting: n_trans={current_n_trans}")

t_start = time.time()
ic_trial = 0
n_candidates = 0
n_found_2plus_decomps = 0
SA_ACCEPT_WORSE = 0.05

TRANS_THRESHOLD = 400  # Only do full decomp search if ≥ this many transversals

while True:
    ic_trial += 1

    r1, r2 = rng.sample(range(N), 2)
    c1, c2 = rng.sample(range(N), 2)
    new_L, changed = apply_ic_move(current_L, r1, r2, c1, c2)

    if not changed:
        continue

    new_n_trans = count_transversals_fast(new_L, max_count=5000)

    # SA acceptance
    if new_n_trans > current_n_trans or rng.random() < SA_ACCEPT_WORSE:
        current_L = new_L.copy()
        current_n_trans = new_n_trans

    if new_n_trans > best_n_trans:
        best_n_trans = new_n_trans
        best_L = new_L.copy()
        elapsed = time.time() - t_start
        log(f"IC trial {ic_trial}: NEW BEST n_trans={new_n_trans} t={elapsed:.0f}s")

    # Periodically search for double decompositions in candidates with many transversals
    if ic_trial % 200 == 0:
        elapsed = time.time() - t_start
        log(f"IC trial {ic_trial}: n_trans={current_n_trans}, best={best_n_trans} t={elapsed:.0f}s")

    # Only do expensive decomp search when we hit a multiple of 300 AND have many transversals
    if ic_trial % 300 == 0 and new_n_trans >= TRANS_THRESHOLD:
        n_candidates += 1
        # Enumerate transversals
        trans = enum_transversals(new_L)
        M = len(trans)

        if M < N: continue

        log(f"  Candidate {n_candidates}: n_trans={M} at IC trial {ic_trial}")

        # Find multiple decompositions
        decomps = find_decompositions(new_L, trans, max_decomps=20, timeout_s=15)
        n_decomps = len(decomps)

        if n_decomps >= 2:
            n_found_2plus_decomps += 1
            log(f"  Found {n_decomps} decompositions! Checking orthogonality...")

            for i in range(n_decomps):
                for j in range(i+1, n_decomps):
                    is_orth, decomp_pair = check_orthogonal_decomps(
                        decomps[i], decomps[j], trans)

                    if is_orth:
                        T1, T2 = decomp_pair
                        B, C, cl_AB, cl_AC, cl_BC = build_mols_from_decomps(new_L, T1, T2)
                        log(f"  *** ORTHOGONAL DECOMPS FOUND! ***")
                        log(f"  cl(L,B)={cl_AB}, cl(L,C)={cl_AC}, cl(B,C)={cl_BC}")

                        if cl_AB == 0 and cl_AC == 0 and cl_BC == 0:
                            log("*** 3-MOLS(10) FOUND! ***")
                            result = {
                                'found': True, 'method': 'double_decomp',
                                'L1': new_L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                                'cl12': int(cl_AB), 'cl13': int(cl_AC), 'cl23': int(cl_BC),
                                'n_trans': M, 'n_decomps': n_decomps,
                            }
                            FOUND_FILE.write_text(json.dumps(result, indent=2))
                            git_push("3-MOLS(10) FOUND via double_decomp!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                            import os; os._exit(0)
                    else:
                        pass  # non-orthogonal pair
        elif n_decomps == 1:
            log(f"  Only 1 decomposition found (need ≥2)")
        else:
            log(f"  No decomposition found (timeout or infeasible)")
