#!/usr/bin/env python3
"""
Investigate L1 (tv_254) for multiple transversal decompositions.
If L1 has 2+ decompositions that are orthogonal, we get 3-MOLS!

Also: enumerate all decompositions of L1 and check all pairs.
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def enum_transversals(L):
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
    return all_t

def find_all_decompositions(trans, max_decomps=200, timeout_per_s=30):
    """Find up to max_decomps transversal decompositions via CP-SAT."""
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []

    for attempt in range(max_decomps):
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(M)]
        for r in range(N):
            for c in range(N):
                model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)
        model.add(sum(x) == N)
        for prev in forbidden:
            model.add(sum(x[i] for i in prev) < N)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_per_s
        solver.parameters.num_workers = 4
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            log(f"  Attempt {attempt+1}: no more decompositions (status={solver.status_name(status)})")
            break

        selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
        decomps.append(selected)
        forbidden.append(selected)
        log(f"  Found decomposition #{len(decomps)}: indices {selected[:3]}...")

    return decomps

def check_orthogonal_pair(d1, d2, trans):
    """Check if two decompositions are orthogonal: |Ti ∩ Sj| = 1 for all i,j."""
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    for i, ti in enumerate(T1):
        for j, sj in enumerate(T2):
            matches = sum(1 for r in range(N) if ti[r] == sj[r])
            if matches != 1:
                return False
    return True

def build_mates(L, d1, d2, trans):
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    B = np.zeros((N,N), dtype=np.int8)
    C = np.zeros((N,N), dtype=np.int8)
    for i, t in enumerate(T1):
        for row, col in enumerate(t):
            B[row, col] = i
    for j, t in enumerate(T2):
        for row, col in enumerate(t):
            C[row, col] = j
    return B, C

# Load L1 and L2 from tv_254
log("Loading tv_254...")
d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L1 = np.array(d['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(d['L2'], dtype=np.int8).reshape(N,N)

log(f"cl(L1, L2) = {count_clashes(L1, L2)}")

log("Enumerating all transversals of L1...")
t_start = time.time()
trans = enum_transversals(L1)
log(f"L1 has {len(trans)} transversals (took {time.time()-t_start:.1f}s)")

log("\nFinding ALL transversal decompositions of L1...")
t_start = time.time()
decomps = find_all_decompositions(trans, max_decomps=200, timeout_per_s=30)
log(f"Found {len(decomps)} decompositions in {time.time()-t_start:.1f}s")

if len(decomps) < 2:
    log("Only 1 (or 0) decompositions found — cannot form 3-MOLS from L1 alone")
    sys.exit(0)

log(f"\nChecking all {len(decomps)*(len(decomps)-1)//2} pairs of decompositions for orthogonality...")
n_orth = 0
for i in range(len(decomps)):
    for j in range(i+1, len(decomps)):
        if check_orthogonal_pair(decomps[i], decomps[j], trans):
            n_orth += 1
            log(f"*** ORTHOGONAL PAIR #{n_orth}: decomps {i} and {j} ***")
            B, C = build_mates(L1, decomps[i], decomps[j], trans)
            cl12 = count_clashes(L1, B)
            cl13 = count_clashes(L1, C)
            cl23 = count_clashes(B, C)
            log(f"  cl(L1,B)={cl12}, cl(L1,C)={cl13}, cl(B,C)={cl23}")
            if cl12 == 0 and cl13 == 0 and cl23 == 0:
                log("*** 3-MOLS(10) FOUND! ***")
                result = {
                    'found': True, 'method': 'l1_double_decomp',
                    'L1': L1.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                    'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
                    'decomp1_idx': list(decomps[i]), 'decomp2_idx': list(decomps[j]),
                }
                FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
                FOUND_FILE.write_text(json.dumps(result, indent=2))

log(f"\nTotal orthogonal pairs: {n_orth} / {len(decomps)*(len(decomps)-1)//2} pairs")
log(f"Total decompositions of L1: {len(decomps)}")
