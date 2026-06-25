#!/usr/bin/env python3
"""
Deep search for ALL transversal decompositions of L1 (tv_254).
Uses 120s timeout per attempt to find as many as possible.
Then checks all pairs for orthogonality.
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

def find_decompositions_deep(trans, max_decomps=500, timeout_per_s=120):
    """Find up to max_decomps transversal decompositions, proving exhaustiveness."""
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []
    proved_complete = False

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
        solver.parameters.num_workers = 6
        status = solver.solve(model)

        if status == cp_model.INFEASIBLE:
            log(f"  Proved INFEASIBLE: exactly {len(decomps)} decompositions total")
            proved_complete = True
            break
        elif status == cp_model.UNKNOWN:
            log(f"  UNKNOWN at attempt {attempt+1}: at least {len(decomps)} decompositions (timeout)")
            break
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
            decomps.append(selected)
            forbidden.append(selected)
            log(f"  Found decomposition #{len(decomps)}: starts with {selected[:4]}...")
        else:
            log(f"  Unexpected status: {solver.status_name(status)}")
            break

    return decomps, proved_complete

def check_all_pairs(decomps, trans):
    """Check all pairs of decompositions for orthogonality."""
    n = len(decomps)
    T = [trans[i] for i in range(len(trans))]

    orthogonal_pairs = []
    for i in range(n):
        T1 = [trans[decomps[i][k]] for k in range(N)]
        for j in range(i+1, n):
            T2 = [trans[decomps[j][k]] for k in range(N)]

            is_orth = True
            for a in range(N):
                for b in range(N):
                    matches = sum(1 for r in range(N) if T1[a][r] == T2[b][r])
                    if matches != 1:
                        is_orth = False
                        break
                if not is_orth: break

            if is_orth:
                orthogonal_pairs.append((i, j, T1, T2))
                log(f"  *** ORTHOGONAL PAIR: decomps {i} and {j} ***")

    return orthogonal_pairs

# Load L1
log("Loading tv_254 L1...")
d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L1 = np.array(d['L1'], dtype=np.int8).reshape(N,N)

log("Enumerating all transversals of L1...")
t0 = time.time()
trans = enum_transversals(L1)
log(f"L1 has {len(trans)} transversals (took {time.time()-t0:.2f}s)")

log(f"\nSearching for ALL transversal decompositions (120s timeout each)...")
t0 = time.time()
decomps, proved_complete = find_decompositions_deep(trans, max_decomps=500, timeout_per_s=120)
log(f"Found {len(decomps)} decompositions in {time.time()-t0:.1f}s (complete={proved_complete})")

if len(decomps) < 2:
    log("Not enough decompositions for orthogonality check")
    sys.exit(0)

log(f"\nChecking all {len(decomps)*(len(decomps)-1)//2} pairs for orthogonality...")
t0 = time.time()
orth_pairs = check_all_pairs(decomps, trans)
log(f"Done in {time.time()-t0:.2f}s. Found {len(orth_pairs)} orthogonal pairs.")

if orth_pairs:
    for i, j, T1, T2 in orth_pairs:
        B = np.zeros((N,N), dtype=np.int8)
        C = np.zeros((N,N), dtype=np.int8)
        for a, t in enumerate(T1):
            for row, col in enumerate(t): B[row, col] = a
        for b, t in enumerate(T2):
            for row, col in enumerate(t): C[row, col] = b
        cl_LB = count_clashes(L1, B)
        cl_LC = count_clashes(L1, C)
        cl_BC = count_clashes(B, C)
        log(f"cl(L1,B)={cl_LB}, cl(L1,C)={cl_LC}, cl(B,C)={cl_BC}")
        if cl_LB == 0 and cl_LC == 0 and cl_BC == 0:
            log("*** 3-MOLS(10) FOUND! ***")
            import json
            (REPO/"mols10/results/MOLS10_FOUND.json").write_text(json.dumps({
                'found': True, 'L1': L1.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                'cl12': 0, 'cl13': 0, 'cl23': 0, 'method': 'l1_deep_decomp',
            }, indent=2))
