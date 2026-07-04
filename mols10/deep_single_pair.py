#!/usr/bin/env python3
"""
Deep investigation of a specific pair's L1 for orthogonal transversal decompositions.
Uses unlimited decomposition search with long timeout.

Usage: python3 deep_single_pair.py <pair_id>
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_94'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def enum_transversals(L):
    all_t = []
    rows = [[int(L[r,c]) for c in range(N)] for r in range(N)]
    def bt(row, cm, vm, path):
        if row == N:
            all_t.append(tuple(path)); return
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    path.append(col)
                    bt(row+1, cm|(1<<col), vm|(1<<v), path)
                    path.pop()
    bt(0, 0, 0, [])
    return all_t

def find_all_decomps(trans, timeout_s=180, max_decomps=200):
    """Find ALL decompositions with long timeout."""
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []
    proved = False

    for attempt in range(max_decomps + 1):
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(M)]
        for r in range(N):
            for c in range(N):
                model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)
        model.add(sum(x) == N)
        for prev in forbidden:
            model.add(sum(x[i] for i in prev) < N)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_s
        solver.parameters.num_workers = 8
        status = solver.solve(model)

        if status == cp_model.INFEASIBLE:
            proved = True
            log(f"  Proved INFEASIBLE: exactly {len(decomps)} decompositions")
            break
        elif status == cp_model.UNKNOWN:
            log(f"  UNKNOWN at attempt {attempt+1}: at least {len(decomps)} decomps (timed out)")
            break
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if attempt == max_decomps:
                log(f"  Reached max_decomps={max_decomps}")
                break
            selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
            decomps.append(selected)
            forbidden.append(selected)
            log(f"  Found decomp #{len(decomps)}")

    return decomps, proved

def check_orth(d1, d2, trans):
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    for ti in T1:
        for sj in T2:
            if sum(1 for r in range(N) if ti[r] == sj[r]) != 1:
                return False
    return True

# Load specific pair
log(f"Loading pair: {PAIR_ID}")
pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair = next((p for p in pairs if p['pair_id'] == PAIR_ID), None)
if pair is None:
    log(f"Pair {PAIR_ID} not found!")
    sys.exit(1)

L1 = np.array(pair['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(pair['L2'], dtype=np.int8).reshape(N,N)
log(f"cl(L1,L2) = {count_clashes(L1,L2)}")

log("Enumerating transversals of L1...")
t0 = time.time()
trans = enum_transversals(L1)
log(f"L1 has {len(trans)} transversals (took {time.time()-t0:.2f}s)")

log(f"\nFinding ALL decompositions of L1 (180s timeout each)...")
t0 = time.time()
decomps, proved = find_all_decomps(trans, timeout_s=180, max_decomps=200)
log(f"Found {len(decomps)} decompositions in {time.time()-t0:.1f}s (proved={proved})")

if len(decomps) < 2:
    log("Not enough decompositions")
    sys.exit(0)

log(f"\nChecking all {len(decomps)*(len(decomps)-1)//2} pairs for orthogonality...")
n_orth = 0
for i in range(len(decomps)):
    for j in range(i+1, len(decomps)):
        if check_orth(decomps[i], decomps[j], trans):
            n_orth += 1
            log(f"*** ORTHOGONAL: decomps {i} and {j} ***")
            T1l = [trans[k] for k in decomps[i]]
            T2l = [trans[k] for k in decomps[j]]
            B = np.zeros((N,N), dtype=np.int8)
            C = np.zeros((N,N), dtype=np.int8)
            for a, t in enumerate(T1l):
                for r, c in enumerate(t): B[r, c] = a
            for b, t in enumerate(T2l):
                for r, c in enumerate(t): C[r, c] = b
            cl12=count_clashes(L1,B); cl13=count_clashes(L1,C); cl23=count_clashes(B,C)
            log(f"  cl(L1,B)={cl12}, cl(L1,C)={cl13}, cl(B,C)={cl23}")
            if cl12==0 and cl13==0 and cl23==0:
                log("*** 3-MOLS(10) FOUND! ***")
                (REPO/"mols10/results/MOLS10_FOUND.json").write_text(json.dumps({
                    'found': True, 'L1': L1.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                    'cl12': 0, 'cl13': 0, 'cl23': 0, 'method': 'deep_single_pair',
                    'pair_id': PAIR_ID,
                }, indent=2))

log(f"\nSummary: {len(decomps)} decompositions, {n_orth} orthogonal pairs (proved={proved})")
