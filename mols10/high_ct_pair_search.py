#!/usr/bin/env python3
"""
Search for orthogonal pairs (A, B) with many common transversals.

Strategy:
1. Generate random Latin squares A (via SA from cyclic base)
2. Count transversals of A
3. If A has many transversals (>1000), find an orthogonal mate B via CP-SAT
4. Count common transversals of (A, B)
5. If ct(A, B) >= 10, search for L'' completing 3-MOLS

Also tries: SA to find B while maximizing ct(A, B).
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
from ortools.sat.python import cp_model

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else '0'
LOG = REPO / f"mols10/results/high_ct_{INSTANCE}.log"
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

def count_transversals(L, max_count=5000):
    count = [0]
    def bt(row, cols, vals):
        if row == N:
            count[0] += 1
            return count[0] >= max_count
        for col in range(N):
            if col not in cols:
                v = int(L[row,col])
                if v not in vals:
                    cols.add(col); vals.add(v)
                    if bt(row+1, cols, vals): return True
                    cols.discard(col); vals.discard(v)
        return False
    bt(0, set(), set())
    return count[0]

def count_ct(A, B, max_count=200):
    found = [0]
    def bt(row, cols, a_used, b_used):
        if row == N:
            found[0] += 1
            return found[0] >= max_count
        for col in range(N):
            if col not in cols:
                va = int(A[row,col]); vb = int(B[row,col])
                if va not in a_used and vb not in b_used:
                    cols.add(col); a_used.add(va); b_used.add(vb)
                    if bt(row+1, cols, a_used, b_used): return True
                    cols.discard(col); a_used.discard(va); b_used.discard(vb)
        return False
    bt(0, set(), set(), set())
    return found[0]

def enum_ct(A, B):
    result = []
    def bt(row, cols, a_used, b_used, path):
        if row == N:
            result.append(tuple(path))
            return
        for col in range(N):
            if col not in cols:
                va = int(A[row,col]); vb = int(B[row,col])
                if va not in a_used and vb not in b_used:
                    cols.add(col); a_used.add(va); b_used.add(vb)
                    path.append((row,col,va,vb))
                    bt(row+1, cols, a_used, b_used, path)
                    path.pop()
                    cols.discard(col); a_used.discard(va); b_used.discard(vb)
    bt(0, set(), set(), set(), [])
    return result

def make_random_ls(rng):
    base = np.array([[(i+j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    row_p = list(range(N)); rng.shuffle(row_p)
    col_p = list(range(N)); rng.shuffle(col_p)
    sym_p = list(range(N)); rng.shuffle(sym_p)
    L = base[row_p][:, col_p]
    Ln = np.zeros_like(L)
    for i in range(N):
        for j in range(N): Ln[i,j] = sym_p[int(L[i,j])]
    return Ln

def find_mate_cpsat(A, hint_B=None, timeout_s=60):
    """Use CP-SAT to find B with cl(A,B)=0."""
    model = cp_model.CpModel()
    B_vars = [[model.new_int_var(0, N-1, f"B{i}{j}") for j in range(N)] for i in range(N)]

    for i in range(N):
        model.add_all_different(B_vars[i])
        model.add_all_different([B_vars[r][i] for r in range(N)])

    # Orthogonality with A
    Af = A.ravel().astype(int).tolist()
    AB = []
    for i in range(N):
        for j in range(N):
            v = model.new_int_var(0, N*N-1, f"AB{i}{j}")
            model.add(v == Af[i*N+j] * N + B_vars[i][j])
            AB.append(v)
    model.add_all_different(AB)

    if hint_B is not None:
        for i in range(N):
            for j in range(N):
                model.add_hint(B_vars[i][j], int(hint_B[i,j]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = 2
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        B = np.array([[solver.value(B_vars[i][j]) for j in range(N)] for i in range(N)], dtype=np.int8)
        return B
    return None

def search_high_ct_pair_via_sa(A, rng, n_trials=20):
    """Try SA to find B orthogonal to A with max CTs."""
    Af = A.ravel().astype(np.int32)
    best_cl = 999
    best_ct = 0
    best_B = None

    for _ in range(n_trials):
        B_init = make_random_ls(rng)
        pc1 = make_pc(A, B_init)
        pc2 = make_pc(A, B_init)
        Bf = B_init.ravel().astype(np.int32).copy()
        E2, B_out = _run_sa_core(Af, Af, Bf, pc1, pc2, 3_000_000, 2.0, 0.01, rng.randint(1,2**31-1))
        B = B_out.astype(np.int8).reshape(N,N)
        cl = count_clashes(A, B)

        if cl < best_cl or (cl == 0 and best_cl == 0):
            best_cl = cl
            if cl == 0:
                ct = count_ct(A, B)
                if ct > best_ct:
                    best_ct = ct
                    best_B = B.copy()

    return best_cl, best_ct, best_B

log("="*70)
log(f"High-CT Pair Search — instance={INSTANCE}")
log("Strategy: Generate LS with many transversals, find mates with high ct")
log("="*70)

rng = random.Random(271828 + int(INSTANCE) * 314159)
t_start = time.time()
trial = 0
best_ct_global = 0
best_trans_global = 0

while True:
    trial += 1

    # Generate a random Latin square
    A = make_random_ls(rng)

    # Count transversals
    n_trans = count_transversals(A, max_count=5000)

    if n_trans > best_trans_global:
        best_trans_global = n_trans
        elapsed = time.time() - t_start
        log(f"Trial {trial}: new best transversal count={n_trans} t={elapsed:.0f}s")

    # Only search for mate if A has many transversals
    if n_trans >= 500:
        log(f"Trial {trial}: A has {n_trans} transversals, searching for mate...")

        # Try SA to find mate
        best_cl, ct, best_B = search_high_ct_pair_via_sa(A, rng, n_trials=10)

        if best_cl == 0 and ct > best_ct_global:
            best_ct_global = ct
            elapsed = time.time() - t_start
            log(f"  *** NEW BEST ct={ct} trial={trial} t={elapsed:.0f}s ***")

            if ct >= N:
                log(f"  *** ct >= {N}! Checking for 3-MOLS... ***")
                all_cts = enum_ct(A, best_B)
                log(f"  Total CTs: {len(all_cts)}")
                # Try CP-SAT for exact cover
                # ... (simplified: check if N disjoint exist)
                # Build L'' from partition
                cell_to_ct = {}
                for r in range(N):
                    for c in range(N):
                        cell_to_ct[(r,c)] = []
                for idx, t in enumerate(all_cts):
                    for r,c,va,vb in t:
                        cell_to_ct[(r,c)].append(idx)

                ec_model = cp_model.CpModel()
                x = [ec_model.new_bool_var(f"x{i}") for i in range(len(all_cts))]
                for r in range(N):
                    for c in range(N):
                        ec_model.add(sum(x[i] for i in cell_to_ct[(r,c)]) == 1)
                ec_model.add(sum(x) == N)

                ec_solver = cp_model.CpSolver()
                ec_solver.parameters.max_time_in_seconds = 120
                ec_solver.parameters.num_workers = 4
                ec_status = ec_solver.solve(ec_model)

                if ec_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    selected = [i for i in range(len(all_cts)) if ec_solver.value(x[i])]
                    L_pp = np.zeros((N,N), dtype=np.int8)
                    for symbol, idx in enumerate(selected):
                        for r,c,va,vb in all_cts[idx]:
                            L_pp[r,c] = symbol
                    cl_A = count_clashes(A, L_pp)
                    cl_B = count_clashes(best_B, L_pp)
                    log(f"  L'': cl(A,L'')={cl_A}, cl(B,L'')={cl_B}")
                    if cl_A == 0 and cl_B == 0:
                        log("*** 3-MOLS FOUND! ***")
                        FOUND_FILE.write_text(json.dumps({
                            'found': True, 'method': 'high_ct_pair_search',
                            'L1': A.tolist(), 'L2': best_B.tolist(), 'L3': L_pp.tolist(),
                            'cl12': 0, 'cl13': cl_A, 'cl23': cl_B
                        }, indent=2))
                        git_push("MOLS10 FOUND via high_ct_pair_search!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                        import os; os._exit(0)

        elif best_cl == 0:
            log(f"  Found mate with ct={ct} (no improvement)")
        else:
            log(f"  SA couldn't find mate (best_cl={best_cl})")

    if trial % 20 == 0:
        elapsed = time.time() - t_start
        rate = trial / elapsed * 3600
        log(f"Trial {trial} | best_ct={best_ct_global} | best_trans={best_trans_global} | {rate:.0f}/hr | t={elapsed:.0f}s")
