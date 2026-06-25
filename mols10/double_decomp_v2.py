#!/usr/bin/env python3
"""
Search for Latin squares with two ORTHOGONAL transversal decompositions.

Strategy:
- IC moves on starting LS to maximize transversal count
- When new best_L found: enumerate all transversals, find up to 20 decompositions (60s each)
- Check if any two decompositions are orthogonal → 3-MOLS!

Also runs from different starting points per instance.
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LOG = REPO / f"mols10/results/ddecomp2_{INSTANCE}.log"
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

def count_transversals_fast(L, max_count=3000):
    count = [0]
    rows = [[int(L[r,c]) for c in range(N)] for r in range(N)]
    def bt(row, cm, vm):
        if row == N:
            count[0] += 1; return count[0] >= max_count
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    if bt(row+1, cm|(1<<col), vm|(1<<v)): return True
        return False
    bt(0, 0, 0)
    return count[0]

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

def find_decompositions(trans, max_decomps=20, timeout_s=60, n_workers=4):
    """Find up to max_decomps decompositions. Returns (list, proved_exhaustive)."""
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []

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
        solver.parameters.num_workers = n_workers
        status = solver.solve(model)

        if status == cp_model.INFEASIBLE:
            return decomps, True  # proved exhaustive
        elif status == cp_model.UNKNOWN:
            return decomps, False  # timed out, might have more
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if attempt == max_decomps:
                return decomps, False  # found max_decomps, stop
            selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
            decomps.append(selected)
            forbidden.append(selected)
        else:
            break

    return decomps, False

def check_orthogonal_decomps(d1, d2, trans):
    """True if two decompositions P1, P2 have |Ti ∩ Sj| = 1 for all i,j."""
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    for ti in T1:
        for sj in T2:
            matches = sum(1 for r in range(N) if ti[r] == sj[r])
            if matches != 1:
                return False
    return True

def build_and_verify(L, d1, d2, trans):
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    B = np.zeros((N,N), dtype=np.int8)
    C = np.zeros((N,N), dtype=np.int8)
    for i, t in enumerate(T1):
        for r, c in enumerate(t): B[r, c] = i
    for j, t in enumerate(T2):
        for r, c in enumerate(t): C[r, c] = j
    return B, C, count_clashes(L, B), count_clashes(L, C), count_clashes(B, C)

def apply_ic_move(L, r1, r2, c1, c2):
    v11=int(L[r1,c1]); v12=int(L[r1,c2]); v21=int(L[r2,c1]); v22=int(L[r2,c2])
    if v11==v22 and v12==v21:
        L2=L.copy(); L2[r1,c1]=v12; L2[r1,c2]=v11; L2[r2,c1]=v22; L2[r2,c2]=v21
        return L2, True
    return L, False

def search_decompositions(candidate_L, label):
    """Find all decompositions of candidate_L, check orthogonality."""
    log(f"  [{label}] Enumerating transversals...")
    trans = enum_transversals(candidate_L)
    M = len(trans)
    log(f"  [{label}] {M} transversals. Finding decompositions (60s each)...")

    if M < N:
        log(f"  [{label}] Too few transversals ({M} < {N}), skipping")
        return None

    t0 = time.time()
    decomps, proved = find_decompositions(trans, max_decomps=20, timeout_s=60, n_workers=4)
    log(f"  [{label}] Found {len(decomps)} decompositions in {time.time()-t0:.1f}s (proved={proved})")

    if len(decomps) < 2:
        return None

    # Check all pairs for orthogonality
    for i in range(len(decomps)):
        for j in range(i+1, len(decomps)):
            if check_orthogonal_decomps(decomps[i], decomps[j], trans):
                log(f"  [{label}] *** ORTHOGONAL DECOMPS: pair ({i},{j}) ***")
                B, C, cl_LB, cl_LC, cl_BC = build_and_verify(candidate_L, decomps[i], decomps[j], trans)
                log(f"  [{label}] cl(L,B)={cl_LB}, cl(L,C)={cl_LC}, cl(B,C)={cl_BC}")
                if cl_LB == 0 and cl_LC == 0 and cl_BC == 0:
                    log(f"  [{label}] *** 3-MOLS(10) FOUND! ***")
                    result = {
                        'found': True, 'method': 'double_decomp_v2',
                        'L1': candidate_L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': int(cl_LB), 'cl13': int(cl_LC), 'cl23': int(cl_BC),
                        'label': label, 'n_trans': M, 'n_decomps': len(decomps),
                    }
                    FOUND_FILE.write_text(json.dumps(result, indent=2))
                    git_push("3-MOLS(10) FOUND via double_decomp_v2!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)
                return {'cl_BC': int(cl_BC), 'n_decomps': len(decomps)}

    log(f"  [{label}] {len(decomps)} decompositions, no orthogonal pair found")
    return None

# Choose starting LS based on instance
log("="*70)
log(f"Double-Decomp V2 Search — instance={INSTANCE}")

d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L1 = np.array(d['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(d['L2'], dtype=np.int8).reshape(N,N)

# Different starting points per instance
# Instance 0,1: start from L1 (different IC seeds)
# Instance 2,3: start from L2 (different IC seeds)
if INSTANCE in (0, 1):
    start_L = L1.copy()
    log(f"Starting from L1 (tv_254)")
else:
    start_L = L2.copy()
    log(f"Starting from L2 (tv_254)")
log("="*70)

rng = random.Random(271828182 + INSTANCE * 1_000_003)

# First check the starting LS itself
start_n_trans = count_transversals_fast(start_L)
log(f"Starting LS has {start_n_trans} transversals")
log(f"Checking starting LS for multiple decompositions...")
search_decompositions(start_L, f"start_inst{INSTANCE}")

current_L = start_L.copy()
current_n_trans = start_n_trans
best_n_trans = start_n_trans
best_L = start_L.copy()
last_searched_n_trans = start_n_trans

t_start = time.time()
ic_trial = 0
SA_ACCEPT_WORSE = 0.05

while True:
    ic_trial += 1
    r1, r2 = rng.sample(range(N), 2)
    c1, c2 = rng.sample(range(N), 2)
    new_L, changed = apply_ic_move(current_L, r1, r2, c1, c2)
    if not changed: continue

    new_n_trans = count_transversals_fast(new_L, max_count=3000)

    if new_n_trans > current_n_trans or rng.random() < SA_ACCEPT_WORSE:
        current_L = new_L.copy()
        current_n_trans = new_n_trans

    if new_n_trans > best_n_trans:
        best_n_trans = new_n_trans
        best_L = new_L.copy()
        elapsed = time.time() - t_start
        log(f"IC trial {ic_trial}: NEW BEST n_trans={new_n_trans} t={elapsed:.0f}s")

        # Check best_L when it improves significantly
        if best_n_trans > last_searched_n_trans + 20:
            last_searched_n_trans = best_n_trans
            log(f"  Searching decompositions for best_L (n_trans={best_n_trans})...")
            search_decompositions(best_L, f"best_{best_n_trans}_ic{ic_trial}")

    if ic_trial % 500 == 0:
        elapsed = time.time() - t_start
        rate = ic_trial / elapsed * 3600
        log(f"IC trial {ic_trial}: n_trans={current_n_trans}, best={best_n_trans} t={elapsed:.0f}s ({rate:.0f}/hr)")
