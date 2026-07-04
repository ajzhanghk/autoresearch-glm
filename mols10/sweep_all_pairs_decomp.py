#!/usr/bin/env python3
"""
Systematic sweep: for each L1 in promising_pairs.json,
find all transversal decompositions (up to 20) and check
if any two decompositions are ORTHOGONAL → 3-MOLS(10).

Run as: python3 sweep_all_pairs_decomp.py <instance> <n_instances>
Each instance processes every n_instances-th pair.
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_INSTANCES = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LOG = REPO / f"mols10/results/sweep_decomp_{INSTANCE}.log"
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

def find_decompositions(trans, max_decomps=20, timeout_s=90):
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    decomps = []
    forbidden = []
    proved_complete = False

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
        solver.parameters.num_workers = 2  # 2 workers per process (2 parallel processes)
        status = solver.solve(model)

        if status == cp_model.INFEASIBLE:
            proved_complete = True
            break
        elif status == cp_model.UNKNOWN:
            break
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if attempt == max_decomps:
                break
            selected = tuple(sorted(i for i in range(M) if solver.value(x[i])))
            decomps.append(selected)
            forbidden.append(selected)

    return decomps, proved_complete

def check_orth(d1, d2, trans):
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    for ti in T1:
        for sj in T2:
            if sum(1 for r in range(N) if ti[r] == sj[r]) != 1:
                return False
    return True

def process_pair(pair_id, L1, L2):
    """Check if L1 has multiple orthogonal transversal decompositions."""
    trans = enum_transversals(L1)
    M = len(trans)
    if M < N:
        log(f"  [{pair_id}] Only {M} transversals, skip")
        return None

    t0 = time.time()
    decomps, proved = find_decompositions(trans, max_decomps=20, timeout_s=90)
    elapsed = time.time() - t0

    if len(decomps) < 2:
        log(f"  [{pair_id}] {M} trans, {len(decomps)} decomp(s) in {elapsed:.1f}s (proved={proved})")
        return None

    log(f"  [{pair_id}] {M} trans, {len(decomps)} decomps in {elapsed:.1f}s (proved={proved}) — checking pairs...")

    for i in range(len(decomps)):
        for j in range(i+1, len(decomps)):
            if check_orth(decomps[i], decomps[j], trans):
                T1 = [trans[k] for k in decomps[i]]
                T2 = [trans[k] for k in decomps[j]]
                B = np.zeros((N,N), dtype=np.int8)
                C = np.zeros((N,N), dtype=np.int8)
                for a, t in enumerate(T1):
                    for r, c in enumerate(t): B[r, c] = a
                for b, t in enumerate(T2):
                    for r, c in enumerate(t): C[r, c] = b
                cl_LB = count_clashes(L1, B)
                cl_LC = count_clashes(L1, C)
                cl_BC = count_clashes(B, C)
                log(f"  [{pair_id}] *** ORTHOGONAL DECOMPS ({i},{j}): cl={cl_LB},{cl_LC},{cl_BC} ***")
                if cl_LB == 0 and cl_LC == 0 and cl_BC == 0:
                    log(f"  [{pair_id}] *** 3-MOLS(10) FOUND! ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'pair_id': pair_id,
                        'L1': L1.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': 0, 'cl13': 0, 'cl23': 0,
                        'method': 'sweep_decomp',
                    }, indent=2))
                    git_push("3-MOLS(10) FOUND via sweep_decomp!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)
                return (pair_id, len(decomps), i, j, cl_BC)

    log(f"  [{pair_id}] {len(decomps)} decomps, no orthogonal pair")
    return None

# Load all pairs
log("="*70)
log(f"Sweep Decomp — instance={INSTANCE}/{N_INSTANCES}")
log("Checking all L1's in promising_pairs for orthogonal decompositions")
log("="*70)

pairs_data = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
log(f"Total pairs: {len(pairs_data)}")

# Sort by transversal count (quick count to prioritize)
log("Quick transversal counting for prioritization...")
pair_trans = []
for pdata in pairs_data:
    pair_id = pdata['pair_id']
    L1 = np.array(pdata['L1'], dtype=np.int8).reshape(N,N)
    # Quick count with max 5000
    count = [0]
    rows = [[int(L1[r,c]) for c in range(N)] for r in range(N)]
    def bt_quick(row, cm, vm):
        if row == N:
            count[0] += 1; return count[0] >= 5000
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    if bt_quick(row+1, cm|(1<<col), vm|(1<<v)): return True
        return False
    bt_quick(0, 0, 0)
    pair_trans.append((count[0], pair_id, pdata))

pair_trans.sort(reverse=True)  # highest transversal count first
log(f"Top 5 pairs by transversal count: {[(n,p) for n,p,_ in pair_trans[:5]]}")

# Process every N_INSTANCES-th pair (this instance handles INSTANCE, INSTANCE+N_INSTANCES, ...)
my_pairs = pair_trans[INSTANCE::N_INSTANCES]
log(f"This instance handles {len(my_pairs)} pairs")

# Resume: skip pairs already fully processed (logged with a result line)
done_ids = set()
if LOG.exists():
    for line in LOG.read_text().splitlines():
        # a pair is done if we logged its decomp-count result
        if "] [" in line and (" decomp(s) in " in line or " decomps in " in line or " decomps, no orthogonal" in line or "Only " in line):
            try:
                pid = line.split("] [",1)[1].split("]",1)[0]
                done_ids.add(pid)
            except Exception:
                pass
my_pairs = [(n, p, d) for (n, p, d) in my_pairs if p not in done_ids]
log(f"Resume: {len(done_ids)} already done, {len(my_pairs)} remaining")

t_global = time.time()
for idx, (n_trans, pair_id, pdata) in enumerate(my_pairs):
    L1 = np.array(pdata['L1'], dtype=np.int8).reshape(N,N)
    L2 = np.array(pdata['L2'], dtype=np.int8).reshape(N,N)
    elapsed = time.time() - t_global
    log(f"[{idx+1}/{len(my_pairs)}] Processing {pair_id} (n_trans={n_trans}) t={elapsed:.0f}s")
    process_pair(pair_id, L1, L2)

log(f"Sweep complete! Processed {len(my_pairs)} pairs")
