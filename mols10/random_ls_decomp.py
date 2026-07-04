#!/usr/bin/env python3
"""
Generate truly random Latin squares of order 10 via random backtracking,
then check each for multiple orthogonal transversal decompositions.

This explores LS space outside the isotopy classes of our known 324 pairs.

3-MOLS(10) exists iff some LS has 2 orthogonal transversal decompositions.
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
LOG = REPO / f"mols10/results/random_ls_{INSTANCE}.log"
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

def generate_random_ls(rng):
    """Generate a random Latin square of order N by backtracking."""
    grid = [[None]*N for _ in range(N)]
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]

    def fill(pos):
        if pos == N*N:
            return True
        r, c = pos // N, pos % N
        vals = list(range(N))
        rng.shuffle(vals)
        for v in vals:
            if v not in row_used[r] and v not in col_used[c]:
                grid[r][c] = v
                row_used[r].add(v)
                col_used[c].add(v)
                if fill(pos+1):
                    return True
                row_used[r].discard(v)
                col_used[c].discard(v)
                grid[r][c] = None
        return False

    if fill(0):
        return np.array(grid, dtype=np.int8)
    return None

def count_transversals_fast(L, max_count=2000):
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

def find_two_decompositions(trans, timeout_s=60):
    """Find up to 2 decompositions. Returns (d1, d2) or (d1, None) or (None, None)."""
    M = len(trans)
    if M < N: return None, None

    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for row, col in enumerate(t):
            cell_to_t[row][col].append(i)

    # Find first decomposition
    def solve(forbidden=[]):
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
        solver.parameters.num_workers = 2
        status = solver.solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return tuple(sorted(i for i in range(M) if solver.value(x[i])))
        return None

    d1 = solve([])
    if d1 is None: return None, None

    d2 = solve([d1])
    return d1, d2

def check_orth(d1, d2, trans):
    T1 = [trans[i] for i in d1]
    T2 = [trans[j] for j in d2]
    for ti in T1:
        for sj in T2:
            if sum(1 for r in range(N) if ti[r] == sj[r]) != 1:
                return False
    return True

log("="*70)
log(f"Random LS Decomp Search — instance={INSTANCE}")
log("Generating random LS, checking for orthogonal transversal decompositions")
log("="*70)

rng = random.Random(577215664 + INSTANCE * 1_000_003)
t_start = time.time()
n_generated = 0
n_had_decomp = 0
n_had_two_decomps = 0
n_orth = 0

MIN_TRANS = 300  # Skip LS with few transversals

while True:
    # Generate random LS
    L = generate_random_ls(rng)
    if L is None:
        continue

    n_generated += 1

    # Quick transversal count
    n_trans = count_transversals_fast(L, max_count=MIN_TRANS)
    if n_trans < MIN_TRANS:
        if n_generated % 1000 == 0:
            elapsed = time.time() - t_start
            rate = n_generated / elapsed * 3600
            log(f"Generated {n_generated}: {n_had_two_decomps} had 2+ decomps, {n_orth} orthogonal pairs | {rate:.0f}/hr")
        continue

    # Has many transversals - enumerate and find decompositions
    trans = enum_transversals(L)
    n_trans_full = len(trans)

    d1, d2 = find_two_decompositions(trans, timeout_s=60)

    if d1 is not None:
        n_had_decomp += 1
        if d2 is not None:
            n_had_two_decomps += 1
            elapsed = time.time() - t_start
            log(f"Gen {n_generated}: n_trans={n_trans_full}, 2+ decomps! Checking orthogonality... t={elapsed:.0f}s")

            if check_orth(d1, d2, trans):
                n_orth += 1
                log(f"*** ORTHOGONAL DECOMPOSITIONS FOUND! ***")
                T1 = [trans[i] for i in d1]
                T2 = [trans[j] for j in d2]
                B = np.zeros((N,N), dtype=np.int8)
                C = np.zeros((N,N), dtype=np.int8)
                for a, t in enumerate(T1):
                    for r, c in enumerate(t): B[r, c] = a
                for b, t in enumerate(T2):
                    for r, c in enumerate(t): C[r, c] = b
                cl12 = count_clashes(L, B)
                cl13 = count_clashes(L, C)
                cl23 = count_clashes(B, C)
                log(f"cl(L,B)={cl12}, cl(L,C)={cl13}, cl(B,C)={cl23}")
                if cl12==0 and cl13==0 and cl23==0:
                    log("*** 3-MOLS(10) FOUND! ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'random_ls_decomp',
                        'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
                    }, indent=2))
                    git_push("3-MOLS(10) FOUND via random_ls_decomp!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)
            else:
                log(f"  Non-orthogonal (n_trans={n_trans_full})")
        else:
            log(f"Gen {n_generated}: n_trans={n_trans_full}, only 1 decomp")
    else:
        if n_generated % 100 == 0:
            log(f"Gen {n_generated}: n_trans={n_trans_full}, no decomp in 60s")
