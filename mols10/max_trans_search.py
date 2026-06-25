#!/usr/bin/env python3
"""
Search for LS with maximum transversal count using IC moves.
Then find orthogonal mates and check common transversals.

The intuition: LS with more transversals might yield pairs with more CTs.
Uses IC moves (not isotopy) to explore the Latin square space.
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
LOG = REPO / f"mols10/results/max_trans_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def count_transversals(L, max_count=10000):
    ct = [0]
    def bt(row, cols, vals):
        if row == N:
            ct[0] += 1; return ct[0] >= max_count
        for col in range(N):
            if col not in cols:
                v = int(L[row,col])
                if v not in vals:
                    cols.add(col); vals.add(v)
                    if bt(row+1, cols, vals): return True
                    cols.discard(col); vals.discard(v)
        return False
    bt(0, set(), set())
    return ct[0]

def count_ct(A, B, max_count=1000):
    found = [0]
    def bt(row, cols, a_used, b_used):
        if row == N:
            found[0] += 1; return found[0] >= max_count
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

def apply_ic_move(L, r1, r2, c1, c2, rng):
    """Apply random IC move: find valid 2x2 intercalate and swap."""
    L2 = L.copy()
    v11 = int(L[r1,c1]); v12 = int(L[r1,c2])
    v21 = int(L[r2,c1]); v22 = int(L[r2,c2])
    if v11 == v22 and v12 == v21:
        L2[r1,c1] = v12; L2[r1,c2] = v11
        L2[r2,c1] = v22; L2[r2,c2] = v21
        return L2, True
    return L, False

def find_mate_cpsat(A, timeout_s=60):
    """Find orthogonal mate of A via CP-SAT transversal decomposition."""
    all_trans = []
    def enum_t(row, cols, vals, path):
        if row == N:
            all_trans.append(tuple(path)); return
        for col in range(N):
            if col not in cols:
                v = int(A[row,col])
                if v not in vals:
                    cols.add(col); vals.add(v)
                    path.append((row,col,v))
                    enum_t(row+1, cols, vals, path)
                    path.pop()
                    cols.discard(col); vals.discard(v)
    enum_t(0, set(), set(), [])
    M = len(all_trans)
    if M == 0: return None, 0

    cell_to_t = {}
    for r in range(N):
        for c in range(N): cell_to_t[(r,c)] = []
    for i, t in enumerate(all_trans):
        for r,c,v in t: cell_to_t[(r,c)].append(i)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    for r in range(N):
        for c in range(N):
            model.add(sum(x[i] for i in cell_to_t[(r,c)]) == 1)
    model.add(sum(x) == N)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = 4
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [i for i in range(M) if solver.value(x[i])]
        B = np.zeros((N,N), dtype=np.int8)
        for symbol, tidx in enumerate(selected):
            for r,c,v in all_trans[tidx]: B[r,c] = symbol
        return B, M
    return None, M

# Load starting LS from tv_254
d = json.loads((REPO/"mols10/results/pt_best_tv_254.json").read_text())
L_start = np.array(d['L1'], dtype=np.int8).reshape(N,N)  # Use L1 as starting point

log("="*70)
log(f"Max-Transversal Search — instance={INSTANCE}")
log("Using IC moves to maximize transversal count, then find mates")
log("="*70)

rng = random.Random(314159265 + int(INSTANCE) * 1000003)
current_L = L_start.copy()
current_n_trans = count_transversals(current_L)
best_n_trans = current_n_trans
best_L = current_L.copy()
log(f"Starting: n_trans={current_n_trans}")

t_start = time.time()
ic_trial = 0
mate_trial = 0
SA_ACCEPT_WORSE = 0.1  # Probability of accepting worse state

while True:
    ic_trial += 1

    # Apply a random IC move
    r1, r2 = rng.sample(range(N), 2)
    c1, c2 = rng.sample(range(N), 2)
    new_L, changed = apply_ic_move(current_L, r1, r2, c1, c2, rng)

    if not changed:
        continue

    new_n_trans = count_transversals(new_L)

    # Accept if better, or with small probability
    if new_n_trans > current_n_trans or rng.random() < SA_ACCEPT_WORSE:
        current_L = new_L.copy()
        current_n_trans = new_n_trans

    if new_n_trans > best_n_trans:
        best_n_trans = new_n_trans
        best_L = new_L.copy()
        elapsed = time.time() - t_start
        log(f"IC trial {ic_trial}: NEW BEST n_trans={new_n_trans} t={elapsed:.0f}s")

    # Periodically try to find a mate for the current best
    if ic_trial % 500 == 0:
        mate_trial += 1
        elapsed = time.time() - t_start
        log(f"IC trial {ic_trial}: n_trans={current_n_trans}, best={best_n_trans} t={elapsed:.0f}s")

        if best_n_trans > 900:  # Only search for mate if many transversals
            log(f"Searching for mate (n_trans={best_n_trans})...")
            mate, M = find_mate_cpsat(best_L, timeout_s=30)
            if mate is not None:
                ct = count_ct(best_L, mate)
                cl = count_clashes(best_L, mate)
                log(f"  Found mate! cl={cl}, ct={ct}, M={M}")

                if ct >= N:
                    log("*** ct >= N! Looking for 3-MOLS ***")
                    # TODO: exact cover + 3-MOLS check
                    pass
