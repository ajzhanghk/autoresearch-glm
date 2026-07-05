#!/usr/bin/env python3
"""
Definitive CP-SAT attack: does the 5504-transversal turn-square L belong
to a 3-MOLS(10)?

L is in a triple (L,B,C) iff there exist TWO decompositions X (->B) and
Z (->C) of L's transversal pool such that every t_i in X and t_j in Z
intersect in exactly one cell.

KEY REDUCTION: X's 10 transversals partition the 100 cells, so for any
transversal t_j, sum_{i in X} |t_i ∩ t_j| = 10. Hence if no pair
intersects in >= 2 cells, every pair intersects in exactly 1. We only
need to forbid I(i,j) >= 2 pairs across (X, Z).

Model:
  x_i, z_j booleans over the 5504-transversal pool
  exact cover constraints for X and for Z (100 cells each)
  x_i = 1  =>  sum_{j in bad(i)} z_j = 0     (bad(i) = {j : I(i,j) >= 2})
  symmetry breaking: index of X's (0,0)-transversal < index of Z's

SAT        => 3-MOLS(10) FOUND (verify, save, push, exit)
INFEASIBLE => theorem: this square is in no 3-MOLS(10)

Usage: python3 triple_cpsat_turnsq.py <seed_json> [timeout_s]
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

SEED = sys.argv[1] if len(sys.argv) > 1 else "mols10/results/turnsq_best_ct_0.json"
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 21600
TAG = Path(SEED).stem
LOG = REPO / f"mols10/results/triple_cpsat_{TAG}.log"
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
    rows = [[int(L[r, c]) for c in range(N)] for r in range(N)]
    def bt(row, cm, vm, path):
        if row == N:
            all_t.append(tuple(path)); return
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    path.append(col)
                    bt(row+1, cm | (1 << col), vm | (1 << v), path)
                    path.pop()
    bt(0, 0, 0, [])
    return all_t

def main():
    log("=" * 70)
    log(f"Triple CP-SAT on turn-square — seed={SEED}, timeout={TIMEOUT}s")
    log("=" * 70)

    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    log(f"{M} transversals")

    TA = np.array(trans, dtype=np.int8)  # M x N: col per row

    # Pairwise intersection counts, chunked
    log("Computing pairwise intersection counts...")
    t0 = time.time()
    bad = [None] * M   # bad[i] = array of j with I(i,j) >= 2
    n_bad_total = 0
    CH = 256
    for lo in range(0, M, CH):
        hi = min(lo + CH, M)
        eq = (TA[lo:hi, None, :] == TA[None, :, :])   # (ch, M, N)
        cnt = eq.sum(axis=2, dtype=np.int8)           # (ch, M)
        for k in range(hi - lo):
            row = np.nonzero(cnt[k] >= 2)[0]
            # exclude self (I(i,i)=10)
            row = row[row != (lo + k)]
            bad[lo + k] = row
            n_bad_total += len(row)
    log(f"Intersections done in {time.time()-t0:.0f}s; avg |bad(i)| = {n_bad_total/M:.0f}")

    # cell -> transversal indices
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    log("Building CP-SAT model...")
    t0 = time.time()
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    z = [model.new_bool_var(f"z{i}") for i in range(M)]

    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            model.add_exactly_one([z[i] for i in cell_to_t[r][c]])

    # x_i => no z_j in bad(i)
    for i in range(M):
        bj = bad[i]
        if len(bj):
            model.add(sum(z[int(j)] for j in bj) == 0).only_enforce_if(x[i])

    # Symmetry breaking: X's transversal through cell (0,0) has smaller index
    # than Z's. (Swapping X<->Z swaps B<->C.)
    t00 = [i for i in cell_to_t[0][0]]
    ix = model.new_int_var(0, M, "ix")
    iz = model.new_int_var(0, M, "iz")
    model.add(ix == sum(i * x[i] for i in t00))
    model.add(iz == sum(i * z[i] for i in t00))
    model.add(ix < iz)

    log(f"Model built in {time.time()-t0:.0f}s")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_workers = 4
    solver.parameters.log_search_progress = False

    t0 = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0
    log(f"STATUS: {solver.status_name(status)} in {elapsed:.0f}s")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selX = [i for i in range(M) if solver.value(x[i])]
        selZ = [i for i in range(M) if solver.value(z[i])]
        B = np.zeros((N, N), dtype=np.int8)
        C = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(selX):
            for r in range(N):
                B[r, trans[ti][r]] = sym
        for sym, ti in enumerate(selZ):
            for r in range(N):
                C[r, trans[ti][r]] = sym
        cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
        log(f"VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
        if cl12 == 0 and cl13 == 0 and cl23 == 0:
            log("*** 3-MOLS(10) FOUND! E=0 verified ***")
            FOUND_FILE.write_text(json.dumps({
                'found': True, 'method': 'triple_cpsat_turnsq', 'seed': SEED,
                'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
            }, indent=2))
            git_push("3-MOLS(10) FOUND via triple CP-SAT on turn-square!")
        else:
            log("!!! verification failed — model bug, investigate !!!")
    elif status == cp_model.INFEASIBLE:
        log(f"THEOREM: this 5504-transversal turn-square is in NO 3-MOLS(10)")
        out = REPO / f"mols10/results/triple_infeasible_{TAG}.json"
        out.write_text(json.dumps({
            'seed': SEED, 'status': 'INFEASIBLE', 'elapsed_s': elapsed,
            'n_transversals': M,
            'statement': 'No two mutually-orthogonal transversal decompositions exist',
            'L': L.tolist(),
        }, indent=2))
    else:
        log("Timed out (UNKNOWN) — rerun with longer budget")

if __name__ == "__main__":
    main()
