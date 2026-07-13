#!/usr/bin/env python3
"""
DIRECT 3-MOLS(10) SEARCH: solve for all three squares simultaneously,
independent of any square-generation heuristic. This covers the case
where a triple's squares lie outside every construction family our
decomposition-based hunt samples.

CP-SAT model:
  L1[r][c], L2[r][c], L3[r][c] in 0..9
  each square Latin (rows and columns all-different)
  pairwise orthogonality: for squares A,B the 100 pairs (A[r][c],B[r][c])
    are all distinct -- encoded by an integer P = 10*A + B per cell being
    all-different across the grid.
Symmetry breaking (safe for existence):
  L1 = reduced (first row and first column = 0..9)
  L2 first row = 0..9 ; L2[1][0] ordered
  value-symmetry on L3 first row anchored.

Different seeds/workers explore different regions. SAT => decode, verify
cl=0 all pairwise => 3-MOLS(10). This is the hardest possible instance
(that is why N(10)>=3 is open); we run it as the no-stone-unturned line
in parallel to the exact per-square decisions.

Usage: python3 direct_triple_sat.py <instance> [cap_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 7200
LOG = REPO / f"mols10/results/directtriple_{INSTANCE}.log"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    subprocess.run(["git","-C",str(REPO),"add","mols10/results/"], check=False)
    r = subprocess.run(["git","-C",str(REPO),"commit","-m",msg], capture_output=True, check=False)
    if r.returncode == 0:
        subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH], capture_output=True, check=False)
        subprocess.run(["git","-C",str(REPO),"push","origin",BRANCH], capture_output=True, check=False)

def build():
    m = cp_model.CpModel()
    L = [[[m.new_int_var(0, N-1, f"L{s}_{r}_{c}") for c in range(N)]
          for r in range(N)] for s in range(3)]
    for s in range(3):
        for r in range(N):
            m.add_all_different([L[s][r][c] for c in range(N)])
        for c in range(N):
            m.add_all_different([L[s][r][c] for r in range(N)])
    # pairwise orthogonality via all-different on 10*A+B
    for a in range(3):
        for b in range(a+1, 3):
            pairs = []
            for r in range(N):
                for c in range(N):
                    p = m.new_int_var(0, N*N-1, f"p{a}{b}_{r}_{c}")
                    m.add(p == N*L[a][r][c] + L[b][r][c])
                    pairs.append(p)
            m.add_all_different(pairs)
    # symmetry breaking: L1 reduced
    for c in range(N):
        m.add(L[0][0][c] == c)
        m.add(L[0][c][0] == c)
    # L2 first row identity (WLOG relabel symbols of L2)
    for c in range(N):
        m.add(L[1][0][c] == c)
    # L3 first row identity (WLOG relabel symbols of L3)
    for c in range(N):
        m.add(L[2][0][c] == c)
    # L2[1][0] < L3[1][0] to break the L2<->L3 swap symmetry
    m.add(L[1][1][0] <= L[2][1][0])
    return m, L

def main():
    log("="*70)
    log(f"DIRECT 3-MOLS(10) SAT — instance={INSTANCE}, cap={CAP}s")
    m, L = build()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CAP
    solver.parameters.num_workers = 2
    solver.parameters.random_seed = 1000 + INSTANCE
    # diversify search across instances
    solver.parameters.search_branching = [
        cp_model.AUTOMATIC_SEARCH, cp_model.FIXED_SEARCH,
        cp_model.PORTFOLIO_SEARCH, cp_model.LP_SEARCH][INSTANCE % 4]
    log("model built; solving...")
    t0 = time.time()
    st = solver.solve(m)
    el = time.time() - t0
    log(f"STATUS: {solver.status_name(st)} in {el:.0f}s")
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sq = [np.array([[solver.value(L[s][r][c]) for c in range(N)]
                        for r in range(N)], dtype=np.int8) for s in range(3)]
        cl = [count_clashes(sq[0], sq[1]), count_clashes(sq[0], sq[2]),
              count_clashes(sq[1], sq[2])]
        log(f"SAT! cl(1,2)={cl[0]} cl(1,3)={cl[1]} cl(2,3)={cl[2]}")
        if cl == [0, 0, 0]:
            log("*** *** 3-MOLS(10) FOUND — THE BIG FISH (direct SAT) *** ***")
            FOUND.write_text(json.dumps({
                'found': True, 'method': 'direct_triple_sat',
                'L1': sq[0].tolist(), 'L2': sq[1].tolist(), 'L3': sq[2].tolist(),
                'cl12': int(cl[0]), 'cl13': int(cl[1]), 'cl23': int(cl[2])}, indent=1))
            git_push("*** 3-MOLS(10) FOUND via direct_triple_sat! ***")
        else:
            log("!!! SAT but cl != 0 -- model bug, investigate !!!")
    elif st == cp_model.INFEASIBLE:
        log("INFEASIBLE: no 3-MOLS(10) in this symmetry-reduced class "
            "(would be a MAJOR result -- verify encoding before claiming)")
        git_push(f"direct_triple_sat i{INSTANCE}: INFEASIBLE (verify before claiming!)")
    else:
        log("UNKNOWN (timed out) -- expected for this hard instance")

if __name__ == "__main__":
    main()
