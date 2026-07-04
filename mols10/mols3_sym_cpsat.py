#!/usr/bin/env python3
"""
Prescribed-automorphism CP-SAT search for 3-MOLS(10).

Require the triple (L1,L2,L3) to be invariant under the automorphism
    sigma: (r, c, s) -> (r+d mod 10, c+d mod 10, s+d mod 10)
for a chosen shift d. This cuts the free cells by the orbit size
(order of d in Z_10), possibly making exhaustive search tractable.

d=2 or d=4: order 5 symmetry  (100 cells -> 20 orbits per square)
d=5:        order 2 symmetry  (100 cells -> 50 orbits per square)

Usage: python3 mols3_sym_cpsat.py <d> [timeout_s]
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
LOG = REPO / f"mols10/results/sym_cpsat_d{D}.log"
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

log("="*70)
log(f"Symmetric 3-MOLS(10) CP-SAT — shift d={D}, timeout={TIMEOUT}s")
log(f"Constraint: L_k(r+{D}, c+{D}) = L_k(r,c)+{D}  (mod 10), k=1,2,3")
log("="*70)

model = cp_model.CpModel()

# Boolean encoding: x[k][r][c][v] <=> L_k(r,c) = v
x = [[[[model.new_bool_var(f"x{k}_{r}_{c}_{v}") for v in range(N)]
       for c in range(N)] for r in range(N)] for k in range(3)]

for k in range(3):
    # Exactly one symbol per cell
    for r in range(N):
        for c in range(N):
            model.add_exactly_one(x[k][r][c])
    # Latin: each symbol once per row and once per column
    for v in range(N):
        for r in range(N):
            model.add_exactly_one([x[k][r][c][v] for c in range(N)])
        for c in range(N):
            model.add_exactly_one([x[k][r][c][v] for r in range(N)])

# Symmetry: x[k][(r+D)%N][(c+D)%N][(v+D)%N] == x[k][r][c][v]
for k in range(3):
    for r in range(N):
        for c in range(N):
            for v in range(N):
                model.add(x[k][(r+D)%N][(c+D)%N][(v+D)%N] == x[k][r][c][v])

# Orthogonality for each pair (k1,k2): each symbol-pair (a,b) occurs exactly once
pair_count = 0
for k1 in range(3):
    for k2 in range(k1+1, 3):
        for a in range(N):
            for b in range(N):
                prods = []
                for r in range(N):
                    for c in range(N):
                        y = model.new_bool_var(f"y{k1}{k2}_{r}_{c}_{a}_{b}")
                        model.add_bool_and([x[k1][r][c][a], x[k2][r][c][b]]).only_enforce_if(y)
                        model.add_bool_or([x[k1][r][c][a].Not(), x[k2][r][c][b].Not(), y])
                        # y => both
                        model.add_implication(y, x[k1][r][c][a])
                        model.add_implication(y, x[k2][r][c][b])
                        prods.append(y)
                model.add_exactly_one(prods)
                pair_count += 1

log(f"Model built: {pair_count} orthogonality constraints")

# Mild symmetry breaking that is compatible with the prescribed automorphism:
# fix L1(0,0)=0 (we can always relabel all three squares' symbols by the same
# translation z -> z - L1(0,0), which preserves the automorphism).
model.add(x[0][0][0][0] == 1)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIMEOUT
solver.parameters.num_workers = 4
solver.parameters.log_search_progress = False

t0 = time.time()
status = solver.solve(model)
elapsed = time.time() - t0
log(f"Status: {solver.status_name(status)} in {elapsed:.1f}s")

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    Ls = []
    for k in range(3):
        L = np.zeros((N,N), dtype=np.int8)
        for r in range(N):
            for c in range(N):
                for v in range(N):
                    if solver.value(x[k][r][c][v]):
                        L[r,c] = v
        Ls.append(L)
    L1, L2, L3 = Ls
    cl12 = count_clashes(L1, L2)
    cl13 = count_clashes(L1, L3)
    cl23 = count_clashes(L2, L3)
    log(f"VERIFY: cl12={cl12}, cl13={cl13}, cl23={cl23}")
    if cl12 == 0 and cl13 == 0 and cl23 == 0:
        log("*** 3-MOLS(10) FOUND (symmetric)! ***")
        FOUND_FILE.write_text(json.dumps({
            'found': True, 'method': f'sym_cpsat_d{D}',
            'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3.tolist(),
            'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
        }, indent=2))
        git_push(f"3-MOLS(10) FOUND via symmetric CP-SAT d={D}!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
elif status == cp_model.INFEASIBLE:
    log(f"PROVED: no 3-MOLS(10) invariant under (r,c,s)->(r+{D},c+{D},s+{D}) mod 10")
    result_file = REPO / f"mols10/results/sym_infeasible_d{D}.json"
    result_file.write_text(json.dumps({
        'shift': D, 'status': 'INFEASIBLE', 'elapsed_s': elapsed,
        'statement': f"No 3-MOLS(10) invariant under (r,c,s)->(r+{D},c+{D},s+{D}) mod 10",
    }, indent=2))
else:
    log("Timed out (UNKNOWN)")
