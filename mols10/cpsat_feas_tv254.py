#!/usr/bin/env python3
"""
CP-SAT pure feasibility attack: does E <= TARGET exist for tv_254?

Adds a hard upper-bound constraint (obj <= TARGET-1) making this a
pure SAT/feasibility problem rather than optimization. For CP-SAT,
feasibility is typically much faster than proving optimality.

Runs multiple trials with different random seeds, each with TIMEOUT seconds.
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, build_and_solve, N

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
TARGET   = int(sys.argv[2]) if len(sys.argv) > 2 else 22
TIMEOUT  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
WORKERS  = int(sys.argv[4]) if len(sys.argv) > 4 else 3

LOG        = REPO / f"mols10/results/cpsat_feas_{PAIR_ID}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE  = REPO / f"mols10/results/escape_best_{PAIR_ID}.json"
BRANCH     = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add","-A"], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def build_and_solve_feas(L1, L2, n, hint_L3, max_clashes, timeout_s, num_workers, random_seed):
    """Build CP-SAT model with hard upper-bound constraint max_clashes."""
    from ortools.sat.python import cp_model
    from mols_cpsat_worker import relabel_L3_canonical
    model = cp_model.CpModel()

    L3v = [[model.new_int_var(0, n-1, f"L3_{i}_{j}") for j in range(n)] for i in range(n)]

    # Latin square constraints for L3
    for i in range(n):
        model.add_all_different(L3v[i])
    for j in range(n):
        model.add_all_different([L3v[i][j] for i in range(n)])

    # Hint from best known L3
    hint_canon = relabel_L3_canonical(hint_L3, n)
    for i in range(n):
        for j in range(n):
            model.add_hint(L3v[i][j], int(hint_canon[i, j]))

    # Coverage variables for L1⊗L3 and L2⊗L3
    covered13, covered23 = [], []
    all_ind13, all_ind23 = [], []
    all_cov13, all_cov23 = [], []

    for a in range(n):
        for b in range(n):
            # L1⊗L3 coverage: (a,b) covered if exists i s.t. L3[i, c(i,a)] = b
            ind_vars13 = []
            for i in range(n):
                c_ia = int(np.where(L1[i] == a)[0][0])
                ind = model.new_bool_var(f"ind13_{a}_{b}_{i}")
                model.add(L3v[i][c_ia] == b).only_enforce_if(ind)
                model.add(L3v[i][c_ia] != b).only_enforce_if(ind.negated())
                ind_vars13.append(ind)
                hint_val = 1 if int(hint_canon[i, c_ia]) == b else 0
                all_ind13.append((ind, hint_val))
            cov13 = model.new_bool_var(f"cov13_{a}_{b}")
            model.add_bool_or(ind_vars13).only_enforce_if(cov13)
            model.add_bool_and([x.negated() for x in ind_vars13]).only_enforce_if(cov13.negated())
            covered13.append(cov13)
            hint_val = 1 if any(int(hint_canon[i, int(np.where(L1[i]==a)[0][0])]) == b for i in range(n)) else 0
            all_cov13.append((cov13, hint_val))

            # L2⊗L3 coverage
            ind_vars23 = []
            for i in range(n):
                c_ia = int(np.where(L2[i] == a)[0][0])
                ind = model.new_bool_var(f"ind23_{a}_{b}_{i}")
                model.add(L3v[i][c_ia] == b).only_enforce_if(ind)
                model.add(L3v[i][c_ia] != b).only_enforce_if(ind.negated())
                ind_vars23.append(ind)
                hint_val = 1 if int(hint_canon[i, c_ia]) == b else 0
                all_ind23.append((ind, hint_val))
            cov23 = model.new_bool_var(f"cov23_{a}_{b}")
            model.add_bool_or(ind_vars23).only_enforce_if(cov23)
            model.add_bool_and([x.negated() for x in ind_vars23]).only_enforce_if(cov23.negated())
            covered23.append(cov23)
            hint_val = 1 if any(int(hint_canon[i, int(np.where(L2[i]==a)[0][0])]) == b for i in range(n)) else 0
            all_cov23.append((cov23, hint_val))

    # Set hints
    for ind, val in all_ind13 + all_ind23: model.add_hint(ind, val)
    for cov, val in all_cov13 + all_cov23: model.add_hint(cov, val)

    # Objective: minimize uncovered
    uncovered = [x.negated() for x in covered13] + [x.negated() for x in covered23]
    obj = model.new_int_var(0, 2*n*n, "obj")
    model.add(obj == sum(uncovered))

    # HARD UPPER BOUND: require obj <= max_clashes (feasibility constraint)
    model.add(obj <= max_clashes)
    model.minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False
    if random_seed is not None:
        solver.parameters.random_seed = random_seed

    status = solver.solve(model)
    timed_out = (status == cp_model.UNKNOWN)
    infeasible = (status == cp_model.INFEASIBLE)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best_obj = int(solver.objective_value)
        L3_sol = np.array([[solver.value(L3v[i][j]) for j in range(n)] for i in range(n)], dtype=np.int8)
        return best_obj, L3_sol, timed_out, infeasible, (status == cp_model.OPTIMAL)
    return -1, None, timed_out, infeasible, False

# Load pair
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)

# Load best L3 hint
hint_L3 = None
current_best_E = 999
for fp in [
    REPO/f"mols10/results/escape_best_{PAIR_ID}.json",
    REPO/f"mols10/results/pt_agg_best_{PAIR_ID}_b.json",
    REPO/f"mols10/results/pt_best_{PAIR_ID}.json",
]:
    if fp.exists():
        d = json.loads(fp.read_text())
        if d.get('pair_id') == PAIR_ID and d.get('E', 999) < current_best_E:
            hint_L3 = np.array(d['L3'], dtype=np.int8).reshape(N,N)
            current_best_E = d['E']

if hint_L3 is None:
    log("ERROR: no hint found"); sys.exit(1)

log("="*70)
log(f"CP-SAT feasibility — pair={PAIR_ID}  hint_E={current_best_E}  target=E<={TARGET}")
log(f"timeout={TIMEOUT}s per trial  workers={WORKERS}")
log(f"Strategy: hard constraint obj<={TARGET} — pure SAT/feasibility")
log("="*70)

seeds = [42, 137, 271828, 314159, 999999, 1234567, 7654321, 88888888,
         12345, 98765, 55555, 33333, 11111, 77777, 99999, 66666]
trial = 0

while True:
    trial += 1
    seed = seeds[(trial-1) % len(seeds)]
    log(f"Trial {trial} | seed={seed} | target E<={TARGET} | timeout={TIMEOUT}s")

    t0 = time.time()
    best_obj, L3_sol, timed_out, infeasible, is_optimal = build_and_solve_feas(
        L1, L2, N, hint_L3, TARGET, TIMEOUT, WORKERS, seed)
    elapsed = time.time() - t0

    if infeasible:
        log(f"  PROVED INFEASIBLE: no L3 with E<={TARGET} exists for {PAIR_ID}!")
        log(f"  This proves E={current_best_E} is the global minimum.")
        TARGET += 1  # Try harder: prove E<=TARGET+1
        if TARGET >= current_best_E:
            log("All targets exhausted — search complete."); break
    elif L3_sol is not None:
        cl13 = int(count_clashes(L1, L3_sol))
        cl23 = int(count_clashes(L2, L3_sol))
        E = cl13 + cl23
        status = "OPTIMAL" if is_optimal else ("FEASIBLE" if not timed_out else "TIMEOUT+FEASIBLE")
        log(f"  {status}: E={E} cl13={cl13} cl23={cl23}  elapsed={elapsed:.1f}s")
        if E < current_best_E:
            current_best_E = E
            hint_L3 = L3_sol.copy()
            TARGET = E - 1
            log(f"*** NEW BEST E={E}! Updating target to E<={TARGET} ***")
            BEST_FILE.write_text(json.dumps({
                'E': E, 'cl13': cl13, 'cl23': cl23, 'pair_id': PAIR_ID,
                'trial': trial, 'method': 'cpsat_feas',
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist()
            }, indent=2))
            git_push(f"cpsat_feas: {PAIR_ID} E={E}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
            if E == 0:
                log("*** 3-MOLS FOUND! ***")
                FOUND_FILE.write_text(json.dumps({
                    'found': True, 'pair_id': PAIR_ID,
                    'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist(),
                    'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
                }, indent=2))
                import os; os._exit(0)
    else:
        log(f"  TIMEOUT (no solution found in {elapsed:.1f}s)")
