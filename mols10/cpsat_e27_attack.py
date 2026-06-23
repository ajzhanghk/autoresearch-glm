#!/usr/bin/env python3
"""
CP-SAT attack from E=27 pairs targeting E<=26.

Strategy: The original E=26 breakthrough came from CP-SAT jumping E=29->E=26.
Now we attack from E=27 (iso2_tv21_0_5, tv_254) with constraint obj<=26.
Only a 1-unit improvement needed -- potentially much easier.

New ideas vs previous CP-SAT runs:
  1. Upper-bound constraint (obj<=26) forces CP-SAT to find improvement
  2. Multiple restarts with different random seeds
  3. No symmetry breaking on some runs (hint L3 may not have canonical row 0)
  4. Try both iso2_tv21_0_5 and tv_254 pairs
  5. Partial fix: fix rows of L3 that contribute most to coverage, free the rest
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, build_and_solve, N, build_col_index
from ortools.sat.python import cp_model

LOG       = REPO / "mols10/results/cpsat_e27_attack.log"
BEST_FILE = REPO / "mols10/results/cpsat_e27_best.json"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

TIMEOUT   = 600      # 10 min per attempt
N_WORKERS = 8        # CP-SAT parallel workers
GOAL_E    = 26       # must beat current global best

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    for f in [str(BEST_FILE.relative_to(REPO)), str(FOUND_FILE.relative_to(REPO))]:
        subprocess.run(["git","-C",str(REPO),"add",f], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def relabel_canonical(L3):
    perm = np.zeros(N, dtype=np.int32)
    for j in range(N):
        perm[int(L3[0, j])] = j
    return perm[L3.astype(np.int32)].astype(np.int8)

def build_and_solve_bounded(L1, L2, hint_L3, upper_bound, timeout_s, num_workers, rseed, symbreak=True):
    """CP-SAT: find L3 with cl13+cl23 <= upper_bound. Returns (E, L3) or (None, None) if timeout/no improvement."""
    model = cp_model.CpModel()
    L3v = [[model.new_int_var(0, N-1, f"L3_{i}_{j}") for j in range(N)] for i in range(N)]

    for i in range(N):
        model.add_all_different(L3v[i])
    for j in range(N):
        model.add_all_different([L3v[i][j] for i in range(N)])

    if symbreak:
        for j in range(N):
            model.add(L3v[0][j] == j)

    # Warm start
    hint_c = relabel_canonical(hint_L3) if symbreak else hint_L3.copy()
    for i in range(N):
        for j in range(N):
            model.add_hint(L3v[i][j], int(hint_c[i,j]))

    col1 = build_col_index(L1, N)
    col2 = build_col_index(L2, N)

    covered13, covered23 = [], []
    ind_hints = []

    for a in range(N):
        for b in range(N):
            cov = model.new_bool_var(f"c13_{a}_{b}")
            inds = []
            any_c = False
            for i in range(N):
                j = int(col1[a, i])
                ind = model.new_bool_var(f"i13_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                inds.append(ind)
                val = 1 if int(hint_c[i,j]) == b else 0
                ind_hints.append((ind, val))
                if val: any_c = True
            model.add_bool_or(inds).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in inds]).only_enforce_if(cov.negated())
            covered13.append(cov)
            ind_hints.append((cov, 1 if any_c else 0))

    for a in range(N):
        for b in range(N):
            cov = model.new_bool_var(f"c23_{a}_{b}")
            inds = []
            any_c = False
            for i in range(N):
                j = int(col2[a, i])
                ind = model.new_bool_var(f"i23_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                inds.append(ind)
                val = 1 if int(hint_c[i,j]) == b else 0
                ind_hints.append((ind, val))
                if val: any_c = True
            model.add_bool_or(inds).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in inds]).only_enforce_if(cov.negated())
            covered23.append(cov)
            ind_hints.append((cov, 1 if any_c else 0))

    for var, val in ind_hints:
        model.add_hint(var, val)

    uncovered = [x.negated() for x in covered13] + [x.negated() for x in covered23]
    obj = model.new_int_var(0, 2*N*N, "obj")
    model.add(obj == sum(uncovered))
    # KEY: hard upper bound -- only look for solutions better than current best
    model.add(obj <= upper_bound)
    model.minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False
    solver.parameters.random_seed = rseed

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        E = int(solver.objective_value)
        L3_sol = np.array([[solver.value(L3v[i][j]) for j in range(N)] for i in range(N)], dtype=np.int8)
        return E, L3_sol
    return None, None  # INFEASIBLE or TIMEOUT

# Load pair data
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

# Target pairs with E=27 seeds (1 unit above global best E=26)
target_seeds = {
    'iso2_tv21_0_5': best_data['iso2_tv21_0_5'],
    'tv_254':        best_data['tv_254'],
}

global_best_E = GOAL_E
if BEST_FILE.exists():
    bd = json.loads(BEST_FILE.read_text())
    global_best_E = min(global_best_E, bd.get('E', GOAL_E))

log("="*70)
log(f"CP-SAT attack from E=27 pairs, targeting E<={GOAL_E-1}")
log(f"Pairs: {list(target_seeds.keys())}")
log(f"Timeout: {TIMEOUT}s per attempt, {N_WORKERS} workers")
log("="*70)

rng = random.Random(27182818284)
attempt = 0

while True:
    for pair_id, seed_data in target_seeds.items():
        attempt += 1
        p = pair_map[pair_id]
        L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
        L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
        hint_L3 = np.array(seed_data['L3'], dtype=np.int8).reshape(N,N)

        # Verify hint energy
        e_hint = count_clashes(L1, hint_L3) + count_clashes(L2, hint_L3)

        rseed = rng.randint(0, 2**31-1)
        symbreak = (attempt % 3 != 0)  # skip symbreak every 3rd attempt

        log(f"Attempt {attempt} | {pair_id} | hint_E={e_hint} | upper={global_best_E-1} | symbreak={symbreak} | rseed={rseed}")

        t0 = time.time()
        E, L3_sol = build_and_solve_bounded(
            L1, L2, hint_L3,
            upper_bound=global_best_E - 1,
            timeout_s=TIMEOUT,
            num_workers=N_WORKERS,
            rseed=rseed,
            symbreak=symbreak,
        )
        elapsed = time.time() - t0

        if L3_sol is not None:
            cl13 = int(count_clashes(L1, L3_sol))
            cl23 = int(count_clashes(L2, L3_sol))
            E_verified = cl13 + cl23
            log(f"  *** FOUND E={E_verified} (cl13={cl13}, cl23={cl23}) t={elapsed:.0f}s ***")

            if E_verified < global_best_E:
                global_best_E = E_verified
                entry = {
                    'E': E_verified, 'cl13': cl13, 'cl23': cl23,
                    'pair_id': pair_id, 'attempt': attempt,
                    'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist()
                }
                BEST_FILE.write_text(json.dumps(entry, indent=2))
                git_push(f"cpsat_e27: {pair_id} E={E_verified} (NEW BEST from E=27!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")

                if E_verified == 0:
                    log("*** 3-MOLS FOUND! ***")
                    found = {
                        'found': True, 'pair_id': pair_id,
                        'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist(),
                        'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
                    }
                    FOUND_FILE.write_text(json.dumps(found, indent=2))
                    git_push(f"*** 3-MOLS FOUND! E=0 pair={pair_id} ***\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    sys.exit(0)
        else:
            log(f"  No improvement found (INFEASIBLE or timeout) t={elapsed:.0f}s")
