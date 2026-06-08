#!/usr/bin/env python3
"""
Feasibility attack: hard constraint E<=target, find L3 satisfying it.
Much faster than optimization when near the true minimum.
"""

import json, sys, time, fcntl
from datetime import datetime
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))

from mols_cpsat_worker import (
    count_clashes, build_col_index, relabel_L3_canonical, N,
    TRIPLE_MISS_FILE, POOL_LOCK, RESULTS_DIR, FOUND_FILE
)

LOG_FILE = REPO / "mols10/results/l3_e32_feas.log"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_to_pool(L1, L2, L3, cl12, cl13, cl23, tag):
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12, "L1_key": str(L1.tolist()),
        "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": f"feas_{tag}", "solver": "cpsat_feas",
    }
    lock_fd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best = min((p["clashes"] for p in pool), default=999)
        if E > best + 2:
            return False
        pool.append(entry)
        pool.sort(key=lambda e: e["clashes"])
        pool = pool[:16]
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def solve_feasibility(L1, L2, n, hint_L3, target_E, timeout_s, num_workers, rseed, symbreak=True):
    """Find L3 with cl13+cl23 <= target_E (feasibility, not optimization)."""
    model = cp_model.CpModel()
    L3v = [[model.new_int_var(0, n-1, f"L3_{i}_{j}") for j in range(n)]
           for i in range(n)]
    for i in range(n):
        model.add_all_different(L3v[i])
    for j in range(n):
        model.add_all_different([L3v[i][j] for i in range(n)])
    if symbreak:
        for j in range(n):
            model.add(L3v[0][j] == j)

    if hint_L3 is not None:
        hint_canon = relabel_L3_canonical(hint_L3, n) if symbreak else hint_L3
        for i in range(n):
            for j in range(n):
                model.add_hint(L3v[i][j], int(hint_canon[i, j]))

    col_of_1 = build_col_index(L1, n)
    col_of_2 = build_col_index(L2, n)

    covered13, covered23 = [], []
    for a in range(n):
        for b in range(n):
            cov = model.new_bool_var(f"c13_{a}_{b}")
            indicators = []
            for i in range(n):
                j = int(col_of_1[a, i])
                ind = model.new_bool_var(f"i13_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                indicators.append(ind)
            model.add_bool_or(indicators).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in indicators]).only_enforce_if(cov.negated())
            covered13.append(cov)

    for a in range(n):
        for b in range(n):
            cov = model.new_bool_var(f"c23_{a}_{b}")
            indicators = []
            for i in range(n):
                j = int(col_of_2[a, i])
                ind = model.new_bool_var(f"i23_{a}_{b}_{i}")
                model.add(L3v[i][j] == b).only_enforce_if(ind)
                model.add(L3v[i][j] != b).only_enforce_if(ind.negated())
                indicators.append(ind)
            model.add_bool_or(indicators).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in indicators]).only_enforce_if(cov.negated())
            covered23.append(cov)

    # HARD constraint: total clashes <= target_E
    uncovered = [x.negated() for x in covered13] + [x.negated() for x in covered23]
    model.add(sum(uncovered) <= target_E)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False
    if random_seed := rseed:
        solver.parameters.random_seed = random_seed

    status = solver.solve(model)
    timed_out = (status == cp_model.UNKNOWN)
    infeasible = (status == cp_model.INFEASIBLE)

    if status == cp_model.FEASIBLE:
        L3_sol = np.array([[solver.value(L3v[i][j]) for j in range(n)]
                           for i in range(n)], dtype=np.int8)
        return L3_sol, timed_out, infeasible
    return None, timed_out, infeasible


def main():
    log("="*60)
    log("Feasibility attack: targeting E<=target")

    pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
    best_entry = min(pool, key=lambda e: e["clashes"]) if pool else None
    if best_entry is None:
        log("ERROR: no pool"); return

    L1 = np.array(best_entry["L1"], dtype=np.int8).reshape(N, N)
    L2 = np.array(best_entry["L2"], dtype=np.int8).reshape(N, N)
    hint = np.array(best_entry["L3"], dtype=np.int8).reshape(N, N)
    session_best = best_entry["clashes"]
    log(f"Best entry E={session_best}")

    # Configs: (target_E, timeout_s, workers, symbreak, seed)
    configs = []
    for target in [30, 28, 25, 20]:
        for rseed in range(60, 80):
            configs.append((target, 300, 6, True, rseed))
    for target in [30, 28]:
        for rseed in range(80, 100):
            configs.append((target, 600, 8, True, rseed))

    for target, timeout, workers, symbreak, rseed in configs:
        # Reload best
        try:
            pool = json.loads(TRIPLE_MISS_FILE.read_text())
            cur_best = min(pool, key=lambda e: e["clashes"])
            if cur_best["clashes"] < session_best:
                best_entry = cur_best
                L1 = np.array(best_entry["L1"], dtype=np.int8).reshape(N, N)
                L2 = np.array(best_entry["L2"], dtype=np.int8).reshape(N, N)
                hint = np.array(best_entry["L3"], dtype=np.int8).reshape(N, N)
                session_best = cur_best["clashes"]
                log(f"  → Reloaded best E={session_best}")
        except Exception:
            pass

        log(f"target={target} timeout={timeout}s {workers}w sym={symbreak} seed={rseed}")
        t0 = time.time()
        L3_sol, timed_out, infeasible = solve_feasibility(
            L1, L2, N, hint, target, timeout, workers, rseed, symbreak)
        elapsed = time.time() - t0

        if L3_sol is not None:
            cl13 = count_clashes(L1, L3_sol)
            cl23 = count_clashes(L2, L3_sol)
            E = cl13 + cl23
            cl12 = count_clashes(L1, L2)
            log(f"  FOUND! E={E} cl13={cl13} cl23={cl23} t={elapsed:.1f}s")

            if E == 0:
                log("!!! N(10) >= 3 FOUND !!!")
                FOUND_FILE.write_text(json.dumps({
                    "found": True, "L1": L1.tolist(), "L2": L2.tolist(),
                    "L3": L3_sol.tolist(), "cl12": cl12, "cl13": cl13, "cl23": cl23,
                }, indent=2))
                return

            if E < session_best:
                session_best = E
                hint = L3_sol.copy()
                saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23, rseed)
                log(f"  ★★★ NEW BEST E={E} t={elapsed:.1f}s saved={saved} ★★★")
            else:
                hint = L3_sol.copy()
        elif infeasible:
            log(f"  PROVED INFEASIBLE: no L3 with E<={target} for this (L1,L2) t={elapsed:.1f}s")
        else:
            log(f"  TIMEOUT t={elapsed:.1f}s")


if __name__ == "__main__":
    main()
