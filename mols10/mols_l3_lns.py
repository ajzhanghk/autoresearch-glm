#!/usr/bin/env python3
"""
L3 Large-Neighborhood Search (LNS) targeting E<31.

Fixes 10-k rows of L3, optimizes the k free rows with CP-SAT.
Uses the same efficient coverage encoding as mols_cpsat_worker.
"""
import json, sys, time, fcntl, argparse, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N, TRIPLE_MISS_FILE, POOL_LOCK, FOUND_FILE
from mols_cpsat_worker import build_col_index, relabel_L3_canonical

from ortools.sat.python import cp_model

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6001)
parser.add_argument("--timeout", type=int, default=120)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--free-rows", type=int, default=3)
args = parser.parse_args()

LOG_FILE = REPO / f"mols10/results/l3_lns_{args.seed}.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def save_to_pool(L1, L2, L3, cl12, cl13, cl23):
    if not is_latin(L3):
        return False
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12, "L1_key": str(L1.tolist()),
        "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": f"lns_{args.seed}", "solver": "lns",
    }
    lfd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lfd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best = min((p["clashes"] for p in pool), default=999)
        if E > best:
            return False
        pool.append(entry)
        pool.sort(key=lambda e: e["clashes"])
        pool = pool[:16]
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lfd, fcntl.LOCK_UN)
        lfd.close()


def is_latin(A):
    for row in A:
        if len(set(row.tolist())) != N:
            return False
    for c in range(N):
        if len(set(A[:, c].tolist())) != N:
            return False
    return True


def lns_solve(L1, L2, L3_hint, free_rows, timeout_s, num_workers, rseed):
    """
    Fix rows NOT in free_rows to their hint values.
    Optimize the free rows via CP-SAT using efficient coverage encoding.
    """
    model = cp_model.CpModel()
    n = N
    fixed_rows = [r for r in range(n) if r not in free_rows]

    # Decision variables: L3[i][j]
    L3v = [[model.new_int_var(0, n - 1, f"L3_{i}_{j}") for j in range(n)]
           for i in range(n)]

    # Fix fixed rows
    for r in fixed_rows:
        for c in range(n):
            model.add(L3v[r][c] == int(L3_hint[r, c]))

    # Latin square: all-different per row (only free rows need it; fixed rows are trivially unique)
    for r in free_rows:
        model.add_all_different(L3v[r])

    # All-different per column (all rows participate — links fixed and free)
    for c in range(n):
        model.add_all_different([L3v[r][c] for r in range(n)])

    # Efficient coverage encoding (same as build_and_solve)
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

    uncovered = [x.negated() for x in covered13] + [x.negated() for x in covered23]
    obj = model.new_int_var(0, 2 * n * n, "obj")
    model.add(obj == sum(uncovered))
    model.minimize(obj)

    # Warm-start hint for free rows (use hint values)
    for r in free_rows:
        for c in range(n):
            model.add_hint(L3v[r][c], int(L3_hint[r, c]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = num_workers
    solver.parameters.random_seed = rseed
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        L3_sol = np.array([[solver.Value(L3v[r][c]) for c in range(n)]
                           for r in range(n)], dtype=np.int8)
        return L3_sol
    return None


def main():
    rng = random.Random(args.seed)
    log("=" * 60)
    log(f"L3-LNS seed={args.seed} timeout={args.timeout}s {args.workers}w free={args.free_rows}rows")

    trial = 0
    session_best = 999
    L1 = L2 = hint = None

    while True:
        try:
            pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
            be = min(pool, key=lambda e: e["clashes"]) if pool else None
            if be is None:
                log("Empty pool, sleeping 30s")
                time.sleep(30)
                continue
            if be["clashes"] < session_best or trial == 0:
                L1 = np.array(be["L1"], dtype=np.int8).reshape(N, N)
                L2 = np.array(be["L2"], dtype=np.int8).reshape(N, N)
                hint = np.array(be["L3"], dtype=np.int8).reshape(N, N)
                session_best = be["clashes"]
                if trial > 0:
                    log(f"  → Reloaded best E={session_best}")
        except Exception as ex:
            log(f"Pool error: {ex}")
            time.sleep(5)
            continue

        trial += 1
        free_rows = sorted(rng.sample(range(N), args.free_rows))
        rseed = rng.randint(0, 2 ** 31)

        log(f"trial={trial} best={session_best} free={free_rows}")
        t0 = time.time()
        L3_sol = lns_solve(L1, L2, hint, free_rows, args.timeout, args.workers, rseed)
        elapsed = time.time() - t0

        if L3_sol is not None:
            cl12 = count_clashes(L1, L2)
            cl13 = count_clashes(L1, L3_sol)
            cl23 = count_clashes(L2, L3_sol)
            E = cl13 + cl23
            log(f"  E={E} cl13={cl13} cl23={cl23} t={elapsed:.1f}s")

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
                saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23)
                log(f"  ★ NEW BEST E={E} saved={saved}")
            elif E <= session_best + 1:
                hint = L3_sol.copy()
        else:
            log(f"  TIMEOUT t={elapsed:.1f}s")


if __name__ == "__main__":
    main()
