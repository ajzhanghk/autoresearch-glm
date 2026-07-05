#!/usr/bin/env python3
"""
CP-SAT optimization: over ALL orthogonal mates B of the 5504-transversal
turn-square L, MAXIMIZE ct(L,B).

Variables: x_i = transversal i is in B's decomposition,
           y_j = transversal j is a common transversal of (L,B).
Constraints: exact cover by x; y_j => sum_{i in bad(j)} x_i = 0
             (bad(j) = transversals intersecting t_j in >= 2 cells;
              the partition argument makes this sufficient).
Objective: maximize sum y_j.

- Every incumbent = new record-ct MOLS(10) pair (saved immediately).
- If incumbent >= 10: try exact-cover of commons into C => 3-MOLS(10).
- If OPTIMAL proved with value < 10: theorem — L is in no 3-MOLS(10).

Warm start: hint from the best known ct=6 mate.

Usage: python3 max_ct_cpsat.py <seed_json> [timeout_s]
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
LOG = REPO / f"mols10/results/maxct_{TAG}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE = REPO / f"mols10/results/maxct_best_{TAG}.json"
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

def try_complete_triple(L, B, trans, TA):
    """Exact-cover the common transversals of (L,B) into C."""
    BV = B[np.arange(N)[None, :], TA]
    S = np.sort(BV, axis=1)
    mask = (np.diff(S, axis=1) != 0).all(axis=1)
    commons = [trans[i] for i in np.nonzero(mask)[0]]
    log(f"    commons: {len(commons)}")
    if len(commons) < N:
        return None
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(commons):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    m2 = cp_model.CpModel()
    w = [m2.new_bool_var(f"w{i}") for i in range(len(commons))]
    feasible_cells = True
    for r in range(N):
        for c in range(N):
            if cell_to_t[r][c]:
                m2.add_exactly_one([w[i] for i in cell_to_t[r][c]])
            else:
                feasible_cells = False
    if not feasible_cells:
        log("    some cell uncovered by commons — no C")
        return None
    s2 = cp_model.CpSolver()
    s2.parameters.max_time_in_seconds = 300
    s2.parameters.num_workers = 2
    st = s2.solve(m2)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        C = np.zeros((N, N), dtype=np.int8)
        for sym, i in enumerate([i for i in range(len(commons)) if s2.value(w[i])]):
            for r in range(N):
                C[r, commons[i][r]] = sym
        return C
    log(f"    commons cover: {s2.status_name(st)}")
    return None

class Incumbent(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, trans, TA, L, t_start):
        super().__init__()
        self.x = x
        self.trans = trans
        self.TA = TA
        self.L = L
        self.t_start = t_start
        self.best = -1

    def on_solution_callback(self):
        obj = int(self.objective_value)
        if obj <= self.best:
            return
        self.best = obj
        sel = [i for i in range(len(self.x)) if self.value(self.x[i])]
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            t = self.trans[ti]
            for r in range(N):
                B[r, t[r]] = sym
        elapsed = time.time() - self.t_start
        log(f"INCUMBENT ct={obj} at t={elapsed:.0f}s")
        BEST_FILE.write_text(json.dumps({
            'best_ct': obj, 'L': self.L.tolist(), 'B': B.tolist(),
            'elapsed_s': elapsed,
        }, indent=2))
        if obj >= N:
            log(f"*** incumbent ct >= {N}! Trying triple completion... ***")
            C = try_complete_triple(self.L, B, self.trans, self.TA)
            if C is not None:
                cl12 = count_clashes(self.L, B); cl13 = count_clashes(self.L, C); cl23 = count_clashes(B, C)
                log(f"VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
                if cl12 == 0 and cl13 == 0 and cl23 == 0:
                    log("*** 3-MOLS(10) FOUND! E=0 verified ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'max_ct_cpsat', 'seed': SEED,
                        'L1': self.L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
                    }, indent=2))
                    git_push("3-MOLS(10) FOUND via max-ct CP-SAT!")
                    self.stop_search()

def main():
    log("=" * 70)
    log(f"Max-CT CP-SAT — seed={SEED}, timeout={TIMEOUT}s")
    log("=" * 70)

    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    log(f"{M} transversals")
    TA = np.array(trans, dtype=np.int64)

    log("Computing bad(j) sets...")
    t0 = time.time()
    TA8 = TA.astype(np.int8)
    bad = [None] * M
    CH = 256
    for lo in range(0, M, CH):
        hi = min(lo + CH, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            # keep self (I=10 >= 2): y_j must force x_j = 0, else selected
            # transversals count as phantom commons
            bad[lo + k] = np.nonzero(cnt[k] >= 2)[0]
    log(f"done in {time.time()-t0:.0f}s")

    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    log("Building model...")
    t0 = time.time()
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    y = [model.new_bool_var(f"y{j}") for j in range(M)]

    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell_to_t[r][c]])

    for j in range(M):
        bj = bad[j]
        if len(bj):
            model.add(sum(x[int(i)] for i in bj) == 0).only_enforce_if(y[j])

    model.maximize(sum(y))

    # Warm start from saved B (ct=6 mate)
    if 'B' in seed:
        B0 = np.array(seed['B'], dtype=np.int8)
        tidx = {t: i for i, t in enumerate(trans)}
        hint_ok = True
        hints = []
        for sym in range(N):
            cols = tuple(int(np.nonzero(B0[r] == sym)[0][0]) for r in range(N))
            if cols in tidx:
                hints.append(tidx[cols])
            else:
                hint_ok = False; break
        if hint_ok:
            hset = set(hints)
            for i in range(M):
                model.add_hint(x[i], 1 if i in hset else 0)
            log(f"Hinted with saved mate (ct={seed.get('best_ct','?')})")

    log(f"Model built in {time.time()-t0:.0f}s")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_workers = 4
    cb = Incumbent(x, trans, TA, L, time.time())
    t0 = time.time()
    status = solver.solve(model, cb)
    elapsed = time.time() - t0
    log(f"STATUS: {solver.status_name(status)} in {elapsed:.0f}s")
    log(f"best objective: {solver.objective_value if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 'n/a'}, bound: {solver.best_objective_bound}")

    if status == cp_model.OPTIMAL:
        opt = int(solver.objective_value)
        if opt < N:
            log(f"THEOREM: max ct over ALL mates of this square = {opt} < {N}")
            log("=> this turn-square is in NO 3-MOLS(10)")
            out = REPO / f"mols10/results/maxct_theorem_{TAG}.json"
            out.write_text(json.dumps({
                'seed': SEED, 'max_ct_over_all_mates': opt,
                'status': 'OPTIMAL', 'elapsed_s': elapsed,
                'L': L.tolist(),
            }, indent=2))

if __name__ == "__main__":
    main()
