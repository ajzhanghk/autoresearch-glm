#!/usr/bin/env python3
"""
DECISION PROCEDURE: does the given square L have an orthogonal mate B with
ct(L,B) >= 10?  (Necessary for L to belong to a 3-MOLS(10); if SAT, we
immediately attempt completion into a full triple.)

Model = exact cover (x) + common-transversal indicators (y) as in
max_ct_cpsat.py, but with the HARD constraint sum(y) >= 10 and no
objective — pure feasibility, so bound propagation prunes any subtree
that cannot reach 10.

INFEASIBLE  => THEOREM: max ct over all mates of L is <= 9 < 10,
               hence L belongs to NO 3-MOLS(10).
SAT         => candidate: check whether >= 10 pairwise-disjoint commons
               exist and exact-cover them into C; verify; if E=0, victory.

Usage: python3 ct10_decide.py <seed_json> [timeout_s]
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

SEED = sys.argv[1] if len(sys.argv) > 1 else "mols10/results/turnsq_ct6_squareA.json"
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 7200
TAG = Path(SEED).stem
LOG = REPO / f"mols10/results/ct10_{TAG}.log"
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
    log(f"ct>=10 DECISION — seed={SEED}, timeout={TIMEOUT}s")
    log("=" * 70)

    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    log(f"{M} transversals")
    TA8 = np.array(trans, dtype=np.int8)

    log("Computing bad(j) sets (I >= 2, self included)...")
    t0 = time.time()
    bad = [None] * M
    CH = 256
    for lo in range(0, M, CH):
        hi = min(lo + CH, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            bad[lo + k] = np.nonzero(cnt[k] >= 2)[0]
    log(f"done in {time.time()-t0:.0f}s")

    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    log("Building feasibility model (sum y >= 10)...")
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

    model.add(sum(y) >= N)  # THE decision constraint

    # Warm-start hint from best known mate (ct=6) — helps the SAT direction
    if 'B' in seed:
        B0 = np.array(seed['B'], dtype=np.int8)
        tidx = {t: i for i, t in enumerate(trans)}
        hints, ok = [], True
        for sym in range(N):
            cols = tuple(int(np.nonzero(B0[r] == sym)[0][0]) for r in range(N))
            if cols in tidx: hints.append(tidx[cols])
            else: ok = False; break
        if ok:
            hset = set(hints)
            for i in range(M):
                model.add_hint(x[i], 1 if i in hset else 0)
            log("Hinted x with the ct=6 mate")

    log(f"Model built in {time.time()-t0:.0f}s")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_workers = 4
    t0 = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0
    log(f"STATUS: {solver.status_name(status)} in {elapsed:.0f}s")

    if status == cp_model.INFEASIBLE:
        log("*" * 60)
        log(f"THEOREM: no orthogonal mate of this square has ct >= 10.")
        log(f"=> This 5504-transversal turn-square belongs to NO 3-MOLS(10).")
        log("*" * 60)
        out = REPO / f"mols10/results/ct10_theorem_{TAG}.json"
        out.write_text(json.dumps({
            'seed': SEED, 'status': 'INFEASIBLE', 'elapsed_s': elapsed,
            'statement': 'max ct over all orthogonal mates <= 9; square is in no 3-MOLS(10)',
            'L': L.tolist(),
        }, indent=2))
        git_push(f"THEOREM: {TAG} has no mate with ct>=10 (CP-SAT INFEASIBLE)")
    elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = [i for i in range(M) if solver.value(x[i])]
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            for r in range(N):
                B[r, trans[ti][r]] = sym
        yc = sum(int(solver.value(y[j])) for j in range(M))
        log(f"SAT! Found mate with >= {yc} certified commons. Verifying + completing...")
        # Independent ct check
        BV = B[np.arange(N)[None, :], TA8.astype(np.int64)]
        S = np.sort(BV, axis=1)
        mask = (np.diff(S, axis=1) != 0).all(axis=1)
        commons = [trans[i] for i in np.nonzero(mask)[0]]
        log(f"independent ct = {len(commons)}, cl(L,B) = {count_clashes(L, B)}")
        # exact-cover commons into C
        c2t = [[[] for _ in range(N)] for _ in range(N)]
        for i, t in enumerate(commons):
            for r in range(N):
                c2t[r][t[r]].append(i)
        m2 = cp_model.CpModel()
        w = [m2.new_bool_var(f"w{i}") for i in range(len(commons))]
        coverable = True
        for r in range(N):
            for c in range(N):
                if c2t[r][c]: m2.add_exactly_one([w[i] for i in c2t[r][c]])
                else: coverable = False
        if coverable:
            s2 = cp_model.CpSolver()
            s2.parameters.max_time_in_seconds = 600
            s2.parameters.num_workers = 4
            st2 = s2.solve(m2)
            log(f"commons exact-cover: {s2.status_name(st2)}")
            if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                C = np.zeros((N, N), dtype=np.int8)
                for sym, i in enumerate([i for i in range(len(commons)) if s2.value(w[i])]):
                    for r in range(N):
                        C[r, commons[i][r]] = sym
                cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
                log(f"VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
                if cl12 == 0 and cl13 == 0 and cl23 == 0:
                    log("*** 3-MOLS(10) FOUND! E=0 verified ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'ct10_decide', 'seed': SEED,
                        'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                        'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23),
                    }, indent=2))
                    git_push("3-MOLS(10) FOUND via ct10_decide!")
        else:
            log("commons do not cover all cells — no C from this mate; "
                "rerun to enumerate further ct>=10 mates")
        # Save the >=10 pair regardless — record material
        out = REPO / f"mols10/results/ct10_sat_{TAG}.json"
        out.write_text(json.dumps({
            'seed': SEED, 'ct': len(commons), 'L': L.tolist(), 'B': B.tolist(),
        }, indent=2))
        git_push(f"ct>=10 mate found for {TAG} (ct={len(commons)})")
    else:
        log("UNKNOWN — rerun with a longer budget or split the search")

if __name__ == "__main__":
    main()
