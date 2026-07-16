#!/usr/bin/env python3
"""
Threshold decision with sigma symmetry-breaking: does the 5504-transversal
turn-square L have an orthogonal mate B with ct(L,B) >= K?

If INFEASIBLE for K=7: max-ct of the turn square is EXACTLY 6 (incumbent
known), i.e. even the transversal-richest order-10 squares cannot come
close to the ct >= 10 a triple requires — the strongest single datapoint
for the weak CT-barrier conjecture (and it would close the last gap in
the proven max-ct map).

New ingredient vs ct10_decide.py: sigma symmetry-breaking. sigma:
(r,c) -> (r+5,c+5) maps transversals to transversals and mate
decompositions to mate decompositions, preserving ct. For a decomposition
X, compare u = index of X's (0,0)-covering transversal with
v = index of sigma(X's (5,5)-covering transversal) — which is exactly
sigma(X)'s (0,0)-cover. Requiring u <= v selects one representative per
{X, sigma(X)} orbit, halving the search space.

Usage: python3 ct_threshold_decide.py <seed_json> <K> [timeout_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

SEED = sys.argv[1]
K = int(sys.argv[2])
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 21600
TAG = f"{Path(SEED).stem}_ge{K}"
LOG = REPO / f"mols10/results/ctthr_{TAG}.log"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    subprocess.run(["git","-C",str(REPO),"add","mols10/results/"], check=False)
    r = subprocess.run(["git","-C",str(REPO),"commit","-m",msg],
                       capture_output=True, check=False)
    if r.returncode == 0:
        subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH],
                       capture_output=True, check=False)
        subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH],
                       capture_output=True, check=False)

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
    log(f"ct >= {K} threshold decision + sigma break — {SEED}, timeout={TIMEOUT}s")
    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    tidx = {t: i for i, t in enumerate(trans)}
    log(f"{M} transversals")

    # sigma action on transversals: sigma(t)[r] = t[(r-5) mod 10] + 5 mod 10
    sig = np.zeros(M, dtype=np.int64)
    ok = True
    for i, t in enumerate(trans):
        st = tuple((t[(r - 5) % N] + 5) % N for r in range(N))
        j = tidx.get(st)
        if j is None:
            ok = False
            break
        sig[i] = j
    if not ok:
        log("pool not closed under sigma — aborting symmetry break")
        return
    log(f"sigma is a permutation of the pool ({(sig == np.arange(M)).sum()} fixed)")

    TA8 = np.array(trans, dtype=np.int8)
    bad = [None] * M
    for lo in range(0, M, 256):
        hi = min(lo + 256, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            bad[lo + k] = np.nonzero(cnt[k] >= 2)[0]

    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    y = [model.new_bool_var(f"y{j}") for j in range(M)]
    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
    for j in range(M):
        if len(bad[j]):
            model.add(sum(x[int(i)] for i in bad[j]) == 0).only_enforce_if(y[j])
    model.add(sum(y) >= K)

    # sigma symmetry-breaking: u = idx of X's (0,0)-cover;
    # v = sig[idx of X's (5,5)-cover] (= sigma(X)'s (0,0)-cover); u <= v
    t00 = cell_to_t[0][0]
    t55 = cell_to_t[5][5]
    u = model.new_int_var(0, M, "u")
    v = model.new_int_var(0, M, "v")
    model.add(u == sum(i * x[i] for i in t00))
    model.add(v == sum(int(sig[i]) * x[i] for i in t55))
    model.add(u <= v)

    # hint from the ct=6 mate if present
    if 'B' in seed and seed['B'] is not None:
        B0 = np.array(seed['B'], dtype=np.int8)
        hints, hok = [], True
        for sym in range(N):
            cols = tuple(int(np.nonzero(B0[r] == sym)[0][0]) for r in range(N))
            if cols in tidx: hints.append(tidx[cols])
            else: hok = False; break
        if hok:
            hset = set(hints)
            for i in range(M):
                model.add_hint(x[i], 1 if i in hset else 0)
            log("hinted with saved mate")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_workers = 4
    t0 = time.time()
    st = solver.solve(model)
    el = time.time() - t0
    log(f"STATUS: {solver.status_name(st)} in {el:.0f}s")

    if st == cp_model.INFEASIBLE:
        log(f"THEOREM: no mate of this square has ct >= {K}.")
        out = REPO / f"mols10/results/ctthr_theorem_{TAG}.json"
        out.write_text(json.dumps({
            'seed': SEED, 'threshold': K, 'status': 'INFEASIBLE',
            'elapsed_s': el, 'statement': f'max ct over all mates < {K}',
            'L': L.tolist()}, indent=1))
        git_push(f"THEOREM: turn-square max-ct < {K} (INFEASIBLE, sigma-break)")
    elif st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = [i for i in range(M) if solver.value(x[i])]
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            for r in range(N):
                B[r, trans[ti][r]] = sym
        TA = TA8.astype(np.int64)
        BV = B[np.arange(N)[None, :], TA]
        S = np.sort(BV, axis=1)
        ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
        log(f"SAT: mate with ct={ct} (cl={count_clashes(L, B)})")
        out = REPO / f"mols10/results/ctthr_sat_{TAG}.json"
        out.write_text(json.dumps({'seed': SEED, 'ct': ct,
                                   'L': L.tolist(), 'B': B.tolist()}, indent=1))
        git_push(f"turn-square mate with ct={ct} >= {K} found!")
    else:
        log("UNKNOWN — rerun with longer budget")

if __name__ == "__main__":
    main()
