#!/usr/bin/env python3
"""Fish distinct high-ct mates of turn square A from random isotopy
corners, normalize each to the pairct model's symmetry-broken form,
and save a rotating hint pool for K=7/8 warm starts."""
import json, random, sys, time
import numpy as np

sys.path.insert(0, 'mols10')
from ortools.sat.python import cp_model

N = 10
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 600

def enum_trans(L):
    rows = [[int(L[r, c]) for c in range(N)] for r in range(N)]
    out = []
    def bt(row, cm, vm, path):
        if row == N:
            out.append(tuple(path)); return
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    path.append(col); bt(row+1, cm | 1 << col, vm | 1 << v, path); path.pop()
    bt(0, 0, 0, [])
    return out

def commons_of(A, B):
    rows = [[int(A[r, c]) for c in range(N)] for r in range(N)]
    brow = [[int(B[r, c]) for c in range(N)] for r in range(N)]
    outs = []
    def bt(row, cm, am, bm, path):
        if row == N:
            outs.append(tuple(path)); return
        for col in range(N):
            if not (cm >> col & 1):
                a, b = rows[row][col], brow[row][col]
                if not (am >> a & 1) and not (bm >> b & 1):
                    path.append(col); bt(row+1, cm | 1 << col, am | 1 << a, bm | 1 << b, path); path.pop()
    bt(0, 0, 0, 0, [])
    return outs

def normalize(A, B):
    cperm = np.zeros(N, dtype=int)
    for c in range(N): cperm[A[0, c]] = c
    A = A[:, cperm]; B = B[:, cperm]
    rperm = np.zeros(N, dtype=int)
    for r in range(N): rperm[A[r, 0]] = r
    A = A[rperm, :]; B = B[rperm, :]
    smap = np.zeros(N, dtype=int)
    for c in range(N): smap[B[0, c]] = c
    B = smap[B].astype(np.int8)
    return A, B

def main():
    d = json.load(open('mols10/results/turnsq_ct6_squareA.json'))
    L0 = np.array(d['L'], dtype=np.int8)
    rng = random.Random(int(time.time()))
    pool = []
    # include the existing normalized ct=6 hint
    h = json.load(open('mols10/results/pairct_hint.json'))
    pool.append(h)
    for trial in range(TRIALS):
        rp = list(range(N)); rng.shuffle(rp)
        cp_ = list(range(N)); rng.shuffle(cp_)
        sp = list(range(N)); rng.shuffle(sp)
        L = np.zeros_like(L0)
        for r in range(N):
            for c in range(N):
                L[rp[r], cp_[c]] = sp[L0[r, c]]
        trans = enum_trans(L)
        TA = np.array(trans, dtype=np.int64)
        c2t = [[[] for _ in range(N)] for _ in range(N)]
        for i, t in enumerate(trans):
            for r in range(N): c2t[r][t[r]].append(i)
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(len(trans))]
        for r in range(N):
            for c in range(N):
                model.add_exactly_one([x[i] for i in c2t[r][c]])
        found = {'best': -1, 'B': None}
        ar = np.arange(N)[None, :]
        class CB(cp_model.CpSolverSolutionCallback):
            def __init__(s):
                super().__init__(); s.t0 = time.time()
            def on_solution_callback(s):
                sel = [i for i in range(len(x)) if s.value(x[i])]
                B = np.zeros((N, N), dtype=np.int8)
                for sym, ti in enumerate(sel):
                    for r in range(N): B[r, trans[ti][r]] = sym
                BV = B[ar, TA]; S = np.sort(BV, axis=1)
                ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
                if ct > found['best']:
                    found['best'], found['B'] = ct, B.copy()
                if time.time() - s.t0 > BUDGET: s.stop_search()
        sv = cp_model.CpSolver()
        sv.parameters.enumerate_all_solutions = True
        sv.parameters.num_workers = 1
        sv.solve(model, CB())
        print(f"corner {trial}: best ct={found['best']}", flush=True)
        if found['best'] >= 5:
            An, Bn = normalize(L, np.array(found['B'], dtype=np.int8))
            cms = sorted(commons_of(An, Bn))
            pool.append({'A': An.tolist(), 'B': Bn.tolist(),
                         'commons': [list(t) for t in cms]})
    json.dump(pool, open('mols10/results/pairct_hintpool.json', 'w'))
    print(f"pool saved: {len(pool)} normalized pairs, cts="
          f"{[len(p['commons']) for p in pool]}")

if __name__ == "__main__":
    main()
