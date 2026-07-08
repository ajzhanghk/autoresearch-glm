#!/usr/bin/env python3
"""
Near-turn-square search: break the sigma symmetry of a 5504-transversal
turn-square by turning extra (non-structural) intercalates, keeping the
transversal count as high as possible, then stream orthogonal mates and
score ct.

Rationale (literature): McKay-Meynert-Myrvold 2007 prove every square in
a 3-MOLS(10) has trivial autoparatopism group; turn-squares all have the
nontrivial autotopism sigma: (r,c)->(r+5,c+5). So the triple hunt must
leave the turn family — but its immediate IC-neighborhood keeps most of
the transversal richness while (generically) breaking sigma. Literature
ct record for MOLS(10) pairs is 7 (previously 4); target: beat 7, dream:
reach >= 10 with decomposable commons.

Phase 1: enumerate all intercalates of the base square; for each flip
         (or flip pair, depth 2), count transversals + check sigma broken.
Phase 2: for the top squares, stream mates (CP-SAT enumeration, budget)
         scoring ct vectorized; checkpoint best pairs; git-push records.

Usage: python3 nearturn_search.py <instance> [depth] [stream_budget_s]
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
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
STREAM_S = int(sys.argv[3]) if len(sys.argv) > 3 else 600
LOG = REPO / f"mols10/results/nearturn_{INSTANCE}.log"
BEST = REPO / f"mols10/results/nearturn_best_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
SEED_SQ = REPO / "mols10/results/turnsq_ct6_squareA.json"

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

def count_transversals(L, cap=6000):
    rows = [[int(L[r, c]) for c in range(N)] for r in range(N)]
    cnt = [0]
    def bt(row, cm, vm):
        if row == N:
            cnt[0] += 1
            return cnt[0] >= cap
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    if bt(row+1, cm | (1 << col), vm | (1 << v)):
                        return True
        return False
    bt(0, 0, 0)
    return cnt[0]

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

def sigma_invariant(L):
    for r in range(N):
        for c in range(N):
            if L[r, c] != L[(r+5) % N, (c+5) % N]:
                return False
    return True

def enum_intercalates(L):
    ics = []
    for r1 in range(N):
        for r2 in range(r1+1, N):
            for c1 in range(N):
                for c2 in range(c1+1, N):
                    if (L[r1,c1] == L[r2,c2] and L[r1,c2] == L[r2,c1]
                            and L[r1,c1] != L[r1,c2]):
                        ics.append((r1,r2,c1,c2))
    return ics

def flip_ic(L, ic):
    r1,r2,c1,c2 = ic
    L2 = L.copy()
    L2[r1,c1], L2[r1,c2] = L[r1,c2], L[r1,c1]
    L2[r2,c1], L2[r2,c2] = L[r2,c2], L[r2,c1]
    return L2

def stream_mates(L, trans, budget_s):
    """Enumerate mates via CP-SAT; return (n_mates, best_ct, best_B, hist)."""
    M = len(trans)
    TA = np.array(trans, dtype=np.int64)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    feasible = True
    for r in range(N):
        for c in range(N):
            if cell_to_t[r][c]: model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            else: feasible = False
    if not feasible:
        return 0, -1, None, {}

    state = {'n': 0, 'best_ct': -1, 'best_B': None, 'hist': {}}
    ar = np.arange(N)[None, :]

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.time()
        def on_solution_callback(self):
            sel = [i for i in range(M) if self.value(x[i])]
            B = np.zeros((N, N), dtype=np.int8)
            for sym, ti in enumerate(sel):
                t = trans[ti]
                for r in range(N):
                    B[r, t[r]] = sym
            BV = B[ar, TA]
            S = np.sort(BV, axis=1)
            ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
            state['n'] += 1
            state['hist'][ct] = state['hist'].get(ct, 0) + 1
            if ct > state['best_ct']:
                state['best_ct'] = ct
                state['best_B'] = B.copy()
            if time.time() - self.t0 > budget_s:
                self.stop_search()

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    solver.solve(model, CB())
    return state['n'], state['best_ct'], state['best_B'], state['hist']

def main():
    log("=" * 70)
    log(f"Near-turn search — instance={INSTANCE}, depth={DEPTH}, stream={STREAM_S}s")
    seed = json.loads(SEED_SQ.read_text())
    L0 = np.array(seed['L'], dtype=np.int8)
    assert sigma_invariant(L0)
    ics = enum_intercalates(L0)
    log(f"base square: 5504 transversals, {len(ics)} intercalates")

    # Phase 1: rank sigma-breaking flips by transversal count
    cands = []
    t0 = time.time()
    mine = [k for k in range(len(ics)) if k % 2 == INSTANCE]  # stripe by instance
    for k in mine:
        L1 = flip_ic(L0, ics[k])
        if sigma_invariant(L1):
            continue
        n1 = count_transversals(L1)
        cands.append((n1, k))
    cands.sort(reverse=True)
    log(f"phase 1 done in {time.time()-t0:.0f}s: {len(cands)} sigma-breaking flips; "
        f"top counts: {[c[0] for c in cands[:10]]}")

    results = []
    best_overall = {'ct': -1}
    for n1, k in cands[:12]:
        L1 = flip_ic(L0, ics[k])
        trans = enum_transversals(L1)
        log(f"flip {k} (ic={ics[k]}): {len(trans)} transversals; streaming {STREAM_S}s...")
        n, bct, bB, hist = stream_mates(L1, trans, STREAM_S)
        log(f"  {n} mates, best_ct={bct}, hist={dict(sorted(hist.items(), reverse=True))}")
        results.append({'flip': k, 'ic': list(ics[k]), 'n_trans': len(trans),
                        'n_mates': n, 'best_ct': bct,
                        'hist': {str(a): b for a, b in hist.items()}})
        if bct > best_overall['ct']:
            best_overall = {'ct': bct, 'flip': k, 'L': L1.tolist(),
                            'B': bB.tolist() if bB is not None else None,
                            'n_trans': len(trans)}
            BEST.write_text(json.dumps({'results': results, 'best': best_overall}, indent=1))
            if bct >= 7:
                log(f"*** ct={bct} >= 7: at/beyond literature record! ***")
                git_push(f"near-turn ct={bct} (flip {k}, {len(trans)} transversals)")
        BEST.write_text(json.dumps({'results': results, 'best': best_overall}, indent=1))

    git_push(f"near-turn i{INSTANCE} phase complete: best_ct={best_overall['ct']}")
    log(f"DONE: best_ct={best_overall['ct']}")

if __name__ == "__main__":
    main()
