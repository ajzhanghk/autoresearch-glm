#!/usr/bin/env python3
"""
Near-turn search v2: break sigma via ROW-CYCLE SWAPS.

The 5504-transversal turn-square has only its 25 structural intercalates
(all sigma-preserving), so no single 2x2 flip leaves the excluded family.
The next-smallest Latin-preserving move is a row-pair cycle swap: for rows
(r1, r2), the permutation p = row_r2^{-1} o row_r1 decomposes into cycles;
swapping L[r1][c] <-> L[r2][c] along one cycle of length >= 3 yields a new
Latin square and generically kills sigma.

Phase 1: all (r1,r2) pairs x all cycles of length >= 3 -> transversal
         count + sigma check.  Also column-cycle swaps via transpose.
Phase 2: stream mates of the top squares (CP-SAT enumeration), score ct.

Literature targets: beat ct = 7 (record, Bright et al); ct >= 10 with
decomposable commons = 3-MOLS(10).

Usage: python3 nearturn_cycles.py <instance> <n_inst> [stream_s] [topk]
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
N_INST = int(sys.argv[2]) if len(sys.argv) > 2 else 2
STREAM_S = int(sys.argv[3]) if len(sys.argv) > 3 else 600
TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 10
LOG = REPO / f"mols10/results/nearturn2_{INSTANCE}.log"
BEST = REPO / f"mols10/results/nearturn2_best_{INSTANCE}.json"
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

def row_cycles(L, r1, r2):
    """Cycles of the permutation mapping column c to the column of row r2
    holding value L[r1][c]."""
    pos2 = np.zeros(N, dtype=int)
    for c in range(N):
        pos2[L[r2, c]] = c
    p = [int(pos2[L[r1, c]]) for c in range(N)]
    seen = [False] * N
    cycles = []
    for c0 in range(N):
        if not seen[c0]:
            cyc = []
            c = c0
            while not seen[c]:
                seen[c] = True
                cyc.append(c)
                c = p[c]
            # length-N "cycles" are full row/column transpositions, i.e.
            # isotopies: the result stays in the (MMM-excluded) turn class
            if 3 <= len(cyc) <= N - 1:
                cycles.append(cyc)
    return cycles

def apply_cycle_swap(L, r1, r2, cyc):
    L2 = L.copy()
    for c in cyc:
        L2[r1, c], L2[r2, c] = L[r2, c], L[r1, c]
    return L2

def gen_moves(L):
    """All row-cycle and column-cycle swap results (as (desc, L2))."""
    moves = []
    for r1 in range(N):
        for r2 in range(r1 + 1, N):
            for ci, cyc in enumerate(row_cycles(L, r1, r2)):
                moves.append((f"R{r1}-{r2}c{ci}len{len(cyc)}",
                              apply_cycle_swap(L, r1, r2, cyc)))
    LT = np.ascontiguousarray(L.T)
    for c1 in range(N):
        for c2 in range(c1 + 1, N):
            for ci, cyc in enumerate(row_cycles(LT, c1, c2)):
                moves.append((f"C{c1}-{c2}c{ci}len{len(cyc)}",
                              np.ascontiguousarray(
                                  apply_cycle_swap(LT, c1, c2, cyc).T)))
    return moves

def stream_mates(L, trans, budget_s):
    M = len(trans)
    TA = np.array(trans, dtype=np.int64)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    for r in range(N):
        for c in range(N):
            if cell_to_t[r][c]:
                model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            else:
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
    log(f"Near-turn CYCLE search — instance={INSTANCE}/{N_INST}, stream={STREAM_S}s")
    seed = json.loads(SEED_SQ.read_text())
    L0 = np.array(seed['L'], dtype=np.int8)
    assert sigma_invariant(L0)

    moves = gen_moves(L0)
    log(f"{len(moves)} cycle-swap moves generated")

    t0 = time.time()
    cands = []
    mine = [k for k in range(len(moves)) if k % N_INST == INSTANCE]
    for k in mine:
        desc, L1 = moves[k]
        if sigma_invariant(L1):
            continue
        n1 = count_transversals(L1)
        cands.append((n1, k, desc))
    cands.sort(reverse=True)
    log(f"phase 1: {len(cands)} sigma-breaking moves in {time.time()-t0:.0f}s; "
        f"top: {[(c[0], c[2]) for c in cands[:8]]}")

    results = []
    best_overall = {'ct': -1}
    for n1, k, desc in cands[:TOPK]:
        L1 = moves[k][1]
        trans = enum_transversals(L1)
        log(f"move {desc}: {len(trans)} transversals; streaming {STREAM_S}s...")
        n, bct, bB, hist = stream_mates(L1, trans, STREAM_S)
        log(f"  {n} mates, best_ct={bct}, hist={dict(sorted(hist.items(), reverse=True))}")
        results.append({'move': desc, 'n_trans': len(trans), 'n_mates': n,
                        'best_ct': bct,
                        'hist': {str(a): b for a, b in hist.items()}})
        if bct > best_overall['ct']:
            best_overall = {'ct': bct, 'move': desc, 'L': L1.tolist(),
                            'B': bB.tolist() if bB is not None else None,
                            'n_trans': len(trans)}
            if bct >= 7:
                log(f"*** ct={bct} >= 7: at/beyond literature record! ***")
                BEST.write_text(json.dumps({'results': results,
                                            'best': best_overall}, indent=1))
                git_push(f"near-turn CYCLE ct={bct} ({desc}, {len(trans)} trans)")
        BEST.write_text(json.dumps({'results': results, 'best': best_overall}, indent=1))

    git_push(f"near-turn cycles i{INSTANCE}: best_ct={best_overall['ct']}")
    log(f"DONE: best_ct={best_overall['ct']}")

if __name__ == "__main__":
    main()
