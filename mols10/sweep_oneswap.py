#!/usr/bin/env python3
"""
Randomized multi-pass ct sweep over the library of one-swap near-turn
squares (2112 transversals, sigma-broken, trivial-symmetry candidates).

DFS mate enumeration only ever samples one corner of a square's mate
space. Each pass applies a random isotopy (row/col/symbol permutation)
to the square before building the model — ct is isotopy-invariant, but
the enumeration order scrambles, so different passes sample different
mate-space corners.

Library: all sigma-breaking partial cycle swaps (len 3..9) of the known
5504-transversal base patterns. Passes loop forever, checkpointing the
global best pair and pushing records (target: ct >= 4 on an asymmetric
square; literature record over ALL pairs is 7).

Usage: python3 sweep_oneswap.py <instance> <n_inst> [stream_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_INST = int(sys.argv[2]) if len(sys.argv) > 2 else 2
STREAM_S = int(sys.argv[3]) if len(sys.argv) > 3 else 300
N = 10
LOG = REPO / f"mols10/results/sweep1s_{INSTANCE}.log"
BEST = REPO / f"mols10/results/sweep1s_best_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
BASES = ["mols10/results/turnsq_ct6_squareA.json",
         "mols10/results/turnsq_best_ct_1.json"]

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

def sigma_invariant(L):
    for r in range(N):
        for c in range(N):
            if L[r, c] != L[(r+5) % N, (c+5) % N]:
                return False
    return True

def row_cycles(L, r1, r2):
    pos2 = np.zeros(N, dtype=int)
    for c in range(N):
        pos2[L[r2, c]] = c
    p = [int(pos2[L[r1, c]]) for c in range(N)]
    seen = [False] * N
    out = []
    for c0 in range(N):
        if not seen[c0]:
            cyc = []
            c = c0
            while not seen[c]:
                seen[c] = True
                cyc.append(c)
                c = p[c]
            if 3 <= len(cyc) <= N - 1:
                out.append(cyc)
    return out

def one_swap_library():
    lib = []
    for bpath in BASES:
        seed = json.loads((REPO / bpath).read_text())
        L0 = np.array(seed['L'], dtype=np.int8)
        if not sigma_invariant(L0):
            continue
        for transpose in (False, True):
            M0 = np.ascontiguousarray(L0.T) if transpose else L0
            for r1 in range(N):
                for r2 in range(r1 + 1, N):
                    for cyc in row_cycles(M0, r1, r2):
                        M1 = M0.copy()
                        for c in cyc:
                            M1[r1, c], M1[r2, c] = M0[r2, c], M0[r1, c]
                        L1 = np.ascontiguousarray(M1.T) if transpose else M1
                        if not sigma_invariant(L1):
                            lib.append(L1)
    return lib

def random_isotopy(L, rng):
    rp = list(range(N)); rng.shuffle(rp)
    cp = list(range(N)); rng.shuffle(cp)
    sp = list(range(N)); rng.shuffle(sp)
    L2 = np.zeros_like(L)
    for r in range(N):
        for c in range(N):
            L2[rp[r], cp[c]] = sp[L[r, c]]
    return L2

def stream_ct(L, budget_s):
    trans = enum_transversals(L)
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
    rng = random.Random(31337 + INSTANCE * 7919 + int(time.time()) % 99991)
    log("=" * 70)
    log(f"One-swap ct sweep — instance={INSTANCE}/{N_INST}, stream={STREAM_S}s")
    lib = one_swap_library()
    log(f"library: {len(lib)} sigma-breaking one-swap squares")

    best = {'ct': -1}
    if BEST.exists():
        try: best = json.loads(BEST.read_text())
        except Exception: pass

    passno = 0
    while True:
        passno += 1
        order = [k for k in range(len(lib)) if k % N_INST == INSTANCE]
        rng.shuffle(order)
        for k in order:
            Lk = random_isotopy(lib[k], rng)
            n, bct, bB, hist = stream_ct(Lk, STREAM_S)
            tag = f"pass{passno} sq{k}"
            log(f"{tag}: {n} mates, best_ct={bct}, "
                f"top={dict(sorted(hist.items(), reverse=True)[:3])}")
            if bct > best.get('ct', -1):
                best = {'ct': bct, 'sq': k, 'pass': passno,
                        'L': Lk.tolist(),
                        'B': bB.tolist() if bB is not None else None}
                BEST.write_text(json.dumps(best, indent=1))
                log(f"*** NEW GLOBAL BEST ct={bct} ({tag}) ***")
                if bct >= 4:
                    git_push(f"sweep1s i{INSTANCE}: asymmetric ct={bct} (sq{k})")

if __name__ == "__main__":
    main()
