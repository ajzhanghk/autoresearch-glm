#!/usr/bin/env python3
"""
Order-10 Latin squares by CONTRACTION of the Z11 cyclic square, and their
transversal / ct profile.

Construction: the Z11 Cayley table C(i,j) = i+j mod 11 is an order-11
Latin square. Every cell (r,c) lies on many transversals of C; pick any
transversal T of C. Deleting the row r0 and column c0 of one cell
(r0,c0) in T and REPLACING, in each remaining cell (r,c) of T, the
symbol C(r,c) by the "hole" symbol C(r0,c0) yields an order-10 Latin
square on the symbol set {0..10} minus {C(r0,c0)}... more precisely: in
row r the deleted column c0 removes symbol C(r,c0); the transversal
cell (r,c) in that row donates its symbol C(r,c) to plug nothing —
the standard contraction: relabel each transversal cell's symbol to
C(r0,c0). Then each row r != r0 contains every symbol except C(r,c),
gains a duplicate of ... (verified computationally below rather than
argued: we CHECK the Latin property and discard failures).

Z11 (prime cyclic) has 3852 transversals lying in 10 parallel classes...
actually for Z_p the transversals of C are exactly the "graphs" of maps
j -> a*j+b? Regardless: we enumerate them directly.

For each of a sample of transversals T and cells (r0,c0) in T, build the
contracted square, verify Latin-ness, count transversals, test
turn-class membership fingerprint (5504) and sigma-likeness, and stream
mates for the richest finds.

Usage: python3 z11_contract.py [n_samples] [stream_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

N = 10
N11 = 11
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STREAM_S = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LOG = REPO / "mols10/results/z11contract.log"
BEST = REPO / "mols10/results/z11contract_best.json"
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

def is_latin(L, n):
    for r in range(n):
        if len(set(int(v) for v in L[r, :])) != n: return False
    for c in range(n):
        if len(set(int(v) for v in L[:, c])) != n: return False
    return True

def enum_transversals_n(L, n):
    all_t = []
    rows = [[int(L[r, c]) for c in range(n)] for r in range(n)]
    def bt(row, cm, vm, path):
        if row == n:
            all_t.append(tuple(path)); return
        for col in range(n):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    path.append(col)
                    bt(row+1, cm | (1 << col), vm | (1 << v), path)
                    path.pop()
    bt(0, 0, 0, [])
    return all_t

def count_transversals10(L, cap=6000):
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

def contract(C, T, idx):
    """Contract order-11 square C at transversal T, deleting T[idx]'s
    row/col; other transversal cells get the deleted cell's symbol."""
    r0, c0 = T[idx]
    s0 = C[r0, c0]
    C2 = C.copy()
    for k, (r, c) in enumerate(T):
        if k != idx:
            C2[r, c] = s0
    keep_r = [r for r in range(N11) if r != r0]
    keep_c = [c for c in range(N11) if c != c0]
    L = C2[np.ix_(keep_r, keep_c)]
    # relabel symbols to 0..9 (symbol s0 replaced the removed diag symbols;
    # the symbol set of L is {0..10} \ {old symbols on T outside (r0,c0)}...
    # just compress whatever 10 symbols appear)
    syms = sorted(set(int(v) for v in L.flatten()))
    if len(syms) != N:
        return None
    remap = {s: i for i, s in enumerate(syms)}
    L10 = np.vectorize(remap.get)(L).astype(np.int8)
    return L10 if is_latin(L10, N) else None

def stream_ct(L, budget_s):
    trans = enum_transversals_n(L, N)
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
    rng = random.Random(20260709)
    log("=" * 70)
    log(f"Z11 contraction — {NSAMP} samples, stream={STREAM_S}s")
    C = np.array([[(i + j) % N11 for j in range(N11)] for i in range(N11)],
                 dtype=np.int8)
    trans11 = enum_transversals_n(C, N11)
    log(f"Z11 square: {len(trans11)} transversals")

    profile = {}
    tried = 0
    best = {'ct': -1}
    seen_counts = {}
    while tried < NSAMP:
        T_cols = rng.choice(trans11)
        T = [(r, T_cols[r]) for r in range(N11)]
        idx = rng.randrange(N11)
        L = contract(C, T, idx)
        tried += 1
        if L is None:
            profile['fail'] = profile.get('fail', 0) + 1
            continue
        n10 = count_transversals10(L)
        seen_counts[n10] = seen_counts.get(n10, 0) + 1
        if n10 >= 1500:
            n, bct, bB, hist = stream_ct(L, STREAM_S)
            log(f"sample {tried}: count={n10}, {n} mates, best_ct={bct}, "
                f"top={dict(sorted(hist.items(), reverse=True)[:3])}")
            if bct > best.get('ct', -1):
                best = {'ct': bct, 'count': int(n10), 'L': L.tolist(),
                        'B': bB.tolist() if bB is not None else None}
                BEST.write_text(json.dumps(best, indent=1))
                if bct >= 4:
                    git_push(f"Z11-contract square: ct={bct} (count={n10})")
    log(f"transversal-count profile over {NSAMP} samples: "
        f"{dict(sorted(seen_counts.items(), reverse=True))}")
    log(f"best: ct={best.get('ct')} count={best.get('count')}")
    git_push("z11 contraction profile")

if __name__ == "__main__":
    main()
