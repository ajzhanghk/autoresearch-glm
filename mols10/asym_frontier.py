#!/usr/bin/env python3
"""
The genuine asymmetric frontier: maximize the transversal count (and hence
the reachable CT) over order-10 Latin squares with TRIVIAL autotopism
group -- the only squares eligible to sit in a 3-MOLS(10) by the
McKay-Meynert-Myrvold theorem.

Discovery motivating this: every square whose CT we pushed above 2
(one-swap=3, turn=6, corpus ct>=4) turned out to have a nontrivial
autotopism, hence is MMM-excluded. Generic trivial-autotopism squares
sit at ~800 transversals with CT<=2. The open question: what is the
MAXIMUM transversal count of a trivial-autotopism square, and does any
such square reach CT>=10 (a triple)?

Method: SA over partial row/column cycle swaps maximizing the transversal
count. Every accepted move is gated to STAY in trivial-autotopism
territory (moves landing on a nontrivial-autotopism square are rejected),
so the walk never leaves the MMM-eligible region. New transversal-count
records get a CP-SAT mate stream (CT histogram). Any CT>=10 with
decomposable commons is a 3-MOLS(10); the square is asymmetric by
construction, so it would be a genuine counterexample to N(10)=2.

Usage: python3 asym_frontier.py <instance> [stream_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from autotopism import has_nontrivial_autotopism
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STREAM_S = int(sys.argv[2]) if len(sys.argv) > 2 else 240
N = 10
LOG = REPO / f"mols10/results/asymfront_{INSTANCE}.log"
BEST = REPO / f"mols10/results/asymfront_best_{INSTANCE}.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
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
        subprocess.run(["git","-C",str(REPO),"push","origin",BRANCH],
                       capture_output=True, check=False)

def count_trans(L, cap=6000):
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

def enum_trans(L):
    out = []
    rows = [[int(L[r, c]) for c in range(N)] for r in range(N)]
    def bt(row, cm, vm, p):
        if row == N:
            out.append(tuple(p)); return
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    p.append(col); bt(row+1, cm | (1 << col), vm | (1 << v), p); p.pop()
    bt(0, 0, 0, [])
    return out

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
            if 2 <= len(cyc) <= N:
                out.append(cyc)
    return out

def random_move(L, rng):
    for _ in range(40):
        transpose = rng.random() < 0.5
        M0 = np.ascontiguousarray(L.T) if transpose else L
        r1, r2 = rng.sample(range(N), 2)
        cyc = rng.choice(row_cycles(M0, r1, r2))
        if len(cyc) < 2:
            continue
        M1 = M0.copy()
        for c in cyc:
            M1[r1, c], M1[r2, c] = M0[r2, c], M0[r1, c]
        return np.ascontiguousarray(M1.T) if transpose else M1
    return None

def stream_ct(L, budget_s):
    trans = enum_trans(L)
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
    st = {'n': 0, 'best': -1, 'B': None, 'hist': {}}
    ar = np.arange(N)[None, :]
    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__(); self.t0 = time.time()
        def on_solution_callback(self):
            sel = [i for i in range(M) if self.value(x[i])]
            B = np.zeros((N, N), dtype=np.int8)
            for sym, ti in enumerate(sel):
                for r in range(N):
                    B[r, trans[ti][r]] = sym
            BV = B[ar, TA]; S = np.sort(BV, axis=1)
            ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
            st['n'] += 1; st['hist'][ct] = st['hist'].get(ct, 0) + 1
            if ct > st['best']:
                st['best'] = ct; st['B'] = B.copy()
            if time.time() - self.t0 > budget_s:
                self.stop_search()
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    solver.solve(model, CB())
    return st['n'], st['best'], st['B'], st['hist']

def main():
    rng = random.Random(12345 + INSTANCE * 6791 + int(time.time()) % 90001)
    log("=" * 70)
    log(f"ASYMMETRIC FRONTIER — instance={INSTANCE}, stream={STREAM_S}s")
    # randomize a starting asymmetric square
    L = np.array([[(i + j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    for _ in range(400):
        m = random_move(L, rng)
        if m is not None:
            L = m
    while has_nontrivial_autotopism(L):
        m = random_move(L, rng)
        if m is not None:
            L = m
    cur = count_trans(L)
    best = cur
    best_ct = -1
    log(f"start: {cur} transversals, trivial-autotopism")

    T = 60.0
    step = 0
    last_push = time.time()
    while True:
        step += 1
        L2 = random_move(L, rng)
        if L2 is None:
            continue
        n2 = count_trans(L2)
        if n2 >= cur or rng.random() < np.exp((n2 - cur) / T):
            # gate: stay in trivial-autotopism territory
            if not has_nontrivial_autotopism(L2):
                L, cur = L2, n2
        T = max(3.0, T * 0.9997)

        if cur > best:
            best = cur
            log(f"step {step}: NEW ASYM transversal record {cur} (T={T:.1f})")
            if cur >= 1000:
                n, bct, bB, hist = stream_ct(L, STREAM_S)
                log(f"  CT stream: {n} mates, best_ct={bct}, "
                    f"top={dict(sorted(hist.items(), reverse=True)[:4])}")
                rec = {'transversals': int(cur), 'stream_best_ct': bct,
                       'L': L.tolist(), 'B': bB.tolist() if bB is not None else None,
                       'trivial_autotopism': True}
                BEST.write_text(json.dumps(rec, indent=1))
                if bct > best_ct:
                    best_ct = bct
                    if bct >= 6:
                        git_push(f"ASYM FRONTIER: trivial-autotopism ct={bct} "
                                 f"({cur} transversals)!")
                    if bct >= 10:
                        log("*** trivial-autotopism ct>=10 — attempting triple ***")
        if step % 100 == 0:
            log(f"step {step}: cur={cur} best={best} T={T:.1f}")
        if time.time() - last_push > 1800:
            git_push(f"asym frontier i{INSTANCE}: best={best} transversals, "
                     f"best_ct={best_ct}")
            last_push = time.time()

if __name__ == "__main__":
    main()
