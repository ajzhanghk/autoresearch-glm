#!/usr/bin/env python3
"""
DIRECT pair+witnesses model: find a MOLS(10) pair (A,B) admitting at
least K common transversals, with the K witnesses as first-class model
objects. The solver constructs the squares AROUND the required common
transversals, bypassing the transversal-richness bottleneck of
square-first search.

Variables:
  A[r][c], B[r][c] in 0..9      (flattened to 100-vectors)
  rows/cols Latin: alldiff per row and column of each square
  orthogonality: alldiff over the 100 codes 10*A+B
  witness w in 0..K-1: columns wcol[w][r] with alldiff (permutation);
    values av[w][r] = A[r][wcol[w][r]], bv[w][r] = B[r][wcol[w][r]]
    via element constraints; alldiff(av[w]), alldiff(bv[w]).
Symmetry breaking: A row 0 = 0..9; A col 0 = 0..9 (reduced square);
  B row 0 = 0..9; witness 0's column vector lex-min... witnesses ordered
  by their column in row 0: wcol[w][0] strictly increasing.

K = 8 SAT would beat the literature record of 7 common transversals.
K = 10 SAT meets the necessary condition of a 3-MOLS(10); its commons
are then checked for a decomposition (third square) separately.

On SAT: verify everything from scratch in Python (Latin, orthogonal,
count ALL common transversals by enumeration), save JSON, push.

Usage: python3 pairct_direct.py <K> [timeout_s] [workers]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 14400
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N = 10
LOG = REPO / f"mols10/results/pairct_K{K}.log"
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

def enum_common_transversals(A, B):
    """Count transversals of A that are also transversals of B."""
    rows = [[int(A[r, c]) for c in range(N)] for r in range(N)]
    browz = [[int(B[r, c]) for c in range(N)] for r in range(N)]
    cnt = [0]
    def bt(row, cm, am, bm):
        if row == N:
            cnt[0] += 1; return
        for col in range(N):
            if not (cm >> col & 1):
                a = rows[row][col]; b = browz[row][col]
                if not (am >> a & 1) and not (bm >> b & 1):
                    bt(row+1, cm | 1 << col, am | 1 << a, bm | 1 << b)
    bt(0, 0, 0, 0)
    return cnt[0]

def main():
    log("=" * 70)
    log(f"DIRECT pair-ct model — K={K}, timeout={TIMEOUT}s, workers={WORKERS}")
    m = cp_model.CpModel()
    A = [m.new_int_var(0, N-1, f"A{i}") for i in range(N*N)]
    B = [m.new_int_var(0, N-1, f"B{i}") for i in range(N*N)]

    for r in range(N):
        m.add_all_different([A[r*N + c] for c in range(N)])
        m.add_all_different([B[r*N + c] for c in range(N)])
    for c in range(N):
        m.add_all_different([A[r*N + c] for r in range(N)])
        m.add_all_different([B[r*N + c] for r in range(N)])

    # orthogonality: codes 10*A+B all different
    codes = []
    for i in range(N*N):
        code = m.new_int_var(0, N*N - 1, f"p{i}")
        m.add(code == A[i] * N + B[i])
        codes.append(code)
    m.add_all_different(codes)

    # symmetry breaking: A reduced (row0 = col0 = identity), B row0 = identity
    for c in range(N):
        m.add(A[c] == c)
        m.add(B[c] == c)
    for r in range(N):
        m.add(A[r*N] == r)

    # K witness common transversals
    wcols = []
    for w in range(K):
        wcol = [m.new_int_var(0, N-1, f"w{w}c{r}") for r in range(N)]
        m.add_all_different(wcol)
        av = [m.new_int_var(0, N-1, f"w{w}a{r}") for r in range(N)]
        bv = [m.new_int_var(0, N-1, f"w{w}b{r}") for r in range(N)]
        for r in range(N):
            m.add_element(wcol[r], [A[r*N + c] for c in range(N)], av[r])
            m.add_element(wcol[r], [B[r*N + c] for c in range(N)], bv[r])
        m.add_all_different(av)
        m.add_all_different(bv)
        wcols.append(wcol)

    # strict lexicographic ordering between adjacent witnesses (complete
    # symmetry break; distinct transversals may share row-0 columns, so
    # ordering by any single row is unsound)
    for w in range(K - 1):
        u, v = wcols[w], wcols[w + 1]
        eq_prefix = m.new_bool_var(f"lex{w}_start")
        m.add(eq_prefix == 1)
        for r in range(N):
            b_eq = m.new_bool_var(f"lex{w}eq{r}")
            m.add(u[r] == v[r]).only_enforce_if(b_eq)
            m.add(u[r] != v[r]).only_enforce_if(b_eq.Not())
            m.add(u[r] <= v[r]).only_enforce_if(eq_prefix)
            nxt = m.new_bool_var(f"lex{w}pre{r}")
            m.add_bool_and([eq_prefix, b_eq]).only_enforce_if(nxt)
            m.add_bool_or([eq_prefix.Not(), b_eq.Not()]).only_enforce_if(nxt.Not())
            eq_prefix = nxt
        m.add(eq_prefix == 0)  # strict: witnesses distinct

    # warm start from the normalized ct=6 turn pair (squares + up to 6 witnesses)
    hint_file = REPO / "mols10/results/pairct_hint.json"
    if hint_file.exists():
        h = json.loads(hint_file.read_text())
        Ah, Bh = h['A'], h['B']
        for i in range(N * N):
            m.add_hint(A[i], Ah[i // N][i % N])
            m.add_hint(B[i], Bh[i // N][i % N])
        commons = sorted(tuple(t) for t in h['commons'])
        for w in range(min(K, len(commons))):
            for r in range(N):
                m.add_hint(wcols[w][r], commons[w][r])
        log(f"hinted with normalized ct={len(commons)} pair")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIMEOUT
    solver.parameters.num_workers = WORKERS
    # fresh trajectory each (re)launch: container restarts make long runs
    # Las Vegas anyway, so embrace seed rotation
    solver.parameters.random_seed = int(time.time()) % 2_000_000_000
    t0 = time.time()
    st = solver.solve(m)
    el = time.time() - t0
    log(f"STATUS: {solver.status_name(st)} in {el:.0f}s")

    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        Av = np.array([solver.value(A[i]) for i in range(N*N)],
                      dtype=np.int8).reshape(N, N)
        Bv = np.array([solver.value(B[i]) for i in range(N*N)],
                      dtype=np.int8).reshape(N, N)
        # independent verification
        okA = all(len(set(map(int, Av[r]))) == N for r in range(N)) and \
              all(len(set(map(int, Av[:, c]))) == N for c in range(N))
        okB = all(len(set(map(int, Bv[r]))) == N for r in range(N)) and \
              all(len(set(map(int, Bv[:, c]))) == N for c in range(N))
        orth = len({(int(Av[r, c]), int(Bv[r, c]))
                    for r in range(N) for c in range(N)}) == N*N
        ct = enum_common_transversals(Av, Bv)
        log(f"VERIFY: latinA={okA} latinB={okB} orth={orth} full_ct={ct}")
        if okA and okB and orth and ct >= K:
            out = REPO / f"mols10/results/pairct_K{K}_SAT.json"
            out.write_text(json.dumps({
                'K': K, 'ct_full': ct,
                'A': Av.tolist(), 'B': Bv.tolist()}, indent=1))
            log(f"*** PAIR WITH ct={ct} SAVED (K={K}) ***")
            git_push(f"DIRECT MODEL: MOLS(10) pair with ct={ct} (K={K} witnesses)"
                     + (" — BEATS LITERATURE RECORD 7" if ct >= 8 else ""))
    elif st == cp_model.INFEASIBLE:
        log(f"INFEASIBLE: no MOLS(10) pair has {K} common transversals (!!)")
        out = REPO / f"mols10/results/pairct_K{K}_INFEASIBLE.json"
        out.write_text(json.dumps({'K': K, 'status': 'INFEASIBLE',
                                   'elapsed': el}, indent=1))
        git_push(f"THEOREM(?): no MOLS(10) pair with {K} common transversals")

if __name__ == "__main__":
    main()
