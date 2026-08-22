#!/usr/bin/env python3
"""
Purpose-built CNF encoding of k-MOLS(n) for kissat -- far stronger propagation
than a generic integer AllDifferent model, and kissat typically beats CP-SAT
on hard combinatorial SAT.

Boolean one-hot symbol variables s[t][r][c][a] = "square t has symbol a at (r,c)".
Latin: exactly-one symbol per cell; each symbol once per row and per column.
Orthogonality of squares t<u: pair variables p[t,u][r][c][a][b] channel
(s_t=a & s_u=b); each ordered symbol pair (a,b) occurs in exactly one cell.
Symmetry breaking: square 0 reduced (row0=id, col0=id); squares 1..k-1 row0=id;
order squares by L[t][1][0] increasing.

A SAT solution IS a k-MOLS(n); for n=10,k=3 that is a 3-MOLS(10) => N(10)>=3.
UNSAT (with sound symmetry breaking) => none in canonical form => N(10)=2.

Usage: python3 direct_triple_cnf.py <n> <k> [timeout_s] [seed]
Writes/streams to kissat; on SAT decodes+verifies and (for n=10,k=3) writes
MOLS10_FOUND.json; on UNSAT writes direct_cnf_UNSAT.json.
"""
import sys, subprocess, itertools, json, time
from datetime import datetime
from pathlib import Path
from pysat.solvers import Cadical195 as SatSolver

REPO = Path("/home/user/autoresearch-glm")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else int(time.time()) % 999999
SCRATCH = Path("/tmp") / f"mols_cnf_{n}_{k}"
LOG = REPO / "mols10/results/direct_cnf.log"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f: f.write(line + "\n")
    except Exception: pass

class CNF:
    def __init__(s): s.nv = 0; s.cl = []
    def var(s): s.nv += 1; return s.nv
    def add(s, c): s.cl.append(c)
    def exactly_one(s, lits):
        s.add(list(lits))                      # at least one
        for a, b in itertools.combinations(lits, 2): s.add([-a, -b])  # at most one

def build(n, k):
    cnf = CNF()
    # s[t][r][c][a]
    s = [[[[cnf.var() for a in range(n)] for c in range(n)]
          for r in range(n)] for t in range(k)]
    for t in range(k):
        for r in range(n):
            for c in range(n):
                cnf.exactly_one(s[t][r][c])           # one symbol per cell
            for a in range(n):
                cnf.exactly_one([s[t][r][c][a] for c in range(n)])  # symbol once per row
        for c in range(n):
            for a in range(n):
                cnf.exactly_one([s[t][r][c][a] for r in range(n)])  # symbol once per col
    # orthogonality via pair channeling
    for t in range(k):
        for u in range(t + 1, k):
            # pair[r][c][a][b] <=> s_t=a & s_u=b ; each (a,b) exactly once over cells
            for a in range(n):
                for b in range(n):
                    occ = []
                    for r in range(n):
                        for c in range(n):
                            p = cnf.var()
                            cnf.add([-p, s[t][r][c][a]])
                            cnf.add([-p, s[u][r][c][b]])
                            cnf.add([p, -s[t][r][c][a], -s[u][r][c][b]])
                            occ.append(p)
                    cnf.exactly_one(occ)
    # symmetry breaking (unit clauses)
    for c in range(n): cnf.add([s[0][0][c][c]])   # sq0 row0 = identity
    for r in range(n): cnf.add([s[0][r][0][r]])   # sq0 col0 = identity
    for t in range(1, k):
        for c in range(n): cnf.add([s[t][0][c][c]])  # row0 identity for all squares
    # order squares by symbol at (1,0): L[t][1][0] < L[t+1][1][0]  (encode via <=)
    # forbid L[t][1][0] >= L[t+1][1][0]: not( s[t][1][0][a] & s[t+1][1][0][b] ) for a>=b
    for t in range(1, k - 1):
        for a in range(n):
            for b in range(n):
                if a >= b:
                    cnf.add([-s[t][1][0][a], -s[t + 1][1][0][b]])
    return cnf, s

def main():
    log("=" * 60)
    log(f"CNF {k}-MOLS({n}) via Cadical  timeout={TIMEOUT}s seed={SEED}")
    cnf, s = build(n, k)
    log(f"CNF: {cnf.nv} vars, {len(cnf.cl)} clauses")
    solver = SatSolver(bootstrap_with=cnf.cl)
    try: solver.set_phases(seed=SEED)
    except Exception: pass
    t0 = time.time()
    solver.conf_budget(-1)
    # run with a wall-clock cap via a timer thread
    import threading
    done = {'r': None}
    def _solve(): done['r'] = solver.solve()
    th = threading.Thread(target=_solve); th.start()
    th.join(TIMEOUT)
    el = time.time() - t0
    if th.is_alive():
        solver.interrupt(); th.join(5); done['r'] = None
    res = done['r']
    if res is True:
        model = set(v for v in solver.get_model() if v > 0)
        def sym(t, r, c):
            for a in range(n):
                if s[t][r][c][a] in model: return a
            return -1
        L = [[[sym(t, r, c) for c in range(n)] for r in range(n)] for t in range(k)]
        # verify Latin + pairwise orthogonal
        def latin(M):
            return all(len(set(M[r]))==n for r in range(n)) and \
                   all(len({M[r][c] for r in range(n)})==n for c in range(n))
        def orth(A, B):
            return len({(A[r][c], B[r][c]) for r in range(n) for c in range(n)}) == n*n
        ok = all(latin(L[t]) for t in range(k)) and \
             all(orth(L[t], L[u]) for t in range(k) for u in range(t+1, k))
        log(f"SAT in {el:.0f}s; verified={ok}")
        if ok and n == 10 and k == 3:
            log("*** 3-MOLS(10) FOUND — N(10) >= 3 COUNTEREXAMPLE! ***")
            (REPO/"mols10/results/MOLS10_FOUND.json").write_text(json.dumps(
                {'found':True,'method':'cnf_kissat','L1':L[0],'L2':L[1],'L3':L[2]}, indent=2))
            subprocess.run(["git","-C",str(REPO),"add","mols10/results/"], check=False)
            subprocess.run(["git","-C",str(REPO),"commit","-m",
                            "3-MOLS(10) COUNTEREXAMPLE FOUND via CNF/kissat!"], capture_output=True)
            subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH], capture_output=True)
            subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], capture_output=True)
        elif ok:
            log(f"validated {k}-MOLS({n}) found (sanity ok)")
            for t in range(k): log(f"  L{t+1}={L[t]}")
    elif res is False:
        log(f"UNSAT in {el:.0f}s")
        if n == 10 and k == 3:
            (REPO/"mols10/results/direct_cnf_UNSAT.json").write_text(json.dumps(
                {'status':'UNSAT','elapsed_s':el}, indent=1))
    else:
        log(f"no verdict in {el:.0f}s (timeout)")
    solver.delete()

if __name__ == "__main__":
    main()
