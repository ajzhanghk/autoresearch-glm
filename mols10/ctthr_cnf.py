#!/usr/bin/env python3
"""
Pure-SAT (kissat) encoding of the threshold decision: does square L have
an orthogonal mate B with ct(L,B) >= K?

CNF structure:
  x_i  (i < M): transversal i selected in the mate decomposition
  y_j  (j < M): transversal j counted as a common transversal
  - exact-one over each cell's covering transversals
    (ALO clause + sequential AMO ladder)
  - y_j -> not x_i for every i with |t_i ^ t_j| >= 2 (self included)
    [~8M binary clauses — kissat's home turf]
  - sequential counter enforcing sum(y) >= K
Solution decoding + independent numpy verification; artifacts saved and
pushed like the CP-SAT variant.

Usage: python3 ctthr_cnf.py <seed_json> <K> [kissat_seed] [timeout_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

KISSAT = ("/tmp/claude-0/-home-user-autoresearch-glm/"
          "6c15e52c-04fd-5e17-80ad-627a8c92c1f0/scratchpad/"
          "Myrvold-MOLS/kissat/build/kissat")

SEED = sys.argv[1]
K = int(sys.argv[2])
KSEED = int(sys.argv[3]) if len(sys.argv) > 3 else int(time.time()) % 999999
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 10800
TAG = f"cnf_{Path(SEED).stem}_ge{K}"
LOG = REPO / f"mols10/results/ctthr_{TAG}.log"
BRANCH = "claude/mols-order-10-search-yfQXK"
SCRATCH = Path("/tmp") / f"ctthr_cnf_ge{K}"

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
    log(f"kissat CNF ct >= {K} — {SEED}, seed={KSEED}, timeout={TIMEOUT}s")
    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    log(f"{M} transversals")

    TA8 = np.array(trans, dtype=np.int8)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    # variable numbering (DIMACS, 1-based)
    X = lambda i: 1 + i                 # x_i
    Y = lambda j: 1 + M + j             # y_j
    next_var = [1 + 2 * M]
    def new_var():
        v = next_var[0]; next_var[0] += 1
        return v

    t0 = time.time()
    clauses = []

    # exact-one per cell: ALO + sequential AMO
    for r in range(N):
        for c in range(N):
            lst = cell_to_t[r][c]
            clauses.append([X(i) for i in lst])
            # sequential AMO ladder
            n = len(lst)
            if n > 1:
                s_prev = None
                for idx in range(n - 1):
                    s = new_var()
                    clauses.append([-X(lst[idx]), s])
                    if s_prev is not None:
                        clauses.append([-s_prev, s])
                        clauses.append([-s_prev, -X(lst[idx])])
                    s_prev = s
                clauses.append([-s_prev, -X(lst[n - 1])])

    # y_j -> not x_i for i in bad(j) (I >= 2, self included)
    n_bin = 0
    for lo in range(0, M, 256):
        hi = min(lo + 256, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            j = lo + k
            for i in np.nonzero(cnt[k] >= 2)[0]:
                clauses.append([-Y(j), -X(int(i))])
                n_bin += 1

    # sum(y) >= K via sequential counter with SOUNDNESS direction:
    # s[j][k] = "y_0..y_j contain >= k+1 trues"; clauses ensure s can only
    # be true when justified, and the final unit forces s[M-1][K-1].
    #   s[0][0] -> y_0 ;  s[0][k>=1] = false
    #   s[j][k] -> s[j-1][k] v y_j
    #   s[j][k] -> s[j-1][k] v s[j-1][k-1]   (k >= 1)
    prev = None
    for j in range(M):
        cur = [new_var() for _ in range(K)]
        if prev is None:
            clauses.append([-cur[0], Y(0)])
            for k in range(1, K):
                clauses.append([-cur[k]])
        else:
            for k in range(K):
                clauses.append([-cur[k], prev[k], Y(j)])
                if k >= 1:
                    clauses.append([-cur[k], prev[k], prev[k-1]])
        prev = cur
    clauses.append([prev[K-1]])   # at least K trues overall

    nv = next_var[0] - 1
    log(f"CNF built: {nv} vars, {len(clauses)} clauses "
        f"({n_bin} indicator binaries) in {time.time()-t0:.0f}s")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    cnf_path = SCRATCH / "problem.cnf"
    with open(cnf_path, "w") as f:
        f.write(f"p cnf {nv} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
    log(f"DIMACS written: {cnf_path} "
        f"({cnf_path.stat().st_size/1e6:.0f} MB)")

    t0 = time.time()
    proc = subprocess.run(
        [KISSAT, f"--time={TIMEOUT}", f"--seed={KSEED}", str(cnf_path)],
        capture_output=True, text=True)
    el = time.time() - t0
    out = proc.stdout
    if "s SATISFIABLE" in out:
        log(f"kissat: SATISFIABLE in {el:.0f}s")
        assign = set()
        for line in out.splitlines():
            if line.startswith("v "):
                for tok in line[2:].split():
                    v = int(tok)
                    if v > 0:
                        assign.add(v)
        sel = [i for i in range(M) if X(i) in assign]
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            for r in range(N):
                B[r, trans[ti][r]] = sym
        TA = TA8.astype(np.int64)
        BV = B[np.arange(N)[None, :], TA]
        S = np.sort(BV, axis=1)
        ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
        cl0 = count_clashes(L, B)
        log(f"decoded mate: ct={ct}, cl={cl0} (need ct >= {K}, cl == 0)")
        if cl0 == 0 and ct >= K:
            outp = REPO / f"mols10/results/ctthr_sat_{TAG}.json"
            outp.write_text(json.dumps({'seed': SEED, 'ct': ct, 'K': K,
                                        'engine': 'kissat',
                                        'L': L.tolist(),
                                        'B': B.tolist()}, indent=1))
            git_push(f"kissat: turn-square mate with ct={ct} >= {K}!")
        else:
            log("!!! decode/verify mismatch — encoding bug, investigate !!!")
    elif "s UNSATISFIABLE" in out:
        log(f"kissat: UNSATISFIABLE in {el:.0f}s")
        outp = REPO / f"mols10/results/ctthr_theorem_{TAG}.json"
        outp.write_text(json.dumps({'seed': SEED, 'threshold': K,
                                    'status': 'UNSAT', 'engine': 'kissat',
                                    'elapsed_s': el,
                                    'L': L.tolist()}, indent=1))
        git_push(f"kissat THEOREM: turn-square max-ct < {K}")
    else:
        log(f"kissat: no verdict in {el:.0f}s (timeout/killed)")

if __name__ == "__main__":
    main()
