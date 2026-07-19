#!/usr/bin/env python3
"""
Cube-and-conquer kissat attack on "exists mate with ct >= K" for the
turn square — RESTART-PROOF version for a hostile environment.

Cubes: fix the mate's (0,0)-covering transversal (608 choices). Each
cube adds one unit clause to the base CNF and solves in minutes; every
verdict is checkpointed to the repo and pushed periodically, so
container rebuilds lose at most the in-flight cube.

ALL cubes UNSAT  => THEOREM max-ct < K.
Any cube SAT     => decoded+verified mate with ct >= K; for K >= 10 the
                    commons are tested for exact-cover into a third
                    square (a success would BE a 3-MOLS(10)).

Usage: python3 ctthr_cubes.py <seed_json> <K> [per_cube_timeout_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

KISSAT = ("/tmp/claude-0/-home-user-autoresearch-glm/"
          "6c15e52c-04fd-5e17-80ad-627a8c92c1f0/scratchpad/"
          "Myrvold-MOLS/kissat/build/kissat")

SEED = sys.argv[1]
K = int(sys.argv[2])
CUBE_T = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
TAG = f"cubes_{Path(SEED).stem}_ge{K}"
LOG = REPO / f"mols10/results/ctthr_{TAG}.log"
CKPT = REPO / f"mols10/results/ctthr_{TAG}_ckpt.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
SCRATCH = Path("/tmp") / f"ctthr_cubes_ge{K}"
PUSH_EVERY = 8

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

def build_cnf(L, trans, K):
    M = len(trans)
    TA8 = np.array(trans, dtype=np.int8)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    X = lambda i: 1 + i
    Y = lambda j: 1 + M + j
    nv = [1 + 2 * M]
    def new_var():
        v = nv[0]; nv[0] += 1
        return v
    clauses = []
    for r in range(N):
        for c in range(N):
            lst = cell_to_t[r][c]
            clauses.append([X(i) for i in lst])
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
    for lo in range(0, M, 256):
        hi = min(lo + 256, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            j = lo + k
            for i in np.nonzero(cnt[k] >= 2)[0]:
                clauses.append([-Y(j), -X(int(i))])
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
    clauses.append([prev[K-1]])
    return clauses, nv[0] - 1, cell_to_t, X

def try_complete(L, B, trans, TA8):
    TA = TA8.astype(np.int64)
    BV = B[np.arange(N)[None, :], TA]
    S = np.sort(BV, axis=1)
    commons = [trans[i] for i in np.nonzero((np.diff(S, axis=1) != 0).all(axis=1))[0]]
    log(f"  commons: {len(commons)}; attempting exact cover into C...")
    c2t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(commons):
        for r in range(N):
            c2t[r][t[r]].append(i)
    m2 = cp_model.CpModel()
    w = [m2.new_bool_var(f"w{i}") for i in range(len(commons))]
    for r in range(N):
        for c in range(N):
            if c2t[r][c]:
                m2.add_exactly_one([w[i] for i in c2t[r][c]])
            else:
                log("  a cell is uncovered by commons — no C")
                return None
    s2 = cp_model.CpSolver()
    s2.parameters.max_time_in_seconds = 600
    st2 = s2.solve(m2)
    log(f"  commons cover: {s2.status_name(st2)}")
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        C = np.zeros((N, N), dtype=np.int8)
        for sym, i in enumerate([i for i in range(len(commons)) if s2.value(w[i])]):
            for r in range(N):
                C[r, commons[i][r]] = sym
        return C
    return None

def main():
    log("=" * 70)
    log(f"CUBES kissat ct >= {K} — {SEED}, per-cube cap={CUBE_T}s")
    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    trans = enum_transversals(L)
    M = len(trans)
    TA8 = np.array(trans, dtype=np.int8)
    clauses, nv, cell_to_t, X = build_cnf(L, trans, K)
    t00 = cell_to_t[0][0]
    log(f"{M} transversals; base CNF {nv} vars {len(clauses)} clauses; "
        f"{len(t00)} cubes")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    base = SCRATCH / "base.cnf"
    with open(base, "w") as f:
        f.write(f"p cnf {nv} {len(clauses) + 1}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
        f.write("CUBELINE\n")
    base_text = base.read_text()

    ckpt = {}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: pass

    n_since = 0
    for ci, t_fix in enumerate(t00):
        key = str(ci)
        if key in ckpt:
            continue
        cnf_i = SCRATCH / f"cube_{ci}.cnf"
        cnf_i.write_text(base_text.replace("CUBELINE", f"{X(t_fix)} 0"))
        t0 = time.time()
        proc = subprocess.run(
            [KISSAT, f"--time={CUBE_T}", f"--seed={int(time.time())%99991}",
             str(cnf_i)], capture_output=True, text=True)
        el = time.time() - t0
        out = proc.stdout
        cnf_i.unlink(missing_ok=True)
        if "s SATISFIABLE" in out:
            assign = set()
            for line in out.splitlines():
                if line.startswith("v "):
                    for tok in line[2:].split():
                        v = int(tok)
                        if v > 0: assign.add(v)
            sel = [i for i in range(M) if (1 + i) in assign]
            B = np.zeros((N, N), dtype=np.int8)
            for sym, ti in enumerate(sel):
                for r in range(N):
                    B[r, trans[ti][r]] = sym
            TA = TA8.astype(np.int64)
            BV = B[np.arange(N)[None, :], TA]
            S = np.sort(BV, axis=1)
            ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
            cl0 = count_clashes(L, B)
            log(f"cube {ci}: SAT in {el:.0f}s — decoded ct={ct} cl={cl0}")
            if cl0 == 0 and ct >= K:
                ckpt[key] = {'status': 'SAT', 'ct': ct, 'elapsed': round(el)}
                CKPT.write_text(json.dumps(ckpt, indent=0))
                outp = REPO / f"mols10/results/ctthr_sat_{TAG}.json"
                outp.write_text(json.dumps({'seed': SEED, 'ct': ct, 'K': K,
                                            'cube': ci, 'engine': 'kissat-cubes',
                                            'L': L.tolist(), 'B': B.tolist()},
                                           indent=1))
                git_push(f"kissat-cube: mate with ct={ct} >= {K} (cube {ci})")
                if K >= N:
                    C = try_complete(L, B, trans, TA8)
                    if C is not None:
                        cl12 = count_clashes(L, B); cl13 = count_clashes(L, C)
                        cl23 = count_clashes(B, C)
                        log(f"VERIFY TRIPLE: {cl12} {cl13} {cl23}")
                        if cl12 == cl13 == cl23 == 0:
                            log("*** 3-MOLS(10) FOUND ***")
                            (REPO / "mols10/results/MOLS10_FOUND.json").write_text(
                                json.dumps({'found': True, 'method': 'ctthr_cubes',
                                            'L1': L.tolist(), 'L2': B.tolist(),
                                            'L3': C.tolist()}, indent=2))
                            git_push("3-MOLS(10) FOUND via cube attack!")
                return
            else:
                log(f"  !!! decode mismatch (encoding bug?) — recording anyway")
                ckpt[key] = {'status': 'SAT_MISMATCH', 'ct': ct,
                             'elapsed': round(el)}
        elif "s UNSATISFIABLE" in out:
            ckpt[key] = {'status': 'UNSAT', 'elapsed': round(el)}
            n_unsat = sum(1 for v in ckpt.values() if v['status'] == 'UNSAT')
            log(f"cube {ci}: UNSAT in {el:.0f}s [{len(ckpt)}/{len(t00)}, "
                f"{n_unsat} UNSAT]")
        else:
            ckpt[key] = {'status': 'TIMEOUT', 'elapsed': round(el)}
            log(f"cube {ci}: TIMEOUT at {CUBE_T}s")
        CKPT.write_text(json.dumps(ckpt, indent=0))
        n_since += 1
        if n_since >= PUSH_EVERY:
            git_push(f"cubes ge{K}: {len(ckpt)}/{len(t00)}")
            n_since = 0

    n_unsat = sum(1 for v in ckpt.values() if v['status'] == 'UNSAT')
    n_to = sum(1 for v in ckpt.values() if v['status'] == 'TIMEOUT')
    log(f"ALL CUBES DONE: {n_unsat} UNSAT, {n_to} TIMEOUT of {len(t00)}")
    if n_unsat == len(t00):
        log(f"THEOREM: max-ct < {K} for this square (all cubes UNSAT)")
        outp = REPO / f"mols10/results/ctthr_theorem_{TAG}.json"
        outp.write_text(json.dumps({'seed': SEED, 'threshold': K,
                                    'status': 'ALL_CUBES_UNSAT',
                                    'L': L.tolist()}, indent=1))
    git_push(f"cubes ge{K} COMPLETE: {n_unsat} UNSAT, {n_to} TIMEOUT")

if __name__ == "__main__":
    main()
