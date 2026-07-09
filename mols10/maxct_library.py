#!/usr/bin/env python3
"""
Theorem factory: PROVE the max-ct value over all orthogonal mates for
every square in the one-swap near-turn library (160 squares, 2112
transversals each).

The exact max-ct CP-SAT model closes on a 2112-transversal square in
~23 minutes (OPTIMAL=3 for sq144, first proof). Each OPTIMAL < 10
verdict is a theorem "this square belongs to no 3-MOLS(10)"; collected
over the library it becomes a family theorem. Any incumbent >= 10 would
be a sensation handled by the SAT branch.

Checkpointed per square; progress pushed to git every PUSH_EVERY
verdicts (container-restart-proof).

Usage: python3 maxct_library.py <instance> <n_inst> [cap_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_INST = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 2400
N = 10
LOG = REPO / f"mols10/results/maxctlib_{INSTANCE}.log"
CKPT = REPO / f"mols10/results/maxctlib_ckpt_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
BASES = ["mols10/results/turnsq_ct6_squareA.json",
         "mols10/results/turnsq_best_ct_1.json"]
PUSH_EVERY = 4

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

def prove_maxct(L, cap_s):
    trans = enum_transversals(L)
    M = len(trans)
    TA8 = np.array(trans, dtype=np.int8)
    bad = [None] * M
    for lo in range(0, M, 256):
        hi = min(lo + 256, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi - lo):
            bad[lo + k] = np.nonzero(cnt[k] >= 2)[0]
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    y = [model.new_bool_var(f"y{j}") for j in range(M)]
    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
    for j in range(M):
        if len(bad[j]):
            model.add(sum(x[int(i)] for i in bad[j]) == 0).only_enforce_if(y[j])
    model.maximize(sum(y))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cap_s
    solver.parameters.num_workers = 2
    st = solver.solve(model)
    name = solver.status_name(st)
    obj = int(solver.objective_value) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1
    bound = int(solver.best_objective_bound)
    B = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) and obj >= N:
        sel = [i for i in range(M) if solver.value(x[i])]
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            for r in range(N):
                B[r, trans[ti][r]] = sym
    return name, obj, bound, M, B

def main():
    log("=" * 70)
    log(f"Max-ct THEOREM FACTORY — instance={INSTANCE}/{N_INST}, cap={CAP}s")
    lib = one_swap_library()
    log(f"library: {len(lib)} squares")

    ckpt = {}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: pass

    mine = [k for k in range(len(lib)) if k % N_INST == INSTANCE]
    pending = [k for k in mine if str(k) not in ckpt]
    log(f"stripe {len(mine)}, done {len(mine)-len(pending)}, pending {len(pending)}")

    n_since = 0
    for k in pending:
        t0 = time.time()
        name, obj, bound, M, B = prove_maxct(lib[k], CAP)
        el = time.time() - t0
        ckpt[str(k)] = {'status': name, 'maxct': obj, 'bound': bound,
                        'n_trans': M, 'elapsed': round(el)}
        CKPT.write_text(json.dumps(ckpt, indent=0))
        n_opt = sum(1 for v in ckpt.values() if v['status'] == 'OPTIMAL')
        log(f"sq{k}: {name} maxct={obj} bound={bound} in {el:.0f}s "
            f"[{len(ckpt)}/{len(mine)}, {n_opt} proved]")
        if B is not None:
            out = REPO / f"mols10/results/maxctlib_ct10_{k}.json"
            out.write_text(json.dumps({'sq': k, 'ct': obj,
                                       'L': lib[k].tolist(),
                                       'B': B.tolist()}, indent=1))
            git_push(f"SENSATION: one-swap sq{k} has mate with ct={obj} >= 10")
        n_since += 1
        if n_since >= PUSH_EVERY:
            git_push(f"maxct library i{INSTANCE}: {len(ckpt)}/{len(mine)} "
                     f"({n_opt} proved)")
            n_since = 0

    vals = {}
    for v in ckpt.values():
        if v['status'] == 'OPTIMAL':
            vals[v['maxct']] = vals.get(v['maxct'], 0) + 1
    log(f"STRIPE COMPLETE: proved-maxct distribution: {dict(sorted(vals.items()))}")
    git_push(f"maxct library i{INSTANCE} COMPLETE: {dict(sorted(vals.items()))}")

if __name__ == "__main__":
    main()
