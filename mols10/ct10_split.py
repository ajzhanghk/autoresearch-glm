#!/usr/bin/env python3
"""
Checkpointed split decision: does square L have a mate B with ct(L,B) >= 10?

The mate's decomposition contains exactly one transversal covering cell
(0,0) (i.e. t[0] == 0). Splitting on that choice gives ~550 independent
subproblems: subproblem k asserts x_{t0[k]} = 1 on top of the base model
(exact cover + y-indicators + sum y >= 10).

Each subproblem verdict (INFEASIBLE / SAT / UNKNOWN) is checkpointed to a
per-instance JSON and auto-committed to git every BATCH subproblems, so
container rebuilds lose at most one batch. ALL INFEASIBLE => theorem:
square is in no 3-MOLS(10). Any SAT => attempt triple completion.

Usage: python3 ct10_split.py <seed_json> <instance> <n_instances> [cap_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

SEED = sys.argv[1]
INSTANCE = int(sys.argv[2])
N_INST = int(sys.argv[3])
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 900
TAG = Path(SEED).stem
CKPT = REPO / f"mols10/results/ct10_split_{TAG}_{INSTANCE}.json"
LOG = REPO / f"mols10/results/ct10_split_{TAG}_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
BATCH = 10

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_commit_push(msg):
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
    log(f"ct>=10 SPLIT DECISION — {SEED}, instance {INSTANCE}/{N_INST}, cap={CAP}s")
    seed = json.loads((REPO / SEED).read_text())
    L = np.array(seed['L'], dtype=np.int8)
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

    t0_list = cell_to_t[0][0]          # transversals covering (0,0)
    log(f"{M} transversals; {len(t0_list)} subproblems total")

    # base model
    tb = time.time()
    base = cp_model.CpModel()
    x = [base.new_bool_var(f"x{i}") for i in range(M)]
    y = [base.new_bool_var(f"y{j}") for j in range(M)]
    for r in range(N):
        for c in range(N):
            base.add_exactly_one([x[i] for i in cell_to_t[r][c]])
    for j in range(M):
        if len(bad[j]):
            base.add(sum(x[int(i)] for i in bad[j]) == 0).only_enforce_if(y[j])
    base.add(sum(y) >= N)
    log(f"base model built in {time.time()-tb:.0f}s")

    ckpt = {}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: ckpt = {}

    mine = [k for k in range(len(t0_list)) if k % N_INST == INSTANCE]
    pending = [k for k in mine if str(k) not in ckpt]
    log(f"my stripe: {len(mine)}, pending: {len(pending)}, "
        f"done: {len(mine)-len(pending)}")

    since_commit = 0
    for k in pending:
        ti = t0_list[k]
        sub = cp_model.CpModel()
        sub.proto.copy_from(base.proto)
        # re-reference variable by index in the copied proto
        xk = sub.get_bool_var_from_proto_index(ti)
        sub.add(xk == 1)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = CAP
        solver.parameters.num_workers = 2
        t1 = time.time()
        st = solver.solve(sub)
        el = time.time() - t1
        name = solver.status_name(st)
        ckpt[str(k)] = {'status': name, 'elapsed': round(el, 1)}
        CKPT.write_text(json.dumps(ckpt, indent=0))
        n_inf = sum(1 for v in ckpt.values() if v['status'] == 'INFEASIBLE')
        n_unk = sum(1 for v in ckpt.values() if v['status'] == 'UNKNOWN')
        log(f"sub {k} (t={ti}): {name} in {el:.0f}s "
            f"[{len(ckpt)}/{len(mine)} done: {n_inf} INF, {n_unk} UNK]")

        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sel = [i for i in range(M) if solver.value(x[i])]
            B = np.zeros((N, N), dtype=np.int8)
            for sym, t_i in enumerate(sel):
                for r in range(N):
                    B[r, trans[t_i][r]] = sym
            BV = B[np.arange(N)[None, :], TA8.astype(np.int64)]
            S = np.sort(BV, axis=1)
            ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
            log(f"*** SAT: mate with ct={ct} (cl={count_clashes(L,B)}) ***")
            out = REPO / f"mols10/results/ct10_sat_{TAG}_{k}.json"
            out.write_text(json.dumps({'seed': SEED, 'sub': k, 'ct': ct,
                                       'L': L.tolist(), 'B': B.tolist()}, indent=1))
            git_commit_push(f"ct>=10 SAT for {TAG} sub {k} (ct={ct})!")
            # completion attempt left to the driver session

        since_commit += 1
        if since_commit >= BATCH:
            git_commit_push(f"ct10 split {TAG} i{INSTANCE}: "
                            f"{len(ckpt)}/{len(mine)} ({n_inf} INF, {n_unk} UNK)")
            since_commit = 0

    n_inf = sum(1 for v in ckpt.values() if v['status'] == 'INFEASIBLE')
    n_unk = sum(1 for v in ckpt.values() if v['status'] == 'UNKNOWN')
    log(f"STRIPE COMPLETE: {len(ckpt)} subproblems, {n_inf} INFEASIBLE, {n_unk} UNKNOWN")
    git_commit_push(f"ct10 split {TAG} i{INSTANCE} COMPLETE: "
                    f"{n_inf} INF, {n_unk} UNK of {len(ckpt)}")

if __name__ == "__main__":
    main()
