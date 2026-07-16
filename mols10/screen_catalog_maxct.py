#!/usr/bin/env python3
"""
Theorem-level max-ct screening of the entire pair catalog.

For every DISTINCT square appearing in promising_pairs.json (L1 or L2),
prove max ct over all its orthogonal mates with the exact CP-SAT model
(~100-200s per ~870-transversal square). Any square with proven
max-ct < 10 can never belong to a 3-MOLS(10) — with any partner.
A square with incumbent >= 10 would be the big fish's fin.

Checkpointed; pushes every PUSH_EVERY verdicts.

Usage: python3 screen_catalog_maxct.py [cap_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
N = 10
LOG = REPO / "mols10/results/screen_maxct.log"
CKPT = REPO / "mols10/results/screen_maxct_ckpt.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
PUSH_EVERY = 10

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

def prove_maxct(L, cap_s):
    trans = enum_transversals(L)
    M = len(trans)
    if M < N:
        return 'NO_DECOMP', 0, 0, M
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
    feasible = True
    for r in range(N):
        for c in range(N):
            if cell_to_t[r][c]:
                model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            else:
                feasible = False
    if not feasible:
        return 'NO_DECOMP', 0, 0, M
    for j in range(M):
        if len(bad[j]):
            model.add(sum(x[int(i)] for i in bad[j]) == 0).only_enforce_if(y[j])
    model.maximize(sum(y))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cap_s
    solver.parameters.num_workers = 4
    st = solver.solve(model)
    name = solver.status_name(st)
    if st == cp_model.INFEASIBLE:
        # square has transversals but no decomposition -> no mates at all
        return 'NO_MATE', 0, 0, M
    obj = int(solver.objective_value) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1
    bound = int(solver.best_objective_bound)
    return name, obj, bound, M

def main():
    log("=" * 70)
    log(f"Catalog-wide max-ct screening — cap={CAP}s")
    pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
    squares = {}
    for p in pairs:
        for key in ('L1', 'L2'):
            L = p.get(key)
            if L is None: continue
            fp = tuple(tuple(row) for row in L)
            if fp not in squares:
                squares[fp] = f"{p.get('pair_id','?')}_{key}"
    sq_list = list(squares.items())
    log(f"{len(pairs)} pairs -> {len(sq_list)} distinct squares")

    ckpt = {}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: pass

    n_since = 0
    for idx, (fp, name0) in enumerate(sq_list):
        key = str(idx)
        if key in ckpt: continue
        L = np.array(fp, dtype=np.int8)
        t0 = time.time()
        name, obj, bound, M = prove_maxct(L, CAP)
        el = time.time() - t0
        ckpt[key] = {'label': name0, 'status': name, 'maxct': obj,
                     'bound': bound, 'n_trans': M, 'elapsed': round(el)}
        CKPT.write_text(json.dumps(ckpt, indent=0))
        log(f"[{idx+1}/{len(sq_list)}] {name0}: {name} maxct={obj} "
            f"bound={bound} trans={M} in {el:.0f}s")
        if obj >= N:
            log(f"*** BIG FISH FIN: {name0} has a mate with ct={obj} >= 10 ***")
            git_push(f"SENSATION: catalog square {name0} max-ct={obj} >= 10")
        n_since += 1
        if n_since >= PUSH_EVERY:
            from collections import Counter
            dist = Counter(v['maxct'] for v in ckpt.values()
                           if v['status'] == 'OPTIMAL')
            git_push(f"maxct screening: {len(ckpt)}/{len(sq_list)} "
                     f"dist={dict(sorted(dist.items()))}")
            n_since = 0

    from collections import Counter
    dist = Counter(v['maxct'] for v in ckpt.values() if v['status'] == 'OPTIMAL')
    log(f"SCREENING COMPLETE: proven max-ct distribution: {dict(sorted(dist.items()))}")
    git_push(f"maxct screening COMPLETE: {dict(sorted(dist.items()))}")

if __name__ == "__main__":
    main()
