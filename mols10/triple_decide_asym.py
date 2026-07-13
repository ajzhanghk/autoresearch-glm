#!/usr/bin/env python3
"""
THE SHARP FISH HUNT: for each TRIVIAL-autotopism (MMM-eligible) order-10
square L, decide EXACTLY whether L belongs to a 3-MOLS(10), via the
two-orthogonal-decompositions model (Stage 8 reformulation):

  L is in a 3-MOLS(10)  <=>  L admits two transversal decompositions
  X = {T_1..T_10}, Z = {S_1..S_10} that are mutually orthogonal
  (|T_i cap S_j| = 1 for all i,j).

Model (booleans x_i, z_j over L's transversal pool P):
  - X and Z each exact-cover the 100 cells (10 disjoint transversals),
  - x_i = 1  =>  sum_{j in bad(i)} z_j = 0, where bad(i) = { j : |t_i cap t_j| >= 2 }.
    (Partition argument: the 10 members of X partition the cells, so every
     transversal meets them with total intersection 10; forbidding any
     >=2 overlap forces exactly-1 overlaps, i.e. mutual orthogonality.)
  - symmetry break: X's (0,0)-transversal index < Z's.

SAT  => decode (L, B, C), verify cl=0 all around => 3-MOLS(10) FOUND.
UNSAT => L is in no triple (theorem).

Because trivial-autotopism squares are transversal-poor (~800-1100),
each model is small (~2k vars) and decides in seconds-to-minutes -- unlike
the 5504-transversal turn class. Squares are supplied by a companion
generator (asym_pool.py) or the on-the-fly randomizer here.

Usage: python3 triple_decide_asym.py <instance> [min_trans] [cap_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from autotopism import has_nontrivial_autotopism
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MIN_TRANS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 300
LOG = REPO / f"mols10/results/tripledec_{INSTANCE}.log"
CKPT = REPO / f"mols10/results/tripledec_ckpt_{INSTANCE}.json"
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
                    p.append(col); bt(row+1, cm|(1<<col), vm|(1<<v), p); p.pop()
    bt(0, 0, 0, [])
    return out

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
                    if bt(row+1, cm|(1<<col), vm|(1<<v)):
                        return True
        return False
    bt(0, 0, 0)
    return cnt[0]

def row_cycles(L, r1, r2):
    pos2 = np.zeros(N, dtype=int)
    for c in range(N):
        pos2[L[r2, c]] = c
    p = [int(pos2[L[r1, c]]) for c in range(N)]
    seen = [False]*N; out = []
    for c0 in range(N):
        if not seen[c0]:
            cyc = []; c = c0
            while not seen[c]:
                seen[c] = True; cyc.append(c); c = p[c]
            if 2 <= len(cyc) <= N:
                out.append(cyc)
    return out

def random_move(L, rng):
    for _ in range(40):
        tp = rng.random() < 0.5
        M0 = np.ascontiguousarray(L.T) if tp else L
        r1, r2 = rng.sample(range(N), 2)
        cyc = rng.choice(row_cycles(M0, r1, r2))
        if len(cyc) < 2:
            continue
        M1 = M0.copy()
        for c in cyc:
            M1[r1, c], M1[r2, c] = M0[r2, c], M0[r1, c]
        return np.ascontiguousarray(M1.T) if tp else M1
    return None

def gen_trivial_square(rng, min_trans):
    """Random trivial-autotopism square, climbed toward min_trans transversals."""
    L = np.array([[(i+j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    for _ in range(300):
        m = random_move(L, rng)
        if m is not None: L = m
    cur = count_trans(L)
    tries = 0
    while (cur < min_trans or has_nontrivial_autotopism(L)) and tries < 4000:
        tries += 1
        m = random_move(L, rng)
        if m is None: continue
        n2 = count_trans(m)
        if n2 >= cur - 40 or rng.random() < 0.1:
            L, cur = m, n2
    return L if not has_nontrivial_autotopism(L) else None

def decide_triple(L, cap_s):
    trans = enum_trans(L)
    M = len(trans)
    TA8 = np.array(trans, dtype=np.int8)
    bad = [None]*M
    for lo in range(0, M, 256):
        hi = min(lo+256, M)
        eq = (TA8[lo:hi, None, :] == TA8[None, :, :])
        cnt = eq.sum(axis=2, dtype=np.int8)
        for k in range(hi-lo):
            bad[lo+k] = np.nonzero(cnt[k] >= 2)[0]
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    z = [model.new_bool_var(f"z{i}") for i in range(M)]
    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            model.add_exactly_one([z[i] for i in cell_to_t[r][c]])
    for i in range(M):
        bi = bad[i]
        if len(bi):
            model.add(sum(z[int(j)] for j in bi) == 0).only_enforce_if(x[i])
    t00 = cell_to_t[0][0]
    ix = model.new_int_var(0, M, "ix"); iz = model.new_int_var(0, M, "iz")
    model.add(ix == sum(i*x[i] for i in t00))
    model.add(iz == sum(i*z[i] for i in t00))
    model.add(ix < iz)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cap_s
    solver.parameters.num_workers = 2
    st = solver.solve(model)
    name = solver.status_name(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selX = [i for i in range(M) if solver.value(x[i])]
        selZ = [i for i in range(M) if solver.value(z[i])]
        B = np.zeros((N, N), dtype=np.int8); C = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(selX):
            for r in range(N): B[r, trans[ti][r]] = sym
        for sym, ti in enumerate(selZ):
            for r in range(N): C[r, trans[ti][r]] = sym
        return name, M, B, C
    return name, M, None, None

def main():
    rng = random.Random(555 + INSTANCE*3313 + int(time.time()) % 90001)
    log("="*70)
    log(f"TRIPLE DECISION on trivial-autotopism squares — inst={INSTANCE}, "
        f"min_trans={MIN_TRANS}, cap={CAP}s")
    ckpt = {}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: ckpt = {}
    n_done = len(ckpt)
    n_unsat = sum(1 for v in ckpt.values() if v['status'] == 'INFEASIBLE')
    log(f"resuming: {n_done} decided, {n_unsat} UNSAT so far")

    while True:
        L = gen_trivial_square(rng, MIN_TRANS)
        if L is None:
            continue
        assert not has_nontrivial_autotopism(L)
        t0 = time.time()
        name, M, B, C = decide_triple(L, CAP)
        el = time.time() - t0
        key = str(len(ckpt))
        ckpt[key] = {'status': name, 'n_trans': M, 'elapsed': round(el)}
        CKPT.write_text(json.dumps(ckpt, indent=0))
        n_done = len(ckpt)
        n_unsat = sum(1 for v in ckpt.values() if v['status'] == 'INFEASIBLE')
        log(f"square {key}: {M} trans -> {name} in {el:.0f}s "
            f"[{n_done} decided, {n_unsat} UNSAT]")
        if name in ('OPTIMAL', 'FEASIBLE') and B is not None:
            cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
            log(f"*** SAT! cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23} ***")
            if cl12 == 0 and cl13 == 0 and cl23 == 0:
                log("*** *** 3-MOLS(10) FOUND — THE BIG FISH *** ***")
                FOUND.write_text(json.dumps({
                    'found': True, 'method': 'triple_decide_asym',
                    'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                    'trivial_autotopism_L1': True,
                    'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23)}, indent=1))
                git_push("*** 3-MOLS(10) FOUND via triple_decide_asym! ***")
                return
        if n_done % 5 == 0:
            git_push(f"triple_decide_asym i{INSTANCE}: {n_done} decided, {n_unsat} UNSAT")

if __name__ == "__main__":
    main()
