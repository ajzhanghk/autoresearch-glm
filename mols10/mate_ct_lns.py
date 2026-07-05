#!/usr/bin/env python3
"""
LNS/SA in MATE SPACE, maximizing ct(L,B).

State: a transversal decomposition (orthogonal mate B) of the fixed
5504-transversal turn-square L, as a set of 10 transversal indices.

Move: remove k (2..4) transversals from the decomposition; the freed cells
(k per row/col) are re-covered by an alternative k-set of disjoint
transversals found by exact-cover backtracking over the transversal pool.
Accept by simulated annealing on ct(L,B).

ct evaluation is vectorized with numpy over all 5504 transversals.

If ct >= 10: enumerate common transversals of (L,B), exact-cover into C
=> (L,B,C) = 3-MOLS(10). Verify with count_clashes, save, push, exit.

Usage: python3 mate_ct_lns.py <instance> [seed_file]
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEED_FILE = sys.argv[2] if len(sys.argv) > 2 else "mols10/results/turnsq_best_ct_0.json"
LOG = REPO / f"mols10/results/matelns_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE = REPO / f"mols10/results/matelns_best_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add","-A"], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

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

class CtEvaluator:
    """Vectorized ct(L,B) over the full transversal pool of L."""
    def __init__(self, trans):
        self.M = len(trans)
        self.TA = np.array(trans, dtype=np.int64)         # M x N (col per row)
        self.rows_idx = np.arange(N, dtype=np.int64)[None, :]  # 1 x N

    def ct(self, B):
        BV = B[self.rows_idx, self.TA]                    # M x N symbol matrix
        S = np.sort(BV, axis=1)
        distinct = (np.diff(S, axis=1) != 0).all(axis=1)
        return int(distinct.sum())

    def common_mask(self, B):
        BV = B[self.rows_idx, self.TA]
        S = np.sort(BV, axis=1)
        return (np.diff(S, axis=1) != 0).all(axis=1)

def build_B(sel, trans):
    B = np.zeros((N, N), dtype=np.int8)
    for sym, ti in enumerate(sel):
        t = trans[ti]
        for r in range(N):
            B[r, t[r]] = sym
    return B

def initial_decomposition(trans, rng, timeout_s=60):
    """Random-ish exact cover via CP-SAT with shuffled variable order."""
    M = len(trans)
    order = list(range(M)); rng.shuffle(order)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i in order:
        t = trans[i]
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = {i: model.new_bool_var(f"x{i}") for i in order}
    for r in range(N):
        for c in range(N):
            model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = rng.randint(1, 10**6)
    status = solver.solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [i for i in order if solver.value(x[i])]
    return None

def find_recoveries(freed_rows_cols, candidates, trans, limit=200):
    """Enumerate exact covers of the freed cells by candidate transversals.
    freed_rows_cols: per-row set of freed cols. Returns list of tuples of indices."""
    # Group candidate transversal indices by their col in row 0's freed set? General BT:
    freed_cells = [(r, c) for r in range(N) for c in freed_rows_cols[r]]
    cellset = set(freed_cells)
    covers = []
    cand = list(candidates)
    # index candidates by (row0_col within freed)
    def bt(remaining, chosen):
        if len(covers) >= limit:
            return
        if not remaining:
            covers.append(tuple(chosen)); return
        # pick lexicographically first remaining cell
        r0, c0 = min(remaining)
        for ti in cand:
            if ti in chosen:
                continue
            t = trans[ti]
            if t[r0] != c0:
                continue
            cells = [(r, t[r]) for r in range(N)]
            if any(cell not in remaining for cell in cells):
                continue
            chosen.append(ti)
            bt(remaining - set(cells), chosen)
            chosen.pop()
    bt(frozenset(freed_cells), [])
    return covers

def try_complete_triple(L, B, ev, trans):
    """ct >= N: exact-cover the common transversals into C."""
    mask = ev.common_mask(B)
    commons = [trans[i] for i in np.nonzero(mask)[0]]
    log(f"    common transversals: {len(commons)}")
    if len(commons) < N:
        return None
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(commons):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"c{i}") for i in range(len(commons))]
    for r in range(N):
        for c in range(N):
            model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600
    solver.parameters.num_workers = 4
    status = solver.solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        C = np.zeros((N, N), dtype=np.int8)
        for sym, i in enumerate([i for i in range(len(commons)) if solver.value(x[i])]):
            for r in range(N):
                C[r, commons[i][r]] = sym
        return C
    log(f"    exact cover of commons: {solver.status_name(status)}")
    return None

def main():
    log("=" * 70)
    log(f"Mate-CT LNS — instance={INSTANCE}, seed={SEED_FILE}")
    log("=" * 70)

    rng = random.Random(1618033988 + INSTANCE * 104729)

    seed = json.loads((REPO / SEED_FILE).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    log(f"Loaded L; enumerating transversals...")
    trans = enum_transversals(L)
    M = len(trans)
    log(f"{M} transversals")
    ev = CtEvaluator(trans)

    # map from transversal tuple -> index (to seed from saved B)
    tidx = {t: i for i, t in enumerate(trans)}

    # Seed decomposition from saved B if present, else fresh
    sel = None
    if 'B' in seed:
        B0 = np.array(seed['B'], dtype=np.int8)
        cols_by_sym = {}
        ok = True
        for sym in range(N):
            cols = tuple(int(np.nonzero(B0[r] == sym)[0][0]) for r in range(N))
            if cols in tidx:
                cols_by_sym[sym] = tidx[cols]
            else:
                ok = False; break
        if ok:
            sel = [cols_by_sym[s] for s in range(N)]
            log(f"Seeded from saved B, ct={ev.ct(B0)}")
    if sel is None:
        sel = initial_decomposition(trans, rng)
        if sel is None:
            log("No initial decomposition found!"); return
        log("Fresh initial decomposition")

    # Precompute per-row col->candidate lists for speed
    by_row_col = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            by_row_col[r][t[r]].append(i)

    B = build_B(sel, trans)
    cur_ct = ev.ct(B)
    best_ct = cur_ct
    best_sel = list(sel)
    log(f"Start ct={cur_ct}")

    T0, T_END = 1.2, 0.05
    STEPS_PER_CYCLE = 4000
    step = 0
    t_start = time.time()
    n_moves = n_accept = 0

    while True:
        step += 1
        frac = (step % STEPS_PER_CYCLE) / STEPS_PER_CYCLE
        T = T0 * (T_END / T0) ** frac  # sawtooth reheating schedule

        k = rng.choice([2, 2, 3, 3, 4])
        out_idx = rng.sample(range(N), k)
        removed = [sel[i] for i in out_idx]
        kept = [sel[i] for i in range(N) if i not in out_idx]

        # freed cells per row
        freed = [set() for _ in range(N)]
        for ti in removed:
            t = trans[ti]
            for r in range(N):
                freed[r].add(t[r])

        # candidates: transversals fully inside freed cells
        cand = set()
        r0 = 0
        for c in freed[0]:
            for ti in by_row_col[0][c]:
                t = trans[ti]
                if all(t[r] in freed[r] for r in range(N)):
                    cand.add(ti)
        covers = find_recoveries(freed, cand, trans, limit=100)
        if len(covers) <= 1:
            continue  # only the original cover exists
        n_moves += 1

        # Evaluate a random alternative cover (not the original)
        orig = tuple(sorted(removed))
        alts = [cv for cv in covers if tuple(sorted(cv)) != orig]
        if not alts:
            continue
        cv = rng.choice(alts)
        new_sel = kept + list(cv)
        newB = build_B(new_sel, trans)
        new_ct = ev.ct(newB)

        d = new_ct - cur_ct
        if d >= 0 or rng.random() < np.exp(d / max(T, 1e-9)):
            sel = new_sel
            cur_ct = new_ct
            n_accept += 1

            if cur_ct > best_ct:
                best_ct = cur_ct
                best_sel = list(sel)
                elapsed = time.time() - t_start
                log(f"step {step}: NEW BEST ct={best_ct} t={elapsed:.0f}s (moves={n_moves}, acc={n_accept})")
                Bb = build_B(best_sel, trans)
                BEST_FILE.write_text(json.dumps({
                    'best_ct': int(best_ct), 'L': L.tolist(), 'B': Bb.tolist(),
                    'step': step,
                }, indent=2))

                if best_ct >= N:
                    log(f"*** ct >= {N}! Attempting triple completion ***")
                    C = try_complete_triple(L, Bb, ev, trans)
                    if C is not None:
                        cl12 = count_clashes(L, Bb); cl13 = count_clashes(L, C); cl23 = count_clashes(Bb, C)
                        log(f"VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
                        if cl12 == 0 and cl13 == 0 and cl23 == 0:
                            log("*** 3-MOLS(10) FOUND via mate-CT LNS! ***")
                            FOUND_FILE.write_text(json.dumps({
                                'found': True, 'method': 'mate_ct_lns',
                                'L1': L.tolist(), 'L2': Bb.tolist(), 'L3': C.tolist(),
                                'cl12': 0, 'cl13': 0, 'cl23': 0,
                            }, indent=2))
                            git_push("3-MOLS(10) FOUND via mate-CT LNS!")
                            sys.exit(0)

        if step % 2000 == 0:
            elapsed = time.time() - t_start
            rate = step / max(elapsed, 1) * 3600
            log(f"step {step}: cur_ct={cur_ct} best_ct={best_ct} T={T:.3f} ({rate:.0f} steps/hr, moves={n_moves}, acc={n_accept})")

        # Occasionally restart from best
        if step % 20000 == 0:
            sel = list(best_sel)
            B = build_B(sel, trans)
            cur_ct = ev.ct(B)

if __name__ == "__main__":
    main()
