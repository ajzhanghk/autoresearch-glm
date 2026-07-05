#!/usr/bin/env python3
"""
Turn-square pipeline for 3-MOLS(10).

Turn-squares: start from the Z10 Cayley table B(i,j)=i+j mod 10. For each
(a,b) in a 5x5 pattern, the cells {a,a+5}x{b,b+5} form an intercalate
(values {a+b, a+b+5}); "turning" it swaps the two values. 2^25 turn-squares.
The maximum transversal count over ALL order-10 Latin squares (5504) is
attained by turn-squares. Our tv-family squares only have ~872.

Pipeline:
 1. Hill-climb/SA over 5x5 patterns to maximize transversal count.
 2. For top squares: enumerate all transversals, then stream orthogonal
    mates via CP-SAT solution enumeration (each mate = exact cover of
    cells by 10 transversals).
 3. For each mate B: ct(L,B) = # transversals of L that are also
    transversals of B. If ct >= 10: enumerate common transversals,
    exact-cover them into C => (L,B,C) is 3-MOLS(10). Verify, save, push.

Usage: python3 turn_square_search.py <instance>
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
LOG = REPO / f"mols10/results/turnsq_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_CT_FILE = REPO / f"mols10/results/turnsq_best_ct_{INSTANCE}.json"
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

def build_turn_square(pattern):
    """pattern: 5x5 0/1 array. Returns the turn-square as int8 array."""
    L = np.array([[(i + j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    for a in range(5):
        for b in range(5):
            if pattern[a][b]:
                v = (a + b) % N
                w = (a + b + 5) % N
                L[a, b] = w;         L[a, (b+5)] = v
                L[(a+5), b] = v;     L[(a+5), (b+5)] = w
    return L

def count_transversals(L, cap=100000):
    """Bitmask backtracking count of transversals."""
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

def pattern_hillclimb(rng, n_restarts=40, n_steps=400):
    """SA/hill-climb over 5x5 turn patterns maximizing transversal count.
    Returns list of (count, pattern) best-first, deduped by count."""
    results = {}
    for restart in range(n_restarts):
        pat = [[rng.randint(0, 1) for _ in range(5)] for _ in range(5)]
        cur = count_transversals(build_turn_square(pat))
        for step in range(n_steps):
            a = rng.randrange(5); b = rng.randrange(5)
            pat[a][b] ^= 1
            new = count_transversals(build_turn_square(pat))
            if new >= cur:
                cur = new
            else:
                pat[a][b] ^= 1  # revert
        key = tuple(tuple(row) for row in pat)
        results[key] = cur
    ranked = sorted(((c, k) for k, c in results.items()), reverse=True)
    return ranked

class MateCollector(cp_model.CpSolverSolutionCallback):
    """Streams exact-cover solutions (orthogonal mates); computes ct per mate."""
    def __init__(self, x, trans, L, deadline, max_mates):
        super().__init__()
        self.x = x
        self.trans = trans
        self.L = L
        self.deadline = deadline
        self.max_mates = max_mates
        self.n_mates = 0
        self.best_ct = -1
        self.best_mate_decomp = None
        self.ct_hist = {}
        # Precompute per-transversal cell list
        self.tcells = [[(r, t[r]) for r in range(N)] for t in trans]

    def on_solution_callback(self):
        self.n_mates += 1
        sel = [i for i in range(len(self.x)) if self.value(self.x[i])]
        # Build mate B
        B = np.zeros((N, N), dtype=np.int8)
        for sym, ti in enumerate(sel):
            for (r, c) in self.tcells[ti]:
                B[r, c] = sym
        # ct(L,B): transversals of L whose B-values are all distinct
        ct = 0
        for t in self.trans:
            mask = 0
            ok = True
            for r in range(N):
                v = int(B[r, t[r]])
                if mask >> v & 1:
                    ok = False; break
                mask |= 1 << v
            if ok:
                ct += 1
        self.ct_hist[ct] = self.ct_hist.get(ct, 0) + 1
        if ct > self.best_ct:
            self.best_ct = ct
            self.best_mate_decomp = (sel, B.copy())
        if ct >= N:
            # Potential 3-MOLS! Stop enumeration; handled by caller.
            self.stop_search()
        if self.n_mates >= self.max_mates or time.time() > self.deadline:
            self.stop_search()

def stream_mates_and_ct(L, trans, time_budget_s=600, max_mates=200000):
    """Enumerate mates of L; return (n_mates, best_ct, best (sel,B), ct histogram)."""
    M = len(trans)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    for r in range(N):
        for c in range(N):
            model.add(sum(x[i] for i in cell_to_t[r][c]) == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_s
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1  # required for enumeration
    cb = MateCollector(x, trans, L, time.time() + time_budget_s, max_mates)
    solver.solve(model, cb)
    return cb

def try_complete_triple(L, B):
    """Given ct(L,B) >= 10 candidate: enumerate common transversals of (L,B),
    exact-cover into C. Returns C or None."""
    commons = []
    def bt(row, cm, lm, bm, path):
        if row == N:
            commons.append(tuple(path)); return
        for col in range(N):
            if not (cm >> col & 1):
                lv = int(L[row, col]); bv = int(B[row, col])
                if not (lm >> lv & 1) and not (bm >> bv & 1):
                    path.append(col)
                    bt(row+1, cm | (1<<col), lm | (1<<lv), bm | (1<<bv), path)
                    path.pop()
    bt(0, 0, 0, 0, [])
    log(f"    common transversals of (L,B): {len(commons)}")
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
    solver.parameters.max_time_in_seconds = 300
    solver.parameters.num_workers = 4
    status = solver.solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        C = np.zeros((N, N), dtype=np.int8)
        for sym, i in enumerate([i for i in range(len(commons)) if solver.value(x[i])]):
            for r in range(N):
                C[r, commons[i][r]] = sym
        return C
    return None

def process_square(L, pat, cnt, label, mate_budget_s=600):
    """Phase 2+3 for one turn-square: stream mates, track ct, complete triple if ct>=N."""
    trans = enum_transversals(L)
    log(f"  [{label}] {len(trans)} transversals enumerated. Streaming mates ({mate_budget_s}s budget)...")
    t0 = time.time()
    cb = stream_mates_and_ct(L, trans, time_budget_s=mate_budget_s, max_mates=200000)
    hist_top = sorted(cb.ct_hist.items(), reverse=True)[:8]
    log(f"  [{label}] {cb.n_mates} mates in {time.time()-t0:.0f}s; best_ct={cb.best_ct}; ct hist(top)={hist_top}")

    if cb.best_ct >= 0 and cb.best_mate_decomp is not None:
        # Never clobber a better result from an earlier rank/run
        prev_best = -1
        if BEST_CT_FILE.exists():
            try:
                prev_best = json.loads(BEST_CT_FILE.read_text()).get('best_ct', -1)
            except Exception:
                pass
        if cb.best_ct <= prev_best:
            return cb
        BEST_CT_FILE.write_text(json.dumps({
            'pattern': pat, 'n_trans': len(trans), 'n_mates_seen': cb.n_mates,
            'best_ct': int(cb.best_ct),
            'L': L.tolist(), 'B': cb.best_mate_decomp[1].tolist(),
            'ct_hist': {str(k): v for k, v in sorted(cb.ct_hist.items(), reverse=True)},
        }, indent=2))

    if cb.best_ct >= N:
        sel, B = cb.best_mate_decomp
        log(f"  [{label}] *** ct >= {N}! Attempting triple completion... ***")
        C = try_complete_triple(L, B)
        if C is not None:
            cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
            log(f"  [{label}] VERIFY: cl(L,B)={cl12} cl(L,C)={cl13} cl(B,C)={cl23}")
            if cl12 == 0 and cl13 == 0 and cl23 == 0:
                log("*** 3-MOLS(10) FOUND via turn-square! ***")
                FOUND_FILE.write_text(json.dumps({
                    'found': True, 'method': 'turn_square',
                    'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                    'cl12': 0, 'cl13': 0, 'cl23': 0, 'pattern': pat,
                }, indent=2))
                git_push("3-MOLS(10) FOUND via turn-square search!")
                sys.exit(0)
    return cb


def main():
    log("=" * 70)
    log(f"Turn-Square Search — instance={INSTANCE}")
    log("Z10 turn-squares: max transversals known for order 10 (5504)")
    log("=" * 70)

    rng = random.Random(2718281828 + INSTANCE * 104729)

    log("Phase 1: pattern hill-climb for high-transversal turn-squares...")
    t0 = time.time()
    ranked = pattern_hillclimb(rng, n_restarts=30, n_steps=300)
    log(f"Hill-climb done in {time.time()-t0:.0f}s. Top counts: {[c for c, _ in ranked[:8]]}")

    TOP_K = 5
    for rank, (cnt, key) in enumerate(ranked[:TOP_K]):
        pat = [list(row) for row in key]
        L = build_turn_square(pat)
        log(f"Phase 2 [{rank+1}/{TOP_K}]: turn-square with {cnt} transversals, pattern={key}")
        process_square(L, pat, cnt, f"rank{rank+1}")

    log("Turn-square sweep of top patterns complete.")


if __name__ == "__main__":
    main()
