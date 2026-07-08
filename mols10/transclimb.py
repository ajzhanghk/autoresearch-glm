#!/usr/bin/env python3
"""
Transversal-count climbing over general order-10 Latin squares, with
inline mate-streaming ct scoring of every new record square.

Motivation: ct (common transversals with a mate) correlates strongly
with transversal richness (872-pool -> ct 2; 2112 -> ct 3; 5504 -> ct 6,
but the 5504 turn class is excluded from triples by MMM 2007). The open
frontier is squares with trivial symmetry and MANY transversals; no
public census exists beyond the (symmetric) 5504 maximum. This script
mines that frontier directly.

Moves: random partial row/column cycle swaps (length 3..9, Latin-
preserving, symmetry-breaking). Accept if transversal count does not
decrease (occasional SA uphill). The turn class is avoided by rejecting
sigma-translation-invariant states.

Every new per-instance record with count >= SCORE_THRESH gets a
STREAM_S-second CP-SAT mate stream with vectorized ct scoring; records
are checkpointed to git.

Usage: python3 transclimb.py <instance> [score_thresh] [stream_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SCORE_THRESH = int(sys.argv[2]) if len(sys.argv) > 2 else 2600
STREAM_S = int(sys.argv[3]) if len(sys.argv) > 3 else 300
N = 10
LOG = REPO / f"mols10/results/transclimb_{INSTANCE}.log"
BEST = REPO / f"mols10/results/transclimb_best_{INSTANCE}.json"
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

def count_transversals(L, cap=6000):
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

def turnlike(L):
    """Reject states invariant under (r,c)->(r+d,c+d) symbol-translations
    (the turn-class symmetry family), for any d in 1..9 combined with
    symbol shift e in 0..9: L[r+d,c+d] == L[r,c]+e."""
    for d in range(1, N):
        for e in range(N):
            ok = True
            for r in range(N):
                for c in range(N):
                    if L[(r+d) % N, (c+d) % N] != (L[r, c] + e) % N:
                        ok = False; break
                if not ok: break
            if ok:
                return True
    return False

def row_cycles(L, r1, r2, rng):
    pos2 = np.zeros(N, dtype=int)
    for c in range(N):
        pos2[L[r2, c]] = c
    p = [int(pos2[L[r1, c]]) for c in range(N)]
    seen = [False] * N
    cycles = []
    for c0 in range(N):
        if not seen[c0]:
            cyc = []
            c = c0
            while not seen[c]:
                seen[c] = True
                cyc.append(c)
                c = p[c]
            if 3 <= len(cyc) <= N - 1:
                cycles.append(cyc)
    return cycles

def random_move(L, rng):
    for _ in range(50):
        transpose = rng.random() < 0.5
        M0 = np.ascontiguousarray(L.T) if transpose else L
        r1, r2 = rng.sample(range(N), 2)
        cycles = row_cycles(M0, r1, r2, rng)
        if not cycles:
            continue
        cyc = rng.choice(cycles)
        M1 = M0.copy()
        for c in cyc:
            M1[r1, c], M1[r2, c] = M0[r2, c], M0[r1, c]
        return np.ascontiguousarray(M1.T) if transpose else M1
    return None

def stream_ct(L, budget_s):
    trans = enum_transversals(L)
    M = len(trans)
    TA = np.array(trans, dtype=np.int64)
    cell_to_t = [[[] for _ in range(N)] for _ in range(N)]
    for i, t in enumerate(trans):
        for r in range(N):
            cell_to_t[r][t[r]].append(i)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(M)]
    for r in range(N):
        for c in range(N):
            if cell_to_t[r][c]:
                model.add_exactly_one([x[i] for i in cell_to_t[r][c]])
            else:
                return 0, -1, None, {}
    state = {'n': 0, 'best_ct': -1, 'best_B': None, 'hist': {}}
    ar = np.arange(N)[None, :]

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.time()
        def on_solution_callback(self):
            sel = [i for i in range(M) if self.value(x[i])]
            B = np.zeros((N, N), dtype=np.int8)
            for sym, ti in enumerate(sel):
                t = trans[ti]
                for r in range(N):
                    B[r, t[r]] = sym
            BV = B[ar, TA]
            S = np.sort(BV, axis=1)
            ct = int((np.diff(S, axis=1) != 0).all(axis=1).sum())
            state['n'] += 1
            state['hist'][ct] = state['hist'].get(ct, 0) + 1
            if ct > state['best_ct']:
                state['best_ct'] = ct
                state['best_B'] = B.copy()
            if time.time() - self.t0 > budget_s:
                self.stop_search()

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_workers = 1
    solver.solve(model, CB())
    return state['n'], state['best_ct'], state['best_B'], state['hist']

def main():
    rng = random.Random(77777 + INSTANCE * 1009 + int(time.time()) % 99991)
    log("=" * 70)
    log(f"Transversal climb — instance={INSTANCE}, "
        f"score_thresh={SCORE_THRESH}, stream={STREAM_S}s")

    # start from the depth-1 near-turn square (2112) or its partner
    seed = json.loads((REPO / f"mols10/results/nearturn2_seed_{INSTANCE % 2}.json"
                       ).read_text())
    L = np.array(seed['L'], dtype=np.int8)
    # diversify instances: random warm-up walk before climbing
    for _ in range(INSTANCE + 3):
        L2 = random_move(L, rng)
        if L2 is not None:
            L = L2
    cur = count_transversals(L)
    best_score = cur
    best_ct_overall = -1
    log(f"start count={cur} (after {INSTANCE+3} warm-up moves)")

    T = 40.0
    step = 0
    last_push = time.time()
    while True:
        step += 1
        L2 = random_move(L, rng)
        if L2 is None:
            continue
        n2 = count_transversals(L2)
        if n2 >= 5504:
            # 5504 is (empirically) attained only by the MMM-excluded turn
            # class; its isotopes evade the aligned turnlike() test. Save
            # once for a later isotopy audit, never enter.
            marker = REPO / f"mols10/results/transclimb_5504_{INSTANCE}.json"
            if not marker.exists():
                marker.write_text(json.dumps({'count': int(n2),
                                              'L': L2.tolist()}))
                log(f"step {step}: hit count={n2} (turn-suspect), rejected")
            continue
        accept = n2 >= cur or rng.random() < np.exp(min(0, (n2 - cur)) / T)
        if accept and not turnlike(L2):
            L, cur = L2, n2
        T = max(2.0, T * 0.99995)

        if cur > best_score:
            best_score = cur
            log(f"step {step}: NEW COUNT RECORD {cur} (T={T:.1f})")
            if cur >= SCORE_THRESH:
                n, bct, bB, hist = stream_ct(L, STREAM_S)
                log(f"  ct stream: {n} mates, best_ct={bct}, "
                    f"hist={dict(sorted(hist.items(), reverse=True))}")
                rec = {'count': int(cur), 'L': L.tolist(),
                       'stream_best_ct': bct, 'n_mates': n,
                       'B': bB.tolist() if bB is not None else None}
                BEST.write_text(json.dumps(rec, indent=1))
                if bct > best_ct_overall:
                    best_ct_overall = bct
                    if bct >= 4:
                        git_push(f"transclimb i{INSTANCE}: count={cur} ct={bct}")
        if step % 200 == 0:
            log(f"step {step}: cur={cur} best={best_score} T={T:.1f}")
        if time.time() - last_push > 1800:
            git_push(f"transclimb i{INSTANCE} checkpoint: best={best_score}")
            last_push = time.time()

if __name__ == "__main__":
    main()
