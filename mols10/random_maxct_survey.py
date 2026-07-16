#!/usr/bin/env python3
"""
First-ever survey (to our knowledge) of the PROVEN max-ct distribution
over UNIFORMLY RANDOM order-10 Latin squares.

All historical searches (ours included) mined biased construction
families — and every family we can build is now proven dead (max-ct <= 3).
This measures the unbiased landscape: Jacobson-Matthews MCMC generates
near-uniform random Latin squares; each sample gets (i) its transversal
count, (ii) a PROVEN max-ct over all orthogonal mates via the exact
CP-SAT model (~1-2 min in the ~800-transversal league).

Outputs the empirical joint distribution (n_transversals, max_ct) — both
a new structural datapoint for the weak CT-barrier conjecture and a
fishing net cast over unbiased square space (any max-ct >= 4 square is
instantly interesting; >= 10 is the fish).

Usage: python3 random_maxct_survey.py <instance> [cap_s]
"""
import json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 600
N = 10
LOG = REPO / f"mols10/results/randsurvey_{INSTANCE}.log"
CKPT = REPO / f"mols10/results/randsurvey_ckpt_{INSTANCE}.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
PUSH_EVERY = 15
JM_STEPS = 3000  # moves between samples (n^3 = 1000 is standard mixing)

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

class JacobsonMatthews:
    """Uniform random Latin square sampler — Jacobson & Matthews (1996).

    Standard incidence-cube formulation: f[r,c,s] in {0,1} with all line
    sums 1 (proper). A move picks f(r,c,s)=0, reads the unique r',c',s'
    with f(r',c,s)=f(r,c',s)=f(r,c,s')=1 and flips the 2x2x2 subcube;
    f(r',c',s') may drop to -1 (improper), in which case the next move is
    anchored at the negative cell with the two candidate lines chosen at
    random, until the cube is proper again.
    """
    def __init__(self, rng):
        self.rng = rng
        self.f = np.zeros((N, N, N), dtype=np.int8)
        for r in range(N):
            for c in range(N):
                self.f[r, c, (r + c) % N] = 1

    def _flip(self, r, c, s, r2, c2, s2):
        f = self.f
        f[r, c, s] += 1; f[r, c, s2] -= 1
        f[r2, c, s] -= 1; f[r, c2, s] -= 1
        f[r2, c2, s] += 1; f[r2, c, s2] += 1
        f[r, c2, s2] += 1; f[r2, c2, s2] -= 1
        return (r2, c2, s2)

    def step(self):
        rng, f = self.rng, self.f
        # pick a zero cell of the cube
        while True:
            r = rng.randrange(N); c = rng.randrange(N); s = rng.randrange(N)
            if f[r, c, s] == 0:
                break
        r2 = int(np.nonzero(f[:, c, s] == 1)[0][0])
        c2 = int(np.nonzero(f[r, :, s] == 1)[0][0])
        s2 = int(np.nonzero(f[r, c, :] == 1)[0][0])
        neg = self._flip(r, c, s, r2, c2, s2)
        for _ in range(100000):
            r, c, s = neg
            if f[r, c, s] >= 0:
                return  # proper again
            # improper: f(r,c,s) = -1; each of the three lines through it
            # now has two 1-entries; choose uniformly
            r2 = int(rng.choice(np.nonzero(f[:, c, s] == 1)[0]))
            c2 = int(rng.choice(np.nonzero(f[r, :, s] == 1)[0]))
            s2 = int(rng.choice(np.nonzero(f[r, c, :] == 1)[0]))
            neg = self._flip(r, c, s, r2, c2, s2)

    def sample(self, k):
        for _ in range(k):
            self.step()
        L = np.argmax(self.f, axis=2).astype(np.int8)
        if not (self.f >= 0).all():
            return None  # should not happen: step() always ends proper
        for r in range(N):
            if len(set(int(v) for v in L[r])) != N: return None
        for c in range(N):
            if len(set(int(v) for v in L[:, c])) != N: return None
        return L

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
        return 'NO_MATE', 0, 0, M
    obj = int(solver.objective_value) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1
    bound = int(solver.best_objective_bound)
    return name, obj, bound, M

def main():
    rng = random.Random(4242 + INSTANCE * 101 + int(time.time()) % 99991)
    jm = JacobsonMatthews(rng)
    log("=" * 70)
    log(f"Uniform-random max-ct survey — instance={INSTANCE}, cap={CAP}s")

    ckpt = {'samples': []}
    if CKPT.exists():
        try: ckpt = json.loads(CKPT.read_text())
        except Exception: pass

    # burn-in
    jm.sample(20000)
    n_since = 0
    while True:
        L = jm.sample(JM_STEPS)
        if L is None:
            continue
        t0 = time.time()
        name, obj, bound, M = prove_maxct(L, CAP)
        el = time.time() - t0
        rec = {'status': name, 'maxct': obj, 'bound': bound,
               'n_trans': M, 'elapsed': round(el)}
        if obj >= 4:
            rec['L'] = L.tolist()
        ckpt['samples'].append(rec)
        CKPT.write_text(json.dumps(ckpt, indent=0))
        n = len(ckpt['samples'])
        from collections import Counter
        dist = Counter(s['maxct'] for s in ckpt['samples']
                       if s['status'] == 'OPTIMAL')
        log(f"sample {n}: {name} maxct={obj} trans={M} in {el:.0f}s "
            f"| dist={dict(sorted(dist.items()))}")
        if obj >= 4:
            git_push(f"random survey: square with proven max-ct={obj}!")
        if obj >= N:
            log(f"*** THE FISH: uniform random square with max-ct={obj} ***")
        n_since += 1
        if n_since >= PUSH_EVERY:
            git_push(f"random maxct survey i{INSTANCE}: {n} samples, "
                     f"dist={dict(sorted(dist.items()))}")
            n_since = 0

if __name__ == "__main__":
    main()
