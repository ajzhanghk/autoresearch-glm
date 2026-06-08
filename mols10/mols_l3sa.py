#!/usr/bin/env python3
"""SA over L3 only (L1,L2 fixed from pool). Fast moves, targets E<31."""
import json, sys, time, fcntl, argparse, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N, TRIPLE_MISS_FILE, POOL_LOCK, FOUND_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=5001)
parser.add_argument("--budget", type=int, default=300)
args = parser.parse_args()

LOG_FILE = REPO / f"mols10/results/l3_only_sa_{args.seed}.log"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_to_pool(L1, L2, L3, cl12, cl13, cl23):
    if not is_latin(L3): return False  # guard: never save invalid LS
    E = cl13 + cl23
    entry = {"clashes": E, "cl12": cl12, "L1_key": str(L1.tolist()),
             "ts": datetime.now().isoformat(), "L1": L1.tolist(), "L2": L2.tolist(),
             "L3": L3.tolist(), "seed": f"l3sa_{args.seed}", "solver": "sa_l3only"}
    lfd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lfd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best = min((p["clashes"] for p in pool), default=999)
        if E > best + 2: return False
        pool.append(entry); pool.sort(key=lambda e: e["clashes"]); pool = pool[:16]
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lfd, fcntl.LOCK_UN); lfd.close()

def compute_E(L1, L2, L3):
    return count_clashes(L1, L3) + count_clashes(L2, L3)

def is_latin(A):
    for row in A:
        if len(set(row.tolist())) != N: return False
    for c in range(N):
        if len(set(A[:,c].tolist())) != N: return False
    return True

def sa_l3(L1, L2, L3_start, budget_s, rng_seed):
    r = random.Random(rng_seed)
    L3 = L3_start.copy()
    assert is_latin(L3), "L3_start must be a valid Latin square"
    E = compute_E(L1, L2, L3)
    bL3 = L3.copy(); bE = E
    T = 1.0; T_min = 0.001; alpha = 0.99995
    t0 = time.time()
    while time.time() - t0 < budget_s:
        T = max(T_min, T * alpha)
        op = r.randint(0, 3)  # 4 moves only — cell_swap removed (always breaks LS)
        if op == 0:
            r1, r2 = r.sample(range(N), 2)
            Ln = L3.copy(); Ln[r1], Ln[r2] = L3[r2].copy(), L3[r1].copy()
        elif op == 1:
            c1, c2 = r.sample(range(N), 2)
            Ln = L3.copy(); Ln[:,c1], Ln[:,c2] = L3[:,c2].copy(), L3[:,c1].copy()
        elif op == 2:
            r1, r2 = r.sample(range(N), 2); c1, c2 = r.sample(range(N), 2)
            v11=int(L3[r1,c1]); v12=int(L3[r1,c2]); v21=int(L3[r2,c1]); v22=int(L3[r2,c2])
            if not (v11==v22 and v12==v21): continue
            Ln=L3.copy(); Ln[r1,c1]=v21; Ln[r1,c2]=v22; Ln[r2,c1]=v11; Ln[r2,c2]=v12
        else:
            s1, s2 = r.sample(range(N), 2)
            Ln = L3.copy(); tmp = Ln.copy(); Ln[tmp==s1]=s2; Ln[tmp==s2]=s1
        En = compute_E(L1, L2, Ln); delta = En - E
        if delta <= 0 or r.random() < math.exp(-delta / T):
            L3 = Ln; E = En
            if E < bE: bE = E; bL3 = L3.copy()
    return bE, bL3

def perturb_intercalate(L3, r, n_flips=3):
    """Small perturbation: apply n_flips random intercalates."""
    import random as _r
    L = L3.copy()
    for _ in range(n_flips):
        rows = _r.sample(range(N), 2); cols = _r.sample(range(N), 2)
        r1, r2, c1, c2 = rows[0], rows[1], cols[0], cols[1]
        v11=int(L[r1,c1]); v12=int(L[r1,c2]); v21=int(L[r2,c1]); v22=int(L[r2,c2])
        if v11==v22 and v12==v21:
            L[r1,c1]=v21; L[r1,c2]=v22; L[r2,c1]=v11; L[r2,c2]=v12
    return L

def main():
    rng = np.random.default_rng(args.seed)
    log("="*60)
    log(f"L3-only SA (seed={args.seed}, budget={args.budget}s)")
    trial=0; session_best=999; L1=None; L2=None; hint=None
    while True:
        try:
            pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
            be = min(pool, key=lambda e: e["clashes"]) if pool else None
            if be is None: log("Empty pool"); time.sleep(30); continue
            if be["clashes"] < session_best or trial == 0:
                L1 = np.array(be["L1"], dtype=np.int8).reshape(N, N)
                L2 = np.array(be["L2"], dtype=np.int8).reshape(N, N)
                hint = np.array(be["L3"], dtype=np.int8).reshape(N, N)
                session_best = be["clashes"]
                if trial > 0: log(f"  → Reloaded E={session_best}")
        except Exception as ex: log(f"Pool error: {ex}"); time.sleep(5); continue
        trial += 1
        # Small perturbation: 1-5 intercalate flips (preserve Latin square property)
        n_flips = int(rng.integers(1, 6))
        L3_start = perturb_intercalate(hint, None, n_flips)
        log(f"trial={trial} best={session_best} flips={n_flips}")
        rseed = int(rng.integers(0, 2**31))
        t0 = time.time()
        E, L3_sol = sa_l3(L1, L2, L3_start, args.budget, rseed)
        cl13 = count_clashes(L1, L3_sol); cl23 = count_clashes(L2, L3_sol)
        cl12 = count_clashes(L1, L2)
        log(f"  E={E} cl13={cl13} cl23={cl23} t={time.time()-t0:.1f}s")
        if E == 0:
            log("!!! N(10) >= 3 FOUND !!!")
            FOUND_FILE.write_text(json.dumps({"found": True, "L1": L1.tolist(), "L2": L2.tolist(),
                "L3": L3_sol.tolist(), "cl12": cl12, "cl13": cl13, "cl23": cl23}, indent=2))
            return
        if E < session_best: session_best = E; hint = L3_sol.copy(); saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23); log(f"  ★ E={E} saved={saved}")
        elif E <= session_best + 2: hint = L3_sol.copy()

if __name__ == "__main__":
    main()
