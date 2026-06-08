#!/usr/bin/env python3
"""
Targeted SA on E=32 pair: fix L1,L2 from best pool entry, 
run SA over L3 space with perturbations focusing on clash positions.
"""

import json, sys, time, fcntl, argparse
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))

from mols_cpsat_worker import count_clashes, N, TRIPLE_MISS_FILE, POOL_LOCK, RESULTS_DIR, FOUND_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9001)
parser.add_argument("--log-suffix", type=str, default="")
args = parser.parse_args()

LOG_FILE = REPO / f"mols10/results/l3_sa_e32_{args.seed}{args.log_suffix}.log"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_to_pool(L1, L2, L3, cl12, cl13, cl23):
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12, "L1_key": str(L1.tolist()),
        "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": f"sa_tgt_{args.seed}", "solver": "sa_targeted",
    }
    lock_fd = open(POOL_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
        best = min((p["clashes"] for p in pool), default=999)
        if E > best + 2:
            return False
        pool.append(entry)
        pool.sort(key=lambda e: e["clashes"])
        pool = pool[:16]
        TRIPLE_MISS_FILE.write_text(json.dumps(pool, indent=2))
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def compute_E(L1, L2, L3):
    return count_clashes(L1, L3) + count_clashes(L2, L3)

def swap_row_pair(L3, row, rng):
    """Swap two elements in a row of L3 (preserving Latin square row property)."""
    new_L3 = L3.copy()
    i, j = rng.choice(N, size=2, replace=False)
    new_L3[row, i], new_L3[row, j] = new_L3[row, j], new_L3[row, i]
    return new_L3

def swap_cols_in_row(L3, row, rng):
    """Swap two cells in a row."""
    new_L3 = L3.copy()
    c1, c2 = rng.choice(N, size=2, replace=False)
    new_L3[row, c1], new_L3[row, c2] = new_L3[row, c2], new_L3[row, c1]
    return new_L3

def intercalate_rows(L3, r1, r2, rng):
    """Latin rectangle intercalate on two rows."""
    new_L3 = L3.copy()
    # Find cycles between r1 and r2
    perm = {new_L3[r1,j]: new_L3[r2,j] for j in range(N)}
    inv_perm = {v: k for k, v in perm.items()}
    # Pick a random starting symbol and follow its cycle
    sym = rng.choice(N)
    cycle = []
    cur = sym
    while True:
        cycle.append(cur)
        cur = perm[cur]
        if cur == sym:
            break
    if len(cycle) == N:
        return L3  # Full permutation cycle, no point
    # Apply the cycle swap on one row
    for sym in cycle:
        cols1 = [j for j in range(N) if new_L3[r1,j] == sym]
        cols2 = [j for j in range(N) if new_L3[r2,j] == perm[sym]]
        if cols1 and cols2:
            j1, j2 = cols1[0], cols2[0]
            new_L3[r1,j1], new_L3[r2,j2] = new_L3[r2,j2], new_L3[r1,j1]
    return new_L3

def sa_search(L1, L2, L3_init, rng, max_iters=200000):
    """SA over L3 fixing L1, L2."""
    L3 = L3_init.copy()
    E = compute_E(L1, L2, L3)
    best_L3 = L3.copy()
    best_E = E

    T = 5.0
    T_min = 0.01
    alpha = 0.9999
    iters_per_check = 1000
    check_interval = iters_per_check

    for it in range(max_iters):
        T = max(T_min, T * alpha)
        
        # Choose perturbation type
        op = rng.integers(0, 4)
        if op == 0:
            row = rng.integers(0, N)
            L3_new = swap_cols_in_row(L3, row, rng)
        elif op == 1:
            r1, r2 = rng.choice(N, size=2, replace=False)
            L3_new = intercalate_rows(L3, r1, r2, rng)
        elif op == 2:
            # Swap two cells in the same column (may break Latin)
            # Instead, swap two complete rows
            r1, r2 = rng.choice(N, size=2, replace=False)
            L3_new = L3.copy()
            L3_new[r1], L3_new[r2] = L3[r2].copy(), L3[r1].copy()
        else:
            # Random row permutation (shuffle a row's columns)
            row = rng.integers(0, N)
            L3_new = L3.copy()
            L3_new[row] = rng.permutation(N).astype(np.int8)
            # Check if still valid column-wise... it won't be, so just do swap
            r1, r2 = rng.choice(N, size=2, replace=False)
            L3_new = swap_cols_in_row(L3, r1, rng)

        # Verify Latin square property
        valid = True
        for col in range(N):
            if len(set(L3_new[:, col].tolist())) != N:
                valid = False
                break
        if not valid:
            continue

        E_new = compute_E(L1, L2, L3_new)
        delta = E_new - E

        if delta <= 0 or rng.random() < np.exp(-delta / T):
            L3 = L3_new
            E = E_new
            if E < best_E:
                best_E = E
                best_L3 = L3.copy()

    return best_E, best_L3


def main():
    log("="*60)
    log(f"Targeted SA on E=32 pair (seed={args.seed})")

    rng = np.random.default_rng(args.seed)
    session_best = 999
    trial = 0

    while True:
        # Reload best from pool
        try:
            pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
            best_entry = min(pool, key=lambda e: e["clashes"]) if pool else None
            if best_entry is None:
                log("Empty pool, waiting..."); time.sleep(60); continue
            
            if best_entry["clashes"] < session_best or trial == 0:
                L1 = np.array(best_entry["L1"], dtype=np.int8).reshape(N, N)
                L2 = np.array(best_entry["L2"], dtype=np.int8).reshape(N, N)
                hint = np.array(best_entry["L3"], dtype=np.int8).reshape(N, N)
                session_best = best_entry["clashes"]
                if trial > 0:
                    log(f"  → New global best E={session_best}")
        except Exception as ex:
            log(f"  Pool read error: {ex}"); time.sleep(5); continue

        trial += 1
        log(f"trial={trial}  session_best={session_best}")

        # Perturb hint as starting point
        perturb_strength = min(trial // 5 + 1, 5)
        L3_start = hint.copy()
        for _ in range(perturb_strength):
            r = rng.integers(0, N)
            c1, c2 = rng.choice(N, size=2, replace=False)
            # Swap in row (valid Latin row operation)
            L3_start[r, c1], L3_start[r, c2] = L3_start[r, c2], L3_start[r, c1]

        t0 = time.time()
        E, L3_sol = sa_search(L1, L2, L3_start, rng, max_iters=500000)
        elapsed = time.time() - t0

        cl13 = count_clashes(L1, L3_sol)
        cl23 = count_clashes(L2, L3_sol)
        cl12 = count_clashes(L1, L2)
        log(f"  E={E} cl13={cl13} cl23={cl23} t={elapsed:.1f}s")

        if E == 0:
            log("!!! N(10) >= 3 FOUND !!!")
            FOUND_FILE.write_text(json.dumps({
                "found": True, "L1": L1.tolist(), "L2": L2.tolist(),
                "L3": L3_sol.tolist(), "cl12": cl12, "cl13": cl13, "cl23": cl23,
            }, indent=2))
            return

        if E <= session_best + 3:
            hint = L3_sol.copy()
            if E < session_best:
                session_best = E
                saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23)
                log(f"  ★ NEW BEST E={E}  saved={saved}")


if __name__ == "__main__":
    main()
