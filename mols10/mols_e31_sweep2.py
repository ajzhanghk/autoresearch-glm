#!/usr/bin/env python3
"""900s sweep on E=31 pair — targeting E<=30."""

import json, sys, time, fcntl, argparse
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))

from mols_cpsat_worker import (
    build_and_solve, count_clashes, N,
    TRIPLE_MISS_FILE, POOL_LOCK, RESULTS_DIR, FOUND_FILE
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed-start", type=int, default=200)
parser.add_argument("--seed-end", type=int, default=300)
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--symbreak", type=int, default=1)
parser.add_argument("--log-suffix", type=str, default="")
args = parser.parse_args()

LOG_FILE = REPO / f"mols10/results/l3_e31_sweep2{args.log_suffix}.log"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_to_pool(L1, L2, L3, cl12, cl13, cl23, seed):
    E = cl13 + cl23
    entry = {
        "clashes": E, "cl12": cl12, "L1_key": str(L1.tolist()),
        "ts": datetime.now().isoformat(),
        "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(),
        "seed": f"sweep2_{seed}", "solver": "cpsat",
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

def main():
    log("="*60)
    log(f"E=31 sweep s={args.seed_start}-{args.seed_end} t={args.timeout}s {args.workers}w sym={bool(args.symbreak)}")

    pool = json.loads(TRIPLE_MISS_FILE.read_text()) if TRIPLE_MISS_FILE.exists() else []
    best_entry = min(pool, key=lambda e: e["clashes"]) if pool else None
    if best_entry is None:
        log("ERROR: no pool"); return

    L1 = np.array(best_entry["L1"], dtype=np.int8).reshape(N, N)
    L2 = np.array(best_entry["L2"], dtype=np.int8).reshape(N, N)
    hint = np.array(best_entry["L3"], dtype=np.int8).reshape(N, N)
    session_best = best_entry["clashes"]
    log(f"Best entry E={session_best}")

    symbreak = bool(args.symbreak)

    for rseed in range(args.seed_start, args.seed_end):
        try:
            pool_data = json.loads(TRIPLE_MISS_FILE.read_text())
            cur_best = min(pool_data, key=lambda e: e["clashes"])
            if cur_best["clashes"] < session_best:
                best_entry = cur_best
                L1 = np.array(best_entry["L1"], dtype=np.int8).reshape(N, N)
                L2 = np.array(best_entry["L2"], dtype=np.int8).reshape(N, N)
                hint = np.array(best_entry["L3"], dtype=np.int8).reshape(N, N)
                session_best = cur_best["clashes"]
                log(f"  → new global best E={session_best}")
        except Exception:
            pass

        log(f"  seed={rseed}  best_so_far={session_best}")
        t0 = time.time()
        result = build_and_solve(L1, L2, N, hint_L3=hint,
            timeout_s=args.timeout, num_workers=args.workers,
            symbreak=symbreak, random_seed=rseed)
        elapsed = time.time() - t0

        best_obj, L3_sol, timed_out, is_optimal = result
        if L3_sol is not None:
            cl13 = count_clashes(L1, L3_sol)
            cl23 = count_clashes(L2, L3_sol)
            E = cl13 + cl23
            cl12 = count_clashes(L1, L2)
            log(f"    {'OPT' if is_optimal else 'FEAS'}: E={E} cl13={cl13} cl23={cl23} t={elapsed:.1f}s")

            if E == 0:
                log("!!! N(10) >= 3 FOUND !!!")
                FOUND_FILE.write_text(json.dumps({
                    "found": True, "L1": L1.tolist(), "L2": L2.tolist(),
                    "L3": L3_sol.tolist(), "cl12": cl12, "cl13": cl13, "cl23": cl23,
                }, indent=2))
                return

            if E < session_best:
                session_best = E
                hint = L3_sol.copy()
                saved = save_to_pool(L1, L2, L3_sol, cl12, cl13, cl23, rseed)
                log(f"  ★★★ NEW BEST E={E} seed={rseed}  saved={saved} ★★★")
            elif E <= session_best + 3:
                hint = L3_sol.copy()

            if is_optimal and E > 0:
                log(f"    PROVED lower bound E>={E}")
        else:
            log(f"    NO_SOL t={elapsed:.1f}s")

    log(f"Sweep done: best={session_best}")

if __name__ == "__main__":
    main()
