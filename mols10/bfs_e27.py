#!/usr/bin/env python3
"""
BFS depth-2 exact search from E=27 local minimum.

SA gets stuck at E=27 because all 1-IC neighbors have E>=29 (strict local min).
To escape via IC moves, we need at least 2 steps: E=27 → E=29+ → E<=26.
SA rarely finds such paths because going uphill by 2+ is exponentially unlikely.

This script enumerates ALL 2-IC reachable states from E=27, checking if any
has E < 27. If yes: we have a path! If no: E=27 is a strict 2-deep minimum.

Run for both iso2_tv21_0_5 (E=27) and tv_254 (E=27).
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG   = REPO / "mols10/results/bfs_e27.log"
BEST  = REPO / "mols10/results/bfs_e27_best.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg, file=None):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(L1, L2, L3, E, pair_id):
    import subprocess, json as J
    entry = {'E': E, 'pair_id': pair_id, 'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3.tolist()}
    BEST.write_text(J.dumps(entry, indent=2))
    for fn in [str(BEST.relative_to(REPO))]:
        subprocess.run(["git","-C",str(REPO),"add",fn], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",
        f"bfs_e27: {pair_id} E={E} (NEW BEST!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
        check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def find_valid_ics(L3):
    """Return all valid IC moves as (r1,r2,c1,c2)."""
    ics = []
    for r1 in range(N):
        for r2 in range(r1+1, N):
            for c1 in range(N):
                A = int(L3[r1,c1]); B = int(L3[r2,c1])
                if A == B: continue
                # Find c2 where L3[r1,c2]=B and L3[r2,c2]=A
                for c2 in range(N):
                    if c2 == c1: continue
                    if int(L3[r1,c2]) == B and int(L3[r2,c2]) == A:
                        ics.append((r1, r2, c1, c2))
    return ics

def apply_ic(L3, r1, r2, c1, c2):
    """Apply IC move in place and return (A,B) for rollback."""
    A = int(L3[r1,c1]); B = int(L3[r1,c2])
    L3[r1,c1] = B; L3[r1,c2] = A
    L3[r2,c1] = A; L3[r2,c2] = B
    return A, B

def undo_ic(L3, r1, r2, c1, c2, A, B):
    L3[r1,c1] = A; L3[r1,c2] = B
    L3[r2,c1] = B; L3[r2,c2] = A

def L3_key(L3):
    return L3.tobytes()

# Load pairs
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']:p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'iso2_tv21_0_5'
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L3_seed = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
E0 = count_clashes(L1, L3_seed) + count_clashes(L2, L3_seed)

log("="*70)
log(f"BFS depth-2 from E={E0} minimum | pair={PAIR_ID}")
log("="*70)

t0 = time.time()

# Depth 1: enumerate all 1-IC neighbors
ics_d1 = find_valid_ics(L3_seed)
log(f"Depth 1: {len(ics_d1)} valid IC moves from E={E0} state")

depth1_energies = {}
depth1_min = E0
depth1_min_state = None

for ic1 in ics_d1:
    r1,r2,c1,c2 = ic1
    A,B = apply_ic(L3_seed, r1,r2,c1,c2)
    E1 = count_clashes(L1, L3_seed) + count_clashes(L2, L3_seed)
    depth1_energies[ic1] = E1
    if E1 < depth1_min:
        depth1_min = E1
        depth1_min_state = L3_seed.copy()
    undo_ic(L3_seed, r1,r2,c1,c2, A,B)

log(f"Depth 1: min_E={depth1_min}, distribution: {sorted(set(depth1_energies.values()))[:10]}")
log(f"Depth 1 done in {time.time()-t0:.1f}s")

if depth1_min < E0:
    log(f"IMPROVEMENT at depth 1! E={depth1_min} < {E0}")

# Depth 2: for each depth-1 state, enumerate all its 1-IC neighbors
log(f"Starting depth-2 search ({len(ics_d1)} × ~200 = ~{len(ics_d1)*200} states)...")

depth2_min = depth1_min
depth2_min_L3 = depth1_min_state
n_depth2 = 0
n_improve = 0
best_per_depth1 = []
seen = set()
seen.add(L3_key(L3_seed))

for idx, ic1 in enumerate(ics_d1):
    r1,r2,c1,c2 = ic1
    A,B = apply_ic(L3_seed, r1,r2,c1,c2)
    E1 = depth1_energies[ic1]

    ics_d2 = find_valid_ics(L3_seed)
    local_min = E1
    local_min_L3 = None

    for ic2 in ics_d2:
        r3,r4,c3,c4 = ic2
        A2,B2 = apply_ic(L3_seed, r3,r4,c3,c4)
        key = L3_key(L3_seed)
        if key not in seen:
            seen.add(key)
            E2 = count_clashes(L1, L3_seed) + count_clashes(L2, L3_seed)
            n_depth2 += 1
            if E2 < local_min:
                local_min = E2
                local_min_L3 = L3_seed.copy()
            if E2 < depth2_min:
                depth2_min = E2
                depth2_min_L3 = L3_seed.copy()
                n_improve += 1
                cl13 = int(count_clashes(L1, L3_seed))
                cl23 = int(count_clashes(L2, L3_seed))
                log(f"  *** DEPTH-2 NEW BEST E={E2} (cl13={cl13},cl23={cl23}) via ic1={ic1} ic2={ic2} ***")
                if E2 == 0:
                    log("*** 3-MOLS FOUND! ***")
                    git_push(L1, L2, L3_seed.copy(), 0, PAIR_ID)
                    sys.exit(0)
                git_push(L1, L2, L3_seed.copy(), E2, PAIR_ID)
        undo_ic(L3_seed, r3,r4,c3,c4, A2,B2)

    best_per_depth1.append(local_min)
    undo_ic(L3_seed, r1,r2,c1,c2, A,B)

    if (idx+1) % 50 == 0:
        log(f"  Depth-2 progress: {idx+1}/{len(ics_d1)} depth-1 states done, "
            f"{n_depth2} total states, depth2_min={depth2_min}, t={time.time()-t0:.0f}s")

log("="*70)
log(f"BFS depth-2 COMPLETE | {n_depth2} unique states explored | "
    f"depth2_min={depth2_min} | t={time.time()-t0:.0f}s")

if depth2_min < E0:
    log(f"IMPROVEMENT FOUND at depth 2! E={depth2_min} < {E0}")
    log(f"This means E={E0} is NOT a 2-deep local minimum!")
else:
    log(f"No improvement at depth 2. E={E0} is a strict 2-deep IC local minimum.")
    log(f"Min depth-1 energy: {min(depth1_energies.values())}")
    log(f"Min depth-2 energy: {depth2_min}")

log("="*70)

# Now run depth 3 if time permits (may be large)
log(f"Depth-2 energy distribution (top 20 best): {sorted(best_per_depth1)[:20]}")
