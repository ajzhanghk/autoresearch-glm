#!/usr/bin/env python3
"""
BFS to depth 4 from E=27 local minimum.
Depth 1,2 already confirmed strict 2-deep min.
Check if depth 3 or 4 has any E<27 state.
"""
import json, sys, time
from collections import deque
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG   = REPO / "mols10/results/bfs_e27_deep.log"
BEST  = REPO / "mols10/results/bfs_e27_deep_best.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
MAX_DEPTH = 4

def log(msg):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(L1, L2, L3, E, pair_id, depth):
    import subprocess, json as J
    entry = {'E': E, 'pair_id': pair_id, 'depth': depth,
             'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3.tolist()}
    BEST.write_text(J.dumps(entry, indent=2))
    subprocess.run(["git","-C",str(REPO),"add",str(BEST.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",
        f"bfs_e27_deep: {pair_id} E={E} depth={depth}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
        check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def find_ics(L3):
    ics = []
    for r1 in range(N):
        for r2 in range(r1+1, N):
            for c1 in range(N):
                A = int(L3[r1,c1]); B = int(L3[r2,c1])
                if A == B: continue
                for c2 in range(N):
                    if c2 == c1: continue
                    if int(L3[r1,c2]) == B and int(L3[r2,c2]) == A:
                        ics.append((r1,r2,c1,c2))
    return ics

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
log(f"BFS depth-{MAX_DEPTH} from E={E0} | pair={PAIR_ID}")
log("="*70)

t0 = time.time()

# BFS using explicit queue: (L3_bytes, depth)
# State = L3 as bytes for hashing
seen = {}  # key -> (E, depth)
seen[L3_seed.tobytes()] = (E0, 0)

queue = deque()
queue.append((L3_seed.copy(), 0))

best_E = E0
best_L3 = L3_seed.copy()
depth_counts = {0: 1}
depth_min = {0: E0}
total = 0

while queue:
    L3_cur, depth = queue.popleft()
    if depth >= MAX_DEPTH:
        continue

    ics = find_ics(L3_cur)
    for (r1,r2,c1,c2) in ics:
        # Apply IC
        A = int(L3_cur[r1,c1]); B = int(L3_cur[r1,c2])
        L3_new = L3_cur.copy()
        L3_new[r1,c1] = B; L3_new[r1,c2] = A
        L3_new[r2,c1] = A; L3_new[r2,c2] = B

        key = L3_new.tobytes()
        if key in seen:
            continue

        E_new = count_clashes(L1, L3_new) + count_clashes(L2, L3_new)
        seen[key] = (E_new, depth+1)
        total += 1

        d = depth + 1
        depth_counts[d] = depth_counts.get(d, 0) + 1
        if E_new < depth_min.get(d, 9999):
            depth_min[d] = E_new

        if E_new < best_E:
            best_E = E_new
            best_L3 = L3_new.copy()
            cl13 = int(count_clashes(L1, L3_new))
            cl23 = int(count_clashes(L2, L3_new))
            log(f"  *** NEW BEST E={E_new} cl13={cl13} cl23={cl23} depth={d} t={time.time()-t0:.1f}s ***")
            git_push(L1, L2, L3_new, E_new, PAIR_ID, d)
            if E_new == 0:
                log("*** 3-MOLS FOUND! ***")
                sys.exit(0)

        queue.append((L3_new, d))

        if total % 10000 == 0:
            elapsed = time.time()-t0
            log(f"  {total} states | depth_counts={dict(sorted(depth_counts.items()))} | "
                f"depth_min={dict(sorted(depth_min.items()))} | best={best_E} | t={elapsed:.1f}s")

log("="*70)
log(f"BFS COMPLETE | {total} total unique states | best_E={best_E}")
log(f"Depth counts: {dict(sorted(depth_counts.items()))}")
log(f"Min E per depth: {dict(sorted(depth_min.items()))}")
if best_E < E0:
    log(f"IMPROVEMENT FOUND at depth {seen.get(best_L3.tobytes(),(None,'?'))[1]}!")
else:
    log(f"E={E0} is a strict {MAX_DEPTH}-deep IC local minimum for {PAIR_ID}.")
log(f"Total time: {time.time()-t0:.1f}s")
