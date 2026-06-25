#!/usr/bin/env python3
"""
Search for orthogonal pairs with many common transversals.

A common transversal of (A,B) is a set of N cells with:
- All rows distinct, all cols distinct, all A-values distinct, all B-values distinct

For 3-MOLS(N), need a pair with N disjoint common transversals.

Strategy: start with a known orthogonal pair, apply isotopy operations
(row/col/symbol permutations) to maximize common transversals.
Also try completely new random orthogonal pairs.
"""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else '0'
LOG = REPO / f"mols10/results/ct_search_{INSTANCE}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
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

def count_common_transversals(A, B, max_count=200):
    found = [0]
    def bt(row, cols, a_used, b_used):
        if row == N:
            found[0] += 1
            return found[0] >= max_count
        for col in range(N):
            if col not in cols:
                va = int(A[row,col]); vb = int(B[row,col])
                if va not in a_used and vb not in b_used:
                    cols.add(col); a_used.add(va); b_used.add(vb)
                    if bt(row+1, cols, a_used, b_used): return True
                    cols.discard(col); a_used.discard(va); b_used.discard(vb)
        return False
    bt(0, set(), set(), set())
    return found[0]

def find_disjoint_ct(A, B, all_cts, target=N):
    """Try to find N disjoint common transversals."""
    trans_masks = [0] * len(all_cts)
    for i, t in enumerate(all_cts):
        for r, c, va, vb in t:
            trans_masks[i] |= (1 << (r*N + c))

    result = _find_disjoint(trans_masks, 0, 0, [], target)
    return result

def _find_disjoint(masks, used_mask, start_idx, chosen, target):
    if len(chosen) == target:
        return chosen
    remaining_count = sum(1 for i in range(start_idx, len(masks))
                         if not (masks[i] & used_mask))
    if remaining_count < target - len(chosen):
        return None
    for i in range(start_idx, len(masks)):
        if not (masks[i] & used_mask):
            result = _find_disjoint(masks, used_mask | masks[i], i+1,
                                    chosen + [i], target)
            if result is not None:
                return result
    return None

def enum_common_transversals(A, B):
    """Enumerate all common transversals of A and B."""
    result = []
    def bt(row, cols, a_used, b_used, path):
        if row == N:
            result.append(tuple(path))
            return
        for col in range(N):
            if col not in cols:
                va = int(A[row,col]); vb = int(B[row,col])
                if va not in a_used and vb not in b_used:
                    cols.add(col); a_used.add(va); b_used.add(vb)
                    path.append((row,col,va,vb))
                    bt(row+1, cols, a_used, b_used, path)
                    path.pop()
                    cols.discard(col); a_used.discard(va); b_used.discard(vb)
    bt(0, set(), set(), set(), [])
    return result

def apply_isotopy(A, B, rng):
    """Apply random isotopy to orthogonal pair (A, B), preserving orthogonality."""
    op = rng.randint(0, 5)
    if op == 0:
        # Permute rows
        perm = list(range(N)); rng.shuffle(perm)
        return A[perm], B[perm]
    elif op == 1:
        # Permute columns
        perm = list(range(N)); rng.shuffle(perm)
        return A[:,perm], B[:,perm]
    elif op == 2:
        # Permute symbols of A
        perm = list(range(N)); rng.shuffle(perm)
        An = np.zeros_like(A)
        for i in range(N):
            for j in range(N):
                An[i,j] = perm[int(A[i,j])]
        return An, B
    elif op == 3:
        # Permute symbols of B
        perm = list(range(N)); rng.shuffle(perm)
        Bn = np.zeros_like(B)
        for i in range(N):
            for j in range(N):
                Bn[i,j] = perm[int(B[i,j])]
        return A, Bn
    elif op == 4:
        # Swap A and B
        return B.copy(), A.copy()
    else:
        # Transpose both and swap
        return A.T.copy(), B.T.copy()

# Load all pairs and find the one with most common transversals as start
pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
# Filter to truly orthogonal pairs
orth_pairs = []
for p in pairs:
    L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
    L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
    if count_clashes(L1, L2) == 0:
        orth_pairs.append((L1, L2, p['pair_id']))

log("="*70)
log(f"CT Search — instance={INSTANCE}")
log(f"Starting with {len(orth_pairs)} orthogonal pairs")
log("Maximizing common transversals via isotopy search")
log("="*70)

rng = random.Random(314159265 + int(INSTANCE) * 999983)
t_start = time.time()

# Start from a random pair
pair_idx = rng.randint(0, len(orth_pairs)-1)
best_A, best_B, best_id = orth_pairs[pair_idx]
best_ct = count_common_transversals(best_A, best_B)
log(f"Starting pair: {best_id}, ct={best_ct}")

current_A, current_B = best_A.copy(), best_B.copy()
current_ct = best_ct

global_best_ct = best_ct
global_best_A = best_A.copy()
global_best_B = best_B.copy()

trial = 0
restart_interval = 500

while True:
    trial += 1

    # Periodically restart from a different random pair
    if trial % restart_interval == 0:
        pair_idx = rng.randint(0, len(orth_pairs)-1)
        new_A, new_B, new_id = orth_pairs[pair_idx]
        new_ct = count_common_transversals(new_A, new_B)
        current_A, current_B = new_A.copy(), new_B.copy()
        current_ct = new_ct
        elapsed = time.time() - t_start
        log(f"Restart {trial//restart_interval}: new pair={new_id}, ct={new_ct}, global_best={global_best_ct}, t={elapsed:.0f}s")

    # Apply random isotopy
    new_A, new_B = apply_isotopy(current_A, current_B, rng)
    new_ct = count_common_transversals(new_A, new_B)

    # Greedy acceptance: accept if ct improves
    if new_ct >= current_ct:
        current_A, current_B = new_A, new_B
        current_ct = new_ct

    if new_ct > global_best_ct:
        global_best_ct = new_ct
        global_best_A = new_A.copy()
        global_best_B = new_B.copy()
        elapsed = time.time() - t_start
        log(f"*** NEW BEST ct={new_ct} trial={trial} t={elapsed:.0f}s ***")

        if new_ct >= N:
            # Check if N disjoint CTs exist
            log(f"ct={new_ct} >= N={N}! Looking for {N} disjoint common transversals...")
            all_cts = enum_common_transversals(new_A, new_B)
            log(f"  Total CTs: {len(all_cts)}")
            chosen_idxs = find_disjoint_ct(new_A, new_B, all_cts, target=N)
            if chosen_idxs is not None:
                log(f"*** FOUND {N} DISJOINT COMMON TRANSVERSALS! Building L'' ***")
                # Build L'' from the disjoint common transversals
                L_pp = np.zeros((N,N), dtype=np.int8)
                for symbol, idx in enumerate(chosen_idxs):
                    for r,c,va,vb in all_cts[idx]:
                        L_pp[r,c] = symbol
                cl_A = count_clashes(new_A, L_pp)
                cl_B = count_clashes(new_B, L_pp)
                log(f"  cl(A, L'') = {cl_A}, cl(B, L'') = {cl_B}")
                if cl_A == 0 and cl_B == 0:
                    log("*** 3-MOLS FOUND! ***")
                    FOUND_FILE.write_text(json.dumps({
                        'found': True, 'method': 'ct_search',
                        'instance': INSTANCE,
                        'L1': new_A.tolist(), 'L2': new_B.tolist(), 'L3': L_pp.tolist(),
                        'cl12': int(count_clashes(new_A, new_B)),
                        'cl13': cl_A, 'cl23': cl_B
                    }, indent=2))
                    git_push("MOLS10 FOUND via ct_search!\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                    import os; os._exit(0)

    if trial % 1000 == 0:
        elapsed = time.time() - t_start
        rate = trial / elapsed * 3600
        log(f"Trial {trial} | best_ct={global_best_ct} | current_ct={current_ct} | {rate:.0f}/hr | t={elapsed:.0f}s")
