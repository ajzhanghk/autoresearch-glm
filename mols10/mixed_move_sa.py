#!/usr/bin/env python3
"""
Mixed-move SA: IC moves + Row swaps + Column swaps of L3.

Key insight from BFS: E=27 is a strict 4-deep IC local minimum.
IC-only SA cannot escape it efficiently.

Row/col swaps of L3 are VALID moves (result is still a Latin square)
that explore a fundamentally different neighborhood -- they can jump
directly to different basins unreachable by IC moves alone.

The mixed move set:
  - 40% IC moves (2x2 block swap, preserves row/col structure)
  - 30% Row swaps (swap two entire rows of L3)
  - 30% Column swaps (swap two entire columns of L3)

For SA: energy delta for row/col swaps computed in O(N).
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG   = REPO / "mols10/results/mixed_sa.log"
BEST  = REPO / "mols10/results/mixed_sa_best.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

T_START  = 5.0
T_END    = 0.001
N_STEPS  = 50_000_000  # 50M steps per run
SAVE_THRESH = 26
GOAL_E   = 26

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add",str(BEST.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def build_freq(L1, L2, L3):
    f13 = np.zeros((N,N), dtype=np.int32)
    f23 = np.zeros((N,N), dtype=np.int32)
    for r in range(N):
        for c in range(N):
            f13[int(L1[r,c]), int(L3[r,c])] += 1
            f23[int(L2[r,c]), int(L3[r,c])] += 1
    return f13, f23

def E_from_freq(f13, f23):
    return int(np.sum(f13==0)) + int(np.sum(f23==0))

def dE_row_swap(L1, L2, f13, f23, r1, r2):
    """Compute dE from swapping rows r1 and r2 of L3."""
    dE = 0
    # Remove contributions of rows r1, r2; add swapped contributions
    changes_13 = {}  # (a,b) -> net delta to f13
    changes_23 = {}
    for c in range(N):
        # Current: row r1 contributes (L1[r1,c], old_L3[r1,c])
        #          row r2 contributes (L1[r2,c], old_L3[r2,c])
        # After swap: row r1 contributes (L1[r1,c], old_L3[r2,c])
        #             row r2 contributes (L1[r2,c], old_L3[r1,c])
        # (stored in L3_r1c and L3_r2c which we pass as part of L3)
        # We'll compute this inline after noting that we need L3 current values
        pass
    # We'll do this with direct L3 access -- need to pass L3
    return None  # placeholder; use compute_delta_row below

def compute_dE_row(L1, L2, L3, f13, f23, r1, r2):
    """dE for swapping rows r1 and r2 of L3. O(N)."""
    net13 = {}; net23 = {}
    for c in range(N):
        a1 = int(L1[r1,c]); b_old_r1 = int(L3[r1,c]); b_new_r1 = int(L3[r2,c])
        a2 = int(L1[r2,c]); b_old_r2 = int(L3[r2,c]); b_new_r2 = int(L3[r1,c])
        # (a1, old_b_r1) loses 1, (a1, new_b_r1) gains 1
        net13[(a1, b_old_r1)] = net13.get((a1,b_old_r1), 0) - 1
        net13[(a1, b_new_r1)] = net13.get((a1,b_new_r1), 0) + 1
        # (a2, old_b_r2) loses 1, (a2, new_b_r2) gains 1
        net13[(a2, b_old_r2)] = net13.get((a2,b_old_r2), 0) - 1
        net13[(a2, b_new_r2)] = net13.get((a2,b_new_r2), 0) + 1

        x1 = int(L2[r1,c]); y_old_r1 = int(L3[r1,c]); y_new_r1 = int(L3[r2,c])
        x2 = int(L2[r2,c]); y_old_r2 = int(L3[r2,c]); y_new_r2 = int(L3[r1,c])
        net23[(x1, y_old_r1)] = net23.get((x1,y_old_r1), 0) - 1
        net23[(x1, y_new_r1)] = net23.get((x1,y_new_r1), 0) + 1
        net23[(x2, y_old_r2)] = net23.get((x2,y_old_r2), 0) - 1
        net23[(x2, y_new_r2)] = net23.get((x2,y_new_r2), 0) + 1

    dE = 0
    for (k, delta) in net13.items():
        old = f13[k]; new = old + delta
        if old > 0 and new == 0: dE += 1
        if old == 0 and new > 0: dE -= 1
    for (k, delta) in net23.items():
        old = f23[k]; new = old + delta
        if old > 0 and new == 0: dE += 1
        if old == 0 and new > 0: dE -= 1
    return dE, net13, net23

def compute_dE_col(L1, L2, L3, f13, f23, c1, c2):
    """dE for swapping columns c1 and c2 of L3. O(N)."""
    net13 = {}; net23 = {}
    for r in range(N):
        a1 = int(L1[r,c1]); b_old_c1 = int(L3[r,c1]); b_new_c1 = int(L3[r,c2])
        a2 = int(L1[r,c2]); b_old_c2 = int(L3[r,c2]); b_new_c2 = int(L3[r,c1])
        net13[(a1,b_old_c1)] = net13.get((a1,b_old_c1),0)-1
        net13[(a1,b_new_c1)] = net13.get((a1,b_new_c1),0)+1
        net13[(a2,b_old_c2)] = net13.get((a2,b_old_c2),0)-1
        net13[(a2,b_new_c2)] = net13.get((a2,b_new_c2),0)+1

        x1 = int(L2[r,c1]); y_old_c1 = int(L3[r,c1]); y_new_c1 = int(L3[r,c2])
        x2 = int(L2[r,c2]); y_old_c2 = int(L3[r,c2]); y_new_c2 = int(L3[r,c1])
        net23[(x1,y_old_c1)] = net23.get((x1,y_old_c1),0)-1
        net23[(x1,y_new_c1)] = net23.get((x1,y_new_c1),0)+1
        net23[(x2,y_old_c2)] = net23.get((x2,y_old_c2),0)-1
        net23[(x2,y_new_c2)] = net23.get((x2,y_new_c2),0)+1

    dE = 0
    for (k, delta) in net13.items():
        old = f13[k]; new = old + delta
        if old > 0 and new == 0: dE += 1
        if old == 0 and new > 0: dE -= 1
    for (k, delta) in net23.items():
        old = f23[k]; new = old + delta
        if old > 0 and new == 0: dE += 1
        if old == 0 and new > 0: dE -= 1
    return dE, net13, net23

def apply_net(f13, f23, net13, net23):
    for k,d in net13.items(): f13[k] += d
    for k,d in net23.items(): f23[k] += d

def dE_ic(L3, f13, f23, L1, L2, r1, r2, c1, c2):
    """dE for IC move. Reuse logic from fast_targeted_sa2."""
    A = int(L3[r1,c1]); B = int(L3[r1,c2])
    cells = [(r1,c1,A,B),(r1,c2,B,A),(r2,c1,B,A),(r2,c2,A,B)]
    net13={}; net23={}
    for (r,c,ov,nv) in cells:
        x1=int(L1[r,c]); x2=int(L2[r,c])
        net13[(x1,ov)]=net13.get((x1,ov),0)-1; net13[(x1,nv)]=net13.get((x1,nv),0)+1
        net23[(x2,ov)]=net23.get((x2,ov),0)-1; net23[(x2,nv)]=net23.get((x2,nv),0)+1
    dE=0
    for k,d in net13.items():
        old=f13[k]; new=old+d
        if old>0 and new==0: dE+=1
        if old==0 and new>0: dE-=1
    for k,d in net23.items():
        old=f23[k]; new=old+d
        if old>0 and new==0: dE+=1
        if old==0 and new>0: dE-=1
    return dE, net13, net23

def probe_ic(L3):
    for _ in range(300):
        r1=random.randint(0,N-1); r2=random.randint(0,N-1)
        if r1==r2: continue
        c1=random.randint(0,N-1); c2=random.randint(0,N-1)
        if c1==c2: continue
        A=int(L3[r1,c1]); B=int(L3[r1,c2])
        if A!=B and L3[r2,c1]==B and L3[r2,c2]==A:
            return r1,r2,c1,c2
    return None

# Load pairs + seeds
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']:p for p in all_pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'iso2_tv21_0_5'
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)
L3_anchor = np.array(best_data[PAIR_ID]['L3'], dtype=np.int8).reshape(N,N)
E_anchor = count_clashes(L1,L3_anchor)+count_clashes(L2,L3_anchor)

global_best_E = E_anchor
global_best_L3 = L3_anchor.copy()
if BEST.exists():
    bd=json.loads(BEST.read_text())
    if bd.get('pair_id')==PAIR_ID and bd.get('E',999)<global_best_E:
        global_best_E=bd['E']
        global_best_L3=np.array(bd['L3'],dtype=np.int8).reshape(N,N)

log("="*70)
log(f"Mixed-move SA (IC + row/col swaps) | pair={PAIR_ID} seed_E={E_anchor}")
log(f"T: {T_START}→{T_END}, {N_STEPS//1_000_000}M steps/run")
log("="*70)

rng = random.Random(27182818284 + hash(PAIR_ID)%10000)
run = 0

while True:
    run += 1
    n_perturb = rng.randint(0, 5)
    L3 = global_best_L3.copy()
    # Perturb: mix of IC and row swaps
    for _ in range(n_perturb):
        if rng.random() < 0.5:
            ic = probe_ic(L3)
            if ic:
                r1,r2,c1,c2=ic; A=int(L3[r1,c1]); B=int(L3[r1,c2])
                L3[r1,c1]=B;L3[r1,c2]=A;L3[r2,c1]=A;L3[r2,c2]=B
        else:
            r1,r2 = rng.sample(range(N),2)
            L3[[r1,r2]] = L3[[r2,r1]]

    f13, f23 = build_freq(L1, L2, L3)
    E_curr = E_from_freq(f13, f23)
    E_best_run = E_curr; L3_best = L3.copy()

    t0 = time.time()
    accepted = tried = 0
    move_counts = {'ic':0, 'row':0, 'col':0}

    for step in range(N_STEPS):
        T = T_START * (T_END/T_START)**(step/N_STEPS)
        move_type = rng.random()

        if move_type < 0.40:
            # IC move
            ic = probe_ic(L3)
            if ic is None: continue
            r1,r2,c1,c2 = ic
            dE, net13, net23 = dE_ic(L3, f13, f23, L1, L2, r1,r2,c1,c2)
            if dE < 0 or rng.random() < math.exp(-dE/T):
                apply_net(f13,f23,net13,net23)
                A=int(L3[r1,c1]);B=int(L3[r1,c2])
                L3[r1,c1]=B;L3[r1,c2]=A;L3[r2,c1]=A;L3[r2,c2]=B
                E_curr += dE; accepted += 1; move_counts['ic']+=1
        elif move_type < 0.70:
            # Row swap
            r1,r2 = rng.randint(0,N-1), rng.randint(0,N-1)
            if r1==r2: continue
            dE, net13, net23 = compute_dE_row(L1,L2,L3,f13,f23,r1,r2)
            if dE < 0 or rng.random() < math.exp(-dE/T):
                apply_net(f13,f23,net13,net23)
                L3[[r1,r2]] = L3[[r2,r1]]
                E_curr += dE; accepted += 1; move_counts['row']+=1
        else:
            # Column swap
            c1,c2 = rng.randint(0,N-1), rng.randint(0,N-1)
            if c1==c2: continue
            dE, net13, net23 = compute_dE_col(L1,L2,L3,f13,f23,c1,c2)
            if dE < 0 or rng.random() < math.exp(-dE/T):
                apply_net(f13,f23,net13,net23)
                L3[:,c1],L3[:,c2] = L3[:,c2].copy(),L3[:,c1].copy()
                E_curr += dE; accepted += 1; move_counts['col']+=1

        tried += 1
        if E_curr < E_best_run:
            E_best_run = E_curr; L3_best = L3.copy()
            if E_curr < global_best_E:
                global_best_E = E_curr; global_best_L3 = L3.copy()
                cl13=int(count_clashes(L1,L3)); cl23=int(count_clashes(L2,L3))
                log(f"*** NEW BEST E={global_best_E} cl13={cl13} cl23={cl23} run={run} step={step} ***")
                BEST.write_text(json.dumps({'E':global_best_E,'cl13':cl13,'cl23':cl23,
                    'pair_id':PAIR_ID,'L1':L1.tolist(),'L2':L2.tolist(),'L3':L3.tolist()},indent=2))
                git_push(f"mixed_sa: {PAIR_ID} E={global_best_E}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                if global_best_E == 0:
                    log("*** 3-MOLS FOUND! ***")
                    FOUND.write_text(json.dumps({'found':True,'pair_id':PAIR_ID,
                        'L1':L1.tolist(),'L2':L2.tolist(),'L3':L3.tolist(),
                        'cl13':cl13,'cl23':cl23},indent=2))
                    sys.exit(0)

    elapsed = time.time()-t0
    log(f"run={run:3d} perturb={n_perturb} best_run={E_best_run} global={global_best_E} "
        f"acc={accepted/max(tried,1)*100:.0f}% moves={move_counts} "
        f"{N_STEPS/elapsed:.0f}steps/s t={elapsed:.0f}s")
