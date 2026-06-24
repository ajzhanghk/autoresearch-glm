#!/usr/bin/env python3
"""
CP-SAT feasibility attack on tv_254: can we achieve E <= max_E?

Runs CP-SAT with the E=23 hint L3 and target E <= TARGET.
If OPTIMAL at TARGET → proves global optimum for tv_254.
If FEASIBLE below 23 → new record!
If TIMEOUT → no conclusion (but incremental bound from best feasible).
"""
import json, sys, time, argparse
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, build_and_solve, N

PAIR_ID = sys.argv[1] if len(sys.argv) > 1 else 'tv_254'
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 600
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LOG = REPO / f"mols10/results/cpsat_attack_{PAIR_ID}.log"
FOUND_FILE = REPO / "mols10/results/MOLS10_FOUND.json"
BEST_FILE = REPO / f"mols10/results/cpsat_attack_best_{PAIR_ID}.json"
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

# Load pair
all_pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}
p = pair_map[PAIR_ID]
L1 = np.array(p['L1'], dtype=np.int8).reshape(N,N)
L2 = np.array(p['L2'], dtype=np.int8).reshape(N,N)

# Load best known L3 as hint
hint_L3 = None
current_best_E = 999
for fp in [
    REPO/f"mols10/results/escape_best_{PAIR_ID}.json",
    REPO/f"mols10/results/pt_agg_best_{PAIR_ID}_b.json",
    REPO/f"mols10/results/pt_best_{PAIR_ID}.json",
]:
    if fp.exists():
        d = json.loads(fp.read_text())
        if d.get('pair_id') == PAIR_ID and d.get('E', 999) < current_best_E:
            hint_L3 = np.array(d['L3'], dtype=np.int8).reshape(N,N)
            current_best_E = d['E']

if hint_L3 is None:
    log("ERROR: no hint L3 found"); sys.exit(1)

log("="*70)
log(f"CP-SAT attack — pair={PAIR_ID}  hint_E={current_best_E}  timeout={TIMEOUT}s  workers={WORKERS}")
log(f"Goal: find E<{current_best_E} or prove {current_best_E} is optimal")
log("="*70)

trial = 0
seeds = [42, 137, 271828, 314159, 999999, 1234567, 7654321, 88888888]

while True:
    trial += 1
    seed = seeds[(trial-1) % len(seeds)]
    log(f"Trial {trial} | seed={seed} | timeout={TIMEOUT}s | hint_E={current_best_E}")

    t0 = time.time()
    best_obj, L3_sol, timed_out, is_optimal = build_and_solve(
        L1, L2, N,
        hint_L3=hint_L3,
        timeout_s=TIMEOUT,
        num_workers=WORKERS,
        random_seed=seed,
    )
    elapsed = time.time() - t0

    if L3_sol is not None:
        cl13 = int(count_clashes(L1, L3_sol))
        cl23 = int(count_clashes(L2, L3_sol))
        E = cl13 + cl23
        status = "OPTIMAL" if is_optimal else ("FEASIBLE" if not timed_out else "TIMEOUT+FEASIBLE")
        log(f"  {status}: E={E} cl13={cl13} cl23={cl23}  elapsed={elapsed:.1f}s")

        if E < current_best_E:
            current_best_E = E
            hint_L3 = L3_sol.copy()
            log(f"*** NEW BEST E={E} cl13={cl13} cl23={cl23} ***")
            BEST_FILE.write_text(json.dumps({
                'E': E, 'cl13': cl13, 'cl23': cl23, 'pair_id': PAIR_ID,
                'trial': trial, 'method': 'cpsat_attack',
                'is_optimal': is_optimal,
                'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist()
            }, indent=2))
            git_push(f"cpsat_attack: {PAIR_ID} E={E}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
            if E == 0:
                log("*** 3-MOLS FOUND! ***")
                FOUND_FILE.write_text(json.dumps({
                    'found': True, 'pair_id': PAIR_ID,
                    'L1': L1.tolist(), 'L2': L2.tolist(), 'L3': L3_sol.tolist(),
                    'cl12': int(count_clashes(L1,L2)), 'cl13': cl13, 'cl23': cl23
                }, indent=2))
                import os; os._exit(0)

        if is_optimal:
            log(f"  PROVED OPTIMAL: E={E} is the global minimum for {PAIR_ID}!")
            log("  Switching to exploration with new random seeds...")
            # Rotate seeds and try longer timeout next
            TIMEOUT = min(TIMEOUT * 2, 3600)

    else:
        status = "TIMEOUT-NO_SOLUTION" if timed_out else "INFEASIBLE"
        log(f"  {status}  elapsed={elapsed:.1f}s")
        if not timed_out:
            log(f"  PROVED INFEASIBLE: no L3 exists with E<{current_best_E} for {PAIR_ID}!")
            # This would mean current_best_E is the true minimum
            break
