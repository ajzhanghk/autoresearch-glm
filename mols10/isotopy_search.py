#!/usr/bin/env python3
"""
Isotopy search: for a given base pair (L1, L2), generate random isotopic
transformations and survey each with SA to find which isotopy achieves
the lowest E. Targets pairs found by tv_unique_survey with E <= THRESHOLD.

An isotopy of L is: L'[rp[r]][cp[c]] = sp[L[r][c]]
For a pair, we apply (possibly different) isotopies to L1 and L2.
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG          = REPO / "mols10/results/isotopy_search.log"
RESULTS_FILE = REPO / "mols10/results/isotopy_search.json"
BRANCH       = "claude/mols-order-10-search-yfQXK"

E_THRESHOLD = 36     # survey pairs to isotopy-search
N_ISOTOPIES = 50     # random isotopies per base pair per round
N_STEPS     = 500_000
T_START     = 8.0
T_END       = 0.04


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    for f in [str(RESULTS_FILE.relative_to(REPO))]:
        subprocess.run(["git", "-C", str(REPO), "add", f], check=False)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg], check=False)
    subprocess.run(["git", "-C", str(REPO), "push", "-u", "origin", BRANCH], check=False)


def probe_ic(L, max_tries=400):
    for _ in range(max_tries):
        r1 = random.randint(0, N-1); r2 = random.randint(0, N-1)
        if r1 == r2: continue
        c1 = random.randint(0, N-1); c2 = random.randint(0, N-1)
        if c1 == c2: continue
        a, b = int(L[r1, c1]), int(L[r1, c2])
        if a != b and L[r2, c1] == b and L[r2, c2] == a:
            return r1, r2, c1, c2
    return None


def apply_isotopy(L, rp, cp, sp):
    """Apply row perm rp, col perm cp, symbol perm sp to L."""
    L2 = np.empty((N, N), dtype=np.int8)
    for r in range(N):
        for c in range(N):
            L2[rp[r], cp[c]] = sp[int(L[r, c])]
    return L2


def random_isotopy(L1, L2, rng):
    """Generate a random isotopic transformation of the pair (L1, L2)."""
    rp1 = list(range(N)); rng.shuffle(rp1)
    cp1 = list(range(N)); rng.shuffle(cp1)
    sp1 = list(range(N)); rng.shuffle(sp1)
    rp2 = list(range(N)); rng.shuffle(rp2)
    cp2 = list(range(N)); rng.shuffle(cp2)
    sp2 = list(range(N)); rng.shuffle(sp2)
    return apply_isotopy(L1, rp1, cp1, sp1), apply_isotopy(L2, rp2, cp2, sp2)


def run_sa(L1, L2, L3_init, n_steps, seed, n_perturb=0):
    random.seed(seed)
    L3 = L3_init.copy()
    for _ in range(n_perturb):
        ic = probe_ic(L3)
        if ic:
            r1, r2, c1, c2 = ic
            a, b = int(L3[r1,c1]), int(L3[r1,c2])
            L3[r1,c1]=b; L3[r1,c2]=a; L3[r2,c1]=a; L3[r2,c2]=b
    E = count_clashes(L1, L3) + count_clashes(L2, L3)
    E_best = E; L3_best = L3.copy()
    for step in range(n_steps):
        T = T_START * (T_END/T_START)**(step/n_steps)
        ic = probe_ic(L3)
        if ic is None: continue
        r1, r2, c1, c2 = ic
        L3p = L3.copy()
        a, b = int(L3[r1,c1]), int(L3[r1,c2])
        L3p[r1,c1]=b; L3p[r1,c2]=a; L3p[r2,c1]=a; L3p[r2,c2]=b
        E_new = count_clashes(L1, L3p) + count_clashes(L2, L3p)
        dE = E_new - E
        if dE < 0 or random.random() < math.exp(-dE/T):
            L3 = L3p; E = E_new
            if E < E_best:
                E_best = E; L3_best = L3.copy()
    return E_best, L3_best


# Load all pairs
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Load cross-seeds for initializing SA
seed_data = []
for f in ['iso_tv21_0_best_l3.json', 'mdecomp113_best.json']:
    fpath = REPO / "mols10/results" / f
    if fpath.exists():
        d = json.loads(fpath.read_text())
        if 'L3' in d:
            seed_data.append(np.array(d['L3'], dtype=np.int8).reshape(N, N))

# Load existing results
existing = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}

rng = random.Random(31415926)
global_best_E = 9999

log("=" * 70)
log(f"Isotopy search: E_threshold={E_THRESHOLD}, N_ISOTOPIES={N_ISOTOPIES}, N_STEPS={N_STEPS//1000}k")

while True:
    # Load tv_unique_survey results (updated as survey runs)
    tv_survey = {}
    fpath = REPO / "mols10/results/tv_unique_survey.json"
    if fpath.exists():
        raw = json.loads(fpath.read_text())
        for pid, entry in raw.items():
            if entry.get('E', 9999) <= E_THRESHOLD and entry.get('L3') is not None:
                tv_survey[pid] = entry

    if not tv_survey:
        log("No tv_unique_survey pairs with E <= threshold yet. Waiting 60s...")
        time.sleep(60)
        continue

    for base_pid, base_entry in sorted(tv_survey.items(), key=lambda x: x[1]['E']):
        if base_pid not in pair_map:
            continue
        p = pair_map[base_pid]
        L1_base = np.array(p['L1'], dtype=np.int8).reshape(N, N)
        L2_base = np.array(p['L2'], dtype=np.int8).reshape(N, N)
        base_E = base_entry['E']

        log(f"--- Isotopy search for {base_pid} (base E={base_E}) ---")

        pair_best_E = existing.get(base_pid, {}).get('best_E', 9999)
        best_isotopy = existing.get(base_pid, {}).get('best_isotopy', None)

        for i in range(N_ISOTOPIES):
            L1_iso, L2_iso = random_isotopy(L1_base, L2_base, rng)

            # Quick SA with each cross-seed
            run_best_E = 9999
            run_best_L3 = None
            for L3_seed in seed_data:
                seed = rng.randint(1, 2**31-1)
                n_p = rng.randint(0, 6)
                E, L3 = run_sa(L1_iso, L2_iso, L3_seed, N_STEPS, seed, n_perturb=n_p)
                if E < run_best_E:
                    run_best_E = E; run_best_L3 = L3.copy()

            if run_best_E < pair_best_E:
                pair_best_E = run_best_E
                cl13 = count_clashes(L1_iso, run_best_L3)
                cl23 = count_clashes(L2_iso, run_best_L3)
                log(f"  *** Isotopy {i}: NEW BEST E={pair_best_E} cl13={cl13} cl23={cl23} ***")
                existing[base_pid] = {
                    'base_pair': base_pid,
                    'base_E': int(base_E),
                    'best_E': int(pair_best_E),
                    'isotopy_index': i,
                    'L1': L1_iso.tolist(),
                    'L2': L2_iso.tolist(),
                    'L3': run_best_L3.tolist(),
                    'cl13': int(cl13), 'cl23': int(cl23)
                }
                RESULTS_FILE.write_text(json.dumps(existing, indent=2))
                git_push(f"isotopy_search: {base_pid} isotopy→E={pair_best_E}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                if pair_best_E == 0:
                    log("*** 3-MOLS FOUND! ***"); sys.exit(0)
                if pair_best_E < global_best_E:
                    global_best_E = pair_best_E
            elif (i + 1) % 10 == 0:
                log(f"  Isotopy {i+1}/{N_ISOTOPIES}: current best E={pair_best_E}")

        log(f"  Done {base_pid}: best isotopy E={pair_best_E} (base was E={base_E})")
