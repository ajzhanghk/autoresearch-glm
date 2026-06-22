#!/usr/bin/env python3
"""
Isotopy search for multiple promising pairs.
For each pair with survey E <= THRESHOLD, applies random isotopic transformations
to (L1, L2) to find a representation with lower minimum E.

Key pairs:
- iso_tv_21_0: global best E=26; looking for E<26 isotopy
- iso2_tv21_0_*: survey E=32; looking for E<26 isotopy
- tv_147: tv_unique_survey E=32; looking for E<26 isotopy
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG          = REPO / "mols10/results/iso_tv21_isotopy.log"
RESULTS_FILE = REPO / "mols10/results/iso_tv21_isotopy_best.json"
BRANCH       = "claude/mols-order-10-search-yfQXK"

N_STEPS_PER_ISO = 1_000_000  # SA steps per isotopy
T_START = 8.0
T_END   = 0.03


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


def build_ics(L):
    ics = []
    for r1 in range(N):
        for r2 in range(r1 + 1, N):
            for c1 in range(N):
                A = int(L[r1, c1])
                for c2 in range(N):
                    if c2 == c1: continue
                    B = int(L[r1, c2])
                    if A != B and L[r2, c1] == B and L[r2, c2] == A:
                        ics.append((r1, r2, c1, c2))
    return ics


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
    L2 = np.empty((N, N), dtype=np.int8)
    for r in range(N):
        for c in range(N):
            L2[rp[r], cp[c]] = sp[int(L[r, c])]
    return L2


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

    # Precomputed IC list for speed
    ics = build_ics(L3)
    rebuild_counter = 0

    for step in range(n_steps):
        rebuild_counter += 1
        if rebuild_counter >= 200:
            ics = build_ics(L3); rebuild_counter = 0

        T = T_START * (T_END / T_START) ** (step / n_steps)

        ic = None
        if ics:
            cand = ics[random.randint(0, len(ics) - 1)]
            r1, r2, c1, c2 = cand
            a, b = int(L3[r1,c1]), int(L3[r1,c2])
            if a != b and L3[r2,c1] == b and L3[r2,c2] == a:
                ic = cand
        if ic is None:
            ic = probe_ic(L3)
            if ic is None: continue

        r1, r2, c1, c2 = ic
        L3p = L3.copy()
        a, b = int(L3[r1,c1]), int(L3[r1,c2])
        L3p[r1,c1]=b; L3p[r1,c2]=a; L3p[r2,c1]=a; L3p[r2,c2]=b
        E_new = count_clashes(L1, L3p) + count_clashes(L2, L3p)
        dE = E_new - E
        if dE < 0 or random.random() < math.exp(-dE / T):
            L3 = L3p; E = E_new
            if E < E_best:
                E_best = E; L3_best = L3.copy()
    return E_best, L3_best


# Load all pairs and promising ones
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Target pairs for isotopy search (ordered by priority)
TARGET_PAIRS = [
    'tv_222',          # survey E=31
    'tv_223',          # survey E=31
    'tv_254',          # survey E=31
    'tv_270',          # survey E=31
    'iso_tv_21_0',     # global best E=26 → looking for E<26
    'iso2_tv21_0_1',   # focused SA E=31
    'tv_147',          # survey E=32
    'iso2_tv21_0_0',   # survey E=32
    'iso2_tv21_0_5',   # survey E=32
    'iso_tv_10_2',     # survey E=32
]

# Load seeds
seed_data = []
for f in ['iso_tv21_0_best_l3.json', 'mdecomp113_best.json']:
    fpath = REPO / "mols10/results" / f
    if fpath.exists():
        d = json.loads(fpath.read_text())
        if 'L3' in d:
            seed_data.append(np.array(d['L3'], dtype=np.int8).reshape(N, N))
# Also add E=31 survey seeds
tv_survey_f = REPO / "mols10/results/tv_unique_survey.json"
if tv_survey_f.exists():
    tv_data = json.loads(tv_survey_f.read_text())
    for pid in ['tv_222', 'tv_223', 'tv_254', 'tv_270', 'tv_147']:
        if pid in tv_data and tv_data[pid].get('L3'):
            seed_data.append(np.array(tv_data[pid]['L3'], dtype=np.int8).reshape(N, N))

# Load existing results
existing = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
global_best_E = existing.get('global_best_E', 26)
if global_best_E > 26: global_best_E = 26  # Must beat current record

log("=" * 70)
log(f"Isotopy search for multiple promising pairs (goal: E < 26)")
log(f"Targets: {TARGET_PAIRS}")
log(f"N_STEPS={N_STEPS_PER_ISO//1000}k per isotopy per seed, precomputed IC list")
log("=" * 70)

rng = random.Random(16180339)
iso_num = existing.get('isotopies_tried', 0)

# Cycle through target pairs
pair_cycle_idx = iso_num % len(TARGET_PAIRS)

while True:
    pair_id = TARGET_PAIRS[pair_cycle_idx % len(TARGET_PAIRS)]
    pair_cycle_idx += 1
    iso_num += 1

    if pair_id not in pair_map:
        # Try to load from tv_unique_survey
        tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
        if tv_survey_file.exists():
            tv_survey = json.loads(tv_survey_file.read_text())
            if pair_id in tv_survey and tv_survey[pair_id].get('L3'):
                # Use the survey's L1, L2
                pass
        log(f"  {pair_id}: NOT IN PAIR MAP, skipping")
        continue

    p = pair_map[pair_id]
    L1_base = np.array(p['L1'], dtype=np.int8).reshape(N, N)
    L2_base = np.array(p['L2'], dtype=np.int8).reshape(N, N)

    # Generate random isotopy of (L1_base, L2_base)
    rp1 = list(range(N)); rng.shuffle(rp1)
    cp1 = list(range(N)); rng.shuffle(cp1)
    sp1 = list(range(N)); rng.shuffle(sp1)
    rp2 = list(range(N)); rng.shuffle(rp2)
    cp2 = list(range(N)); rng.shuffle(cp2)
    sp2 = list(range(N)); rng.shuffle(sp2)

    L1_iso = apply_isotopy(L1_base, rp1, cp1, sp1)
    L2_iso = apply_isotopy(L2_base, rp2, cp2, sp2)

    # Run SA with each seed
    best_E_this = 9999; best_L3_this = None
    for L3_seed in seed_data:
        seed = rng.randint(1, 2**31-1)
        n_p = rng.randint(0, 6)
        E, L3 = run_sa(L1_iso, L2_iso, L3_seed, N_STEPS_PER_ISO, seed, n_perturb=n_p)
        if E < best_E_this:
            best_E_this = E; best_L3_this = L3.copy()

    if best_E_this < global_best_E:
        global_best_E = best_E_this
        cl13 = count_clashes(L1_iso, best_L3_this)
        cl23 = count_clashes(L2_iso, best_L3_this)
        log(f"*** ISO {iso_num} ({pair_id}): NEW GLOBAL BEST E={global_best_E} cl13={cl13} cl23={cl23} ***")
        existing.update({
            'global_best_E': int(global_best_E),
            'isotopies_tried': iso_num,
            'best_pair': pair_id,
            'best_iso_index': iso_num,
            'L1': L1_iso.tolist(), 'L2': L2_iso.tolist(),
            'L3': best_L3_this.tolist(),
            'cl13': int(cl13), 'cl23': int(cl23)
        })
        RESULTS_FILE.write_text(json.dumps(existing, indent=2))
        git_push(f"isotopy_search: {pair_id} isotopy E={global_best_E} (beats E=26!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if global_best_E == 0:
            log("*** 3-MOLS FOUND! ***"); sys.exit(0)
    else:
        log(f"Isotopy {iso_num} ({pair_id}): best_E={best_E_this}, global_best={global_best_E}")
        if iso_num % 10 == 0:
            existing['isotopies_tried'] = iso_num
            RESULTS_FILE.write_text(json.dumps(existing, indent=2))
