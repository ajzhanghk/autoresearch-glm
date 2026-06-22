#!/usr/bin/env python3
"""
Adaptive isotopy search for promising pairs.

Two-phase per isotopy:
 Phase 1: 500k-step quick SA to estimate energy
 Phase 2: 8M-step deep SA only if phase1 E <= DEEP_THRESHOLD

Focus pairs: all with survey E <= 34, ordered by best known E.
Saves promising isotopies (E <= SAVE_THRESHOLD) for later analysis.
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG          = REPO / "mols10/results/adaptive_isotopy.log"
RESULTS_FILE = REPO / "mols10/results/adaptive_isotopy_best.json"
PROMISING_FILE = REPO / "mols10/results/adaptive_isotopy_promising.json"
BRANCH       = "claude/mols-order-10-search-yfQXK"

N_STEPS_QUICK = 500_000    # Phase 1: quick filter
N_STEPS_DEEP  = 5_000_000  # Phase 2: deep search if promising
DEEP_THRESHOLD = 31        # Run deep SA if phase1 E <= this (lowered from 30)
SAVE_THRESHOLD = 32        # Save isotopy data if E <= this
GOAL_E = 26                # Must beat current global best

T_START = 10.0
T_END   = 0.03

# Target pairs ordered by priority (best known E first)
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
    'iso_tv_21_1',     # survey E=33
    'iso_tv_21_2',     # survey E=33
    'iso_tv_10_1',     # survey E=33
    'iso2_tv21_0_3',   # survey E=34
    'iso2_tv21_0_4',  # survey E=34
    'iso_tv_10_0',    # survey E=34
]


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


def run_sa(L1, L2, L3_init, n_steps, seed, n_perturb=0, rebuild_every=200):
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

    ics = build_ics(L3)
    rebuild_counter = 0

    for step in range(n_steps):
        rebuild_counter += 1
        if rebuild_counter >= rebuild_every:
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


# Load all pairs
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Load seeds - sorted by E
seed_data = []
for f, label in [('iso_tv21_0_best_l3.json', 'iso_tv21_0_E26'),
                 ('mdecomp113_best.json', 'md113_E28'),
                 ('mdecomp56_best.json', 'md56_E30')]:
    fpath = REPO / "mols10/results" / f
    if fpath.exists():
        d = json.loads(fpath.read_text())
        if 'L3' in d:
            seed_data.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), label))

# Also load tv_unique_survey seeds
tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
if tv_survey_file.exists():
    tv_survey = json.loads(tv_survey_file.read_text())
    for pid in ['tv_222', 'tv_223', 'tv_254', 'tv_270', 'tv_147']:
        if pid in tv_survey and tv_survey[pid].get('L3'):
            E_s = tv_survey[pid]['E']
            seed_data.append((np.array(tv_survey[pid]['L3'], dtype=np.int8).reshape(N, N), f'{pid}_E{E_s}'))

# Load existing results
existing = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
promising = json.loads(PROMISING_FILE.read_text()) if PROMISING_FILE.exists() else []
global_best_E = min(existing.get('global_best_E', GOAL_E), GOAL_E)

log("=" * 70)
log(f"Adaptive isotopy search (phase1={N_STEPS_QUICK//1000}k, phase2={N_STEPS_DEEP//1_000_000}M)")
log(f"Deep threshold: E<={DEEP_THRESHOLD}, Goal: E<{GOAL_E}")
log(f"Pairs: {TARGET_PAIRS}")
log(f"Seeds: {[s[1] for s in seed_data]}")
log("=" * 70)

rng = random.Random(27182818)
iso_num = existing.get('isotopies_tried', 0)
pair_cycle_idx = iso_num % len(TARGET_PAIRS)

while True:
    pair_id = TARGET_PAIRS[pair_cycle_idx % len(TARGET_PAIRS)]
    pair_cycle_idx += 1
    iso_num += 1

    if pair_id not in pair_map:
        log(f"  {pair_id}: NOT IN PAIR MAP, skipping")
        continue

    p = pair_map[pair_id]
    L1_base = np.array(p['L1'], dtype=np.int8).reshape(N, N)
    L2_base = np.array(p['L2'], dtype=np.int8).reshape(N, N)

    # Generate random isotopy for this pair
    rp1 = list(range(N)); rng.shuffle(rp1)
    cp1 = list(range(N)); rng.shuffle(cp1)
    sp1 = list(range(N)); rng.shuffle(sp1)
    rp2 = list(range(N)); rng.shuffle(rp2)
    cp2 = list(range(N)); rng.shuffle(cp2)
    sp2 = list(range(N)); rng.shuffle(sp2)

    L1_iso = apply_isotopy(L1_base, rp1, cp1, sp1)
    L2_iso = apply_isotopy(L2_base, rp2, cp2, sp2)

    t_start = time.time()

    # Phase 1: Quick SA across all seeds
    best_E_phase1 = 9999; best_L3_phase1 = None
    for L3_seed, seed_label in seed_data:
        seed = rng.randint(1, 2**31-1)
        n_p = rng.randint(0, 4)
        E, L3 = run_sa(L1_iso, L2_iso, L3_seed, N_STEPS_QUICK, seed, n_perturb=n_p)
        if E < best_E_phase1:
            best_E_phase1 = E; best_L3_phase1 = L3.copy()

    phase1_time = time.time() - t_start

    if best_E_phase1 <= DEEP_THRESHOLD:
        # Phase 2: Deep SA on this promising isotopy
        log(f"ISO {iso_num} ({pair_id}): phase1 E={best_E_phase1} → running DEEP SA ({N_STEPS_DEEP//1_000_000}M steps)...")
        best_E_deep = best_E_phase1; best_L3_deep = best_L3_phase1.copy()
        for L3_seed, seed_label in seed_data:
            seed = rng.randint(1, 2**31-1)
            n_p = rng.randint(0, 4)
            E, L3 = run_sa(L1_iso, L2_iso, L3_seed, N_STEPS_DEEP, seed, n_perturb=n_p)
            if E < best_E_deep:
                best_E_deep = E; best_L3_deep = L3.copy()

        best_E_this = best_E_deep; best_L3_this = best_L3_deep
        total_time = time.time() - t_start
        log(f"ISO {iso_num} ({pair_id}): DEEP done → E={best_E_this} t={total_time:.0f}s")
    else:
        best_E_this = best_E_phase1; best_L3_this = best_L3_phase1
        log(f"ISO {iso_num} ({pair_id}): phase1 E={best_E_this} t={phase1_time:.0f}s (skipping deep)")

    # Save promising isotopies
    if best_E_this <= SAVE_THRESHOLD:
        entry = {
            'iso_num': iso_num, 'pair_id': pair_id, 'E': int(best_E_this),
            'cl13': int(count_clashes(L1_iso, best_L3_this)),
            'cl23': int(count_clashes(L2_iso, best_L3_this)),
            'L1_iso': L1_iso.tolist(), 'L2_iso': L2_iso.tolist(),
            'L3': best_L3_this.tolist(),
            'rp1': rp1, 'cp1': cp1, 'sp1': sp1,
            'rp2': rp2, 'cp2': cp2, 'sp2': sp2
        }
        promising.append(entry)
        promising.sort(key=lambda x: x['E'])
        PROMISING_FILE.write_text(json.dumps(promising[:50], indent=2))
        log(f"  *** Saved promising isotopy: {pair_id} E={best_E_this} (total saved: {len(promising)})")

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
        git_push(f"adaptive_isotopy: {pair_id} isotopy E={global_best_E} (new global best!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if global_best_E == 0:
            log("*** 3-MOLS FOUND! ***"); sys.exit(0)

    if iso_num % 10 == 0:
        existing['isotopies_tried'] = iso_num
        RESULTS_FILE.write_text(json.dumps(existing, indent=2))
