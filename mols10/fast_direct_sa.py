#!/usr/bin/env python3
"""
Fast direct-pair SA using numba (360k+ steps/s).
Replaces the slow e31_deep_sa.py (4k steps/s).

For each promising pair, runs multiple 10M-step SA runs with
varied seeds and perturbations. Saves new bests to file.
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

sys.path.insert(0, "/tmp")
from fast_sa_numba import run_sa_fast, make_pc, _run_sa_core

LOG       = REPO / "mols10/results/fast_deep_sa.log"
BEST_FILE = REPO / "mols10/results/fast_deep_sa_best.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

N_STEPS   = 10_000_000  # 10M steps per run = 28s at 360k steps/s
T_START   = 10.0
T_END     = 0.03
GOAL_E    = 26  # must beat current global best

PAIR_IDS = [
    'tv_222', 'tv_223', 'tv_254', 'tv_270',
    'iso2_tv21_0_1',
    'tv_147', 'tv_225', 'tv_238',
    'iso2_tv21_0_0', 'iso2_tv21_0_5',
    'iso_tv_10_2', 'iso_tv_10_1',
    'iso_tv_21_0',  # global best E=26, try harder
    'iso_tv_21_1', 'iso_tv_21_2', 'iso_tv_10_0',
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    subprocess.run(["git", "-C", str(REPO), "add", str(BEST_FILE.relative_to(REPO))], check=False)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg], check=False)
    subprocess.run(["git", "-C", str(REPO), "push", "-u", "origin", BRANCH], check=False)


def probe_ic_py(L, max_tries=100):
    for _ in range(max_tries):
        r1 = random.randint(0, N-1); r2 = random.randint(0, N-1)
        if r1 == r2: continue
        c1 = random.randint(0, N-1); c2 = random.randint(0, N-1)
        if c1 == c2: continue
        a, b = int(L[r1, c1]), int(L[r1, c2])
        if a != b and L[r2, c1] == b and L[r2, c2] == a:
            return r1, r2, c1, c2
    return None


# Load pairs
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Load seeds
seeds = []
for fname, label in [
    ('iso_tv21_0_best_l3.json', 'iso0_E26'),
    ('mdecomp113_best.json', 'md113_E28'),
    ('mdecomp56_best.json', 'md56_E30'),
    ('tv222_e30_seed.json', 'tv222_E30'),
]:
    fpath = REPO / "mols10/results" / fname
    if fpath.exists():
        d = json.loads(fpath.read_text())
        if 'L3' in d:
            seeds.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), label, d.get('E', 30)))

# Survey seeds
tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
if tv_survey_file.exists():
    tv_survey = json.loads(tv_survey_file.read_text())
    for pid in ['tv_222', 'tv_223', 'tv_254', 'tv_270']:
        if pid in tv_survey and tv_survey[pid].get('L3'):
            L3s = np.array(tv_survey[pid]['L3'], dtype=np.int8).reshape(N, N)
            seeds.append((L3s, f'{pid}_E{tv_survey[pid]["E"]}', tv_survey[pid]['E']))

# Also load from fast_deep_sa_best.json for pair-specific seeds
bests = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
pair_best_seeds = {}
for pid, entry in bests.items():
    if entry.get('L3') and entry.get('E', 999) <= 32:
        pair_best_seeds[pid] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])

log("=" * 70)
log(f"Fast direct-pair SA (numba, {N_STEPS//1_000_000}M steps/run)")
log(f"Seeds: {[s[1] for s in seeds]}")
log(f"Pairs: {PAIR_IDS}")
log("=" * 70)

rng = random.Random(27182818)
run_global = 0
pair_run_count = {}

while True:
    # Reload dynamic seeds periodically
    if run_global % 50 == 0:
        for fname, label in [('tv222_e30_seed.json', 'tv222_E30'),
                              ('fast_iso_best.json', 'fast_iso')]:
            fpath = REPO / "mols10/results" / fname
            if fpath.exists():
                try:
                    d = json.loads(fpath.read_text())
                    e = d.get('global_best_E', d.get('E', 999))
                    if 'L3' in d and e <= 30 and label not in [s[1] for s in seeds]:
                        seeds.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), label, e))
                except: pass
        bests = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}

    for pair_id in PAIR_IDS:
        if pair_id not in pair_map:
            continue

        p = pair_map[pair_id]
        L1 = np.array(p['L1'], dtype=np.int8).reshape(N, N)
        L2 = np.array(p['L2'], dtype=np.int8).reshape(N, N)

        pair_best_E = bests.get(pair_id, {}).get('E', 9999)
        L3_pair_best = None
        if pair_id in bests and bests[pair_id].get('L3'):
            L3_pair_best = np.array(bests[pair_id]['L3'], dtype=np.int8).reshape(N, N)
            pair_best_seeds[pair_id] = (L3_pair_best, pair_best_E)

        run_global += 1
        pair_run = pair_run_count.get(pair_id, 0)
        pair_run_count[pair_id] = pair_run + 1

        seed_rng = rng.randint(1, 2**31-1)
        random.seed(seed_rng)

        # Build start options
        pair_specific = []
        tv_survey_file2 = REPO / "mols10/results/tv_unique_survey.json"
        if tv_survey_file2.exists():
            tv_data = json.loads(tv_survey_file2.read_text())
            if pair_id in tv_data and tv_data[pair_id].get('L3'):
                L3s = np.array(tv_data[pair_id]['L3'], dtype=np.int8).reshape(N, N)
                pair_specific.append((L3s, f'survey_E{tv_data[pair_id]["E"]}', tv_data[pair_id]['E']))
        if pair_id in pair_best_seeds:
            L3b, Eb = pair_best_seeds[pair_id]
            pair_specific.append((L3b, 'pair_best', Eb))
        pair_specific.sort(key=lambda x: x[2])

        # Select starting seed
        all_starts = pair_specific + seeds
        L3_seed, seed_label, seed_E = all_starts[pair_run % len(all_starts)]

        # Perturb starting L3
        L3 = L3_seed.copy()
        max_p = 2 if seed_E <= 31 else 5
        n_perturb = rng.randint(0, max_p)
        for _ in range(n_perturb):
            ic = probe_ic_py(L3)
            if ic:
                r1, r2, c1, c2 = ic
                a, b = int(L3[r1,c1]), int(L3[r1,c2])
                L3[r1,c1]=b; L3[r1,c2]=a; L3[r2,c1]=a; L3[r2,c2]=b

        t0 = time.time()
        E_best, L3_best = run_sa_fast(L1, L2, L3, N_STEPS, seed_rng, T_START, T_END)
        elapsed = time.time() - t0

        log_msg = (f"run={run_global:4d} {pair_id}: perturb={n_perturb} seed={seed_label}({seed_E}) "
                   f"E={E_best} pair_best={pair_best_E} t={elapsed:.0f}s {N_STEPS/elapsed:.0f}s/s")

        if E_best < pair_best_E:
            pair_best_E = E_best
            cl13 = count_clashes(L1, L3_best)
            cl23 = count_clashes(L2, L3_best)
            log(f"  *** {log_msg} → NEW PAIR BEST! cl13={cl13} cl23={cl23}")
            bests[pair_id] = {'E': int(E_best), 'cl13': int(cl13), 'cl23': int(cl23),
                               'L3': L3_best.tolist()}
            pair_best_seeds[pair_id] = (L3_best, E_best)
            BEST_FILE.write_text(json.dumps(bests, indent=2))

            if E_best < GOAL_E:
                log(f"*** {pair_id}: NEW GLOBAL BEST E={E_best}! Pushing...")
                git_push(f"fast_deep_sa: {pair_id} E={E_best} (new global best!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                if E_best == 0:
                    log("*** 3-MOLS FOUND! ***"); sys.exit(0)
        else:
            log(log_msg)
