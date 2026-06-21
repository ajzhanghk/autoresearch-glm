#!/usr/bin/env python3
"""
Focused deep SA on tv_222 (survey E=31) - new best non-mdecomp pair.
Also includes tv_147 (survey E=32, isotopy E=31).
Uses old probe_random_ic approach which runs at ~5k steps/s.
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG       = REPO / "mols10/results/tv222_focused.log"
BEST_FILE = REPO / "mols10/results/tv222_focused_best.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

T_START = 10.0
T_END   = 0.03
N_STEPS = 5_000_000

PAIR_IDS = [
    'tv_222',          # survey E=31 ← NEW BEST!
    'tv_147',          # survey E=32, isotopy E=31
    'iso2_tv21_0_0',   # survey E=32
    'iso2_tv21_0_1',   # survey E=32
    'iso2_tv21_0_5',   # survey E=32
    'iso_tv_10_2',     # survey E=32
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


def probe_random_ic(L, max_tries=400):
    for _ in range(max_tries):
        r1 = random.randint(0, N-1); r2 = random.randint(0, N-1)
        if r1 == r2: continue
        c1 = random.randint(0, N-1); c2 = random.randint(0, N-1)
        if c1 == c2: continue
        a, b = int(L[r1, c1]), int(L[r1, c2])
        if a != b and L[r2, c1] == b and L[r2, c2] == a:
            return r1, r2, c1, c2
    return None


all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

best_per_pair = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}

# Load seeds
seeds = []
for fname in ['iso_tv21_0_best_l3.json', 'mdecomp113_best.json', 'mdecomp56_best.json']:
    fpath = REPO / "mols10/results" / fname
    if fpath.exists():
        d = json.loads(fpath.read_text())
        seeds.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), int(d['E'])))

# Load tv_unique_survey seeds (including tv_222 E=31 L3)
tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
tv_seeds = {}
if tv_survey_file.exists():
    tv_data = json.loads(tv_survey_file.read_text())
    for pid, entry in tv_data.items():
        if entry.get('L3'):
            tv_seeds[pid] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])

log("=" * 70)
log(f"Focused deep SA on tv_222 (E=31) + tv_147 (E=32)")
log(f"N_STEPS={N_STEPS//1000}k, T: {T_START}→{T_END}")
log(f"Seeds: {[s[1] for s in seeds]}, tv_seeds: {list(tv_seeds.keys())}")
log("=" * 70)

rng = random.Random(31415926)
run_global = 0
pair_run_count = {}

while True:
    # Reload tv_survey to catch updates
    if tv_survey_file.exists():
        tv_data = json.loads(tv_survey_file.read_text())
        for pid, entry in tv_data.items():
            if entry.get('L3') and pid not in tv_seeds:
                tv_seeds[pid] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])

    for pair_id in PAIR_IDS:
        if pair_id not in pair_map:
            log(f"  {pair_id}: NOT IN PAIR MAP, skipping")
            continue

        p = pair_map[pair_id]
        L1 = np.array(p['L1'], dtype=np.int8).reshape(N, N)
        L2 = np.array(p['L2'], dtype=np.int8).reshape(N, N)

        if BEST_FILE.exists():
            best_per_pair = json.loads(BEST_FILE.read_text())

        pair_best = best_per_pair.get(pair_id, {}).get('E', 9999)
        L3_pair_best = None
        if pair_id in best_per_pair:
            L3_pair_best = np.array(best_per_pair[pair_id]['L3'], dtype=np.int8).reshape(N, N)

        run_global += 1
        pair_run = pair_run_count.get(pair_id, 0)
        pair_run_count[pair_id] = pair_run + 1
        seed = rng.randint(1, 2**31-1)
        random.seed(seed)

        # Build start options: pair-specific seeds first, then generic
        pair_seeds = []
        if pair_id in tv_seeds:
            pair_seeds.append(tv_seeds[pair_id])
        if L3_pair_best is not None and (pair_id not in tv_seeds or pair_best < tv_seeds[pair_id][1]):
            pair_seeds.append((L3_pair_best, pair_best))
        pair_seeds.sort(key=lambda x: x[1])
        start_options = pair_seeds + list(seeds)
        L3_seed, seed_E = start_options[pair_run % len(start_options)]

        L3 = L3_seed.copy()
        n_perturb = rng.randint(0, 8)
        for _ in range(n_perturb):
            ic = probe_random_ic(L3)
            if ic:
                r1, r2, c1, c2 = ic
                a, b = int(L3[r1,c1]), int(L3[r1,c2])
                L3[r1,c1]=b; L3[r1,c2]=a; L3[r2,c1]=a; L3[r2,c2]=b

        E_curr = count_clashes(L1, L3) + count_clashes(L2, L3)
        E_best_run = E_curr
        L3_best_run = L3.copy()

        t0 = time.time()
        accepted = tried = 0

        for step in range(N_STEPS):
            T = T_START * (T_END / T_START) ** (step / N_STEPS)
            ic = probe_random_ic(L3)
            if ic is None: continue
            tried += 1
            r1, r2, c1, c2 = ic
            L3p = L3.copy()
            a, b = int(L3[r1,c1]), int(L3[r1,c2])
            L3p[r1,c1]=b; L3p[r1,c2]=a; L3p[r2,c1]=a; L3p[r2,c2]=b
            Ep = count_clashes(L1, L3p) + count_clashes(L2, L3p)
            dE = Ep - E_curr
            if dE < 0 or random.random() < math.exp(-dE / T):
                L3 = L3p; E_curr = Ep; accepted += 1
                if E_curr < E_best_run:
                    E_best_run = E_curr; L3_best_run = L3.copy()
                    cl13 = count_clashes(L1, L3)
                    cl23 = count_clashes(L2, L3)
                    log(f"  {pair_id}: new best E={E_best_run} cl13={cl13} cl23={cl23}")
                    if E_best_run < pair_best:
                        pair_best = E_best_run
                        best_per_pair[pair_id] = {
                            'E': int(pair_best), 'cl13': int(cl13), 'cl23': int(cl23),
                            'L3': L3_best_run.tolist()
                        }
                        BEST_FILE.write_text(json.dumps(best_per_pair, indent=2))
                        git_push(f"tv222_focused: {pair_id} E={pair_best}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                        if pair_best == 0:
                            log("*** 3-MOLS FOUND! ***"); sys.exit(0)

        elapsed = time.time() - t0
        log(f"run={run_global:4d} {pair_id}: perturb={n_perturb} E0={seed_E} best={E_best_run} "
            f"pair_best={pair_best} acc={accepted/max(tried,1)*100:.0f}% "
            f"{N_STEPS/elapsed:.0f}steps/s t={elapsed:.0f}s")
