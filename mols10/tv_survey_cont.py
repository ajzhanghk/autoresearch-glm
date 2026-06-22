#!/usr/bin/env python3
"""
Continuation of tv_unique_survey for remaining 7 pairs.
Surveys: tv_280, tv_29, tv_315, tv_32, tv_35, tv_42, tv_44
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG       = REPO / "mols10/results/tv_unique_survey.log"
OUT_FILE  = REPO / "mols10/results/tv_unique_survey.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

# Remaining 7 canonical tv_* pairs
REMAINING = ['tv_280', 'tv_29', 'tv_315', 'tv_32', 'tv_35', 'tv_42', 'tv_44']

T_START = 10.0
T_END   = 0.03
N_STEPS_MAIN = 500_000
N_STEPS_RANDOM = 200_000
N_CROSS = 5
N_RANDOM = 2


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    subprocess.run(["git", "-C", str(REPO), "add", str(OUT_FILE.relative_to(REPO))], check=False)
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


def run_sa(L1, L2, L3_init, n_steps, seed, n_perturb=0):
    random.seed(seed)
    L3 = L3_init.copy()
    for _ in range(n_perturb):
        ic = probe_random_ic(L3)
        if ic:
            r1, r2, c1, c2 = ic
            a, b = int(L3[r1,c1]), int(L3[r1,c2])
            L3[r1,c1]=b; L3[r1,c2]=a; L3[r2,c1]=a; L3[r2,c2]=b
    E = count_clashes(L1, L3) + count_clashes(L2, L3)
    E_best = E; L3_best = L3.copy()
    for step in range(n_steps):
        T = T_START * (T_END / T_START) ** (step / n_steps)
        ic = probe_random_ic(L3)
        if ic is None: continue
        r1, r2, c1, c2 = ic
        L3p = L3.copy()
        a, b = int(L3[r1,c1]), int(L3[r1,c2])
        L3p[r1,c1]=b; L3p[r1,c2]=a; L3p[r2,c1]=a; L3p[r2,c2]=b
        Ep = count_clashes(L1, L3p) + count_clashes(L2, L3p)
        dE = Ep - E
        if dE < 0 or random.random() < math.exp(-dE / T):
            L3 = L3p; E = Ep
            if E < E_best:
                E_best = E; L3_best = L3.copy()
    return E_best, L3_best


all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Load existing results
results = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}
done_so_far = len(results)

# Load seeds
cross_seeds = []
for fname, label in [
    ('iso_tv21_0_best_l3.json', 'iso0_E26'),
    ('mdecomp113_best.json', 'md113_E28'),
    ('mdecomp56_best.json', 'md56_E30'),
]:
    fpath = REPO / "mols10/results" / fname
    if fpath.exists():
        d = json.loads(fpath.read_text())
        cross_seeds.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), label))

# Also add best E=31 seeds from survey
for pid in ['tv_222', 'tv_223', 'tv_254', 'tv_270']:
    if pid in results and results[pid].get('L3'):
        cross_seeds.append((np.array(results[pid]['L3'], dtype=np.int8).reshape(N, N), f'{pid}_E31'))

log("=" * 70)
log(f"tv_unique_survey CONTINUATION: {len(REMAINING)} remaining pairs")
log(f"Seeds: {[s[1] for s in cross_seeds]}")
log("=" * 70)

rng = random.Random(27182818)
total_done = done_so_far

for pair_id in REMAINING:
    if pair_id in results:
        log(f"  {pair_id}: already done (E={results[pair_id]['E']}), skipping")
        continue

    if pair_id not in pair_map:
        log(f"  {pair_id}: not in pair_map, skipping")
        continue

    p = pair_map[pair_id]
    L1 = np.array(p['L1'], dtype=np.int8).reshape(N, N)
    L2 = np.array(p['L2'], dtype=np.int8).reshape(N, N)

    t0 = time.time()
    best_E = 9999; best_L3 = None

    # Use top N_CROSS cross-seeds with random perturbations
    seeds_to_use = list(cross_seeds)[:N_CROSS]
    for L3_seed, seed_label in seeds_to_use:
        seed = rng.randint(1, 2**31-1)
        n_p = rng.randint(0, 4)
        E, L3 = run_sa(L1, L2, L3_seed, N_STEPS_MAIN, seed, n_perturb=n_p)
        if E < best_E:
            best_E = E; best_L3 = L3.copy()

    # Random seeds
    for _ in range(N_RANDOM):
        L3_rand = np.zeros((N, N), dtype=np.int8)
        for r in range(N):
            perm = list(range(N)); random.shuffle(perm)
            L3_rand[r] = perm
        seed = rng.randint(1, 2**31-1)
        E, L3 = run_sa(L1, L2, L3_rand, N_STEPS_RANDOM, seed)
        if E < best_E:
            best_E = E; best_L3 = L3.copy()

    elapsed = time.time() - t0
    total_done += 1
    is_new_best = best_E <= min((v['E'] for v in results.values()), default=9999)
    marker = " *** NEW BEST! ***" if is_new_best else ""

    cl13 = count_clashes(L1, best_L3)
    cl23 = count_clashes(L2, best_L3)
    log(f"[{total_done}/{done_so_far + len(REMAINING)}] {pair_id}: E={best_E} cl13={cl13} cl23={cl23} t={elapsed:.0f}s{marker}")

    results[pair_id] = {
        'E': int(best_E), 'cl13': int(cl13), 'cl23': int(cl23),
        'L3': best_L3.tolist()
    }
    OUT_FILE.write_text(json.dumps(results, indent=2))

    overall_best = min(v['E'] for v in results.values())
    git_push(f"tv_unique_survey: {total_done}/{done_so_far + len(REMAINING)} done, best={overall_best}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")

log(f"Survey complete! {len(REMAINING)} remaining pairs done.")
best_pair = min(results.items(), key=lambda x: x[1]['E'])
log(f"Best: {best_pair[0]} E={best_pair[1]['E']}")
