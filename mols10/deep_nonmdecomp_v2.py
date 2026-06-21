#!/usr/bin/env python3
"""
Optimized deep SA for non-mdecomp pairs.
Uses precomputed IC list (rebuild every 200 steps) for ~8x speedup over random probe.
"""
import json, sys, time, random, math
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N

LOG       = REPO / "mols10/results/deep_nonmdecomp.log"
BEST_FILE = REPO / "mols10/results/deep_nonmdecomp_best.json"
BRANCH    = "claude/mols-order-10-search-yfQXK"

T_START = 10.0
T_END   = 0.03
N_STEPS = 30_000_000   # 30M steps per run (~17 min at 30k steps/s)
REBUILD_EVERY = 200

PAIR_IDS = [
    'iso2_tv21_0_0',  # survey E=32, deep best=32
    'iso2_tv21_0_1',  # survey E=32
    'iso2_tv21_0_5',  # survey E=32
    'iso_tv_10_2',    # survey E=32
    'tv_147',         # tv_unique_survey E=32 (NEW!)
    'iso_tv_21_1',    # survey E=33
    'iso_tv_21_2',    # survey E=33
    'iso_tv_10_1',    # survey E=33
    'iso2_tv21_0_3',  # survey E=34
    'iso2_tv21_0_4',  # survey E=34
    'iso_tv_10_0',    # survey E=34
    'iso2_tv21_0_2',  # survey E=35
    'tv_21',          # BFS: strict 6-IC local min at E=31; lowest priority
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    for f in [str(BEST_FILE.relative_to(REPO))]:
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


def probe_random_ic(L3, max_tries=400):
    for _ in range(max_tries):
        r1 = random.randint(0, N-1); r2 = random.randint(0, N-1)
        if r1 == r2: continue
        c1 = random.randint(0, N-1); c2 = random.randint(0, N-1)
        if c1 == c2: continue
        A = int(L3[r1, c1]); B = int(L3[r1, c2])
        if A != B and L3[r2, c1] == B and L3[r2, c2] == A:
            return r1, r2, c1, c2
    return None


# Load all pairs
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Load global best tracking
if BEST_FILE.exists():
    best_per_pair = json.loads(BEST_FILE.read_text())
else:
    best_per_pair = {}

# Load seeds
seeds = []
for fname in ['iso_tv21_0_best_l3.json', 'mdecomp113_best.json', 'mdecomp56_best.json']:
    fpath = REPO / "mols10/results" / fname
    if fpath.exists():
        d = json.loads(fpath.read_text())
        seeds.append((np.array(d['L3'], dtype=np.int8).reshape(N, N), d['E']))

# Load survey results
survey_results = {}
survey_file = REPO / "mols10/results/nonmdecomp_survey.json"
if survey_file.exists():
    survey_raw = json.loads(survey_file.read_text())
    for pid, entry in survey_raw.items():
        if entry.get('L3') is not None:
            survey_results[pid] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])
    if 'sa_ct_80' in survey_raw and survey_raw['sa_ct_80'].get('L3'):
        survey_results['tv_21'] = survey_results['sa_ct_80']

# Load tv_unique_survey results
tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
if tv_survey_file.exists():
    tv_survey_raw = json.loads(tv_survey_file.read_text())
    for pid, entry in tv_survey_raw.items():
        if entry.get('L3') is not None and pid not in survey_results:
            survey_results[pid] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])

log("=" * 70)
log(f"FAST Deep SA v2 (precomputed IC list, {N_STEPS//1_000_000}M steps)")
log(f"T: {T_START}→{T_END}, rebuild_every={REBUILD_EVERY}")
log(f"Seeds: {[s[1] for s in seeds]}")
log("=" * 70)

rng = random.Random(14142135)
run_global = 0
pair_run_count = {}

while True:
    for pair_id in PAIR_IDS:
        if pair_id not in pair_map:
            log(f"  {pair_id}: NOT IN PAIR MAP, skipping")
            continue

        p = pair_map[pair_id]
        L1 = np.array(p['L1'], dtype=np.int8).reshape(N, N)
        L2 = np.array(p['L2'], dtype=np.int8).reshape(N, N)

        if BEST_FILE.exists():
            best_per_pair = json.loads(BEST_FILE.read_text())

        if pair_id in best_per_pair:
            pair_best = best_per_pair[pair_id]['E']
            L3_pair_best = np.array(best_per_pair[pair_id]['L3'], dtype=np.int8).reshape(N, N)
        else:
            pair_best = 9999
            L3_pair_best = None

        # Also reload tv_unique_survey to get updated seeds
        if tv_survey_file.exists():
            tv_survey_raw = json.loads(tv_survey_file.read_text())
            for pid2, entry in tv_survey_raw.items():
                if entry.get('L3') is not None and pid2 not in survey_results:
                    survey_results[pid2] = (np.array(entry['L3'], dtype=np.int8).reshape(N, N), entry['E'])

        run_global += 1
        pair_run = pair_run_count.get(pair_id, 0)
        pair_run_count[pair_id] = pair_run + 1
        seed = rng.randint(1, 2**31 - 1)
        random.seed(seed)

        pair_seeds = []
        if pair_id in survey_results:
            pair_seeds.append(survey_results[pair_id])
        if L3_pair_best is not None:
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
                A = int(L3[r1, c1]); B = int(L3[r1, c2])
                L3[r1, c1] = B; L3[r1, c2] = A; L3[r2, c1] = A; L3[r2, c2] = B

        E_curr = count_clashes(L1, L3) + count_clashes(L2, L3)
        E_best_run = E_curr
        L3_best_run = L3.copy()

        t0 = time.time()
        accepted = tried = invalid = 0
        ics = build_ics(L3)

        for step in range(N_STEPS):
            if step > 0 and step % REBUILD_EVERY == 0:
                ics = build_ics(L3)

            if not ics:
                ic = probe_random_ic(L3)
                if ic is None: continue
            else:
                ic = ics[random.randint(0, len(ics) - 1)]
                r1, r2, c1, c2 = ic
                A = int(L3[r1, c1]); B = int(L3[r1, c2])
                if not (A != B and L3[r2, c1] == B and L3[r2, c2] == A):
                    invalid += 1
                    ic = probe_random_ic(L3)
                    if ic is None: continue

            tried += 1
            r1, r2, c1, c2 = ic
            Lp = L3.copy()
            a, b = int(L3[r1, c1]), int(L3[r1, c2])
            Lp[r1, c1] = b; Lp[r1, c2] = a; Lp[r2, c1] = a; Lp[r2, c2] = b
            T = T_START * (T_END / T_START) ** (step / N_STEPS)
            Ep = count_clashes(L1, Lp) + count_clashes(L2, Lp)
            dE = Ep - E_curr

            if dE < 0 or random.random() < math.exp(-dE / T):
                L3 = Lp; E_curr = Ep; accepted += 1
                if E_curr < E_best_run:
                    E_best_run = E_curr; L3_best_run = L3.copy()
                    log(f"  {pair_id}: new best E={E_best_run} cl13={count_clashes(L1,L3)} cl23={count_clashes(L2,L3)}")
                    if E_best_run < pair_best:
                        pair_best = E_best_run
                        best_per_pair[pair_id] = {
                            'E': int(pair_best),
                            'cl13': int(count_clashes(L1, L3_best_run)),
                            'cl23': int(count_clashes(L2, L3_best_run)),
                            'L3': L3_best_run.tolist()
                        }
                        BEST_FILE.write_text(json.dumps(best_per_pair, indent=2))
                        git_push(f"deep_nonmdecomp_v2: {pair_id} E={pair_best}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
                        if pair_best == 0:
                            log("*** 3-MOLS FOUND! ***"); sys.exit(0)

        elapsed = time.time() - t0
        log(f"run={run_global:4d} {pair_id}: perturb={n_perturb} best_run={E_best_run} "
            f"pair_best={pair_best} acc={accepted/max(tried,1)*100:.0f}% "
            f"invalid%={invalid/max(step,1)*100:.1f}% "
            f"{N_STEPS/elapsed:.0f}steps/s t={elapsed:.0f}s")
