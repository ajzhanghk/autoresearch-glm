#!/usr/bin/env python3
"""
Fast isotopy search using numba-JIT SA (360k+ steps/s).
21x speedup over the probe_random_ic approach.

Two-phase:
  Phase 1: 4 seeds × 5M steps (54s) per isotopy — quick filter
  Phase 2: 8 seeds × 30M steps (~660s) — if phase1 E <= DEEP_THRESH
  Phase 3: 8 seeds × 150M steps (~3300s) — if phase2 E <= ULTRA_THRESH

All promising isotopies (E <= SAVE_THRESH) saved to file.
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

INSTANCE    = sys.argv[1] if len(sys.argv) > 1 else "0"
LOG         = REPO / f"mols10/results/fast_iso_{INSTANCE}.log"
BEST_FILE   = REPO / "mols10/results/fast_iso_best.json"
PROMISING_F = REPO / "mols10/results/fast_iso_promising.json"
BRANCH      = "claude/mols-order-10-search-yfQXK"

N_SEEDS_P1      = 4          # seeds for phase 1
N_STEPS_P1      = 5_000_000  # steps per seed, phase 1
N_SEEDS_P2      = 8          # seeds for phase 2
N_STEPS_P2      = 30_000_000 # steps per seed, phase 2
N_SEEDS_P3      = 8          # seeds for phase 3 (ultra-deep)
N_STEPS_P3      = 150_000_000
DEEP_THRESH     = 30         # trigger phase2 if phase1 E <= this
ULTRA_THRESH    = 29         # trigger phase3 if phase2 E <= this
SAVE_THRESH     = 31         # save isotopy if E <= this
GOAL_E          = 26         # must beat current global best

T_START = 8.0
T_END   = 0.03

TARGET_PAIRS = [
    'tv_222', 'tv_223', 'tv_254', 'tv_270',
    'iso_tv_21_0',
    'iso2_tv21_0_1', 'iso2_tv21_0_0', 'iso2_tv21_0_5',
    'iso_tv_10_2', 'iso_tv_10_1', 'iso_tv_10_0',
    'tv_147', 'tv_225', 'tv_238',
    'iso_tv_21_1', 'iso_tv_21_2',
    'iso2_tv21_0_3', 'iso2_tv21_0_4',
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def git_push(msg):
    import subprocess
    for f in [str(BEST_FILE.relative_to(REPO)), str(PROMISING_F.relative_to(REPO))]:
        subprocess.run(["git", "-C", str(REPO), "add", f], check=False)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg], check=False)
    subprocess.run(["git", "-C", str(REPO), "push", "-u", "origin", BRANCH], check=False)


def apply_isotopy(L, rp, cp, sp):
    L2 = np.empty((N, N), dtype=np.int8)
    for r in range(N):
        for c in range(N):
            L2[rp[r], cp[c]] = sp[int(L[r, c])]
    return L2


def run_phase(L1, L2, seeds_L3, n_seeds, n_steps, rng_base, label):
    """Run SA on multiple seeds, return best (E, L3)."""
    best_E = 9999; best_L3 = None
    L1f = L1.ravel().astype(np.int32)
    L2f = L2.ravel().astype(np.int32)
    for i, L3_seed in enumerate(seeds_L3[:n_seeds]):
        seed = int(rng_base ^ (i * 1234567))
        if seed <= 0: seed = 1
        L3f = L3_seed.ravel().astype(np.int32).copy()
        pc1 = make_pc(L1, L3_seed)
        pc2 = make_pc(L2, L3_seed)
        E, L3_flat = _run_sa_core(L1f, L2f, L3f, pc1, pc2,
                                   n_steps, T_START, T_END, seed)
        if E < best_E:
            best_E = E
            best_L3 = L3_flat.astype(np.int8).reshape(N, N)
    return best_E, best_L3


# Load pair data
all_pairs = json.loads((REPO / "mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']: p for p in all_pairs}

# Build seed list (all known good L3s)
seeds_L3 = []
seed_labels = []

for fname, label in [
    ('iso_tv21_0_best_l3.json', 'iso0_E26'),
    ('mdecomp113_best.json', 'md113_E28'),
    ('mdecomp56_best.json', 'md56_E30'),
]:
    fpath = REPO / "mols10/results" / fname
    if fpath.exists():
        d = json.loads(fpath.read_text())
        if 'L3' in d:
            seeds_L3.append(np.array(d['L3'], dtype=np.int8).reshape(N, N))
            seed_labels.append(label)

# New: tv_222 E=30 seed from e31_deep_sa
tv222_seed_file = REPO / "mols10/results/tv222_e30_seed.json"
if tv222_seed_file.exists():
    d = json.loads(tv222_seed_file.read_text())
    seeds_L3.append(np.array(d['L3'], dtype=np.int8).reshape(N, N))
    seed_labels.append(f'tv222_E{d["E"]}')

# Survey seeds (E=31 pairs)
tv_survey_file = REPO / "mols10/results/tv_unique_survey.json"
if tv_survey_file.exists():
    tv_survey = json.loads(tv_survey_file.read_text())
    for pid in ['tv_222', 'tv_223', 'tv_254', 'tv_270', 'tv_147', 'tv_225', 'tv_238']:
        if pid in tv_survey and tv_survey[pid].get('L3'):
            seeds_L3.append(np.array(tv_survey[pid]['L3'], dtype=np.int8).reshape(N, N))
            seed_labels.append(f'{pid}_E{tv_survey[pid]["E"]}')

# Load existing results
best_data = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
promising = json.loads(PROMISING_F.read_text()) if PROMISING_F.exists() else []
global_best_E = min(best_data.get('global_best_E', GOAL_E), GOAL_E)

log("=" * 70)
log(f"Fast isotopy search (numba) instance={INSTANCE}")
log(f"Phase1: {N_SEEDS_P1}×{N_STEPS_P1//1_000_000}M, Phase2: {N_SEEDS_P2}×{N_STEPS_P2//1_000_000}M")
log(f"Phase3 (if E<={ULTRA_THRESH}): {N_SEEDS_P3}×{N_STEPS_P3//1_000_000}M")
log(f"Deep thresh: {DEEP_THRESH}, Ultra thresh: {ULTRA_THRESH}")
log(f"Seeds ({len(seeds_L3)}): {seed_labels}")
log(f"Pairs ({len(TARGET_PAIRS)}): {TARGET_PAIRS[:8]}...")
log("=" * 70)

# Stagger instance start by pair offset
instance_offset = int(INSTANCE) * 3
rng = random.Random(31415926 + int(INSTANCE) * 997)
iso_num = best_data.get(f'isotopies_inst{INSTANCE}', 0)
pair_cycle_idx = (iso_num + instance_offset) % len(TARGET_PAIRS)

while True:
    pair_id = TARGET_PAIRS[pair_cycle_idx % len(TARGET_PAIRS)]
    pair_cycle_idx += 1
    iso_num += 1

    if pair_id not in pair_map:
        log(f"  {pair_id}: not in pair_map, skipping")
        continue

    p = pair_map[pair_id]
    L1_base = np.array(p['L1'], dtype=np.int8).reshape(N, N)
    L2_base = np.array(p['L2'], dtype=np.int8).reshape(N, N)

    # Random independent isotopy of (L1, L2)
    rp1 = list(range(N)); rng.shuffle(rp1)
    cp1 = list(range(N)); rng.shuffle(cp1)
    sp1 = list(range(N)); rng.shuffle(sp1)
    rp2 = list(range(N)); rng.shuffle(rp2)
    cp2 = list(range(N)); rng.shuffle(cp2)
    sp2 = list(range(N)); rng.shuffle(sp2)

    L1_iso = apply_isotopy(L1_base, rp1, cp1, sp1)
    L2_iso = apply_isotopy(L2_base, rp2, cp2, sp2)

    rng_base = rng.randint(1, 2**31 - 1)
    t_start = time.time()

    # --- Phase 1 ---
    E1, L3_p1 = run_phase(L1_iso, L2_iso, seeds_L3, N_SEEDS_P1, N_STEPS_P1, rng_base, "P1")
    t_p1 = time.time() - t_start

    if E1 > DEEP_THRESH:
        log(f"ISO {iso_num} ({pair_id}): P1 E={E1} t={t_p1:.0f}s (skip)")
        if iso_num % 20 == 0:
            best_data[f'isotopies_inst{INSTANCE}'] = iso_num
            BEST_FILE.write_text(json.dumps(best_data, indent=2))
        continue

    # Phase 1 promising: run Phase 2
    log(f"ISO {iso_num} ({pair_id}): P1 E={E1} t={t_p1:.0f}s → Phase2...")
    rng_base2 = rng.randint(1, 2**31 - 1)
    E2, L3_p2 = run_phase(L1_iso, L2_iso, seeds_L3, N_SEEDS_P2, N_STEPS_P2, rng_base2, "P2")
    t_p2 = time.time() - t_start
    best_E = min(E1, E2)
    best_L3 = L3_p2 if E2 <= E1 else L3_p1
    log(f"ISO {iso_num} ({pair_id}): P2 E={E2} best={best_E} t={t_p2:.0f}s")

    if best_E <= ULTRA_THRESH:
        log(f"ISO {iso_num} ({pair_id}): best={best_E}<={ULTRA_THRESH} → ULTRA-DEEP Phase3...")
        rng_base3 = rng.randint(1, 2**31 - 1)
        # Start Phase3 from the Phase2 best L3 + global seeds
        p3_seeds = [best_L3] + seeds_L3[:N_SEEDS_P3 - 1]
        E3, L3_p3 = run_phase(L1_iso, L2_iso, p3_seeds, N_SEEDS_P3, N_STEPS_P3, rng_base3, "P3")
        t_p3 = time.time() - t_start
        log(f"ISO {iso_num} ({pair_id}): P3 E={E3} t={t_p3:.0f}s")
        if E3 < best_E:
            best_E = E3; best_L3 = L3_p3

    # Save if promising (read-merge-write to avoid race condition)
    if best_E <= SAVE_THRESH:
        cl13 = int(count_clashes(L1_iso, best_L3))
        cl23 = int(count_clashes(L2_iso, best_L3))
        entry = {
            'inst': INSTANCE, 'iso_num': iso_num, 'pair_id': pair_id,
            'E': int(best_E), 'cl13': cl13, 'cl23': cl23,
            'L1_iso': L1_iso.tolist(), 'L2_iso': L2_iso.tolist(),
            'L3': best_L3.tolist(),
        }
        # Re-read from disk to merge entries from other instances
        on_disk = json.loads(PROMISING_F.read_text()) if PROMISING_F.exists() else []
        # Merge: add our entry if not already present
        key = (INSTANCE, iso_num)
        if not any(e.get('inst') == INSTANCE and e.get('iso_num') == iso_num for e in on_disk):
            on_disk.append(entry)
        # Also include our in-memory entries not yet on disk
        for e in promising:
            if not any(ex.get('inst') == e.get('inst') and ex.get('iso_num') == e.get('iso_num') for ex in on_disk):
                on_disk.append(e)
        on_disk.sort(key=lambda x: x['E'])
        promising = on_disk[:100]
        PROMISING_F.write_text(json.dumps(promising, indent=2))
        log(f"  *** Saved: {pair_id} E={best_E} cl13={cl13} cl23={cl23} (saved: {len(promising)})")

    if best_E < global_best_E:
        global_best_E = best_E
        cl13 = int(count_clashes(L1_iso, best_L3))
        cl23 = int(count_clashes(L2_iso, best_L3))
        log(f"*** ISO {iso_num} ({pair_id}): NEW GLOBAL BEST E={global_best_E} cl13={cl13} cl23={cl23} ***")
        best_data.update({
            'global_best_E': int(global_best_E),
            'best_pair': pair_id, 'best_iso': iso_num,
            'L1': L1_iso.tolist(), 'L2': L2_iso.tolist(),
            'L3': best_L3.tolist(), 'cl13': cl13, 'cl23': cl23,
        })
        BEST_FILE.write_text(json.dumps(best_data, indent=2))
        git_push(f"fast_iso: {pair_id} E={global_best_E} (new global best!)\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if global_best_E == 0:
            log("*** 3-MOLS FOUND! ***"); sys.exit(0)

    if iso_num % 20 == 0:
        best_data[f'isotopies_inst{INSTANCE}'] = iso_num
        BEST_FILE.write_text(json.dumps(best_data, indent=2))
