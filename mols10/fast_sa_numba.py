#!/usr/bin/env python3
"""
Fast SA using numba JIT + incremental energy updates.
Maintains pair-count arrays so each SA step is O(1) instead of O(N^2).

Speedup target: ~100x over current count_clashes approach.
"""
import numpy as np
from numba import njit
import math, random, time

N = 10

@njit(cache=True)
def _run_sa_core(L1f, L2f, L3f, pc1, pc2, n_steps, T_start, T_end, seed):
    """
    numba-accelerated SA inner loop with incremental energy updates.
    L1f, L2f, L3f: flat int32 arrays of size N*N
    pc1, pc2: pair count arrays, pc[i*N+j] = number of cells where L1=i, L3=j
    Returns: (E_best, L3_best_flat)
    """
    NN = 10  # N, literal for numba

    # Initial energy (count missing pairs = 0 entries in pc)
    E1 = 0
    for i in range(NN*NN):
        if pc1[i] == 0:
            E1 += 1
    E2 = 0
    for i in range(NN*NN):
        if pc2[i] == 0:
            E2 += 1
    E = E1 + E2
    E_best = E
    L3_best = L3f.copy()

    # Park-Miller LCG
    rng = seed & 0x7fffffff
    if rng == 0:
        rng = 1

    scratch = np.zeros(NN*NN, dtype=np.int32)

    for step in range(n_steps):
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)

        # Find valid IC (random probe, up to 400 tries)
        found = 0
        r1 = 0; r2 = 0; c1 = 0; c2 = 0; a = 0; b = 0
        for _ in range(400):
            rng = (16807 * rng) % 2147483647
            r1 = rng % NN
            rng = (16807 * rng) % 2147483647
            r2 = rng % NN
            if r1 == r2:
                continue
            rng = (16807 * rng) % 2147483647
            c1 = rng % NN
            rng = (16807 * rng) % 2147483647
            c2 = rng % NN
            if c1 == c2:
                continue
            a = L3f[r1*NN+c1]
            b = L3f[r1*NN+c2]
            if a != b and L3f[r2*NN+c1] == b and L3f[r2*NN+c2] == a:
                found = 1
                break

        if not found:
            continue

        # L1 values at the 4 changed cells
        l1_r1c1 = L1f[r1*NN+c1]; l1_r1c2 = L1f[r1*NN+c2]
        l1_r2c1 = L1f[r2*NN+c1]; l1_r2c2 = L1f[r2*NN+c2]
        # L2 values
        l2_r1c1 = L2f[r1*NN+c1]; l2_r1c2 = L2f[r1*NN+c2]
        l2_r2c1 = L2f[r2*NN+c1]; l2_r2c2 = L2f[r2*NN+c2]

        # --- Compute dE1 using scratch array for pc1 ---
        # IC move: (r1,c1) a→b, (r1,c2) b→a, (r2,c1) b→a, (r2,c2) a→b
        # Pairs removed from pc1: (l1_r1c1,a), (l1_r1c2,b), (l1_r2c1,b), (l1_r2c2,a)
        # Pairs added to pc1:     (l1_r1c1,b), (l1_r1c2,a), (l1_r2c1,a), (l1_r2c2,b)

        scratch[l1_r1c1*NN+a] -= 1; scratch[l1_r1c2*NN+b] -= 1
        scratch[l1_r2c1*NN+b] -= 1; scratch[l1_r2c2*NN+a] -= 1
        scratch[l1_r1c1*NN+b] += 1; scratch[l1_r1c2*NN+a] += 1
        scratch[l1_r2c1*NN+a] += 1; scratch[l1_r2c2*NN+b] += 1

        dE1 = 0
        # Process the 8 potentially-changed pair indices for pc1
        # (pairs with net zero change have scratch=0 and are skipped)
        p = l1_r1c1*NN+a; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r1c2*NN+b; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r2c1*NN+b; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r2c2*NN+a; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r1c1*NN+b; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r1c2*NN+a; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r2c1*NN+a; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0
        p = l1_r2c2*NN+b; d = scratch[p]
        if d != 0:
            oc = pc1[p]; nc = oc + d
            if oc == 0 and nc > 0: dE1 -= 1
            elif oc > 0 and nc == 0: dE1 += 1
            scratch[p] = 0

        # --- Compute dE2 for pc2 ---
        scratch[l2_r1c1*NN+a] -= 1; scratch[l2_r1c2*NN+b] -= 1
        scratch[l2_r2c1*NN+b] -= 1; scratch[l2_r2c2*NN+a] -= 1
        scratch[l2_r1c1*NN+b] += 1; scratch[l2_r1c2*NN+a] += 1
        scratch[l2_r2c1*NN+a] += 1; scratch[l2_r2c2*NN+b] += 1

        dE2 = 0
        p = l2_r1c1*NN+a; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r1c2*NN+b; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r2c1*NN+b; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r2c2*NN+a; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r1c1*NN+b; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r1c2*NN+a; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r2c1*NN+a; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0
        p = l2_r2c2*NN+b; d = scratch[p]
        if d != 0:
            oc = pc2[p]; nc = oc + d
            if oc == 0 and nc > 0: dE2 -= 1
            elif oc > 0 and nc == 0: dE2 += 1
            scratch[p] = 0

        dE = dE1 + dE2

        # SA acceptance criterion
        accept = False
        if dE < 0:
            accept = True
        elif T > 1e-10:
            rng = (16807 * rng) % 2147483647
            u = rng / 2147483647.0
            if u < math.exp(-dE / T):
                accept = True

        if accept:
            # Apply IC move to L3
            L3f[r1*NN+c1] = b; L3f[r1*NN+c2] = a
            L3f[r2*NN+c1] = a; L3f[r2*NN+c2] = b

            # Update pc1 (net change same as scratch fills above)
            pc1[l1_r1c1*NN+a] -= 1; pc1[l1_r1c2*NN+b] -= 1
            pc1[l1_r2c1*NN+b] -= 1; pc1[l1_r2c2*NN+a] -= 1
            pc1[l1_r1c1*NN+b] += 1; pc1[l1_r1c2*NN+a] += 1
            pc1[l1_r2c1*NN+a] += 1; pc1[l1_r2c2*NN+b] += 1

            # Update pc2
            pc2[l2_r1c1*NN+a] -= 1; pc2[l2_r1c2*NN+b] -= 1
            pc2[l2_r2c1*NN+b] -= 1; pc2[l2_r2c2*NN+a] -= 1
            pc2[l2_r1c1*NN+b] += 1; pc2[l2_r1c2*NN+a] += 1
            pc2[l2_r2c1*NN+a] += 1; pc2[l2_r2c2*NN+b] += 1

            E += dE
            if E < E_best:
                E_best = E
                for i in range(NN*NN):
                    L3_best[i] = L3f[i]

    return E_best, L3_best


def make_pc(L1, L3, n=N):
    """Build pair-count array pc[i*n+j] = count of cells where L1=i, L3=j."""
    pc = np.zeros(n*n, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            pc[int(L1[r,c])*n + int(L3[r,c])] += 1
    return pc


def run_sa_fast(L1, L2, L3_init, n_steps, seed, T_start=8.0, T_end=0.03):
    """
    Run fast numba SA.
    L1, L2, L3_init: (N,N) int8 numpy arrays
    Returns: (E_best, L3_best as (N,N) int8 array)
    """
    L1f = L1.ravel().astype(np.int32)
    L2f = L2.ravel().astype(np.int32)
    L3f = L3_init.ravel().astype(np.int32).copy()
    pc1 = make_pc(L1, L3_init)
    pc2 = make_pc(L2, L3_init)
    E_best, L3_best_flat = _run_sa_core(L1f, L2f, L3f, pc1, pc2, n_steps,
                                         float(T_start), float(T_end), int(seed))
    return int(E_best), L3_best_flat.astype(np.int8).reshape(N, N)


if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    REPO = Path("/home/user/autoresearch-glm")
    sys.path.insert(0, str(REPO / "mols10"))
    from mols_cpsat_worker import count_clashes

    print("Warming up numba JIT (compiling)...", flush=True)
    # Trigger compilation with a small test
    L_test = np.arange(N*N, dtype=np.int8).reshape(N, N) % N
    t0 = time.time()
    E, _ = run_sa_fast(L_test, L_test, L_test, 100, 42)
    print(f"JIT compile done in {time.time()-t0:.1f}s")

    # Benchmark on tv_222
    print("\nBenchmark: 1M steps on tv_222...", flush=True)
    survey_file = REPO / "mols10/results/tv_unique_survey.json"
    pairs_file = REPO / "mols10/results/promising_pairs.json"

    all_pairs = json.loads(pairs_file.read_text())
    pair_map = {p['pair_id']: p for p in all_pairs}
    tv_survey = json.loads(survey_file.read_text())

    p = pair_map['tv_222']
    L1 = np.array(p['L1'], dtype=np.int8).reshape(N, N)
    L2 = np.array(p['L2'], dtype=np.int8).reshape(N, N)
    L3_seed = np.array(tv_survey['tv_222']['L3'], dtype=np.int8).reshape(N, N)

    t0 = time.time()
    E, L3 = run_sa_fast(L1, L2, L3_seed, 1_000_000, 12345)
    elapsed = time.time() - t0
    cl13 = count_clashes(L1, L3)
    cl23 = count_clashes(L2, L3)
    print(f"1M steps in {elapsed:.2f}s = {1e6/elapsed:.0f} steps/s")
    print(f"Result: E={E} (cl13={cl13}, cl23={cl23})")
    print(f"Expected E from count_clashes: {cl13+cl23}")

    # Longer run
    print("\nRunning 10M steps...", flush=True)
    t0 = time.time()
    E, L3 = run_sa_fast(L1, L2, L3_seed, 10_000_000, 99999)
    elapsed = time.time() - t0
    cl13 = count_clashes(L1, L3)
    cl23 = count_clashes(L2, L3)
    print(f"10M steps in {elapsed:.2f}s = {10e6/elapsed:.0f} steps/s")
    print(f"Result: E={E} (cl13={cl13}, cl23={cl23})")
