#!/usr/bin/env python3
"""
Standalone verification of saved MOLS(10) pairs.

Usage:
    python verify_results.py                        # verify all in results/
    python verify_results.py results/mols_pair_seed42.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def is_latin_square(L: list[list[int]], n: int) -> bool:
    arr = np.array(L, dtype=np.int8)
    for i in range(n):
        if set(arr[i, :]) != set(range(n)):
            return False
        if set(arr[:, i]) != set(range(n)):
            return False
    return True


def are_orthogonal(L1: list[list[int]], L2: list[list[int]], n: int) -> bool:
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True


def verify_file(path: Path) -> bool:
    data = json.loads(path.read_text())
    squares = []
    # Support single pair {L1, L2} or collection [{seed, L1, L2}, ...]
    if isinstance(data, list):
        entries = data
    else:
        entries = [data]

    all_ok = True
    for idx, entry in enumerate(entries):
        label = f"entry[{idx}]" if isinstance(data, list) else path.name
        if isinstance(data, list):
            seed = entry.get("seed", "?")
            label = f"seed={seed}"

        L1 = entry["L1"]
        L2 = entry["L2"]
        n = len(L1)

        ok_L1 = is_latin_square(L1, n)
        ok_L2 = is_latin_square(L2, n)
        ok_orth = are_orthogonal(L1, L2, n)

        status = "OK" if (ok_L1 and ok_L2 and ok_orth) else "FAIL"
        if not (ok_L1 and ok_L2 and ok_orth):
            all_ok = False
        print(f"  {label}: L1_latin={ok_L1}  L2_latin={ok_L2}  orthogonal={ok_orth}  [{status}]")

    return all_ok


def main():
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        results_dir = Path(__file__).parent / "results"
        files = sorted(results_dir.glob("*.json"))

    if not files:
        print("No JSON files found.")
        sys.exit(1)

    print(f"Verifying {len(files)} file(s)...\n")
    overall_ok = True
    for f in files:
        print(f"{f.name}:")
        ok = verify_file(f)
        if not ok:
            overall_ok = False
        print()

    if overall_ok:
        print("All pairs verified ✓")
    else:
        print("Some pairs FAILED verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
