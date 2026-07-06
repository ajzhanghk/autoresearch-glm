#!/usr/bin/env python3
"""
Collect and canonically classify turn-square patterns by transversal count.

Pattern symmetries that induce isotopies of the turn-square:
  - permuting the 5 row-blocks (S5 on pattern rows)
  - permuting the 5 col-blocks (S5 on pattern cols)
  - transpose
Canonical form = lexicographic minimum over all 5! x 5! x 2 = 28800 images.

Massive random-restart hill-climb; every pattern with count >= THRESH is
canonicalized and checkpointed to a JSON set. The resulting class list is
the worklist for per-square ct>=10 decision runs (ct10_decide.py).

Usage: python3 turnsq_classes.py <instance>
"""
import json, sys, time, random
from datetime import datetime
from itertools import permutations
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = 10
LOG = REPO / f"mols10/results/tsclasses_{INSTANCE}.log"
DB = REPO / "mols10/results/turnsq_classes.json"   # shared, merge-on-write
THRESH = 5504

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

PERMS5 = list(permutations(range(5)))

def build_turn_square(pattern):
    L = np.array([[(i + j) % N for j in range(N)] for i in range(N)], dtype=np.int8)
    for a in range(5):
        for b in range(5):
            if pattern[a][b]:
                v = (a + b) % N; w = (a + b + 5) % N
                L[a, b] = w; L[a, b+5] = v
                L[a+5, b] = v; L[a+5, b+5] = w
    return L

def count_transversals(L, cap=6000):
    rows = [[int(L[r, c]) for c in range(N)] for r in range(N)]
    cnt = [0]
    def bt(row, cm, vm):
        if row == N:
            cnt[0] += 1
            return cnt[0] >= cap
        for col in range(N):
            if not (cm >> col & 1):
                v = rows[row][col]
                if not (vm >> v & 1):
                    if bt(row+1, cm | (1 << col), vm | (1 << v)):
                        return True
        return False
    bt(0, 0, 0)
    return cnt[0]

def canonical(pat):
    """Lexicographic min over S5 x S5 x transpose of the 25-bit string."""
    A = np.array(pat, dtype=np.int8)
    best = None
    for Mx in (A, A.T):
        for rp in PERMS5:
            R = Mx[list(rp), :]
            # For fixed row perm, best col perm = sort columns lexicographically
            # (valid because rows are already fixed: sorting col tuples gives lex-min)
            cols = sorted(tuple(R[:, c]) for c in range(5))
            key = tuple(v for col in cols for v in col)
            if best is None or key < best:
                best = key
    return best

def load_db():
    if DB.exists():
        try:
            d = json.loads(DB.read_text())
            return {tuple(k): v for k, v in zip([tuple(e['key']) for e in d], d)}
        except Exception:
            return {}
    return {}

def save_db(db):
    DB.write_text(json.dumps(list(db.values()), indent=1))

def main():
    log(f"Turn-square class collector — instance={INSTANCE}, THRESH={THRESH}")
    rng = random.Random(9973 * (INSTANCE + 1) + int(time.time()) % 100000)
    db = load_db()
    log(f"DB loaded: {len(db)} classes")

    t_start = time.time()
    n_climbs = 0
    n_hits = 0

    while True:
        # hill-climb from random pattern
        pat = [[rng.randint(0, 1) for _ in range(5)] for _ in range(5)]
        cur = count_transversals(build_turn_square(pat))
        for _ in range(300):
            a = rng.randrange(5); b = rng.randrange(5)
            pat[a][b] ^= 1
            new = count_transversals(build_turn_square(pat))
            if new >= cur:
                cur = new
            else:
                pat[a][b] ^= 1
        n_climbs += 1

        if cur >= THRESH:
            n_hits += 1
            key = canonical(pat)
            if key not in db:
                db = load_db()  # merge with other instance's writes
                if key not in db:
                    db[key] = {'key': list(key), 'count': cur,
                               'pattern': [list(r) for r in pat]}
                    save_db(db)
                    log(f"NEW CLASS #{len(db)}: count={cur} "
                        f"(climbs={n_climbs}, hits={n_hits})")

        if n_climbs % 20 == 0:
            el = time.time() - t_start
            log(f"climbs={n_climbs} hits={n_hits} classes={len(db)} "
                f"({n_climbs/max(el,1)*3600:.0f} climbs/hr)")

if __name__ == "__main__":
    main()
