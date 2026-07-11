#!/usr/bin/env python3
"""Process solved Myrvold cases: extract, verify, count transversals, save JSONs + INDEX.md.

Usage: save_pairs.py <squares_dir> <out_dir>
squares_dir contains files named squares-<CASE>.txt (decode.py output).
"""
import sys, os, json, glob

n = 10

def latin(N):
    return all(sorted(N[i]) == list(range(n)) for i in range(n)) and \
           all(sorted(N[i][j] for i in range(n)) == list(range(n)) for j in range(n))

def orthogonal(A, B):
    return len({(A[i][j], B[i][j]) for i in range(n) for j in range(n)}) == n*n

def transversal_rep(P, Q):
    for i in range(n):
        N = []
        for j in range(n):
            for k in range(n):
                if P[i][j] == Q[k][j]:
                    N.append(k)
        if sorted(N) != list(range(n)):
            return False
    return True

def derive_mate(P, Q):
    """M[k][j] = i where P[i][j] == Q[k][j].  Requires transversal_rep(P,Q); then M is Latin and orthogonal to Q."""
    M = [[None]*n for _ in range(n)]
    for j in range(n):
        col = {}  # symbol -> row of Q
        for k in range(n):
            col[Q[k][j]] = k
        for i in range(n):
            M[col[P[i][j]]][j] = i
    return M

def count_transversals(L):
    """Number of transversals of a single Latin square."""
    count = 0
    cols = [False]*n
    syms = [False]*n
    def dfs(r):
        nonlocal count
        if r == n:
            count += 1
            return
        for c in range(n):
            if not cols[c] and not syms[L[r][c]]:
                cols[c] = True; syms[L[r][c]] = True
                dfs(r+1)
                cols[c] = False; syms[L[r][c]] = False
    dfs(0)
    return count

def count_common_transversals(L, B):
    """Transversals of L whose B-values are also all distinct."""
    count = 0
    cols = [False]*n
    ls = [False]*n
    bs = [False]*n
    def dfs(r):
        nonlocal count
        if r == n:
            count += 1
            return
        for c in range(n):
            if not cols[c] and not ls[L[r][c]] and not bs[B[r][c]]:
                cols[c] = True; ls[L[r][c]] = True; bs[B[r][c]] = True
                dfs(r+1)
                cols[c] = False; ls[L[r][c]] = False; bs[B[r][c]] = False
    dfs(0)
    return count

def parse(path):
    blocks, cur = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            if cur:
                blocks.append(cur); cur = []
            continue
        cur.append(line.split())
    if cur:
        blocks.append(cur)
    assert len(blocks) == 4, f"{path}: expected 4 blocks, got {len(blocks)}"
    P = [[int(x) for x in row] for row in blocks[2]]
    Q = [[int(x) for x in row] for row in blocks[3]]
    return P, Q

def main(sq_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    k = 1
    seen = set()
    for path in sorted(glob.glob(os.path.join(sq_dir, "squares-*.txt"))):
        case = os.path.basename(path)[8:-4]
        P, Q = parse(path)
        assert latin(P) and latin(Q), f"{case}: not Latin"
        assert transversal_rep(P, Q) and transversal_rep(Q, P), f"{case}: not mutual TR"
        direct = orthogonal(P, Q)
        # Derived orthogonal pairs
        MP = derive_mate(P, Q)   # MP orthogonal to Q
        MQ = derive_mate(Q, P)   # MQ orthogonal to P
        cands = []
        if direct:
            cands.append((P, Q, f"(P,Q) directly orthogonal, types {case[0]}{case[1]}"))
        cands.append((Q, MP, f"L=Q (TR type {case[1]}), B=mate derived from rows of P (type {case[0]}) as transversals of Q"))
        cands.append((P, MQ, f"L=P (TR type {case[0]}), B=mate derived from rows of Q (type {case[1]}) as transversals of P"))
        for L, B, desc in cands:
            assert latin(L) and latin(B) and orthogonal(L, B), f"{case}: verification failed for {desc}"
            key = (tuple(map(tuple, L)), tuple(map(tuple, B)))
            key2 = (tuple(map(tuple, B)), tuple(map(tuple, L)))
            if key in seen or key2 in seen:
                continue
            seen.add(key)
            tL = count_transversals(L)
            tB = count_transversals(B)
            ct = count_common_transversals(L, B)
            fname = f"pair_{k}.json"
            json.dump({"L": L, "B": B}, open(os.path.join(out_dir, fname), "w"))
            entries.append({"file": fname, "case": case, "desc": desc,
                            "transversals_L": tL, "transversals_B": tB,
                            "common_transversals": ct,
                            "P_orth_Q_directly": direct})
            print(f"{fname}: case {case} | {desc} | tL={tL} tB={tB} common={ct}")
            k += 1
    return entries

if __name__ == "__main__":
    entries = main(sys.argv[1], sys.argv[2])
    json.dump(entries, open(os.path.join(sys.argv[2], "entries.json"), "w"), indent=1)
