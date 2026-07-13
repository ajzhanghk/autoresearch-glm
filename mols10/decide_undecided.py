#!/usr/bin/env python3
"""
Deep re-attack on the UNDECIDED (UNKNOWN) rich trivial-autotopism squares:
the only territory where a 3-MOLS(10) could still hide. Each is the richest
MMM-eligible material (~1800 transversals) that the fast triple-decision
could not resolve within its short cap. Here we give each a long CP-SAT
budget (and more workers). SAT => the big fish; INFEASIBLE => another
brick; still UNKNOWN => escalate further.

Usage: python3 decide_undecided.py [cap_s]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from triple_decide_asym import decide_triple
from mols_cpsat_worker import count_clashes, N

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
LOG = REPO / "mols10/results/decide_undecided.log"
UDIR = REPO / "mols10/results/undecided"
DONE = REPO / "mols10/results/undecided_verdicts.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    subprocess.run(["git","-C",str(REPO),"add","mols10/results/"], check=False)
    r = subprocess.run(["git","-C",str(REPO),"commit","-m",msg],
                       capture_output=True, check=False)
    if r.returncode == 0:
        subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH],
                       capture_output=True, check=False)
        subprocess.run(["git","-C",str(REPO),"push","origin",BRANCH],
                       capture_output=True, check=False)

def main():
    log("=" * 70)
    log(f"DEEP RE-ATTACK on undecided rich squares, cap={CAP}s each")
    verdicts = {}
    if DONE.exists():
        try: verdicts = json.loads(DONE.read_text())
        except Exception: verdicts = {}
    while True:
        files = sorted(UDIR.glob("u_*.json")) if UDIR.exists() else []
        pending = [f for f in files if f.name not in verdicts]
        if not pending:
            log(f"no pending undecided squares; sleeping "
                f"({len(verdicts)} resolved)")
            time.sleep(120)
            continue
        # richest first (most fish potential)
        pending.sort(key=lambda f: -int(f.stem.split("_")[-1]))
        f = pending[0]
        L = np.array(json.loads(f.read_text())['L'], dtype=np.int8)
        M = int(f.stem.split("_")[-1])
        log(f"attacking {f.name} ({M} transversals), cap={CAP}s...")
        t0 = time.time()
        name, MM, B, C = decide_triple(L, CAP)
        el = time.time() - t0
        verdicts[f.name] = {'status': name, 'n_trans': M, 'elapsed': round(el)}
        DONE.write_text(json.dumps(verdicts, indent=1))
        log(f"  {f.name}: {name} in {el:.0f}s")
        if name in ('OPTIMAL', 'FEASIBLE') and B is not None:
            cl12 = count_clashes(L, B); cl13 = count_clashes(L, C); cl23 = count_clashes(B, C)
            log(f"*** SAT! cl={cl12},{cl13},{cl23} ***")
            if cl12 == cl13 == cl23 == 0:
                log("*** *** 3-MOLS(10) FOUND — THE BIG FISH *** ***")
                FOUND.write_text(json.dumps({
                    'found': True, 'method': 'decide_undecided',
                    'L1': L.tolist(), 'L2': B.tolist(), 'L3': C.tolist(),
                    'cl12': int(cl12), 'cl13': int(cl13), 'cl23': int(cl23)}, indent=1))
                git_push("*** 3-MOLS(10) FOUND via decide_undecided! ***")
                return
        git_push(f"deep re-attack: {f.name} -> {name} "
                 f"({sum(1 for v in verdicts.values() if v['status']=='INFEASIBLE')} INF, "
                 f"{sum(1 for v in verdicts.values() if v['status'] not in ('INFEASIBLE','OPTIMAL','FEASIBLE'))} still unknown)")

if __name__ == "__main__":
    main()
