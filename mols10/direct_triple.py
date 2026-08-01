#!/usr/bin/env python3
"""
DIRECT counterexample search for 3-MOLS(10): solve for three mutually
orthogonal 10x10 Latin squares L1,L2,L3 as ONE CP-SAT instance. Any SAT
solution IS a 3-MOLS(10) -> N(10) >= 3 (immediately verified). This is the
only COMPLETE method (no expensive per-square max-CT proxy): it searches the
whole triple at once.

Symmetry breaking (sound: a triple exists iff one in this canonical form does):
  * L1 reduced: L1[0][c]=c, L1[r][0]=r.
  * L2,L3 first row = identity (0..9); relabel symbols so a 3-MOLS in
    "standard form" has L2[0]=L3[0]=(0..9) and L1[0]=(0..9).
  * order the pair (L2,L3): L2[1][0] < L3[1][0] (swap L2<->L3 symmetry).
  * L2[1][0], L3[1][0] != 0,1 (derangement of first column below row 0).

Orthogonality via AllDifferent on the 100 encoded pairs for each of the
three pairings (L1,L2),(L1,L3),(L2,L3).

Las Vegas: fresh random seed per relaunch (the environment kills long runs);
each attempt explores a different trajectory. Restart-proof by design (a
single SAT instance; on SAT it writes MOLS10_FOUND and pushes).

Usage: python3 direct_triple.py [timeout_s] [seed]
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else int(time.time()) % 99991
LOG = REPO / "mols10/results/direct_triple.log"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(m):
    line=f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line,flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def git_push(msg):
    subprocess.run(["git","-C",str(REPO),"add","mols10/results/"],check=False)
    r=subprocess.run(["git","-C",str(REPO),"commit","-m",msg],capture_output=True)
    if r.returncode==0:
        subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH],capture_output=True)
        subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH],capture_output=True)

def main():
    log("="*66)
    log(f"DIRECT 3-MOLS(10) counterexample search — timeout={TIMEOUT}s seed={SEED}")
    m=cp_model.CpModel()
    # L[s][r][c] in 0..9 for s in {0,1,2}
    L=[[[m.new_int_var(0,N-1,f"L{s}_{r}_{c}") for c in range(N)]
        for r in range(N)] for s in range(3)]
    for s in range(3):
        for r in range(N): m.add_all_different(L[s][r])
        for c in range(N): m.add_all_different([L[s][r][c] for r in range(N)])
    # symmetry breaking
    for c in range(N):
        m.add(L[0][0][c]==c)          # L1 reduced first row
        m.add(L[1][0][c]==c)          # L2 first row identity
        m.add(L[2][0][c]==c)          # L3 first row identity
    for r in range(N):
        m.add(L[0][r][0]==r)          # L1 reduced first column
    m.add(L[1][1][0] < L[2][1][0])    # order L2,L3
    # orthogonality: encode pair (a,b) -> a*N+b must be all-different over cells
    def orth(sa, sb):
        codes=[]
        for r in range(N):
            for c in range(N):
                v=m.new_int_var(0,N*N-1,f"p{sa}{sb}_{r}_{c}")
                m.add(v==L[sa][r][c]*N+L[sb][r][c])
                codes.append(v)
        m.add_all_different(codes)
    orth(0,1); orth(0,2); orth(1,2)

    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=TIMEOUT
    solver.parameters.num_workers=4
    solver.parameters.random_seed=SEED
    solver.parameters.log_search_progress=False
    t0=time.time()
    st=solver.solve(m)
    el=time.time()-t0
    log(f"STATUS: {solver.status_name(st)} in {el:.0f}s")
    if st in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        sq=[np.array([[solver.value(L[s][r][c]) for c in range(N)]
                      for r in range(N)],dtype=np.int8) for s in range(3)]
        c12=count_clashes(sq[0],sq[1]); c13=count_clashes(sq[0],sq[2])
        c23=count_clashes(sq[1],sq[2])
        log(f"VERIFY clashes: (L1,L2)={c12} (L1,L3)={c13} (L2,L3)={c23}")
        if c12==0 and c13==0 and c23==0:
            log("*** 3-MOLS(10) FOUND — N(10) >= 3 COUNTEREXAMPLE! ***")
            FOUND.write_text(json.dumps({'found':True,'method':'direct_triple',
                'L1':sq[0].tolist(),'L2':sq[1].tolist(),'L3':sq[2].tolist()},indent=2))
            git_push("3-MOLS(10) COUNTEREXAMPLE FOUND via direct CP-SAT triple search!")
        else:
            log("!!! verify failed — model bug, investigate !!!")
    elif st==cp_model.INFEASIBLE:
        log("INFEASIBLE: no 3-MOLS(10) in this canonical form -> N(10)=2 PROVEN")
        (REPO/"mols10/results/direct_triple_UNSAT.json").write_text(
            json.dumps({'status':'INFEASIBLE','elapsed_s':el},indent=1))
        git_push("DIRECT SEARCH INFEASIBLE: N(10)=2 proven (canonical 3-MOLS SAT)")
    else:
        log("UNKNOWN (timeout) — relaunch with fresh seed")

if __name__=="__main__":
    main()
