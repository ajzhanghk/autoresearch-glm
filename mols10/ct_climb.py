#!/usr/bin/env python3
"""
COUNTEREXAMPLE HUNT for 3-MOLS(10).

Target the ONLY region a counterexample can live (per this project's proven
map + McKay-Meynert-Myrvold): a Latin square L with
  (a) TRIVIAL autoparatopism group  (MMM: required of every square in a triple)
  (b) max-CT(L) >= 10   (10 common transversals of some mate B that themselves
                         decompose into a third square C -> (L,B,C) is 3-MOLS)

We have NO known trivial-symmetry square with max-CT even >= 4 (catalog<=2,
one-swap near-turn=3). So we HUNT: simulated annealing directly in L-space,
objective = ct(L) := best common-transversal count over streamed orthogonal
mates, moving by Latin-preserving intercalate flips, REJECTING states with
any nontrivial translation symmetry (to stay in the MMM-eligible region).

On ct(L) >= 10: verify, then try to exact-cover the >=10 common transversals
into a third square C; if cl(L,B)=cl(L,C)=cl(B,C)=0  ->  3-MOLS(10) FOUND.

Every new ct record (per instance) is checkpointed + pushed, so progress
accumulates across container restarts. If ct climbs 3->4->5..., we are on
the trail of a fish; if it plateaus at 3, that is itself strong evidence.

Usage: python3 ct_climb.py <instance> [stream_s] [seed_json]
"""
import json, subprocess, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, N
from ortools.sat.python import cp_model

INSTANCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STREAM_S = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
SEED_JSON = sys.argv[3] if len(sys.argv) > 3 else "mols10/results/nearturn2_seed_0.json"
LOG = REPO / f"mols10/results/ctclimb_{INSTANCE}.log"
BEST = REPO / f"mols10/results/ctclimb_best_{INSTANCE}.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"

def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def git_push(msg):
    subprocess.run(["git","-C",str(REPO),"add","mols10/results/"], check=False)
    r = subprocess.run(["git","-C",str(REPO),"commit","-m",msg], capture_output=True)
    if r.returncode == 0:
        subprocess.run(["git","-C",str(REPO),"pull","--rebase","origin",BRANCH], capture_output=True)
        subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], capture_output=True)

def is_latin(L):
    for r in range(N):
        if len(set(int(v) for v in L[r]))!=N: return False
    for c in range(N):
        if len(set(int(v) for v in L[:,c]))!=N: return False
    return True

def translation_symmetric(L):
    """Reject any (r,c)->(r+d,c+d) symbol-translation invariance (the turn-class
    family and its relatives): exists d in 1..9, e in 0..9 with
    L[r+d,c+d] == L[r,c]+e for all r,c.  (A cheap necessary filter for the
    excluded symmetric region; full autoparatopism-triviality is audited only
    if a high-ct candidate appears.)"""
    for d in range(1, N):
        for e in range(N):
            ok = True
            for r in range(N):
                rr = (r+d) % N
                for c in range(N):
                    if L[rr, (c+d)%N] != (L[r,c]+e) % N:
                        ok = False; break
                if not ok: break
            if ok: return True
    return False

def enum_transversals(L):
    out=[]; rows=[[int(L[r,c]) for c in range(N)] for r in range(N)]
    def bt(row,cm,vm,p):
        if row==N: out.append(tuple(p)); return
        for col in range(N):
            if not(cm>>col&1):
                v=rows[row][col]
                if not(vm>>v&1):
                    p.append(col); bt(row+1,cm|1<<col,vm|1<<v,p); p.pop()
    bt(0,0,0,[]); return out

def best_ct(L, budget_s):
    """Lower-bound max-CT(L) fast by the DIRECTED exact model (maximize sum of
    common-transversal indicators y) with a short time cap; return
    (best_ct, best_B, best_commons). Far tighter per second than blind
    mate streaming."""
    trans = enum_transversals(L)
    M=len(trans); TA8=np.array(trans,dtype=np.int8)
    cell=[[[] for _ in range(N)] for _ in range(N)]
    for i,t in enumerate(trans):
        for r in range(N): cell[r][t[r]].append(i)
    # feasibility: every cell must be covered by some transversal
    for r in range(N):
        for c in range(N):
            if not cell[r][c]: return -1, None, None
    # bad(j) = transversals meeting t_j in >=2 cells (self included): y_j forces x_i=0
    bad=[None]*M
    for lo in range(0,M,256):
        hi=min(lo+256,M)
        eq=(TA8[lo:hi,None,:]==TA8[None,:,:])
        cnt=eq.sum(axis=2,dtype=np.int8)
        for k in range(hi-lo): bad[lo+k]=np.nonzero(cnt[k]>=2)[0]
    model=cp_model.CpModel()
    x=[model.new_bool_var(f"x{i}") for i in range(M)]
    y=[model.new_bool_var(f"y{j}") for j in range(M)]
    for r in range(N):
        for c in range(N):
            model.add_exactly_one([x[i] for i in cell[r][c]])
    for j in range(M):
        if len(bad[j]):
            model.add(sum(x[int(i)] for i in bad[j])==0).only_enforce_if(y[j])
    model.maximize(sum(y))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=budget_s
    solver.parameters.num_workers=2
    st=solver.solve(model)
    if st not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return 0, None, None
    sel=[i for i in range(M) if solver.value(x[i])]
    B=np.zeros((N,N),dtype=np.int8)
    for sym,ti in enumerate(sel):
        for r in range(N): B[r,trans[ti][r]]=sym
    ar=np.arange(N)[None,:]; BV=B[ar,TA8.astype(np.int64)]
    S=np.sort(BV,axis=1); mask=(np.diff(S,axis=1)!=0).all(axis=1)
    ct=int(mask.sum()); com=[trans[i] for i in np.nonzero(mask)[0]]
    return ct, B, com

def try_third_square(L, B, commons):
    """Exact-cover the >=10 common transversals into a Latin square C."""
    if len(commons) < N: return None
    c2t=[[[] for _ in range(N)] for _ in range(N)]
    for i,t in enumerate(commons):
        for r in range(N): c2t[r][t[r]].append(i)
    m=cp_model.CpModel()
    w=[m.new_bool_var(f"w{i}") for i in range(len(commons))]
    for r in range(N):
        for c in range(N):
            if c2t[r][c]: m.add_exactly_one([w[i] for i in c2t[r][c]])
            else: return None
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=120
    if s.solve(m) not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None
    C=np.zeros((N,N),dtype=np.int8)
    for sym,i in enumerate([i for i in range(len(commons)) if s.value(w[i])]):
        for r in range(N): C[r,commons[i][r]]=sym
    return C

def intercalates(L):
    ics=[]
    for r1 in range(N):
        for r2 in range(r1+1,N):
            for c1 in range(N):
                for c2 in range(c1+1,N):
                    if (L[r1,c1]==L[r2,c2] and L[r1,c2]==L[r2,c1]
                            and L[r1,c1]!=L[r1,c2]):
                        ics.append((r1,r2,c1,c2))
    return ics

def flip(L, ic):
    r1,r2,c1,c2=ic; M=L.copy()
    M[r1,c1],M[r1,c2]=L[r1,c2],L[r1,c1]
    M[r2,c1],M[r2,c2]=L[r2,c2],L[r2,c1]
    return M

def main():
    rng=random.Random(1234+INSTANCE*97+int(time.time())%9999)
    log("="*66)
    log(f"CT-CLIMB counterexample hunt inst={INSTANCE} stream={STREAM_S}s seed={SEED_JSON}")
    seed=json.loads((REPO/SEED_JSON).read_text())
    L=np.array(seed['L'],dtype=np.int8)
    # warm-up perturbation for diversity; ensure trivial-symmetry start
    for _ in range(INSTANCE):
        ics=intercalates(L)
        if ics: L=flip(L, rng.choice(ics))
    cur,_,_=best_ct(L, STREAM_S)
    best=cur; log(f"start ct={cur}")
    T=1.5; step=0; last_push=time.time()
    while True:
        step+=1
        ics=intercalates(L)
        if not ics: break
        L2=flip(L, rng.choice(ics))
        if not is_latin(L2): continue
        if translation_symmetric(L2): continue   # stay in MMM-eligible region
        ct2,B2,com2=best_ct(L2, STREAM_S)
        if ct2>=cur or rng.random()<np.exp((ct2-cur)/T):
            L,cur=L2,ct2
        T=max(0.25,T*0.9995)
        if cur>best:
            best=cur
            log(f"step {step}: NEW ct RECORD {cur} (T={T:.2f})")
            BEST.write_text(json.dumps({'ct':int(cur),'L':L.tolist()},indent=1))
            if cur>=4:
                git_push(f"ct-climb i{INSTANCE}: trivial-symmetry square with ct={cur}")
            if cur>=N:  # candidate fish
                log(f"*** ct>={N}! verifying + completing triple ***")
                ctv,Bv,comv=best_ct(L, max(30,STREAM_S*5))
                C=try_third_square(L,Bv,comv) if comv else None
                if C is not None:
                    cl=(count_clashes(L,Bv),count_clashes(L,C),count_clashes(Bv,C))
                    log(f"VERIFY triple clashes: {cl}")
                    if cl==(0,0,0):
                        log("*** 3-MOLS(10) FOUND — COUNTEREXAMPLE! ***")
                        FOUND.write_text(json.dumps({'found':True,'method':'ct_climb',
                            'L1':L.tolist(),'L2':Bv.tolist(),'L3':C.tolist()},indent=2))
                        git_push("3-MOLS(10) COUNTEREXAMPLE FOUND via ct-climb!")
                        return
        if step%200==0:
            log(f"step {step}: cur={cur} best={best} T={T:.2f} ({len(ics)} ics)")
        if time.time()-last_push>1800:
            git_push(f"ct-climb i{INSTANCE} checkpoint: best={best}, step={step}")
            last_push=time.time()

if __name__=="__main__":
    main()
