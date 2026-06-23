#!/usr/bin/env python3
"""CP-SAT: iso2_tv21_0_5 and tv_254, targeting E<=26 first (stepping stone), then E<=25."""
import json, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path("/home/user/autoresearch-glm")
sys.path.insert(0, str(REPO / "mols10"))
from mols_cpsat_worker import count_clashes, build_col_index, N
from ortools.sat.python import cp_model

LOG   = REPO / "mols10/results/cpsat_e27_step.log"
BEST  = REPO / "mols10/results/cpsat_e27_step_best.json"
FOUND = REPO / "mols10/results/MOLS10_FOUND.json"
BRANCH = "claude/mols-order-10-search-yfQXK"
TIMEOUT = 600

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def git_push(msg):
    import subprocess
    subprocess.run(["git","-C",str(REPO),"add",str(BEST.relative_to(REPO))], check=False)
    subprocess.run(["git","-C",str(REPO),"commit","-m",msg], check=False)
    subprocess.run(["git","-C",str(REPO),"push","-u","origin",BRANCH], check=False)

def relabel(L3):
    perm = np.zeros(N, dtype=np.int32)
    for j in range(N): perm[int(L3[0,j])] = j
    return perm[L3.astype(np.int32)].astype(np.int8)

def solve(L1, L2, hint_L3, upper_bound, timeout_s, rseed, symbreak=True):
    model = cp_model.CpModel()
    L3v = [[model.new_int_var(0,N-1,f"L3_{i}_{j}") for j in range(N)] for i in range(N)]
    for i in range(N): model.add_all_different(L3v[i])
    for j in range(N): model.add_all_different([L3v[i][j] for i in range(N)])
    if symbreak:
        for j in range(N): model.add(L3v[0][j]==j)
    hint = relabel(hint_L3) if symbreak else hint_L3.copy()
    for i in range(N):
        for j in range(N): model.add_hint(L3v[i][j], int(hint[i,j]))
    col1 = build_col_index(L1, N); col2 = build_col_index(L2, N)
    cov13=[]; cov23=[]; hints=[]
    for a in range(N):
        for b in range(N):
            cov=model.new_bool_var(f"c13_{a}_{b}"); inds=[]; any_c=False
            for i in range(N):
                j=int(col1[a,i]); ind=model.new_bool_var(f"i13_{a}_{b}_{i}")
                model.add(L3v[i][j]==b).only_enforce_if(ind)
                model.add(L3v[i][j]!=b).only_enforce_if(ind.negated())
                inds.append(ind); v=1 if int(hint[i,j])==b else 0
                hints.append((ind,v));
                if v: any_c=True
            model.add_bool_or(inds).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in inds]).only_enforce_if(cov.negated())
            cov13.append(cov); hints.append((cov,1 if any_c else 0))
    for a in range(N):
        for b in range(N):
            cov=model.new_bool_var(f"c23_{a}_{b}"); inds=[]; any_c=False
            for i in range(N):
                j=int(col2[a,i]); ind=model.new_bool_var(f"i23_{a}_{b}_{i}")
                model.add(L3v[i][j]==b).only_enforce_if(ind)
                model.add(L3v[i][j]!=b).only_enforce_if(ind.negated())
                inds.append(ind); v=1 if int(hint[i,j])==b else 0
                hints.append((ind,v));
                if v: any_c=True
            model.add_bool_or(inds).only_enforce_if(cov)
            model.add_bool_and([x.negated() for x in inds]).only_enforce_if(cov.negated())
            cov23.append(cov); hints.append((cov,1 if any_c else 0))
    for var,val in hints: model.add_hint(var,val)
    uncov=[x.negated() for x in cov13]+[x.negated() for x in cov23]
    obj=model.new_int_var(0,2*N*N,"obj"); model.add(obj==sum(uncov))
    model.add(obj<=upper_bound)
    model.minimize(obj)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=timeout_s
    solver.parameters.num_search_workers=8
    solver.parameters.random_seed=rseed
    status=solver.solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        E=int(solver.objective_value)
        L3_sol=np.array([[solver.value(L3v[i][j]) for j in range(N)] for i in range(N)],dtype=np.int8)
        return E, L3_sol
    return None, None

pairs = json.loads((REPO/"mols10/results/promising_pairs.json").read_text())
pair_map = {p['pair_id']:p for p in pairs}
best_data = json.loads((REPO/"mols10/results/fast_deep_sa_best.json").read_text())

targets = {
    'iso2_tv21_0_5': (best_data['iso2_tv21_0_5'], 26),  # (seed, starting upper_bound)
    'tv_254':        (best_data['tv_254'],        26),
}

rng = random.Random(27182818285)
attempt=0; global_best={}

log("="*70)
log("CP-SAT stepping: E=27 pairs → E<=26 then E<=25 ...")
log("="*70)

while True:
    for pair_id, (seed_data, ub) in list(targets.items()):
        attempt += 1
        p = pair_map[pair_id]
        L1=np.array(p['L1'],dtype=np.int8).reshape(N,N)
        L2=np.array(p['L2'],dtype=np.int8).reshape(N,N)
        hint=np.array(seed_data['L3'],dtype=np.int8).reshape(N,N)
        e_hint=count_clashes(L1,hint)+count_clashes(L2,hint)
        rseed=rng.randint(0,2**31-1)
        log(f"[{attempt}] {pair_id} hint_E={e_hint} upper={ub} rseed={rseed}")
        E,L3_sol=solve(L1,L2,hint,ub,TIMEOUT,rseed,symbreak=(attempt%3!=0))
        if L3_sol is not None:
            cl13=int(count_clashes(L1,L3_sol)); cl23=int(count_clashes(L2,L3_sol))
            Ev=cl13+cl23
            log(f"  *** FOUND E={Ev} (cl13={cl13},cl23={cl23}) ***")
            # Update seed for this pair and tighten bound
            targets[pair_id] = ({'L3':L3_sol.tolist(),'E':Ev}, Ev-1)
            entry={'E':Ev,'cl13':cl13,'cl23':cl23,'pair_id':pair_id,
                   'L1':L1.tolist(),'L2':L2.tolist(),'L3':L3_sol.tolist()}
            BEST.write_text(json.dumps(entry,indent=2))
            git_push(f"cpsat_step: {pair_id} E={Ev}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
            if Ev==0:
                log("*** 3-MOLS FOUND! ***")
                FOUND.write_text(json.dumps({'found':True,'pair_id':pair_id,
                    'L1':L1.tolist(),'L2':L2.tolist(),'L3':L3_sol.tolist(),
                    'cl13':cl13,'cl23':cl23},indent=2))
                sys.exit(0)
        else:
            log(f"  No solution (INFEASIBLE/timeout)")
