#!/bin/bash
# watchdog.sh — monitors MOLS search workers and restarts any that die.
# Run once; checks every 60s. Uses exact seed-based process matching.

cd /home/user/autoresearch-glm || exit 1
PYTHON=.venv/bin/python3
LOG=mols10/results/watchdog.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "Watchdog started (PID=$$). Grace period 30s..."
sleep 30  # Let workers start up before first check

declare -A CMDS=(
  ["42"]="mols10/mols_l3_search.py --seed 42 >> mols10/results/l3_v3_s42.log 2>&1"
  ["137"]="mols10/mols_l3_search.py --seed 137 >> mols10/results/l3_v3_s137.log 2>&1"
  ["271"]="mols10/mols_l3_search.py --seed 271 >> mols10/results/l3_v3_s271.log 2>&1"
  ["503"]="mols10/mols_l3_search.py --seed 503 >> mols10/results/l3_v3_s503.log 2>&1"
  ["999"]="mols10/mols_l3_search.py --seed 999 >> mols10/results/l3_longrun2.log 2>&1"
  ["7777"]="mols10/mols_reverse_search.py --seed 7777 --budget 120 >> mols10/results/l3_reverse.log 2>&1"
  ["11111"]="mols10/mols_sat_worker.py --seed 11111 --timeout 20 >> mols10/results/l3_sat.log 2>&1"
  ["4242"]="mols10/mols_crossover_search.py --seed 4242 --budget 90 >> mols10/results/l3_crossover.log 2>&1"
  ["8888"]="mols10/mols_crossover_search.py --seed 8888 --budget 90 >> mols10/results/l3_crossover.log 2>&1"
  ["1337"]="mols10/mols_maxsat_search.py --seed 1337 --timeout 60 >> mols10/results/l3_maxsat.log 2>&1"
)

while true; do
  for SEED in "${!CMDS[@]}"; do
    # Check if a python process with this exact seed is running
    if ! pgrep -af "python.*seed $SEED" > /dev/null 2>&1; then
      CMD="${CMDS[$SEED]}"
      log "DEAD seed=$SEED — restarting: $CMD"
      eval "nohup $PYTHON $CMD &"
      log "  Restarted PID $!"
    fi
  done
  sleep 60
done
