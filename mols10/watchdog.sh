#!/bin/bash
# watchdog.sh — restarts any dead MOLS search worker automatically.
# Run: nohup bash mols10/watchdog.sh >> mols10/results/watchdog.log 2>&1 &

REPO=/home/user/autoresearch-glm
cd "$REPO" || exit 1

declare -A WORKERS=(
  ["mols_l3_search.py --seed 42"]="mols10/results/l3_v3_s42.log"
  ["mols_l3_search.py --seed 137"]="mols10/results/l3_v3_s137.log"
  ["mols_l3_search.py --seed 271"]="mols10/results/l3_v3_s271.log"
  ["mols_l3_search.py --seed 503"]="mols10/results/l3_v3_s503.log"
  ["mols_l3_search.py --seed 999"]="mols10/results/l3_longrun2.log"
  ["mols_reverse_search.py --seed 7777 --budget 120"]="mols10/results/l3_reverse.log"
  ["mols_sat_worker.py --seed 11111 --timeout 20"]="mols10/results/l3_sat.log"
  ["mols_crossover_search.py --seed 4242 --budget 90"]="mols10/results/l3_crossover.log"
  ["mols_crossover_search.py --seed 8888 --budget 90"]="mols10/results/l3_crossover.log"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Watchdog started. Monitoring ${#WORKERS[@]} workers."

while true; do
  for CMD in "${!WORKERS[@]}"; do
    LOG="${WORKERS[$CMD]}"
    SCRIPT=$(echo "$CMD" | awk '{print $1}')
    # Check if process is running
    if ! pgrep -f "python.*$SCRIPT.*$(echo $CMD | grep -o 'seed [0-9]*' | awk '{print $2}')" > /dev/null 2>&1; then
      log "DEAD: $CMD — restarting…"
      nohup .venv/bin/python3 mols10/$CMD >> "$LOG" 2>&1 &
      log "  Restarted PID $!"
    fi
  done
  sleep 30
done
