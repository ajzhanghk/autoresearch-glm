#!/bin/bash
# Self-healing Las Vegas supervisor for the ct>=8 threshold decision.
# Each round runs a fresh-seed attempt (3h cap). Stops as soon as a
# verdict artifact exists (SAT json or theorem json).
cd /home/user/autoresearch-glm
SAT_FILE=mols10/results/ctthr_sat_ctthr_sat_turnsq_ct6_squareA_ge7_ge8.json
THM_FILE=mols10/results/ctthr_theorem_ctthr_sat_turnsq_ct6_squareA_ge7_ge8.json
while true; do
  if [ -f "$SAT_FILE" ] || [ -f "$THM_FILE" ]; then
    echo "verdict exists; supervisor exiting" >> /tmp/thr8_super.log
    exit 0
  fi
  echo "$(date '+%H:%M:%S') new attempt" >> /tmp/thr8_super.log
  python3 mols10/ct_threshold_decide.py \
    mols10/results/ctthr_sat_turnsq_ct6_squareA_ge7.json 8 10800 \
    >> /tmp/thr8.txt 2>&1
  sleep 5
done
