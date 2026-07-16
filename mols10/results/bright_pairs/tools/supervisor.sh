#!/bin/bash
# Self-healing Las Vegas supervisor for Myrvold-MOLS SAT search (7200s era).
# 3 concurrent kissat; a case is retired after 2 timed-out 7200s attempts.
SP=/tmp/claude-0/-home-user-autoresearch-glm/6c15e52c-04fd-5e17-80ad-627a8c92c1f0/scratchpad
MM=$SP/Myrvold-MOLS
STATE=$SP/queue_state.txt
LOG=$SP/supervisor.log
CONC=3
MAXATT=2
TIMEOUT=7200
CASES="UX VX SX UU UW WX XX"
cd "$MM" || exit 1
touch "$STATE"
solved() { for f in log/$1-*.log; do grep -q "^s SATISFIABLE" "$f" 2>/dev/null && return 0; done; return 1; }
attempts() { local a; a=$(grep "^$1 " "$STATE" | awk '{print $2}' | tail -1); echo ${a:-0}; }
running() { pgrep -af "run.sh.*$1\$" >/dev/null; }
while true; do
  n=$(pgrep -cx kissat)
  if [ "$n" -lt "$CONC" ]; then
    best=""; besta=999999; active=0
    for c in $CASES; do
      solved $c && continue
      running $c && { active=1; continue; }
      a=$(attempts $c)
      [ "$a" -ge "$MAXATT" ] && continue
      if [ "$a" -lt "$besta" ]; then besta=$a; best=$c; fi
    done
    if [ -z "$best" ]; then
      if [ "$n" -eq 0 ] && [ "$active" -eq 0 ]; then
        echo "$(date +%F,%T) QUEUE FINISHED (all cases solved or retired after $MAXATT x ${TIMEOUT}s)" >> "$LOG"
        exit 0
      fi
      sleep 60; continue
    fi
    a=$((besta+1)); seed=$(shuf -i 1-999999999 -n 1)
    grep -v "^$best " "$STATE" > "$STATE.tmp"; mv "$STATE.tmp" "$STATE"; echo "$best $a $seed" >> "$STATE"
    echo "$(date +%F,%T) LAUNCH $best attempt=$a seed=$seed" >> "$LOG"
    ./run.sh -t $TIMEOUT -s $seed $best > "sol-$best-$seed.out" 2>&1 &
    sleep 5; continue
  fi
  sleep 60
done
