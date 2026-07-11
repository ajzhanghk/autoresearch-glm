#!/bin/bash
# Self-healing Las Vegas supervisor for the Myrvold-MOLS SAT search.
SP=/tmp/claude-0/-home-user-autoresearch-glm/6c15e52c-04fd-5e17-80ad-627a8c92c1f0/scratchpad
MM=$SP/Myrvold-MOLS
STATE=$SP/queue_state.txt
LOG=$SP/supervisor.log
CONC=2
CASES="WW UX VX SX UU UW WX XX"
cd "$MM" || exit 1
touch "$STATE"
solved() {
  for f in log/$1-*.log; do
    grep -q "^s SATISFIABLE" "$f" 2>/dev/null && return 0
  done
  return 1
}
attempts() { local a; a=$(grep "^$1 " "$STATE" | awk '{print $2}' | tail -1); echo ${a:-0}; }
while true; do
  n=$(pgrep -cx kissat)
  if [ "$n" -lt "$CONC" ]; then
    best=""; besta=999999; allsolved=1
    for c in $CASES; do
      solved $c && continue
      allsolved=0
      a=$(attempts $c)
      if [ "$a" -lt "$besta" ]; then besta=$a; best=$c; fi
    done
    if [ "$allsolved" = 1 ]; then echo "$(date +%F,%T) ALL CASES SOLVED" >> "$LOG"; exit 0; fi
    a=$((besta+1))
    seed=$(shuf -i 1-999999999 -n 1)
    grep -v "^$best " "$STATE" > "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
    echo "$best $a $seed" >> "$STATE"
    echo "$(date +%F,%T) LAUNCH $best attempt=$a seed=$seed" >> "$LOG"
    ./run.sh -t 1800 -s $seed $best > "sol-$best-$seed.out" 2>&1 &
    sleep 5
    continue
  fi
  sleep 60
done
