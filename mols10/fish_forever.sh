#!/bin/bash
# Continuous corner-fishing lottery: repeated random-isotopy mate streams
# of turn square A. Any batch pair with ct >= 8 = NEW WORLD RECORD ->
# saved + pushed immediately.
cd /home/user/autoresearch-glm
while true; do
  python3 mols10/fish_hints.py 4 600 >> /tmp/fish_forever.txt 2>&1
  python3 - <<'EOF'
import json, subprocess
pool = json.load(open('mols10/results/pairct_hintpool.json'))
best = max(pool, key=lambda p: len(p['commons']))
ct = len(best['commons'])
if ct >= 8:
    json.dump({'ct': ct, **best},
              open(f'mols10/results/CT{ct}_RECORD.json', 'w'), indent=1)
    subprocess.run(['git', 'add', 'mols10/results/'])
    subprocess.run(['git', 'commit', '-m',
                    f'NEW RECORD: MOLS(10) pair with ct={ct} common transversals (streamed)'])
    subprocess.run(['git', 'pull', '--rebase', 'origin',
                    'claude/mols-order-10-search-yfQXK'], capture_output=True)
    subprocess.run(['git', 'push', '-u', 'origin',
                    'claude/mols-order-10-search-yfQXK'], capture_output=True)
    print(f'RECORD ct={ct} PUSHED', flush=True)
EOF
done
