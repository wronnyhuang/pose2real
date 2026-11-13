#!/bin/bash
# POD=a ./fetch.sh : pull finished runs (outputs, cfg, samples, logs; not weights) into ROOT/runs
cd "$(dirname "$0")"
A=$(python3 pod.py sshargs); SSH="${A%|*}"; HOST="${A#*|}"
rsync -az --timeout=60 -e "$SSH" --exclude='last.pt' --exclude='smoke*' $HOST:/root/runs/ ../runs/
rsync -az --timeout=60 -e "$SSH" $HOST:/root/q.log ../pods/q_$POD.log 2>/dev/null
