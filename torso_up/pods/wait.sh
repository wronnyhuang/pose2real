#!/bin/bash
# wait.sh : block up to ~9 min until the number of finished runs (DONE/FAIL lines in pod q.logs) changes; prints status
cd "$(dirname "$0")"
cnt() { n=0; for p in a b d; do python3 -c "import json,sys;sys.exit(0 if not json.load(open('state_$p.json')).get('terminated_at') else 1)" 2>/dev/null || continue; x=$(timeout 30 env POD=$p python3 pod.py ssh "grep -cE 'DONE|FAIL' /root/q.log" </dev/null 2>/dev/null | tail -1); case "$x" in ''|*[!0-9]*) x=$(cat .cnt_$p 2>/dev/null || echo 0);; esac; echo $x > .cnt_$p; n=$((n+x)); done; echo $n; }
c0=$(cat .cnt_total 2>/dev/null || echo 0); end=$((SECONDS+520))
while [ $SECONDS -lt $end ]; do c=$(cnt); [ "$c" != "$c0" ] && break; sleep 45; done
echo $c > .cnt_total; echo "$(date +%T) finished-count $c0 -> $c"; ./status.sh
