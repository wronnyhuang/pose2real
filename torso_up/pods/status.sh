#!/bin/bash
# one line per pod
cd "$(dirname "$0")"
for p in ${PODS:-a b d}; do
  [ -f state_$p.json ] || continue
  python3 -c "import json,sys;sys.exit(0 if not json.load(open('state_$p.json')).get('terminated_at') else 1)" || continue
  o=$(timeout 40 env POD=$p python3 pod.py ssh 'q=$(tail -1 /root/q.log 2>/dev/null); n=$(echo $q | awk "{print \$3}"); echo "$q | $(tail -1 /root/prep.log 2>/dev/null | cut -c1-40) | $(tail -1 /root/runs_$n.log 2>/dev/null | cut -c1-48) | left: $(wc -l < /root/queue.txt 2>/dev/null) | data $(du -sm /root/data | cut -f1) $(ls /root/PULLED 2>/dev/null)"' < /dev/null 2>&1 | tail -1)
  echo "$p: $o"
done
