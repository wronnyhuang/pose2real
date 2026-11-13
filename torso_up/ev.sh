#!/bin/bash
# ev.sh : fetch finished runs from live pods, evaluate any run with 20 outputs and no result yet (with judge), rebuild table
cd $WORK
PY=$WORK/eval/venv/bin/python
for p in a b d; do
  python3 -c "import json,sys;sys.exit(0 if not json.load(open('pods/state_$p.json')).get('terminated_at') else 1)" 2>/dev/null || continue
  for i in 1 2 3; do POD=$p timeout 600 pods/fetch.sh && break; sleep 5; done
done
for d in runs/*/; do
  r=$(basename $d); n=$(ls $d/out/*.mp4 2>/dev/null | wc -l)
  [ "$n" = 20 ] || continue; [ -f $d/infer.json ] || continue
  [ -f eval/results/$r.json ] && continue
  echo "eval $r"; $PY eval/run_eval.py --run $r --workers 12 > eval/logs/lead_$r.log 2>&1
done
$PY eval/table.py > /dev/null 2>&1; ls eval/results
