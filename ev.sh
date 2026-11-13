#!/bin/bash
# ev.sh <run> [extra args] : score runs/<run>/out and out_alips (no judge unless JUDGE=1)
cd $WORK
PY=$WORK/eval/venv/bin/python
J="--no-judge"; [ "$JUDGE" = 1 ] && J=""
for r in "$@"; do
  nice $PY eval/run_eval.py --run $r $J --workers 12 > eval/log_$r.txt 2>&1
  for od in runs/$r/out_*; do [ -d "$od" ] || continue; m=${od##*/out_}
    [ -f eval/results/${r}__$m.json ] || nice $PY eval/run_eval.py --run ${r}__$m --dir $od $J --workers 12 > eval/log_${r}__$m.txt 2>&1
  done
done
$PY eval/table.py > /dev/null 2>&1
