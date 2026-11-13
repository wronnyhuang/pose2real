#!/bin/bash
# queue runner on the pod: q.sh <queuefile>; each line: <run_name> <minutes> <train args...>
# Pass --bs per line (train.py default 4). UNet gradient checkpointing is OFF (--no_ckpt) unless the line adds --ckpt.
# After training: infer (all heldout clips, mode full) + bench (plain + compiled ms/frame) into /root/runs/<name>/.
cd /root/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  line=$(head -n 1 "$1" 2>/dev/null); [ -z "$line" ] && { sleep 20; [ -f /root/STOPQ ] && exit 0; continue; }
  sed -i '1d' "$1"
  set -- "$1" $line; q=$1; name=$2; mins=$3; shift 3
  echo "$(date +%T) START $name" >> /root/q.log
  python train.py --out /root/runs/$name --minutes $mins --no_ckpt "$@" > /root/runs_$name.log 2>&1; rc=$?
  if [ -f /root/runs/$name/last.pt ]; then
    python infer.py --run /root/runs/$name --bench --bs 8 >> /root/runs_$name.log 2>&1
    echo "$(date +%T) DONE $name (train exit $rc)" >> /root/q.log
  else
    echo "$(date +%T) FAIL $name (no last.pt; see /root/runs_$name.log)" >> /root/q.log
  fi
  set -- "$q"
done
