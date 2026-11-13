#!/bin/bash
# queue runner on the pod: q.sh <queuefile>; each line: <run_name> <minutes> <train args...>
cd /root/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  line=$(head -n 1 "$1" 2>/dev/null); [ -z "$line" ] && { sleep 20; [ -f /root/STOPQ ] && exit 0; continue; }
  sed -i '1d' "$1"
  set -- "$1" $line; q=$1; name=$2; mins=$3; shift 3
  echo "$(date +%T) START $name" >> /root/q.log
  python train.py --out /root/runs/$name --minutes $mins --bs 6 --no_ckpt "$@" > /root/runs_$name.log 2>&1
  python infer.py --run /root/runs/$name --bench >> /root/runs_$name.log 2>&1
  echo "$(date +%T) DONE $name" >> /root/q.log
  set -- "$q"
done
