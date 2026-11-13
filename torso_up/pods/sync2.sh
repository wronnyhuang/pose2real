#!/bin/bash
# POD=d ./sync2.sh <include-glob e.g. 'X0*'> : upload only matching clips, with retries
cd "$(dirname "$0")"
A=$(python3 pod.py sshargs); SSH="${A%|*}"; HOST="${A#*|}"
for try in $(seq 1 30); do
  rsync -a --timeout=60 --copy-links -e "$SSH" --include='manifest.json' --include='clips/' --include="clips/$1/" --include='real.mp4' --include='audio_pad.wav' --include='lm.npz' --exclude='*' ../data/ $HOST:/root/data/ && { echo DONE $1; exit 0; }
  echo "retry $try"; sleep 4
done
