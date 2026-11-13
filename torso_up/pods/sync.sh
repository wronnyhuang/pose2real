#!/bin/bash
# usage: POD=a ./sync.sh [code]   (code up; without "code" also data up)
cd "$(dirname "$0")"
A=$(python3 pod.py sshargs); SSH="${A%|*}"; HOST="${A#*|}"
rsync -az --timeout=40 -e "$SSH" --exclude=venv --exclude=hf --exclude=scratch --exclude=__pycache__ ../code/ $HOST:/root/code/
[ "$1" = code ] && exit 0
for try in 1 2 3 4 5 6; do
  rsync -az --timeout=60 --copy-links -e "$SSH" --include='manifest.json' --include='clips/' --include='clips/*/' --include='real.mp4' --include='audio_pad.wav' --include='lm.npz' --exclude='*' ../data/ $HOST:/root/data/ && break
  echo "retry $try"; sleep 3
done
