#!/bin/bash
# usage: lm_one.sh <clipdir>   -> lm.npz from real.mp4 (skip if present)
d=$1; [ -s $d/lm.npz ] && exit 0
nice -n 10 $WORK/eval/venv/bin/python -I $WORK/common/lmk.py $d/real.mp4 $d/lm.tmp.npz > $d/lm.log 2>&1 && mv $d/lm.tmp.npz $d/lm.npz
