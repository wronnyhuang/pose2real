#!/bin/bash
# usage: run.sh <idlist file> <parallel>   -- one proc.py per id, nice 10, 1 thread per process; resumable (qc.json = done)
cd $WORK/data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 GLOG_minloglevel=2
xargs -a "$1" -P "$2" -n 1 nice -n 10 $WORK/eval/venv/bin/python -I src/proc.py 2>/dev/null | grep -v '^\(INFO\|WARNING\|W0000\|I0000\)'
