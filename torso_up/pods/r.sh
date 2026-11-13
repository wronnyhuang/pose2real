#!/bin/bash
# r.sh <pod> <cmd...> : ssh with retries
cd "$(dirname "$0")"
p=$1; shift
for i in 1 2 3 4 5; do POD=$p python3 pod.py ssh "$@" && exit 0; rc=$?; [ $rc -ne 255 ] && exit $rc; sleep 6; done; exit 255
