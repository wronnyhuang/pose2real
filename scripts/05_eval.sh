#!/bin/bash
# Stage 4: CPU eval harness: landmark errors, SyncNet LSE-C/D, blind Claude judge for realism. See eval/README.md.
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
PY=${PY:-python3}
$PY eval/run_eval.py --run ${1:-v3_seen} --workers 12 ${NOJUDGE:+--no-judge}
$PY eval/table.py        # -> eval/results/table.json
