#!/bin/bash
# Stage 3: generate the 12 heldout clips (full = true landmarks + audio; alips = lower-face landmarks hidden, lips from audio only).
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh; cd code
RUN=${1:-$RUNS/v3_seen}            # folder with cfg.json + last.pt  (weights/<run>/ from the artifacts)
python infer.py --run $RUN --modes full,alips --splits heldout       # -> $RUN/out, $RUN/out_alips
python infer.py --run $RUN --bench                                    # ms/frame (reported 25.6 ms, 5090, fused LoRA + torch.compile)
