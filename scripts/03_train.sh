#!/bin/bash
# Stage 2: train. Recipe of the best models (run_configs/*.json hold the exact arguments of every reported run). 1x RTX 5090, bs 6.
#   1) encoder warm start (minutes)      2) 'v2' full model, 62 min        3) 'v4_more' / 'v3_seen' continue from v2, 50 min each
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh; cd code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_xy.py --out $RUNS/xy_long --audio win --steps 24000 --bs 32 --lr 3e-4 --origin all   # -> $RUNS/xy_long/enc.pt (15 min)
python train.py --out $RUNS/v2 --minutes 62 --max_steps 5200 --bs 6 --audio win --au_la -1 --refbase --w_lips 3 --origin all --enc_init $RUNS/xy_long/enc.pt --no_ckpt
python train.py --out $RUNS/v4_more --minutes 50 --bs 6 --audio win --au_la -1 --refbase --w_lips 3 --origin all --gan_start 300 --resume $RUNS/v2/last.pt --no_ckpt                        # unseen-identity model
python train.py --out $RUNS/v3_seen --minutes 50 --bs 6 --audio win --au_la -1 --refbase --w_lips 3 --origin all --gan_start 300 --splits train,heldout_ref --boost 8 --resume $RUNS/v2/last.pt --no_ckpt  # seen-identity model
# torso-up + hands variant: see torso_up/code/README.md (same layout, 512x640)
