#!/bin/bash
# Stage 1b: frame jpg cache + HuBERT audio features for every clip that has real.mp4 / audio_pad.wav (needs a GPU; run after 02_data.sh --media-only).
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh; cd code
python - <<'P'
import data as D
ids = [c['id'] for c in D.manifest() if c['split'] in ('heldout', 'heldout_ref')]
D.prep_frames(ids); D.prep_audio_win(ids); print('PREP_OK', len(ids))
P
