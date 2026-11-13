# source this: . scripts/env.sh
export WORK=${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}      # repo root
export R2R_DATA=${R2R_DATA:-$WORK/../render2real/data}   # render2real pairs (the 138 seed clips come from there)
export R2R_ROOT=${R2R_ROOT:-$WORK/../render2real}
export SEAMLESS_ROOT=${SEAMLESS_ROOT:-$HOME/datasets/seamless-interaction}   # Meta Seamless Interaction, see DATA.md
export DATA=${DATA:-$WORK/data}            # clips/<id>/{real.mp4,audio.wav,audio_pad.wav,lm.npz,meta.json} + manifest.json
export FRAMES=${FRAMES:-$WORK/frames}      # jpg frame cache
export RUNS=${RUNS:-$WORK/runs}
export OUT_DIR=${OUT_DIR:-$WORK/out}
