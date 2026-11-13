#!/bin/bash
# Stage 1: data. 853 clips of 8 s, 384x384, 30 fps: real.mp4 + audio.wav (16 kHz mono) + lm.npz (478 face pts + 33 pose pts + 52 blendshapes per frame).
#  --media-only : you fetched lm.npz/meta.json; re-cut real.mp4 + audio from Seamless using meta.json (src_video, src_start_s, crop).
#  default      : full rebuild: window selection -> cut + audio + landmarks + QC -> manifest (data/src/*.py; see page/summary.md for counts).
set -e; cd "$(dirname "$0")/.."; . scripts/env.sh
if [ "$1" = "--media-only" ]; then
python3 - <<'P'
import json,os,subprocess,glob
S=os.environ['SEAMLESS_ROOT']; D=os.environ['DATA']
for mp in glob.glob(f'{D}/clips/*/meta.json'):
    m=json.load(open(mp)); d=os.path.dirname(mp); x,y,s=m['crop']['x'],m['crop']['y'],m['crop']['S']
    src=m['src_video'].replace('$SEAMLESS_ROOT',S)
    if not os.path.exists(f'{d}/real.mp4'):
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(m['src_start_s']),'-i',src,'-t','8','-vf',f'crop={s}:{s}:{x}:{y},scale=384:384:flags=lanczos,fps=30','-c:v','libx264','-pix_fmt','yuv420p','-an',f'{d}/real.mp4'],check=True)
    if not os.path.exists(f'{d}/audio.wav'):
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(m['src_start_s']),'-i',src,'-t','8','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',f'{d}/audio.wav'],check=True)
P
else
# render2real clips first (needs render2real/data), then the expansion windows
python3 data/src/seed.py            # see its docstring: <audio_src> <shift_samples>
python3 data/src/select2.py         # -> data/work/sel2.json
python3 data/src/proc.py $(python3 -c "import json;print(' '.join(w['id'] for w in json.load(open('data/work/sel2.json'))))")
python3 data/src/manifest.py --final
fi
