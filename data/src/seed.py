"""Seed the 138 render2real clips: meta.json + audio.wav + audio_pad.wav (real.mp4 copied, lm.npz by lm_one.sh). usage: seed.py <audio_src> <shift_samples>"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio as A
T48 = __import__('os').path.expandvars('$R2R_DATA'); D = __import__('os').path.expandvars('$WORK/data')
src = sys.argv[1] if len(sys.argv) > 1 else None; shift = int(sys.argv[2]) if len(sys.argv) > 2 else 0
man = json.load(open(f'{T48}/manifest.json'))
for c in man['clips']:
    m = json.load(open(f"{T48}/pairs/{c['id']}/meta.json")); cd = f"{D}/clips/{c['id']}"
    meta = dict(id=c['id'], identity=c['identity'], split=c['split'], origin='render2real', file_id=m['file_id'],
                src_video=m['src_video'], src_start_s=m['src_start_s'], src_start_frame=m['src_start_frame'],
                frames=240, fps=30, size=384, crop=dict(x=m['crop_src_1080x1920'][0], y=m['crop_src_1080x1920'][1], S=m['crop_src_1080x1920'][2], src_wh=[1080, 1920]),
                speech_frac=m['speech_frac'], frontal_frac=m['frontal_frac'])
    if src:
        a, p = A.cut(m['src_video'], m['src_start_s'], src, shift)
        A.write_wav(f'{cd}/audio.wav', a); A.write_wav(f'{cd}/audio_pad.wav', p)
        meta['audio'] = dict(src=('mp4 AAC track ch0' if src == 'mp4' else 'audio/<fid>.wav'), sr=16000, shift_samples=shift)
    json.dump(meta, open(f'{cd}/meta.json.tmp', 'w'), indent=1); os.replace(f'{cd}/meta.json.tmp', f'{cd}/meta.json')
print('seeded', len(man['clips']))
