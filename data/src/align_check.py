"""A/V alignment check on finished clips: corr(mouth signal from lm.npz, audio envelope from audio_pad.wav) at lags -6..+6 frames.
Positive lag L = audio taken L frames later than the video frame. usage: align_check.py <out.json> <clipdir>..."""
import sys, json, wave, numpy as np
sys.path.insert(0, __import__('os').path.expandvars('$WORK/data/src')); import align
SR = 16000
def env_lin(a, t0):
    h = SR // 30; out = np.zeros(240)
    for i in range(240):
        c = t0 + int(round(i * SR / 30)); seg = a[max(c - h, 0):max(c + h, 0)]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0
    return out
res = {}
for cd in sys.argv[2:]:
    d = np.load(f'{cd}/lm.npz'); w = wave.open(f'{cd}/audio_pad.wav')
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767
    jaw = d['bs'][:, 25].astype(float); jaw[~d['face_ok']] = np.nan
    lip = align.mouth(d)
    r = {}
    for name, m in (('jaw', jaw), ('lip', lip)):
        r[name] = [align.corr(m, env_lin(a, SR + int(round(L * SR / 30)))) for L in range(-6, 7)]
    res[cd.rstrip('/').split('/')[-1]] = r
json.dump(res, open(sys.argv[1], 'w'))
