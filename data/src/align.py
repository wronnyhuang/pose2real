"""A/V alignment check: corr(mouth opening from lm.npz, audio RMS envelope) vs audio shift.
mode 'src' : uses the cached full source audio (mp4 / wav) around meta src_start_s at sub-frame shifts
mode 'clip': uses the clip's own audio.wav at integer frame lags.
usage: align.py src <clipdir>...   |  align.py clip <clipdir>...  -> prints JSON lines"""
import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.expandvars('$WORK/data/src'))
import audio as A
SR = 16000


def mouth(d):
    f = d['face']; ok = d['face_ok']
    fw = np.linalg.norm(f[:, 234, :2] - f[:, 454, :2], axis=1)
    m = np.linalg.norm(f[:, 13, :2] - f[:, 14, :2], axis=1) / np.maximum(fw, 1e-4)
    m[~ok] = np.nan
    return m


def env(a, t0_samples, T=240):
    # RMS of a 1/30 s window centred on each frame time
    h = SR // 60
    out = np.zeros(T)
    for i in range(T):
        c = t0_samples + int(round(i * SR / 30))
        seg = a[max(c - h, 0):max(c + h, 0)]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0
    return np.log(out + 1e-4)


def corr(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 50 or x[m].std() == 0 or y[m].std() == 0: return float('nan')
    return float(np.corrcoef(x[m], y[m])[0, 1])


if __name__ == '__main__':
    mode = sys.argv[1]
    for cd in sys.argv[2:]:
        meta = json.load(open(f'{cd}/meta.json')); d = np.load(f'{cd}/lm.npz'); m = mouth(d)
        if mode == 'src':
            res = {}
            for src in ('mp4', 'wav'):
                a = A.load(meta['src_video'], src); s0 = int(round(meta['src_start_s'] * SR))
                # shift in ms: audio sample at frame i taken from (i/30 + shift)
                res[src] = [corr(m, env(a, s0 + int(round(ms * SR / 1000)))) for ms in range(-150, 151, 5)]
            print(json.dumps(dict(id=meta['id'], **res)), flush=True)
        else:
            import wave
            w = wave.open(f'{cd}/audio_pad.wav'); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767
            lags = {L: corr(m, env(a, SR + int(round(L * SR / 30)))) for L in range(-6, 7)}
            best = max(lags, key=lambda k: -9 if np.isnan(lags[k]) else lags[k])
            print(json.dumps(dict(id=meta['id'], lag0=lags[0], best_lag=best, best=lags[best], lags=lags)), flush=True)
