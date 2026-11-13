"""Audio cutting. Source audio decoded once per file to 16 kHz float32 mono and cached as .npy.
src='mp4' : the AAC track muxed in the speaker's mp4 (channel 0 = that speaker's own channel; ch1 is silent)
src='wav' : the per-speaker audio/<fid>.wav
cut(fid_path, start_s, src) -> (audio 8 s, audio_pad 10 s) int16 arrays, sample 0 of audio == frame 0 at start_s."""
import os, subprocess, numpy as np, wave
CACHE = __import__('os').path.expandvars('$WORK/data/work/aud16')
SR = 16000


def _decode(path, ch0):
    args = ['ffmpeg', '-v', 'error', '-i', path, '-vn']
    args += (['-af', 'pan=mono|c0=c0'] if ch0 else ['-ac', '1'])
    args += ['-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(args, capture_output=True, check=True).stdout, np.float32)


def load(video_path, src):
    fid = os.path.basename(video_path)[:-4]
    os.makedirs(CACHE, exist_ok=True)
    cp = f'{CACHE}/{fid}_{src}.npy'
    if os.path.exists(cp):
        return np.load(cp)
    if src == 'mp4':
        a = _decode(video_path, True)
    else:
        a = _decode(video_path.replace('/video/', '/audio/')[:-4] + '.wav', False)
    tmp = cp + f'.{os.getpid()}.npy'
    np.save(tmp, a); os.replace(tmp, cp)
    return a


def cut(video_path, start_s, src='mp4', shift_samples=0):
    a = load(video_path, src)
    s0 = int(round(start_s * SR)) + shift_samples
    def seg(b, n):
        out = np.zeros(n, np.float32)
        lo, hi = max(b, 0), min(b + n, len(a))
        if hi > lo:
            out[lo - b:hi - b] = a[lo:hi]
        return out
    main = seg(s0, 8 * SR); pad = seg(s0 - SR, 10 * SR)
    q = lambda x: (np.clip(x, -1, 1) * 32767).round().astype(np.int16)
    return q(main), q(pad)


def write_wav(path, x):
    tmp = path + '.tmp'
    with wave.open(tmp, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.tobytes())
    os.replace(tmp, path)
