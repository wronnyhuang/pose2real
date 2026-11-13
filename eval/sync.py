"""SyncNet LSE-C / LSE-D / AV offset, following joonson/syncnet_python (run_pipeline.py + SyncNetInstance.evaluate)
as used by Wav2Lip's evaluation (vshift 15, batch 20, 25 fps, 224 px crops, MFCC of 16 kHz audio, syncnet_v2.model).

Differences from the reference pipeline (documented in README):
  * the face box comes from a detector of choice per frame: 'mp' = bbox of the 468 MediaPipe face-mesh points from
    common/lmk.py (default; already computed for the landmark metrics), or 's3fd' = the original S3FD detector
    (conf 0.9, facedet_scale 0.25) — both then go through the SAME crop policy as run_pipeline.crop_video:
    s = max(w,h)/2, medfilt(13) on s/x/y at 25 fps, pad 110, crop [y-s, y+1.8s] x [x-1.4s, x+1.4s] (crop_scale 0.4), resize 224.
  * frames are resampled 30 -> 25 fps in memory (nearest timestamp, like ffmpeg -r 25) instead of via an .avi/jpg round trip.
  * one face track per video (single-speaker clips); frames without a box are linearly interpolated (edges held).
"""
import os, sys, math, subprocess, tempfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, 'third_party')
sys.path.insert(0, os.path.join(TP, 'pylib'))
SN_DIR = os.path.join(TP, 'syncnet_python')
WEIGHTS = os.path.join(TP, 'weights', 'syncnet_v2.model')
S3FD_WEIGHTS = os.path.join(TP, 'weights', 'sfd_face.pth')
FPS_OUT = 25
VSHIFT = 15
BATCH = 20
CROP_SCALE = 0.40
# MediaPipe mesh bbox is ~0.904x the S3FD box on the 12 real heldout clips (validate.py, 2026-11-13) -> scale it up
MP_BOX_SCALE = 1.106

_NET = None
_S3FD = None


def _load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def net():
    global _NET
    if _NET is None:
        import torch
        S = _load_mod('sn_model', os.path.join(SN_DIR, 'SyncNetModel.py')).S
        m = S(num_layers_in_fc_layers=1024)
        sd = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        own = m.state_dict()
        for k, v in sd.items(): own[k].copy_(v)
        m.eval(); _NET = m
    return _NET


def s3fd():
    global _S3FD
    if _S3FD is None:
        import torch
        sys.path.insert(0, SN_DIR)
        from detectors.s3fd.nets import S3FDNet
        from detectors.s3fd.box_utils import nms_
        n = S3FDNet(device='cpu'); n.load_state_dict(torch.load(S3FD_WEIGHTS, map_location='cpu', weights_only=True)); n.eval()
        _S3FD = (n, nms_)
    return _S3FD


def s3fd_boxes(frames, scale=0.25, conf_th=0.9):
    """per-frame best box [x1,y1,x2,y2] or nan (same preprocessing as detectors/s3fd/__init__.py)."""
    import torch
    n, nms_ = s3fd(); mean = np.array([104., 117., 123.], np.float32)[:, None, None]
    out = np.full((len(frames), 4), np.nan)
    with torch.no_grad():
        for t, image in enumerate(frames):
            h, w = image.shape[:2]
            im = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            x = np.transpose(im, (2, 0, 1))[[2, 1, 0]].astype(np.float32); x -= mean; x = x[[2, 1, 0]]
            y = n(torch.from_numpy(x)[None]).data
            sc = np.array([w, h, w, h]); bb = []
            for i in range(y.size(1)):
                j = 0
                while y[0, i, j, 0] > conf_th:
                    bb.append(np.r_[y[0, i, j, 1:].numpy() * sc, float(y[0, i, j, 0])]); j += 1
            if bb:
                bb = np.array(bb); bb = bb[nms_(bb, 0.1)]
                out[t] = bb[np.argmax(bb[:, 4]), :4]
    return out


def mp_boxes(face, face_ok, W, H):
    """bbox of the 468 face-mesh points (pixels) per frame scaled about its centre by MP_BOX_SCALE, nan where no face."""
    out = np.full((len(face), 4), np.nan)
    p = face[:, :468, :2] * np.array([W, H])
    out[face_ok] = np.c_[p[face_ok, :, 0].min(1), p[face_ok, :, 1].min(1), p[face_ok, :, 0].max(1), p[face_ok, :, 1].max(1)]
    c = (out[:, :2] + out[:, 2:]) / 2; h = (out[:, 2:] - out[:, :2]) / 2 * MP_BOX_SCALE
    return np.c_[c - h, c + h]


def resample_idx(n_in, fps_in=30.0, fps_out=FPS_OUT):
    n_out = int(math.floor(n_in / fps_in * fps_out + 1e-6))
    return np.clip(np.round(np.arange(n_out) * fps_in / fps_out).astype(int), 0, n_in - 1)


def interp_nan(a):
    a = a.copy()
    for j in range(a.shape[1]):
        ok = ~np.isnan(a[:, j])
        if ok.sum() == 0: return None
        a[:, j] = np.interp(np.arange(len(a)), np.where(ok)[0], a[ok, j])
    return a


def crops(frames25, boxes25):
    """run_pipeline.crop_video crop policy -> (T,224,224,3) uint8 BGR"""
    from scipy import signal
    b = interp_nan(boxes25)
    if b is None: return None
    s = np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2
    y = (b[:, 1] + b[:, 3]) / 2; x = (b[:, 0] + b[:, 2]) / 2
    s = signal.medfilt(s, 13); x = signal.medfilt(x, 13); y = signal.medfilt(y, 13)
    cs = CROP_SCALE; out = []
    for t, img in enumerate(frames25):
        bs = s[t]; bsi = int(bs * (1 + 2 * cs))
        fr = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = y[t] + bsi; mx = x[t] + bsi
        f = fr[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        out.append(cv2.resize(f, (224, 224)))
    return np.stack(out)


def mfcc(audio, sr=16000):
    import python_speech_features
    return np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio, sr))])  # (13, N)


def score(crops25, audio, sr=16000, vshift=VSHIFT):
    """SyncNetInstance.evaluate math. audio = int16 mono 16 kHz aligned with frame 0."""
    import torch
    from scipy import signal
    torch.set_num_threads(int(os.environ.get('SYNC_THREADS', '2')))
    m = net()
    im = np.transpose(crops25, (3, 0, 1, 2))[None]  # 1,3,T,H,W  (BGR, 0..255, as the reference)
    imtv = torch.from_numpy(im.astype(np.float32))
    cc = mfcc(audio, sr)[None, None]
    cct = torch.from_numpy(cc.astype(np.float32))
    min_length = min(len(crops25), math.floor(len(audio) / 640))
    lastframe = min_length - 5
    im_feat, cc_feat = [], []
    with torch.no_grad():
        for i in range(0, lastframe, BATCH):
            rng = range(i, min(lastframe, i + BATCH))
            im_feat.append(m.forward_lip(torch.cat([imtv[:, :, v:v + 5] for v in rng], 0)))
            cc_feat.append(m.forward_aud(torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in rng], 0)))
    f1 = torch.cat(im_feat, 0); f2 = torch.cat(cc_feat, 0)
    win = vshift * 2 + 1
    f2p = torch.nn.functional.pad(f2, (0, 0, vshift, vshift))
    dists = [torch.nn.functional.pairwise_distance(f1[[i], :].repeat(win, 1), f2p[i:i + win, :]) for i in range(len(f1))]
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    offset = vshift - int(minidx)
    conf = float(torch.median(mdist) - minval)
    # lag-0 distance too (useful when the best offset is spurious)
    d0 = float(mdist[vshift])
    return dict(lse_c=round(conf, 4), lse_d=round(float(minval), 4), offset_frames=int(offset), lse_d_at0=round(d0, 4),
                n_windows=len(f1))


def read_wav(path):
    from scipy.io import wavfile
    sr, a = wavfile.read(path)
    if a.ndim > 1: a = a.mean(1).astype(np.int16)
    return sr, a


def mux_and_extract(video_path, wav_path, tmpdir):
    """mux generated video + audio.wav into a temp .mkv (video copied, PCM audio), then read the audio back out of it."""
    mkv = os.path.join(tmpdir, 'muxed.mkv'); wav = os.path.join(tmpdir, 'a.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video_path, '-i', wav_path, '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'copy', '-c:a', 'pcm_s16le', '-shortest', mkv], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mkv, '-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', wav],
                   check=True)
    return mkv, read_wav(wav)


def read_frames(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return np.stack(fr) if fr else None, fps


def sync_for(frames30, fps_in, boxes30, audio, sr=16000):
    idx = resample_idx(len(frames30), fps_in)
    c = crops(frames30[idx], boxes30[idx])
    if c is None: return dict(error='no face box in any frame')
    return score(c, audio, sr)
